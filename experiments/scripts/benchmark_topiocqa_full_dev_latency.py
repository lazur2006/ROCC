#!/usr/bin/env python3
"""Measure the five primary TopiOCQA preprocessing conditions on full dev."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


ROOT = Path(os.environ.get("ROCC_ROOT") or os.environ.get(
    "MASTER_THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]
)).expanduser().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rocc import (
    EncoderCrfHistorySelector,
    add_conversation_columns,
    build_topiocqa_conversation_samples,
    collate_selector_inference_batch,
    encode_selector_inference_row,
    load_itercqr_rewriter,
    load_itercqr_tokenizer,
    load_topiocqa_frame,
    normalize_rewrite,
    resolve_topiocqa_resources,
)


OUTPUT_ROOT = ROOT / "experiments/results/10_efficiency_analysis/latency"
T5_DIR = ROOT / "experiments/model/IterCQR/IterCQR Model"
SELECTOR_DIR = (
    ROOT
    / "experiments/results/05_history_selector/full_candidates/collapsed_crf"
)
POPULATION_PATH = (
    ROOT
    / "experiments/results/07a_topiocqa_final/protocol/population.json"
)
INPUTS_PATH = (
    ROOT
    / "experiments/results/07a_topiocqa_final/query_bundle/serialization"
    / "main_inputs.csv"
)
DATA_DIR = ROOT / "experiments/data/topiocqa"
CACHE_DB = Path(os.environ.get("ROCC_NB07_PIPELINE_CACHE_DB", str(
    ROOT / ".cache/master_thesis/03_oracle_headroom_analysis/runtime_cache.sqlite3"
))).expanduser().resolve()
SELECTOR_ASSETS = ROOT / "models/rocc"
QUERY_MANIFEST = INPUTS_PATH.parents[1] / "manifest.json"

BATCH_SIZE = 16
WARMUP_BATCHES = 5
SEED = 13
POPULATION_QUERIES = 2_514
HISTORY_ZERO_QUERIES = 205
EXPECTED_POPULATION_SHA256 = (
    "f2a71cab647da97ea16471715df20afb"
    "268c92833e2341b9bc64859e9838b6fe"
)
EXPECTED_GPU = "NVIDIA GeForce RTX 4070 Laptop GPU"
EXPECTED_HASHES = {
    SELECTOR_DIR / "model.pt": (
        "a07271abefbc37ef08dba4eb94adb0e1"
        "65f53a652060c069c909a7a1614115c4"
    ),
    SELECTOR_DIR / "manifest.json": (
        "1e4f580a142ce66bd86547ffe57cb67794f"
        "8171b4106cc1d6353ea56b65dff4d"
    ),
    T5_DIR / "pytorch_model.bin": (
        "1a25ceaf597bf92ff2897cccb22e9c46"
        "b807efbf78a3234186066e1f62f16ae3"
    ),
    T5_DIR / "config.json": (
        "a1f20ac0527995010a72aa51a3c7dd1"
        "fec6bcb15c0d69bcbdca5611f035ee2fa"
    ),
    POPULATION_PATH: (
        "11f10ba5689257c945ac8bfbd216c1829"
        "f05ffe0d06f10d48c8713a432f351bf"
    ),
    INPUTS_PATH: (
        "8fe7f909e08d2ad1bb83544324b70af3"
        "150ae9d4d079643f8d6e5b3b34f3982e"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def chunks(values: Sequence[Any]) -> list[Sequence[Any]]:
    return [
        values[start : start + BATCH_SIZE]
        for start in range(0, len(values), BATCH_SIZE)
    ]


def timed(call, device: torch.device) -> tuple[Any, float]:
    torch.cuda.synchronize(device)
    started = time.perf_counter_ns()
    result = call()
    torch.cuda.synchronize(device)
    seconds = (time.perf_counter_ns() - started) / 1e9
    return result, seconds


def gpu_state() -> dict[str, str]:
    command = [
        "nvidia-smi", "--query-gpu=name,persistence_mode,driver_version",
        "--format=csv,noheader",
    ]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        selected = visible.split(",", 1)[0].strip()
        if not re.fullmatch(r"\d+|GPU-[0-9a-fA-F-]+", selected):
            raise RuntimeError("Set CUDA_VISIBLE_DEVICES to a GPU number or GPU-UUID; CUDA device 0 must be visible.")
        if selected.isdigit() and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
            raise RuntimeError("Numeric CUDA_VISIBLE_DEVICES requires CUDA_DEVICE_ORDER=PCI_BUS_ID; alternatively select a GPU-UUID.")
        command.append(f"--id={selected}")
    output = subprocess.check_output(
        command,
        text=True,
    ).strip()
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("GPU selection is ambiguous. Set CUDA_VISIBLE_DEVICES to the GPU number or GPU-UUID to measure as CUDA device 0.")
    fields = [item.strip() for item in lines[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError(f"Unexpected GPU state: {output}")
    return {
        "name": fields[0],
        "persistence_mode": fields[1],
        "driver_version": fields[2],
    }


def lineage_mode() -> str:
    mode = os.environ.get("ROCC_LINEAGE", "historical").strip().lower()
    if mode not in {"historical", "local"}:
        raise ValueError("ROCC_LINEAGE must be historical or local.")
    return mode


def input_identity() -> dict[str, str]:
    """Validate the current NB05/NB07 chain, without models or cache writes."""
    observed = {}
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing upstream input; finish NB05/NB07a first: {path}")
        observed[str(path.relative_to(ROOT))] = sha256_file(path)
        if (lineage_mode() == "historical" or path.parent == T5_DIR) and observed[str(path.relative_to(ROOT))] != expected:
            raise RuntimeError(f"Frozen artifact changed: {path}")
    for path in (QUERY_MANIFEST, SELECTOR_ASSETS / "config.json",
                 SELECTOR_ASSETS / "tokenizer.json", SELECTOR_ASSETS / "manifest.json"):
        observed[str(path.relative_to(ROOT))] = sha256_file(path)
    selector = json.loads((SELECTOR_DIR / "manifest.json").read_text())
    query = json.loads(QUERY_MANIFEST.read_text())
    population = json.loads(POPULATION_PATH.read_text())
    checkpoint_hash = observed[str((SELECTOR_DIR / "model.pt").relative_to(ROOT))]
    if (selector.get("complete") is not True or query.get("complete") is not True
            or selector.get("checkpoint_sha256") != checkpoint_hash
            or query.get("checkpoint_sha256", {}).get("pretrained") != checkpoint_hash
            or query.get("itercqr_model_sha256") != observed[str((T5_DIR / "pytorch_model.bin").relative_to(ROOT))]
            or query.get("population_sha256") != EXPECTED_POPULATION_SHA256
            or population.get("sample_id_sha256") != EXPECTED_POPULATION_SHA256):
        raise RuntimeError("NB05/NB07a model or population identities disagree.")
    config = selector["config"]
    backbone = json.loads((SELECTOR_ASSETS / "manifest.json").read_text())["backbone"]
    required = {"label_space": "collapsed", "decoder": "crf", "adaptation": "full",
                "max_length": 512, "history_order": "recent_first",
                "model_name": backbone["repository"], "model_revision": backbone["revision"]}
    if any(config.get(key) != value for key, value in required.items()):
        raise RuntimeError("NB05 checkpoint does not implement the canonical collapsed-CRF architecture.")
    if not CACHE_DB.is_file():
        raise FileNotFoundError(f"NB07a token cache missing; set ROCC_NB07_PIPELINE_CACHE_DB: {CACHE_DB}")
    observed["experiments/scripts/benchmark_topiocqa_full_dev_latency.py"] = sha256_file(Path(__file__))
    return observed


def preflight() -> tuple[dict[str, str], dict[str, str]]:
    observed = input_identity()
    state = gpu_state()
    if lineage_mode() == "historical" and state["name"] != EXPECTED_GPU:
        raise RuntimeError(f"Unexpected GPU: {state['name']}")
    if state["persistence_mode"] not in {"Enabled", "Disabled"}:
        raise RuntimeError(
            f"Unexpected persistence mode: {state['persistence_mode']}"
        )
    return observed, state


def load_dev_rows() -> tuple[list[str], list[dict[str, Any]], int]:
    resources = resolve_topiocqa_resources(DATA_DIR)
    frame = add_conversation_columns(
        load_topiocqa_frame(resources, splits=("dev",))
    )
    samples = build_topiocqa_conversation_samples(
        frame,
        minimum_history_depth=0,
        progress=False,
    )
    by_id = {str(sample.sample_id): sample for sample in samples}
    sample_ids = sorted(by_id)
    population_hash = hashlib.sha256(
        "\n".join(sample_ids).encode("utf-8")
    ).hexdigest()
    if (
        len(sample_ids) != POPULATION_QUERIES
        or len(by_id) != POPULATION_QUERIES
        or population_hash != EXPECTED_POPULATION_SHA256
    ):
        raise RuntimeError("TopiOCQA full-dev population changed.")
    selector_rows = []
    history_zero = 0
    for sample_id in sample_ids:
        sample = by_id[sample_id]
        if not sample.history:
            history_zero += 1
            continue
        selector_rows.append(
            {
                "sample_id": sample_id,
                "current_query": str(sample.current_query),
                "history": [
                    {
                        "turn_id": int(turn.turn_id),
                        "question": str(turn.question),
                        "answer": str(turn.answer),
                    }
                    for turn in sample.history
                ],
            }
        )
    if history_zero != HISTORY_ZERO_QUERIES:
        raise RuntimeError(
            f"Expected {HISTORY_ZERO_QUERIES} no-history queries, "
            f"found {history_zero}."
        )
    if len(selector_rows) != POPULATION_QUERIES - HISTORY_ZERO_QUERIES:
        raise RuntimeError("ROCC population is incomplete.")
    return sample_ids, selector_rows, history_zero


def select_t5_inputs(
    source: pd.DataFrame,
    sample_ids: Sequence[str],
    *,
    arm: str,
    budget: int,
) -> pd.DataFrame:
    selected = source.loc[
        source["arm"].eq(arm) & source["budget"].eq(int(budget))
    ].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    if (
        len(selected) != POPULATION_QUERIES
        or selected["sample_id"].duplicated().any()
        or set(selected["sample_id"]) != set(sample_ids)
    ):
        raise RuntimeError(f"Incomplete {arm}-B{budget} T5 inputs.")
    return selected.set_index("sample_id").loc[list(sample_ids)].reset_index()


def prepare_t5_batches(
    frame: pd.DataFrame,
    pad_token_id: int,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    """Read NB07a's existing token cache; exactly the original dynamic padding."""
    prepared: list[dict[str, Any]] = []
    expected: list[list[str]] = []
    connection = sqlite3.connect(CACHE_DB.as_uri() + "?mode=ro", uri=True)
    try:
        for part in chunks(frame.to_dict("records")):
            sequences = []
            for row in part:
                cached = connection.execute(
                    "SELECT token_blob FROM token_inputs WHERE input_key = ?", (str(row["input_key"]),)
                ).fetchone()
                if cached is None:
                    raise RuntimeError("NB07a token cache is incomplete; rerun upstream serialization.")
                sequence = np.frombuffer(cached[0], dtype="<i4").astype(np.int64).tolist()
                if len(sequence) != int(row["input_length"]):
                    raise RuntimeError("Cached T5 inputs changed.")
                sequences.append(sequence)
            maximum = max(map(len, sequences))
            prepared.append({
                "bt_input_ids": torch.tensor([s + [pad_token_id] * (maximum - len(s)) for s in sequences], dtype=torch.long),
                "bt_attention_mask": torch.tensor([[1] * len(s) + [0] * (maximum - len(s)) for s in sequences], dtype=torch.long),
            })
            expected.append([str(row["rewrite_norm"]) for row in part])
    finally:
        connection.close()
    return prepared, expected


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true", help="Validate input files and cached batches only; no model loading, GPU calls or output writes.")
    args = parser.parse_args()
    if args.run_index < 1:
        raise ValueError("run-index must be positive.")

    OUTPUT_ROOT = args.output_dir.expanduser().resolve()
    output_dir = OUTPUT_ROOT / "runs" / f"run_{args.run_index}"
    invocation_started = time.perf_counter()
    if args.dry_run:
        input_identity()
        ids, _, _ = load_dev_rows()
        source = pd.read_csv(INPUTS_PATH)
        for arm, budget in (("I", 64), ("I", 512), ("pretrained_R", 64)):
            frame = select_t5_inputs(source, ids, arm=arm, budget=budget)
            prepare_t5_batches(frame, 0)
        print("Full-dev input files, population and cached batches validated; no measurement executed.")
        return
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Do not overwrite a completed or previously recorded run: {output_dir}")
    observed_hashes, observed_gpu = preflight()
    if not torch.cuda.is_available():
        raise RuntimeError("This GPU benchmark requires a CUDA-enabled PyTorch build and an available CUDA GPU.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    sample_ids, selector_rows, history_zero = load_dev_rows()
    selector_manifest = json.loads(
        (SELECTOR_DIR / "manifest.json").read_text(encoding="utf-8")
    )["config"]
    selector_tokenizer = AutoTokenizer.from_pretrained(
        SELECTOR_ASSETS, local_files_only=True,
        use_fast=True,
    )
    state = torch.load(SELECTOR_DIR / "model.pt", map_location="cpu", weights_only=True)
    selector = EncoderCrfHistorySelector(
        str(SELECTOR_ASSETS), num_labels=3, class_weights=state["class_weights"].tolist(),
        crf_loss_weight=1.0, token_loss_weight=1.0,
        encoder=AutoModel.from_config(AutoConfig.from_pretrained(SELECTOR_ASSETS, local_files_only=True), trust_remote_code=False),
    )
    selector.load_state_dict(state, strict=True)
    selector.to(device)
    del state
    selector.eval()
    encoded = [
        encode_selector_inference_row(
            row,
            selector_tokenizer,
            max_length=int(selector_manifest["max_length"]),
            history_order=str(selector_manifest["history_order"]),
        )
        for row in tqdm(
            selector_rows,
            desc=f"run {args.run_index}: encode ROCC",
            unit="query",
            dynamic_ncols=True,
        )
    ]
    rocc_batches = [
        collate_selector_inference_batch(
            part,
            int(selector_tokenizer.pad_token_id),
            torch.device("cpu"),
        )
        for part in chunks(encoded)
    ]

    tokenizer = load_itercqr_tokenizer(T5_DIR)
    source = pd.read_csv(INPUTS_PATH)
    source["sample_id"] = source["sample_id"].astype(str)
    i64 = select_t5_inputs(
        source, sample_ids, arm="I", budget=64
    )
    i512 = select_t5_inputs(
        source, sample_ids, arm="I", budget=512
    )
    r64 = select_t5_inputs(
        source, sample_ids, arm="pretrained_R", budget=64
    )
    paired = i64[["sample_id", "input_key", "history_depth"]].merge(
        r64[["sample_id", "input_key"]],
        on="sample_id",
        suffixes=("_I", "_R"),
        validate="one_to_one",
    )
    duplicate_pairs = int(
        paired["input_key_I"].eq(paired["input_key_R"]).sum()
    )
    no_history_duplicates = int(
        (
            paired["history_depth"].eq(0)
            & paired["input_key_I"].eq(paired["input_key_R"])
        ).sum()
    )
    if (lineage_mode() == "historical" and duplicate_pairs != 212) or no_history_duplicates != history_zero:
        raise RuntimeError(
            "The frozen I/R input-identity structure changed."
        )
    ird64 = pd.concat([i64, r64], ignore_index=True).drop_duplicates(
        "input_key", keep="first"
    )
    if lineage_mode() == "historical" and len(ird64) != 4_816:
        raise RuntimeError(
            f"Expected 4,816 unique IRD-B64 inputs, found {len(ird64)}."
        )

    condition_frames = {
        "T5_I_B64": i64,
        "T5_I_B512": i512,
        "T5_R_B64": r64,
        "T5_IRD_B64_UNIQUE": ird64,
    }
    t5_batches: dict[str, list[dict[str, Any]]] = {}
    t5_expected: dict[str, list[list[str]]] = {}
    for name, frame in condition_frames.items():
        t5_batches[name], t5_expected[name] = prepare_t5_batches(
            frame, int(tokenizer.pad_token_id)
        )
    del source

    rewriter = load_itercqr_rewriter(model_dir=T5_DIR, device=device)
    rewriter.model.eval()

    def run_rocc(batch: dict[str, Any]) -> Any:
        return selector(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            history_mask=batch["history_mask"].to(device),
        )

    def run_t5(
        batch: dict[str, Any],
        *,
        measure: bool,
    ) -> tuple[list[str], float | None, float | None]:
        def encode() -> tuple[Any, torch.Tensor]:
            input_ids = batch["bt_input_ids"].to(device)
            attention_mask = batch["bt_attention_mask"].to(device)
            outputs = rewriter.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return outputs, attention_mask

        if measure:
            encoded_result, encoder_seconds = timed(encode, device)
        else:
            encoded_result, encoder_seconds = encode(), None
        encoder_outputs, attention_mask = encoded_result

        def decode() -> list[str]:
            generated = rewriter.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=rewriter.generation_config.max_length,
                num_beams=rewriter.generation_config.num_beams,
                num_return_sequences=1,
            )
            return rewriter.tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )

        if measure:
            decoded, decoder_seconds = timed(decode, device)
        else:
            decoded, decoder_seconds = decode(), None
        return decoded, encoder_seconds, decoder_seconds

    conditions: list[tuple[str, list[dict[str, Any]]]] = [
        ("ROCC_B64", rocc_batches),
        *[(name, t5_batches[name]) for name in condition_frames],
    ]
    if args.run_index % 2 == 0:
        conditions = conditions[::-1]

    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for name in condition_frames:
            decoded, _, _ = run_t5(t5_batches[name][0], measure=False)
            actual = [normalize_rewrite(value) for value in decoded]
            if actual != t5_expected[name][0]:
                raise RuntimeError(f"Split T5 rewrite drift for {name}.")

        for name, items in tqdm(
            conditions,
            desc=f"run {args.run_index}: warm-up",
            unit="component",
            dynamic_ncols=True,
        ):
            for batch in items[:WARMUP_BATCHES]:
                if name == "ROCC_B64":
                    run_rocc(batch)
                else:
                    run_t5(batch, measure=False)
        torch.cuda.synchronize(device)

        progress = tqdm(
            total=sum(len(items) for _, items in conditions),
            desc=f"run {args.run_index}: full dev",
            unit="batch",
            dynamic_ncols=True,
        )
        for name, items in conditions:
            for batch_index, batch in enumerate(items):
                if name == "ROCC_B64":
                    batch_size = int(batch["input_ids"].shape[0])
                    _, seconds = timed(lambda: run_rocc(batch), device)
                    records.append(
                        {
                            "component": name,
                            "phase": "rocc",
                            "batch_index": batch_index,
                            "batch_size": batch_size,
                            "seconds": seconds,
                        }
                    )
                else:
                    batch_size = int(batch["bt_input_ids"].shape[0])
                    _, encoder_seconds, decoder_seconds = run_t5(
                        batch,
                        measure=True,
                    )
                    records.extend(
                        [
                            {
                                "component": name,
                                "phase": "t5_encoder",
                                "batch_index": batch_index,
                                "batch_size": batch_size,
                                "seconds": float(encoder_seconds),
                            },
                            {
                                "component": name,
                                "phase": "t5_decoder",
                                "batch_index": batch_index,
                                "batch_size": batch_size,
                                "seconds": float(decoder_seconds),
                            },
                        ]
                    )
                progress.update()
        progress.close()

    components = pd.DataFrame(records)
    component_totals = (
        components.groupby(["component", "phase"], sort=False)
        .agg(
            processed_inputs=("batch_size", "sum"),
            batches=("batch_index", "size"),
            seconds=("seconds", "sum"),
        )
        .reset_index()
    )

    def seconds(component: str, phase: str) -> float:
        selected = component_totals.loc[
            component_totals["component"].eq(component)
            & component_totals["phase"].eq(phase),
            "seconds",
        ]
        if len(selected) != 1:
            raise RuntimeError(f"Missing timing for {component}/{phase}.")
        return float(selected.iloc[0])

    rocc_seconds = seconds("ROCC_B64", "rocc")
    route_specs = (
        ("I", 64, 0.0, "T5_I_B64", POPULATION_QUERIES),
        ("I", 512, 0.0, "T5_I_B512", POPULATION_QUERIES),
        ("D", 64, rocc_seconds, None, 0),
        ("R", 64, rocc_seconds, "T5_R_B64", POPULATION_QUERIES),
        ("IRD", 64, rocc_seconds, "T5_IRD_B64_UNIQUE", len(ird64)),
    )
    route_rows = []
    for route, budget, route_rocc, component, t5_inputs in route_specs:
        encoder_seconds = (
            seconds(component, "t5_encoder") if component else 0.0
        )
        decoder_seconds = (
            seconds(component, "t5_decoder") if component else 0.0
        )
        total_seconds = route_rocc + encoder_seconds + decoder_seconds
        route_rows.append(
            {
                "route": route,
                "budget": budget,
                "queries": POPULATION_QUERIES,
                "independent_run": args.run_index,
                "within_invocation_repeats": 1,
                "rocc_processed_queries": (
                    len(selector_rows) if route in {"D", "R", "IRD"} else 0
                ),
                "t5_processed_inputs": t5_inputs,
                "rocc_ms_per_query": (
                    1_000 * route_rocc / POPULATION_QUERIES
                ),
                "t5_encoder_ms_per_query": (
                    1_000 * encoder_seconds / POPULATION_QUERIES
                ),
                "t5_decoder_ms_per_query": (
                    1_000 * decoder_seconds / POPULATION_QUERIES
                ),
                "total_ms_per_query": (
                    1_000 * total_seconds / POPULATION_QUERIES
                ),
            }
        )
    routes = pd.DataFrame(route_rows)
    component_error = (
        routes["total_ms_per_query"]
        - routes[
            [
                "rocc_ms_per_query",
                "t5_encoder_ms_per_query",
                "t5_decoder_ms_per_query",
            ]
        ].sum(axis=1)
    ).abs()
    if not component_error.lt(1e-10).all():
        raise RuntimeError("Route/component sums disagree.")

    output_dir.mkdir(parents=True, exist_ok=True)
    components.to_csv(output_dir / "components_by_batch.csv", index=False)
    component_totals.to_csv(
        output_dir / "component_totals.csv", index=False
    )
    routes.to_csv(
        output_dir / "route_components_summary.csv", index=False
    )
    manifest = {
        "schema_version": 1,
        "lineage": lineage_mode(),
        "run_index": args.run_index,
        "sample": "complete TopiOCQA development split",
        "population_queries": POPULATION_QUERIES,
        "population_sha256": EXPECTED_POPULATION_SHA256,
        "history_zero_queries": history_zero,
        "rocc_processed_queries": len(selector_rows),
        "batch_size": BATCH_SIZE,
        "within_invocation_repeats": 1,
        "warmup_batches_per_component": WARMUP_BATCHES,
        "dynamic_padding": True,
        "scope": "prepared inputs -> model outputs; no retrieval/RRF",
        "route_time": "paired sum of synchronized component wall clocks",
        "primary_conditions": [
            "I-B64",
            "I-B512",
            "R-B64",
            "D-B64",
            "IRD-B64",
        ],
        "ird_b64": {
            "requested_i_plus_r_inputs": 2 * POPULATION_QUERIES,
            "identical_i_r_input_pairs": duplicate_pairs,
            "identical_no_history_pairs": no_history_duplicates,
            "unique_materialized_t5_inputs": len(ird64),
        },
        "gpu": observed_gpu,
        "tf32": False,
        "seed": SEED,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "input_sha256": observed_hashes,
        "invocation_wall_seconds": time.perf_counter() - invocation_started,
        "complete": True,
    }
    atomic_json(output_dir / "manifest.json", manifest)
    print(routes.to_string(index=False), flush=True)

    del rewriter, selector, selector_tokenizer, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
