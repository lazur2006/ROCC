#!/usr/bin/env python3
"""Evaluate saved original and BM25-corrected Teachers on TopiOCQA Dev.

Consumes NB04 labels, NB05/NB06 checkpoints and NB07a BM25 results. It calls
no Teacher API and performs no training, but retrieval needs the local full
TopiOCQA BM25 index, IterCQR weights and the selector backbone. Use
``--check-inputs`` for a file-presence check without loading models or indexes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rocc import (  # noqa: E402
    CachedIterCQRBM25Config,
    CachedIterCQRBM25Pipeline,
    LABEL_TO_ID,
    SelectorExperimentConfig,
    add_conversation_columns,
    build_raw_teacher_target_histories,
    build_topiocqa_conversation_samples,
    collate_selector_batch,
    gather_history_sequences,
    hard_relabel_gold_score_dataset,
    load_itercqr_tokenizer,
    load_topiocqa_frame,
    materialize_gold_bm25_token_scores_for_rows,
    prepare_gold_score_selector_dataset,
    prepare_selector_datasets,
    read_label_jsonl,
    resolve_topiocqa_resources,
    run_history_arm_evaluation,
)
from experiments.rocc.crf_sampling import load_crf_inference_model  # noqa: E402
from experiments.rocc.post_training import _bm25_index_signature  # noqa: E402


RESULT_DIR = ROOT / "experiments/results/10_efficiency_analysis/teacher_eval"
TEACHER_LABELS = (
    ROOT
    / "experiments/results/04_teacher"
    / "rocc_history_labels_topiocqa_dev.jsonl"
)
TEACHER_MANIFEST = TEACHER_LABELS.with_suffix(".manifest.json")
TRAIN_LABELS = (
    ROOT
    / "experiments/results/04_teacher"
    / "rocc_history_labels_topiocqa_train.jsonl"
)
TRAIN_MANIFEST = TRAIN_LABELS.with_suffix(".manifest.json")
TOPIOCQA_DATA = ROOT / "experiments/data/topiocqa"
ITERCQR_MODEL = ROOT / "experiments/model/IterCQR/IterCQR Model"
ITERCQR_WEIGHTS = ITERCQR_MODEL / "pytorch_model.bin"
BM25_INDEX = (
    ROOT / "experiments/data/pyserini_bm25_lucene_topiocqa/lucene_index"
)
PRETRAINED_CHECKPOINT = (
    ROOT
    / "experiments/results/05_history_selector/full_candidates"
    / "collapsed_crf/model.pt"
)
TREATMENT_CHECKPOINT = (
    ROOT
    / "experiments/results/06_post_training"
    / "controlled_hard_relabel_nostop_e6/training/bm25_treatment"
    / "checkpoint_epoch_6.pt"
)
NB07_METRICS = (
    ROOT
    / "experiments/results/07a_topiocqa_final/backends/bm25"
    / "route_metrics_by_query.csv"
)
NB07_MANIFEST = NB07_METRICS.parent / "manifest.json"
NB07_BUNDLE_MANIFEST = NB07_METRICS.parents[2] / "query_bundle/manifest.json"
CACHE_DB = ROOT / ".cache/teacher_eval/runtime_cache.sqlite3"
ENCODING_CACHE = ROOT / ".cache/teacher_eval/selector_encodings"
DEVICE = "auto"
REQUIRE_ORIGINAL_ARTIFACTS = False
BUDGETS = (64, 128, 256, 512)
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CHUNK = 250
EXPECTED = {
    "full_population": (
        "f2a71cab647da97ea16471715df20afb268c92833e2341b9bc64859e9838b6fe"
    ),
    "teacher_labels": (
        "43ced1a1f302672a29737d14fe02a1d9e6c751ea0eb1225aebe8222b72f49c37"
    ),
    "teacher_manifest": (
        "548ad3eb185d6315ad9b359721cbeba211005962a41d6f6b6bcc66ace5af742b"
    ),
    "pretrained": (
        "a07271abefbc37ef08dba4eb94adb0e165f53a652060c069c909a7a1614115c4"
    ),
    "bm25_student": (
        "a8d00d1cdcc7116e4f20833429a1acb879b2346bea4840319b555c073410992d"
    ),
    "itercqr_weights": (
        "1a25ceaf597bf92ff2897cccb22e9c46b807efbf78a3234186066e1f62f16ae3"
    ),
}
EXPECTED_BM25_INDEX_SIGNATURE = (
    "c511bfb24c95bf3239dcdf16ce8ceea989c2cd9d7d39271145762c23542152d0"
)
IMPLEMENTATION_FILES = {
    "evaluator": Path(__file__).resolve(),
    "cached_itercqr": ROOT / "experiments/rocc/cached_itercqr.py",
    "controlled_post_training": (
        ROOT / "experiments/rocc/controlled_post_training.py"
    ),
    "crf_sampling": ROOT / "experiments/rocc/crf_sampling.py",
    "post_training": ROOT / "experiments/rocc/post_training.py",
    "selector_experiment": ROOT / "experiments/rocc/selector_experiment.py",
    "teacher_evaluation": ROOT / "experiments/rocc/teacher_evaluation.py",
}
OUTPUT_FILES = (
    "cohort.csv",
    "retrieval_metrics_by_query.csv",
    "retrieval_summary.csv",
    "paired_comparisons.csv",
    "agreement_counts_by_query.csv",
    "agreement_summary.csv",
    "selection_summary.csv",
    "serialization_summary.csv",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments/results",
                        help="Notebook result tree (04_teacher, 05_history_selector, 06_post_training, 07a_topiocqa_final).")
    parser.add_argument("--data-dir", type=Path, default=TOPIOCQA_DATA,
                        help="Prepared TopiOCQA directory from NB00b.")
    parser.add_argument("--itercqr-model-dir", type=Path, default=ITERCQR_MODEL,
                        help="Complete local IterCQR model/tokenizer directory.")
    parser.add_argument("--bm25-index-dir", type=Path, default=BM25_INDEX,
                        help="Complete local Lucene TopiOCQA BM25 index.")
    parser.add_argument("--pretrained-checkpoint", type=Path,
                        help="Override the NB05 collapsed-CRF model.pt.")
    parser.add_argument("--treatment-checkpoint", type=Path,
                        help="Override the NB06 BM25 treatment epoch-6 checkpoint.")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: RESULTS_DIR/10_efficiency_analysis/teacher_eval.")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / ".cache/teacher_eval",
                        help="Local rewrite/retrieval and selector-encoding caches.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Check required paths only; no model/index loading, downloads or output writes.")
    parser.add_argument("--require-original-artifacts", action="store_true",
                        help="Additionally require the original thesis file hashes and machine-specific index signature; normally omit for relocated/rebuilt artifacts.")
    return parser.parse_args(argv)


def configure_paths(args: argparse.Namespace) -> None:
    global RESULT_DIR, TEACHER_LABELS, TEACHER_MANIFEST, TRAIN_LABELS, TRAIN_MANIFEST
    global TOPIOCQA_DATA, ITERCQR_MODEL, ITERCQR_WEIGHTS, BM25_INDEX
    global PRETRAINED_CHECKPOINT, TREATMENT_CHECKPOINT, NB07_METRICS, NB07_MANIFEST, NB07_BUNDLE_MANIFEST
    global CACHE_DB, ENCODING_CACHE, DEVICE, REQUIRE_ORIGINAL_ARTIFACTS
    results = args.results_dir.expanduser().resolve()
    RESULT_DIR = (args.output_dir or results / "10_efficiency_analysis/teacher_eval").expanduser().resolve()
    TEACHER_LABELS = results / "04_teacher/rocc_history_labels_topiocqa_dev.jsonl"
    TEACHER_MANIFEST = TEACHER_LABELS.with_suffix(".manifest.json")
    TRAIN_LABELS = results / "04_teacher/rocc_history_labels_topiocqa_train.jsonl"
    TRAIN_MANIFEST = TRAIN_LABELS.with_suffix(".manifest.json")
    TOPIOCQA_DATA = args.data_dir.expanduser().resolve()
    ITERCQR_MODEL = args.itercqr_model_dir.expanduser().resolve()
    ITERCQR_WEIGHTS = ITERCQR_MODEL / "pytorch_model.bin"
    BM25_INDEX = args.bm25_index_dir.expanduser().resolve()
    PRETRAINED_CHECKPOINT = (args.pretrained_checkpoint or results / "05_history_selector/full_candidates/collapsed_crf/model.pt").expanduser().resolve()
    TREATMENT_CHECKPOINT = (args.treatment_checkpoint or results / "06_post_training/controlled_hard_relabel_nostop_e6/training/bm25_treatment/checkpoint_epoch_6.pt").expanduser().resolve()
    NB07_METRICS = results / "07a_topiocqa_final/backends/bm25/route_metrics_by_query.csv"
    NB07_MANIFEST = NB07_METRICS.parent / "manifest.json"
    NB07_BUNDLE_MANIFEST = results / "07a_topiocqa_final/query_bundle/manifest.json"
    cache = args.cache_dir.expanduser().resolve()
    CACHE_DB = cache / "runtime_cache.sqlite3"
    ENCODING_CACHE = cache / "selector_encodings"
    DEVICE = args.device
    REQUIRE_ORIGINAL_ARTIFACTS = args.require_original_artifacts


def required_input_files() -> tuple[Path, ...]:
    return (
        TEACHER_LABELS, TEACHER_MANIFEST, TRAIN_LABELS, TRAIN_MANIFEST,
        PRETRAINED_CHECKPOINT, TREATMENT_CHECKPOINT, ITERCQR_WEIGHTS,
        ITERCQR_MODEL / "config.json", ITERCQR_MODEL / "tokenizer_config.json",
        NB07_METRICS, NB07_MANIFEST, NB07_BUNDLE_MANIFEST, *IMPLEMENTATION_FILES.values(),
    )


def check_inputs() -> None:
    """Check local prerequisites; this does not prove index or model validity."""
    missing = [str(path) for path in required_input_files() if not path.is_file()]
    if not TOPIOCQA_DATA.is_dir():
        missing.append(str(TOPIOCQA_DATA))
    if not BM25_INDEX.is_dir() or not any(BM25_INDEX.glob("segments_*")):
        missing.append(f"{BM25_INDEX} (Lucene segments_* commit required)")
    if missing:
        raise FileNotFoundError("Missing Teacher-evaluation inputs:\n  " + "\n  ".join(missing))
    print("Required local files are present; contents, index and model compatibility not yet checked.")
    print(f"Output: {RESULT_DIR}; caches: {CACHE_DB.parent}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_frame(name: str, frame: pd.DataFrame) -> None:
    path = RESULT_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


def source_identity() -> dict[str, Any]:
    required = required_input_files()
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Teacher-evaluation inputs: " + str(missing))
    observed = {
        "teacher_labels": sha256(TEACHER_LABELS),
        "teacher_manifest": sha256(TEACHER_MANIFEST),
        "pretrained": sha256(PRETRAINED_CHECKPOINT),
        "bm25_student": sha256(TREATMENT_CHECKPOINT),
        "itercqr_weights": sha256(ITERCQR_WEIGHTS),
    }
    for key, expected in EXPECTED.items():
        if key == "full_population":
            continue
        if REQUIRE_ORIGINAL_ARTIFACTS and observed[key] != expected:
            raise RuntimeError(f"Pinned source drift for {key}: {observed[key]}")
    teacher_manifest = json.loads(TEACHER_MANIFEST.read_text(encoding="utf-8"))
    if (
        teacher_manifest.get("dataset_sha256") != observed["teacher_labels"]
        or teacher_manifest.get("model") != "gpt-5.4"
        or teacher_manifest.get("protocol_hash") != selector_config().expected_teacher_protocol_hash
        or teacher_manifest.get("seed") != 42
        or teacher_manifest.get("eligible_queries") != 2_104
        or teacher_manifest.get("error_count") != 0
    ):
        raise RuntimeError("Teacher-label manifest does not match the supplied labels and original Teacher protocol.")
    index_signature = _bm25_index_signature(BM25_INDEX, k1=0.9, b=0.4)
    if REQUIRE_ORIGINAL_ARTIFACTS and index_signature != EXPECTED_BM25_INDEX_SIGNATURE:
        raise RuntimeError("Pinned TopiOCQA BM25 index signature drifted.")
    nb07 = json.loads(NB07_MANIFEST.read_text(encoding="utf-8"))
    if (
        nb07.get("complete") is not True
        or nb07.get("backend") != "bm25"
        or nb07.get("population_sha256") != EXPECTED["full_population"]
        or nb07.get("files", {}).get(NB07_METRICS.name) != sha256(NB07_METRICS)
    ):
        raise RuntimeError("NB07a BM25 source manifest is not canonical.")
    if (
        nb07.get("checkpoint_sha256", {}).get("pretrained") != observed["pretrained"]
        or nb07.get("checkpoint_sha256", {}).get("bm25_treatment_e6") != observed["bm25_student"]
        or tuple(nb07.get("budgets", ())) != BUDGETS
        or nb07.get("retrieval_top_k") != 1_000
        or nb07.get("backend_identity", {}).get("k1") != 0.9
        or nb07.get("backend_identity", {}).get("b") != 0.4
    ):
        raise RuntimeError("NB07a students, budgets or BM25 parameters do not match this evaluation.")
    bundle = json.loads(NB07_BUNDLE_MANIFEST.read_text(encoding="utf-8"))
    if (
        bundle.get("complete") is not True
        or bundle.get("population_sha256") != EXPECTED["full_population"]
        or bundle.get("query_bundle_sha256") != nb07.get("query_bundle_sha256")
        or bundle.get("itercqr_model_sha256") != observed["itercqr_weights"]
    ):
        raise RuntimeError("NB07a query bundle does not match the supplied IterCQR model and population.")
    return {
        "protocol": "topiocqa_teacher_labeled_dev_v2",
        "teacher_labels_sha256": observed["teacher_labels"],
        "teacher_manifest_sha256": observed["teacher_manifest"],
        "pretrained_checkpoint_sha256": observed["pretrained"],
        "bm25_student_checkpoint_sha256": observed["bm25_student"],
        "itercqr_weights_sha256": observed["itercqr_weights"],
        "bm25_index_signature": index_signature,
        "train_labels_sha256": sha256(TRAIN_LABELS),
        "train_manifest_sha256": sha256(TRAIN_MANIFEST),
        "require_original_artifacts": REQUIRE_ORIGINAL_ARTIFACTS,
        "implementation_sha256": {
            name: sha256(path)
            for name, path in IMPLEMENTATION_FILES.items()
        },
        "nb07_metrics_sha256": sha256(NB07_METRICS),
        "nb07_manifest_sha256": sha256(NB07_MANIFEST),
        "nb07_bundle_manifest_sha256": sha256(NB07_BUNDLE_MANIFEST),
        "budgets": list(BUDGETS),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "cohort_rule": "teacher_labeled_history_depth_ge_2",
        "no_history_queries_excluded": 205,
        "single_history_queries_excluded": 205,
        "retrieval_queries": 2_104,
        "agreement_queries": 2_104,
    }


def reusable(identity: Mapping[str, Any]) -> bool:
    manifest_path = RESULT_DIR / "manifest.json"
    paths = [RESULT_DIR / name for name in OUTPUT_FILES]
    if not manifest_path.is_file() or not all(path.is_file() for path in paths):
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or manifest.get("identity") != identity:
        return False
    return manifest.get("files") == {
        path.name: sha256(path) for path in paths
    }


def load_population() -> tuple[
    list[Any],
    list[dict[str, Any]],
    dict[str, Sequence[Any]],
    dict[str, Sequence[Mapping[str, Any]]],
    pd.DataFrame,
]:
    resources = resolve_topiocqa_resources(TOPIOCQA_DATA)
    frame = add_conversation_columns(
        load_topiocqa_frame(resources, splits=("dev",))
    )
    samples = build_topiocqa_conversation_samples(
        frame,
        minimum_history_depth=0,
        progress=True,
    )
    full_ids = [str(sample.sample_id) for sample in samples]
    if len(full_ids) != 2_514 or text_sha256("\n".join(sorted(full_ids))) != EXPECTED[
        "full_population"
    ]:
        raise RuntimeError("TopiOCQA Full-Dev population drifted.")
    depth = {str(sample.sample_id): len(sample.history) for sample in samples}
    if {
        value: sum(observed == value for observed in depth.values())
        for value in (0, 1)
    } != {0: 205, 1: 205} or sum(value >= 2 for value in depth.values()) != 2_104:
        raise RuntimeError("TopiOCQA history-depth counts drifted.")

    cohort_samples = [sample for sample in samples if len(sample.history) >= 2]
    cohort_ids = [str(sample.sample_id) for sample in cohort_samples]
    teacher_rows = read_label_jsonl(TEACHER_LABELS, progress=True)
    teacher_ids = [str(row["sample_id"]) for row in teacher_rows]
    expected_teacher_ids = {
        sample_id for sample_id, value in depth.items() if value >= 2
    }
    if (
        len(cohort_ids) != 2_104
        or len(teacher_rows) != 2_104
        or set(cohort_ids) != expected_teacher_ids
        or set(teacher_ids) != expected_teacher_ids
    ):
        raise RuntimeError("Teacher-evaluable cohort alignment failed.")

    indexed = frame.set_index("sample_id")
    gold = {
        sample_id: indexed.at[sample_id, "positive_ctx_passage_ids"]
        for sample_id in cohort_ids
    }
    positive = {
        sample_id: list(indexed.at[sample_id, "positive_ctxs"] or [])
        for sample_id in teacher_ids
    }
    cohort = pd.DataFrame(
        {
            "sample_id": cohort_ids,
            "history_depth": [depth[sample_id] for sample_id in cohort_ids],
            "teacher_labeled": 1,
        }
    )
    return cohort_samples, teacher_rows, gold, positive, cohort


def selector_config() -> SelectorExperimentConfig:
    return SelectorExperimentConfig(
        train_labels=TRAIN_LABELS,
        train_manifest=TRAIN_MANIFEST,
        topiocqa_data_dir=TOPIOCQA_DATA,
        result_dir=RESULT_DIR,
        encoding_cache_dir=ENCODING_CACHE,
        itercqr_model_dir=ITERCQR_MODEL,
        bm25_index_dir=BM25_INDEX,
        pipeline_cache_db=CACHE_DB,
        batch_size=32,
        rewrite_batch_size=16,
        retrieval_batch_size=64,
        retrieval_workers=8,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
    )


def retrieval_evaluation(
    *,
    samples: Sequence[Any],
    teacher_rows: Sequence[dict[str, Any]],
    gold: Mapping[str, Sequence[Any]],
    positive: Mapping[str, Sequence[Mapping[str, Any]]],
    config: SelectorExperimentConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Any]:
    teacher_ids = {str(row["sample_id"]) for row in teacher_rows}
    score = materialize_gold_bm25_token_scores_for_rows(
        rows=teacher_rows,
        gold_by_sample={sample_id: gold[sample_id] for sample_id in teacher_ids},
        positive_contexts_by_sample=positive,
        bm25_index_dir=BM25_INDEX,
        output_path=RESULT_DIR / "gold_bm25_token_scores.jsonl",
        source_dataset_sha256=sha256(TEACHER_LABELS),
        progress=True,
    )
    if (
        score.identity.get("bm25_index_signature")
        != _bm25_index_signature(BM25_INDEX, k1=0.9, b=0.4)
    ):
        raise RuntimeError("TopiOCQA BM25 index changed during evaluation.")
    histories, selection = build_raw_teacher_target_histories(
        rows=teacher_rows,
        score_result=score,
        progress=True,
    )
    sample_ids = {str(sample.sample_id) for sample in samples}
    if any(set(arm_histories) != sample_ids for arm_histories in histories.values()):
        raise RuntimeError("Teacher histories do not cover the labeled 2,104-query cohort.")

    tokenizer = load_itercqr_tokenizer(ITERCQR_MODEL)
    pipeline = CachedIterCQRBM25Pipeline(
        config=CachedIterCQRBM25Config(
            cache_db=CACHE_DB,
            model_dir=ITERCQR_MODEL,
            bm25_index_dir=BM25_INDEX,
            device=str(device),
            rewrite_batch_size=16,
            retrieval_batch_size=64,
            retrieval_workers=8,
            retrieval_top_k=1_000,
            eval_ks=(3, 10, 100, 1000),
            k1=0.9,
            b=0.4,
            progress=True,
        ),
        tokenizer=tokenizer,
    )
    try:
        evaluated = run_history_arm_evaluation(
            samples=samples,
            histories_by_arm=histories,
            arm_order=("teacher_original", "teacher_bm25_corrected"),
            tokenizer=tokenizer,
            budgets=BUDGETS,
            pipeline=pipeline,
            gold_by_sample=gold,
            progress=True,
            progress_desc="serialize held-out Teacher arms",
        )
    finally:
        pipeline.close()

    teacher = evaluated.pipeline_results[
        ["sample_id", "budget", "arm", "MRR"]
    ].copy()
    teacher["system"] = teacher["arm"].map(
        {
            "teacher_original": "gpt54_teacher",
            "teacher_bm25_corrected": "bm25_teacher",
        }
    )
    teacher = teacher.drop(columns="arm")

    nb07 = pd.read_csv(NB07_METRICS)
    students = nb07.loc[
        nb07["arm"].eq("R")
        & nb07["system"].isin(("pretrained", "bm25_treatment_e6"))
        & nb07["sample_id"].astype(str).isin(sample_ids),
        ["sample_id", "budget", "system", "MRR"],
    ].copy()
    students["system"] = students["system"].map(
        {"pretrained": "student", "bm25_treatment_e6": "bm25_student"}
    )
    expected_rows = 2 * len(BUDGETS) * len(samples)
    if len(students) != expected_rows or students.duplicated(
        ["sample_id", "budget", "system"]
    ).any():
        raise RuntimeError("NB07a student subset is incomplete.")
    metrics = pd.concat([teacher, students], ignore_index=True)
    metrics["sample_id"] = metrics["sample_id"].astype(str)
    metrics = metrics.sort_values(
        ["system", "budget", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    selection_summary = (
        selection.groupby("arm", observed=True)
        .agg(
            queries=("sample_id", "size"),
            history_tokens=("history_tokens", "sum"),
            keep_tokens=("keep_tokens", "sum"),
            mean_selected_words=("selected_words", "mean"),
            empty_selections=("empty_selection", "sum"),
        )
        .reset_index()
    )
    selection_summary["keep_share"] = (
        selection_summary["keep_tokens"] / selection_summary["history_tokens"]
    )
    return (
        metrics,
        evaluated.serialization_summary,
        selection_summary,
        score,
    )


def bio_spans(sequence: Sequence[int]) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    start = None
    for index, label in enumerate(sequence):
        if int(label) == LABEL_TO_ID["B-KEEP"]:
            if start is not None:
                spans.add((start, index))
            start = index
        elif int(label) == LABEL_TO_ID["I-KEEP"]:
            if start is None:
                start = index
        elif start is not None:
            spans.add((start, index))
            start = None
    if start is not None:
        spans.add((start, len(sequence)))
    return spans


@torch.no_grad()
def agreement_counts(
    *,
    system: str,
    reference: str,
    checkpoint: Path,
    dataset: Any,
    tokenizer: Any,
    config: SelectorExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_selector_batch(
            batch,
            int(tokenizer.pad_token_id),
        ),
        num_workers=0,
    )
    model = load_crf_inference_model(
        checkpoint,
        model_name=config.model_name,
        model_revision=config.model_revision,
        num_labels=len(LABEL_TO_ID),
        device=device,
    )
    rows = []
    from tqdm.auto import tqdm

    try:
        for batch in tqdm(
            loader,
            desc=f"agreement {system}",
            unit="batch",
            dynamic_ncols=True,
        ):
            moved = {
                key: value.to(device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            output = model(
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
                history_mask=moved["history_mask"],
                labels=moved["labels"],
            )
            _, mask, gold = gather_history_sequences(
                output["emissions"],
                moved["history_mask"],
                moved["labels"],
            )
            for index, sample_id in enumerate(batch["sample_ids"]):
                length = int(mask[index].sum().item())
                gold_sequence = gold[index, :length].cpu().tolist()
                predicted = list(output["decoded"][index][:length])
                gold_keep = np.asarray(gold_sequence) != LABEL_TO_ID["O"]
                predicted_keep = np.asarray(predicted) != LABEL_TO_ID["O"]
                gold_spans = bio_spans(gold_sequence)
                predicted_spans = bio_spans(predicted)
                rows.append(
                    {
                        "sample_id": str(sample_id),
                        "system": system,
                        "reference": reference,
                        "keep_tp": int(np.sum(gold_keep & predicted_keep)),
                        "keep_fp": int(np.sum(~gold_keep & predicted_keep)),
                        "keep_fn": int(np.sum(gold_keep & ~predicted_keep)),
                        "span_tp": len(gold_spans & predicted_spans),
                        "span_fp": len(predicted_spans - gold_spans),
                        "span_fn": len(gold_spans - predicted_spans),
                    }
                )
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = pd.DataFrame(rows)
    if len(result) != 2_104 or result["sample_id"].duplicated().any():
        raise RuntimeError(f"Agreement population failed for {system}.")
    return result


def f1_from_counts(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def score(offset: int) -> np.ndarray:
        tp, fp, fn = values[..., offset], values[..., offset + 1], values[..., offset + 2]
        denominator = 2.0 * tp + fp + fn
        return np.divide(
            2.0 * tp,
            denominator,
            out=np.zeros_like(tp, dtype=float),
            where=denominator > 0,
        )

    return score(0), score(3)


def bootstrap_agreement(counts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    columns = ["keep_tp", "keep_fp", "keep_fn", "span_tp", "span_fp", "span_fn"]
    for system, group in counts.groupby("system", observed=True, sort=False):
        group = group.sort_values("sample_id", kind="mergesort")
        values = group[columns].to_numpy(dtype=np.int64)
        point = values.sum(axis=0, keepdims=True)
        keep, span = f1_from_counts(point)
        seed = int.from_bytes(
            hashlib.sha256(f"teacher_agreement_v1\0{system}".encode()).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(seed)
        keep_draws = np.empty(BOOTSTRAP_REPLICATES)
        span_draws = np.empty(BOOTSTRAP_REPLICATES)
        for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK):
            stop = min(start + BOOTSTRAP_CHUNK, BOOTSTRAP_REPLICATES)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            sampled = values[indices].sum(axis=1)
            keep_draws[start:stop], span_draws[start:stop] = f1_from_counts(sampled)
        reference = str(group["reference"].iloc[0])
        for metric, value, draws in (
            ("KEEP F1", float(keep[0]), keep_draws),
            ("Span F1", float(span[0]), span_draws),
        ):
            low, high = np.quantile(draws, (0.025, 0.975))
            rows.append(
                {
                    "system": system,
                    "reference": reference,
                    "metric": metric,
                    "n": len(values),
                    "value": value,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "bootstrap_seed": seed,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                }
            )
    return pd.DataFrame(rows)


def agreement_evaluation(
    *,
    teacher_rows: Sequence[dict[str, Any]],
    score: Any,
    config: SelectorExperimentConfig,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = tuple(str(row["sample_id"]) for row in teacher_rows)
    prepared = prepare_selector_datasets(
        teacher_rows,
        {"dev": ids},
        config,
        class_weight_split="dev",
        dataset_sha256=sha256(TEACHER_LABELS),
        label_spaces=("collapsed",),
        progress=True,
    )
    original = prepared["collapsed"]["dev"]
    tokenizer = prepared["tokenizer"]
    scored = prepare_gold_score_selector_dataset(
        rows=teacher_rows,
        base_dataset=original,
        tokenizer=tokenizer,
        score_result=score,
        selector_config=config,
        exclude_stopwords=True,
        require_complete_train=False,
        progress=True,
    )
    corrected = hard_relabel_gold_score_dataset(scored)
    counts = pd.concat(
        [
            agreement_counts(
                system="student",
                reference="GPT-5.4 Teacher",
                checkpoint=PRETRAINED_CHECKPOINT,
                dataset=original,
                tokenizer=tokenizer,
                config=config,
                device=device,
            ),
            agreement_counts(
                system="bm25_student",
                reference="BM25 Teacher",
                checkpoint=TREATMENT_CHECKPOINT,
                dataset=corrected,
                tokenizer=tokenizer,
                config=config,
                device=device,
            ),
        ],
        ignore_index=True,
    )
    return counts, bootstrap_agreement(counts)


def bootstrap_retrieval(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    arrays: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    for (system, budget), group in metrics.groupby(
        ["system", "budget"], observed=True, sort=False
    ):
        group = group.sort_values("sample_id", kind="mergesort")
        ids = group["sample_id"].astype(str).to_numpy()
        values = group["MRR"].to_numpy(dtype=float)
        if len(values) != 2_104 or len(set(ids)) != len(ids):
            raise RuntimeError(f"Retrieval population failed for {system}/B{budget}.")
        arrays[(str(system), int(budget))] = (ids, values)
        seed = int.from_bytes(
            hashlib.sha256(f"teacher_mrr_v1\0{system}\0{budget}".encode()).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(seed)
        draws = np.empty(BOOTSTRAP_REPLICATES)
        for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK):
            stop = min(start + BOOTSTRAP_CHUNK, BOOTSTRAP_REPLICATES)
            indices = rng.integers(0, len(values), size=(stop - start, len(values)))
            draws[start:stop] = values[indices].mean(axis=1)
        low, high = np.quantile(draws, (0.025, 0.975))
        summaries.append(
            {
                "system": system,
                "budget": int(budget),
                "n": len(values),
                "MRR": float(values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_seed": seed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )

    comparisons = []
    pairs = []
    for budget in BUDGETS:
        pairs.extend(
            [
                ("student_minus_gpt54_teacher", "student", "gpt54_teacher", budget),
                ("bm25_teacher_minus_gpt54_teacher", "bm25_teacher", "gpt54_teacher", budget),
            ]
        )
    pairs.append(("bm25_student_minus_bm25_teacher", "bm25_student", "bm25_teacher", 64))
    for name, left, right, budget in pairs:
        left_ids, left_values = arrays[(left, budget)]
        right_ids, right_values = arrays[(right, budget)]
        if not np.array_equal(left_ids, right_ids):
            raise RuntimeError(f"Paired population mismatch: {name}.")
        difference = left_values - right_values
        seed = int.from_bytes(hashlib.sha256(f"teacher_delta_v1\0{name}".encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        draws = np.empty(BOOTSTRAP_REPLICATES)
        for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK):
            stop = min(start + BOOTSTRAP_CHUNK, BOOTSTRAP_REPLICATES)
            indices = rng.integers(0, len(difference), size=(stop - start, len(difference)))
            draws[start:stop] = difference[indices].mean(axis=1)
        low, high = np.quantile(draws, (0.025, 0.975))
        comparisons.append(
            {
                "comparison": name,
                "budget": budget,
                "left_system": left,
                "right_system": right,
                "n": len(difference),
                "delta_mrr": float(difference.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_seed": seed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(comparisons)


def main() -> None:
    identity = source_identity()
    if reusable(identity):
        print(f"Reusing complete Teacher evaluation: {RESULT_DIR}", flush=True)
        return
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if DEVICE == "auto" else DEVICE
    )
    print(f"Building held-out Teacher evaluation on {device}.", flush=True)
    samples, teacher_rows, gold, positive, cohort = load_population()
    config = selector_config()
    metrics, serialization, selection, score = retrieval_evaluation(
        samples=samples,
        teacher_rows=teacher_rows,
        gold=gold,
        positive=positive,
        config=config,
        device=device,
    )
    retrieval_summary, comparisons = bootstrap_retrieval(metrics)
    agreement_counts_frame, agreement_summary = agreement_evaluation(
        teacher_rows=teacher_rows,
        score=score,
        config=config,
        device=device,
    )

    write_frame("cohort.csv", cohort)
    write_frame("retrieval_metrics_by_query.csv", metrics)
    write_frame("retrieval_summary.csv", retrieval_summary)
    write_frame("paired_comparisons.csv", comparisons)
    write_frame("agreement_counts_by_query.csv", agreement_counts_frame)
    write_frame("agreement_summary.csv", agreement_summary)
    write_frame("selection_summary.csv", selection)
    write_frame("serialization_summary.csv", serialization)
    paths = [RESULT_DIR / name for name in OUTPUT_FILES]
    manifest = {
        "schema_version": 2,
        "identity": identity,
        "complete": True,
        "device": str(device),
        "cohort_sample_id_sha256": text_sha256(
            "\n".join(sorted(cohort["sample_id"].astype(str)))
        ),
        "teacher_api_called": False,
        "training_performed": False,
        "single_history_queries_evaluated": False,
        "no_history_queries_evaluated": False,
        "files": {path.name: sha256(path) for path in paths},
        "gold_bm25_score_sha256": score.dataset_sha256,
    }
    write_json(RESULT_DIR / "manifest.json", manifest)
    print(f"Teacher evaluation complete: {RESULT_DIR}", flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    configure_paths(arguments)
    check_inputs()
    if not arguments.check_inputs:
        main()
