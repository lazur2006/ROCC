#!/usr/bin/env python3
"""Profile population-derived TopiOCQA T5-encoder shapes (six in the thesis)."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from math import isclose
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(os.environ.get("ROCC_ROOT") or os.environ.get(
    "MASTER_THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]
)).expanduser().resolve()
MODEL_DIR = PROJECT_ROOT / "experiments/model/IterCQR/IterCQR Model"
OUTPUT_DIR = (
    PROJECT_ROOT / "experiments/results/10_efficiency_analysis/nsight"
)
MAIN_INPUTS_PATH = (
    PROJECT_ROOT
    / "experiments/results/07a_topiocqa_final"
    / "query_bundle/serialization/main_inputs.csv"
)

BATCH_SIZE = 16
DUMMY_TOKEN_ID = 42
EXPECTED_POPULATION_QUERIES = 2514
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4070 Laptop GPU"
EXPECTED_MAIN_INPUTS_SHA256 = (
    "8fe7f909e08d2ad1bb83544324b70af"
    "3150ae9d4d079643f8d6e5b3b34f3982e"
)
EXPECTED_MODEL_SHA256 = (
    "1a25ceaf597bf92ff2897cccb22e9c46b"
    "807efbf78a3234186066e1f62f16ae3"
)
EXPECTED_CONFIG_SHA256 = (
    "a1f20ac0527995010a72aa51a3c7dd1f"
    "ec6bcb15c0d69bcbdca5611f035ee2fa"
)

RANGE_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes.sum",
    "dram__bytes.sum.per_second",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
]
FP32_COUNTER_METRICS = [
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
]
KERNEL_COUNTER_METRICS = ["dram__bytes.sum", *FP32_COUNTER_METRICS]
COMPONENT_DOMAIN = "rocc-t5-encoder-components"
COMPONENTS = (
    "embedding",
    "self_attention",
    "ffn",
    "layernorm_residual_other",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _topiocqa_profile_configs() -> list[tuple[str, str, int]]:
    """Round the full-Dev mean length per condition, as in the original profile."""

    if _lineage_mode() == "historical" and _sha256_file(MAIN_INPUTS_PATH) != EXPECTED_MAIN_INPUTS_SHA256:
        raise RuntimeError("The frozen NB07a main_inputs.csv has changed.")
    if (
        _sha256_file(MODEL_DIR / "pytorch_model.bin")
        != EXPECTED_MODEL_SHA256
    ):
        raise RuntimeError("The frozen IterCQR checkpoint has changed.")
    if (
        _sha256_file(MODEL_DIR / "config.json")
        != EXPECTED_CONFIG_SHA256
    ):
        raise RuntimeError("The frozen IterCQR configuration has changed.")

    frame = pd.read_csv(
        MAIN_INPUTS_PATH,
        usecols=["sample_id", "budget", "arm", "input_length"],
    )
    expected_lengths = {
        ("I", 64): 57,
        ("I", 128): 98,
        ("I", 256): 141,
        ("I", 512): 150,
        ("pretrained_R", 64): 41,
        ("pretrained_R", 128): 42,
        ("pretrained_R", 256): 42,
        ("pretrained_R", 512): 42,
    }
    lengths = {}
    population = None
    for (arm, budget), expected_length in expected_lengths.items():
        rows = frame.loc[
            frame["arm"].eq(arm) & frame["budget"].eq(budget)
        ]
        if (len(rows) != EXPECTED_POPULATION_QUERIES
                or rows["sample_id"].nunique() != EXPECTED_POPULATION_QUERIES
                or rows["input_length"].isna().any()
                or not rows["input_length"].between(1, budget).all()):
            raise RuntimeError(
                f"Unexpected TopiOCQA Dev population for {arm}, B{budget}."
            )
        ids = set(rows["sample_id"].astype(str))
        if population is None:
            population = ids
        elif ids != population:
            raise RuntimeError("I/R profiling conditions use different populations.")
        observed_length = int(round(rows["input_length"].mean()))
        lengths[(arm, budget)] = observed_length
        if _lineage_mode() == "historical" and observed_length != expected_length:
            raise RuntimeError(
                f"Unexpected canonical length for {arm}, B{budget}: "
                f"{observed_length} != {expected_length}."
            )

    population_manifest = json.loads((MAIN_INPUTS_PATH.parents[2] / "protocol/population.json").read_text())
    observed_population = hashlib.sha256("\n".join(sorted(population)).encode()).hexdigest()
    if observed_population != population_manifest.get("sample_id_sha256"):
        raise RuntimeError("Profiling inputs disagree with the NB07a population.")
    profiles = [("I", f"B{budget}", lengths[("I", budget)]) for budget in (64, 128, 256, 512)]
    profiles.append(("R", "B64", lengths[("pretrained_R", 64)]))
    if len({lengths[("pretrained_R", b)] for b in (128, 256, 512)}) == 1:
        profiles.append(("R", "B128-B512", lengths[("pretrained_R", 128)]))
    else:
        profiles.extend(("R", f"B{b}", lengths[("pretrained_R", b)]) for b in (128, 256, 512))
    return profiles


def _lineage_mode() -> str:
    mode = os.environ.get("ROCC_LINEAGE", "historical").strip().lower()
    if mode not in {"historical", "local"}:
        raise ValueError("ROCC_LINEAGE must be historical or local.")
    return mode


def _configure_ncu(ncu_dir: Path | None) -> None:
    """Use the user's installation, never a machine-specific home path."""
    if ncu_dir is not None:
        ncu_dir = ncu_dir.expanduser().resolve()
        binary_dir = ncu_dir if (ncu_dir / "ncu").is_file() else ncu_dir / "bin"
        if not (binary_dir / "ncu").is_file():
            raise FileNotFoundError("--ncu-dir / NCU_HOME must contain ncu or bin/ncu.")
        os.environ["PATH"] = str(binary_dir) + os.pathsep + os.environ.get("PATH", "")
    binary = shutil.which("ncu")
    if binary is None:
        raise RuntimeError("Install NVIDIA Nsight Compute and set NCU_HOME or put ncu on PATH.")
    version_text = subprocess.run([binary, "--version"], check=True, capture_output=True, text=True).stdout
    match = re.search(r"Version\s+(\d+)\.(\d+)\.(\d+)", version_text, re.IGNORECASE)
    if match is None or tuple(map(int, match.groups())) < (2026, 2, 1):
        raise RuntimeError("nsight-python requires Nsight Compute 2026.2.1 or newer. Set NCU_HOME / --ncu-dir to that installation and check ncu --version.")
    binary_dir = Path(binary).resolve().parent
    roots = ([ncu_dir] if ncu_dir else []) + [binary_dir, binary_dir.parent]
    for root in roots:
        report_python = root / "extras/python"
        if report_python.is_dir() and str(report_python) not in sys.path:
            sys.path.insert(0, str(report_python))


def _require_expected_gpu() -> None:
    command = [
        "nvidia-smi", "--query-gpu=name,persistence_mode",
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
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError("GPU selection is ambiguous. Set CUDA_VISIBLE_DEVICES to the GPU number or GPU-UUID to measure as CUDA device 0.")
    expected = f"{EXPECTED_GPU_NAME}, Enabled"
    if (not rows[0].endswith(", Enabled")
            or (_lineage_mode() == "historical" and rows != [expected])):
        raise RuntimeError(
            "Expected one GPU with Persistence Mode enabled (RTX 4070 Laptop "
            f"for historical lineage); observed: {rows}."
        )


def _annotation(route: str, budget_label: str) -> str:
    normalized_budget = budget_label.lower().replace("-", "_")
    return f"topiocqa_t5_encoder_{route.lower()}_{normalized_budget}"


def profile_itercqr_t5_encoder_range(
    route: str,
    budget_label: str,
    sequence_length: int,
) -> None:
    """Measure time and DRAM traffic for the complete encoder range."""

    _run_encoder(
        annotation=_annotation(route, budget_label),
        sequence_length=sequence_length,
    )


def _sum_kernel_metrics(left: float, right: float) -> float:
    """Add counters from every kernel launched by the T5 encoder."""

    return left + right


def profile_itercqr_t5_encoder_fp32(
    route: str,
    budget_label: str,
    sequence_length: int,
) -> None:
    """Collect additive DRAM and FP32 counters for every encoder kernel."""

    _run_encoder(
        annotation=_annotation(route, budget_label),
        sequence_length=sequence_length,
    )


def _run_encoder(annotation: str, sequence_length: int) -> None:
    """Load IterCQR and execute one warm-up and one measured encoder pass."""

    device = torch.device("cuda:0")
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    ).to(device)
    model.eval()
    encoder = model.get_encoder()

    input_ids = torch.full(
        (BATCH_SIZE, sequence_length),
        DUMMY_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        # Warm-up is intentionally outside the measured annotation.
        encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        torch.cuda.synchronize(device)

        # Plain NVTX child ranges partition kernels without nesting the
        # nsight-python annotation itself (which Nsight Python forbids).
        hooks = _component_hooks(encoder)
        try:
            with nsight.annotate(annotation):
                encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
        finally:
            for hook in hooks:
                hook.remove()

        torch.cuda.synchronize(device)


def _component_hooks(encoder) -> list:
    """Mark three ranges; unmarked kernels form the fourth component."""

    hooks = []

    def register(module, component: str) -> None:
        contexts = []

        def enter(_module, _inputs) -> None:
            context = nvtx.annotate(component, domain=COMPONENT_DOMAIN)
            context.__enter__()
            contexts.append(context)

        def exit_range(_module, _inputs, _output) -> None:
            contexts.pop().__exit__(None, None, None)

        hooks.append(module.register_forward_pre_hook(enter))
        hooks.append(
            module.register_forward_hook(exit_range, always_call=True)
        )

    register(encoder.embed_tokens, "embedding")
    for block in encoder.block:
        register(block.layer[0].SelfAttention, "self_attention")
        register(block.layer[-1].DenseReluDense, "ffn")
    return hooks


def _component_metrics(report_path: Path, config: tuple) -> pd.DataFrame:
    """Read per-kernel counters and aggregate the exhaustive NVTX partition."""

    import ncu_report

    report = ncu_report.load_report(str(report_path))
    if len(report) != 1:
        raise RuntimeError(f"Expected one range in {report_path}.")
    route, budget_label, sequence_length = config
    expected_outer = _annotation(route, budget_label)
    rows = []
    for action in report[0]:
        component = "layernorm_residual_other"
        state = action.nvtx_state()
        marked = []
        outer = []
        if state is not None:
            for domain_id in state.domains():
                domain = state[domain_id]
                if domain.name() == COMPONENT_DOMAIN:
                    marked.extend(domain.push_pop_ranges())
                elif domain.name() == "nsight-python":
                    outer.extend(domain.push_pop_ranges())
        if outer != [expected_outer]:
            raise RuntimeError(f"Wrong outer range in {report_path}: {outer}.")
        if len(marked) > 1:
            raise RuntimeError(
                f"Overlapping encoder component ranges: {marked}."
            )
        if marked:
            component = marked[0]
        if component not in COMPONENTS:
            raise RuntimeError(f"Unknown encoder component: {component}.")
        values = {
            metric: float(action[metric].value())
            for metric in KERNEL_COUNTER_METRICS
        }
        rows.append({"component": component, **values})

    kernels = pd.DataFrame(rows)
    if kernels.empty or not set(COMPONENTS[:3]).issubset(
        kernels["component"]
    ):
        raise RuntimeError(f"Incomplete component ranges in {report_path}.")
    grouped = kernels.groupby("component", sort=False)
    components = []
    for component in COMPONENTS:
        group = (
            grouped.get_group(component)
            if component in grouped.groups
            else kernels.iloc[:0]
        )
        counters = {
            metric: float(group[metric].sum())
            for metric in KERNEL_COUNTER_METRICS
        }
        components.append(
            {
                "route": route,
                "budget_label": budget_label,
                "sequence_length": sequence_length,
                "component": component,
                "kernels": len(group),
                **counters,
                "fp32_flops": (
                    counters[FP32_COUNTER_METRICS[0]]
                    + counters[FP32_COUNTER_METRICS[1]]
                    + 2.0 * counters[FP32_COUNTER_METRICS[2]]
                ),
            }
        )
    return pd.DataFrame(components)


def main() -> None:
    global OUTPUT_DIR, nsight, nvtx, torch, T5ForConditionalGeneration
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--ncu-dir", type=Path, default=os.environ.get("NCU_HOME"), help="Nsight Compute installation directory (or NCU_HOME / ncu on PATH).")
    parser.add_argument("--dry-run", action="store_true", help="Validate local inputs and list encoder shapes; no profiling, CUDA, model loading or output writes.")
    args = parser.parse_args()
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    configs = _topiocqa_profile_configs()
    if args.dry_run:
        print(json.dumps({"lineage": _lineage_mode(), "batch_size": BATCH_SIZE, "profiles": configs}, indent=2))
        return
    if any(OUTPUT_DIR.glob("*.ncu-rep")) or (OUTPUT_DIR / "topiocqa_t5_encoder_summary.csv").exists():
        raise FileExistsError("Profiling output already exists; keep it and use a new --output-dir.")
    _configure_ncu(args.ncu_dir)
    # Required before torch, transformers, nvtx or any CUDA/NVTX calls.
    try:
        nsight = importlib.import_module("nsight")
        major = int(importlib.metadata.version("nsight-python").split(".")[0])
        if major < 1:
            raise RuntimeError("This entry point requires nsight-python >= 1.0 (in-process profiling).")
        nvtx = importlib.import_module("nvtx")
        importlib.import_module("ncu_report")
    except ImportError as exc:
        raise RuntimeError("Install nsight-python>=1.0 and nvtx, and expose Nsight Compute extras/python.") from exc
    torch = importlib.import_module("torch")
    T5ForConditionalGeneration = importlib.import_module("transformers").T5ForConditionalGeneration
    _require_expected_gpu()
    if not torch.cuda.is_available():
        raise RuntimeError("Encoder profiling requires a CUDA-enabled PyTorch build and an available CUDA GPU.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    range_profiler = nsight.analyze.kernel(
        metrics=RANGE_METRICS, runs=1, replay_mode="range", clock_control="base",
        cache_control="all", thermal_mode="auto", output_csv=True,
        output_prefix=str(OUTPUT_DIR / "topiocqa_t5_encoder_range_"),
    )(profile_itercqr_t5_encoder_range)
    kernel_profiler = nsight.analyze.kernel(
        metrics=KERNEL_COUNTER_METRICS, runs=1, replay_mode="kernel",
        combine_kernel_metrics=_sum_kernel_metrics, clock_control="base",
        cache_control="all", thermal_mode="auto", output_csv=True,
        output_prefix=str(OUTPUT_DIR / "topiocqa_t5_encoder_fp32_"),
    )(profile_itercqr_t5_encoder_fp32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    # Profile each shape separately. nsight-python associates one NVTX
    # annotation with every config passed to a single call; separate calls
    # therefore avoid ambiguous extraction and permit shape-dependent kernel
    # counts in the kernel-replay pass.
    range_frames = []
    for config in tqdm(
        configs,
        desc="Nsight DRAM ranges",
        unit="shape",
        dynamic_ncols=True,
    ):
        result = range_profiler(configs=[config])
        range_frames.append(result.to_dataframe())
    range_metrics = pd.concat(range_frames, ignore_index=True)

    fp32_frames = []
    component_frames = []
    kernel_dram_by_config = {}
    for report_index, config in enumerate(
        tqdm(
            configs,
            desc="Nsight additive kernel counters",
            unit="shape",
            dynamic_ncols=True,
        )
    ):
        result = kernel_profiler(configs=[config])
        frame = result.to_dataframe()
        kernel_dram_by_config[config] = float(
            frame.loc[
                frame["Metric"].eq("dram__bytes.sum"), "AvgValue"
            ].iloc[0]
        )
        fp32_frames.append(
            frame.loc[frame["Metric"].isin(FP32_COUNTER_METRICS)]
        )
        report_path = OUTPUT_DIR / (
            "topiocqa_t5_encoder_fp32_ncu-output-"
            "profile_itercqr_t5_encoder_fp32-"
            f"{report_index}.ncu-rep"
        )
        component_frames.append(_component_metrics(report_path, config))
    fp32_metrics = pd.concat(fp32_frames, ignore_index=True)
    component_metrics = pd.concat(component_frames, ignore_index=True)
    raw_metrics = pd.concat(
        [range_metrics, fp32_metrics],
        ignore_index=True,
    )
    raw_metrics.to_csv(
        OUTPUT_DIR / "topiocqa_t5_encoder_raw_metrics.csv",
        index=False,
    )
    summary_rows = []
    for route, budget_label, sequence_length in configs:
        annotation = _annotation(route, budget_label)
        metrics = raw_metrics.loc[
            raw_metrics["Annotation"].eq(annotation)
            & raw_metrics["route"].eq(route)
            & raw_metrics["budget_label"].eq(budget_label)
            & raw_metrics["sequence_length"].eq(sequence_length)
        ]
        if set(metrics["Metric"]) != set(
            RANGE_METRICS + FP32_COUNTER_METRICS
        ):
            raise RuntimeError(
                f"Incomplete NVIDIA metrics for {route}, {budget_label}."
            )
        values = metrics.set_index("Metric")["AvgValue"].astype(float)
        range_rows = metrics.loc[metrics["Metric"].isin(RANGE_METRICS)]
        if not range_rows["NumRuns"].eq(1).all():
            raise RuntimeError(
                f"Expected one range run for {route}, {budget_label}."
            )
        fp32_rows = metrics.loc[
            metrics["Metric"].isin(FP32_COUNTER_METRICS)
        ]
        if not fp32_rows["NumRuns"].eq(1).all():
            raise RuntimeError(
                f"Expected one FP32 counter run for {route}, "
                f"{budget_label}."
            )
        duration_row = metrics.loc[
            metrics["Metric"].eq("gpu__time_duration.sum")
        ].iloc[0]
        duration_ns = values["gpu__time_duration.sum"]
        duration_seconds = duration_ns * 1e-9
        dram_bytes = values["dram__bytes.sum"]
        kernel_dram_bytes = kernel_dram_by_config[
            (route, budget_label, sequence_length)
        ]
        # https://archive.docs.nvidia.com/nsight-compute/2024.2/NsightComputeCli/index.html#id29
        fp32_flops = (
            values[
                "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
            ]
            + values[
                "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
            ]
            + 2.0
            * values[
                "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
            ]
        )
        shape_components = component_metrics.loc[
            component_metrics["route"].eq(route)
            & component_metrics["budget_label"].eq(budget_label)
            & component_metrics["sequence_length"].eq(sequence_length)
        ]
        component_flops = float(shape_components["fp32_flops"].sum())
        component_dram = float(
            shape_components["dram__bytes.sum"].sum()
        )
        if not (
            isclose(component_flops, fp32_flops, rel_tol=1e-12)
            and isclose(
                component_dram,
                kernel_dram_bytes,
                rel_tol=1e-12,
            )
        ):
            raise RuntimeError(
                f"Component closure failed for {route}, {budget_label}."
            )
        summary_rows.append(
            {
                "annotation": annotation,
                "route": route,
                "budget_label": budget_label,
                "sequence_length": sequence_length,
                "batch_size": BATCH_SIZE,
                "population_queries": EXPECTED_POPULATION_QUERIES,
                "duration_ns": duration_ns,
                "fp32_flops": fp32_flops,
                # Range replay preserves inter-kernel cache reuse; the
                # component partition comes from cold-cache kernel replay.
                "range_replay_dram_bytes": dram_bytes,
                "dram_bytes": dram_bytes,
                "kernel_replay_dram_bytes": kernel_dram_bytes,
                "component_fp32_flops_sum": component_flops,
                "component_kernel_dram_bytes_sum": component_dram,
                "component_fp32_closure_error": (
                    component_flops - fp32_flops
                ),
                "component_kernel_dram_closure_error": (
                    component_dram - kernel_dram_bytes
                ),
                "kernel_vs_range_dram_delta_bytes": (
                    kernel_dram_bytes - dram_bytes
                ),
                "kernel_vs_range_dram_delta_pct": (
                    100.0 * (kernel_dram_bytes - dram_bytes) / dram_bytes
                ),
                "dram_bandwidth_gb_per_s": (
                    values["dram__bytes.sum.per_second"] / 1e9
                ),
                "dram_throughput_pct_of_peak_sustained_elapsed": values[
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed"
                ],
                "arithmetic_intensity_flop_per_byte": (
                    fp32_flops / dram_bytes
                ),
                "achieved_fp32_tflop_per_s": (
                    fp32_flops / duration_seconds / 1e12
                ),
                "gpu": duration_row["GPU"],
                "compute_clock_khz": int(duration_row["ComputeClock"]),
                "memory_clock_khz": int(duration_row["MemoryClock"]),
                "range_runs": 1,
                "fp32_counter_runs": 1,
            }
        )
    component_metrics.to_csv(
        OUTPUT_DIR / "topiocqa_t5_encoder_component_metrics.csv",
        index=False,
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUTPUT_DIR / "topiocqa_t5_encoder_summary.csv",
        index=False,
    )
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps({
        "complete": True, "lineage": _lineage_mode(), "batch_size": BATCH_SIZE,
        "profiles": configs, "input_sha256": _sha256_file(MAIN_INPUTS_PATH),
        "model_sha256": _sha256_file(MODEL_DIR / "pytorch_model.bin"),
        "nsight_python": importlib.metadata.version("nsight-python"),
        "torch": torch.__version__, "tf32": False,
        "warmup_encoder_passes": 1, "range_runs": 1, "kernel_runs": 1,
        "clock_control": "base", "cache_control": "all",
        "scope": "synthetic fixed-shape T5 encoder only; no decoder, selector or retrieval",
    }, indent=2) + "\n", encoding="utf-8")

    print("Raw NVIDIA metrics:")
    print(
        raw_metrics[
            ["Annotation", "Metric", "AvgValue", "Unit"]
        ].to_string(index=False)
    )
    print("\nAdditive encoder component metrics:")
    print(
        component_metrics[
            [
                "route",
                "budget_label",
                "component",
                "kernels",
                "fp32_flops",
                "dram__bytes.sum",
            ]
        ].to_string(index=False)
    )
    print("\nDerived whole-encoder values:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
