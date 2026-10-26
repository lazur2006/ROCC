#!/usr/bin/env python3
"""Run and aggregate five independent full-dev latency invocations."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("ROCC_ROOT") or os.environ.get(
    "MASTER_THESIS_PROJECT_ROOT", Path(__file__).resolve().parents[2]
)).expanduser().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT_ROOT = ROOT / "experiments/results/10_efficiency_analysis/latency"
BENCHMARK = Path(__file__).with_name("benchmark_topiocqa_full_dev_latency.py")
RUNS = 5
T95_DF4 = 2.7764451051977987
MEASURES = (
    "rocc_ms_per_query",
    "t5_encoder_ms_per_query",
    "t5_decoder_ms_per_query",
    "total_ms_per_query",
)
EXPECTED_CONDITIONS = {
    ("I", 64),
    ("I", 512),
    ("R", 64),
    ("D", 64),
    ("IRD", 64),
}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_benchmarks(*, dry_run: bool = False) -> None:
    if dry_run:
        subprocess.run([sys.executable, str(BENCHMARK), "--run-index", "1", "--dry-run"], cwd=ROOT, check=True)
        return
    from experiments.scripts.benchmark_topiocqa_full_dev_latency import preflight, lineage_mode
    identity, gpu = preflight()
    for run in range(1, RUNS + 1):
        manifest_path = OUTPUT_ROOT / "runs" / f"run_{run}" / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("complete") is True
                and manifest.get("run_index") == run
                and manifest.get("within_invocation_repeats") == 1
            ):
                if (manifest.get("input_sha256") != identity or manifest.get("gpu") != gpu
                        or manifest.get("lineage", "historical") != lineage_mode()
                        or not (manifest_path.parent / "route_components_summary.csv").is_file()):
                    raise RuntimeError(f"Run {run} belongs to different inputs/settings or is incomplete. Use a new --output-dir.")
                print(
                    f"\n=== Full-dev latency run {run}/{RUNS} already complete ===",
                    flush=True,
                )
                continue
        print(f"\n=== Full-dev latency run {run}/{RUNS} ===", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(BENCHMARK),
                "--run-index",
                str(run),
                "--output-dir",
                str(OUTPUT_ROOT),
            ],
            cwd=ROOT,
            check=True,
        )


def aggregate() -> None:
    frames = []
    manifests = []
    for run in range(1, RUNS + 1):
        run_dir = OUTPUT_ROOT / "runs" / f"run_{run}"
        frame = pd.read_csv(run_dir / "route_components_summary.csv")
        manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        if (
            manifest.get("complete") is not True
            or manifest.get("within_invocation_repeats") != 1
            or manifest.get("run_index") != run
        ):
            raise RuntimeError(f"Run {run} manifest is invalid.")
        if (len(frame) != len(EXPECTED_CONDITIONS)
                or frame.duplicated(["route", "budget"]).any()
                or not frame["independent_run"].eq(run).all()
                or not frame["queries"].eq(2514).all()
                or not frame["within_invocation_repeats"].eq(1).all()):
            raise RuntimeError(f"Run {run} does not contain exactly the five full-dev conditions.")
        if (manifest.get("population_queries") != 2514
                or manifest.get("rocc_processed_queries") != 2309
                or manifest.get("history_zero_queries") != 205
                or manifest.get("batch_size") != 16
                or manifest.get("warmup_batches_per_component") != 5):
            raise RuntimeError(f"Run {run} measurement protocol changed.")
        if manifests:
            for key in ("input_sha256", "gpu", "population_sha256", "ird_b64", "tf32", "python", "torch", "lineage"):
                if manifest.get(key) != manifests[0].get(key):
                    raise RuntimeError(f"Run {run} has inconsistent {key}; do not mix experiment chains.")
        frames.append(frame)
        manifests.append(manifest)

    runs = pd.concat(frames, ignore_index=True)
    observed = set(
        runs[["route", "budget"]].itertuples(index=False, name=None)
    )
    if observed != EXPECTED_CONDITIONS:
        raise RuntimeError(f"Unexpected latency conditions: {observed}")
    if (
        runs.groupby(["route", "budget"])["independent_run"]
        .nunique()
        .ne(RUNS)
        .any()
    ):
        raise RuntimeError("Not every condition has five independent runs.")
    if runs[list(MEASURES)].isna().any().any():
        raise RuntimeError("Latency measurements contain missing values.")
    if not runs[list(MEASURES)].map(lambda value: math.isfinite(value) and value >= 0).all().all():
        raise RuntimeError("Latency measurements must be finite and nonnegative.")
    component_error = (
        runs["total_ms_per_query"]
        - runs[list(MEASURES[:-1])].sum(axis=1)
    ).abs()
    if not component_error.lt(1e-9).all():
        raise RuntimeError("Latency component sums are invalid.")

    grouped = runs.groupby(["route", "budget"], sort=False)
    summary = grouped.size().rename("runs").reset_index()
    for source, short in zip(
        MEASURES,
        ("rocc", "t5_encoder", "t5_decoder", "total"),
        strict=True,
    ):
        stats = grouped[source].agg(["mean", "std", "min", "max"]).reset_index()
        half_width = T95_DF4 * stats["std"] / math.sqrt(RUNS)
        stats["ci95_low"] = stats["mean"] - half_width
        stats["ci95_high"] = stats["mean"] + half_width
        summary = summary.merge(
            stats.rename(
                columns={
                    "mean": f"mean_{short}_ms",
                    "std": f"std_{short}_ms",
                    "min": f"min_{short}_ms",
                    "max": f"max_{short}_ms",
                    "ci95_low": f"ci95_low_{short}_ms",
                    "ci95_high": f"ci95_high_{short}_ms",
                }
            ),
            on=["route", "budget"],
            validate="one_to_one",
        )
    summary["cv_total_percent"] = (
        100 * summary["std_total_ms"] / summary["mean_total_ms"]
    )
    summary["component_sum_error_ms"] = (
        summary["mean_total_ms"]
        - summary[
            ["mean_rocc_ms", "mean_t5_encoder_ms", "mean_t5_decoder_ms"]
        ].sum(axis=1)
    ).abs()

    pivot = runs.pivot(
        index="independent_run",
        columns=["route", "budget"],
        values="total_ms_per_query",
    )
    comparisons = []
    reference = pivot[("I", 512)]
    for route, budget in (("I", 64), ("R", 64), ("D", 64), ("IRD", 64)):
        deltas = pivot[(route, budget)] - reference
        mean = float(deltas.mean())
        std = float(deltas.std(ddof=1))
        half_width = T95_DF4 * std / math.sqrt(RUNS)
        comparisons.append(
            {
                "condition": f"{route}-B{budget}",
                "reference": "I-B512",
                "runs": RUNS,
                "mean_delta_ms_per_query": mean,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }
        )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(
        OUTPUT_ROOT / "reproducibility_route_components_runs.csv",
        index=False,
    )
    summary.to_csv(
        OUTPUT_ROOT / "reproducibility_route_components_summary.csv",
        index=False,
    )
    pd.DataFrame(comparisons).to_csv(
        OUTPUT_ROOT / "latency_vs_i512_paired_t_intervals.csv",
        index=False,
    )
    atomic_json(
        OUTPUT_ROOT / "reproducibility_manifest.json",
        {
            "schema_version": 1,
            "lineage": manifests[0].get("lineage", "historical"),
            "sample": "complete TopiOCQA development split",
            "population_queries": 2_514,
            "independent_script_invocations": RUNS,
            "within_invocation_repeats": 1,
            "total_measurements_per_condition": RUNS,
            "conditions": [
                "I-B64",
                "I-B512",
                "R-B64",
                "D-B64",
                "IRD-B64",
            ],
            "scope": "prepared inputs -> model outputs; no retrieval/RRF",
            "run_invocation_wall_seconds": [
                float(manifest["invocation_wall_seconds"])
                for manifest in manifests
            ],
            "all_integrity_checks_passed": True,
        },
    )
    print("\n=== Aggregated full-dev latency ===", flush=True)
    print(
        summary[
            [
                "route",
                "budget",
                "runs",
                "mean_rocc_ms",
                "mean_t5_encoder_ms",
                "mean_t5_decoder_ms",
                "mean_total_ms",
                "ci95_low_total_ms",
                "ci95_high_total_ms",
                "cv_total_percent",
            ]
        ].to_string(index=False),
        flush=True,
    )


def main() -> None:
    global OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--aggregate-only", action="store_true", help="Recompute summaries from five existing runs, without loading models or accessing CUDA.")
    modes.add_argument("--dry-run", action="store_true", help="Validate NB05/NB07a inputs only; no measurements or output writes.")
    args = parser.parse_args()
    OUTPUT_ROOT = args.output_dir.expanduser().resolve()
    if args.dry_run:
        run_benchmarks(dry_run=True)
        return
    if not args.aggregate_only:
        run_benchmarks()
    aggregate()


if __name__ == "__main__":
    main()
