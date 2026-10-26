#!/usr/bin/env python3
"""Export frozen context-length, B64 projection and DQ-CIS plot data.

Source results are read-only. No models, retrieval or bootstrap runs are invoked.
The original figure's saved intervals are retained verbatim. Outputs are written
to RESULTS_DIR/10_efficiency_analysis/figures/data unless overridden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results"
OVERVIEW = RESULTS / "10_efficiency_analysis/figures"
OUT = OVERVIEW / "data"
MANIFEST = None
used_sources: dict[str, dict[str, str]] = {}


def source(key: str) -> Path:
    if MANIFEST is None:
        raise RuntimeError("Load the figure manifest through main() before exporting.")
    item = MANIFEST["sources"][key]
    recorded = Path(item["path"])
    # Manifests may have been moved with the result tree. Never silently read
    # an old author's absolute path even if that checkout still exists.
    parts = recorded.parts
    anchor = next((i for i in range(len(parts) - 1)
                   if parts[i:i + 2] == ("experiments", "results")), None)
    if anchor is not None:
        relative = Path(*parts[anchor + 2:])
        figure_prefix = Path("10_efficiency_analysis/figures")
        path = (OVERVIEW / relative.relative_to(figure_prefix)
                if relative.is_relative_to(figure_prefix) else RESULTS / relative)
    elif recorded.is_absolute() and recorded.is_relative_to(OVERVIEW):
        path = recorded
    elif recorded.is_absolute() and recorded.is_relative_to(RESULTS):
        path = recorded
    else:
        raise ValueError(f"Source {key} is outside the explicitly selected result tree: {recorded}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == item["sha256"], f"Source changed: {key}"
    used_sources[key] = {"path": str(Path("experiments/results/10_efficiency_analysis/figures") / path.relative_to(OVERVIEW))
                        if path.is_relative_to(OVERVIEW) else str(Path("experiments/results") / path.relative_to(RESULTS)),
                        "sha256": digest}
    return path


def read(key: str) -> pd.DataFrame:
    return pd.read_csv(source(key))


def write(name: str, rows: pd.DataFrame) -> None:
    rows.to_csv(OUT / f"results_context_{name}.csv", index=False,
                float_format="%.17g")


def one(rows: pd.DataFrame) -> pd.Series:
    assert len(rows) == 1, f"Expected one record, found {len(rows)}"
    return rows.iloc[0]


def intervals(row: pd.Series, value: float) -> tuple[float, float]:
    low, high = float(row.ci95_low), float(row.ci95_high)
    assert low <= value <= high
    return value - low, high - value


def lengths() -> None:
    topi = read("topiocqa_context_inputs")
    qrecc = pd.read_json(source("qrecc_context_inputs"), lines=True,
                         compression="gzip")
    qrecc = qrecc.loc[qrecc.family.eq("rocc") & qrecc.system.eq("pretrained")]
    summaries, saturation = [], []
    for dataset, frame, n, arm, names in (
        ("topiocqa", topi, 2514, "arm", ("I", "I", "pretrained_R")),
        ("qrecc", qrecc, 8209, "view", ("I", "I", "R")),
    ):
        hist = {"length": np.arange(1, 513)}
        reference = None
        for name, source_arm, budget in zip(("I512", "I64", "R512"), names,
                                            (512, 64, 512)):
            rows = frame.loc[frame[arm].eq(source_arm) & frame.budget.eq(budget)]
            rows = rows.sort_values("sample_id", kind="mergesort")
            assert len(rows) == n and not rows.sample_id.duplicated().any()
            ids = rows.sample_id.to_numpy()
            if reference is None:
                reference = ids
            else:
                assert np.array_equal(reference, ids), "Unpaired lengths"
            values = rows.input_length.to_numpy(dtype=int)
            assert np.all((values >= 1) & (values <= budget))
            counts = np.bincount(values, minlength=513)[1:513]
            assert counts.sum() == n
            hist[name] = counts
            summaries.append(dict(dataset=dataset, series=name, n=n,
                                  mean=values.mean(), maximum=values.max(),
                                  over64=int((values > 64).sum()),
                                  over64_percent=100 * (values > 64).mean()))
        # One-token histogram edges, matching the source's bins 0.5, 1.5, ... .
        centers = pd.DataFrame(hist)
        edges = pd.DataFrame({"edge": np.arange(0.5, 513.5)})
        for series in ("I512", "I64", "R512"):
            edges[series] = np.r_[centers[series], 0]
        write(f"hist_{dataset}", edges)
        # Display-only aggregation reduces one-token noise without smoothing.
        # Bins contain lengths 1-8, 9-16, ..., 505-512. Counts still sum to n.
        grouped = pd.DataFrame({"edge": np.arange(0.5, 513.5, 8)})
        for series in ("I512", "I64", "R512"):
            counts = centers[series].to_numpy().reshape(64, 8).sum(axis=1)
            assert counts.sum() == n
            grouped[series] = np.r_[counts, 0]
        write(f"hist8_{dataset}", grouped)
        for backend in ("bm25", "ance"):
            frame = read(f"{dataset}_{backend}_saturation")
            if dataset == "topiocqa":
                row = one(frame.loc[frame.system.eq("pretrained") &
                                    frame.arm.eq("R") &
                                    frame.comparison.eq("B128-B64")])
                value = float(row.mrr_delta)
            else:
                row = one(frame.loc[frame.arm.eq("R") &
                                    frame.comparison.eq("B128 - B64") &
                                    frame.metric.eq("MRR")])
                value = float(row.delta)
            assert int(row.n) == n and int(row.replicates) == 10000
            low, high = intervals(row, value)
            saturation.append(dict(dataset=dataset, backend=backend, n=n,
                                   delta=value, ci95_low=row.ci95_low,
                                   ci95_high=row.ci95_high, minus=low, plus=high))
    write("length_summary", pd.DataFrame(summaries))
    write("saturation", pd.DataFrame(saturation))


def granularity() -> None:
    bars = read("granularity_projection_bar_bootstrap_cis")
    quality = read("quality_bar_bootstrap_cis")
    paired = read("granularity_projection_vs_i512_comparisons")
    summary = read("granularity_projection_summary")
    baseline = one(read("granularity_projection_baseline_length_bootstrap_ci"))
    length_summary = read("granularity_projection_input_lengths")
    length_ci = read("granularity_projection_length_bootstrap_cis")
    rows = []
    for x, label, budget, gran in ((1, "I512", 512, None), (2, "I64", 64, None),
                                   (3, "V1", 64, "turn"), (4, "V2", 64, "qa"),
                                   (5, "V3", 64, "token")):
        if gran is None:
            row = one(quality.loc[quality.dataset.eq("TopiOCQA Dev") &
                                  quality.backend.eq("BM25") &
                                  quality.system.eq("pretrained") &
                                  quality.budget.eq(budget) & quality.route.eq("I")])
            value = float(row.MRR)
            low, high = float(row.MRR_ci95_low), float(row.MRR_ci95_high)
        else:
            row = one(bars.loc[bars.budget.eq(64) & bars.arm.eq("R") &
                              bars.granularity.eq(gran)])
            value, low, high = float(row.MRR), float(row.ci95_low), float(row.ci95_high)
            original = one(summary.loc[summary.budget.eq(64) & summary.arm.eq("R") &
                                       summary.granularity.eq(gran)])
            assert np.isclose(value, original.MRR, atol=1e-12)
        assert int(row.n) == 2514 and low <= value <= high
        rows.append(dict(x=x, label=label, n=2514, mean=value,
                         ci95_low=low, ci95_high=high,
                         minus=value-low, plus=high-value))
    write("granularity_mrr", pd.DataFrame(rows))
    write("granularity_mrr_paired", paired)
    rows = []
    for x, gran in enumerate(("turn", "qa", "token"), start=1):
        original = one(length_summary.loc[length_summary.budget.eq(64) &
                                          length_summary.granularity.eq(gran)])
        row = one(length_ci.loc[length_ci.granularity.eq(gran)])
        value = float(original.mean_R_t5_input_tokens) - baseline.mean_t5_input_tokens
        assert int(row.n) == 2514 and np.isclose(value, row.mean_delta_t5_tokens)
        low, high = intervals(row, value)
        rows.append(dict(x=x, granularity=gran, n=2514, delta=value,
                         selected_mean=original.mean_R_t5_input_tokens,
                         baseline_mean=baseline.mean_t5_input_tokens,
                         percentage=100*value/baseline.mean_t5_input_tokens,
                         ci95_low=row.ci95_low, ci95_high=row.ci95_high,
                         minus=low, plus=high))
    write("granularity_lengths", pd.DataFrame(rows))
    write("granularity_baseline", baseline.to_frame().T)


def dqcis() -> None:
    bars = read("dqcis_fusion_bar_bootstrap_cis")
    summary = read("dqcis_fusion_summary")
    metrics = read("dqcis_fusion_metrics_by_query")
    paired = read("dqcis_fusion_paired_comparisons")
    original_ids = None
    rows = []
    for x, system in enumerate(("DQ", "DQ+I", "DQ+R"), start=1):
        row = one(bars.loc[bars.system.eq(system)])
        raw = metrics.loc[metrics.system.eq(system)].sort_values("sample_id")
        assert len(raw) == 2514 and not raw.sample_id.duplicated().any()
        if original_ids is None:
            original_ids = raw.sample_id.to_numpy()
        else:
            assert np.array_equal(original_ids, raw.sample_id.to_numpy())
        value = float(row.MRR)
        assert np.isclose(value, raw.MRR.mean(), atol=1e-12)
        assert np.isclose(value, one(summary.loc[summary.system.eq(system)]).MRR)
        low, high = intervals(row, value)
        rows.append(dict(x=x, system=system, n=2514, mean=value,
                         ci95_low=row.ci95_low, ci95_high=row.ci95_high,
                         minus=low, plus=high))
    write("dqcis_mrr", pd.DataFrame(rows))
    rows = []
    for y, comparison in enumerate(("DQ+I_minus_DQ", "DQ+R_minus_DQ",
                                    "DQ+R_minus_DQ+I"), start=1):
        row = one(paired.loc[paired.comparison.eq(comparison)])
        assert int(row.n_queries) == 2514 and int(row.replicates) == 10000
        value = float(row.delta_mrr)
        low, high = intervals(row, value)
        rows.append(dict(y=y, comparison=comparison, delta=value,
                         ci95_low=row.ci95_low, ci95_high=row.ci95_high,
                         minus=low, plus=high))
    write("dqcis_paired", pd.DataFrame(rows))


def main() -> None:
    global MANIFEST
    MANIFEST = json.loads((OVERVIEW / "manifest.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    lengths()
    granularity()
    dqcis()
    write("sources", pd.DataFrame([dict(key=k, **v) for k, v in used_sources.items()]))
    print(pd.read_csv(OUT / "results_context_length_summary.csv").to_string(index=False))
    print(pd.read_csv(OUT / "results_context_saturation.csv").to_string(index=False))
    print(f"Exported frozen data to {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--figures-dir", type=Path,
                        help="Default: RESULTS_DIR/10_efficiency_analysis/figures; requires plot_topiocqa_efficiency.py manifest.")
    parser.add_argument("--output-dir", type=Path, help="Default: FIGURES_DIR/data.")
    args = parser.parse_args()
    RESULTS = args.results_dir.expanduser().resolve()
    OVERVIEW = (args.figures_dir or RESULTS / "10_efficiency_analysis/figures").expanduser().resolve()
    OUT = (args.output_dir or OVERVIEW / "data").expanduser().resolve()
    main()
