#!/usr/bin/env python3
"""Build final cost/quality figures and their aggregate CSVs from saved runs.

Requires completed NB03, NB04, NB07a/b (including query-only and granularity
extensions), NB08, Teacher evaluation, five Full-Dev latency runs and Nsight
summaries. No retrieval or inference is performed. Local IterCQR and pinned
MiniLM tokenizer files are needed for the original length/label projections.
Use --check-inputs for path checks without bootstrapping or rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections import Counter
from pathlib import Path
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.textpath import TextPath
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS_ROOT = ROOT / "experiments/results"
RESULT_DIR = RESULTS_ROOT / "10_efficiency_analysis"
LATENCY_DIR = RESULT_DIR / "latency"
LATENCY_RUNS_SOURCE = (
    LATENCY_DIR / "reproducibility_route_components_runs.csv"
)
NSIGHT_DIR = RESULT_DIR / "nsight"
BM25_DIR = ROOT / "experiments/results/07a_topiocqa_final/backends/bm25"
ANCE_DIR = ROOT / "experiments/results/07a_topiocqa_final/backends/ance"
QRECC_DIR = ROOT / "experiments/results/07b_qrecc_ood/backends"
CURRENT_QUERY_ONLY_DIRS = {
    "TopiOCQA Dev": (
        ROOT
        / "experiments/results/07a_topiocqa_final/current_query_only"
    ),
    "QReCC Test (dataset OOD)": (
        ROOT
        / "experiments/results/07b_qrecc_ood/current_query_only"
    ),
}
FIGURE_DIR = RESULT_DIR / "figures"
CURRENT_QUERY_ONLY_CI_SOURCE = (
    FIGURE_DIR / "current_query_only_mrr_bootstrap_ci.csv"
)
TOPIOCQA_CONTEXT_SOURCE = (
    ROOT
    / "experiments/results/07a_topiocqa_final/query_bundle"
    / "serialization/main_inputs.csv"
)
QRECC_CONTEXT_SOURCE = (
    ROOT
    / "experiments/results/07b_qrecc_ood/query_bundle"
    / "query_bundle.jsonl.gz"
)
GRANULARITY_PROJECTION_DIR = BM25_DIR / "granularity_projection"
GRANULARITY_COMPARISON_SOURCE = (
    BM25_DIR / "granularity_mrr_comparisons.csv"
)
GRANULARITY_LENGTH_SOURCE = (
    GRANULARITY_PROJECTION_DIR / "input_length_summary.csv"
)
GRANULARITY_METRICS_SOURCE = (
    GRANULARITY_PROJECTION_DIR / "metrics_by_query.csv"
)
GRANULARITY_SUMMARY_SOURCE = GRANULARITY_PROJECTION_DIR / "summary.csv"
GRANULARITY_BAR_CI_SOURCE = (
    FIGURE_DIR / "granularity_projection_bar_bootstrap_ci.csv"
)
GRANULARITY_BASELINE_COMPARISON_SOURCE = (
    FIGURE_DIR / "granularity_projection_vs_i512_comparisons.csv"
)
GRANULARITY_LENGTH_CI_SOURCE = (
    FIGURE_DIR / "granularity_projection_length_bootstrap_ci.csv"
)
GRANULARITY_BASELINE_LENGTH_CI_SOURCE = (
    FIGURE_DIR / "granularity_projection_baseline_length_bootstrap_ci.csv"
)
DQCIS_DIR = ROOT / "experiments/results/08_dqcis_external_ablation"
DQCIS_SUMMARY_SOURCE = DQCIS_DIR / "summary.csv"
DQCIS_METRICS_SOURCE = DQCIS_DIR / "metrics_by_query.csv.gz"
DQCIS_COMPARISONS_SOURCE = DQCIS_DIR / "paired_comparisons.csv"
DQCIS_MANIFEST_SOURCE = DQCIS_DIR / "manifest.json"
DQCIS_BAR_CI_SOURCE = FIGURE_DIR / "dqcis_fusion_bar_bootstrap_ci.csv"
TEACHER_EVAL_DIR = RESULT_DIR / "teacher_eval"
TEACHER_STUDENT_MRR_SOURCE = TEACHER_EVAL_DIR / "retrieval_summary.csv"
TEACHER_STUDENT_COMPARISON_SOURCE = (
    TEACHER_EVAL_DIR / "paired_comparisons.csv"
)
TEACHER_STUDENT_F1_SOURCE = TEACHER_EVAL_DIR / "agreement_summary.csv"
TEACHER_STUDENT_AGREEMENT_COUNTS_SOURCE = (
    TEACHER_EVAL_DIR / "agreement_counts_by_query.csv"
)
TEACHER_STUDENT_POPULATION_SOURCE = TEACHER_EVAL_DIR / "cohort.csv"
TEACHER_STUDENT_MANIFEST_SOURCE = TEACHER_EVAL_DIR / "manifest.json"
TEACHER_STUDENT_MRR_EXPORT = FIGURE_DIR / "teacher_student_mrr.csv"
TEACHER_STUDENT_F1_EXPORT = FIGURE_DIR / "teacher_student_f1.csv"
TEACHER_STUDENT_F1_COMPARISON_EXPORT = (
    FIGURE_DIR / "teacher_student_f1_paired_comparisons.csv"
)
TEACHER_LABELS_SOURCE = (
    ROOT
    / "experiments/results/04_teacher"
    / "rocc_history_labels_topiocqa_dev.jsonl"
)
TEACHER_LABEL_DISTRIBUTION_EXPORT = (
    FIGURE_DIR / "teacher_label_distribution.csv"
)
DATASET_STRUCTURE_EXPORT = FIGURE_DIR / "dataset_structure.csv"
STRUCTURE_QUALITY_EXPORT = (
    FIGURE_DIR / "retrieval_quality_by_dataset_structure.csv"
)
ORACLE_DIR = (
    ROOT / "experiments/results/03_oracle_headroom_analysis"
)
ORACLE_MANIFEST_SOURCE = ORACLE_DIR / "manifest.json"
ORACLE_SOURCES = {
    "population": ORACLE_DIR / "population.csv",
    "budget_degradation": ORACLE_DIR / "budget_degradation.csv",
    "space_oracle": ORACLE_DIR / "space_oracle_summary.csv",
    "control_relation": ORACLE_DIR / "control_relation_summary.csv",
    "dedup": ORACLE_DIR / "dedup_summary.csv",
    "generator": ORACLE_DIR / "generator_summary.csv",
    "monte_carlo": ORACLE_DIR / "mc_summary_all_budgets.csv",
    "entity_anchor": ORACLE_DIR / "entity_anchor_summary.csv",
}
LATENCY_COMPARISON_EXPORT = (
    FIGURE_DIR / "latency_vs_i512_paired_t_intervals.csv"
)
TOPIOCQA_DATA_DIR = ROOT / "experiments/data/topiocqa"
QRECC_DATA_DIR = ROOT / "experiments/data/qrecc"
TOPIOCQA_DEV_SOURCE = (
    TOPIOCQA_DATA_DIR / "downloads/data/retriever/original/dev.json"
)
QRECC_TEST_METADATA_SOURCE = (
    QRECC_DATA_DIR / "collection/qrecc-test.json"
)
ITERCQR_MODEL_DIR = ROOT / "experiments/model/IterCQR/IterCQR Model"
PRETRAINED_VITERBI_SOURCE = (
    ROOT
    / "experiments/results/07a_topiocqa_final"
    / "predictions/pretrained/viterbi.jsonl"
)
SATURATION_SOURCES = {
    "topiocqa": {
        "BM25": BM25_DIR / "budget_mrr_comparisons.csv",
        "ANCE": ANCE_DIR / "budget_mrr_comparisons.csv",
    },
    "qrecc": {
        "BM25": QRECC_DIR / "bm25/saturation_comparisons.csv",
        "ANCE": QRECC_DIR / "ance/saturation_comparisons.csv",
    },
}

try:
    font_manager.findfont("Arial", fallback_to_default=False)
    PLOT_FONT = "Arial"
except ValueError:
    PLOT_FONT = "DejaVu Sans"
PLOT_INK = "#475467"
plt.rcParams.update(
    {
        "font.family": PLOT_FONT,
        "font.sans-serif": [PLOT_FONT],
        "pdf.fonttype": 42,
        "text.color": PLOT_INK,
        "axes.edgecolor": PLOT_INK,
        "axes.labelcolor": PLOT_INK,
        "axes.titlecolor": PLOT_INK,
        "axes.linewidth": 0.6,
        "xtick.color": PLOT_INK,
        "ytick.color": PLOT_INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
    }
)

BUDGETS = (64, 128, 256, 512)
ROUTES = ("I", "R", "D", "IRD")
COLORS = {
    "I": "#9099a3",
    "R": "#adc6da",
    "D": "#7098b8",
    "IRD": "#315f88",
}
RECALL_COLORS = {
    "I": "#9099a3",
    "R": "#adc6da",
    "D": "#7098b8",
    "IRD": "#315f88",
}
GRANULARITY_STYLES = (
    ("turn", "v1 Turn", "#c5d9e8", -0.22),
    ("qa", "v2 Q/A", "#8fb3cd", 0.0),
    ("token", "v3 Token/span", "#315f88", 0.22),
)
GRANULARITY_LENGTH_STYLES = (
    ("turn", "v1 Turn", "#d6c8db", -0.22),
    ("qa", "v2 Q/A", "#a98bb4", 0.0),
    ("token", "v3 Token/span", "#6f4778", 0.22),
)
ENCODER_COMPONENTS = (
    ("embedding", "Embedding", "#eee8f0"),
    ("self_attention", "Self-attention", "#a98bb4"),
    ("ffn", "FFN", "#6f4778"),
    (
        "layernorm_residual_other",
        "Other",
        "#cdbed2",
    ),
)
DISPLAY_ENCODER_COMPONENTS = (
    ("self_attention", "Self-attention", "#a98bb4"),
    ("ffn", "FFN", "#6f4778"),
    (
        "display_other",
        "Other",
        "#cdbed2",
    ),
)
COMPONENTS = (
    ("mean_rocc_ms", "ROCC selector", "#a98bb4"),
    ("mean_t5_encoder_ms", "T5 encoder", "#6f4778"),
    ("mean_t5_decoder_ms", "T5 decoder", "#d6c8db"),
)
MIDDLE_BUDGET_ALPHA = 0.35
BAR_GAP_RATIO = 0.20
BAR_GROUP_GAP_RATIO = 1.25
COMPARISON_FAMILY_GAP_MULTIPLIER = 1.0
COMPARISON_BRACKET_AXES_Y = 0.82
METRIC_COMPARISON_BRACKET_AXES_Y = 0.80
METRIC_COMPARISON_BRACKET_AXES_Y_TOPIOCQA_BM25 = 0.62
METRIC_COMPARISON_BRACKET_AXES_Y_TOPIOCQA_ANCE = 0.68
METRIC_COMPARISON_SUBLABEL_POINTS = -7.0
METRIC_COMPARISON_VALUE_LABEL_POINTS = -58.0
METRIC_SIGNIFICANCE_MARKER_GAP_POINTS = 21.0
COMPARISON_BRACKET_LEG_POINTS = 5.0
COMPARISON_FAMILY_TITLE_POINTS = 4.0
COMPARISON_SUBLABEL_POINTS = -7.0
COMPARISON_VALUE_LABEL_POINTS = -50.0
QUALITY_METRICS = ("MRR", "nDCG@3", "R@10", "R@100", "R@1000")
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_CHUNK_SIZE = 250
FONT_PAGE_TITLE = 18.0
FONT_PANEL_TITLE = 13.5
FONT_AXIS = 13.5
FONT_TICK = 13.5
FONT_BODY = 11.5
STRUCTURE_FIGURE_FONT_SCALE = 1.45
BAR_ANNOTATION_MAX_AXES_Y = 0.80
RECALL_LAYOUT_SCALE = 1.55
LEGEND_HANDLE_LENGTH = 0.72
LEGEND_HANDLE_HEIGHT = 1.00
LEGEND_HANDLE_TEXT_PAD = 0.45
LEGEND_COLUMN_SPACING = 1.00
CONTEXT_LENGTH_STYLES = (
    ("IterCQR full-history (I-B512)", "#8d98a5"),
    ("IterCQR truncated (I-B64)", "#bcaec4"),
    ("ROCC (not truncated)", "#6f4778"),
)


def grouped_bar_layout(
    group_sizes: tuple[int, ...],
    width: float,
    *,
    gap_ratio: float = BAR_GAP_RATIO,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, tuple[tuple[float, float], ...]]:
    """Return one uniform, visibly separated geometry for every bar plot."""
    if not group_sizes or any(size < 1 for size in group_sizes):
        raise ValueError("Bar groups must contain at least one bar.")
    step = width * (1.0 + gap_ratio)
    group_gap = width * BAR_GROUP_GAP_RATIO
    groups: list[np.ndarray] = []
    spans: list[tuple[float, float]] = []
    cursor = 0.0
    for size in group_sizes:
        centers = cursor + width / 2.0 + np.arange(size) * step
        left = float(centers[0] - width / 2.0)
        right = float(centers[-1] + width / 2.0)
        groups.append(centers)
        spans.append((left, right))
        cursor = right + group_gap
    shift = (spans[0][0] + spans[-1][1]) / 2.0
    shifted_groups = tuple(group - shift for group in groups)
    shifted_spans = tuple(
        (left - shift, right - shift) for left, right in spans
    )
    group_centers = np.asarray(
        [(left + right) / 2.0 for left, right in shifted_spans]
    )
    return shifted_groups, group_centers, shifted_spans


def draw_comparison_family_header(
    axis,
    *,
    left: float,
    right: float,
    label: str,
    y: float = COMPARISON_BRACKET_AXES_Y,
) -> None:
    """Draw one comparison header with fixed physical spacing."""
    transform = axis.get_xaxis_transform()
    axis.plot(
        [left, right],
        [y, y],
        transform=transform,
        color="#475467",
        linewidth=0.9,
        clip_on=False,
    )
    for x in (left, right):
        axis.annotate(
            "",
            xy=(x, y),
            xycoords=transform,
            xytext=(0, -COMPARISON_BRACKET_LEG_POINTS),
            textcoords="offset points",
            arrowprops={
                "arrowstyle": "-",
                "color": "#475467",
                "linewidth": 0.9,
                "shrinkA": 0,
                "shrinkB": 0,
            },
            annotation_clip=False,
        )
    axis.annotate(
        label,
        xy=((left + right) / 2.0, y),
        xycoords=transform,
        xytext=(0, COMPARISON_FAMILY_TITLE_POINTS),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9.8,
        fontweight="bold",
        color="#475467",
        annotation_clip=False,
    )
DEPTH_STRATA = (
    ("d00", "0", 0, 0),
    ("d01", "1", 1, 1),
    ("d02_04", "2–4", 2, 4),
    ("d05_06", "5–6", 5, 6),
    ("d07_08", "7–8", 7, 8),
    ("d09_10", "9–10", 9, 10),
    ("d11_14", "11–14", 11, 14),
    ("d15_plus", "15+", 15, None),
)


def fade_middle_budgets(bars) -> None:
    for budget, bar in zip(BUDGETS, bars, strict=True):
        if budget in (128, 256):
            bar.set_alpha(MIDDLE_BUDGET_ALPHA)


def add_preference_hint(
    axis,
    *,
    higher_is_better: bool,
    fontsize: float = 8,
    axes_x: float = 0.985,
    axes_y: float = 0.985,
    offset_points: tuple[float, float] | None = None,
) -> None:
    text = "↑ Higher is better" if higher_is_better else "↓ Lower is better"
    if offset_points is None:
        axis.text(
            axes_x,
            axes_y,
            text,
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=fontsize,
            color="#475467",
            zorder=20,
        )
    else:
        axis.annotate(
            text,
            xy=(axes_x, axes_y),
            xycoords=axis.transAxes,
            xytext=offset_points,
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=fontsize,
            color="#475467",
            annotation_clip=False,
            zorder=20,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def bootstrap_bar_intervals(
    per_query: pd.DataFrame,
    *,
    source_key: str,
    expected_n: int,
) -> pd.DataFrame:
    """Return pointwise percentile CIs using common query resamples."""
    cells = {}
    sample_ids = None
    for (budget, arm), group in per_query.groupby(
        ["budget", "arm"], observed=True, sort=True
    ):
        group = group.sort_values("sample_id", kind="mergesort")
        ids = group["sample_id"].astype(str).to_numpy()
        if len(ids) != expected_n or pd.Series(ids).duplicated().any():
            raise RuntimeError("Invalid population for bar bootstrap.")
        if sample_ids is None:
            sample_ids = ids
        elif not np.array_equal(sample_ids, ids):
            raise RuntimeError("Bar-bootstrap populations are not identical.")
        cells[(int(budget), str(arm))] = group[
            list(QUALITY_METRICS)
        ].to_numpy(dtype=float)

    expected_cells = {
        (budget, arm)
        for budget in BUDGETS
        for arm in ("I", "R", "D", "I+R+D")
    }
    if set(cells) != expected_cells:
        raise RuntimeError("Bar bootstrap is not the complete 4x4 design.")

    seed = int.from_bytes(
        hashlib.sha256(
            f"quality_bar_ci_v1\0{source_key}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    draws = {
        key: np.empty((BOOTSTRAP_REPLICATES, len(QUALITY_METRICS)))
        for key in cells
    }
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            0,
            expected_n,
            size=(stop - start, expected_n),
        )
        for key, values in cells.items():
            draws[key][start:stop] = values[indices].mean(axis=1)

    rows = []
    for (budget, arm), values in cells.items():
        low, high = np.quantile(draws[(budget, arm)], (0.025, 0.975), axis=0)
        row = {
            "budget": budget,
            "route": "IRD" if arm == "I+R+D" else arm,
            "bootstrap_seed": seed,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        }
        means = values.mean(axis=0)
        for index, metric in enumerate(QUALITY_METRICS):
            row[f"{metric}_bootstrap_mean"] = means[index]
            row[f"{metric}_ci95_low"] = low[index]
            row[f"{metric}_ci95_high"] = high[index]
        rows.append(row)
    return pd.DataFrame(rows)


def load_dqcis_fusion_ablation() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load NB08 and derive pointwise MRR intervals for its three systems."""
    systems = ("DQ", "DQ+I", "DQ+R")
    summary = pd.read_csv(DQCIS_SUMMARY_SOURCE)
    metrics = pd.read_csv(DQCIS_METRICS_SOURCE)
    comparisons = pd.read_csv(DQCIS_COMPARISONS_SOURCE)
    manifest = json.loads(DQCIS_MANIFEST_SOURCE.read_text(encoding="utf-8"))
    identity = manifest.get("input_identity", {})
    if not (
        manifest.get("complete") is True
        and identity.get("dataset") == "TopiOCQA dev"
        and identity.get("budget") == 64
        and identity.get("top_k") == 100
        and identity.get("rrf_k") == 10
        and tuple(identity.get("methods", ())) == systems
    ):
        raise RuntimeError("NB08 DQ-CIS fusion protocol has drifted.")
    if set(summary["system"].astype(str)) != set(systems):
        raise RuntimeError("Unexpected NB08 DQ-CIS summary systems.")

    values_by_system: dict[str, np.ndarray] = {}
    sample_ids: np.ndarray | None = None
    for system in systems:
        group = metrics.loc[metrics["system"].eq(system)].sort_values(
            "sample_id", kind="mergesort"
        )
        ids = group["sample_id"].astype(str).to_numpy()
        values = group["MRR"].to_numpy(dtype=float)
        if len(ids) != 2_514 or pd.Series(ids).duplicated().any():
            raise RuntimeError(f"Invalid NB08 population for {system}.")
        if sample_ids is None:
            sample_ids = ids
        elif not np.array_equal(sample_ids, ids):
            raise RuntimeError("NB08 DQ-CIS populations are not paired.")
        expected_mean = float(
            summary.loc[summary["system"].eq(system), "MRR"].iloc[0]
        )
        if not np.isclose(values.mean(), expected_mean, atol=1e-12):
            raise RuntimeError(f"NB08 MRR drift for {system}.")
        values_by_system[system] = values

    source_key = sha256(DQCIS_METRICS_SOURCE)
    seed = int.from_bytes(
        hashlib.sha256(
            f"dqcis_fusion_bar_ci_v1\0{source_key}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    draws = {
        system: np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for system in systems
    }
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            0,
            2_514,
            size=(stop - start, 2_514),
        )
        for system, values in values_by_system.items():
            draws[system][start:stop] = values[indices].mean(axis=1)

    interval_rows = []
    for system in systems:
        low, high = np.quantile(draws[system], (0.025, 0.975))
        interval_rows.append(
            {
                "system": system,
                "n": 2_514,
                "MRR": float(values_by_system[system].mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_seed": seed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "source_key": source_key,
            }
        )
    intervals = pd.DataFrame(interval_rows)

    pivot = pd.DataFrame(values_by_system)
    required_comparisons = {
        "DQ+R_minus_DQ+I": ("DQ+R", "DQ+I"),
        "DQ+R_minus_DQ": ("DQ+R", "DQ"),
        "DQ+I_minus_DQ": ("DQ+I", "DQ"),
    }
    if set(comparisons["comparison"].astype(str)) != set(
        required_comparisons
    ):
        raise RuntimeError("Unexpected NB08 DQ-CIS comparisons.")
    for name, (left, right) in required_comparisons.items():
        row = comparisons.loc[comparisons["comparison"].eq(name)].iloc[0]
        delta = float((pivot[left] - pivot[right]).mean())
        if not (
            int(row["n_queries"]) == 2_514
            and np.isclose(float(row["delta_mrr"]), delta, atol=1e-12)
            and float(row["ci95_low"]) <= delta <= float(row["ci95_high"])
        ):
            raise RuntimeError(f"NB08 paired comparison drift for {name}.")
    return intervals, comparisons


def load_quality(backend_dirs, expected_n: int):
    quality_frames = []
    comparison_rows = []
    for backend, backend_dir in backend_dirs:
        backend_quality = pd.read_csv(backend_dir / "route_summary.csv")
        backend_quality = backend_quality.loc[
            backend_quality["system"].eq("pretrained")
            & backend_quality["budget"].isin(BUDGETS)
            & backend_quality["arm"].isin(("I", "R", "D", "I+R+D"))
        ].copy()
        backend_quality["route"] = backend_quality["arm"].replace(
            {"I+R+D": "IRD"}
        )
        backend_quality["backend"] = backend

        per_query_path = backend_dir / "route_metrics_by_query.csv"
        per_query = pd.read_csv(
            per_query_path,
            usecols=[
                "system", "budget", "sample_id", "arm",
                "MRR", "nDCG@3", "R@10", "R@100", "R@1000",
            ],
        )
        per_query = per_query.loc[
            per_query["system"].eq("pretrained")
            & per_query["budget"].isin(BUDGETS)
            & per_query["arm"].isin(("I", "R", "D", "I+R+D"))
        ].copy()
        bar_intervals = bootstrap_bar_intervals(
            per_query,
            source_key=str(Path("experiments/results") / per_query_path.relative_to(RESULTS_ROOT)),
            expected_n=expected_n,
        )
        backend_quality = backend_quality.merge(
            bar_intervals,
            on=["budget", "route"],
            how="left",
            validate="one_to_one",
        )
        for metric in QUALITY_METRICS:
            if not np.allclose(
                backend_quality[metric],
                backend_quality[f"{metric}_bootstrap_mean"],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"Bootstrap means do not reproduce {metric}."
                )
            if not (
                backend_quality[f"{metric}_ci95_low"].le(
                    backend_quality[metric]
                ).all()
                and backend_quality[f"{metric}_ci95_high"].ge(
                    backend_quality[metric]
                ).all()
            ):
                raise RuntimeError(f"Invalid bootstrap CI for {metric}.")
        quality_frames.append(backend_quality)

        def add_comparison(
            name: str,
            left_arm: str,
            right_budget: int,
            metric: str,
            seed: int,
        ) -> None:
            left = per_query.loc[
                per_query["system"].eq("pretrained")
                & per_query["budget"].eq(64)
                & per_query["arm"].eq(left_arm),
                ["sample_id", metric],
            ].rename(columns={metric: "left"})
            right = per_query.loc[
                per_query["system"].eq("pretrained")
                & per_query["budget"].eq(right_budget)
                & per_query["arm"].eq("I"),
                ["sample_id", metric],
            ].rename(columns={metric: "right"})
            delta = left.merge(
                right, on="sample_id", validate="one_to_one"
            ).eval("left - right").to_numpy()
            rng = np.random.default_rng(seed)
            draws = np.concatenate(
                [
                    delta[
                        rng.integers(0, len(delta), (size, len(delta)))
                    ].mean(axis=1)
                    for size in (500,) * 20
                ]
            )
            low, high = np.quantile(draws, (0.025, 0.975))
            comparison_rows.append(
                {
                    "backend": backend,
                    "comparison": name,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )

        for name, left_arm, right_budget, seed in (
            ("IRD64_minus_I64", "I+R+D", 64, 1000),
            ("IRD64_minus_I512", "I+R+D", 512, 1060),
            ("R64_minus_I512", "R", 512, 1061),
            ("I64_minus_I512", "I", 512, 1062),
            ("D64_minus_I512", "D", 512, 1063),
        ):
            add_comparison(name, left_arm, right_budget, "MRR", seed)
        for name, left_arm, seed in (
            ("I64_minus_I512", "I", 2000),
            ("R64_minus_I512", "R", 2100),
            ("D64_minus_I512", "D", 2200),
            ("IRD64_minus_I512", "I+R+D", 2300),
        ):
            add_comparison(
                f"{name}_nDCG3",
                left_arm,
                512,
                "nDCG@3",
                seed + 500,
            )
            for metric_index, metric in enumerate(
                ("R@10", "R@100", "R@1000")
            ):
                add_comparison(
                    f"{name}_{metric.replace('@', '')}",
                    left_arm,
                    512,
                    metric,
                    seed + metric_index,
                )
    quality = pd.concat(quality_frames, ignore_index=True)
    comparisons = pd.DataFrame(comparison_rows)
    expected = {
        (backend, route, budget)
        for backend, _ in backend_dirs
        for route in ROUTES
        for budget in BUDGETS
    }
    observed = set(
        quality[["backend", "route", "budget"]].itertuples(
            index=False, name=None
        )
    )
    if observed != expected or quality.duplicated(
        ["backend", "route", "budget"]
    ).any():
        raise RuntimeError("Quality summaries are not complete 4x4 designs.")
    if not quality["n"].eq(expected_n).all():
        raise RuntimeError(f"Quality summaries are not based on n={expected_n}.")
    return quality, comparisons


def load_current_query_only_mrr() -> pd.DataFrame:
    """Validate full-population current-query-only runs and bootstrap MRR."""
    rows = []
    dataset_specs = (
        ("TopiOCQA Dev", "topiocqa", 2_514),
        ("QReCC Test (dataset OOD)", "qrecc", 8_209),
    )
    for dataset, dataset_key, expected_n in dataset_specs:
        root = CURRENT_QUERY_ONLY_DIRS[dataset]
        bundle = json.loads(
            (root / "query_bundle/manifest.json").read_text(encoding="utf-8")
        )
        if not (
            bundle.get("complete") is True
            and bundle.get("dataset") == dataset_key
            and bundle.get("protocol") == "itercqr_current_query_only_v1"
            and bundle.get("history") == []
            and int(bundle.get("rows", -1)) == expected_n
            and int(bundle.get("truncated_inputs", -1)) == 0
        ):
            raise RuntimeError(
                f"Invalid current-query-only bundle for {dataset}."
            )

        reference_ids = None
        for backend, backend_dirname in (("BM25", "bm25"), ("ANCE", "ance")):
            backend_dir = root / backend_dirname
            manifest = json.loads(
                (backend_dir / "final_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            if not (
                manifest.get("complete") is True
                and manifest.get("dataset") == dataset_key
                and manifest.get("backend") == backend_dirname
                and manifest.get("protocol")
                == "itercqr_current_query_only_v1"
                and manifest.get("history") == []
                and manifest.get("selection_performed") is False
                and manifest.get("latency_measured") is False
                and int(manifest.get("budget", -1)) == 64
                and int(manifest.get("top_k", -1)) == 1_000
                and manifest.get("population_sha256")
                == bundle.get("population_sha256")
                and manifest.get("query_bundle_sha256")
                == bundle.get("query_bundle_sha256")
            ):
                raise RuntimeError(
                    f"Invalid current-query-only run for {dataset}/{backend}."
                )
            for filename, expected_hash in manifest.get("files", {}).items():
                if sha256(backend_dir / filename) != expected_hash:
                    raise RuntimeError(
                        f"Current-query-only hash drift: {dataset}/{backend}/"
                        f"{filename}."
                    )

            metrics_path = backend_dir / "metrics_by_query.csv"
            metrics = pd.read_csv(
                metrics_path,
                usecols=["sample_id", "query_id", "MRR"],
            ).sort_values("sample_id", kind="mergesort")
            ids = metrics[["sample_id", "query_id"]].astype(str).to_numpy()
            if (
                len(metrics) != expected_n
                or metrics["sample_id"].astype(str).duplicated().any()
            ):
                raise RuntimeError(
                    f"Invalid current-query-only population for "
                    f"{dataset}/{backend}."
                )
            if reference_ids is None:
                reference_ids = ids
            elif not np.array_equal(reference_ids, ids):
                raise RuntimeError(
                    f"Current-query-only backend populations differ for "
                    f"{dataset}."
                )

            summary = pd.read_csv(backend_dir / "summary.csv")
            if len(summary) != 1:
                raise RuntimeError(
                    f"Invalid current-query-only summary for "
                    f"{dataset}/{backend}."
                )
            summary_row = summary.iloc[0]
            values = metrics["MRR"].to_numpy(dtype=float)
            mean = float(values.mean())
            if not (
                summary_row["system"] == "itercqr"
                and summary_row["arm"] == "CurrentQueryOnly"
                and int(summary_row["n"]) == expected_n
                and np.isclose(
                    float(summary_row["MRR"]), mean, rtol=0.0, atol=1e-15
                )
            ):
                raise RuntimeError(
                    f"Current-query-only summary drift for {dataset}/{backend}."
                )

            source_hash = sha256(metrics_path)
            seed = int.from_bytes(
                hashlib.sha256(
                    f"current_query_only_mrr_ci_v1\0{source_hash}".encode(
                        "utf-8"
                    )
                ).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(seed)
            draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
            for start in range(
                0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK_SIZE
            ):
                stop = min(
                    start + BOOTSTRAP_CHUNK_SIZE,
                    BOOTSTRAP_REPLICATES,
                )
                indices = rng.integers(
                    0,
                    expected_n,
                    size=(stop - start, expected_n),
                )
                draws[start:stop] = values[indices].mean(axis=1)
            low, high = np.quantile(draws, (0.025, 0.975))
            rows.append(
                {
                    "dataset": dataset,
                    "backend": backend,
                    "n": expected_n,
                    "MRR": mean,
                    "MRR_ci95_low": float(low),
                    "MRR_ci95_high": float(high),
                    "bootstrap_seed": seed,
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "source_sha256": source_hash,
                    "query_bundle_sha256": bundle["query_bundle_sha256"],
                }
            )

    result = pd.DataFrame(rows)
    expected = {
        (dataset, backend)
        for dataset, _, _ in dataset_specs
        for backend in ("BM25", "ANCE")
    }
    if (
        set(result[["dataset", "backend"]].itertuples(index=False, name=None))
        != expected
        or result.duplicated(["dataset", "backend"]).any()
    ):
        raise RuntimeError("Incomplete current-query-only MRR design.")
    return result


def load_depth_strata_mrr() -> pd.DataFrame:
    """Build the four requested MRR cells by history depth."""
    topiocqa_depth = pd.read_csv(
        TOPIOCQA_CONTEXT_SOURCE,
        usecols=["sample_id", "history_depth"],
    ).drop_duplicates()
    qrecc_depth = pd.read_json(
        QRECC_CONTEXT_SOURCE,
        lines=True,
        compression="gzip",
    )[["sample_id", "history_depth"]].drop_duplicates()

    def depth_bin(depth: int) -> str:
        for key, _, lower, upper in DEPTH_STRATA:
            if depth >= lower and (upper is None or depth <= upper):
                return key
        raise RuntimeError(f"Uncovered TopiOCQA history depth: {depth}.")

    systems = (
        ("I512", "I", 512, "I-B512"),
        ("I64", "I", 64, "I-B64"),
        ("R", "R", 64, "R-B64"),
        ("IRD", "I+R+D", 64, "IRD-B64"),
    )
    rows = []
    dataset_specs = (
        (
            "TopiOCQA Dev",
            2_514,
            topiocqa_depth,
            (("BM25", BM25_DIR), ("ANCE", ANCE_DIR)),
        ),
        (
            "QReCC Test (dataset OOD)",
            8_209,
            qrecc_depth,
            (
                ("BM25", QRECC_DIR / "bm25"),
                ("ANCE", QRECC_DIR / "ance"),
            ),
        ),
    )
    expected_rows = 0
    for dataset, expected_n, depth_rows, backend_dirs in dataset_specs:
        if (
            len(depth_rows) != expected_n
            or depth_rows["sample_id"].duplicated().any()
        ):
            raise RuntimeError(f"Invalid {dataset} history-depth population.")
        depth_rows = depth_rows.copy()
        depth_rows["depth_bin"] = depth_rows["history_depth"].map(depth_bin)
        nonempty_depths = set(depth_rows["depth_bin"])
        expected_rows += 2 * len(nonempty_depths) * len(systems)

        for backend, backend_dir in backend_dirs:
            metrics = pd.read_csv(
                backend_dir / "route_metrics_by_query.csv",
                usecols=["system", "budget", "sample_id", "arm", "MRR"],
            )
            metrics = metrics.loc[
                metrics["system"].eq("pretrained")
                & (
                    (
                        metrics["arm"].eq("I")
                        & metrics["budget"].isin((64, 512))
                    )
                    | (
                        metrics["arm"].isin(("R", "I+R+D"))
                        & metrics["budget"].eq(64)
                    )
                )
            ].merge(depth_rows, on="sample_id", validate="many_to_one")
            if len(metrics) != 4 * expected_n:
                raise RuntimeError(
                    f"Incomplete depth-strata design for {dataset}/{backend}."
                )

            for depth_key, depth_label, _, _ in DEPTH_STRATA:
                stratum = metrics.loc[metrics["depth_bin"].eq(depth_key)]
                if stratum.empty:
                    continue
                cells = {}
                reference_ids = None
                for system_key, arm, budget, label in systems:
                    cell = stratum.loc[
                        stratum["arm"].eq(arm)
                        & stratum["budget"].eq(budget)
                    ].sort_values("sample_id", kind="mergesort")
                    ids = cell["sample_id"].astype(str).to_numpy()
                    if reference_ids is None:
                        reference_ids = ids
                    elif not np.array_equal(reference_ids, ids):
                        raise RuntimeError(
                            "Depth-strata populations differ by system."
                        )
                    cells[system_key] = (
                        label,
                        cell["MRR"].to_numpy(dtype=float),
                    )
                n = len(reference_ids)
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"depth_strata_mrr_v1\0{dataset}\0{backend}\0"
                        f"{depth_key}".encode()
                    ).digest()[:8],
                    "big",
                )
                rng = np.random.default_rng(seed)
                draws = {
                    key: np.empty(BOOTSTRAP_REPLICATES)
                    for key in cells
                }
                for start in range(
                    0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK_SIZE
                ):
                    stop = min(
                        start + BOOTSTRAP_CHUNK_SIZE,
                        BOOTSTRAP_REPLICATES,
                    )
                    indices = rng.integers(0, n, size=(stop - start, n))
                    for key, (_, values) in cells.items():
                        draws[key][start:stop] = values[indices].mean(axis=1)
                for system_key, (label, values) in cells.items():
                    low, high = np.quantile(
                        draws[system_key], (0.025, 0.975)
                    )
                    rows.append(
                        {
                            "dataset": dataset,
                            "backend": backend,
                            "depth_bin": depth_key,
                            "depth_label": depth_label,
                            "n": n,
                            "system_key": system_key,
                            "system": label,
                            "MRR": float(values.mean()),
                            "MRR_ci95_low": float(low),
                            "MRR_ci95_high": float(high),
                            "bootstrap_seed": seed,
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        }
                    )
    result = pd.DataFrame(rows)
    if len(result) != expected_rows or result.duplicated(
        ["dataset", "backend", "depth_bin", "system_key"]
    ).any():
        raise RuntimeError("Depth-strata MRR summary is incomplete.")
    return result


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed_key: str,
) -> tuple[float, float, float, int]:
    """Return a deterministic pointwise query-bootstrap mean interval."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid bootstrap values for {seed_key}.")
    seed = int.from_bytes(
        hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPLICATES)
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_REPLICATES)
        indices = rng.integers(
            0,
            len(values),
            size=(stop - start, len(values)),
        )
        draws[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(values.mean()), float(low), float(high), seed


def load_dataset_structure_analysis() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset structure and frozen pretrained R-B64 performance."""
    from experiments.rocc import (
        load_qrecc_frame, load_topiocqa_frame,
        resolve_qrecc_resources, resolve_topiocqa_resources,
    )

    topiocqa = load_topiocqa_frame(resolve_topiocqa_resources(TOPIOCQA_DATA_DIR), splits=("dev",)).sort_values(
        ["conv_id", "turn_id"], kind="mergesort"
    )
    topiocqa["sample_id"] = (
        "dev:"
        + topiocqa["conv_id"].astype(str)
        + ":"
        + topiocqa["turn_id"].astype(str)
    )
    # Count actual sequential changes.  topic_switch_depth is the current
    # topic ID minus one and therefore decreases when a conversation returns
    # to an earlier topic; it is not a cumulative switch count.
    topiocqa["structure_key"] = (
        topiocqa.groupby("conv_id", sort=False)["topic_switch"]
        .cumsum()
        .astype(int)
    )
    topiocqa["structure_label"] = topiocqa["structure_key"].astype(str)
    if (
        len(topiocqa) != 2_514
        or topiocqa["sample_id"].duplicated().any()
        or set(topiocqa["structure_key"]) != set(range(9))
    ):
        raise RuntimeError("Invalid TopiOCQA topic-switch population.")

    qrecc = load_qrecc_frame(resolve_qrecc_resources(QRECC_DATA_DIR)).loc[lambda frame: frame["split"].eq("test")]
    qrecc = qrecc.copy()
    qrecc["sample_id"] = (
        "test:"
        + qrecc["conv_id"].astype(str)
        + ":"
        + qrecc["turn_id"].astype(str)
    )
    qrecc["structure_key"] = qrecc["conversation_source"].str.lower()
    source_labels = {"nq": "NQ", "quac": "QuAC", "trec": "TREC"}
    qrecc["structure_label"] = qrecc["structure_key"].map(source_labels)
    if (
        len(qrecc) != 8_209
        or qrecc["sample_id"].duplicated().any()
        or set(qrecc["structure_key"]) != set(source_labels)
        or qrecc["structure_label"].isna().any()
    ):
        raise RuntimeError("Invalid QReCC source-dataset population.")

    dataset_specs = (
        (
            "TopiOCQA Dev",
            "encountered_topic_switches",
            topiocqa[["sample_id", "structure_key", "structure_label"]],
            2_514,
            (("BM25", BM25_DIR), ("ANCE", ANCE_DIR)),
        ),
        (
            "QReCC Test (dataset OOD)",
            "source_dataset",
            qrecc[["sample_id", "structure_key", "structure_label"]],
            8_209,
            (
                ("BM25", QRECC_DIR / "bm25"),
                ("ANCE", QRECC_DIR / "ance"),
            ),
        ),
    )
    distribution_rows = []
    performance_rows = []
    for dataset, structure, metadata, expected_n, backend_dirs in dataset_specs:
        counts = (
            metadata.groupby(
                ["structure_key", "structure_label"],
                observed=True,
                sort=False,
            )
            .size()
            .reset_index(name="n_queries")
        )
        if structure == "encountered_topic_switches":
            counts = counts.sort_values("structure_key", kind="mergesort")
        else:
            counts["order"] = counts["structure_key"].map(
                {"nq": 0, "quac": 1, "trec": 2}
            )
            counts = counts.sort_values("order", kind="mergesort")
        if int(counts["n_queries"].sum()) != expected_n:
            raise RuntimeError(f"Incomplete structure counts for {dataset}.")
        for order, row in enumerate(counts.itertuples(index=False)):
            distribution_rows.append(
                {
                    "dataset": dataset,
                    "structure": structure,
                    "structure_key": str(row.structure_key),
                    "structure_label": str(row.structure_label),
                    "order": order,
                    "n_queries": int(row.n_queries),
                    "share": float(row.n_queries / expected_n),
                }
            )

        systems = (
            ("I-B64", "I", 64),
            ("I-B512", "I", 512),
            ("R-B64", "R", 64),
        )
        for backend, backend_dir in backend_dirs:
            metrics = pd.read_csv(
                backend_dir / "route_metrics_by_query.csv",
                usecols=["system", "budget", "sample_id", "arm", "MRR"],
            )
            metrics = metrics.loc[
                metrics["system"].eq("pretrained")
                & (
                    (
                        metrics["arm"].eq("I")
                        & metrics["budget"].isin((64, 512))
                    )
                    | (
                        metrics["arm"].eq("R")
                        & metrics["budget"].eq(64)
                    )
                )
            ].merge(metadata, on="sample_id", validate="many_to_one")
            if (
                len(metrics) != len(systems) * expected_n
                or metrics.duplicated(["sample_id", "arm", "budget"]).any()
            ):
                raise RuntimeError(
                    f"Incomplete I/R population for {dataset}/{backend}."
                )
            for method, arm, budget in systems:
                method_metrics = metrics.loc[
                    metrics["arm"].eq(arm) & metrics["budget"].eq(budget)
                ]
                for order, row in enumerate(counts.itertuples(index=False)):
                    values = method_metrics.loc[
                        method_metrics["structure_key"].astype(str).eq(
                            str(row.structure_key)
                        ),
                        "MRR",
                    ].to_numpy(dtype=float)
                    mean, low, high, seed = bootstrap_mean_interval(
                        values,
                        seed_key=(
                            f"structure_quality_v2\0{dataset}\0{backend}\0"
                            f"{method}\0{row.structure_key}"
                        ),
                    )
                    performance_rows.append(
                        {
                            "dataset": dataset,
                            "structure": structure,
                            "structure_key": str(row.structure_key),
                            "structure_label": str(row.structure_label),
                            "order": order,
                            "backend": backend,
                            "system": "pretrained",
                            "method": method,
                            "route": arm,
                            "budget": budget,
                            "n_queries": len(values),
                            "MRR": mean,
                            "MRR_ci95_low": low,
                            "MRR_ci95_high": high,
                            "bootstrap_seed": seed,
                            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                        }
                    )

    distribution = pd.DataFrame(distribution_rows)
    performance = pd.DataFrame(performance_rows)
    expected_performance_rows = 3 * 2 * (9 + 3)
    if (
        len(performance) != expected_performance_rows
        or performance.duplicated(
            ["dataset", "backend", "method", "structure_key"]
        ).any()
    ):
        raise RuntimeError("Incomplete dataset-structure performance design.")

    # NB07b already exports the same QReCC source means.  Require exact
    # agreement so this page cannot silently drift from the notebook results.
    for backend in ("BM25", "ANCE"):
        stored = pd.read_csv(
            QRECC_DIR / backend.lower() / "source_strata.csv"
        )
        stored = stored.loc[
            stored["system"].eq("pretrained") & stored["arm"].eq("R")
        ].set_index("conversation_source")
        derived = performance.loc[
            performance["dataset"].eq("QReCC Test (dataset OOD)")
            & performance["backend"].eq(backend)
            & performance["method"].eq("R-B64")
        ].set_index("structure_key")
        for source in source_labels:
            if not (
                int(stored.loc[source, "n"])
                == int(derived.loc[source, "n_queries"])
                and np.isclose(
                    float(stored.loc[source, "MRR"]),
                    float(derived.loc[source, "MRR"]),
                    rtol=0.0,
                    atol=1e-15,
                )
            ):
                raise RuntimeError(
                    f"QReCC source-strata drift for {backend}/{source}."
                )
    return distribution, performance


def validated_inputs():
    latency = pd.read_csv(
        LATENCY_DIR / "reproducibility_route_components_summary.csv"
    )
    quality, comparisons = load_quality(
        (("BM25", BM25_DIR), ("ANCE", ANCE_DIR)),
        2_514,
    )
    qrecc_quality, qrecc_comparisons = load_quality(
        (
            ("BM25", QRECC_DIR / "bm25"),
            ("ANCE", QRECC_DIR / "ance"),
        ),
        8_209,
    )

    measured = pd.read_csv(
        NSIGHT_DIR / "topiocqa_t5_encoder_summary.csv"
    )
    components = pd.read_csv(
        NSIGHT_DIR / "topiocqa_t5_encoder_component_metrics.csv"
    )
    component_lookup = components.set_index(
        ["route", "budget_label", "component"]
    )
    resource_rows = []
    for route in ("I", "R"):
        for budget in BUDGETS:
            exact_label = f"B{budget}"
            has_exact = (
                measured["route"].eq(route)
                & measured["budget_label"].eq(exact_label)
            ).any()
            label = exact_label if has_exact else "B128-B512"
            row = measured.loc[
                measured["route"].eq(route)
                & measured["budget_label"].eq(label)
            ].iloc[0]
            resource = {
                "route": route,
                "budget": budget,
                "fp32_flops": row.fp32_flops,
                "dram_bytes": row.dram_bytes,
                "kernel_replay_dram_bytes": row.kernel_replay_dram_bytes,
            }
            for component, _, _ in ENCODER_COMPONENTS:
                component_row = component_lookup.loc[
                    (route, label, component)
                ]
                resource[f"fp32_flops_{component}"] = (
                    component_row.fp32_flops
                )
                resource[f"dram_bytes_{component}"] = (
                    row.dram_bytes
                    * component_row["dram__bytes.sum"]
                    / row.kernel_replay_dram_bytes
                )
            for metric in ("fp32_flops", "dram_bytes"):
                resource[f"{metric}_display_other"] = (
                    resource[f"{metric}_layernorm_residual_other"]
                    + resource[f"{metric}_embedding"]
                )
            resource_rows.append(resource)
    resources = pd.DataFrame(resource_rows)
    latency_manifest = json.loads(
        (LATENCY_DIR / "reproducibility_manifest.json").read_text()
    )

    expected = {
        ("I", 64),
        ("I", 512),
        ("R", 64),
        ("D", 64),
        ("IRD", 64),
    }
    observed_latency = set(
        latency[["route", "budget"]].itertuples(index=False, name=None)
    )
    if observed_latency != expected or latency.duplicated(
        ["route", "budget"]
    ).any():
        raise RuntimeError(
            "Latency summary is not the complete primary-route design."
        )
    if not latency["runs"].eq(5).all():
        raise RuntimeError("The figure requires five independent latency runs.")
    if not {
        "ci95_low_total_ms",
        "ci95_high_total_ms",
    }.issubset(latency.columns):
        raise RuntimeError("Latency summary lacks the 95% t-interval.")
    if not latency_manifest["all_integrity_checks_passed"]:
        raise RuntimeError("Latency integrity checks did not pass.")
    expected_resources = {
        (route, budget) for route in ("I", "R") for budget in BUDGETS
    }
    observed_resources = set(
        resources[["route", "budget"]].itertuples(index=False, name=None)
    )
    if observed_resources != expected_resources or resources.duplicated(
        ["route", "budget"]
    ).any():
        raise RuntimeError("NCU resources are not the complete I/R 2x4 design.")
    if not resources[
        ["fp32_flops", "dram_bytes"]
    ].gt(0).all().all():
        raise RuntimeError("NCU resource measurements must be positive.")
    for metric in ("fp32_flops", "dram_bytes"):
        component_sum = resources[
            [f"{metric}_{component}" for component, _, _ in ENCODER_COMPONENTS]
        ].sum(axis=1)
        if not np.allclose(
            component_sum,
            resources[metric],
            rtol=1e-12,
            atol=1e-6,
        ):
            raise RuntimeError(f"Encoder {metric} components do not close.")
        display_sum = resources[
            [
                f"{metric}_{component}"
                for component, _, _ in DISPLAY_ENCODER_COMPONENTS
            ]
        ].sum(axis=1)
        if not np.allclose(
            display_sum,
            resources[metric],
            rtol=1e-12,
            atol=1e-6,
        ):
            raise RuntimeError(
                f"Displayed encoder {metric} components do not close."
            )

    return (
        latency,
        quality,
        comparisons,
        qrecc_quality,
        qrecc_comparisons,
        resources,
    )


def load_latency_comparisons() -> pd.DataFrame:
    """Return paired run-level latency differences versus I-B512."""
    runs = pd.read_csv(LATENCY_RUNS_SOURCE)
    run_column = (
        "independent_run"
        if "independent_run" in runs.columns
        else "run"
    )
    baseline = runs.loc[
        runs["route"].eq("I") & runs["budget"].eq(512),
        [run_column, "total_ms_per_query"],
    ].rename(
        columns={
            run_column: "run",
            "total_ms_per_query": "reference_ms",
        }
    )
    definitions = (
        ("I64", "I-B64", "I", 64),
        ("R", "R-B64", "R", 64),
        ("D", "D-B64", "D", 64),
        ("IRD", "IRD-B64", "IRD", 64),
    )
    rows = []
    t_critical_df4 = 2.7764451051977987
    for key, label, route, budget in definitions:
        candidate = runs.loc[
            runs["route"].eq(route) & runs["budget"].eq(budget),
            [run_column, "total_ms_per_query"],
        ].rename(
            columns={
                run_column: "run",
                "total_ms_per_query": "candidate_ms",
            }
        )
        paired = candidate.merge(
            baseline,
            on="run",
            how="inner",
            validate="one_to_one",
        ).sort_values("run")
        if paired["run"].tolist() != [1, 2, 3, 4, 5]:
            raise RuntimeError(
                f"Incomplete paired latency runs for {label}."
            )
        differences = (
            paired["candidate_ms"] - paired["reference_ms"]
        ).to_numpy(dtype=float)
        mean = float(differences.mean())
        half_width = float(
            t_critical_df4
            * differences.std(ddof=1)
            / np.sqrt(len(differences))
        )
        rows.append(
            {
                "system_key": key,
                "comparison": f"{label}_minus_I-B512",
                "candidate": label,
                "reference": "I-B512",
                "runs": len(differences),
                "mean_delta_ms": mean,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "interval": "paired two-sided Student-t, df=4",
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(LATENCY_COMPARISON_EXPORT, index=False)
    return result


def add_quality_bracket(
    axis,
    *,
    left_x: float,
    right_x: float,
    y: float,
    low: float,
    high: float,
) -> None:
    height = 0.004
    axis.plot(
        [left_x, left_x, right_x, right_x],
        [y, y + height, y + height, y],
        color="#344054",
        linewidth=0.9,
        clip_on=False,
    )
    axis.text(
        (left_x + right_x) / 2,
        y + height + 0.003,
        f"95% CI [{low:+.3f}, {high:+.3f}]",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#344054",
    )


def add_bootstrap_errorbars(
    axis,
    x,
    values,
    lows,
    highs,
    *,
    color="#475467",
    fill_color=None,
    bar_origin=0.0,
) -> None:
    x = np.atleast_1d(np.asarray(x, dtype=float))
    values = np.atleast_1d(np.asarray(values, dtype=float))
    lows = np.atleast_1d(np.asarray(lows, dtype=float))
    highs = np.atleast_1d(np.asarray(highs, dtype=float))
    if not (x.shape == values.shape == lows.shape == highs.shape):
        raise ValueError("Bootstrap error-bar arrays must have equal shapes.")

    inside_color = color
    if fill_color is not None:
        red, green, blue = matplotlib.colors.to_rgb(fill_color)
        channels = []
        for channel in (red, green, blue):
            channels.append(
                channel / 12.92
                if channel <= 0.04045
                else ((channel + 0.055) / 1.055) ** 2.4
            )
        luminance = (
            0.2126 * channels[0]
            + 0.7152 * channels[1]
            + 0.0722 * channels[2]
        )
        if luminance < 0.40:
            inside_color = "white"

    for position, value, low, high in zip(
        x, values, lows, highs, strict=True
    ):
        lower_color, upper_color = (
            (inside_color, color)
            if value >= bar_origin
            else (color, inside_color)
        )
        axis.vlines(
            position,
            low,
            value,
            colors=lower_color,
            linewidth=0.8,
            zorder=4,
        )
        axis.vlines(
            position,
            value,
            high,
            colors=upper_color,
            linewidth=0.8,
            zorder=4,
        )
        for cap_y, cap_color in (
            (low, lower_color),
            (high, upper_color),
        ):
            axis.plot(
                position,
                cap_y,
                marker="_",
                markersize=5,
                markeredgewidth=0.8,
                color=cap_color,
                linestyle="none",
                zorder=4,
            )


def add_method_callouts(axis, positions, offsets, *, y: float) -> None:
    transform = axis.get_xaxis_transform()
    for x, label in (
        (
            positions[0] + offsets["R"],
            "R-B64\nROCC primary",
        ),
        (
            positions[3] + offsets["I"],
            "I-B512\nIterCQR baseline",
        ),
    ):
        axis.annotate(
            label,
            xy=(x, 0.0),
            xycoords=transform,
            xytext=(x, y),
            textcoords=transform,
            ha="center",
            va="top",
            fontsize=8.5,
            color="#344054",
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#344054",
                "linewidth": 0.8,
                "shrinkA": 2,
                "shrinkB": 2,
            },
            annotation_clip=False,
        )


def draw_encoder_resources(
    flops_axis,
    dram_axis,
    resources: pd.DataFrame,
) -> None:
    lookup = resources.set_index(["budget", "route"])
    width = 0.30
    bar_groups, positions, _ = grouped_bar_layout(
        (2,) * len(BUDGETS),
        width,
    )
    route_positions = {
        route: np.asarray([group[index] for group in bar_groups])
        for index, route in enumerate(("I", "R"))
    }
    specifications = (
        (
            flops_axis,
            "fp32_flops",
            1e9,
            "Measured T5-encoder FP32 work by component",
            "GFLOP per encoder forward\n(batch size 16, FP32)",
            lambda value: f"{value:.1f}",
            "Nsight Compute; canonical full-Dev mean-length shapes",
        ),
        (
            dram_axis,
            "dram_bytes",
            1e9,
            "Whole-range T5-encoder DRAM traffic by component allocation",
            "GB per encoder forward\n(batch size 16, FP32)",
            lambda value: f"{value:.2f}",
            "Range-replay whole-forward total; kernel-replay component "
            "shares",
        ),
    )
    for panel_index, (
        axis,
        column,
        scale,
        title,
        ylabel,
        formatter,
        subtitle,
    ) in enumerate(specifications):
        values_by_route = {
            route: np.asarray(
                [
                    float(lookup.loc[(budget, route), column]) / scale
                    for budget in BUDGETS
                ]
            )
            for route in ("I", "R")
        }
        panel_max = max(values.max() for values in values_by_route.values())
        for route in ("I", "R"):
            bottoms = np.zeros(len(BUDGETS), dtype=float)
            for component, component_label, color in DISPLAY_ENCODER_COMPONENTS:
                segment = np.asarray(
                    [
                        float(
                            lookup.loc[
                                (budget, route), f"{column}_{component}"
                            ]
                        )
                        / scale
                        for budget in BUDGETS
                    ],
                    dtype=float,
                )
                bars = axis.bar(
                    route_positions[route],
                    segment,
                    width,
                    bottom=bottoms,
                    color=color,
                    edgecolor="none",
                    linewidth=0,
                    label=component_label if route == "I" else None,
                )
                fade_middle_budgets(bars)
                for bar in bars:
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")
                    bar.set_linewidth(0)
                bottoms += segment
            for x, value in zip(
                route_positions[route],
                values_by_route[route],
                strict=True,
            ):
                axis.text(
                    x,
                    value + panel_max * 0.012,
                    f"{route} {formatter(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                    color="#252b33",
                )
        for index, budget in enumerate(BUDGETS):
            i_value = values_by_route["I"][index]
            r_value = values_by_route["R"][index]
            reduction = 100.0 * (1.0 - r_value / i_value)
            axis.text(
                positions[index],
                max(i_value, r_value) + panel_max * 0.085,
                f"R −{reduction:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7.7,
                color="#344054",
            )
        axis.set_ylabel(ylabel)
        axis.set_title(
            title + "\n" + subtitle,
            loc="left",
            fontsize=9.8,
            pad=7,
        )
        axis.set_ylim(0, panel_max * 1.33)
        axis.set_xticks(positions, [f"B{budget}" for budget in BUDGETS])
        axis.grid(axis="x", visible=False)
        axis.grid(visible=False)
        if panel_index == 0:
            axis.legend(
                frameon=True,
                facecolor="white",
                edgecolor="#d0d5dd",
                framealpha=1,
                fontsize=7.5,
                ncol=2,
                loc="upper left",
                borderaxespad=0.45,
            )
        else:
            axis.set_xlabel(
                "Fixed total rewriter-input budget",
                labelpad=34,
            )
        add_preference_hint(axis, higher_is_better=False)


def draw_quality(
    axis,
    quality: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    dataset: str,
    n: int,
    backend: str,
    backend_label: str | None = None,
    show_legend: bool,
    show_callouts: bool,
    show_d_comparison: bool = False,
    ylim_padding: float = 0.20,
) -> None:
    frame = quality.loc[quality["backend"].eq(backend)]
    lookup = frame.set_index(["budget", "route"])
    width = 0.145
    bar_groups, positions, _ = grouped_bar_layout(
        (len(ROUTES),) * len(BUDGETS),
        width,
    )
    route_positions = {
        route: np.asarray([group[index] for group in bar_groups])
        for index, route in enumerate(ROUTES)
    }
    offsets = {
        route: float(route_positions[route][0] - positions[0])
        for route in ROUTES
    }
    quality_max = float(frame["MRR_ci95_high"].max())
    group_maxima = np.asarray(
        [
            max(
                float(lookup.loc[(budget, route), "MRR_ci95_high"])
                for route in ROUTES
            )
            for budget in BUDGETS
        ],
        dtype=float,
    )
    value_label_lift = {
        "I": 0.015,
        "R": 0.045,
        "D": 0.015,
        "IRD": 0.045,
    }
    for route in ROUTES:
        values = np.asarray(
            [lookup.loc[(budget, route), "MRR"] for budget in BUDGETS],
            dtype=float,
        )
        lows = np.asarray(
            [
                lookup.loc[(budget, route), "MRR_ci95_low"]
                for budget in BUDGETS
            ],
            dtype=float,
        )
        highs = np.asarray(
            [
                lookup.loc[(budget, route), "MRR_ci95_high"]
                for budget in BUDGETS
            ],
            dtype=float,
        )
        bars = axis.bar(
            route_positions[route],
            values,
            width,
            color=COLORS[route],
            edgecolor="none",
            linewidth=0,
            label=route,
        )
        fade_middle_budgets(bars)
        add_bootstrap_errorbars(
            axis,
            route_positions[route],
            values,
            lows,
            highs,
            fill_color=COLORS[route],
        )
        for index, (bar, value, high) in enumerate(
            zip(bars, values, highs, strict=True)
        ):
            center = bar.get_x() + bar.get_width() / 2
            axis.annotate(
                f"{value:.3f}",
                xy=(center, high + 0.002),
                xytext=(
                    center,
                    group_maxima[index] + value_label_lift[route],
                ),
                ha="center",
                va="bottom",
                fontsize=8.5,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#69717c",
                    "linewidth": 0.7,
                    "shrinkA": 3,
                    "shrinkB": 1,
                },
                annotation_clip=False,
            )
    axis.set_title(
        f"Retrieval quality, {dataset}, "
        f"{backend_label or backend}, n={n:,}",
        loc="left",
        fontsize=10,
        pad=6,
    )
    axis.set_ylabel("MRR", fontsize=10)
    axis.set_xticks(positions, [f"B{budget}" for budget in BUDGETS])
    axis.tick_params(axis="both", labelsize=10)
    axis.tick_params(axis="x", labelbottom=True)
    legend_headroom = 0.055 if show_legend else 0.0
    axis.set_ylim(
        0,
        quality_max + ylim_padding + legend_headroom,
    )
    axis.grid(axis="x", visible=False)
    if show_legend:
        axis.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#d0d5dd",
            framealpha=1,
            fontsize=8,
            ncol=4,
            loc="upper left",
            borderaxespad=0.45,
        )

    intervals = comparisons.loc[
        comparisons["backend"].eq(backend)
    ].set_index("comparison")
    short = intervals.loc["IRD64_minus_I64"]
    long = intervals.loc["IRD64_minus_I512"]
    compressed = intervals.loc["R64_minus_I512"]
    add_quality_bracket(
        axis,
        left_x=positions[0] + offsets["I"],
        right_x=positions[0] + offsets["IRD"],
        y=quality_max + 0.085,
        low=float(short.ci95_low),
        high=float(short.ci95_high),
    )
    add_quality_bracket(
        axis,
        left_x=positions[0] + offsets["IRD"],
        right_x=positions[3] + offsets["I"],
        y=quality_max + 0.125,
        low=float(long.ci95_low),
        high=float(long.ci95_high),
    )
    add_quality_bracket(
        axis,
        left_x=positions[0] + offsets["R"],
        right_x=positions[3] + offsets["I"],
        y=quality_max + 0.155,
        low=float(compressed.ci95_low),
        high=float(compressed.ci95_high),
    )
    if show_d_comparison:
        direct = intervals.loc["D64_minus_I512"]
        add_quality_bracket(
            axis,
            left_x=positions[0] + offsets["D"],
            right_x=positions[3] + offsets["I"],
            y=quality_max + 0.185,
            low=float(direct.ci95_low),
            high=float(direct.ci95_high),
        )
    if show_callouts:
        axis.set_xlabel(
            "Fixed total rewriter-input budget",
            fontsize=10,
            labelpad=34,
        )
        add_method_callouts(axis, positions, offsets, y=-0.15)
    add_preference_hint(axis, higher_is_better=True)


def draw_recall(
    axis,
    quality: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    dataset: str,
    n: int,
    backend: str,
    backend_label: str | None = None,
    show_legend: bool,
    bold_baseline_values: bool = False,
) -> None:
    frame = quality.loc[quality["backend"].eq(backend)]
    lookup = frame.set_index(["budget", "route"])
    metrics = ("R@10", "R@100", "R@1000")
    systems = (
        ("I64", "I", 64, "I-B64 (IterCQR truncated)"),
        ("I512", "I", 512, "I-B512 (IterCQR baseline)"),
        ("R", "R", 64, "R-B64"),
        ("D", "D", 64, "D-B64"),
        ("IRD", "IRD", 64, "IRD-B64"),
    )
    width = 0.13
    bar_groups, positions, _ = grouped_bar_layout(
        (len(systems),) * len(metrics),
        width,
    )
    recall_group_offsets = np.asarray((-0.14, 0.0, 0.14))
    bar_groups = tuple(
        group + recall_group_offsets[index]
        for index, group in enumerate(bar_groups)
    )
    positions = positions + recall_group_offsets
    system_positions = {
        key: np.asarray([group[index] for group in bar_groups])
        for index, (key, _, _, _) in enumerate(systems)
    }
    offsets = {
        key: float(system_positions[key][0] - positions[0])
        for key, _, _, _ in systems
    }
    baseline = np.asarray(
        [lookup.loc[(512, "I"), metric] for metric in metrics],
        dtype=float,
    )
    group_maxima = np.asarray(
        [
            max(
                float(
                    lookup.loc[
                        (budget, route), f"{metric}_ci95_high"
                    ]
                )
                for _, route, budget, _ in systems
            )
            for metric in metrics
        ],
        dtype=float,
    )
    label_lift = {
        "I64": 0.030 * RECALL_LAYOUT_SCALE,
        "I512": 0.210 * RECALL_LAYOUT_SCALE,
        "R": 0.390 * RECALL_LAYOUT_SCALE,
        "D": 0.030 * RECALL_LAYOUT_SCALE,
        "IRD": 0.210 * RECALL_LAYOUT_SCALE,
    }
    recall_colors = {
        "I64": "#c8cdd2",
        "I512": RECALL_COLORS["I"],
        "R": RECALL_COLORS["R"],
        "D": RECALL_COLORS["D"],
        "IRD": RECALL_COLORS["IRD"],
    }
    family_groups = (
        ("I64", "I512", "IterCQR", "#9099a3"),
        ("R", "IRD", "ROCC B64", "#7098b8"),
    )
    intervals = comparisons.loc[
        comparisons["backend"].eq(backend)
    ].set_index("comparison")
    comparison_names = {
        "I64": "I64_minus_I512",
        "R": "R64_minus_I512",
        "D": "D64_minus_I512",
        "IRD": "IRD64_minus_I512",
    }
    bracket_y_by_metric = group_maxima + 0.55 * RECALL_LAYOUT_SCALE
    for metric_index, center in enumerate(positions):
        bracket_y = bracket_y_by_metric[metric_index]
        for left_key, right_key, _, background in family_groups:
            left = center + offsets[left_key] - width / 2
            right = center + offsets[right_key] + width / 2
            axis.fill(
                [left, right, right, left],
                [0.0, 0.0, bracket_y, bracket_y],
                facecolor=background,
                edgecolor="none",
                alpha=0.075,
                zorder=0,
            )
        axis.hlines(
            baseline[metric_index],
            center + offsets["I64"] - width / 2,
            center + offsets["IRD"] + width / 2,
            colors="#667085",
            linestyles=(0, (2.5, 2.5)),
            linewidth=0.8,
            alpha=0.55,
            zorder=0.5,
        )
    for system_key, route, budget, label in systems:
        values = np.asarray(
            [lookup.loc[(budget, route), metric] for metric in metrics],
            dtype=float,
        )
        lows = np.asarray(
            [
                lookup.loc[(budget, route), f"{metric}_ci95_low"]
                for metric in metrics
            ],
            dtype=float,
        )
        highs = np.asarray(
            [
                lookup.loc[(budget, route), f"{metric}_ci95_high"]
                for metric in metrics
            ],
            dtype=float,
        )
        relative = 100.0 * (values - baseline) / baseline
        bars = axis.bar(
            system_positions[system_key],
            values,
            width,
            color=recall_colors[system_key],
            edgecolor="none",
            linewidth=0,
            label=label,
        )
        add_bootstrap_errorbars(
            axis,
            system_positions[system_key],
            values,
            lows,
            highs,
            fill_color=recall_colors[system_key],
        )
        for metric_index, (bar, value, high, change) in enumerate(zip(
            bars,
            values,
            highs,
            relative,
            strict=True,
        )):
            center = bar.get_x() + bar.get_width() / 2
            annotation_y = (
                group_maxima[metric_index] + label_lift[system_key]
            )
            axis.annotate(
                f"{value:.3f}\nΔ {change:+.1f}%",
                xy=(center, high + 0.006),
                xytext=(center, annotation_y),
                ha="center",
                va="bottom",
                fontsize=13.7,
                fontweight=(
                    "bold"
                    if bold_baseline_values and system_key == "I512"
                    else "normal"
                ),
                linespacing=1.05,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#69717c",
                    "linewidth": 0.7,
                    "shrinkA": 3,
                    "shrinkB": 1,
                },
                annotation_clip=False,
            )
            if system_key in comparison_names:
                metric_key = metrics[metric_index].replace("@", "")
                interval = intervals.loc[
                    f"{comparison_names[system_key]}_{metric_key}"
                ]
                significant = (
                    float(interval.ci95_low) > 0.0
                    or float(interval.ci95_high) < 0.0
                )
                axis.annotate(
                    "†" if significant else "‡",
                    xy=(center, annotation_y),
                    xytext=(-30, 12),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=20.2,
                    fontweight="normal",
                    color="#344054",
                    annotation_clip=False,
                )
    for metric_index, center in enumerate(positions):
        bracket_y = bracket_y_by_metric[metric_index]
        for left_key, right_key, family, _ in family_groups:
            left = center + offsets[left_key] - width / 2
            right = center + offsets[right_key] + width / 2
            axis.plot(
                [left, left, right, right],
                [
                    bracket_y - 0.014,
                    bracket_y,
                    bracket_y,
                    bracket_y - 0.014,
                ],
                color="#475467",
                linewidth=0.9,
                clip_on=False,
            )
            axis.text(
                (left + right) / 2,
                bracket_y + 0.010,
                family,
                ha="center",
                va="bottom",
                fontsize=13.6,
                fontweight="bold",
                color="#475467",
                clip_on=False,
            )
    axis.set_title(
        f"{dataset}, {backend_label or backend}, n={n:,}",
        loc="left",
        fontsize=10,
        pad=6,
    )
    axis.set_ylabel("Recall")
    axis.set_xticks(positions, metrics)
    axis.tick_params(axis="x", labelbottom=True)
    axis.set_ylim(0, 2.20)
    axis.set_yticks(np.arange(0.0, 1.01, 0.2))
    axis.grid(False)
    if show_legend:
        handles, labels = axis.get_legend_handles_labels()
        legend_order = (0, 1, 4, 2, 3)
        axis.legend(
            [handles[index] for index in legend_order],
            [labels[index] for index in legend_order],
            frameon=True,
            facecolor="white",
            edgecolor="#d0d5dd",
            framealpha=1,
            fontsize=13.7,
            ncol=2,
            loc="upper left",
            borderaxespad=0.45,
        )
    add_preference_hint(
        axis,
        higher_is_better=True,
        fontsize=13.6,
    )


def draw_metric_comparison(
    axis,
    quality: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    metric: str,
    dataset: str,
    n: int,
    backend: str,
    backend_label: str | None = None,
    show_legend: bool,
) -> None:
    frame = quality.loc[quality["backend"].eq(backend)]
    lookup = frame.set_index(["budget", "route"])
    systems = (
        ("I64", "I", 64, "I-B64 (IterCQR truncated)"),
        ("I512", "I", 512, "I-B512 (IterCQR baseline)"),
        ("R", "R", 64, "R-B64"),
        ("D", "D", 64, "D-B64"),
        ("IRD", "IRD", 64, "IRD-B64"),
    )
    width = 0.13
    bar_groups, _, family_spans = grouped_bar_layout((2, 3), width)
    extra_family_gap = (
        width
        * BAR_GROUP_GAP_RATIO
        * (COMPARISON_FAMILY_GAP_MULTIPLIER - 1.0)
    )
    half_gap = extra_family_gap / 2.0
    bar_groups = (
        bar_groups[0] - half_gap,
        bar_groups[1] + half_gap,
    )
    family_spans = (
        (
            family_spans[0][0] - half_gap,
            family_spans[0][1] - half_gap,
        ),
        (
            family_spans[1][0] + half_gap,
            family_spans[1][1] + half_gap,
        ),
    )
    offsets = {
        key: float(position)
        for key, position in zip(
            ("I64", "I512", "R", "D", "IRD"),
            np.concatenate(bar_groups),
            strict=True,
        )
    }
    colors = {
        "I64": "#c8cdd2",
        "I512": RECALL_COLORS["I"],
        "R": RECALL_COLORS["R"],
        "D": RECALL_COLORS["D"],
        "IRD": RECALL_COLORS["IRD"],
    }
    baseline = float(lookup.loc[(512, "I"), metric])
    intervals = comparisons.loc[
        comparisons["backend"].eq(backend)
    ].set_index("comparison")
    comparison_names = {
        "I64": "I64_minus_I512",
        "R": "R64_minus_I512",
        "D": "D64_minus_I512",
        "IRD": "IRD64_minus_I512",
    }
    comparison_suffix = {
        "MRR": "",
        "nDCG@3": "_nDCG3",
    }[metric]
    maximum = max(
        float(lookup.loc[(budget, route), f"{metric}_ci95_high"])
        for _, route, budget, _ in systems
    )
    comparison_transform = axis.get_xaxis_transform()
    topiocqa = dataset.startswith("TopiOCQA")
    if topiocqa:
        bracket_y = (
            METRIC_COMPARISON_BRACKET_AXES_Y_TOPIOCQA_BM25
            if backend.lower() == "bm25"
            else METRIC_COMPARISON_BRACKET_AXES_Y_TOPIOCQA_ANCE
        )
    else:
        bracket_y = METRIC_COMPARISON_BRACKET_AXES_Y
    value_label_transform = matplotlib.transforms.offset_copy(
        comparison_transform,
        fig=axis.figure,
        y=METRIC_COMPARISON_VALUE_LABEL_POINTS,
        units="points",
    )
    value_label_font = font_manager.FontProperties(
        family=PLOT_FONT,
        size=8.5,
    )
    significance_marker_font = font_manager.FontProperties(
        family=PLOT_FONT,
        size=13,
    )
    family_groups = (
        ("I64", "I512", "IterCQR", "#9099a3"),
        ("R", "IRD", "ROCC B64", "#7098b8"),
    )
    family_sublabels = {
        "I64": "Truncated",
        "I512": "Baseline",
        "R": "Primary",
        "D": "Direct",
        "IRD": "Fusion",
    }
    for group_index, (_, _, _, background) in enumerate(family_groups):
        left, right = family_spans[group_index]
        axis.fill(
            [left, right, right, left],
            [0.0, 0.0, bracket_y, bracket_y],
            transform=comparison_transform,
            facecolor=background,
            edgecolor="none",
            alpha=0.075,
            zorder=0,
        )
    axis.axhline(
        baseline,
        color="#667085",
        linestyle=(0, (2.5, 2.5)),
        linewidth=0.8,
        alpha=0.55,
        zorder=0.5,
    )
    for system_key, route, budget, label in systems:
        value = float(lookup.loc[(budget, route), metric])
        low = float(
            lookup.loc[(budget, route), f"{metric}_ci95_low"]
        )
        high = float(
            lookup.loc[(budget, route), f"{metric}_ci95_high"]
        )
        change = 100.0 * (value - baseline) / baseline
        significance_marker = None
        if system_key in comparison_names:
            interval = intervals.loc[
                f"{comparison_names[system_key]}{comparison_suffix}"
            ]
            significant = (
                float(interval.ci95_low) > 0.0
                or float(interval.ci95_high) < 0.0
            )
            significance_marker = "†" if significant else "‡"
        bar = axis.bar(
            offsets[system_key],
            value,
            width,
            color=colors[system_key],
            edgecolor="none",
            linewidth=0,
            label=label,
        )[0]
        center = bar.get_x() + bar.get_width() / 2
        add_bootstrap_errorbars(
            axis,
            np.asarray([center]),
            np.asarray([value]),
            np.asarray([low]),
            np.asarray([high]),
            fill_color=colors[system_key],
        )
        axis.annotate(
            f"{value:.3f}\nΔ {change:+.1f}%",
            xy=(center, high + 0.004),
            xycoords="data",
            xytext=(center, bracket_y),
            textcoords=value_label_transform,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if system_key == "I512" else "normal",
            linespacing=1.05,
            arrowprops={
                "arrowstyle": "-",
                "color": "#69717c",
                "linewidth": 0.7,
                "shrinkA": 3,
                "shrinkB": 1,
            },
            annotation_clip=False,
        )
        if significance_marker is not None:
            label_lines = (
                f"{value:.3f}",
                f"Δ {change:+.1f}%",
            )
            label_half_width = 0.5 * max(
                TextPath(
                    (0, 0),
                    line,
                    prop=value_label_font,
                ).get_extents().width
                for line in label_lines
            )
            marker_half_width = 0.5 * TextPath(
                (0, 0),
                significance_marker,
                prop=significance_marker_font,
            ).get_extents().width
            marker_x_offset = -(
                label_half_width
                + marker_half_width
                + METRIC_SIGNIFICANCE_MARKER_GAP_POINTS
            )
            axis.annotate(
                significance_marker,
                xy=(center, bracket_y),
                xycoords=comparison_transform,
                xytext=(
                    marker_x_offset,
                    METRIC_COMPARISON_VALUE_LABEL_POINTS + 10,
                ),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="normal",
                color="#344054",
                annotation_clip=False,
            )
    for group_index, (_, _, family, _) in enumerate(family_groups):
        left, right = family_spans[group_index]
        draw_comparison_family_header(
            axis,
            left=left,
            right=right,
            label=family,
            y=bracket_y,
        )
    for system_key, text in family_sublabels.items():
        axis.annotate(
            text,
            xy=(offsets[system_key], bracket_y),
            xycoords=comparison_transform,
            xytext=(0, METRIC_COMPARISON_SUBLABEL_POINTS),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10.0,
            fontweight="medium",
            color="#475467",
            annotation_clip=False,
        )
    axis.set_title(
        f"{dataset}, {backend_label or backend}, n={n:,}",
        loc="left",
        fontsize=10,
        pad=6,
    )
    axis.set_ylabel(metric)
    axis.set_xticks([0.0], [metric])
    axis.set_xlim(
        family_spans[0][0] - width * 0.55,
        family_spans[-1][1] + width * 0.55,
    )
    axis.set_ylim(0, 0.57 if topiocqa else 0.78)
    axis.set_yticks(
        np.arange(0.0, 0.51 if topiocqa else 0.71, 0.1)
    )
    axis.grid(False)
    if show_legend:
        axis.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#d0d5dd",
            framealpha=1,
            fontsize=8,
            ncol=3,
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0, 0.70, 0.0),
            mode="expand",
            borderaxespad=0.30,
            borderpad=0.20,
            columnspacing=LEGEND_COLUMN_SPACING,
            labelspacing=0.35,
            handlelength=LEGEND_HANDLE_LENGTH,
            handleheight=LEGEND_HANDLE_HEIGHT,
            handletextpad=LEGEND_HANDLE_TEXT_PAD,
        )
    add_preference_hint(
        axis,
        higher_is_better=True,
        axes_x=0.985,
    )


def draw_efficiency_comparison(
    axis,
    values: dict[str, float],
    *,
    title: str,
    ylabel: str,
    decimals: int,
    intervals: dict[str, tuple[float, float]] | None = None,
    comparison_intervals: dict[str, tuple[float, float]] | None = None,
    segments: dict[
        str, tuple[tuple[float, str, str], ...]
    ] | None = None,
    hatched_labels: frozenset[str] = frozenset(),
    show_legend: bool = False,
    family_gap_multiplier: float = 1.0,
) -> None:
    systems = (
        ("I64", "I-B64 (IterCQR truncated)"),
        ("I512", "I-B512 (IterCQR baseline)"),
        ("R", "R-B64"),
        ("D", "D-B64"),
        ("IRD", "IRD-B64"),
    )
    width = 0.13
    bar_groups, _, family_spans = grouped_bar_layout(
        (2, 3),
        width,
        gap_ratio=0.90,
    )
    extra_family_gap = (
        width
        * BAR_GROUP_GAP_RATIO
        * (family_gap_multiplier - 1.0)
    )
    if extra_family_gap:
        half_gap = extra_family_gap / 2.0
        bar_groups = (
            bar_groups[0] - half_gap,
            bar_groups[1] + half_gap,
        )
        family_spans = (
            (
                family_spans[0][0] - half_gap,
                family_spans[0][1] - half_gap,
            ),
            (
                family_spans[1][0] + half_gap,
                family_spans[1][1] + half_gap,
            ),
        )
    offsets = {
        key: float(position)
        for key, position in zip(
            ("I64", "I512", "R", "D", "IRD"),
            np.concatenate(bar_groups),
            strict=True,
        )
    }
    colors = {
        "I64": "#c8cdd2",
        "I512": RECALL_COLORS["I"],
        "R": RECALL_COLORS["R"],
        "D": RECALL_COLORS["D"],
        "IRD": RECALL_COLORS["IRD"],
    }
    family_groups = (
        ("I64", "I512", "IterCQR", "#9099a3"),
        ("R", "IRD", "ROCC B64", "#7098b8"),
    )
    family_sublabels = {
        "I64": "Truncated",
        "I512": "Baseline",
        "R": "Primary",
        "D": "Direct",
        "IRD": "Fusion",
    }
    baseline = values["I512"]
    bracket_y = 0.74 if show_legend else 0.82
    value_label_points = -72.0
    sublabel_points = -12.0
    maximum = max(
        intervals[key][1] if intervals else value
        for key, value in values.items()
    )
    comparison_transform = axis.get_xaxis_transform()
    value_label_transform = matplotlib.transforms.offset_copy(
        comparison_transform,
        fig=axis.figure,
        y=value_label_points,
        units="points",
    )
    for group_index, (_, _, _, background) in enumerate(family_groups):
        left, right = family_spans[group_index]
        axis.fill(
            [left, right, right, left],
            [0.0, 0.0, bracket_y, bracket_y],
            transform=comparison_transform,
            facecolor=background,
            edgecolor="none",
            alpha=0.075,
            zorder=0,
        )
    if comparison_intervals is not None:
        axis.plot(
            [
                offsets["I64"] - width / 2.0,
                offsets["IRD"] + width / 2.0,
            ],
            [baseline, baseline],
            color="#667085",
            linestyle=(0, (3, 2)),
            linewidth=0.8,
            alpha=0.60,
            zorder=3,
        )
    legend_labels: set[str] = set()
    for key, label in systems:
        value = values[key]
        error = None
        anchor = value
        if intervals:
            low, high = intervals[key]
            error = np.asarray([[value - low], [high - value]])
            anchor = high
        bars = []
        bottom = 0.0
        if segments is not None:
            for segment_value, color, segment_label in segments[key]:
                legend_label = None
                if show_legend and segment_label not in legend_labels:
                    legend_label = segment_label
                    legend_labels.add(segment_label)
                bar = axis.bar(
                    offsets[key],
                    segment_value,
                    width,
                    bottom=bottom,
                    color=color,
                    edgecolor="none",
                    linewidth=0,
                    label=legend_label,
                )[0]
                if segment_label in hatched_labels:
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")
                    bar.set_linewidth(0)
                bars.append(
                    bar
                )
                bottom += segment_value
        else:
            bars.append(
                axis.bar(
                    offsets[key],
                    value,
                    width,
                    color=colors[key],
                    edgecolor="none",
                    linewidth=0,
                    label=label,
                )[0]
            )
        change = 100.0 * (value - baseline) / baseline
        center = offsets[key]
        if error is not None:
            axis.errorbar(
                center,
                value,
                yerr=error,
                fmt="none",
                ecolor="#475467",
                elinewidth=0.8,
                capsize=3,
                capthick=0.8,
                zorder=4,
            )
        axis.annotate(
            f"{value:.{decimals}f}\nΔ {change:+.1f}%",
            xy=(center, anchor + maximum * 0.01),
            xycoords="data",
            xytext=(center, bracket_y),
            textcoords=value_label_transform,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if key == "I512" else "normal",
            linespacing=1.48,
            arrowprops={
                "arrowstyle": "-",
                "color": "#69717c",
                "linewidth": 0.7,
                "shrinkA": 3,
                "shrinkB": 1,
            },
            annotation_clip=False,
        )
        if comparison_intervals is not None and key in comparison_intervals:
            ci_low, ci_high = comparison_intervals[key]
            significant = ci_low > 0.0 or ci_high < 0.0
            axis.annotate(
                "†" if significant else "‡",
                xy=(center, bracket_y),
                xycoords=comparison_transform,
                xytext=(-47, value_label_points + 8),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=11,
                fontweight="normal",
                color="#344054",
                annotation_clip=False,
                zorder=6,
            )
    for group_index, (_, _, family, _) in enumerate(family_groups):
        left, right = family_spans[group_index]
        draw_comparison_family_header(
            axis,
            left=left,
            right=right,
            label=family,
            y=bracket_y,
        )
    for key, text in family_sublabels.items():
        axis.annotate(
            text,
            xy=(offsets[key], bracket_y),
            xycoords=comparison_transform,
            xytext=(0, sublabel_points),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=10.0,
            fontweight="medium",
            color="#475467",
            annotation_clip=False,
        )
    title_artist = axis.set_title(
        title,
        loc="left",
        fontsize=10,
        pad=6,
    )
    title_artist.set_linespacing(1.35)
    axis.set_ylabel(ylabel)
    axis.set_xticks([0.0], [""])
    axis.set_xlim(
        family_spans[0][0] - width * 0.55,
        family_spans[-1][1] + width * 0.55,
    )
    axis.set_ylim(0, maximum * (2.15 if show_legend else 1.90))
    axis.grid(False)
    if show_legend:
        axis.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#d0d5dd",
            framealpha=1,
            fontsize=6.5,
            ncol=2,
            loc="upper left",
            columnspacing=LEGEND_COLUMN_SPACING,
            handlelength=LEGEND_HANDLE_LENGTH,
            handleheight=LEGEND_HANDLE_HEIGHT,
            handletextpad=LEGEND_HANDLE_TEXT_PAD,
            labelspacing=0.25,
            bbox_to_anchor=(0.008, 0.992),
            borderaxespad=0.0,
        )
    add_preference_hint(
        axis,
        higher_is_better=False,
        axes_x=1.0,
        axes_y=1.0,
        offset_points=(-5.0, -5.0),
    )


def draw_current_query_only_mrr(
    axis,
    current_query_only: pd.DataFrame,
    *,
    dataset: str,
    n: int,
    backend: str,
    backend_label: str | None = None,
    show_x_label: bool,
) -> None:
    row = current_query_only.loc[
        current_query_only["dataset"].eq(dataset)
        & current_query_only["backend"].eq(backend)
    ]
    if len(row) != 1 or int(row.iloc[0]["n"]) != n:
        raise RuntimeError(
            f"Missing current-query-only MRR for {dataset}/{backend}."
        )
    row = row.iloc[0]
    value = float(row["MRR"])
    low = float(row["MRR_ci95_low"])
    high = float(row["MRR_ci95_high"])
    color = "#5f6b78"
    axis.set_facecolor("#f5f7fa")
    axis.plot(
        [0.0],
        [value],
        marker="o",
        markersize=5.2,
        color=color,
        linestyle="none",
        zorder=30,
    )
    add_bootstrap_errorbars(
        axis,
        [0.0],
        [value],
        [low],
        [high],
        color=color,
    )
    span = max(high - low, value * 0.12, 1e-6)
    axis.text(
        0.0,
        high + 0.12 * span,
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#344054",
    )
    axis.set_title(" ", loc="left", fontsize=10, pad=6)
    axis.set_ylabel("MRR")
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.3f")
    )
    axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axis.set_xticks(
        [0.0],
        ["Current query\nonly" if show_x_label else ""],
    )
    axis.tick_params(axis="both", labelsize=10)
    axis.set_xlim(-0.55, 0.55)
    axis.set_ylim(max(0.0, low - 0.40 * span), high + 0.70 * span)
    axis.grid(axis="y", color="#d9dee5", linewidth=0.5, alpha=0.75)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)


def draw_budget_mrr(
    axis,
    quality: pd.DataFrame,
    *,
    dataset: str,
    n: int,
    backend: str,
    backend_label: str | None = None,
    show_legend: bool,
    show_title: bool = True,
    show_ylabel: bool = True,
) -> None:
    positions = np.arange(len(BUDGETS), dtype=float)
    styles = (
        ("I", "I", COLORS["I"], "o", "-"),
        ("R", "R", COLORS["R"], "s", "--"),
        ("D", "D", COLORS["D"], "^", "-."),
        ("IRD", "IRD", COLORS["IRD"], "D", "-"),
    )
    frame = quality.loc[quality["backend"].eq(backend)]
    panel_lows = []
    panel_highs = []
    for route, label, color, marker, linestyle in styles:
        selected = frame.loc[frame["route"].eq(route)].sort_values("budget")
        if tuple(selected["budget"]) != BUDGETS:
            raise RuntimeError("Incomplete MRR budget trajectory.")
        values = selected["MRR"].to_numpy(dtype=float)
        lows = selected["MRR_ci95_low"].to_numpy(dtype=float)
        highs = selected["MRR_ci95_high"].to_numpy(dtype=float)
        panel_lows.extend(lows)
        panel_highs.extend(highs)
        axis.plot(
            positions,
            values,
            color=color,
            marker=marker,
            markersize=4.8,
            linewidth=1.6,
            linestyle=linestyle,
            label=label,
            zorder=3,
        )
        add_bootstrap_errorbars(
            axis,
            positions,
            values,
            lows,
            highs,
            color=tuple(
                0.72 * channel
                for channel in matplotlib.colors.to_rgb(color)
            ),
        )
        axis.plot(
            positions,
            values,
            color=color,
            marker=marker,
            markersize=4.8,
            linestyle="none",
            label="_nolegend_",
            zorder=30,
        )

    if show_title:
        axis.set_title(
            f"{dataset}, {backend_label or backend}, n={n:,}",
            loc="left",
            fontsize=10,
            pad=6,
        )
    if show_ylabel:
        axis.set_ylabel("MRR")
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.2f")
    )
    axis.set_xticks(positions, [f"B{budget}" for budget in BUDGETS])
    axis.tick_params(axis="both", labelsize=10, labelbottom=True)
    lower = float(min(panel_lows))
    upper = float(max(panel_highs))
    span = upper - lower
    axis.set_ylim(lower - 0.10 * span, upper + 0.18 * span)
    axis.set_xlim(-0.15, 3.15)
    axis.grid(axis="y", color="#d9dee5", linewidth=0.5, alpha=0.75)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    if show_legend:
        legend_handles = [
            Patch(
                facecolor=color,
                edgecolor="none",
                label=label,
            )
            for _, label, color, _, _ in styles
        ]
        axis.legend(
            handles=legend_handles,
            frameon=True,
            facecolor="white",
            edgecolor="#d0d5dd",
            framealpha=1,
            fontsize=7.2,
            ncol=4,
            loc="upper left",
            borderaxespad=0.30,
            borderpad=0.20,
            columnspacing=LEGEND_COLUMN_SPACING,
            handlelength=LEGEND_HANDLE_LENGTH,
            handleheight=LEGEND_HANDLE_HEIGHT,
            handletextpad=LEGEND_HANDLE_TEXT_PAD,
        )
    add_preference_hint(axis, higher_is_better=True)


def draw_depth_strata_mrr(
    axis,
    strata: pd.DataFrame,
    *,
    dataset: str,
    n: int,
    backend: str,
    backend_label: str | None = None,
    show_legend: bool,
    y_max: float,
) -> None:
    """Draw I-B512/I-B64/R-B64/IRD-B64 across NB07 depth strata."""
    systems = (
        ("I512", "I-B512", "#9099a3"),
        ("I64", "I-B64", "#c8cdd2"),
        ("R", "R-B64", COLORS["R"]),
        ("IRD", "IRD-B64", COLORS["IRD"]),
    )
    width = 0.17
    frame = strata.loc[
        strata["dataset"].eq(dataset)
        & strata["backend"].eq(backend)
    ]
    available = set(frame["depth_bin"])
    depth_specs = [
        spec for spec in DEPTH_STRATA if spec[0] in available
    ]
    depth_order = [key for key, _, _, _ in depth_specs]
    labels = [label for _, label, _, _ in depth_specs]
    bar_groups, positions, group_spans = grouped_bar_layout(
        (len(systems),) * len(depth_specs),
        width,
    )
    system_positions = tuple(
        np.asarray([group[index] for group in bar_groups])
        for index in range(len(systems))
    )
    counts = []
    for depth_key in depth_order:
        depth_counts = frame.loc[
            frame["depth_bin"].eq(depth_key), "n"
        ].unique()
        if len(depth_counts) != 1:
            raise RuntimeError("Inconsistent depth-stratum size.")
        counts.append(int(depth_counts[0]))

    for x_values, (system_key, label, color) in zip(
        system_positions, systems, strict=True
    ):
        selected = (
            frame.loc[frame["system_key"].eq(system_key)]
            .set_index("depth_bin")
            .loc[depth_order]
        )
        values = selected["MRR"].to_numpy(dtype=float)
        lows = selected["MRR_ci95_low"].to_numpy(dtype=float)
        highs = selected["MRR_ci95_high"].to_numpy(dtype=float)
        axis.bar(
            x_values,
            values,
            width,
            color=color,
            edgecolor="none",
            linewidth=0,
            label=label,
            zorder=2,
        )
        add_bootstrap_errorbars(
            axis,
            x_values,
            values,
            lows,
            highs,
            color="#566273",
            fill_color=color,
        )

    axis.set_title(
        f"{dataset}, {backend_label or backend}, n={n:,}",
        loc="left",
        fontsize=10,
        pad=6,
    )
    axis.set_ylabel("MRR")
    axis.set_xticks(
        positions,
        [
            f"{label}\nn={count:,}"
            for label, count in zip(labels, counts, strict=True)
        ],
    )
    axis.tick_params(axis="both", labelsize=10)
    axis.set_xlim(
        group_spans[0][0] - width * 0.55,
        group_spans[-1][1] + width * 0.55,
    )
    axis.set_ylim(0, y_max)
    axis.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    axis.yaxis.set_major_formatter(
        matplotlib.ticker.FormatStrFormatter("%.2f")
    )
    axis.grid(axis="y", color="#d9dee5", linewidth=0.5, alpha=0.75)
    axis.grid(axis="x", visible=False)
    axis.set_axisbelow(True)
    if show_legend:
        axis.legend(
            frameon=True,
            facecolor="white",
            edgecolor="#d0d5dd",
            framealpha=1,
            fontsize=7.2,
            ncol=4,
            loc="upper left",
            borderaxespad=0.45,
            columnspacing=1.0,
        )
    add_preference_hint(axis, higher_is_better=True)


def checked_context_lengths(
    selections: list[pd.DataFrame],
    expected_n: int,
) -> list[np.ndarray]:
    reference_ids = None
    values = []
    for rows in selections:
        rows = rows.sort_values("sample_id", kind="mergesort")
        sample_ids = rows["sample_id"].astype(str).to_numpy()
        if len(rows) != expected_n or rows["sample_id"].duplicated().any():
            raise RuntimeError("Invalid context-length population.")
        if reference_ids is None:
            reference_ids = sample_ids
        elif not np.array_equal(reference_ids, sample_ids):
            raise RuntimeError("Context-length populations are not identical.")
        values.append(rows["input_length"].to_numpy(dtype=int))
    return values


def load_context_lengths() -> tuple[list[np.ndarray], list[np.ndarray]]:
    topiocqa = pd.read_csv(
        TOPIOCQA_CONTEXT_SOURCE,
        usecols=["sample_id", "arm", "budget", "input_length"],
    )
    topiocqa_values = checked_context_lengths(
        [
            topiocqa.loc[
                topiocqa["arm"].eq(arm)
                & topiocqa["budget"].eq(budget)
            ]
            for arm, budget in (
                ("I", 512),
                ("I", 64),
                ("pretrained_R", 512),
            )
        ],
        2_514,
    )

    qrecc = pd.read_json(
        QRECC_CONTEXT_SOURCE,
        lines=True,
        compression="gzip",
    )
    base = qrecc["family"].eq("rocc") & qrecc["system"].eq("pretrained")
    qrecc_values = checked_context_lengths(
        [
            qrecc.loc[
                base
                & qrecc["view"].eq(view)
                & qrecc["budget"].eq(budget)
            ]
            for view, budget in (("I", 512), ("I", 64), ("R", 512))
        ],
        8_209,
    )
    return topiocqa_values, qrecc_values


def load_r_b128_minus_b64() -> dict[str, dict[str, tuple[float, float, float]]]:
    """Load the frozen paired MRR saturation comparisons shown in the inset."""
    result = {}
    for dataset, backend_paths in SATURATION_SOURCES.items():
        result[dataset] = {}
        for backend, path in backend_paths.items():
            rows = pd.read_csv(path)
            if dataset == "topiocqa":
                rows = rows.loc[
                    rows["system"].eq("pretrained")
                    & rows["arm"].eq("R")
                    & rows["comparison"].eq("B128-B64")
                ]
                delta_column = "mrr_delta"
                expected_n = 2_514
            else:
                rows = rows.loc[
                    rows["arm"].eq("R")
                    & rows["comparison"].eq("B128 - B64")
                    & rows["metric"].eq("MRR")
                ]
                delta_column = "delta"
                expected_n = 8_209
            if len(rows) != 1:
                raise RuntimeError(
                    f"Missing unique R B128-B64 comparison for "
                    f"{dataset}/{backend}."
                )
            row = rows.iloc[0]
            if int(row["n"]) != expected_n or int(row["replicates"]) != 10_000:
                raise RuntimeError(
                    f"Invalid R B128-B64 bootstrap for {dataset}/{backend}."
                )
            result[dataset][backend] = (
                float(row[delta_column]),
                float(row["ci95_low"]),
                float(row["ci95_high"]),
            )
    return result


def bootstrap_granularity_bar_intervals() -> pd.DataFrame:
    """Rebuild pointwise MRR intervals for the granularity bars."""
    metrics = pd.read_csv(
        GRANULARITY_METRICS_SOURCE,
        usecols=["granularity", "budget", "sample_id", "arm", "MRR"],
    )
    metrics = metrics.loc[
        metrics["budget"].isin((64, 512))
        & metrics["arm"].isin(("R", "D", "I+R+D"))
        & metrics["granularity"].isin(("token", "qa", "turn"))
    ]
    cells: dict[tuple[int, str, str], np.ndarray] = {}
    sample_ids: np.ndarray | None = None
    for (budget, arm, granularity), group in metrics.groupby(
        ["budget", "arm", "granularity"],
        observed=True,
        sort=True,
    ):
        group = group.sort_values("sample_id", kind="mergesort")
        ids = group["sample_id"].astype(str).to_numpy()
        if len(ids) != 2_514 or pd.Series(ids).duplicated().any():
            raise RuntimeError("Invalid granularity bar population.")
        if sample_ids is None:
            sample_ids = ids
        elif not np.array_equal(sample_ids, ids):
            raise RuntimeError("Granularity bar populations are not paired.")
        cells[(int(budget), str(arm), str(granularity))] = group[
            "MRR"
        ].to_numpy(dtype=float)

    expected = {
        (budget, arm, granularity)
        for budget in (64, 512)
        for arm in ("R", "D", "I+R+D")
        for granularity in ("token", "qa", "turn")
    }
    if set(cells) != expected:
        raise RuntimeError("Incomplete granularity bar design.")

    seed = int.from_bytes(
        hashlib.sha256(
            (
                "granularity_bar_ci_v1\0"
                f"{sha256(GRANULARITY_METRICS_SOURCE)}"
            ).encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    draws = {
        key: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for key in cells
    }
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, 2_514, size=(stop - start, 2_514))
        for key, values in cells.items():
            draws[key][start:stop] = values[indices].mean(axis=1)

    rows = []
    for (budget, arm, granularity), values in cells.items():
        low, high = np.quantile(
            draws[(budget, arm, granularity)],
            (0.025, 0.975),
        )
        rows.append(
            {
                "budget": budget,
                "arm": arm,
                "granularity": granularity,
                "n": len(values),
                "MRR": float(values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_seed": seed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["budget", "arm", "granularity"], kind="mergesort"
    )
    GRANULARITY_BAR_CI_SOURCE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(GRANULARITY_BAR_CI_SOURCE, index=False)
    return result


def load_granularity_projection() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the corrected frozen NB07a BM25 granularity projection."""
    projection_manifest = json.loads(
        (GRANULARITY_PROJECTION_DIR / "manifest.json").read_text()
    )
    backend_manifest = json.loads((BM25_DIR / "manifest.json").read_text())
    if (
        not projection_manifest.get("complete")
        or projection_manifest.get("protocol")
        != "nb07a_pretrained_granularity_bm25_v2"
        or projection_manifest["files"]["input_length_summary.csv"]
        != sha256(GRANULARITY_LENGTH_SOURCE)
        or projection_manifest["files"]["metrics_by_query.csv"]
        != sha256(GRANULARITY_METRICS_SOURCE)
        or projection_manifest["files"]["summary.csv"]
        != sha256(GRANULARITY_SUMMARY_SOURCE)
        or backend_manifest["files"]["granularity_mrr_comparisons.csv"]
        != sha256(GRANULARITY_COMPARISON_SOURCE)
    ):
        raise RuntimeError("Granularity-projection artifacts have drifted.")
    comparisons = pd.read_csv(GRANULARITY_COMPARISON_SOURCE)
    comparisons = comparisons.loc[
        comparisons["budget"].isin((64, 512))
        & comparisons["arm"].isin(("R", "D", "I+R+D"))
        & comparisons["comparison"].isin(
            ("Token − Q/A", "Token − Turn")
        )
    ].copy()
    expected = {
        (budget, arm, comparison)
        for budget in (64, 512)
        for arm in ("R", "D", "I+R+D")
        for comparison in ("Token − Q/A", "Token − Turn")
    }
    observed = set(
        comparisons[["budget", "arm", "comparison"]].itertuples(
            index=False,
            name=None,
        )
    )
    if observed != expected or comparisons.duplicated(
        ["budget", "arm", "comparison"]
    ).any():
        raise RuntimeError("Incomplete B64/B512 granularity comparisons.")
    if not comparisons["n"].eq(2_514).all() or not comparisons[
        "replicates"
    ].eq(10_000).all():
        raise RuntimeError("Invalid granularity-comparison population.")
    if not (
        comparisons["ci95_low"].le(comparisons["delta_mrr"]).all()
        and comparisons["ci95_high"].ge(comparisons["delta_mrr"]).all()
    ):
        raise RuntimeError("Invalid granularity-comparison intervals.")

    lengths = pd.read_csv(GRANULARITY_LENGTH_SOURCE)
    lengths = lengths.loc[
        lengths["budget"].isin((64, 512))
        & lengths["granularity"].isin(("token", "qa", "turn"))
    ].copy()
    expected_lengths = {
        (budget, granularity)
        for budget in (64, 512)
        for granularity in ("token", "qa", "turn")
    }
    observed_lengths = set(
        lengths[["budget", "granularity"]].itertuples(
            index=False,
            name=None,
        )
    )
    if observed_lengths != expected_lengths or lengths.duplicated(
        ["budget", "granularity"]
    ).any():
        raise RuntimeError("Incomplete B64/B512 granularity token lengths.")
    if not lengths["n_R"].eq(2_514).all():
        raise RuntimeError("Invalid granularity token-length population.")

    bars = (
        pd.read_csv(GRANULARITY_BAR_CI_SOURCE)
        if GRANULARITY_BAR_CI_SOURCE.is_file()
        else bootstrap_granularity_bar_intervals()
    )
    expected_bars = {
        (budget, arm, granularity)
        for budget in (64, 512)
        for arm in ("R", "D", "I+R+D")
        for granularity in ("token", "qa", "turn")
    }
    observed_bars = set(
        bars[["budget", "arm", "granularity"]].itertuples(
            index=False,
            name=None,
        )
    )
    expected_seed = int.from_bytes(
        hashlib.sha256(
            (
                "granularity_bar_ci_v1\0"
                f"{sha256(GRANULARITY_METRICS_SOURCE)}"
            ).encode("utf-8")
        ).digest()[:8],
        "big",
    )
    cache_invalid = (
        observed_bars != expected_bars
        or bars.duplicated(["budget", "arm", "granularity"]).any()
        or not bars["n"].eq(2_514).all()
        or not bars["bootstrap_replicates"].eq(10_000).all()
        or not bars["bootstrap_seed"].eq(expected_seed).all()
    )
    if cache_invalid:
        bars = bootstrap_granularity_bar_intervals()
        observed_bars = set(
            bars[["budget", "arm", "granularity"]].itertuples(
                index=False,
                name=None,
            )
        )
    if (
        observed_bars != expected_bars
        or bars.duplicated(["budget", "arm", "granularity"]).any()
        or not bars["n"].eq(2_514).all()
        or not bars["bootstrap_replicates"].eq(10_000).all()
        or not bars["bootstrap_seed"].eq(expected_seed).all()
    ):
        raise RuntimeError("Invalid cached granularity bar intervals.")
    summary = pd.read_csv(GRANULARITY_SUMMARY_SOURCE).set_index(
        ["budget", "arm", "granularity"]
    )
    for row in bars.itertuples(index=False):
        expected_mrr = float(
            summary.loc[(row.budget, row.arm, row.granularity), "MRR"]
        )
        if not np.isclose(float(row.MRR), expected_mrr, atol=1e-12):
            raise RuntimeError("Granularity bar MRR has drifted.")
    return bars, lengths


def bootstrap_granularity_baseline_comparisons() -> pd.DataFrame:
    """Compare each compact-panel bar with the paired I-B512 baseline."""
    metrics = pd.read_csv(
        GRANULARITY_METRICS_SOURCE,
        usecols=["granularity", "budget", "sample_id", "arm", "MRR"],
    )
    baseline = metrics.loc[
        metrics["granularity"].eq("token")
        & metrics["budget"].eq(512)
        & metrics["arm"].eq("I")
    ].sort_values("sample_id", kind="mergesort")
    sample_ids = baseline["sample_id"].astype(str).to_numpy()
    if len(sample_ids) != 2_514 or pd.Series(sample_ids).duplicated().any():
        raise RuntimeError("Invalid I-B512 granularity baseline population.")
    baseline_values = baseline["MRR"].to_numpy(dtype=float)
    definitions = (
        ("I-B64", "token", 64, "I"),
        ("v1 Turn", "turn", 64, "R"),
        ("v2 Q/A", "qa", 64, "R"),
        ("v3 Token/span", "token", 64, "R"),
    )
    deltas = {}
    for label, granularity, budget, arm in definitions:
        candidate = metrics.loc[
            metrics["granularity"].eq(granularity)
            & metrics["budget"].eq(budget)
            & metrics["arm"].eq(arm)
        ].sort_values("sample_id", kind="mergesort")
        if not np.array_equal(
            sample_ids,
            candidate["sample_id"].astype(str).to_numpy(),
        ):
            raise RuntimeError(f"Unpaired granularity population for {label}.")
        deltas[label] = candidate["MRR"].to_numpy(dtype=float) - baseline_values

    source_key = sha256(GRANULARITY_METRICS_SOURCE)
    seed = int.from_bytes(
        hashlib.sha256(
            f"granularity_vs_i512_mrr_v1\0{source_key}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    draws = {
        label: np.empty(BOOTSTRAP_REPLICATES, dtype=float) for label in deltas
    }
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, 2_514, size=(stop - start, 2_514))
        for label, values in deltas.items():
            draws[label][start:stop] = values[indices].mean(axis=1)
    rows = []
    for label, values in deltas.items():
        low, high = np.quantile(draws[label], (0.025, 0.975))
        rows.append(
            {
                "comparison": f"{label}_minus_I-B512",
                "candidate": label,
                "reference": "I-B512",
                "n": 2_514,
                "delta_mrr": float(values.mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_seed": seed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "source_key": source_key,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(GRANULARITY_BASELINE_COMPARISON_SOURCE, index=False)
    return result


def bootstrap_granularity_length_intervals(
    lengths: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap paired R-B64 minus I-B512 input-length means."""
    from experiments.rocc import (
        add_conversation_columns,
        build_topiocqa_conversation_samples,
        expand_selected_histories,
        load_itercqr_tokenizer,
        load_topiocqa_frame,
        read_selected_histories_jsonl,
        resolve_topiocqa_resources,
        serialize_itercqr_input,
    )

    projection_manifest = json.loads(
        (GRANULARITY_PROJECTION_DIR / "manifest.json").read_text()
    )
    serializer_sha256 = hashlib.sha256(
        inspect.getsource(serialize_itercqr_input).encode("utf-8")
    ).hexdigest()
    projection_sha256 = hashlib.sha256(
        inspect.getsource(expand_selected_histories).encode("utf-8")
    ).hexdigest()
    if (
        serializer_sha256 != projection_manifest["serializer_sha256"]
        or projection_sha256
        != projection_manifest["projection_function_sha256"]
        or sha256(PRETRAINED_VITERBI_SOURCE)
        != projection_manifest["selector_viterbi_sha256"]
    ):
        raise RuntimeError("Granularity length reconstruction has drifted.")

    resources = resolve_topiocqa_resources(TOPIOCQA_DATA_DIR)
    frame = add_conversation_columns(
        load_topiocqa_frame(resources, splits=("dev",))
    )
    samples = build_topiocqa_conversation_samples(
        frame,
        minimum_history_depth=0,
        progress=False,
    )
    rows = [
        {
            "sample_id": str(sample.sample_id),
            "history": [
                {
                    "turn_id": int(turn.turn_id),
                    "question": str(turn.question),
                    "answer": str(turn.answer),
                }
                for turn in sample.history
            ],
        }
        for sample in samples
    ]
    selected = read_selected_histories_jsonl(
        PRETRAINED_VITERBI_SOURCE,
        progress=False,
    )
    projected = {
        granularity: expand_selected_histories(
            rows,
            selected,
            granularity=granularity,
        )
        for granularity in ("qa", "turn")
    }
    tokenizer = load_itercqr_tokenizer(ITERCQR_MODEL_DIR)

    main_inputs = pd.read_csv(
        TOPIOCQA_CONTEXT_SOURCE,
        usecols=["sample_id", "budget", "arm", "input_length"],
    )
    main_inputs["sample_id"] = main_inputs["sample_id"].astype(str)
    i_b512 = (
        main_inputs.loc[
            main_inputs["budget"].eq(512)
            & main_inputs["arm"].eq("I"),
            ["sample_id", "input_length"],
        ]
        .set_index("sample_id")["input_length"]
        .sort_index()
    )
    r_lengths = {
        "token": (
            main_inputs.loc[
                main_inputs["budget"].eq(64)
                & main_inputs["arm"].eq("pretrained_R"),
                ["sample_id", "input_length"],
            ]
            .set_index("sample_id")["input_length"]
            .sort_index()
        )
    }
    for granularity in ("qa", "turn"):
        r_lengths[granularity] = pd.Series(
            {
                str(sample.sample_id): len(
                    serialize_itercqr_input(
                        tokenizer,
                        sample,
                        projected[granularity][str(sample.sample_id)],
                        budget=64,
                    )[0]
                )
                for sample in samples
            },
            dtype=float,
        ).sort_index()

    if (
        len(i_b512) != 2_514
        or i_b512.index.has_duplicates
        or any(
            len(values) != 2_514
            or values.index.has_duplicates
            or not values.index.equals(i_b512.index)
            for values in r_lengths.values()
        )
    ):
        raise RuntimeError("Incomplete granularity length population.")

    deltas = {
        granularity: values.to_numpy(dtype=float)
        - i_b512.to_numpy(dtype=float)
        for granularity, values in r_lengths.items()
    }
    length_lookup = lengths.set_index(["budget", "granularity"])
    for granularity, values in r_lengths.items():
        expected_i = float(
            length_lookup.loc[
                (512, granularity), "mean_I_t5_input_tokens"
            ]
        )
        expected_r = float(
            length_lookup.loc[
                (64, granularity), "mean_R_t5_input_tokens"
            ]
        )
        if not (
            np.isclose(i_b512.mean(), expected_i, atol=1e-12)
            and np.isclose(values.mean(), expected_r, atol=1e-12)
        ):
            raise RuntimeError("Granularity length means have drifted.")

    source_key = hashlib.sha256(
        "\0".join(
            (
                sha256(TOPIOCQA_CONTEXT_SOURCE),
                sha256(PRETRAINED_VITERBI_SOURCE),
                sha256(GRANULARITY_LENGTH_SOURCE),
            )
        ).encode("utf-8")
    ).hexdigest()
    seed = int.from_bytes(
        hashlib.sha256(
            f"granularity_length_ci_v1\0{source_key}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    draws = {
        granularity: np.empty(BOOTSTRAP_REPLICATES)
        for granularity in deltas
    }
    baseline_draws = np.empty(BOOTSTRAP_REPLICATES)
    baseline_values = i_b512.to_numpy(dtype=float)
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK_SIZE):
        stop = min(start + BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(i_b512), size=(stop - start, len(i_b512)))
        baseline_draws[start:stop] = baseline_values[indices].mean(axis=1)
        for granularity, values in deltas.items():
            draws[granularity][start:stop] = values[indices].mean(axis=1)

    result_rows = []
    for granularity, _, _, _ in GRANULARITY_LENGTH_STYLES:
        low, high = np.quantile(draws[granularity], (0.025, 0.975))
        result_rows.append(
            {
                "granularity": granularity,
                "n": len(i_b512),
                "mean_delta_t5_tokens": float(deltas[granularity].mean()),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_seed": seed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "source_key": source_key,
            }
        )
    baseline_low, baseline_high = np.quantile(
        baseline_draws, (0.025, 0.975)
    )
    baseline_result = pd.DataFrame(
        [
            {
                "system": "I-B512",
                "n": len(i_b512),
                "mean_t5_input_tokens": float(baseline_values.mean()),
                "ci95_low": float(baseline_low),
                "ci95_high": float(baseline_high),
                "bootstrap_seed": seed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "source_key": source_key,
            }
        ]
    )
    return pd.DataFrame(result_rows), baseline_result


def signed_three(value: float) -> str:
    return f"{value:+.3f}".replace("-", "−")


def resize_legend_swatches(legend, fontsize: float) -> None:
    """Match legend swatches to the final, standardized text size."""
    text_height = TextPath((0, 0), "Ag", size=fontsize).get_extents().height
    swatch_height = float(text_height)
    swatch_width = 0.78 * swatch_height
    descent = 0.18 * swatch_height
    for column in legend._legend_handle_box.get_children():
        for item in column.get_children():
            handle_box = item.get_children()[0]
            handle_box.width = swatch_width
            handle_box.height = swatch_height
            handle_box.ydescent = descent
            for artist in handle_box.get_children():
                if isinstance(artist, Rectangle):
                    artist.set_x(0.0)
                    artist.set_y(-descent)
                    artist.set_width(swatch_width)
                    artist.set_height(swatch_height)


def standardize_figure_typography(
    figure,
    *,
    scale: float = 1.0,
    panel_titles_as_body: bool = False,
) -> None:
    """Apply one exact, role-based font-size contract to every figure."""
    if getattr(figure, "_rocc_typography_standardized", False):
        return
    if scale <= 0.0:
        raise ValueError("Typography scale must be positive.")

    page_title_size = FONT_PAGE_TITLE * scale
    panel_title_size = FONT_PANEL_TITLE * scale
    axis_size = FONT_AXIS * scale
    tick_size = FONT_TICK * scale
    body_size = FONT_BODY * scale
    all_text = figure.findobj(
        match=lambda artist: isinstance(artist, matplotlib.text.Text)
    )
    for text in all_text:
        text.set_fontsize(body_size)

    for axis in figure.axes:
        for title in (axis.title, axis._left_title, axis._right_title):
            title.set_fontsize(
                body_size if panel_titles_as_body else panel_title_size
            )
        axis.xaxis.label.set_fontsize(axis_size)
        axis.yaxis.label.set_fontsize(axis_size)
        axis.xaxis.offsetText.set_fontsize(tick_size)
        axis.yaxis.offsetText.set_fontsize(tick_size)
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontsize(tick_size)
        legend = axis.get_legend()
        if legend is not None:
            legend.get_title().set_fontsize(body_size)
            for text in legend.get_texts():
                text.set_fontsize(body_size)
            resize_legend_swatches(legend, body_size)

    suptitle = getattr(figure, "_suptitle", None)
    if suptitle is not None:
        suptitle.set_fontsize(page_title_size)
    for text in figure.texts:
        if text is suptitle:
            continue
        weight = str(text.get_fontweight()).lower()
        if text.get_position()[1] >= 0.97 and weight in {"bold", "700"}:
            text.set_fontsize(page_title_size)
        else:
            text.set_fontsize(body_size)

    allowed = {
        page_title_size,
        panel_title_size,
        axis_size,
        tick_size,
        body_size,
    }
    observed = {float(text.get_fontsize()) for text in all_text}
    if not observed.issubset(allowed):
        raise RuntimeError(
            f"Unstandardized figure font sizes remain: {sorted(observed)}"
        )

    # The larger, uniform typography needs matching vertical headroom.  Keep
    # the highest data-space annotation below the top 20% of every bar axis;
    # legends, titles and the preference hint then retain their own band.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for axis in figure.axes:
        if getattr(
            axis,
            "_rocc_skip_annotation_ylim_adjustment",
            False,
        ):
            continue
        if not any(
            isinstance(container, matplotlib.container.BarContainer)
            for container in axis.containers
        ):
            continue
        annotation_tops = []
        for text in axis.texts:
            if not text.get_visible() or not text.get_text().strip():
                continue
            if text.get_transform() == axis.transAxes:
                continue
            bounds = text.get_window_extent(renderer=renderer)
            if not np.isfinite(bounds.y1):
                continue
            annotation_tops.append(
                float(
                    axis.transData.inverted().transform(
                        (axis.bbox.x0, bounds.y1)
                    )[1]
                )
            )
        if not annotation_tops:
            continue
        lower, upper = axis.get_ylim()
        if upper <= lower:
            continue
        maximum_axes_y = float(
            getattr(
                axis,
                "_rocc_annotation_max_axes_y",
                BAR_ANNOTATION_MAX_AXES_Y,
            )
        )
        required_upper = lower + (
            (max(annotation_tops) - lower) / maximum_axes_y
        )
        if required_upper > upper:
            axis.set_ylim(lower, required_upper)

    figure._rocc_typography_standardized = True
    figure._rocc_typography_scale = scale


def enlarge_recall_outer_typography(figure) -> None:
    """Enlarge only labels outside the Recall panel interiors."""
    scale = STRUCTURE_FIGURE_FONT_SCALE
    for axis in figure.axes:
        for title in (axis.title, axis._left_title, axis._right_title):
            title.set_fontsize(FONT_BODY * scale)
        axis.xaxis.label.set_fontsize(FONT_AXIS * scale)
        axis.yaxis.label.set_fontsize(FONT_AXIS * scale)
        axis.xaxis.offsetText.set_fontsize(FONT_TICK * scale)
        axis.yaxis.offsetText.set_fontsize(FONT_TICK * scale)
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontsize(FONT_TICK * scale)
    for text in figure.texts:
        text.set_fontsize(FONT_BODY * scale)


def draw_context_length_panel(
    axis,
    values_by_series: list[np.ndarray],
    title: str,
    r_b128_minus_b64: dict[str, tuple[float, float, float]],
    *,
    show_ylabel: bool = True,
) -> None:
    bins = np.arange(0.5, 513.5, 1.0)
    rocc_values = values_by_series[-1]
    rocc_counts, _ = np.histogram(rocc_values, bins=bins)
    rocc_tail_counts = rocc_counts.copy()
    rocc_tail_counts[:64] = 0
    rocc_tail_n = int((rocc_values > 64).sum())

    def draw_histogram(target, *, labels: bool) -> None:
        target.stairs(
            rocc_tail_counts,
            bins,
            baseline=0,
            fill=True,
            color=CONTEXT_LENGTH_STYLES[-1][1],
            alpha=0.20,
            zorder=1,
        )
        for (label, color), values in zip(
            CONTEXT_LENGTH_STYLES,
            values_by_series,
            strict=True,
        ):
            target.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=1.2,
                color=color,
                label=(
                    f"{label}  (mean {values.mean():.1f})"
                    if labels
                    else None
                ),
            )

    draw_histogram(axis, labels=True)
    axis.set_xlim(0, 520)
    axis.set_ylim(bottom=0)
    axis.set_xticks([0, 64, 128, 256, 384, 512])
    axis.grid(axis="y", color="#d9dee5", linewidth=0.5, alpha=0.75)
    axis.set_axisbelow(True)
    axis.set_xlabel("Actual serialized T5 input length (tokens)")
    axis.set_ylabel("#queries" if show_ylabel else "", labelpad=2)
    axis.tick_params(axis="both", labelsize=10)
    axis.set_title(title, loc="left", fontsize=10, pad=6)
    axis.legend(
        handles=[
            Patch(
                facecolor=color,
                edgecolor="none",
                label=f"{label}  (mean {values.mean():.1f})",
            )
            for (label, color), values in zip(
                CONTEXT_LENGTH_STYLES,
                values_by_series,
                strict=True,
            )
        ],
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#d0d5dd",
        framealpha=1,
        fontsize=8,
        borderaxespad=0.45,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
    )

    zoom = axis.inset_axes([0.20, 0.19, 0.77, 0.54])
    draw_histogram(zoom, labels=False)
    counts = [np.bincount(values, minlength=513) for values in values_by_series]
    counts[1][64] = 0
    low_frequency_max = max(int(counts_.max()) for counts_ in counts)
    zoom_upper = 10 * np.ceil(1.15 * low_frequency_max / 10)
    axis.add_patch(
        Rectangle(
            (0, 0),
            520,
            zoom_upper,
            facecolor="#f7f4f8",
            edgecolor="#b8a9bf",
            linewidth=0.7,
            alpha=0.55,
            zorder=0.5,
        )
    )
    zoom.set_xlim(0, 520)
    zoom.set_ylim(0, zoom_upper)
    zoom.set_xticks([0, 64, 128, 256, 384, 512])
    zoom.tick_params(axis="both", labelsize=8)
    zoom.grid(axis="y", color="#d9dee5", linewidth=0.45, alpha=0.75)
    zoom.set_axisbelow(True)
    zoom.set_facecolor("#f7f4f8")
    for spine in zoom.spines.values():
        spine.set_color("#b8a9bf")
        spine.set_linewidth(0.8)
    tail_peak_index = 64 + int(np.argmax(rocc_tail_counts[64:]))
    tail_peak_x = tail_peak_index + 1
    tail_peak_y = rocc_tail_counts[tail_peak_index]
    quality_lines = ["R: B128−B64 ΔMRR (paired 95% CI)"]
    for backend in ("BM25", "ANCE"):
        delta, low, high = r_b128_minus_b64[backend]
        quality_lines.append(
            f"{backend}: {signed_three(delta)} "
            f"[{signed_three(low)}, {signed_three(high)}]"
        )
    zoom.annotate(
        (
            "ROCC tail > B64:\n"
            f"n={rocc_tail_n:,} queries "
            f"({rocc_tail_n / len(rocc_values):.1%})\n"
            + "\n".join(quality_lines)
        ),
        xy=(tail_peak_x + 8, 0.07 * tail_peak_y),
        xycoords="data",
        xytext=(tail_peak_x + 45, max(0.58 * zoom_upper, tail_peak_y)),
        textcoords="data",
        ha="left",
        va="center",
        fontsize=7.8,
        color="#000000",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "#cfc4d3",
            "linewidth": 0.6,
            "alpha": 0.90,
        },
        arrowprops={
            "arrowstyle": "-",
            "color": "#76577f",
            "linewidth": 1.0,
            "relpos": (0.0, 0.0),
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=8,
    )
    for x in (0, 520):
        axis.add_artist(
            ConnectionPatch(
                xyA=(x, zoom_upper),
                coordsA=axis.transData,
                xyB=(x, 0),
                coordsB=zoom.transData,
                color="#b8a9bf",
                linewidth=0.65,
                alpha=0.9,
                clip_on=False,
                zorder=4,
            )
        )


def draw_primary_quality_compact(
    axis,
    quality: pd.DataFrame,
    granularity_bars: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    frame = quality.loc[quality["backend"].eq("BM25")].set_index(
        ["budget", "route"]
    )
    width = 0.20
    bar_groups, _, family_spans = grouped_bar_layout((2, 3), width)
    iter_label_clearance = width * 0.12
    bar_groups = (
        bar_groups[0]
        + np.asarray((-iter_label_clearance, iter_label_clearance)),
        bar_groups[1],
    )
    family_spans = (
        (
            family_spans[0][0] - iter_label_clearance,
            family_spans[0][1] + iter_label_clearance,
        ),
        family_spans[1],
    )
    centers = np.asarray(
        (
            bar_groups[0][0],
            bar_groups[0][1],
            float(bar_groups[1].mean()),
        )
    )
    labels = ("I-B512", "I-B64", "R-B64")
    singleton_bars = (
        (centers[0], 512, "Baseline"),
        (centers[1], 64, None),
    )
    baseline = float(frame.loc[(512, "I"), "MRR"])
    comparison_lookup = comparisons.set_index("candidate")
    axis.axhline(
        baseline,
        color="#667085",
        linewidth=0.75,
        linestyle=(0, (3, 2)),
        zorder=1,
    )

    def draw_bar(
        x,
        value,
        low,
        high,
        width,
        color,
        label,
        annotation_y,
        comparison_label=None,
    ):
        axis.bar(
            x,
            value,
            width,
            color=color,
            edgecolor="none",
            linewidth=0,
            label=label,
            zorder=2,
        )
        add_bootstrap_errorbars(
            axis,
            np.asarray([x]),
            np.asarray([value]),
            np.asarray([low]),
            np.asarray([high]),
            fill_color=color,
        )
        axis.annotate(
            f"{value:.3f}",
            xy=(x, high + 0.002),
            xytext=(x, annotation_y),
            ha="center",
            va="bottom",
            fontsize=8.5,
            arrowprops={
                "arrowstyle": "-",
                "color": "#69717c",
                "linewidth": 0.7,
                "shrinkA": 3,
                "shrinkB": 1,
            },
            annotation_clip=False,
        )
        if comparison_label is not None:
            comparison = comparison_lookup.loc[comparison_label]
            add_significance_marker(
                axis,
                x=float(x),
                annotation_y=float(annotation_y),
                ci_low=float(comparison["ci95_low"]),
                ci_high=float(comparison["ci95_high"]),
            )

    for position, budget, label in singleton_bars:
        row = frame.loc[(budget, "I")]
        value = float(row["MRR"])
        low = float(row["MRR_ci95_low"])
        high = float(row["MRR_ci95_high"])
        draw_bar(
            position,
            value,
            low,
            high,
            width,
            "#9099a3",
            label,
            0.205 if budget == 512 else 0.190,
            "I-B64" if budget == 64 else None,
        )

    b64 = granularity_bars.loc[granularity_bars["budget"].eq(64)]
    annotation_y = {"turn": 0.195, "qa": 0.225, "token": 0.255}
    for index, (granularity, legend_label, color, _) in enumerate(
        GRANULARITY_STYLES
    ):
        row = b64.loc[
            b64["arm"].eq("R")
            & b64["granularity"].eq(granularity)
        ].iloc[0]
        draw_bar(
            bar_groups[1][index],
            float(row["MRR"]),
            float(row["ci95_low"]),
            float(row["ci95_high"]),
            width,
            color,
            legend_label,
            annotation_y[granularity],
            legend_label,
        )

    bracket_y = 0.285
    family_groups = (
        ("IterCQR", "#9099a3"),
        ("ROCC B64", "#7098b8"),
    )
    for (left, right), (family, background) in zip(
        family_spans,
        family_groups,
        strict=True,
    ):
        axis.fill(
            [left, right, right, left],
            [0.0, 0.0, bracket_y, bracket_y],
            facecolor=background,
            edgecolor="none",
            alpha=0.075,
            zorder=0,
        )
        axis.plot(
            [left, left, right, right],
            [bracket_y - 0.006, bracket_y, bracket_y, bracket_y - 0.006],
            color="#475467",
            linewidth=0.9,
            clip_on=False,
        )
        axis.text(
            (left + right) / 2,
            bracket_y + 0.005,
            family,
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#475467",
            clip_on=False,
        )

    axis.set_title(
        "TopiOCQA Dev, BM25, n=2,514",
        loc="left",
        fontsize=10,
        pad=6,
    )
    axis.set_ylabel("MRR", fontsize=10)
    axis.set_xlim(
        family_spans[0][0] - width * 0.55,
        family_spans[-1][1] + width * 0.55,
    )
    axis.set_ylim(0.0, 0.380)
    axis.set_yticks(np.arange(0.0, 0.301, 0.05))
    axis.set_xticks(centers, labels)
    axis.grid(False)
    axis._rocc_annotation_max_axes_y = 0.68
    add_preference_hint(
        axis,
        higher_is_better=True,
        axes_y=0.81,
    )
    axis.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#d0d5dd",
        framealpha=1,
        fontsize=7.2,
        ncol=2,
        loc="upper left",
        borderaxespad=0.35,
        columnspacing=0.5,
        handlelength=1.0,
        handletextpad=0.25,
    )
    axis.tick_params(axis="both", labelsize=10)


def draw_granularity_length_panel(
    axis,
    lengths: pd.DataFrame,
    intervals: pd.DataFrame,
    baseline_interval: pd.DataFrame,
) -> None:
    width = 0.21
    position = 0.0
    bar_groups, _, group_spans = grouped_bar_layout((3,), width)
    lookup = lengths.set_index(["budget", "granularity"])
    interval_lookup = intervals.set_index("granularity")
    i_b512_values = lengths.loc[
        lengths["budget"].eq(512),
        "mean_I_t5_input_tokens",
    ].unique()
    if len(i_b512_values) != 1:
        raise RuntimeError("Inconsistent I-B512 input-length reference.")
    i_b512 = float(i_b512_values[0])
    if len(baseline_interval) != 1:
        raise RuntimeError("Invalid I-B512 baseline interval.")
    baseline_row = baseline_interval.iloc[0]
    baseline_low = float(baseline_row["ci95_low"])
    baseline_high = float(baseline_row["ci95_high"])
    if not (
        np.isclose(
            float(baseline_row["mean_t5_input_tokens"]),
            i_b512,
            atol=1e-12,
        )
        and baseline_low <= i_b512 <= baseline_high
    ):
        raise RuntimeError("I-B512 baseline interval has drifted.")
    axis.axhspan(
        baseline_low,
        baseline_high,
        facecolor="#667085",
        edgecolor="none",
        alpha=0.14,
        zorder=0.5,
    )
    for index, (granularity, label, color, _) in enumerate(
        GRANULARITY_LENGTH_STYLES
    ):
        row = lookup.loc[(64, granularity)]
        r_b64 = float(row["mean_R_t5_input_tokens"])
        value = r_b64 - i_b512
        percent = 100.0 * value / i_b512
        low = float(interval_lookup.loc[granularity, "ci95_low"])
        high = float(interval_lookup.loc[granularity, "ci95_high"])
        if not (
            np.isclose(
                value,
                float(
                    interval_lookup.loc[
                        granularity, "mean_delta_t5_tokens"
                    ]
                ),
                atol=1e-12,
            )
            and low <= value <= high
        ):
            raise RuntimeError("Invalid granularity length interval.")
        x = float(bar_groups[0][index])
        axis.bar(
            x,
            value,
            width,
            bottom=i_b512,
            color=color,
            edgecolor="none",
            linewidth=0,
            label=label,
            zorder=2,
        )
        add_bootstrap_errorbars(
            axis,
            np.asarray([x]),
            np.asarray([i_b512 + value]),
            np.asarray([i_b512 + low]),
            np.asarray([i_b512 + high]),
            fill_color=color,
            bar_origin=i_b512,
        )
        label_y = {
            "turn": i_b512 - 110.0,
            "qa": i_b512 - 145.0,
            "token": i_b512 - 180.0,
        }[
            granularity
        ]
        axis.annotate(
            f"{value:.1f} tokens\nΔ {percent:.1f}%",
            xy=(x, i_b512 + low - 1.0),
            xytext=(x, label_y),
            ha="center",
            va="top",
            fontsize=8.5,
            arrowprops={
                "arrowstyle": "-",
                "color": "#69717c",
                "linewidth": 0.7,
                "shrinkA": 3,
                "shrinkB": 1,
            },
            annotation_clip=False,
        )
        add_significance_marker(
            axis,
            x=x,
            annotation_y=label_y,
            ci_low=low,
            ci_high=high,
            va="top",
            x_offset_points=(-38 if granularity == "turn" else -46),
        )
    axis.set_title(
        "TopiOCQA Dev, n=2,514",
        loc="left",
        fontsize=10,
        pad=6,
    )
    axis.set_ylabel(
        "Mean T5 input tokens/query",
        fontsize=10,
        rotation=90,
        labelpad=8,
    )
    axis.yaxis.set_label_position("left")
    axis.set_xlim(
        group_spans[0][0] - width * 0.55,
        group_spans[0][1] + width * 0.55,
    )
    axis.set_ylim(-80.0, 225.0)
    absolute_ticks = (0.0, 30.0, 60.0, 90.0, 120.0, i_b512, 180.0, 210.0)
    axis.set_yticks(
        absolute_ticks,
        [
            f"{tick:.1f}" if np.isclose(tick, i_b512) else f"{tick:.0f}"
            for tick in absolute_ticks
        ],
    )
    axis.set_xticks((position,), ("R-B64 − I-B512",))
    axis.axhline(i_b512, color="#475467", linewidth=0.8)
    axis.text(
        0.0,
        baseline_high + 3.0,
        f"I-B512 95% CI [{baseline_low:.1f}, {baseline_high:.1f}]",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#566273",
    )
    axis.grid(False)
    axis._rocc_annotation_max_axes_y = 0.68
    add_preference_hint(
        axis,
        higher_is_better=False,
        axes_y=0.81,
    )
    handles, legend_labels = axis.get_legend_handles_labels()
    legend_order = (0, 2, 1)
    axis.legend(
        [handles[index] for index in legend_order],
        [legend_labels[index] for index in legend_order],
        frameon=True,
        facecolor="white",
        edgecolor="#d0d5dd",
        framealpha=1,
        fontsize=7.2,
        ncol=2,
        loc="upper left",
        borderaxespad=0.35,
        columnspacing=0.5,
        handlelength=1.0,
        handletextpad=0.25,
    )
    axis.tick_params(axis="both", labelsize=10)


def draw_dqcis_fusion_ablation(
    axis,
    intervals: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    systems = ("DQ", "DQ+I", "DQ+R")
    labels = ("DQ-CIS", "DQ-CIS + I", "DQ-CIS + R")
    colors = ("#9099a3", "#adc6da", "#315f88")
    width = 0.40
    bar_groups, _, family_spans = grouped_bar_layout((1, 2), width)
    positions = np.concatenate(bar_groups)
    lookup = intervals.set_index("system")
    values = np.asarray([lookup.loc[system, "MRR"] for system in systems])
    lows = np.asarray(
        [lookup.loc[system, "ci95_low"] for system in systems]
    )
    highs = np.asarray(
        [lookup.loc[system, "ci95_high"] for system in systems]
    )
    label_y = float(highs.max()) + 0.020
    family_y = float(highs.max()) + 0.065
    comparison_y = float(highs.max()) + 0.120

    family_groups = (
        ("Single ranking", "#9099a3"),
        ("RRF10 fusion · two rankings", "#7098b8"),
    )
    for (left, right), (family, background) in zip(
        family_spans,
        family_groups,
        strict=True,
    ):
        axis.fill(
            [left, right, right, left],
            [0.0, 0.0, family_y, family_y],
            facecolor=background,
            edgecolor="none",
            alpha=0.075,
            zorder=0,
        )
        axis.plot(
            [left, left, right, right],
            [family_y - 0.007, family_y, family_y, family_y - 0.007],
            color="#475467",
            linewidth=0.9,
            clip_on=False,
        )
        axis.text(
            (left + right) / 2,
            family_y + 0.006,
            family,
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color="#475467",
            clip_on=False,
        )

    axis.axhline(
        float(values[0]),
        color="#667085",
        linestyle=(0, (2.5, 2.5)),
        linewidth=0.8,
        alpha=0.55,
        zorder=0.5,
    )

    for position, system, color, value, low, high in zip(
        positions,
        systems,
        colors,
        values,
        lows,
        highs,
        strict=True,
    ):
        axis.bar(
            position,
            value,
            width,
            color=color,
            edgecolor="none",
            linewidth=0,
            zorder=2,
        )
        add_bootstrap_errorbars(
            axis,
            np.asarray([position]),
            np.asarray([value]),
            np.asarray([low]),
            np.asarray([high]),
            fill_color=color,
        )
        axis.annotate(
            f"{value:.3f}",
            xy=(position, high + 0.003),
            xytext=(position, label_y),
            ha="center",
            va="bottom",
            fontsize=8.5,
            arrowprops={
                "arrowstyle": "-",
                "color": "#69717c",
                "linewidth": 0.7,
                "shrinkA": 3,
                "shrinkB": 1,
            },
            annotation_clip=False,
        )
        if system != "DQ":
            comparison = comparisons.loc[
                comparisons["comparison"].eq(f"{system}_minus_DQ")
            ].iloc[0]
            add_significance_marker(
                axis,
                x=float(position),
                annotation_y=label_y,
                ci_low=float(comparison["ci95_low"]),
                ci_high=float(comparison["ci95_high"]),
            )

    primary = comparisons.loc[
        comparisons["comparison"].eq("DQ+R_minus_DQ+I")
    ].iloc[0]
    axis.plot(
        [positions[1], positions[1], positions[2], positions[2]],
        [comparison_y - 0.007, comparison_y, comparison_y, comparison_y - 0.007],
        color="#475467",
        linewidth=0.9,
        clip_on=False,
    )
    axis.text(
        float((positions[1] + positions[2]) / 2.0),
        comparison_y + 0.006,
        (
            f"DQ+R − DQ+I: Δ MRR {float(primary['delta_mrr']):+.3f}\n"
            f"95% CI [{float(primary['ci95_low']):+.3f}, "
            f"{float(primary['ci95_high']):+.3f}]"
        ),
        ha="center",
        va="bottom",
        fontsize=7.8,
        color="#344054",
        linespacing=1.05,
        clip_on=False,
    )

    axis.set_title(
        "TopiOCQA Dev, DQ-CIS ChatGPT/ColBERTv2, n=2,514",
        loc="left",
        fontsize=10,
        pad=6,
    )
    axis.set_ylabel("MRR", fontsize=10)
    axis.set_xlim(
        family_spans[0][0] - width * 0.55,
        family_spans[-1][1] + width * 0.55,
    )
    axis.set_ylim(0.0, max(0.50, comparison_y + 0.055))
    axis.set_yticks(np.arange(0.0, 0.451, 0.05))
    axis.set_xticks(positions, labels)
    axis.set_xlabel(
        "I: IterCQR→ANCE · R: ROCC→IterCQR→ANCE\n"
        "DQ+I and DQ+R use unweighted RRF10 over two Top-100 rankings",
        fontsize=8,
    )
    axis.grid(False)
    axis.legend(
        handles=[
            Patch(facecolor=color, edgecolor="none", label=label)
            for label, color in zip(labels, colors, strict=True)
        ],
        frameon=True,
        facecolor="white",
        edgecolor="#d0d5dd",
        framealpha=1,
        fontsize=8,
        ncol=3,
        loc="upper left",
        borderaxespad=0.35,
        columnspacing=LEGEND_COLUMN_SPACING,
        handlelength=LEGEND_HANDLE_LENGTH,
        handleheight=LEGEND_HANDLE_HEIGHT,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
    )
    add_preference_hint(axis, higher_is_better=True)
    axis.tick_params(axis="both", labelsize=10)


def load_teacher_student_analysis() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    manifest = json.loads(
        TEACHER_STUDENT_MANIFEST_SOURCE.read_text(encoding="utf-8")
    )
    identity = manifest.get("identity", {})
    if (
        manifest.get("complete") is not True
        or identity.get("protocol") != "topiocqa_teacher_labeled_dev_v2"
        or int(identity.get("retrieval_queries", -1)) != 2_104
        or int(identity.get("agreement_queries", -1)) != 2_104
        or int(identity.get("no_history_queries_excluded", -1)) != 205
        or int(identity.get("single_history_queries_excluded", -1)) != 205
        or manifest.get("single_history_queries_evaluated") is not False
        or manifest.get("no_history_queries_evaluated") is not False
    ):
        raise RuntimeError("Held-out Teacher-evaluation manifest drifted.")
    population = pd.read_csv(TEACHER_STUDENT_POPULATION_SOURCE)
    if (
        len(population) != 2_104
        or population["sample_id"].duplicated().any()
        or int(population["history_depth"].eq(0).sum()) != 0
        or int(population["history_depth"].eq(1).sum()) != 0
        or int(population["teacher_labeled"].sum()) != 2_104
    ):
        raise RuntimeError("Teacher-evaluable TopiOCQA cohort drifted.")

    mrr = pd.read_csv(TEACHER_STUDENT_MRR_SOURCE)
    expected_cells = {
        (system, budget)
        for system in (
            "gpt54_teacher",
            "student",
            "bm25_teacher",
            "bm25_student",
        )
        for budget in BUDGETS
    }
    if (
        set(mrr[["system", "budget"]].itertuples(index=False, name=None))
        != expected_cells
        or mrr.duplicated(["system", "budget"]).any()
        or not mrr["n"].eq(2_104).all()
        or not mrr["bootstrap_replicates"].eq(10_000).all()
        or not (
            mrr["ci95_low"].le(mrr["MRR"]).all()
            and mrr["ci95_high"].ge(mrr["MRR"]).all()
        )
    ):
        raise RuntimeError("Teacher/Student retrieval summary is incomplete.")

    f1 = pd.read_csv(TEACHER_STUDENT_F1_SOURCE)
    expected_f1 = {
        (system, metric)
        for system in ("student", "bm25_student")
        for metric in ("KEEP F1", "Span F1")
    }
    if (
        set(f1[["system", "metric"]].itertuples(index=False, name=None))
        != expected_f1
        or not f1["n"].eq(2_104).all()
        or not f1["bootstrap_replicates"].eq(10_000).all()
        or not (
            f1["ci95_low"].le(f1["value"]).all()
            and f1["ci95_high"].ge(f1["value"]).all()
        )
    ):
        raise RuntimeError("Teacher/Student agreement summary is incomplete.")
    comparisons = pd.read_csv(TEACHER_STUDENT_COMPARISON_SOURCE)
    if len(comparisons) != 9 or not comparisons["n"].eq(2_104).all():
        raise RuntimeError("Teacher/Student paired comparisons are incomplete.")
    f1_comparisons = bootstrap_teacher_student_f1_comparisons(f1)
    return mrr, f1, comparisons, f1_comparisons


def load_teacher_label_distribution() -> pd.DataFrame:
    """Project the held-out Teacher labels into both trained label spaces."""
    from transformers import AutoTokenizer

    from experiments.rocc.history_selector import (
        ID_TO_LABEL,
        KEEP_LABELS,
        TAXONOMY_ID_TO_LABEL,
        encode_history_pair,
        project_collapsed_selector_row,
        project_taxonomy_selector_row,
    )

    manifest = json.loads(
        TEACHER_STUDENT_MANIFEST_SOURCE.read_text(encoding="utf-8")
    )
    expected_sha = str(manifest["identity"]["teacher_labels_sha256"])
    if sha256(TEACHER_LABELS_SOURCE) != expected_sha:
        raise RuntimeError("Teacher-label source drifted.")
    cohort_ids = set(
        pd.read_csv(TEACHER_STUDENT_POPULATION_SOURCE)["sample_id"].astype(str)
    )
    rows = [
        json.loads(line)
        for line in TEACHER_LABELS_SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if (
        len(rows) != 2_104
        or {str(row["sample_id"]) for row in rows} != cohort_ids
    ):
        raise RuntimeError("Teacher-label distribution cohort drifted.")

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2",
        revision="a50ef00143b4d5391434df20ae11632588ac25be",
        use_fast=True,
        local_files_only=True,
    )
    taxonomy_counts: Counter[str] = Counter()
    collapsed_counts: Counter[str] = Counter()
    for row in rows:
        pair = encode_history_pair(
            row,
            tokenizer,
            max_length=512,
            history_order="recent_first",
        )
        taxonomy = project_taxonomy_selector_row(row, pair)
        collapsed = project_collapsed_selector_row(
            row,
            pair,
            positive_labels=KEEP_LABELS,
        )
        for label, active in zip(
            taxonomy.labels, taxonomy.history_mask, strict=True
        ):
            if active:
                taxonomy_counts[TAXONOMY_ID_TO_LABEL[int(label)]] += 1
        for label, active in zip(
            collapsed.labels, collapsed.history_mask, strict=True
        ):
            if active:
                collapsed_counts[ID_TO_LABEL[int(label)]] += 1

    definitions = (
        (
            "full_taxonomy",
            ("O", "ENTITY", "CONCEPT", "KEY_TERM", "DEFINITION", "RELATION_CUE"),
            taxonomy_counts,
        ),
        (
            "collapsed_bio",
            ("O", "B-KEEP", "I-KEEP"),
            collapsed_counts,
        ),
    )
    result_rows = []
    for label_space, labels, counts in definitions:
        total = int(sum(counts.values()))
        for order, label in enumerate(labels):
            count = int(counts[label])
            result_rows.append(
                {
                    "label_space": label_space,
                    "order": order,
                    "label": label,
                    "count": count,
                    "share": count / total,
                    "queries": 2_104,
                    "history_subword_tokens": total,
                    "tokenizer": "sentence-transformers/all-MiniLM-L12-v2",
                    "tokenizer_revision": (
                        "a50ef00143b4d5391434df20ae11632588ac25be"
                    ),
                    "max_length": 512,
                    "history_order": "recent_first",
                }
            )
    result = pd.DataFrame(result_rows)
    totals = result.groupby("label_space")["history_subword_tokens"].first()
    if (
        totals.nunique() != 1
        or int(totals.iloc[0]) != 393_722
        or int(taxonomy_counts["O"]) != int(collapsed_counts["O"])
        or sum(taxonomy_counts[label] for label in taxonomy_counts if label != "O")
        != int(collapsed_counts["B-KEEP"] + collapsed_counts["I-KEEP"])
    ):
        raise RuntimeError("Teacher-label spaces do not close exactly.")
    result.to_csv(TEACHER_LABEL_DISTRIBUTION_EXPORT, index=False)
    return result


def bootstrap_teacher_student_f1_comparisons(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Pair the two Student systems by query and bootstrap micro-F1 deltas."""
    counts = pd.read_csv(TEACHER_STUDENT_AGREEMENT_COUNTS_SOURCE)
    systems = ("student", "bm25_student")
    frames = {
        system: counts.loc[counts["system"].eq(system)].sort_values(
            "sample_id", kind="mergesort"
        )
        for system in systems
    }
    sample_ids = frames[systems[0]]["sample_id"].astype(str).to_numpy()
    if (
        len(sample_ids) != 2_104
        or pd.Series(sample_ids).duplicated().any()
        or any(len(frame) != 2_104 for frame in frames.values())
        or not np.array_equal(
            sample_ids,
            frames[systems[1]]["sample_id"].astype(str).to_numpy(),
        )
    ):
        raise RuntimeError("Teacher/Student F1 populations are not paired.")

    source_key = sha256(TEACHER_STUDENT_AGREEMENT_COUNTS_SOURCE)
    rows = []
    for metric, prefix in (("KEEP F1", "keep"), ("Span F1", "span")):
        values = {
            system: frames[system][
                [f"{prefix}_tp", f"{prefix}_fp", f"{prefix}_fn"]
            ].to_numpy(dtype=np.int64)
            for system in systems
        }

        def micro_f1(query_counts: np.ndarray) -> np.ndarray:
            totals = query_counts.sum(axis=-2)
            denominator = 2 * totals[..., 0] + totals[..., 1] + totals[..., 2]
            return np.divide(
                2 * totals[..., 0],
                denominator,
                out=np.zeros_like(denominator, dtype=float),
                where=denominator != 0,
            )

        point_values = {
            system: float(micro_f1(values[system])) for system in systems
        }
        for system, value in point_values.items():
            expected = float(
                summary.loc[
                    summary["system"].eq(system)
                    & summary["metric"].eq(metric),
                    "value",
                ].iloc[0]
            )
            if not np.isclose(value, expected, atol=1e-12):
                raise RuntimeError(f"Teacher/Student {metric} drift for {system}.")

        seed = int.from_bytes(
            hashlib.sha256(
                (
                    f"teacher_student_f1_pair_v1\0{metric}\0{source_key}"
                ).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(seed)
        draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
        for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK_SIZE):
            stop = min(start + BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_REPLICATES)
            indices = rng.integers(0, 2_104, size=(stop - start, 2_104))
            draws[start:stop] = (
                micro_f1(values["bm25_student"][indices])
                - micro_f1(values["student"][indices])
            )
        low, high = np.quantile(draws, (0.025, 0.975))
        rows.append(
            {
                "comparison": "bm25_student_minus_student",
                "metric": metric,
                "left_system": "student",
                "right_system": "bm25_student",
                "n": 2_104,
                "delta_f1": (
                    point_values["bm25_student"] - point_values["student"]
                ),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_seed": seed,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TEACHER_STUDENT_F1_COMPARISON_EXPORT, index=False)
    return result


def add_pair_reference_and_significance(
    axis,
    *,
    left_x: float,
    right_x: float,
    left_value: float,
    right_annotation_y: float,
    ci_low: float,
    ci_high: float,
    bar_width: float,
) -> None:
    """Draw a pair-local left-bar reference and its paired-CI marker."""
    axis.plot(
        [left_x - bar_width / 2.0, right_x + bar_width / 2.0],
        [left_value, left_value],
        color="#667085",
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        alpha=0.60,
        zorder=3,
    )
    add_significance_marker(
        axis,
        x=right_x,
        annotation_y=right_annotation_y,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def add_significance_marker(
    axis,
    *,
    x: float,
    annotation_y: float,
    ci_low: float,
    ci_high: float,
    va: str = "bottom",
    x_offset_points: float = -25,
) -> None:
    """Place the document-wide paired-CI marker beside a bar label."""
    significant = ci_low > 0.0 or ci_high < 0.0
    axis.annotate(
        "†" if significant else "‡",
        xy=(x, annotation_y),
        xytext=(x_offset_points, 0),
        textcoords="offset points",
        ha="center",
        va=va,
        fontsize=11,
        fontweight="normal",
        color="#344054",
        annotation_clip=False,
        zorder=6,
    )


def draw_teacher_student_mrr(
    axis,
    mrr: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    width = 0.32
    bar_groups, group_positions, group_spans = grouped_bar_layout(
        (2, 2),
        width,
    )
    styles = (
        ("gpt54_teacher", "Teacher", RECALL_COLORS["R"], bar_groups[0][0]),
        ("student", "Student", RECALL_COLORS["D"], bar_groups[0][1]),
        ("bm25_teacher", None, RECALL_COLORS["R"], bar_groups[1][0]),
        ("bm25_student", None, RECALL_COLORS["D"], bar_groups[1][1]),
    )
    for (left, right), color in zip(
        group_spans,
        (RECALL_COLORS["I"], RECALL_COLORS["D"]),
        strict=True,
    ):
        axis.axvspan(
            left,
            right,
            ymin=0.0,
            ymax=0.82,
            facecolor=color,
            edgecolor="none",
            alpha=0.075,
            zorder=0,
        )
    b64 = mrr.loc[mrr["budget"].eq(64)].set_index("system")
    for system, label, color, position in styles:
        row = b64.loc[system]
        value = float(row["MRR"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        axis.bar(
            position,
            value,
            width,
            color=color,
            edgecolor="none",
            linewidth=0,
            label=label,
            zorder=2,
        )
        add_bootstrap_errorbars(
            axis,
            np.asarray([position]),
            np.asarray([value]),
            np.asarray([low]),
            np.asarray([high]),
            fill_color=color,
        )
        axis.text(
            position,
            high + (0.065 if system in ("student", "bm25_student") else 0.009),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#344054",
        )
    for group_index, (left_system, right_system, comparison_name) in enumerate(
        (
            ("gpt54_teacher", "student", "student_minus_gpt54_teacher"),
            (
                "bm25_teacher",
                "bm25_student",
                "bm25_student_minus_bm25_teacher",
            ),
        )
    ):
        comparison = comparisons.loc[
            comparisons["comparison"].eq(comparison_name)
            & comparisons["budget"].eq(64)
        ].iloc[0]
        add_pair_reference_and_significance(
            axis,
            left_x=float(bar_groups[group_index][0]),
            right_x=float(bar_groups[group_index][1]),
            left_value=float(b64.loc[left_system, "MRR"]),
            right_annotation_y=(
                float(b64.loc[right_system, "ci95_high"])
                + 0.065
            ),
            ci_low=float(comparison["ci95_low"]),
            ci_high=float(comparison["ci95_high"]),
            bar_width=width,
        )
    axis.set_title(
        "TopiOCQA Dev, BM25, n=2,104\nB64 R-route MRR",
        loc="left",
        fontsize=10,
    )
    axis.set_ylabel("MRR", fontsize=10)
    axis.set_xticks(group_positions, ("GPT-5.4", "BM25-corrected"))
    axis.set_xlim(
        group_spans[0][0] - width * 0.55,
        group_spans[-1][1] + width * 0.55,
    )
    axis.set_ylim(0.0, 0.50)
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.legend(
        loc="upper left",
        frameon=True,
        fontsize=7.2,
        ncol=2,
        handlelength=1.1,
        handletextpad=0.35,
        columnspacing=0.65,
    )
    add_preference_hint(axis, higher_is_better=True)
    axis.tick_params(axis="both", labelsize=10)


def draw_teacher_student_f1(
    axis,
    f1: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    metrics = ("KEEP F1", "Span F1")
    systems = ("student", "bm25_student")
    labels = (
        "Student",
        "BM25 Student",
    )
    colors = (RECALL_COLORS["D"], RECALL_COLORS["D"])
    width = 0.32
    bar_groups, positions, group_spans = grouped_bar_layout(
        (len(systems),) * len(metrics),
        width,
    )
    for left, right in group_spans:
        axis.axvspan(
            left,
            right,
            ymin=0.0,
            ymax=0.82,
            facecolor=RECALL_COLORS["D"],
            edgecolor="none",
            alpha=0.075,
            zorder=0,
        )
    for index, (system, label, color) in enumerate(
        zip(systems, labels, colors, strict=True)
    ):
        selected = f1.loc[f1["system"].eq(system)].set_index("metric")
        values = np.asarray([float(selected.loc[metric, "value"]) for metric in metrics])
        lows = np.asarray([float(selected.loc[metric, "ci95_low"]) for metric in metrics])
        highs = np.asarray([float(selected.loc[metric, "ci95_high"]) for metric in metrics])
        centers = np.asarray([group[index] for group in bar_groups])
        axis.bar(
            centers,
            values,
            width,
            color=color,
            edgecolor="none",
            label=label,
            zorder=2,
        )
        add_bootstrap_errorbars(
            axis,
            centers,
            values,
            lows,
            highs,
            color="#475467",
            fill_color=color,
        )
        for center, value, high in zip(centers, values, highs, strict=True):
            axis.text(
                center,
                high + (0.180 if system == "bm25_student" else 0.035),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#344054",
            )
    comparison_lookup = comparisons.set_index("metric")
    student = f1.loc[f1["system"].eq("student")].set_index("metric")
    bm25_student = f1.loc[
        f1["system"].eq("bm25_student")
    ].set_index("metric")
    for metric_index, metric in enumerate(metrics):
        comparison = comparison_lookup.loc[metric]
        add_pair_reference_and_significance(
            axis,
            left_x=float(bar_groups[metric_index][0]),
            right_x=float(bar_groups[metric_index][1]),
            left_value=float(student.loc[metric, "value"]),
            right_annotation_y=(
                float(bm25_student.loc[metric, "ci95_high"]) + 0.180
            ),
            ci_low=float(comparison["ci95_low"]),
            ci_high=float(comparison["ci95_high"]),
            bar_width=width,
        )
    axis.set_title(
        "TopiOCQA Dev, n=2,104\nSelector agreement",
        loc="left",
        fontsize=10,
    )
    axis.set_ylabel("F1", fontsize=10)
    axis.set_xticks(positions, metrics)
    axis.set_xlim(
        group_spans[0][0] - width * 0.55,
        group_spans[-1][1] + width * 0.55,
    )
    axis.set_ylim(0.0, 1.00)
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.legend(loc="upper left", frameon=True, fontsize=6.2)
    add_preference_hint(axis, higher_is_better=True)
    axis.tick_params(axis="both", labelsize=10)


def draw_teacher_label_distribution(
    axis,
    distribution: pd.DataFrame,
    *,
    label_space: str,
    title: str,
) -> None:
    complete = distribution.loc[
        distribution["label_space"].eq(label_space)
    ].sort_values("order").reset_index(drop=True)
    majority = complete.loc[complete["label"].eq("O")].iloc[0]
    selected = complete.loc[complete["label"].ne("O")].copy()
    display_labels = {
        "O": "O / DROP" if label_space == "collapsed_bio" else "O",
        "ENTITY": "Entity",
        "CONCEPT": "Concept",
        "KEY_TERM": "Key term",
        "DEFINITION": "Definition",
        "RELATION_CUE": "Relation cue",
        "B-KEEP": "B-KEEP",
        "I-KEEP": "I-KEEP",
    }
    palette = {
        label: "#9099a3"
        for label in (
            "ENTITY",
            "CONCEPT",
            "KEY_TERM",
            "DEFINITION",
            "RELATION_CUE",
            "B-KEEP",
            "I-KEEP",
        )
    }
    width = 0.52
    majority_width = 0.34 if label_space == "collapsed_bio" else width
    all_positions = np.arange(len(complete), dtype=float)
    axis_left = -0.55
    axis_right = float(len(complete)) - 0.45
    majority_position = axis_left + 0.15 * (axis_right - axis_left)
    majority_value = float(majority["share"]) * 100.0
    axis.bar(
        [majority_position],
        [majority_value],
        majority_width,
        color="#9099a3",
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )
    axis.text(
        majority_position,
        majority_value + 2.2,
        f"{majority_value:.1f}%",
        ha="center",
        va="bottom",
        color="#344054",
        fontsize=8,
    )
    axis.set_xticks(
        [majority_position],
        [display_labels["O"]],
    )
    axis.set_xlim(axis_left, axis_right)
    axis.set_ylim(0.0, 112.0)
    axis.set_ylabel("Tokens (%)")
    axis.set_title(title, loc="left", fontsize=10)
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=10)

    centers = all_positions[1:]
    values = selected["share"].to_numpy(dtype=float) * 100.0
    zoom_upper = float(values.max()) * 1.34
    zoom = axis.inset_axes([0.30, 0.0, 0.70, 0.86], zorder=5)
    zoom.bar(
        centers,
        values,
        width,
        color=[palette[label] for label in selected["label"]],
        edgecolor="none",
        linewidth=0,
        zorder=2,
    )
    for position, value in zip(centers, values, strict=True):
        zoom.text(
            position,
            value + float(values.max()) * 0.055,
            (
                f"{value:.2f}%"
                if label_space == "collapsed_bio"
                else f"{value:.1f}%"
            ),
            ha="center",
            va="bottom",
            color="#344054",
            fontsize=8,
        )
    zoom.set_xticks(
        centers,
        [display_labels[label] for label in selected["label"]],
    )
    if label_space == "full_taxonomy":
        zoom.tick_params(axis="x", length=0)
        label_rows = {
            "ENTITY": -0.170,
            "CONCEPT": -0.040,
            "KEY_TERM": -0.300,
            "DEFINITION": -0.040,
            "RELATION_CUE": -0.430,
        }
        for position, raw_label, label in zip(
            centers,
            selected["label"],
            zoom.get_xticklabels(),
            strict=True,
        ):
            label_y = label_rows[str(raw_label)]
            label.set_y(label_y)
            zoom.plot(
                [position, position],
                [0.0, label_y + (0.015 if label_y < -0.05 else 0.010)],
                transform=zoom.get_xaxis_transform(),
                color="#667085",
                linewidth=0.7,
                alpha=0.80,
                clip_on=False,
                zorder=4,
            )
    zoom.set_xlim(0.5, float(len(complete)) - 0.5)
    zoom.set_ylim(0.0, zoom_upper)
    zoom.set_yticks(
        [
            tick
            for tick in zoom.get_yticks()
            if 0.0 < float(tick) <= zoom_upper
        ]
    )
    zoom.set_facecolor("#f4f7fa")
    zoom.grid(axis="y", color="#d8dee8", linewidth=0.45, alpha=0.70)
    zoom.set_axisbelow(True)
    zoom.tick_params(axis="both", labelsize=8)
    for spine in zoom.spines.values():
        spine.set_color("#7f8b99")
        spine.set_linewidth(0.75)


def draw_structure_distribution(
    axis,
    distribution: pd.DataFrame,
    *,
    dataset: str,
) -> None:
    selected = distribution.loc[distribution["dataset"].eq(dataset)].sort_values(
        "order", kind="mergesort"
    )
    width = 0.58
    bar_groups, _, group_spans = grouped_bar_layout((len(selected),), width)
    centers = bar_groups[0]
    values = selected["n_queries"].to_numpy(dtype=float)
    colors = (
        ["#7098b8"] * len(selected)
        if dataset == "TopiOCQA Dev"
        else ["#adc6da", "#7098b8", "#315f88"]
    )
    axis.bar(
        centers,
        values,
        width,
        color=colors,
        edgecolor="none",
        zorder=2,
    )
    for center, count, share in zip(
        centers,
        selected["n_queries"],
        selected["share"],
        strict=True,
    ):
        axis.text(
            center,
            float(count) + values.max() * 0.025,
            f"{int(count):,}\n{float(share):.1%}",
            ha="center",
            va="bottom",
            color="#344054",
        )
    axis.set_xticks(centers, selected["structure_label"])
    axis.set_xlim(
        group_spans[0][0] - width * 0.55,
        group_spans[0][1] + width * 0.55,
    )
    axis.set_ylim(0.0, values.max() * 1.27)
    axis.set_ylabel("#queries")
    if dataset == "TopiOCQA Dev":
        axis.set_title(
            "TopiOCQA Dev, n=2,514\nQuery distribution by encountered topic switches",
            loc="left",
        )
    else:
        axis.set_title(
            "QReCC Test (dataset OOD), n=8,209\nQuery distribution by source dataset",
            loc="left",
        )
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)


def draw_structure_performance(
    axis,
    performance: pd.DataFrame,
    *,
    dataset: str,
    backend: str,
) -> None:
    selected = performance.loc[
        performance["dataset"].eq(dataset)
        & performance["backend"].eq(backend)
    ].copy()
    labels = (
        selected[["order", "structure_label"]]
        .drop_duplicates()
        .sort_values("order", kind="mergesort")
    )
    methods = ("I-B64", "I-B512", "R-B64")
    colors = ("#c4cbd2", "#9099a3", "#315f88")
    max_high_by_order = selected.groupby("order")["MRR_ci95_high"].max()
    annotation_offsets = (
        (0.045, 0.180, 0.315)
        if dataset == "TopiOCQA Dev"
        else (0.040, 0.130, 0.220)
    )
    width = 0.22
    bar_groups, positions, group_spans = grouped_bar_layout(
        (len(methods),) * len(labels), width
    )
    for left, right in group_spans:
        axis.axvspan(
            left,
            right,
            ymin=0.0,
            ymax=0.84,
            facecolor="#adc6da",
            edgecolor="none",
            alpha=0.06,
            zorder=0,
        )
    for method_index, (method, color) in enumerate(
        zip(methods, colors, strict=True)
    ):
        method_rows = selected.loc[selected["method"].eq(method)].set_index(
            "order"
        )
        values = np.asarray(
            [float(method_rows.loc[order, "MRR"]) for order in labels["order"]]
        )
        lows = np.asarray(
            [
                float(method_rows.loc[order, "MRR_ci95_low"])
                for order in labels["order"]
            ]
        )
        highs = np.asarray(
            [
                float(method_rows.loc[order, "MRR_ci95_high"])
                for order in labels["order"]
            ]
        )
        centers = np.asarray(
            [group[method_index] for group in bar_groups], dtype=float
        )
        axis.bar(
            centers,
            values,
            width,
            color=color,
            edgecolor="none",
            label=method,
            zorder=2,
        )
        add_bootstrap_errorbars(
            axis,
            centers,
            values,
            lows,
            highs,
            fill_color=color,
        )
        if dataset != "TopiOCQA Dev":
            for center, order, value in zip(
                centers,
                labels["order"],
                values,
                strict=True,
            ):
                axis.text(
                    center,
                    float(max_high_by_order.loc[order])
                    + annotation_offsets[method_index],
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    color="#344054",
                )
    axis.set_xticks(positions, labels["structure_label"])
    axis.set_xlim(
        group_spans[0][0] - width * 0.55,
        group_spans[-1][1] + width * 0.55,
    )
    backend_label = "ANCE (retriever OOD)" if backend == "ANCE" else "BM25"
    if dataset == "TopiOCQA Dev":
        axis.set_title(
            f"TopiOCQA Dev, {backend_label}\n"
            "MRR by encountered topic switches",
            loc="left",
        )
        axis.set_ylim(0.0, 0.92)
    else:
        axis.set_title(
            f"QReCC Test (dataset OOD), {backend_label}\n"
            "MRR by source dataset",
            loc="left",
        )
        axis.set_ylim(0.0, 0.68)
    if backend == "ANCE":
        axis.set_xlabel(
            "Encountered topic switches"
            if dataset == "TopiOCQA Dev"
            else "Source dataset"
        )
    axis.set_ylabel("MRR")
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=True, ncol=3)
    add_preference_hint(axis, higher_is_better=True)


def load_oracle_headroom_analysis() -> dict[str, pd.DataFrame]:
    """Load and validate the refreshed NB03 artifact without fixed values."""
    manifest = json.loads(
        ORACLE_MANIFEST_SOURCE.read_text(encoding="utf-8")
    )
    identity = manifest.get("identity", {})
    if not (
        manifest.get("complete") is True
        and manifest.get("protocol")
        == "nb03_oracle_headroom_analysis_v2"
        and identity.get("split") == "train"
        and identity.get("sample_per_bin") == 100
        and tuple(identity.get("budgets", ())) == BUDGETS
    ):
        raise RuntimeError("NB03 oracle protocol or identity has drifted.")

    manifest_files = manifest.get("files", {})
    frames: dict[str, pd.DataFrame] = {}
    for name, path in ORACLE_SOURCES.items():
        expected_hash = manifest_files.get(path.name)
        if expected_hash is None or sha256(path) != expected_hash:
            raise RuntimeError(
                f"NB03 artifact hash mismatch or undeclared file: {path}"
            )
        frames[name] = pd.read_csv(path)

    population = frames["population"]
    if not (
        len(population) == 600
        and population["sample_id"].astype(str).nunique() == 600
        and population["history_depth"].ge(2).all()
        and population.groupby("depth_bin", observed=True).size().eq(100).all()
    ):
        raise RuntimeError("NB03 Train600 population is not the frozen design.")

    degradation = frames["budget_degradation"]
    oracle = frames["space_oracle"]
    relation = frames["control_relation"]
    dedup = frames["dedup"]
    generator = frames["generator"]
    monte_carlo = frames["monte_carlo"]
    entity = frames["entity_anchor"]
    expected_budgets = set(BUDGETS)
    if any(
        set(frame["budget"].astype(int)) != expected_budgets
        for frame in (degradation, oracle, relation, dedup, monte_carlo)
    ):
        raise RuntimeError("NB03 budget coverage is incomplete.")
    if not (
        set(oracle["control_mode"].astype(str))
        == {"exhaustive", "approximated"}
        and oracle["queries"].eq(300).all()
    ):
        raise RuntimeError("NB03 exhaustive/approximated cohorts drifted.")

    relation_order = ("negative", "neutral", "positive")
    if not (
        set(relation["relation_to_recency"].astype(str))
        == set(relation_order)
        and relation["queries"].eq(600).all()
        and np.allclose(
            relation.groupby("budget", observed=True)["share"].sum(),
            1.0,
            atol=1e-10,
        )
    ):
        raise RuntimeError("NB03 effective-input relation summary drifted.")

    control_dedup = dedup.loc[
        dedup["provenance_role"].eq("control")
    ].copy()
    effective_inputs = (
        relation.groupby("budget", observed=True)["effective_inputs"]
        .sum()
        .rename("effective_inputs")
    )
    mask_flow = (
        control_dedup.set_index("budget")
        .rename(
            columns={
                "rows_before_rewrite_dedup": "generated_masks",
                "rows_after_rewrite_dedup": "unique_rewrites",
            }
        )[["generated_masks", "unique_rewrites"]]
        .join(effective_inputs, how="inner")
        .reset_index()
        .sort_values("budget", kind="mergesort")
    )
    mask_flow = mask_flow[
        ["budget", "generated_masks", "effective_inputs", "unique_rewrites"]
    ]
    if not (
        len(mask_flow) == len(BUDGETS)
        and mask_flow["generated_masks"].nunique() == 1
        and (
            mask_flow["generated_masks"]
            >= mask_flow["effective_inputs"]
        ).all()
        and (
            mask_flow["effective_inputs"]
            >= mask_flow["unique_rewrites"]
        ).all()
    ):
        raise RuntimeError("NB03 mask/input/rewrite collapse is inconsistent.")

    generator_families = tuple(
        identity.get("candidate_config", {}).get("generator_families", ())
    )
    if not (
        len(generator_families) == 4
        and set(monte_carlo["generator_family"].astype(str))
        == set(generator_families)
        and len(monte_carlo) == len(BUDGETS) * len(generator_families)
        and monte_carlo["matched_n"].eq(600).all()
        and monte_carlo["null_support_status"].eq("valid").all()
    ):
        raise RuntimeError("NB03 matched-null generator grid is incomplete.")
    generator_plot = monte_carlo.merge(
        degradation[["budget", "recency_mrr"]],
        on="budget",
        how="left",
        validate="many_to_one",
    )
    generator_plot["generator_oracle_mrr"] = (
        generator_plot["recency_mrr"]
        + generator_plot["headroom_matched"]
    )
    b64_check = generator.merge(
        generator_plot.loc[
            generator_plot["budget"].eq(64),
            [
                "generator_family",
                "recency_mrr",
                "generator_oracle_mrr",
                "headroom_matched",
            ],
        ],
        on="generator_family",
        how="inner",
        suffixes=("_summary", "_mc"),
        validate="one_to_one",
    )
    if not (
        len(b64_check) == 4
        and np.allclose(
            b64_check["recency_mrr_summary"],
            b64_check["recency_mrr_mc"],
        )
        and np.allclose(
            b64_check["generator_oracle_mrr_summary"],
            b64_check["generator_oracle_mrr_mc"],
        )
        and np.allclose(
            b64_check["headroom"],
            b64_check["headroom_matched"],
        )
    ):
        raise RuntimeError("NB03 B64 generator and Monte Carlo results differ.")

    required_entity_populations = {
        "conversation-disjoint analysis sample",
        "eligible TopiOCQA train population",
    }
    entity_indexed = entity.set_index("population")
    if not (
        set(entity_indexed.index.astype(str))
        == required_entity_populations
        and int(
            entity_indexed.loc[
                "conversation-disjoint analysis sample", "queries"
            ]
        )
        == len(population)
        and int(
            entity_indexed.loc[
                "eligible TopiOCQA train population", "queries"
            ]
        )
        > len(population)
        and (
            entity["queries_with_exact_entity_anchor"]
            + entity["queries_without_exact_entity_anchor"]
            == entity["queries"]
        ).all()
        and np.allclose(
            entity["no_entity_anchor_rate"],
            entity["queries_without_exact_entity_anchor"]
            / entity["queries"],
        )
    ):
        raise RuntimeError("NB03 entity-anchor populations have drifted.")

    frames["mask_flow"] = mask_flow
    frames["generator_plot"] = generator_plot
    return frames


ORACLE_GENERATOR_STYLES = {
    "recent_variants": ("Recent variants", "#315f88", "o", "-"),
    "position_anchor": ("Position anchor", "#7098b8", "s", "-"),
    "lexical_overlap": ("Lexical overlap", "#8d98a5", "^", "--"),
    "spacy_recent_entity": ("Recent entity", "#adc6da", "D", "-"),
}


def draw_space_oracle_headroom(axis, oracle: pd.DataFrame) -> None:
    styles = {
        "exhaustive": {
            "label": "Exh.",
            "recency": "#8d98a5",
            "oracle": "#7098b8",
            "marker": "o",
        },
        "approximated": {
            "label": "Approx.",
            "recency": "#667085",
            "oracle": "#315f88",
            "marker": "D",
        },
    }
    positions = np.arange(len(BUDGETS), dtype=float)
    for mode in ("exhaustive", "approximated"):
        selected = oracle.loc[oracle["control_mode"].eq(mode)].sort_values(
            "budget", kind="mergesort"
        )
        style = styles[mode]
        recency = selected["recency_mrr"].to_numpy(dtype=float)
        oracle_mrr = selected["space_oracle_mrr"].to_numpy(dtype=float)
        axis.fill_between(
            positions,
            recency,
            oracle_mrr,
            color=style["oracle"],
            alpha=0.08,
            zorder=1,
        )
        axis.plot(
            positions,
            recency,
            color=style["recency"],
            linestyle="--",
            linewidth=1.35,
            marker=style["marker"],
            markersize=4.5,
            label=f"{style['label']} - Recency",
            zorder=3,
        )
        axis.plot(
            positions,
            oracle_mrr,
            color=style["oracle"],
            linestyle="-",
            linewidth=1.65,
            marker=style["marker"],
            markersize=5.0,
            label=f"{style['label']} - Oracle",
            zorder=4,
        )
        for position, value, headroom in zip(
            positions,
            oracle_mrr,
            selected["space_oracle_headroom"].to_numpy(dtype=float),
            strict=True,
        ):
            offset = 5 if mode == "approximated" else -7
            axis.annotate(
                f"+{headroom:.3f}",
                xy=(position, value),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va="bottom" if offset > 0 else "top",
                color=style["oracle"],
            )
    axis.set_title(
        "Space-oracle MRR and headroom\n"
        "Train600: exhaustive/approximated n=300 each",
        loc="left",
    )
    axis.set_xlabel("Fixed total rewriter-input budget")
    axis.set_ylabel("MRR")
    axis.set_xticks(positions, [f"B{budget}" for budget in BUDGETS])
    plotted_min = float(oracle["recency_mrr"].min())
    plotted_max = float(oracle["space_oracle_mrr"].max())
    plotted_span = plotted_max - plotted_min
    axis.set_ylim(
        max(0.0, plotted_min - plotted_span * 0.12),
        plotted_max + plotted_span * 0.72,
    )
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=True, ncol=2)


def draw_effective_mask_outcomes(axis, relation: pd.DataFrame) -> None:
    order = ("negative", "neutral", "positive")
    styles = {
        "negative": ("Negative", "#8d98a5"),
        "neutral": ("Neutral", "#c7d6e2"),
        "positive": ("Positive", "#315f88"),
    }
    lookup = relation.set_index(["budget", "relation_to_recency"])
    x = np.arange(len(BUDGETS), dtype=float)
    bottom = np.zeros(len(BUDGETS), dtype=float)
    for outcome in order:
        label, color = styles[outcome]
        shares = np.asarray(
            [lookup.loc[(budget, outcome), "share"] for budget in BUDGETS],
            dtype=float,
        )
        counts = np.asarray(
            [
                lookup.loc[(budget, outcome), "effective_inputs"]
                for budget in BUDGETS
            ],
            dtype=int,
        )
        axis.bar(
            x,
            shares * 100.0,
            bottom=bottom * 100.0,
            width=0.62,
            color=color,
            edgecolor="none",
            label=label,
        )
        for position, base, share, count in zip(
            x, bottom, shares, counts, strict=True
        ):
            axis.text(
                position,
                (base + share / 2.0) * 100.0,
                f"{count:,}\n{share:.1%}",
                ha="center",
                va="center",
                color="white" if outcome == "positive" else PLOT_INK,
            )
        bottom += shares
    axis.set_title(
        "Effective mask outcomes vs Recency\n"
        "Input-deduplicated within query; Train600",
        loc="left",
    )
    axis.set_xlabel("Fixed total rewriter-input budget")
    axis.set_ylabel("Mean share (%)")
    axis.set_xticks(x, [f"B{budget}" for budget in BUDGETS])
    axis.set_ylim(0.0, 106.0)
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=True, ncol=3)


def draw_generator_oracle_mrr(
    axis,
    generator_plot: pd.DataFrame,
) -> None:
    positions = np.arange(len(BUDGETS), dtype=float)
    recency = (
        generator_plot[["budget", "recency_mrr"]]
        .drop_duplicates()
        .sort_values("budget", kind="mergesort")
    )
    axis.plot(
        positions,
        recency["recency_mrr"],
        color="#475467",
        linewidth=1.4,
        linestyle="--",
        marker="o",
        markersize=4.5,
        label="Recency",
        zorder=4,
    )
    for family, (label, color, marker, linestyle) in (
        ORACLE_GENERATOR_STYLES.items()
    ):
        selected = generator_plot.loc[
            generator_plot["generator_family"].eq(family)
        ].sort_values("budget", kind="mergesort")
        axis.plot(
            positions,
            selected["generator_oracle_mrr"],
            color=color,
            linewidth=1.5,
            linestyle=linestyle,
            marker=marker,
            markersize=4.7,
            label=label,
            zorder=5,
        )
    axis.set_title(
        "Generator-oracle MRR vs Recency\n"
        "Retrospective best-of-K; Train600",
        loc="left",
    )
    axis.set_xlabel("Fixed total rewriter-input budget")
    axis.set_ylabel("MRR")
    axis.set_xticks(positions, [f"B{budget}" for budget in BUDGETS])
    plotted_values = np.concatenate(
        [
            generator_plot["recency_mrr"].to_numpy(dtype=float),
            generator_plot["generator_oracle_mrr"].to_numpy(dtype=float),
        ]
    )
    plotted_span = float(plotted_values.max() - plotted_values.min())
    axis.set_ylim(
        max(0.0, float(plotted_values.min()) - plotted_span * 0.18),
        float(plotted_values.max()) + plotted_span * 0.55,
    )
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=True, ncol=2)


def draw_generator_excess(
    axis,
    generator_plot: pd.DataFrame,
) -> None:
    positions = np.arange(len(BUDGETS), dtype=float)
    axis.axhline(
        0.0,
        color="#98a2b3",
        linewidth=0.8,
        linestyle=(0, (2, 2)),
        zorder=1,
    )
    for family, (label, color, marker, linestyle) in (
        ORACLE_GENERATOR_STYLES.items()
    ):
        selected = generator_plot.loc[
            generator_plot["generator_family"].eq(family)
        ].sort_values("budget", kind="mergesort")
        axis.plot(
            positions,
            selected["excess"],
            color=color,
            linewidth=1.5,
            linestyle=linestyle,
            marker=marker,
            markersize=4.7,
            label=label,
            zorder=5,
        )
        for position, row in zip(
            positions, selected.itertuples(index=False), strict=True
        ):
            if float(row.p_holm) < 0.05:
                axis.annotate(
                    "†",
                    xy=(position, float(row.excess)),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    color=color,
                )
    axis.set_title(
        "Excess over matched random masks\n"
        "Observed minus matched-null headroom; Train600",
        loc="left",
    )
    axis.set_xlabel("Fixed total rewriter-input budget")
    axis.set_ylabel("Excess MRR")
    axis.set_xticks(positions, [f"B{budget}" for budget in BUDGETS])
    excess_limit = float(generator_plot["excess"].abs().max()) * 1.45
    axis.set_ylim(-excess_limit, excess_limit)
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)


def draw_mask_collapse(axis, mask_flow: pd.DataFrame) -> None:
    stages = (
        ("generated_masks", "Masks", "#8d98a5"),
        ("effective_inputs", "Inputs", "#adc6da"),
        ("unique_rewrites", "Rewrites", "#315f88"),
    )
    width = 0.18
    groups, centers, spans = grouped_bar_layout(
        (len(stages),) * len(BUDGETS), width
    )
    lookup = mask_flow.set_index("budget")
    for stage_index, (column, label, color) in enumerate(stages):
        positions = np.asarray(
            [group[stage_index] for group in groups], dtype=float
        )
        values = np.asarray(
            [lookup.loc[budget, column] for budget in BUDGETS], dtype=float
        )
        axis.bar(
            positions,
            values,
            width=width,
            color=color,
            edgecolor="none",
            label=label,
        )
        maximum_count = float(mask_flow["generated_masks"].max())
        label_offsets = (
            maximum_count * 0.140,
            maximum_count * 0.040,
            maximum_count * 0.010,
        )
        for position, value in zip(positions, values, strict=True):
            axis.text(
                position,
                value + label_offsets[stage_index],
                f"{int(value):,}",
                ha="center",
                va="bottom",
            )
    axis.set_title(
        "Mask → input → rewrite collapse\n"
        "After deterministic deduplication; Train600",
        loc="left",
    )
    axis.set_xlabel("Fixed total rewriter-input budget")
    axis.set_ylabel("Count")
    axis.set_xticks(centers, [f"B{budget}" for budget in BUDGETS])
    axis.set_xlim(spans[0][0] - width * 0.55, spans[-1][1] + width * 0.55)
    axis.set_ylim(0.0, float(mask_flow["generated_masks"].max()) * 1.36)
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)
    axis.legend(loc="upper left", frameon=True, ncol=3)


def draw_entity_anchor_diagnostic(axis, entity: pd.DataFrame) -> None:
    order = (
        "conversation-disjoint analysis sample",
        "eligible TopiOCQA train population",
    )
    labels = ("Train600", "Eligible train")
    selected = entity.set_index("population").loc[list(order)]
    x = np.arange(len(order), dtype=float)
    no_anchor = selected["no_entity_anchor_rate"].to_numpy(dtype=float) * 100
    anchor = selected["entity_anchor_coverage"].to_numpy(dtype=float) * 100
    axis.bar(
        x,
        no_anchor,
        width=0.56,
        color="#315f88",
        edgecolor="none",
        label="No exact entity anchor",
    )
    axis.bar(
        x,
        anchor,
        width=0.56,
        bottom=no_anchor,
        color="#adc6da",
        edgecolor="none",
        label="Exact entity anchor",
    )
    for position, row in zip(x, selected.itertuples(), strict=True):
        axis.text(
            position,
            float(row.no_entity_anchor_rate) * 50.0,
            (
                f"{float(row.no_entity_anchor_rate):.2%}\n"
                f"{int(row.queries_without_exact_entity_anchor):,}/"
                f"{int(row.queries):,}"
            ),
            ha="center",
            va="center",
            color="white",
        )
        axis.text(
            position,
            101.0,
            f"Exact: {float(row.entity_anchor_coverage):.2%}",
            ha="center",
            va="bottom",
        )
    axis.set_title(
        "Naive query-conditioned entity matching\n"
        "spacy_query_entity_overlap exact anchors",
        loc="left",
    )
    axis.set_xlabel("Population")
    axis.set_ylabel("Queries (%)")
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 109.0)
    axis.grid(axis="y", color="#d8dee8", linewidth=0.5, alpha=0.65)
    axis.set_axisbelow(True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_ROOT,
                        help="Notebook result tree; all analysis sources come from this tree.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "experiments/data",
                        help="Prepared topiocqa/ and qrecc/ dataset directories.")
    parser.add_argument("--itercqr-model-dir", type=Path, default=ITERCQR_MODEL_DIR,
                        help="Local IterCQR tokenizer directory; no model weights are loaded.")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: RESULTS_DIR/10_efficiency_analysis/figures.")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Check saved-input paths only; no inference, tokenizer loading, resampling or output writes.")
    return parser.parse_args(argv)


def configure_paths(args) -> None:
    """Relocate the existing path constants, not their statistical definitions."""
    results = args.results_dir.expanduser().resolve()
    output = (args.output_dir or results / "10_efficiency_analysis/figures").expanduser().resolve()
    mappings = (
        (FIGURE_DIR, output),
        (RESULTS_ROOT, results),
        (ROOT / "experiments/data", args.data_dir.expanduser().resolve()),
        (ITERCQR_MODEL_DIR, args.itercqr_model_dir.expanduser().resolve()),
    )

    def relocate(value):
        if isinstance(value, Path):
            for original, destination in mappings:
                if value.is_relative_to(original):
                    return destination / value.relative_to(original)
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        return value

    for name, value in list(globals().items()):
        if name.isupper() and name != "ROOT":
            globals()[name] = relocate(value)


def required_input_files() -> tuple[Path, ...]:
    sources = [
        LATENCY_RUNS_SOURCE, LATENCY_DIR / "reproducibility_route_components_summary.csv",
        LATENCY_DIR / "reproducibility_manifest.json",
        NSIGHT_DIR / "topiocqa_t5_encoder_summary.csv",
        NSIGHT_DIR / "topiocqa_t5_encoder_component_metrics.csv",
        TOPIOCQA_CONTEXT_SOURCE, QRECC_CONTEXT_SOURCE,
        TOPIOCQA_DEV_SOURCE, QRECC_TEST_METADATA_SOURCE,
        QRECC_DATA_DIR / "collection/qrecc-train.json", PRETRAINED_VITERBI_SOURCE,
        ITERCQR_MODEL_DIR / "tokenizer_config.json",
        GRANULARITY_PROJECTION_DIR / "manifest.json", BM25_DIR / "manifest.json",
        GRANULARITY_COMPARISON_SOURCE, GRANULARITY_LENGTH_SOURCE,
        GRANULARITY_METRICS_SOURCE, GRANULARITY_SUMMARY_SOURCE,
        DQCIS_SUMMARY_SOURCE, DQCIS_METRICS_SOURCE, DQCIS_COMPARISONS_SOURCE,
        DQCIS_MANIFEST_SOURCE, TEACHER_STUDENT_MRR_SOURCE,
        TEACHER_STUDENT_COMPARISON_SOURCE, TEACHER_STUDENT_F1_SOURCE,
        TEACHER_STUDENT_AGREEMENT_COUNTS_SOURCE, TEACHER_STUDENT_POPULATION_SOURCE,
        TEACHER_STUDENT_MANIFEST_SOURCE, TEACHER_LABELS_SOURCE,
        ORACLE_MANIFEST_SOURCE, *ORACLE_SOURCES.values(),
    ]
    for backend_dir in (BM25_DIR, ANCE_DIR, QRECC_DIR / "bm25", QRECC_DIR / "ance"):
        sources.extend(backend_dir / name for name in ("route_summary.csv", "route_metrics_by_query.csv"))
    for root in CURRENT_QUERY_ONLY_DIRS.values():
        sources.append(root / "query_bundle/manifest.json")
        for backend in ("bm25", "ance"):
            sources.extend(root / backend / name for name in ("final_manifest.json", "metrics_by_query.csv", "summary.csv"))
    sources.extend(path for mapping in SATURATION_SOURCES.values() for path in mapping.values())
    return tuple(dict.fromkeys(sources))


def check_inputs() -> None:
    missing = [str(path) for path in required_input_files() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing figure inputs (complete the indicated notebook/script first):\n  " + "\n  ".join(missing))
    print(f"{len(required_input_files())} required files are present; contents not yet validated.")
    print("Also requires cached sentence-transformers/all-MiniLM-L12-v2 tokenizer at revision a50ef00143b4d5391434df20ae11632588ac25be; no automatic download.")
    print(f"Output: {FIGURE_DIR}; font: {PLOT_FONT}")


def main() -> None:
    (
        latency,
        quality,
        comparisons,
        qrecc_quality,
        qrecc_comparisons,
        resources,
    ) = validated_inputs()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    latency_comparisons = load_latency_comparisons()
    topiocqa_context_lengths, qrecc_context_lengths = load_context_lengths()
    r_b128_minus_b64 = load_r_b128_minus_b64()
    granularity_bars, granularity_lengths = load_granularity_projection()
    granularity_baseline_comparisons = (
        bootstrap_granularity_baseline_comparisons()
    )
    dqcis_intervals, dqcis_comparisons = load_dqcis_fusion_ablation()
    (
        teacher_student_mrr,
        teacher_student_f1,
        teacher_student_comparisons,
        teacher_student_f1_comparisons,
    ) = (
        load_teacher_student_analysis()
    )
    teacher_label_distribution = load_teacher_label_distribution()
    (
        granularity_length_intervals,
        granularity_baseline_length_interval,
    ) = (
        bootstrap_granularity_length_intervals(granularity_lengths)
    )
    current_query_only_mrr = load_current_query_only_mrr()
    depth_strata_mrr = load_depth_strata_mrr()
    dataset_structure, r_route_structure = load_dataset_structure_analysis()
    oracle_data = load_oracle_headroom_analysis()
    bootstrap_ci_path = FIGURE_DIR / "quality_bar_bootstrap_ci.csv"
    depth_strata_path = FIGURE_DIR / "depth_strata_mrr_bootstrap_ci.csv"
    depth_strata_mrr.to_csv(depth_strata_path, index=False)
    dataset_structure.to_csv(DATASET_STRUCTURE_EXPORT, index=False)
    r_route_structure.to_csv(STRUCTURE_QUALITY_EXPORT, index=False)
    current_query_only_mrr.to_csv(
        CURRENT_QUERY_ONLY_CI_SOURCE,
        index=False,
    )
    granularity_length_intervals.to_csv(
        GRANULARITY_LENGTH_CI_SOURCE,
        index=False,
    )
    granularity_baseline_length_interval.to_csv(
        GRANULARITY_BASELINE_LENGTH_CI_SOURCE,
        index=False,
    )
    dqcis_intervals.to_csv(DQCIS_BAR_CI_SOURCE, index=False)
    shutil.copyfile(TEACHER_STUDENT_MRR_SOURCE, TEACHER_STUDENT_MRR_EXPORT)
    shutil.copyfile(TEACHER_STUDENT_F1_SOURCE, TEACHER_STUDENT_F1_EXPORT)
    bootstrap_columns = [
        "dataset",
        "backend",
        "system",
        "budget",
        "route",
        "n",
        "bootstrap_seed",
        "bootstrap_replicates",
        *[
            column
            for metric in QUALITY_METRICS
            for column in (
                metric,
                f"{metric}_ci95_low",
                f"{metric}_ci95_high",
            )
        ],
    ]
    pd.concat(
        [
            quality.assign(dataset="TopiOCQA Dev"),
            qrecc_quality.assign(dataset="QReCC Test"),
        ],
        ignore_index=True,
    )[bootstrap_columns].to_csv(bootstrap_ci_path, index=False)

    budget_mrr_figure = plt.figure(
        figsize=(15.8, 11.6),
        constrained_layout=True,
    )
    budget_mrr_grid = budget_mrr_figure.add_gridspec(
        4,
        2,
        hspace=0.22,
    )
    budget_left_grid = budget_mrr_grid[:, 0].subgridspec(
        4,
        2,
        width_ratios=(0.90, 5.10),
        wspace=0.0,
    )
    current_query_axes = tuple(
        budget_mrr_figure.add_subplot(budget_left_grid[row, 0])
        for row in range(4)
    )
    first_budget_mrr_axis = budget_mrr_figure.add_subplot(
        budget_left_grid[0, 1]
    )
    budget_mrr_axes = (
        first_budget_mrr_axis,
        *(
            budget_mrr_figure.add_subplot(
                budget_left_grid[row, 1],
                sharex=first_budget_mrr_axis,
            )
            for row in range(1, 4)
        ),
    )
    first_depth_mrr_axis = budget_mrr_figure.add_subplot(
        budget_mrr_grid[0, 1]
    )
    depth_mrr_axes = (
        first_depth_mrr_axis,
        *(
            budget_mrr_figure.add_subplot(
                budget_mrr_grid[row, 1],
                sharey=first_depth_mrr_axis,
            )
            for row in range(1, 4)
        ),
    )
    budget_mrr_specs = (
        (
            current_query_axes[0], budget_mrr_axes[0], quality,
            "TopiOCQA Dev", 2_514,
            "BM25", None, True,
        ),
        (
            current_query_axes[1], budget_mrr_axes[1], quality,
            "TopiOCQA Dev", 2_514,
            "ANCE", "ANCE (retriever OOD)", False,
        ),
        (
            current_query_axes[2], budget_mrr_axes[2], qrecc_quality,
            "QReCC Test (dataset OOD)", 8_209,
            "BM25", None, False,
        ),
        (
            current_query_axes[3], budget_mrr_axes[3], qrecc_quality,
            "QReCC Test (dataset OOD)", 8_209,
            "ANCE", "ANCE (retriever OOD)", False,
        ),
    )
    depth_y_max = float(depth_strata_mrr["MRR_ci95_high"].max()) * 1.28
    row_titles = []
    for (
        current_axis,
        budget_axis,
        mrr_data,
        dataset,
        n,
        backend,
        backend_label,
        show_legend,
    ) in budget_mrr_specs:
        draw_current_query_only_mrr(
            current_axis,
            current_query_only_mrr,
            dataset=dataset,
            n=n,
            backend=backend,
            backend_label=backend_label,
            show_x_label=current_axis is current_query_axes[-1],
        )
        draw_budget_mrr(
            budget_axis,
            mrr_data,
            dataset=dataset,
            n=n,
            backend=backend,
            backend_label=backend_label,
            show_legend=show_legend,
            show_title=False,
            show_ylabel=False,
        )
    for axis in budget_mrr_axes[:-1]:
        axis.tick_params(axis="x", labelbottom=False)
    depth_mrr_specs = (
        (
            depth_mrr_axes[0], "TopiOCQA Dev", 2_514,
            "BM25", None, True,
        ),
        (
            depth_mrr_axes[1], "TopiOCQA Dev", 2_514,
            "ANCE", "ANCE (retriever OOD)", False,
        ),
        (
            depth_mrr_axes[2], "QReCC Test (dataset OOD)", 8_209,
            "BM25", None, False,
        ),
        (
            depth_mrr_axes[3], "QReCC Test (dataset OOD)", 8_209,
            "ANCE", "ANCE (retriever OOD)", False,
        ),
    )
    for axis, dataset, n, backend, backend_label, show_legend in (
        depth_mrr_specs
    ):
        draw_depth_strata_mrr(
            axis,
            depth_strata_mrr,
            dataset=dataset,
            n=n,
            backend=backend,
            backend_label=backend_label,
            show_legend=show_legend,
            y_max=depth_y_max,
        )
    budget_mrr_figure.supxlabel(
        "Fixed total rewriter-input budget\n"
        "Error bars: pointwise query-bootstrap 95% CI",
        x=0.31,
        fontsize=9,
    )
    budget_mrr_figure.text(
        0.75,
        0.006,
        "History-depth stratum (previous turns)\n"
        "Error bars: pointwise query-bootstrap 95% CI",
        ha="center",
        va="bottom",
        fontsize=9,
    )

    width = 0.145
    latency_bar_groups, positions, _ = grouped_bar_layout(
        (len(ROUTES),) * len(BUDGETS),
        width,
    )
    latency_route_positions = {
        route: np.asarray(
            [group[index] for group in latency_bar_groups]
        )
        for index, route in enumerate(ROUTES)
    }
    offsets = {
        route: float(latency_route_positions[route][0] - positions[0])
        for route in ROUTES
    }

    figure = plt.figure(figsize=(15.8, 10.4), constrained_layout=True)
    outer_grid = figure.add_gridspec(
        1,
        2,
        width_ratios=(1.08, 1.16),
    )
    left_grid = outer_grid[0, 0].subgridspec(
        3,
        1,
        height_ratios=(1.80, 1.35, 1.35),
    )
    right_grid = outer_grid[0, 1].subgridspec(2, 1)
    latency_axis = figure.add_subplot(left_grid[0, 0])
    bm25_axis = figure.add_subplot(left_grid[1, 0], sharex=latency_axis)
    ance_axis = figure.add_subplot(left_grid[2, 0], sharex=latency_axis)
    flops_axis = figure.add_subplot(right_grid[0, 0])
    dram_axis = figure.add_subplot(right_grid[1, 0], sharex=flops_axis)

    latency_lookup = latency.set_index(["budget", "route"])

    def available_latency_value(
        budget: int,
        route: str,
        column: str,
    ) -> float:
        key = (budget, route)
        return (
            float(latency_lookup.loc[key, column])
            if key in latency_lookup.index
            else float("nan")
        )

    latency_max = float(latency["ci95_high_total_ms"].max())
    for route in ROUTES:
        totals = np.asarray(
            [
                available_latency_value(budget, route, "mean_total_ms")
                for budget in BUDGETS
            ],
            dtype=float,
        )
        lows = np.asarray(
            [
                available_latency_value(
                    budget,
                    route,
                    "ci95_low_total_ms",
                )
                for budget in BUDGETS
            ],
            dtype=float,
        )
        highs = np.asarray(
            [
                available_latency_value(
                    budget,
                    route,
                    "ci95_high_total_ms",
                )
                for budget in BUDGETS
            ],
            dtype=float,
        )
        bottoms = np.zeros(len(BUDGETS), dtype=float)
        for column, label, color in COMPONENTS:
            values = np.asarray(
                [
                    available_latency_value(budget, route, column)
                    for budget in BUDGETS
                ],
                dtype=float,
            )
            bars = latency_axis.bar(
                latency_route_positions[route],
                values,
                width,
                bottom=bottoms,
                color=color,
                edgecolor="none",
                linewidth=0,
                label=label if route == ROUTES[0] else None,
            )
            fade_middle_budgets(bars)
            if column == "mean_t5_encoder_ms":
                for bar in bars:
                    bar.set_hatch("///")
                    bar.set_edgecolor("white")
                    bar.set_linewidth(0)
            bottoms += values
        latency_axis.errorbar(
            latency_route_positions[route],
            totals,
            yerr=np.vstack([totals - lows, highs - totals]),
            fmt="none",
            ecolor="#344054",
            elinewidth=0.8,
            capsize=2.5,
            capthick=0.8,
            zorder=4,
        )
        for x, value, high in zip(
            latency_route_positions[route], totals, highs, strict=True
        ):
            latency_axis.text(
                x,
                high + latency_max * 0.018,
                f"{route}\n{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                linespacing=0.9,
            )
    latency_axis.set_ylabel("Wall time (ms/query)")
    latency_axis.set_title(
        "Measured preprocessing wall time by route and component\n"
        "Mean with 95% CI; depth-balanced train sample n=600; "
        "warm-up, retrieval and RRF excluded",
        loc="left",
        fontsize=9.8,
        pad=8,
    )
    latency_axis.set_ylim(0, latency_max * 1.18)
    latency_axis.grid(axis="x", visible=False)
    latency_axis.tick_params(axis="x", labelbottom=False)
    latency_axis.legend(
        title="Measured component",
        frameon=True,
        facecolor="white",
        edgecolor="#d0d5dd",
        framealpha=1,
        fontsize=7.7,
        title_fontsize=7.7,
        ncol=3,
        loc="upper left",
        borderaxespad=0.45,
    )
    add_method_callouts(
        latency_axis,
        positions,
        offsets,
        y=-0.055,
    )
    add_preference_hint(latency_axis, higher_is_better=False)

    draw_quality(
        bm25_axis,
        quality,
        comparisons,
        dataset="TopiOCQA Dev",
        n=2_514,
        backend="BM25",
        show_legend=True,
        show_callouts=False,
    )
    draw_quality(
        ance_axis,
        quality,
        comparisons,
        dataset="TopiOCQA Dev",
        n=2_514,
        backend="ANCE",
        backend_label="ANCE (retriever OOD)",
        show_legend=False,
        show_callouts=True,
    )

    draw_encoder_resources(flops_axis, dram_axis, resources)
    figure.supxlabel(
        "Error bars: pointwise query-bootstrap 95% CI",
        fontsize=8.5,
    )

    output_png = FIGURE_DIR / "topiocqa_ir_efficiency.png"
    output_pdf = FIGURE_DIR / "topiocqa_ir_efficiency.pdf"
    standardize_figure_typography(figure)
    figure.savefig(
        output_png,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "ROCC reproducible latency plot"},
    )

    primary_rows = {
        "I64": latency_lookup.loc[(64, "I")],
        "I512": latency_lookup.loc[(512, "I")],
        "R": latency_lookup.loc[(64, "R")],
        "D": latency_lookup.loc[(64, "D")],
        "IRD": latency_lookup.loc[(64, "IRD")],
    }
    primary_latency = {
        key: float(row.mean_total_ms)
        for key, row in primary_rows.items()
    }
    primary_latency_intervals = {
        key: (
            float(row.ci95_low_total_ms),
            float(row.ci95_high_total_ms),
        )
        for key, row in primary_rows.items()
    }
    primary_latency_comparisons = {
        str(row.system_key): (
            float(row.ci95_low),
            float(row.ci95_high),
        )
        for row in latency_comparisons.itertuples(index=False)
    }
    primary_latency_segments = {
        key: tuple(
            (float(row[column]), color, label)
            for column, label, color in COMPONENTS
            if float(row[column]) > 0.0
        )
        for key, row in primary_rows.items()
    }
    resource_lookup = resources.set_index(["route", "budget"])

    def primary_encoder_resource(
        column: str,
        scale: float,
    ) -> tuple[
        dict[str, float],
        dict[str, tuple[tuple[float, str, str], ...]],
    ]:
        i64 = float(resource_lookup.loc[("I", 64), column]) / scale
        i512 = float(resource_lookup.loc[("I", 512), column]) / scale
        r64 = float(resource_lookup.loc[("R", 64), column]) / scale
        component_values = {
            (route, budget): {
                component: float(
                    resource_lookup.loc[
                        (route, budget), f"{column}_{component}"
                    ]
                )
                / scale
                for component, _, _ in DISPLAY_ENCODER_COMPONENTS
            }
            for route, budget in (("I", 64), ("I", 512), ("R", 64))
        }

        def component_segments(route: str, budget: int):
            return tuple(
                (
                    component_values[(route, budget)][component],
                    color,
                    label,
                )
                for component, label, color in DISPLAY_ENCODER_COMPONENTS
            )

        values = {
            "I64": i64,
            "I512": i512,
            "R": r64,
            "D": 0.0,
            "IRD": i64 + r64,
        }
        segments = {
            "I64": component_segments("I", 64),
            "I512": component_segments("I", 512),
            "R": component_segments("R", 64),
            "D": (),
            "IRD": tuple(
                (
                    component_values[("I", 64)][component]
                    + component_values[("R", 64)][component],
                    color,
                    label,
                )
                for component, label, color in DISPLAY_ENCODER_COMPONENTS
            ),
        }
        return values, segments

    compute_values, compute_segments = primary_encoder_resource(
        "fp32_flops", 1e9
    )
    memory_values, memory_segments = primary_encoder_resource(
        "dram_bytes", 1e9
    )

    efficiency_figure = plt.figure(
        figsize=(15.8, 10.4),
        constrained_layout=True,
    )
    efficiency_grid = efficiency_figure.add_gridspec(
        4,
        2,
        height_ratios=(1.0, 1.0, 0.050, 0.022),
        wspace=0.015,
    )
    primary_latency_axis = efficiency_figure.add_subplot(
        efficiency_grid[:2, 0]
    )
    primary_compute_axis = efficiency_figure.add_subplot(
        efficiency_grid[0, 1]
    )
    primary_memory_axis = efficiency_figure.add_subplot(
        efficiency_grid[1, 1]
    )
    latency_note_axis = efficiency_figure.add_subplot(
        efficiency_grid[2, 0]
    )
    encoder_note_axis = efficiency_figure.add_subplot(
        efficiency_grid[2, 1]
    )
    scope_note_axis = efficiency_figure.add_subplot(
        efficiency_grid[3, :]
    )
    for note_axis in (
        latency_note_axis,
        encoder_note_axis,
        scope_note_axis,
    ):
        note_axis.set_axis_off()
    draw_efficiency_comparison(
        primary_latency_axis,
        primary_latency,
        title=(
            "Preprocessing latency\n"
            "Mean with 95% t-CI across five independent runs; n=600"
        ),
        ylabel="Wall time (ms/query)",
        decimals=1,
        intervals=primary_latency_intervals,
        comparison_intervals=primary_latency_comparisons,
        segments=primary_latency_segments,
        hatched_labels=frozenset({"T5 encoder"}),
        show_legend=True,
    )
    draw_efficiency_comparison(
        primary_compute_axis,
        compute_values,
        title=(
            "T5-encoder FP32 work by component\n"
            "Nsight Compute; one B16 route forward"
        ),
        ylabel="GFLOP",
        decimals=1,
        segments=compute_segments,
        hatched_labels=frozenset(
            label for _, label, _ in DISPLAY_ENCODER_COMPONENTS
        ),
        show_legend=True,
        family_gap_multiplier=COMPARISON_FAMILY_GAP_MULTIPLIER,
    )
    draw_efficiency_comparison(
        primary_memory_axis,
        memory_values,
        title=(
            "Range-replay whole-forward T5-encoder DRAM traffic\n"
            "Total allocated by kernel-replay component shares"
        ),
        ylabel="DRAM traffic (GB)",
        decimals=2,
        segments=memory_segments,
        hatched_labels=frozenset(
            label for _, label, _ in DISPLAY_ENCODER_COMPONENTS
        ),
        family_gap_multiplier=COMPARISON_FAMILY_GAP_MULTIPLIER,
    )
    latency_note_axis.text(
        0.5,
        0.5,
        "Latency includes ROCC selection and T5 rewriting\n"
        "Dashed line: I-B512 (0.0%); paired t-CI: "
        "† excludes 0; ‡ includes 0",
        ha="center",
        va="center",
        transform=latency_note_axis.transAxes,
        fontsize=9,
    )
    encoder_note_axis.text(
        0.5,
        0.5,
        "Encoder only (D=0; IRD=I+R). FP32 components are direct.\n"
        "DRAM: range-replay whole-forward totals allocated by "
        "kernel-replay shares",
        ha="center",
        va="center",
        transform=encoder_note_axis.transAxes,
        fontsize=9,
    )
    scope_note_axis.text(
        0.5,
        0.5,
        "Warm-up, retrieval and RRF are excluded",
        ha="center",
        va="center",
        transform=scope_note_axis.transAxes,
        fontsize=9,
    )

    context_length_figure = plt.figure(
        figsize=(15.8, 5.7),
        constrained_layout=False,
    )
    context_length_grid = context_length_figure.add_gridspec(
        1,
        2,
        left=0.070,
        right=0.995,
        bottom=0.145,
        top=0.955,
        wspace=0.170,
    )
    context_length_axes = (
        context_length_figure.add_subplot(context_length_grid[0, 0]),
        context_length_figure.add_subplot(context_length_grid[0, 1]),
    )
    context_length_axes[1].sharex(context_length_axes[0])
    draw_context_length_panel(
        context_length_axes[0],
        topiocqa_context_lengths,
        "TopiOCQA Dev, n=2,514",
        r_b128_minus_b64["topiocqa"],
    )
    draw_context_length_panel(
        context_length_axes[1],
        qrecc_context_lengths,
        "QReCC Test (dataset OOD), n=8,209",
        r_b128_minus_b64["qrecc"],
    )

    qrecc_figure = plt.figure(figsize=(15.8, 10.4), constrained_layout=True)
    qrecc_grid = qrecc_figure.add_gridspec(2, 2)
    qrecc_bm25_axis = qrecc_figure.add_subplot(qrecc_grid[0, 0])
    qrecc_ance_axis = qrecc_figure.add_subplot(
        qrecc_grid[1, 0],
        sharex=qrecc_bm25_axis,
    )
    qrecc_blank_axis = qrecc_figure.add_subplot(qrecc_grid[:, 1])
    qrecc_blank_axis.set_axis_off()
    draw_quality(
        qrecc_bm25_axis,
        qrecc_quality,
        qrecc_comparisons,
        dataset="QReCC Test (dataset OOD)",
        n=8_209,
        backend="BM25",
        show_legend=True,
        show_callouts=False,
        show_d_comparison=True,
        ylim_padding=0.245,
    )
    draw_quality(
        qrecc_ance_axis,
        qrecc_quality,
        qrecc_comparisons,
        dataset="QReCC Test (dataset OOD)",
        n=8_209,
        backend="ANCE",
        backend_label="ANCE (retriever OOD)",
        show_legend=False,
        show_callouts=True,
        show_d_comparison=True,
        ylim_padding=0.245,
    )
    qrecc_figure.supxlabel(
        "Error bars: pointwise query-bootstrap 95% CI",
        x=0.25,
        fontsize=8.5,
    )

    def build_metric_figure(metric: str, title: str):
        metric_figure, metric_axes = plt.subplots(
            2,
            2,
            figsize=(15.8, 10.4),
            sharex=True,
            sharey="row",
            constrained_layout=False,
        )
        metric_figure.subplots_adjust(
            left=0.055,
            right=0.995,
            bottom=0.080,
            top=0.975,
            wspace=0.025,
            hspace=0.105,
        )
        specs = (
            (
                metric_axes[0, 0], quality, comparisons,
                "TopiOCQA Dev", 2_514, "BM25", None, True,
            ),
            (
                metric_axes[0, 1], quality, comparisons,
                "TopiOCQA Dev", 2_514, "ANCE",
                "ANCE (retriever OOD)", False,
            ),
            (
                metric_axes[1, 0], qrecc_quality, qrecc_comparisons,
                "QReCC Test (dataset OOD)", 8_209, "BM25", None, False,
            ),
            (
                metric_axes[1, 1], qrecc_quality, qrecc_comparisons,
                "QReCC Test (dataset OOD)", 8_209, "ANCE",
                "ANCE (retriever OOD)", False,
            ),
        )
        for (
            axis,
            metric_data,
            metric_comparisons,
            dataset,
            n,
            backend,
            backend_label,
            show_legend,
        ) in specs:
            draw_metric_comparison(
                axis,
                metric_data,
                metric_comparisons,
                metric=metric,
                dataset=dataset,
                n=n,
                backend=backend,
                backend_label=backend_label,
                show_legend=show_legend,
            )
        metric_axes[0, 1].set_ylabel("")
        metric_axes[1, 1].set_ylabel("")
        metric_figure.supxlabel(
            "Error bars: pointwise query-bootstrap 95% CI.\n"
            "Paired bootstrap versus I-B512: † 95% CI excludes 0; "
            "‡ 95% CI includes 0 (statistically indistinguishable)",
            fontsize=9,
        )
        return metric_figure

    ndcg_figure = build_metric_figure(
        "nDCG@3", "ROCC nDCG@3 Comparison"
    )
    mrr_figure = build_metric_figure("MRR", "ROCC MRR Comparison")

    recall_figure, recall_axes = plt.subplots(
        2,
        2,
        figsize=(15.8, 10.4),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    recall_figure.subplots_adjust(
        left=0.055,
        right=0.995,
        bottom=0.135,
        top=0.975,
        wspace=0.025,
        hspace=0.180,
    )
    recall_specs = (
        (
            recall_axes[0, 0], quality, comparisons, "TopiOCQA Dev", 2_514,
            "BM25", None, True, True,
        ),
        (
            recall_axes[0, 1],
            quality,
            comparisons,
            "TopiOCQA Dev",
            2_514,
            "ANCE",
            "ANCE (retriever OOD)",
            False,
            True,
        ),
        (
            recall_axes[1, 0],
            qrecc_quality,
            qrecc_comparisons,
            "QReCC Test (dataset OOD)",
            8_209,
            "BM25",
            None,
            False,
            True,
        ),
        (
            recall_axes[1, 1],
            qrecc_quality,
            qrecc_comparisons,
            "QReCC Test (dataset OOD)",
            8_209,
            "ANCE",
            "ANCE (retriever OOD)",
            False,
            True,
        ),
    )
    for (
        axis,
        recall_data,
        recall_comparisons,
        dataset,
        n,
        backend,
        backend_label,
        show_legend,
        bold_baseline_values,
    ) in recall_specs:
        draw_recall(
            axis,
            recall_data,
            recall_comparisons,
            dataset=dataset,
            n=n,
            backend=backend,
            backend_label=backend_label,
            show_legend=show_legend,
            bold_baseline_values=bold_baseline_values,
        )
    recall_axes[0, 1].set_ylabel("")
    recall_axes[1, 1].set_ylabel("")
    recall_figure.supxlabel(
        "Recall cutoff\nError bars: pointwise query-bootstrap 95% CI.\n"
        "Paired bootstrap versus I-B512: "
        "† 95% CI excludes 0; ‡ 95% CI includes 0 "
        "(statistically indistinguishable)",
        fontsize=9,
    )

    projection_figure = plt.figure(
        figsize=(15.8, 10.4),
        constrained_layout=False,
    )
    projection_figure.add_axes((0.0, 0.0, 1.0, 1.0)).axis("off")
    projection_mrr_axis = projection_figure.add_axes(
        (0.045, 0.555, 0.205, 0.350)
    )
    projection_length_axis = projection_figure.add_axes(
        (0.315, 0.555, 0.180, 0.350)
    )
    draw_primary_quality_compact(
        projection_mrr_axis,
        quality,
        granularity_bars,
        granularity_baseline_comparisons,
    )
    projection_mrr_axis.yaxis.set_label_coords(
        0.01389,
        0.730,
        transform=projection_figure.transFigure,
    )
    draw_granularity_length_panel(
        projection_length_axis,
        granularity_lengths,
        granularity_length_intervals,
        granularity_baseline_length_interval,
    )
    dqcis_figure = plt.figure(
        figsize=(15.8, 10.4),
        constrained_layout=False,
    )
    dqcis_figure.add_axes((0.0, 0.0, 1.0, 1.0)).axis("off")
    dqcis_axis = dqcis_figure.add_axes((0.045, 0.555, 0.450, 0.350))
    draw_dqcis_fusion_ablation(
        dqcis_axis,
        dqcis_intervals,
        dqcis_comparisons,
    )
    dqcis_axis.yaxis.set_label_coords(
        0.01389,
        0.730,
        transform=dqcis_figure.transFigure,
    )
    teacher_student_figure = plt.figure(
        figsize=(15.8, 10.4),
        constrained_layout=False,
    )
    teacher_student_figure.add_axes((0.0, 0.0, 1.0, 1.0)).axis("off")
    teacher_panel_left = 0.045
    teacher_panel_gap = 0.040
    teacher_panel_width = 0.205
    teacher_panel_height = 0.125
    teacher_panel_right = (
        teacher_panel_left + teacher_panel_width + teacher_panel_gap
    )
    teacher_student_mrr_axis = teacher_student_figure.add_axes(
        (
            teacher_panel_left,
            0.780,
            teacher_panel_width,
            teacher_panel_height,
        )
    )
    teacher_student_f1_axis = teacher_student_figure.add_axes(
        (
            teacher_panel_right,
            0.780,
            teacher_panel_width,
            teacher_panel_height,
        )
    )
    teacher_taxonomy_axis = teacher_student_figure.add_axes(
        (
            teacher_panel_left,
            0.550,
            teacher_panel_width,
            teacher_panel_height,
        )
    )
    teacher_collapsed_axis = teacher_student_figure.add_axes(
        (
            teacher_panel_right,
            0.550,
            teacher_panel_width,
            teacher_panel_height,
        )
    )
    draw_teacher_student_mrr(
        teacher_student_mrr_axis,
        teacher_student_mrr,
        teacher_student_comparisons,
    )
    teacher_student_mrr_axis.yaxis.set_label_coords(
        0.01389,
        0.8425,
        transform=teacher_student_figure.transFigure,
    )
    draw_teacher_student_f1(
        teacher_student_f1_axis,
        teacher_student_f1,
        teacher_student_f1_comparisons,
    )
    teacher_student_f1_axis.set_ylim(0.0, 1.80)
    draw_teacher_label_distribution(
        teacher_taxonomy_axis,
        teacher_label_distribution,
        label_space="full_taxonomy",
        title="TopiOCQA Dev, n=2,104\nFull Taxonomy",
    )
    draw_teacher_label_distribution(
        teacher_collapsed_axis,
        teacher_label_distribution,
        label_space="collapsed_bio",
        title="TopiOCQA Dev, n=2,104\nCollapsed BIO (KEEP/DROP)",
    )
    teacher_collapsed_axis.set_ylabel("")
    structure_figure, structure_axes = plt.subplots(
        3,
        2,
        figsize=(15.8, 10.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": (0.90, 1.0, 1.0)},
    )
    draw_structure_distribution(
        structure_axes[0, 0],
        dataset_structure,
        dataset="TopiOCQA Dev",
    )
    draw_structure_distribution(
        structure_axes[0, 1],
        dataset_structure,
        dataset="QReCC Test (dataset OOD)",
    )
    draw_structure_performance(
        structure_axes[1, 0],
        r_route_structure,
        dataset="TopiOCQA Dev",
        backend="BM25",
    )
    draw_structure_performance(
        structure_axes[1, 1],
        r_route_structure,
        dataset="QReCC Test (dataset OOD)",
        backend="BM25",
    )
    draw_structure_performance(
        structure_axes[2, 0],
        r_route_structure,
        dataset="TopiOCQA Dev",
        backend="ANCE",
    )
    draw_structure_performance(
        structure_axes[2, 1],
        r_route_structure,
        dataset="QReCC Test (dataset OOD)",
        backend="ANCE",
    )
    structure_figure.supxlabel(
        "Systems: IterCQR I-B64, IterCQR I-B512 and frozen pretrained "
        "ROCC R-B64; "
        "error bars: pointwise query-bootstrap 95% CI"
    )
    oracle_figure = plt.figure(
        figsize=(15.8, 10.4),
        constrained_layout=False,
    )
    oracle_figure.add_axes((0.0, 0.0, 1.0, 1.0)).axis("off")
    oracle_grid = oracle_figure.add_gridspec(
        3,
        2,
        left=0.045,
        right=0.495,
        bottom=0.145,
        top=0.900,
        wspace=0.34,
        hspace=0.62,
    )
    oracle_axes = np.asarray(
        [
            [oracle_figure.add_subplot(oracle_grid[row, column])
             for column in range(2)]
            for row in range(3)
        ],
        dtype=object,
    )
    draw_space_oracle_headroom(
        oracle_axes[0, 0], oracle_data["space_oracle"]
    )
    draw_effective_mask_outcomes(
        oracle_axes[0, 1], oracle_data["control_relation"]
    )
    draw_generator_oracle_mrr(
        oracle_axes[1, 0], oracle_data["generator_plot"]
    )
    draw_generator_excess(
        oracle_axes[1, 1], oracle_data["generator_plot"]
    )
    draw_mask_collapse(oracle_axes[2, 0], oracle_data["mask_flow"])
    draw_entity_anchor_diagnostic(
        oracle_axes[2, 1], oracle_data["entity_anchor"]
    )
    oracle_figure.supxlabel(
        "Train600: informed, conversation-disjoint, depth-balanced "
        "TopiOCQA Train sample.\n"
        "Generator oracle: retrospective best-of-K; excess: observed minus "
        "matched-random headroom.\n"
        "† Holm-adjusted p<.05; one-sided Monte Carlo uses the "
        "Phipson-Smyth correction.",
        x=0.270,
        y=0.035,
        fontsize=8.5,
    )
    pdf_figures = (
        budget_mrr_figure,
        efficiency_figure,
        context_length_figure,
        ndcg_figure,
        mrr_figure,
        recall_figure,
        projection_figure,
        dqcis_figure,
        teacher_student_figure,
        structure_figure,
        oracle_figure,
    )
    for pdf_figure in pdf_figures:
        if pdf_figure is recall_figure:
            continue
        enlarged_embedded_figure = pdf_figure in {
            budget_mrr_figure,
            efficiency_figure,
            context_length_figure,
            ndcg_figure,
            mrr_figure,
            structure_figure,
        }
        standardize_figure_typography(
            pdf_figure,
            scale=(
                STRUCTURE_FIGURE_FONT_SCALE
                if enlarged_embedded_figure
                else 1.0
            ),
            panel_titles_as_body=enlarged_embedded_figure,
        )
    enlarge_recall_outer_typography(recall_figure)
    budget_mrr_figure.canvas.draw()
    budget_mrr_figure.set_layout_engine(None)
    for current_axis, budget_axis, depth_axis in zip(
        current_query_axes,
        budget_mrr_axes,
        depth_mrr_axes,
        strict=True,
    ):
        target = depth_axis.get_position()
        current_source = current_axis.get_position()
        current_axis.set_position(
            (
                current_source.x0,
                target.y0,
                current_source.width,
                target.height,
            )
        )
        budget_source = budget_axis.get_position()
        budget_left_expansion = 0.018
        budget_axis.set_position(
            (
                budget_source.x0 - budget_left_expansion,
                target.y0,
                budget_source.width + budget_left_expansion,
                target.height,
            )
        )
    budget_mrr_figure.canvas.draw()
    current_geometry = np.asarray(
        [
            (axis.get_position().x0, axis.get_position().width)
            for axis in current_query_axes
        ],
        dtype=float,
    )
    budget_geometry = np.asarray(
        [
            (axis.get_position().x0, axis.get_position().width)
            for axis in budget_mrr_axes
        ],
        dtype=float,
    )
    column_gaps = budget_geometry[:, 0] - (
        current_geometry[:, 0] + current_geometry[:, 1]
    )
    current_vertical_geometry = np.asarray(
        [
            (axis.get_position().y0, axis.get_position().height)
            for axis in current_query_axes
        ],
        dtype=float,
    )
    budget_vertical_geometry = np.asarray(
        [
            (axis.get_position().y0, axis.get_position().height)
            for axis in budget_mrr_axes
        ],
        dtype=float,
    )
    depth_vertical_geometry = np.asarray(
        [
            (axis.get_position().y0, axis.get_position().height)
            for axis in depth_mrr_axes
        ],
        dtype=float,
    )
    if not (
        np.allclose(current_geometry, current_geometry[0], atol=1e-12)
        and np.allclose(budget_geometry, budget_geometry[0], atol=1e-12)
        and np.allclose(column_gaps, column_gaps[0], atol=1e-12)
    ):
        raise RuntimeError("Budget-MRR horizontal panel geometry drifted.")
    if not (
        np.allclose(
            current_vertical_geometry,
            depth_vertical_geometry,
            atol=1e-12,
        )
        and np.allclose(
            budget_vertical_geometry,
            depth_vertical_geometry,
            atol=1e-12,
        )
    ):
        raise RuntimeError("Budget/depth MRR panel frames are not aligned.")
    for (
        current_axis,
        _,
        _,
        dataset,
        n,
        backend,
        backend_label,
        _,
    ) in budget_mrr_specs:
        bounds = current_axis.get_position()
        row_title = budget_mrr_figure.text(
            bounds.x0,
            bounds.y1 + 0.004,
            f"{dataset}, {backend_label or backend}, n={n:,}",
            ha="left",
            va="bottom",
            fontsize=FONT_BODY * STRUCTURE_FIGURE_FONT_SCALE,
        )
        row_title.set_in_layout(False)
        row_titles.append(row_title)
    expected_tick_size = FONT_TICK * STRUCTURE_FIGURE_FONT_SCALE
    for axis in (*current_query_axes, *budget_mrr_axes, *depth_mrr_axes):
        visible_tick_labels = [
            label
            for label in (*axis.get_xticklabels(), *axis.get_yticklabels())
            if label.get_visible() and label.get_text().strip()
        ]
        if not all(
            np.isclose(label.get_fontsize(), expected_tick_size)
            for label in visible_tick_labels
        ):
            raise RuntimeError("Budget-page tick typography drifted.")
    expected_panel_title_size = FONT_BODY * STRUCTURE_FIGURE_FONT_SCALE
    panel_titles = (
        *row_titles,
        *(axis._left_title for axis in depth_mrr_axes),
    )
    panel_title_sizes = [title.get_fontsize() for title in panel_titles]
    if not all(
        np.isclose(size, expected_panel_title_size)
        for size in panel_title_sizes
    ):
        raise RuntimeError(
            "Budget-page panel-title typography drifted: "
            f"{panel_title_sizes}."
        )
    with PdfPages(output_pdf) as pdf:
        for pdf_figure in pdf_figures:
            pdf.savefig(pdf_figure)
    plt.close(figure)
    plt.close(budget_mrr_figure)
    plt.close(efficiency_figure)
    plt.close(context_length_figure)
    plt.close(qrecc_figure)
    plt.close(ndcg_figure)
    plt.close(mrr_figure)
    plt.close(recall_figure)
    plt.close(projection_figure)
    plt.close(dqcis_figure)
    plt.close(teacher_student_figure)
    plt.close(structure_figure)
    plt.close(oracle_figure)

    sources = {
        "latency_components": (
            LATENCY_DIR / "reproducibility_route_components_summary.csv"
        ),
        "latency_component_runs": LATENCY_RUNS_SOURCE,
        "latency_vs_i512_paired_t_intervals": (
            LATENCY_COMPARISON_EXPORT
        ),
        "bm25_quality": BM25_DIR / "route_summary.csv",
        "bm25_quality_comparisons": BM25_DIR / "route_metrics_by_query.csv",
        "ance_quality": ANCE_DIR / "route_summary.csv",
        "ance_quality_comparisons": ANCE_DIR / "route_metrics_by_query.csv",
        "qrecc_bm25_quality": QRECC_DIR / "bm25/route_summary.csv",
        "qrecc_bm25_quality_comparisons": (
            QRECC_DIR / "bm25/route_metrics_by_query.csv"
        ),
        "qrecc_ance_quality": QRECC_DIR / "ance/route_summary.csv",
        "qrecc_ance_quality_comparisons": (
            QRECC_DIR / "ance/route_metrics_by_query.csv"
        ),
        "topiocqa_dev_metadata": TOPIOCQA_DEV_SOURCE,
        "qrecc_test_metadata": QRECC_TEST_METADATA_SOURCE,
        "dataset_structure_plot_data": DATASET_STRUCTURE_EXPORT,
        "structure_quality_plot_data": STRUCTURE_QUALITY_EXPORT,
        "nb03_oracle_manifest": ORACLE_MANIFEST_SOURCE,
        **{
            f"nb03_oracle_{name}": path
            for name, path in ORACLE_SOURCES.items()
        },
        "topiocqa_current_query_only_bundle_manifest": (
            CURRENT_QUERY_ONLY_DIRS["TopiOCQA Dev"]
            / "query_bundle/manifest.json"
        ),
        "topiocqa_current_query_only_bm25_manifest": (
            CURRENT_QUERY_ONLY_DIRS["TopiOCQA Dev"]
            / "bm25/final_manifest.json"
        ),
        "topiocqa_current_query_only_bm25_metrics": (
            CURRENT_QUERY_ONLY_DIRS["TopiOCQA Dev"]
            / "bm25/metrics_by_query.csv"
        ),
        "topiocqa_current_query_only_bm25_summary": (
            CURRENT_QUERY_ONLY_DIRS["TopiOCQA Dev"]
            / "bm25/summary.csv"
        ),
        "topiocqa_current_query_only_ance_manifest": (
            CURRENT_QUERY_ONLY_DIRS["TopiOCQA Dev"]
            / "ance/final_manifest.json"
        ),
        "topiocqa_current_query_only_ance_metrics": (
            CURRENT_QUERY_ONLY_DIRS["TopiOCQA Dev"]
            / "ance/metrics_by_query.csv"
        ),
        "topiocqa_current_query_only_ance_summary": (
            CURRENT_QUERY_ONLY_DIRS["TopiOCQA Dev"]
            / "ance/summary.csv"
        ),
        "qrecc_current_query_only_bundle_manifest": (
            CURRENT_QUERY_ONLY_DIRS["QReCC Test (dataset OOD)"]
            / "query_bundle/manifest.json"
        ),
        "qrecc_current_query_only_bm25_manifest": (
            CURRENT_QUERY_ONLY_DIRS["QReCC Test (dataset OOD)"]
            / "bm25/final_manifest.json"
        ),
        "qrecc_current_query_only_bm25_metrics": (
            CURRENT_QUERY_ONLY_DIRS["QReCC Test (dataset OOD)"]
            / "bm25/metrics_by_query.csv"
        ),
        "qrecc_current_query_only_bm25_summary": (
            CURRENT_QUERY_ONLY_DIRS["QReCC Test (dataset OOD)"]
            / "bm25/summary.csv"
        ),
        "qrecc_current_query_only_ance_manifest": (
            CURRENT_QUERY_ONLY_DIRS["QReCC Test (dataset OOD)"]
            / "ance/final_manifest.json"
        ),
        "qrecc_current_query_only_ance_metrics": (
            CURRENT_QUERY_ONLY_DIRS["QReCC Test (dataset OOD)"]
            / "ance/metrics_by_query.csv"
        ),
        "qrecc_current_query_only_ance_summary": (
            CURRENT_QUERY_ONLY_DIRS["QReCC Test (dataset OOD)"]
            / "ance/summary.csv"
        ),
        "current_query_only_mrr_bootstrap_cis": (
            CURRENT_QUERY_ONLY_CI_SOURCE
        ),
        "encoder_resources": NSIGHT_DIR / "topiocqa_t5_encoder_summary.csv",
        "encoder_component_resources": (
            NSIGHT_DIR / "topiocqa_t5_encoder_component_metrics.csv"
        ),
        "topiocqa_context_inputs": TOPIOCQA_CONTEXT_SOURCE,
        "qrecc_context_inputs": QRECC_CONTEXT_SOURCE,
        "granularity_projection_comparisons": (
            GRANULARITY_COMPARISON_SOURCE
        ),
        "granularity_projection_input_lengths": (
            GRANULARITY_LENGTH_SOURCE
        ),
        "granularity_projection_metrics_by_query": (
            GRANULARITY_METRICS_SOURCE
        ),
        "granularity_projection_summary": GRANULARITY_SUMMARY_SOURCE,
        "granularity_projection_bar_bootstrap_cis": (
            GRANULARITY_BAR_CI_SOURCE
        ),
        "granularity_projection_vs_i512_comparisons": (
            GRANULARITY_BASELINE_COMPARISON_SOURCE
        ),
        "granularity_projection_length_bootstrap_cis": (
            GRANULARITY_LENGTH_CI_SOURCE
        ),
        "granularity_projection_baseline_length_bootstrap_ci": (
            GRANULARITY_BASELINE_LENGTH_CI_SOURCE
        ),
        "granularity_projection_manifest": (
            GRANULARITY_PROJECTION_DIR / "manifest.json"
        ),
        "dqcis_fusion_summary": DQCIS_SUMMARY_SOURCE,
        "dqcis_fusion_metrics_by_query": DQCIS_METRICS_SOURCE,
        "dqcis_fusion_paired_comparisons": DQCIS_COMPARISONS_SOURCE,
        "dqcis_fusion_manifest": DQCIS_MANIFEST_SOURCE,
        "dqcis_fusion_bar_bootstrap_cis": DQCIS_BAR_CI_SOURCE,
        "topiocqa_teacher_student_mrr": TEACHER_STUDENT_MRR_SOURCE,
        "topiocqa_teacher_student_mrr_comparisons": (
            TEACHER_STUDENT_COMPARISON_SOURCE
        ),
        "topiocqa_teacher_student_f1": TEACHER_STUDENT_F1_SOURCE,
        "topiocqa_teacher_student_f1_query_counts": (
            TEACHER_STUDENT_AGREEMENT_COUNTS_SOURCE
        ),
        "topiocqa_teacher_labels": TEACHER_LABELS_SOURCE,
        "topiocqa_teacher_label_distribution": (
            TEACHER_LABEL_DISTRIBUTION_EXPORT
        ),
        "topiocqa_teacher_evaluable_population": (
            TEACHER_STUDENT_POPULATION_SOURCE
        ),
        "topiocqa_teacher_evaluation_manifest": (
            TEACHER_STUDENT_MANIFEST_SOURCE
        ),
        "teacher_student_mrr_plot_data": TEACHER_STUDENT_MRR_EXPORT,
        "teacher_student_f1_plot_data": TEACHER_STUDENT_F1_EXPORT,
        "teacher_student_f1_paired_comparisons": (
            TEACHER_STUDENT_F1_COMPARISON_EXPORT
        ),
        **{
            f"{dataset}_{backend.lower()}_saturation": path
            for dataset, backend_paths in SATURATION_SOURCES.items()
            for backend, path in backend_paths.items()
        },
        "quality_bar_bootstrap_cis": bootstrap_ci_path,
        "depth_strata_mrr_bootstrap_cis": depth_strata_path,
    }
    manifest_path = FIGURE_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 90,
                "bar_spacing_definition": (
                    "every bar chart uses the same central geometry: "
                    f"adjacent bars are separated by {BAR_GAP_RATIO:.0%} "
                    "of one bar width and distinct method, budget or metric "
                    "groups by "
                    f"{BAR_GROUP_GAP_RATIO:.0%} of one bar width; the "
                    "IterCQR-versus-ROCC family gap in the MRR, nDCG@3, "
                    "compute and DRAM comparison panels is "
                    f"{COMPARISON_FAMILY_GAP_MULTIPLIER:.1f} times this "
                    "base group gap"
                ),
                "pdf_page_geometry": (
                    "fixed 15.8 x 10.4 inch canvas for every page; no "
                    "content-dependent tight bounding box"
                ),
                "bootstrap_errorbar_style": (
                    "on dark bar fills, the interval segment and cap inside "
                    "the bar are white while the segment and cap outside "
                    "the bar remain dark; signed bars are handled by their "
                    "direction"
                ),
                "typography_definition": (
                    f"{PLOT_FONT} with one enforced role-based contract on every "
                    "page: page titles 18 pt; panel titles, axis labels and "
                    "tick labels 13.5 pt; legends, annotations and notes "
                    "11.5 pt. All local legacy fontsize values are overridden "
                    "before PNG and PDF export"
                ),
                "teacher_student_color_definition": (
                    "Teacher variants use light blue; both Student variants "
                    "use the same medium blue in the retrieval and agreement "
                    "panels; all Teacher-label distribution bars use gray"
                ),
                "teacher_label_distribution_definition": (
                    "the same 2104 held-out Teacher-labeled histories are "
                    "tokenized with the pinned MiniLM-L12-v2 revision, "
                    "max_length=512 and recent-first history; the left panel "
                    "shows the six-class Teacher taxonomy and the right panel "
                    "the exact O/B-KEEP/I-KEEP projection used by the "
                    "collapsed selector"
                ),
                "dataset_structure_definition": (
                    "TopiOCQA is grouped at query level by the cumulative "
                    "number of sequential topic_switch events encountered "
                    "within each Dev conversation; QReCC Test is grouped by "
                    "its recorded NQ, QuAC or TREC conversation source. "
                    "Performance compares IterCQR I-B64 and I-B512 with "
                    "the frozen pretrained ROCC R-B64 route separately for "
                    "BM25 and ANCE using pointwise query-bootstrap 95% CIs"
                ),
                "bar_annotation_headroom_definition": (
                    "after typography normalization, every bar axis is "
                    "expanded when necessary so its highest data-space "
                    "annotation occupies at most "
                    f"{BAR_ANNOTATION_MAX_AXES_Y:.0%} of the vertical axis; "
                    "font sizes remain unchanged. The two compact "
                    "granularity panels reserve separate legend, preference-"
                    "hint and data-annotation bands"
                ),
                "display_decimal_policy": (
                    "all reader-visible plot annotations use at most three "
                    "digits after the decimal point; stored source and "
                    "derived CSV values retain full precision; budget-MRR "
                    "trajectory y-axis ticks use two decimal places"
                ),
                "latency_definition": (
                    "component-wise synchronized wall clocks; route total is "
                    "the paired sum; mean and two-sided 95% Student-t interval "
                    "across five independent process-run means (df=4); each "
                    "run mean averages five measurement passes; the dashed "
                    "reference is the I-B512 mean (0.0%), and dagger markers "
                    "use paired two-sided Student-t intervals over matching "
                    "run IDs versus I-B512"
                ),
                "latency_sample": "NB03 depth-balanced TopiOCQA train600",
                "quality_population": "TopiOCQA Dev n=2514",
                "ood_quality_population": "QReCC Test n=8209",
                "quality_bar_interval_definition": (
                    "pointwise query-level percentile bootstrap 95% CI; "
                    "10000 common resamples per dataset/backend; fixed "
                    "SHA-256-derived seeds"
                ),
                "pdf_pages": [
                    "MRR trajectories and TopiOCQA history-depth strata",
                    "Primary latency, compute and memory comparison",
                    "Serialized T5 input-length distributions",
                    "nDCG@3 comparison by dataset and backend",
                    "MRR comparison by dataset and backend",
                    "R@10/R@100/R@1000 comparison by dataset and backend",
                    "BM25 v1/v2/v3 projection effects for R-B64 quality "
                    "and R-B64 versus I-B512 input length within the "
                    "upper-left page quadrant",
                    "DQ-CIS fusion ablation within the upper-left page "
                    "quadrant",
                    "Held-out GPT-5.4 Teacher, Student, BM25 Teacher and "
                    "BM25 Student evaluation plus full-taxonomy and "
                    "collapsed-BIO Teacher-label distributions on the "
                    "uniform TopiOCQA Dev Teacher-evaluable cohort",
                    "Dataset structure and I-B64/I-B512/R-B64 retrieval "
                    "quality by TopiOCQA topic switches and QReCC source "
                    "dataset",
                    "V1 space-oracle headroom, effective-mask outcomes, "
                    "generator oracle and matched-null excess, mask-to-"
                    "input-to-rewrite collapse, and exact entity-anchor "
                    "availability from the verified NB03 artifact",
                ],
                "budget_mrr_definition": (
                    "full-population IterCQR current-query-only MRR on a "
                    "separate left sub-axis and pretrained I/R/D/IRD MRR "
                    "at B64/B128/B256/B512 on the adjacent budget sub-axis; "
                    "there is no connecting line across the two x-domains; "
                    "each current-query-only point and each budget panel use "
                    "pointwise query-bootstrap 95% CIs; every current-query "
                    "and budget-panel frame is vertically aligned exactly "
                    "with its adjacent history-depth panel, while the budget "
                    "trajectories and current-query-only points retain "
                    "separate focused y-scales; TopiOCQA Dev n=2514 and QReCC "
                    "Test n=8209; "
                    "the current-query-only artifacts have history=[], no "
                    "selection and complete BM25/ANCE Top-1000 retrieval; "
                    "four vertically stacked rows occupy the left page half "
                    "and the current-query-only x-label appears only on the "
                    "bottom row; the right page half compares "
                    "I-B512, I-B64, R-B64 and IRD-B64 across the fixed "
                    "history-depth strata in the same TopiOCQA-BM25, "
                    "TopiOCQA-ANCE, QReCC-BM25, QReCC-ANCE top-to-bottom "
                    "order, with pointwise query-bootstrap 95% CIs"
                ),
                "budget_mrr_horizontal_geometry": {
                    "current_query_x0": float(current_geometry[0, 0]),
                    "current_query_width": float(current_geometry[0, 1]),
                    "budget_x0": float(budget_geometry[0, 0]),
                    "budget_width": float(budget_geometry[0, 1]),
                    "column_gap": float(column_gaps[0]),
                    "identical_across_four_rows": True,
                },
                "budget_mrr_vertical_geometry": {
                    "row_y0": depth_vertical_geometry[:, 0].tolist(),
                    "row_height": depth_vertical_geometry[:, 1].tolist(),
                    "current_query_matches_depth": True,
                    "budget_matches_depth": True,
                    "budget_y_scale": "panel-focused",
                },
                "budget_mrr_typography": {
                    "tick_fontsize_points": expected_tick_size,
                    "panel_title_fontsize_points": (
                        expected_panel_title_size
                    ),
                    "identical_across_current_budget_and_depth_panels": True,
                },
                "context_length_definition": (
                    "absolute query counts in one-token bins for I-B512, "
                    "I-B64 and pretrained R-B512; TopiOCQA Dev n=2514 and "
                    "QReCC Test n=8209; the lower-frequency inset excludes "
                    "the I-B64 token-64 peak only from its y-axis scaling; "
                    "the shaded ROCC tail and annotation count inputs with "
                    "more than 64 tokens; the same annotation reports the "
                    "stored paired R B128-minus-B64 MRR difference and "
                    "95% bootstrap CI for BM25 and ANCE"
                ),
                "granularity_projection_definition": (
                    "TopiOCQA Dev n=2514, pretrained selector and BM25; "
                    "only the upper-left page quadrant is used and is split "
                    "into two side-by-side panels; the left panel shows "
                    "same-grey I-B512 and I-B64 bars plus blue v1 turn, "
                    "v2 Q/A and v3 token/span sub-bars for R-B64, all with "
                    "pointwise 10000-replicate query-bootstrap 95% CIs; "
                    "matching background fields and brackets group the "
                    "first two bars as IterCQR and the projection bars as "
                    "ROCC B64; "
                    "the right panel shows full-population R-B64-minus-"
                    "I-B512 T5 input-token and percentage changes for the "
                    "same three projections with paired 10000-replicate "
                    "query-bootstrap 95% CIs for the token differences; "
                    "floating bars start at the absolute I-B512 mean of "
                    "150.1 tokens and end at each absolute R-B64 mean, so "
                    "the y-axis remains an absolute token-length scale; "
                    "a shaded band and exact label show the independently "
                    "bootstrapped pointwise 95% CI of the I-B512 mean; "
                    "†/‡ markers report whether each paired B64-minus-"
                    "I-B512 95% CI excludes/includes zero; "
                    "the other three page quadrants are blank"
                ),
                "dqcis_fusion_definition": (
                    "TopiOCQA Dev n=2514; published DQ-CIS ChatGPT/"
                    "ColBERTv2 Top-100 ranking compared with unweighted "
                    "RRF10 fusion of that ranking and exactly one frozen "
                    "Top-100 ANCE ranking: either full-history IterCQR I "
                    "or ROCC-selected IterCQR R at B64; the plot therefore "
                    "tests fusion complementarity rather than replacing "
                    "the DQ-CIS system; bars use pointwise 10000-replicate "
                    "query-bootstrap 95% CIs and the displayed DQ+R-minus-"
                    "DQ+I interval is the stored paired NB08 query-level "
                    "bootstrap; †/‡ markers compare DQ+I and DQ+R with the "
                    "dashed DQ-CIS single-ranking reference; only the "
                    "upper-left page quadrant is used"
                ),
                "teacher_student_definition": (
                    "TopiOCQA Dev n=2104 for every retrieval MRR and F1 "
                    "series: only the depth>=2 queries with available "
                    "Teacher labels are evaluated; all 205 no-history and "
                    "all 205 depth=1 queries are excluded from both panels. "
                    "The MRR and F1 inputs therefore contain exactly the "
                    "same sample IDs. GPT-5.4 Teacher and "
                    "the Student use the original Teacher selection; BM25 "
                    "Teacher applies the frozen privileged correction rule "
                    "Teacher_KEEP OR (gold-BM25 score_raw>0 and non-"
                    "stopword), and BM25 Student is the frozen E6 treatment "
                    "checkpoint at B64. Retrieval uses R/Viterbi, IterCQR "
                    "and BM25. The figure shows only the four B64 MRR "
                    "endpoints as bars beside the F1 panel in the upper-"
                    "left page quadrant; the other three quadrants are "
                    "blank. KEEP F1 and exact BIO-span F1 use this same "
                    "cohort n=2104, with each Student "
                    "compared against its corresponding Teacher. Each of "
                    "the four bar pairs includes a dashed reference at the "
                    "left-bar value; † marks a paired 95% bootstrap CI that "
                    "excludes zero and ‡ one that includes zero. MRR uses "
                    "the stored paired query bootstrap, while KEEP and "
                    "Span F1 differences use 10000 common query resamples "
                    "of the stored TP/FP/FN counts. No Train600 or QReCC "
                    "values are shown"
                ),
                "oracle_headroom_definition": (
                    "Verified NB03 v2 artifact; every source CSV is hashed "
                    "against its complete manifest before plotting. Space-"
                    "oracle, effective-mask and generator panels use only "
                    "the informed, conversation-disjoint, depth-balanced "
                    "TopiOCQA Train sample n=600. Exhaustive masks cover "
                    "depth<=8 (n=300); deeper histories use the "
                    "cardinality-stratified approximation (n=300). Mask "
                    "outcome shares are computed within query after "
                    "input-key deduplication and then averaged over 600 "
                    "queries; displayed counts are the corresponding raw "
                    "effective-input totals. Generator oracle MRR equals "
                    "the per-budget Recency MRR plus the stored matched "
                    "headroom and is a retrospective best-of-K upper "
                    "bound. Excess subtracts cardinality- and opportunity-"
                    "matched random-mask headroom; dagger markers denote "
                    "Holm-adjusted p<.05 from one-sided Monte Carlo tests "
                    "with the Phipson-Smyth +1 correction. Mask-collapse "
                    "counts are derived at runtime from control-relation "
                    "and rewrite-dedup summaries. The entity diagnostic "
                    "shows both Train600 and the sampling-eligible "
                    "TopiOCQA Train population n="
                    f"{int(oracle_data['entity_anchor']['queries'].max())}. "
                    "All six panels occupy only the left page half; the "
                    "right page half is intentionally blank"
                ),
                "encoder_resource_definition": (
                    "Nsight Compute scalar FP32 operation counts split "
                    "directly by encoder component; range-replay "
                    "whole-forward DRAM total allocated across components "
                    "in proportion to the "
                    "cold-cache kernel-replay dram__bytes.sum shares; one "
                    "B16 T5-encoder forward; the negligible embedding share "
                    "is folded into Other; R reductions "
                    "are paired against I within each budget"
                ),
                "sources": {
                    name: {
                        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                        "sha256": sha256(path),
                    }
                    for name, path in sources.items()
                },
                "outputs": [str(output_png), str(output_pdf)],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Plot: {output_png}")
    print(f"PDF:  {output_pdf}")


if __name__ == "__main__":
    arguments = parse_args()
    configure_paths(arguments)
    check_inputs()
    if not arguments.check_inputs:
        main()
