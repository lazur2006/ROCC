"""Train-member recall, direct retrieval, and fixed selector fusion APIs."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cached_itercqr import (
    CachedIterCQRBM25Config,
    CachedIterCQRBM25Pipeline,
)
from .evaluation import (
    paired_bootstrap_mean_ci,
    reciprocal_rank_fusion,
    retrieval_metrics,
)
from .itercqr_components import load_itercqr_tokenizer
from .progress import get_tqdm
from .retrievers import (
    load_bm25_retriever,
    normalize_retrieval_scores,
)
from .teacher_evaluation import TEACHER_METRIC_COLUMNS


FUSION_METHOD_ORDER = ("minmax_sum", "rrf10", "rrf60")
FUSION_BOOTSTRAP_SEEDS = {
    "minmax_sum_minus_rrf10": 20,
    "minmax_sum_minus_rrf60": 21,
    "rrf10_minus_rrf60": 22,
    "minmax_sum_minus_best_single": 23,
    "rrf10_minus_best_single": 24,
    "rrf60_minus_best_single": 25,
}
RECALL_BOOTSTRAP_SEEDS = {
    "R@10": 17,
    "R@100": 18,
    "R@1000": 19,
}
VIEW_ORDER = ("I", "R", "D")


def materialize_selector_direct_queries(
    *,
    pipeline_results: pd.DataFrame,
    selector_arm: str,
    budget: int,
    config: Any,
    device: str,
    progress: bool = True,
) -> pd.DataFrame:
    """Decode the exact budgeted selector inputs into direct queries."""

    required = {
        "sample_id",
        "conv_id",
        "turn_id",
        "budget",
        "arm",
        "input_key",
        "input_length",
        "raw_input_length",
        "was_truncated",
    }
    missing = required.difference(pipeline_results.columns)
    if missing:
        raise KeyError(
            "Direct-query input columns missing: "
            + ", ".join(sorted(missing))
        )
    selected = pipeline_results.loc[
        pipeline_results["budget"].eq(int(budget))
        & pipeline_results["arm"].eq(str(selector_arm)),
        sorted(required.difference({"arm"})),
    ].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected = selected.sort_values("sample_id", kind="mergesort")
    if selected.empty or selected["sample_id"].duplicated().any():
        raise ValueError(
            "Direct-query selector rows are empty or duplicated."
        )

    tokenizer = load_itercqr_tokenizer(config.itercqr_model_dir)
    pipeline = _pipeline(
        config,
        device=device,
        tokenizer=tokenizer,
        progress=progress,
    )
    try:
        token_sequences = pipeline.load_token_inputs(
            selected["input_key"].astype(str).tolist()
        )
    finally:
        pipeline.close()
    selected["direct_query"] = [
        str(
            tokenizer.decode(
                sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        ).strip()
        for sequence in token_sequences
    ]
    return selected


def run_selector_direct_bm25(
    *,
    pipeline_results: pd.DataFrame,
    selector_arm: str,
    budget: int,
    gold_by_sample: Mapping[str, Sequence[Any]],
    config: Any,
    output_dir: Path | str,
    device: str,
    progress_prefix: str = "selector direct D",
    progress: bool = True,
) -> dict[str, Any]:
    """Decode the exact budgeted selector input and retrieve it directly."""

    selected = materialize_selector_direct_queries(
        pipeline_results=pipeline_results,
        selector_arm=selector_arm,
        budget=budget,
        config=config,
        device=device,
        progress=progress,
    )
    tokenizer = load_itercqr_tokenizer(config.itercqr_model_dir)
    pipeline = _pipeline(
        config,
        device=device,
        tokenizer=tokenizer,
        progress=progress,
    )
    try:
        result = pipeline.run_bm25(
            selected,
            gold_by_sample=gold_by_sample,
            query_column="direct_query",
            metric_columns=TEACHER_METRIC_COLUMNS,
            progress_prefix=str(progress_prefix),
        )
    finally:
        pipeline.close()

    evaluated = result.evaluated.copy()
    evaluated["arm"] = "D"
    summary = _metric_summary(evaluated)
    resolved_output = Path(output_dir)
    _write_frame(
        resolved_output / "direct_metrics_by_query.csv",
        evaluated.sort_values("sample_id", kind="mergesort"),
    )
    _write_frame(resolved_output / "direct_summary.csv", summary)
    return {
        "queries": selected,
        "evaluated": evaluated,
        "summary": summary,
        "runtime_cache_stats": {
            "unique_queries": result.unique_queries,
            "retrieval_cache_misses": result.retrieval_cache_misses,
            "retriever_loaded": result.retriever_loaded,
        },
    }


def compare_target_recall(
    *,
    pipeline_results: pd.DataFrame,
    target_arm: str,
    budget: int,
    output_dir: Path | str,
    replicates: int = 10_000,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compare target-selector recall with equal-budget Recency."""

    selected = pipeline_results.loc[
        pipeline_results["budget"].eq(int(budget))
        & pipeline_results["arm"].isin([str(target_arm), "recency"])
    ].copy()
    selected["route"] = selected["arm"].map(
        {str(target_arm): "R", "recency": "I"}
    )
    if selected["route"].isna().any():
        raise ValueError("Recall comparison contains unknown routes.")
    summary = (
        selected.groupby("route", observed=True, sort=True)
        .agg(
            n=("sample_id", "size"),
            **{
                metric: (metric, "mean")
                for metric in ("R@10", "R@100", "R@1000")
            },
        )
        .reset_index()
    )
    comparisons = pd.DataFrame(
        [
            _paired_metric(
                selected.rename(columns={"route": "comparison_arm"}),
                arm_column="comparison_arm",
                left_arm="R",
                right_arm="I",
                metric=metric,
                budget=int(budget),
                seed=RECALL_BOOTSTRAP_SEEDS[metric],
                replicates=int(replicates),
                progress=progress,
            )
            for metric in ("R@10", "R@100", "R@1000")
        ]
    )
    resolved_output = Path(output_dir)
    _write_frame(resolved_output / "summary.csv", summary)
    _write_frame(
        resolved_output / "paired_comparisons.csv",
        comparisons,
    )
    return {"summary": summary, "comparisons": comparisons}


def compute_union_recall(
    *,
    pipeline_results: pd.DataFrame,
    target_arm: str,
    budget: int,
    gold_by_sample: Mapping[str, Sequence[Any]],
    config: Any,
    output_dir: Path | str,
    device: str,
    vehicle_sample_id_sha256: str,
    evaluation_scope: str = "train_member",
    cutoffs: Sequence[int] = (10, 100, 1000),
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Measure I/R union recall and mutually exclusive gold hits."""

    paired = _paired_ranking_keys(
        pipeline_results,
        arm_to_view={str(target_arm): "R", "recency": "I"},
        budget=int(budget),
    )
    if _sample_id_sha256(paired.index.astype(str)) != str(
        vehicle_sample_id_sha256
    ):
        raise ValueError("Union vehicle hash does not match the NB03 gate.")
    pipeline = _pipeline(
        config,
        device=device,
        tokenizer=None,
        progress=progress,
    )
    try:
        rankings = pipeline.load_rankings(
            paired.to_numpy().ravel(),
            progress_prefix=f"{evaluation_scope} I/R union",
        )
    finally:
        pipeline.close()

    rows: list[dict[str, Any]] = []
    tqdm = get_tqdm()
    for sample_id, ranking_row in tqdm(
        paired.iterrows(),
        total=len(paired),
        desc=f"{evaluation_scope} I/R union recall",
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    ):
        gold = {str(value) for value in gold_by_sample[str(sample_id)]}
        if not gold:
            raise ValueError(f"Gold is empty for {sample_id}.")
        ranking_i = rankings[str(ranking_row["I"])]
        ranking_r = rankings[str(ranking_row["R"])]
        for cutoff in map(int, cutoffs):
            docs_i = set(ranking_i[:cutoff])
            docs_r = set(ranking_r[:cutoff])
            hit_i = bool(gold.intersection(docs_i))
            hit_r = bool(gold.intersection(docs_r))
            category = (
                "both"
                if hit_i and hit_r
                else "R_only"
                if hit_r
                else "I_only"
                if hit_i
                else "neither"
            )
            rows.append(
                {
                    "sample_id": str(sample_id),
                    "budget": int(budget),
                    "cutoff": cutoff,
                    "I_recall": len(gold.intersection(docs_i))
                    / len(gold),
                    "R_recall": len(gold.intersection(docs_r))
                    / len(gold),
                    "union_recall": len(
                        gold.intersection(docs_i.union(docs_r))
                    )
                    / len(gold),
                    "I_hit": hit_i,
                    "R_hit": hit_r,
                    "hit_category": category,
                }
            )

    by_query = pd.DataFrame(rows).sort_values(
        ["cutoff", "sample_id"],
        kind="mergesort",
    )
    summary_rows: list[dict[str, Any]] = []
    for cutoff, group in by_query.groupby("cutoff", sort=True):
        counts = group["hit_category"].value_counts()
        i_recall = float(group["I_recall"].mean())
        r_recall = float(group["R_recall"].mean())
        union_recall = float(group["union_recall"].mean())
        row: dict[str, Any] = {
            "budget": int(budget),
            "cutoff": int(cutoff),
            "n": int(len(group)),
            "I_recall": i_recall,
            "R_recall": r_recall,
            "union_recall": union_recall,
            "union_gain_vs_best": union_recall
            - max(i_recall, r_recall),
        }
        for category in ("both", "R_only", "I_only", "neither"):
            count = int(counts.get(category, 0))
            row[f"{category}_count"] = count
            row[f"{category}_share"] = count / len(group)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    resolved_output = Path(output_dir)
    _write_frame(resolved_output / "by_query.csv", by_query)
    _write_frame(resolved_output / "summary.csv", summary)
    _write_json(
        resolved_output / "manifest.json",
        {
            "schema_version": 1,
            "evaluation_scope": str(evaluation_scope),
            "budget": int(budget),
            "views": ["I", "R"],
            "cutoffs": list(map(int, cutoffs)),
            "vehicle_sample_id_sha256": str(
                vehicle_sample_id_sha256
            ),
            "by_query_sha256": _sha256_file(
                resolved_output / "by_query.csv"
            ),
            "summary_sha256": _sha256_file(
                resolved_output / "summary.csv"
            ),
            "complete": True,
        },
    )
    return {"by_query": by_query, "summary": summary}


def evaluate_fixed_rrf_fusion(
    *,
    pipeline_results: pd.DataFrame,
    direct_results: pd.DataFrame,
    target_arm: str,
    budget: int,
    gold_by_sample: Mapping[str, Sequence[Any]],
    config: Any,
    output_dir: Path | str,
    fusion_decision_path: Path | str,
    vehicle_sample_id_sha256: str,
    device: str,
    evaluation_scope: str = "train_member",
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate only the RRF rule already locked by Notebook 05."""

    decision_path = Path(fusion_decision_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    selected_parameters = decision.get("selected_parameters", {})
    rrf_k = int(selected_parameters.get("rrf_k", -1))
    view_weights = selected_parameters.get("view_weights", {})
    if (
        decision.get("locked") is not True
        or decision.get("selected_method") != "rrf10"
        or int(decision.get("budget", -1)) != 64
        or rrf_k != 10
        or decision.get("views") != list(VIEW_ORDER)
        or {
            str(view): float(weight)
            for view, weight in view_weights.items()
        }
        != {view: 1.0 for view in VIEW_ORDER}
        or decision.get("vehicle_sample_id_sha256")
        != str(vehicle_sample_id_sha256)
    ):
        raise ValueError(
            "Fusion decision is not the locked NB05 RRF10 rule."
        )

    iterative = pipeline_results.loc[
        pipeline_results["budget"].eq(int(budget))
        & pipeline_results["arm"].isin([str(target_arm), "recency"])
    ].copy()
    iterative["view"] = iterative["arm"].map(
        {str(target_arm): "R", "recency": "I"}
    )
    direct = direct_results.loc[
        direct_results["budget"].eq(int(budget))
    ].copy()
    direct["view"] = "D"
    views = pd.concat(
        [iterative, direct],
        ignore_index=True,
        sort=False,
    )
    required = {
        "sample_id",
        "view",
        "rewrite_key",
        *TEACHER_METRIC_COLUMNS,
    }
    missing = required.difference(views.columns)
    if missing:
        raise KeyError(
            "Fixed-fusion columns missing: "
            + ", ".join(sorted(missing))
        )
    views["sample_id"] = views["sample_id"].astype(str)
    if views.duplicated(["sample_id", "view"]).any():
        raise ValueError("Fixed-fusion views contain duplicate rows.")
    keys = views.pivot(
        index="sample_id",
        columns="view",
        values="rewrite_key",
    ).sort_index(kind="mergesort")
    if (
        tuple(sorted(keys.columns)) != tuple(sorted(VIEW_ORDER))
        or keys.isna().any().any()
    ):
        raise ValueError("I/R/D rankings are not paired completely.")
    if _sample_id_sha256(keys.index.astype(str)) != str(
        vehicle_sample_id_sha256
    ):
        raise ValueError("Fixed-fusion vehicle hash does not match NB03.")

    pipeline = _pipeline(
        config,
        device=device,
        tokenizer=None,
        progress=progress,
    )
    try:
        rankings = pipeline.load_rankings(
            keys.to_numpy().ravel(),
            progress_prefix=f"{evaluation_scope} locked RRF10",
        )
    finally:
        pipeline.close()

    rows: list[dict[str, Any]] = []
    tqdm = get_tqdm()
    for sample_id, row in tqdm(
        keys.iterrows(),
        total=len(keys),
        desc=f"{evaluation_scope} locked RRF10 I/R/D",
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    ):
        rank_lists = [
            rankings[str(row[view])] for view in VIEW_ORDER
        ]
        fused = reciprocal_rank_fusion(
            rank_lists,
            rrf_k=rrf_k,
            top_k=config.retrieval_top_k,
        )
        metrics = retrieval_metrics(
            fused,
            gold_by_sample[str(sample_id)],
            ks=config.eval_ks,
        )
        rows.append(
            {
                "sample_id": str(sample_id),
                "budget": int(budget),
                "arm": "rrf10",
                **{
                    metric: metrics[metric]
                    for metric in TEACHER_METRIC_COLUMNS
                },
            }
        )

    by_query = pd.DataFrame(rows).sort_values(
        "sample_id",
        kind="mergesort",
    )
    summary = _metric_summary(by_query)
    resolved_output = Path(output_dir)
    by_query_path = resolved_output / "rrf10_metrics_by_query.csv"
    summary_path = resolved_output / "rrf10_summary.csv"
    _write_frame(by_query_path, by_query)
    _write_frame(summary_path, summary)
    manifest = {
        "schema_version": 1,
        "evaluation_scope": str(evaluation_scope),
        "budget": int(budget),
        "method": "rrf10",
        "target_arm": str(target_arm),
        "rrf_k": rrf_k,
        "views": list(VIEW_ORDER),
        "vehicle_sample_id_sha256": str(
            vehicle_sample_id_sha256
        ),
        "fusion_decision_sha256": _sha256_file(decision_path),
        "by_query_sha256": _sha256_file(by_query_path),
        "summary_sha256": _sha256_file(summary_path),
        "complete": True,
    }
    _write_json(resolved_output / "manifest.json", manifest)
    return {
        "by_query": by_query,
        "summary": summary,
        "manifest": manifest,
    }


def evaluate_selector_fusions(
    *,
    pipeline_results: pd.DataFrame,
    direct_results: pd.DataFrame,
    target_arm: str,
    budget: int,
    gold_by_sample: Mapping[str, Sequence[Any]],
    config: Any,
    output_dir: Path | str,
    score_cache_db: Path | str,
    architecture_decision_path: Path | str | None = None,
    source_manifest_path: Path | str | None = None,
    selector_checkpoint_sha256: str,
    vehicle_sample_id_sha256: str,
    device: str,
    evaluation_scope: str = "train_member",
    lock_selection: bool = True,
    replicates: int = 10_000,
    progress: bool = True,
) -> dict[str, Any]:
    """Compare fixed three-view Min-Max, RRF10, and RRF60 fusion."""

    iterative = pipeline_results.loc[
        pipeline_results["budget"].eq(int(budget))
        & pipeline_results["arm"].isin([str(target_arm), "recency"])
    ].copy()
    iterative["view"] = iterative["arm"].map(
        {str(target_arm): "R", "recency": "I"}
    )
    direct = direct_results.loc[
        direct_results["budget"].eq(int(budget))
    ].copy()
    direct["view"] = "D"
    views = pd.concat(
        [iterative, direct],
        ignore_index=True,
        sort=False,
    )
    required = {
        "sample_id",
        "view",
        "rewrite_key",
        "rewrite_norm",
        *TEACHER_METRIC_COLUMNS,
    }
    missing = required.difference(views.columns)
    if missing:
        raise KeyError(
            "Fusion view columns missing: "
            + ", ".join(sorted(missing))
        )
    views["sample_id"] = views["sample_id"].astype(str)
    if views.duplicated(["sample_id", "view"]).any():
        raise ValueError("Fusion views contain duplicate rows.")
    keys = views.pivot(
        index="sample_id",
        columns="view",
        values="rewrite_key",
    ).sort_index(kind="mergesort")
    if (
        tuple(sorted(keys.columns)) != tuple(sorted(VIEW_ORDER))
        or keys.isna().any().any()
    ):
        raise ValueError("I/R/D rankings are not paired completely.")
    if _sample_id_sha256(keys.index.astype(str)) != str(
        vehicle_sample_id_sha256
    ):
        raise ValueError("Fusion vehicle hash does not match NB03.")

    pipeline = _pipeline(
        config,
        device=device,
        tokenizer=None,
        progress=progress,
    )
    try:
        rankings = pipeline.load_rankings(
            keys.to_numpy().ravel(),
            progress_prefix=f"{evaluation_scope} I/R/D fusion",
        )
    finally:
        pipeline.close()

    query_table = (
        views[["rewrite_key", "rewrite_norm"]]
        .drop_duplicates("rewrite_key")
        .sort_values("rewrite_key", kind="mergesort")
    )
    score_cache = BM25ScoreRankingCache(
        cache_db=score_cache_db,
        index_dir=config.bm25_index_dir,
        retrieval_workers=config.retrieval_workers,
        top_k=config.retrieval_top_k,
    )
    score_rankings, score_cache_misses = score_cache.materialize(
        query_table,
        expected_rankings=rankings,
        progress=progress,
    )

    fusion_rows: list[dict[str, Any]] = []
    tqdm = get_tqdm()
    for sample_id, row in tqdm(
        keys.iterrows(),
        total=len(keys),
        desc=f"{evaluation_scope} fixed I/R/D fusion",
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    ):
        key_by_view = {
            view: str(row[view]) for view in VIEW_ORDER
        }
        rank_lists = [
            rankings[key_by_view[view]] for view in VIEW_ORDER
        ]
        score_lists = [
            score_rankings[key_by_view[view]] for view in VIEW_ORDER
        ]
        fused_by_method = {
            "minmax_sum": _minmax_sum_fusion(
                score_lists,
                top_k=config.retrieval_top_k,
            ),
            "rrf10": reciprocal_rank_fusion(
                rank_lists,
                rrf_k=10,
                top_k=config.retrieval_top_k,
            ),
            "rrf60": reciprocal_rank_fusion(
                rank_lists,
                rrf_k=60,
                top_k=config.retrieval_top_k,
            ),
        }
        gold = gold_by_sample[str(sample_id)]
        for method in FUSION_METHOD_ORDER:
            metrics = retrieval_metrics(
                fused_by_method[method],
                gold,
                ks=config.eval_ks,
            )
            fusion_rows.append(
                {
                    "sample_id": str(sample_id),
                    "budget": int(budget),
                    "arm": method,
                    **{
                        metric: metrics[metric]
                        for metric in TEACHER_METRIC_COLUMNS
                    },
                }
            )

    fusion_by_query = pd.DataFrame(fusion_rows).sort_values(
        ["arm", "sample_id"],
        kind="mergesort",
    )
    fusion_summary = _metric_summary(fusion_by_query)
    standalone = views[
        ["sample_id", "view", *TEACHER_METRIC_COLUMNS]
    ].rename(columns={"view": "arm"})
    standalone = standalone.sort_values(
        ["arm", "sample_id"],
        kind="mergesort",
    )
    standalone_summary = _metric_summary(standalone)
    best_single_row = standalone_summary.sort_values(
        ["MRR", "arm"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    best_single = str(best_single_row["arm"])
    comparison_frame = pd.concat(
        [fusion_by_query, standalone],
        ignore_index=True,
    )
    comparisons = pd.DataFrame(
        [
            _paired_metric(
                comparison_frame,
                arm_column="arm",
                left_arm=left,
                right_arm=right,
                metric="MRR",
                budget=int(budget),
                seed=seed,
                replicates=int(replicates),
                progress=progress,
            )
            for left, right, seed in (
                (
                    "minmax_sum",
                    "rrf10",
                    FUSION_BOOTSTRAP_SEEDS[
                        "minmax_sum_minus_rrf10"
                    ],
                ),
                (
                    "minmax_sum",
                    "rrf60",
                    FUSION_BOOTSTRAP_SEEDS[
                        "minmax_sum_minus_rrf60"
                    ],
                ),
                (
                    "rrf10",
                    "rrf60",
                    FUSION_BOOTSTRAP_SEEDS[
                        "rrf10_minus_rrf60"
                    ],
                ),
                (
                    "minmax_sum",
                    best_single,
                    FUSION_BOOTSTRAP_SEEDS[
                        "minmax_sum_minus_best_single"
                    ],
                ),
                (
                    "rrf10",
                    best_single,
                    FUSION_BOOTSTRAP_SEEDS[
                        "rrf10_minus_best_single"
                    ],
                ),
                (
                    "rrf60",
                    best_single,
                    FUSION_BOOTSTRAP_SEEDS[
                        "rrf60_minus_best_single"
                    ],
                ),
            )
        ]
    )

    winner = (
        fusion_summary.assign(
            tie_rank=fusion_summary["arm"].map(
                {
                    method: index
                    for index, method in enumerate(
                        FUSION_METHOD_ORDER
                    )
                }
            )
        )
        .sort_values(
            ["MRR", "tie_rank"],
            ascending=[False, True],
            kind="mergesort",
        )
        .iloc[0]
    )
    selected_method = str(winner["arm"])
    selected_parameters = _fusion_parameters(selected_method)
    provenance: dict[str, str] = {}
    if architecture_decision_path is not None:
        architecture_path = Path(architecture_decision_path)
        if not architecture_path.exists():
            raise FileNotFoundError(architecture_path)
        provenance["architecture_decision_sha256"] = (
            _sha256_file(architecture_path)
        )
    if source_manifest_path is not None:
        source_path = Path(source_manifest_path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        provenance["source_manifest_sha256"] = _sha256_file(
            source_path
        )
    if not provenance:
        raise ValueError(
            "Fusion requires an architecture decision or source manifest."
        )
    resolved_output = Path(output_dir)
    decision = {
        "schema_version": 1,
        "locked": bool(lock_selection),
        "evaluation_scope": str(evaluation_scope),
        "descriptive_only": not bool(lock_selection),
        "budget": int(budget),
        "selection_metric": "MRR",
        "selection_rule": "highest_unrounded_MRR_then_fixed_order",
        "tie_break_order": list(FUSION_METHOD_ORDER),
        "selected_method": selected_method,
        "selected_parameters": selected_parameters,
        "selected_mrr": float(winner["MRR"]),
        "views": list(VIEW_ORDER),
        "vehicle_sample_id_sha256": str(
            vehicle_sample_id_sha256
        ),
        "selector_checkpoint_sha256": str(
            selector_checkpoint_sha256
        ),
        **provenance,
        "fusion_summary_sha256": _frame_sha256(fusion_summary),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seeds": dict(FUSION_BOOTSTRAP_SEEDS),
    }
    decision_path = resolved_output / (
        "decision.json" if lock_selection else "diagnostic_best.json"
    )
    if lock_selection and decision_path.exists():
        _write_locked_json(decision_path, decision)
    _write_frame(
        resolved_output / "metrics_by_query.csv",
        fusion_by_query,
    )
    _write_frame(resolved_output / "summary.csv", fusion_summary)
    _write_frame(
        resolved_output / "standalone_metrics_by_query.csv",
        standalone,
    )
    _write_frame(
        resolved_output / "standalone_summary.csv",
        standalone_summary,
    )
    _write_frame(
        resolved_output / "paired_comparisons.csv",
        comparisons,
    )
    if lock_selection:
        _write_locked_json(decision_path, decision)
    else:
        _write_json(decision_path, decision)
    manifest = {
        "schema_version": 1,
        "evaluation_scope": str(evaluation_scope),
        "budget": int(budget),
        "views": list(VIEW_ORDER),
        "fusion_methods": list(FUSION_METHOD_ORDER),
        "vehicle_sample_id_sha256": str(
            vehicle_sample_id_sha256
        ),
        "selector_checkpoint_sha256": str(
            selector_checkpoint_sha256
        ),
        "metrics_by_query_sha256": _sha256_file(
            resolved_output / "metrics_by_query.csv"
        ),
        "summary_sha256": _sha256_file(
            resolved_output / "summary.csv"
        ),
        "paired_comparisons_sha256": _sha256_file(
            resolved_output / "paired_comparisons.csv"
        ),
        "decision_sha256": _sha256_file(decision_path),
        "complete": True,
    }
    _write_json(resolved_output / "manifest.json", manifest)
    return {
        "fusion_by_query": fusion_by_query,
        "fusion_summary": fusion_summary,
        "standalone_by_query": standalone,
        "standalone_summary": standalone_summary,
        "comparisons": comparisons,
        "decision": decision,
        "decision_path": decision_path,
        "runtime_cache_stats": {
            "score_cache_misses": int(score_cache_misses),
        },
    }


class BM25ScoreRankingCache:
    """Persistent raw BM25 scores keyed by canonical rewrite keys."""

    def __init__(
        self,
        *,
        cache_db: Path | str,
        index_dir: Path | str,
        retrieval_workers: int,
        top_k: int,
        k1: float = 0.9,
        b: float = 0.4,
    ) -> None:
        self.cache_db = Path(cache_db)
        self.index_dir = Path(index_dir)
        self.retrieval_workers = int(retrieval_workers)
        self.top_k = int(top_k)
        self.k1 = float(k1)
        self.b = float(b)
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connection()):
            pass

    def materialize(
        self,
        queries: pd.DataFrame,
        *,
        expected_rankings: Mapping[str, Sequence[str]],
        progress: bool,
    ) -> tuple[dict[str, list[dict[str, Any]]], int]:
        required = {"rewrite_key", "rewrite_norm"}
        missing = required.difference(queries.columns)
        if missing:
            raise KeyError(
                "Score-cache query columns missing: "
                + ", ".join(sorted(missing))
            )
        unique = (
            queries[["rewrite_key", "rewrite_norm"]]
            .drop_duplicates("rewrite_key")
            .sort_values("rewrite_key", kind="mergesort")
        )
        expected_query = {
            str(row.rewrite_key): str(row.rewrite_norm)
            for row in unique.itertuples(index=False)
        }
        loaded: dict[str, list[dict[str, Any]]] = {}
        with closing(self._connection()) as connection:
            for key_batch in _chunks(list(expected_query), 500):
                placeholders = ",".join("?" for _ in key_batch)
                rows = connection.execute(
                    f"""
                    SELECT rewrite_key, rewrite_norm, hits_blob
                    FROM score_rankings
                    WHERE rewrite_key IN ({placeholders})
                    """,
                    key_batch,
                ).fetchall()
                for key, query, blob in rows:
                    key = str(key)
                    if str(query) != expected_query[key]:
                        raise ValueError(
                            f"Score-cache query mismatch: {key}"
                        )
                    loaded[key] = _unpack_score_hits(blob)
        missing_keys = [
            key for key in expected_query if key not in loaded
        ]
        if missing_keys:
            retriever = load_bm25_retriever(
                index_dir=self.index_dir,
                k1=self.k1,
                b=self.b,
                retrieval_workers=self.retrieval_workers,
            )
            tqdm = get_tqdm()
            try:
                with closing(self._connection()) as connection:
                    for key_batch in tqdm(
                        list(_chunks(missing_keys, 64)),
                        desc="BM25 raw-score cache misses",
                        unit="batch",
                        dynamic_ncols=True,
                        disable=not progress,
                    ):
                        queries = [
                            expected_query[key] for key in key_batch
                        ]
                        found: list[list[dict[str, Any]]] = [
                            [] for _ in queries
                        ]
                        nonempty = [
                            index
                            for index, query in enumerate(queries)
                            if query
                        ]
                        if nonempty:
                            retrieved = retriever.search_batch(
                                [queries[index] for index in nonempty],
                                top_k=self.top_k,
                            )
                            for index, hits in zip(
                                nonempty,
                                retrieved,
                                strict=True,
                            ):
                                found[index] = hits
                        connection.executemany(
                            """
                            INSERT OR REPLACE INTO score_rankings
                            (rewrite_key, rewrite_norm, hits_blob, num_hits)
                            VALUES (?, ?, ?, ?)
                            """,
                            [
                                (
                                    key,
                                    expected_query[key],
                                    _pack_score_hits(hits),
                                    len(hits),
                                )
                                for key, hits in zip(
                                    key_batch,
                                    found,
                                    strict=True,
                                )
                            ],
                        )
                        connection.commit()
                        loaded.update(
                            {
                                key: [dict(hit) for hit in hits]
                                for key, hits in zip(
                                    key_batch,
                                    found,
                                    strict=True,
                                )
                            }
                        )
            finally:
                retriever.close()

        for key, hits in loaded.items():
            expected = [str(docid) for docid in expected_rankings[key]]
            actual = [str(hit["docid"]) for hit in hits]
            if actual != expected:
                raise ValueError(
                    "Raw-score ranking differs from canonical cache for "
                    f"{key}."
                )
        return loaded, len(missing_keys)

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.cache_db, timeout=120)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS score_rankings (
                rewrite_key TEXT PRIMARY KEY,
                rewrite_norm TEXT NOT NULL,
                hits_blob BLOB NOT NULL,
                num_hits INTEGER NOT NULL
            )
            """
        )
        connection.commit()
        return connection


def _pipeline(
    config: Any,
    *,
    device: str,
    tokenizer: Any | None,
    progress: bool,
) -> CachedIterCQRBM25Pipeline:
    return CachedIterCQRBM25Pipeline(
        config=CachedIterCQRBM25Config(
            cache_db=config.pipeline_cache_db,
            model_dir=config.itercqr_model_dir,
            bm25_index_dir=config.bm25_index_dir,
            device=str(device),
            rewrite_batch_size=config.rewrite_batch_size,
            retrieval_batch_size=config.retrieval_batch_size,
            retrieval_workers=config.retrieval_workers,
            retrieval_top_k=config.retrieval_top_k,
            eval_ks=config.eval_ks,
            progress=progress,
        ),
        tokenizer=tokenizer,
    )


def _paired_ranking_keys(
    frame: pd.DataFrame,
    *,
    arm_to_view: Mapping[str, str],
    budget: int,
) -> pd.DataFrame:
    selected = frame.loc[
        frame["budget"].eq(int(budget))
        & frame["arm"].isin(arm_to_view),
        ["sample_id", "arm", "rewrite_key"],
    ].copy()
    selected["view"] = selected["arm"].map(arm_to_view)
    if selected.duplicated(["sample_id", "view"]).any():
        raise ValueError("Ranking-key pairs contain duplicates.")
    paired = selected.pivot(
        index="sample_id",
        columns="view",
        values="rewrite_key",
    ).sort_index(kind="mergesort")
    if (
        set(paired.columns) != set(arm_to_view.values())
        or paired.isna().any().any()
    ):
        raise ValueError("Ranking keys are not paired completely.")
    return paired


def _paired_metric(
    frame: pd.DataFrame,
    *,
    arm_column: str,
    left_arm: str,
    right_arm: str,
    metric: str,
    budget: int,
    seed: int,
    replicates: int,
    progress: bool,
) -> dict[str, Any]:
    selected = frame.loc[
        frame[arm_column].isin([left_arm, right_arm]),
        ["sample_id", arm_column, metric],
    ].copy()
    if selected.duplicated(["sample_id", arm_column]).any():
        raise ValueError("Paired metric rows contain duplicates.")
    wide = selected.pivot(
        index="sample_id",
        columns=arm_column,
        values=metric,
    ).sort_index(kind="mergesort")
    if (
        left_arm not in wide
        or right_arm not in wide
        or wide[[left_arm, right_arm]].isna().any().any()
    ):
        raise ValueError("Metric arms are not paired completely.")
    deltas = (
        wide[left_arm].to_numpy(dtype=float)
        - wide[right_arm].to_numpy(dtype=float)
    )
    ci_low, ci_high = paired_bootstrap_mean_ci(
        deltas,
        seed=int(seed),
        replicates=int(replicates),
        progress=progress,
        progress_desc=(
            f"bootstrap {metric} {left_arm} - {right_arm}"
        ),
    )
    return {
        "budget": int(budget),
        "metric": str(metric),
        "comparison": f"{left_arm} - {right_arm}",
        "left_arm": left_arm,
        "right_arm": right_arm,
        "n": int(len(deltas)),
        "delta": float(np.mean(deltas)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "seed": int(seed),
        "replicates": int(replicates),
    }


def _minmax_sum_fusion(
    score_lists: Sequence[Sequence[Mapping[str, Any]]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    for hits in score_lists:
        normalized = normalize_retrieval_scores(
            [dict(hit) for hit in hits],
            method="minmax",
        )
        seen: set[str] = set()
        for hit in normalized:
            docid = str(hit["docid"])
            if docid in seen:
                continue
            seen.add(docid)
            scores[docid] += float(hit["score_norm"])
    ordered = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[: int(top_k)]
    return [
        {"docid": docid, "rank": rank, "score": score}
        for rank, (docid, score) in enumerate(ordered, start=1)
    ]


def _fusion_parameters(method: str) -> dict[str, Any]:
    if method == "minmax_sum":
        return {
            "normalization": "query_local_minmax",
            "aggregation": "sum",
            "view_weights": {view: 1.0 for view in VIEW_ORDER},
        }
    if method == "rrf10":
        return {
            "rrf_k": 10,
            "view_weights": {view: 1.0 for view in VIEW_ORDER},
        }
    if method == "rrf60":
        return {
            "rrf_k": 60,
            "view_weights": {view: 1.0 for view in VIEW_ORDER},
        }
    raise ValueError(f"Unknown fusion method: {method}")


def _metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby("arm", observed=True, sort=True)
        .agg(
            n=("sample_id", "size"),
            **{
                metric: (metric, "mean")
                for metric in TEACHER_METRIC_COLUMNS
            },
        )
        .reset_index()
    )


def _pack_score_hits(hits: Sequence[Mapping[str, Any]]) -> bytes:
    payload = json.dumps(
        [
            [str(hit["docid"]), float(hit["score"])]
            for hit in hits
        ],
        separators=(",", ":"),
    ).encode("utf-8")
    return zlib.compress(payload, level=6)


def _unpack_score_hits(blob: bytes) -> list[dict[str, Any]]:
    pairs = json.loads(zlib.decompress(blob).decode("utf-8"))
    return [
        {"docid": str(docid), "rank": rank, "score": float(score)}
        for rank, (docid, score) in enumerate(pairs, start=1)
    ]


def _chunks(values: Sequence[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), int(size)):
        yield list(values[start : start + int(size)])


def _sample_id_sha256(sample_ids: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(map(str, sample_ids))).encode("utf-8")
    ).hexdigest()


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_locked_json(path: Path, value: Mapping[str, Any]) -> None:
    expected = json.loads(
        json.dumps(dict(value), ensure_ascii=False)
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != expected:
            raise RuntimeError(
                f"Locked fusion decision differs: {path}"
            )
        return
    _write_json(path, expected)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BM25ScoreRankingCache",
    "FUSION_BOOTSTRAP_SEEDS",
    "FUSION_METHOD_ORDER",
    "RECALL_BOOTSTRAP_SEEDS",
    "VIEW_ORDER",
    "compare_target_recall",
    "compute_union_recall",
    "evaluate_fixed_rrf_fusion",
    "evaluate_selector_fusions",
    "run_selector_direct_bm25",
]
