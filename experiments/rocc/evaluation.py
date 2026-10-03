"""Retrieval evaluation pipeline for ROCC experiments."""

from __future__ import annotations

import ast
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import ceil, log2
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .progress import get_tqdm


DEFAULT_EVAL_KS = (10, 100, 1000)


@dataclass
class RetrievalEvaluationResult:
    per_query: pd.DataFrame
    aggregate: pd.DataFrame
    bootstrap_summary: pd.DataFrame
    bootstrap_samples: pd.DataFrame


def evaluate_retrieval_pipeline(
    retrieval_result: Any,
    *,
    ks: Sequence[int] = DEFAULT_EVAL_KS,
    bootstrap_samples: int = 1000,
    seed: int = 13,
    confidence: float = 0.95,
) -> RetrievalEvaluationResult:
    """Evaluate a ``run_itercqr_retrieval_pipeline`` result."""

    return evaluate_retrieval_run(
        rewrites=retrieval_result.rewrites,
        hits_by_sample_id=retrieval_result.hits_by_sample_id,
        ks=ks,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        confidence=confidence,
    )


def evaluate_retrieval_run(
    *,
    rewrites: pd.DataFrame,
    hits_by_sample_id: dict[str, list[dict[str, Any]]],
    ks: Sequence[int] = DEFAULT_EVAL_KS,
    bootstrap_samples: int = 1000,
    seed: int = 13,
    confidence: float = 0.95,
) -> RetrievalEvaluationResult:
    """Evaluate ranked retrieval hits against the positive passage ids."""

    normalized_ks = normalize_eval_ks(ks)
    rows: list[dict[str, Any]] = []
    for _, row in rewrites.iterrows():
        sample_id = str(row["sample_id"])
        relevant_docids = normalize_docids(row.get("positive_ctx_passage_ids", []))
        hits = hits_by_sample_id.get(sample_id, [])
        metrics = retrieval_metrics(hits, relevant_docids, ks=normalized_ks)
        rows.append(
            {
                "sample_id": sample_id,
                "split": row.get("split"),
                "conv_id": row.get("conv_id"),
                "turn_id": row.get("turn_id"),
                "query": row.get("rewrite", row.get("query")),
                "num_hits": len(hits),
                "num_gold_passages": len(relevant_docids),
                **metrics,
            }
        )

    per_query = pd.DataFrame(rows)
    metric_columns = metric_columns_from_frame(per_query)
    aggregate = aggregate_metric_frame(per_query, metric_columns=metric_columns)
    bootstrap_summary, bootstrap_draws = bootstrap_metric_summary(
        per_query,
        metric_columns=metric_columns,
        n_samples=bootstrap_samples,
        seed=seed,
        confidence=confidence,
    )
    return RetrievalEvaluationResult(
        per_query=per_query,
        aggregate=aggregate,
        bootstrap_summary=bootstrap_summary,
        bootstrap_samples=bootstrap_draws,
    )


def retrieval_metrics(
    hits: list[dict[str, Any]],
    relevant_docids: Iterable[Any],
    *,
    ks: Sequence[int] = DEFAULT_EVAL_KS,
) -> dict[str, float | int | None]:
    relevant = set(normalize_docids(relevant_docids))
    first_rank = first_relevant_rank(hits, relevant)
    metrics: dict[str, float | int | None] = {
        "target_rank": first_rank,
        "MRR": 1.0 / first_rank if first_rank else 0.0,
    }
    for k in normalize_eval_ks(ks):
        metrics[f"nDCG@{k}"] = ndcg_at_k(hits, relevant, k)
        metrics[f"R@{k}"] = recall_at_k(hits, relevant, k)
    return metrics


def first_relevant_rank(
    hits: list[dict[str, Any]],
    relevant_docids: Iterable[Any],
) -> int | None:
    relevant = set(normalize_docids(relevant_docids))
    if not relevant:
        return None
    for rank, docid in enumerate(ranked_docids(hits), start=1):
        if docid in relevant:
            return rank
    return None


def reciprocal_rank(
    hits: list[dict[str, Any]],
    relevant_docids: Iterable[Any],
) -> tuple[float, int | None]:
    rank = first_relevant_rank(hits, relevant_docids)
    return (1.0 / rank if rank else 0.0, rank)


def reciprocal_rank_fusion(
    ranked_docid_lists: Sequence[Sequence[Any]],
    *,
    rrf_k: int = 60,
    top_k: int = 1_000,
) -> list[dict[str, Any]]:
    """Fuse ranked document lists with deterministic unweighted RRF."""

    if int(rrf_k) < 0:
        raise ValueError("rrf_k muss mindestens 0 sein.")
    if int(top_k) < 1:
        raise ValueError("top_k muss mindestens 1 sein.")
    if len(ranked_docid_lists) < 2:
        raise ValueError(
            "RRF benötigt mindestens zwei Ranking-Listen."
        )

    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_docid_lists:
        seen: set[str] = set()
        for rank, raw_docid in enumerate(ranking, start=1):
            docid = str(raw_docid)
            if docid in seen:
                continue
            seen.add(docid)
            scores[docid] += 1.0 / (int(rrf_k) + rank)

    ordered = sorted(
        scores.items(),
        key=lambda item: (-item[1], item[0]),
    )[: int(top_k)]
    return [
        {
            "docid": docid,
            "rank": rank,
            "score": score,
        }
        for rank, (docid, score) in enumerate(ordered, start=1)
    ]


def evaluate_fixed_rrf_views(
    *,
    query_views: pd.DataFrame,
    hits_by_query_id: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
    gold_by_sample: Mapping[str, Sequence[Any]],
    view_sets: Mapping[str, Sequence[str]],
    group_columns: Sequence[str] = ("system", "budget"),
    rrf_k: int = 10,
    top_k: int = 1_000,
    ks: Sequence[int] = (3, 10, 100, 1000),
    measure_latency: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """Evaluate fixed single-view and RRF systems without selection.

    With ``measure_latency=False``, no fusion timer or timing column is
    created.
    """

    required = {
        "sample_id",
        "view",
        "query_id",
        "query_norm",
        *map(str, group_columns),
    }
    missing = required.difference(query_views.columns)
    if missing:
        raise KeyError(
            "Fixed-RRF query columns missing: "
            + ", ".join(sorted(missing))
        )
    if not view_sets:
        raise ValueError("view_sets must not be empty.")
    normalized_sets = {
        str(arm): tuple(str(view) for view in views)
        for arm, views in view_sets.items()
    }
    if any(not views for views in normalized_sets.values()):
        raise ValueError("Every fixed view set must contain a view.")

    frame = query_views.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["view"] = frame["view"].astype(str)
    frame["query_id"] = frame["query_id"].astype(str)
    frame["query_norm"] = frame["query_norm"].astype(str)
    if frame.duplicated(
        [*map(str, group_columns), "sample_id", "view"]
    ).any():
        raise ValueError("Fixed-RRF query views contain duplicates.")

    grouping = [*map(str, group_columns), "sample_id"]
    groups = frame.groupby(grouping, observed=True, sort=True)
    rows: list[dict[str, Any]] = []
    tqdm = get_tqdm()
    for group_key, sample_frame in tqdm(
        groups,
        total=groups.ngroups,
        desc="evaluate fixed retrieval views",
        unit="query-system",
        dynamic_ncols=True,
        disable=not progress,
    ):
        values = (
            group_key
            if isinstance(group_key, tuple)
            else (group_key,)
        )
        group_values = dict(zip(grouping, values, strict=True))
        sample_id = str(group_values["sample_id"])
        if sample_id not in gold_by_sample:
            raise KeyError(f"Gold missing for {sample_id}.")
        by_view = sample_frame.set_index("view", drop=False)
        for arm, requested_views in normalized_sets.items():
            missing_views = [
                view for view in requested_views if view not in by_view.index
            ]
            if missing_views:
                raise ValueError(
                    f"Views missing for {sample_id}, {arm}: "
                    f"{missing_views}"
                )
            selected_rows = [by_view.loc[view] for view in requested_views]
            unique_rows: list[pd.Series] = []
            seen_queries: set[str] = set()
            for selected in selected_rows:
                query_norm = str(selected["query_norm"])
                if query_norm in seen_queries:
                    continue
                seen_queries.add(query_norm)
                unique_rows.append(selected)
            rankings: list[list[str]] = []
            for selected in unique_rows:
                query_id = str(selected["query_id"])
                if query_id not in hits_by_query_id:
                    raise KeyError(f"Ranking missing for {query_id}.")
                rankings.append(
                    ranked_docids(
                        list(hits_by_query_id[query_id]),
                        k=int(top_k),
                    )
                )
            fusion_started = (
                time.perf_counter() if measure_latency else None
            )
            if len(rankings) == 1:
                fused = [
                    {
                        "docid": docid,
                        "rank": rank,
                    }
                    for rank, docid in enumerate(
                        rankings[0],
                        start=1,
                    )
                ]
            else:
                fused = reciprocal_rank_fusion(
                    rankings,
                    rrf_k=int(rrf_k),
                    top_k=int(top_k),
                )
            fusion_seconds = (
                time.perf_counter() - fusion_started
                if fusion_started is not None
                else None
            )
            metrics = retrieval_metrics(
                fused,
                gold_by_sample[sample_id],
                ks=ks,
            )
            rows.append(
                {
                    **{
                        column: group_values[column]
                        for column in map(str, group_columns)
                    },
                    "sample_id": sample_id,
                    "arm": arm,
                    "requested_views": "+".join(requested_views),
                    "requested_view_count": len(requested_views),
                    "unique_view_count": len(unique_rows),
                    **(
                        {"fusion_seconds": float(fusion_seconds)}
                        if fusion_seconds is not None
                        else {}
                    ),
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(
        [*map(str, group_columns), "arm", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def recall_at_k(
    hits: list[dict[str, Any]],
    relevant_docids: Iterable[Any],
    k: int,
) -> float:
    relevant = set(normalize_docids(relevant_docids))
    if not relevant:
        return 0.0
    retrieved = set(ranked_docids(hits, k=k))
    return len(retrieved & relevant) / len(relevant)


def ndcg_at_k(
    hits: list[dict[str, Any]],
    relevant_docids: Iterable[Any],
    k: int,
) -> float:
    relevant = set(normalize_docids(relevant_docids))
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, docid in enumerate(ranked_docids(hits, k=k), start=1):
        if docid in relevant:
            dcg += 1.0 / log2(rank + 1)
    ideal_hits = min(len(relevant), int(k))
    ideal_dcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def aggregate_metric_frame(
    per_query: pd.DataFrame,
    *,
    metric_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    metric_columns = list(metric_columns or metric_columns_from_frame(per_query))
    rows = []
    for metric in metric_columns:
        values = pd.to_numeric(per_query[metric], errors="coerce").dropna()
        rows.append(
            {
                "metric": metric,
                "n": int(values.shape[0]),
                "mean": float(values.mean()) if not values.empty else float("nan"),
                "variance": float(values.var(ddof=1)) if values.shape[0] > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric_summary(
    per_query: pd.DataFrame,
    *,
    metric_columns: Sequence[str] | None = None,
    n_samples: int = 1000,
    seed: int = 13,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_samples < 1:
        return (
            pd.DataFrame(
                columns=["metric", "bootstrap_samples", "confidence", "mean", "variance", "ci95_low", "ci95_high"]
            ),
            pd.DataFrame(),
        )
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")

    metric_columns = list(metric_columns or metric_columns_from_frame(per_query))
    if per_query.empty or not metric_columns:
        return pd.DataFrame(), pd.DataFrame()

    values = per_query[metric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.shape[0], size=(int(n_samples), values.shape[0]))
    draw_values = values[indices].mean(axis=1)
    draws = pd.DataFrame(draw_values, columns=metric_columns)

    alpha = (1.0 - confidence) / 2.0
    rows = []
    for metric in metric_columns:
        metric_draws = draws[metric].to_numpy()
        rows.append(
            {
                "metric": metric,
                "bootstrap_samples": int(n_samples),
                "confidence": float(confidence),
                "mean": float(metric_draws.mean()),
                "variance": float(metric_draws.var(ddof=1)) if metric_draws.shape[0] > 1 else 0.0,
                "ci95_low": float(np.quantile(metric_draws, alpha)),
                "ci95_high": float(np.quantile(metric_draws, 1.0 - alpha)),
            }
        )
    return pd.DataFrame(rows), draws


def paired_bootstrap_mean_ci(
    deltas: Sequence[float] | np.ndarray,
    *,
    seed: int,
    replicates: int = 10_000,
    chunk_size: int = 500,
    quantiles: tuple[float, float] = (0.025, 0.975),
    progress: bool = True,
    progress_desc: str = "paired bootstrap",
) -> tuple[float, float]:
    """Bootstrap the mean of paired deltas with bounded memory."""

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("deltas muss ein nichtleerer 1D-Vektor sein.")
    if replicates < 1:
        raise ValueError("replicates muss mindestens 1 sein.")
    if chunk_size < 1:
        raise ValueError("chunk_size muss mindestens 1 sein.")

    rng = np.random.default_rng(seed)
    draws = np.empty(int(replicates), dtype=np.float64)
    tqdm = get_tqdm()
    for start in tqdm(
        range(0, int(replicates), int(chunk_size)),
        total=ceil(int(replicates) / int(chunk_size)),
        desc=progress_desc,
        unit="chunk",
        dynamic_ncols=True,
        disable=not progress,
    ):
        size = min(int(chunk_size), int(replicates) - start)
        indices = rng.integers(
            0,
            len(values),
            size=(size, len(values)),
        )
        draws[start : start + size] = values[indices].mean(
            axis=1
        )
    return tuple(
        map(float, np.quantile(draws, quantiles))
    )


def metric_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column == "MRR" or column.startswith("nDCG@") or column.startswith("R@")
    ]


def ranked_docids(hits: list[dict[str, Any]], *, k: int | None = None) -> list[str]:
    ranked = sorted(
        enumerate(hits),
        key=lambda item: int(item[1].get("rank", item[0] + 1)),
    )
    if k is not None:
        ranked = ranked[: int(k)]
    return [str(hit["docid"]) for _, hit in ranked if "docid" in hit]


def normalize_eval_ks(ks: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted({int(k) for k in ks if int(k) > 0}))
    if not normalized:
        raise ValueError("ks must contain at least one positive integer")
    return normalized


def normalize_docids(docids: Iterable[Any] | Any) -> list[str]:
    if docids is None:
        return []
    if isinstance(docids, (float, np.floating)) and np.isnan(docids):
        return []
    if isinstance(docids, str):
        value = docids.strip()
        if not value:
            return []
        if value.startswith("["):
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = value
            else:
                return normalize_docids(parsed)
        return [part for part in value.replace(",", " ").split() if part]
    if isinstance(docids, Iterable):
        return [str(docid) for docid in docids if docid is not None]
    return [str(docids)]
