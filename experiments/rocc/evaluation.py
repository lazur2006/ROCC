"""Retrieval evaluation pipeline for ROCC experiments."""

from __future__ import annotations

import ast
import hashlib
import json
from numbers import Integral, Real
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import log2
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .progress import get_tqdm


DEFAULT_EVAL_KS = (10, 100, 1000)
BOOTSTRAP_METHOD = "conversation-cluster-percentile-v1"


def dataset_bootstrap_seed(dataset, cohort="all") -> int:
    """Explicit-dataset variant for notebooks with metadata already in memory."""
    if dataset not in ("topiocqa", "qrecc"):
        raise ValueError("Unknown evaluation dataset")
    key = f"chiir2027-conversation-bootstrap-v1|20261019|{dataset}|{cohort}"
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def _conversation_id(value: Any) -> str:
    if not np.isscalar(value) or pd.isna(value):
        raise ValueError("Conversation IDs must be non-missing scalar values.")
    if isinstance(value, Real) and not isinstance(value, Integral):
        if not np.isfinite(value):
            raise ValueError("Conversation IDs must be finite.")
        if float(value).is_integer():
            value = int(value)
    result = str(value).strip()
    if not result or result.lower() in {"nan", "none", "null", "<na>", "nat"}:
        raise ValueError("Conversation IDs must not be empty or missing.")
    return result


def conversation_ids_from_frame(
    frame: pd.DataFrame, *, sample_ids: Sequence[Any] | None = None,
) -> np.ndarray:
    """Validate explicit ``conv_id`` metadata, optionally align by sample_id.

    Include dataset/split in the cluster key when available, so equal numeric
    conversation IDs in separate populations cannot silently merge. All rows
    for a paired sample must agree. Never infer IDs from sample-name prefixes.
    """
    if "conv_id" not in frame:
        raise ValueError("Conversation bootstrap requires an explicit conv_id column.")
    columns = [name for name in ("dataset", "split")
               if name in frame and frame[name].notna().any()] + ["conv_id"]
    normalized = frame[columns].apply(lambda column: column.map(_conversation_id))
    keys = np.asarray(
        [json.dumps(row, ensure_ascii=True, separators=(",", ":"))
         for row in normalized.itertuples(index=False, name=None)], dtype=str,
    )
    if sample_ids is None:
        return keys
    if "sample_id" not in frame:
        raise ValueError("Sample alignment requires an explicit sample_id column.")
    lookup = pd.DataFrame({"sample_id": frame["sample_id"].map(_conversation_id), "cluster": keys})
    if lookup.groupby("sample_id")["cluster"].nunique().gt(1).any():
        raise ValueError("Paired sample rows have inconsistent conversation metadata.")
    lookup = lookup.drop_duplicates("sample_id").set_index("sample_id")["cluster"]
    requested = [_conversation_id(value) for value in sample_ids]
    result = lookup.reindex(requested)
    if result.isna().any():
        raise ValueError("Conversation metadata are missing for requested sample IDs.")
    return result.to_numpy(dtype=str)


def conversation_bootstrap_means(
    values: Any,
    conversation_ids: Sequence[Any],
    *,
    n_samples: int = 10_000,
    seed: int = 13,
    chunk_size: int = 500,
    singleton_index_draws: bool = False,
) -> np.ndarray:
    """Return ``(n_samples, n_metrics)`` query-weighted bootstrap means.

    A vector is promoted to one metric column. Cluster ordering is sorted and
    deterministic; chunk size does not change the sampled conversations. A
    single conversation yields a degenerate interval, not invented query-level
    uncertainty. The original point estimate remains ``values.mean(axis=0)``.

    ``singleton_index_draws`` is an explicit replay option for the frozen
    one-query-per-conversation development sample. It retains original row
    order and seeded index draws and rejects repeated conversation IDs.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError("values must be a non-empty finite query-by-metric array.")
    if not isinstance(n_samples, Integral) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if not isinstance(chunk_size, Integral) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    ids = np.asarray(conversation_ids, dtype=object)
    if ids.ndim != 1 or len(ids) != len(array):
        raise ValueError("conversation_ids must contain one ID per query row.")
    ids = np.asarray([_conversation_id(value) for value in ids], dtype=str)
    clusters, inverse = np.unique(ids, return_inverse=True)
    if singleton_index_draws:
        # Preserve the frozen development experiment's seeded index draws.
        # With exactly one evaluated query per conversation these are literal
        # conversation draws, not an IID fallback for multi-turn populations.
        if len(clusters) != len(array):
            raise ValueError("Singleton index draws require one query per conversation.")
        rng = np.random.default_rng(seed)
        draws = np.empty((n_samples, array.shape[1]), dtype=np.float64)
        for start in range(0, n_samples, chunk_size):
            stop = min(start + chunk_size, n_samples)
            indices = rng.integers(0, len(array), size=(stop-start, len(array)))
            draws[start:stop] = array[indices].mean(axis=1)
        return draws
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.zeros((len(clusters), array.shape[1]), dtype=np.float64)
    np.add.at(sums, inverse, array)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_samples, array.shape[1]), dtype=np.float64)
    probabilities = np.full(len(clusters), 1.0 / len(clusters))
    # Multinomial multiplicities match the published paper audit protocol and
    # avoid a draws x clusters x metrics allocation for large evaluations.
    for start in range(0, n_samples, chunk_size):
        stop = min(start + chunk_size, n_samples)
        multiplicities = rng.multinomial(len(clusters), probabilities, size=stop - start)
        denominator = multiplicities @ counts
        draws[start:stop] = (multiplicities @ sums) / denominator[:, None]
    return draws


def attach_conversation_metadata(frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Carry validated conversation metadata through derived/fused metric rows."""
    requested = frame["sample_id"].map(_conversation_id)
    conversation_ids_from_frame(source, sample_ids=requested)
    columns = [name for name in ("dataset", "split")
               if name in source and source[name].notna().any()] + ["conv_id"]
    metadata = source[["sample_id", *columns]].copy()
    metadata["sample_id"] = metadata["sample_id"].map(_conversation_id)
    metadata = metadata.drop_duplicates("sample_id").set_index("sample_id")
    result = frame.copy()
    result["sample_id"] = requested
    for column in columns:
        expected = requested.map(metadata[column])
        if column in result:
            present = result[column].notna()
            if not result.loc[present, column].map(_conversation_id).equals(expected.loc[present].map(_conversation_id)):
                raise ValueError(f"Existing {column} disagrees with source metadata.")
        result[column] = expected
    return result


def conversation_bootstrap_ci(
    values: Any, conversation_ids: Sequence[Any], *,
    n_samples: int = 10_000, seed: int = 13, chunk_size: int = 500,
    quantiles: tuple[float, float] = (0.025, 0.975),
    singleton_index_draws: bool = False,
) -> np.ndarray:
    """Return percentile bounds with shape ``(2, n_metrics)``."""
    if len(quantiles) != 2 or not 0 <= quantiles[0] < quantiles[1] <= 1:
        raise ValueError("quantiles must be two ordered probabilities in [0, 1].")
    return np.quantile(conversation_bootstrap_means(
        values, conversation_ids, n_samples=n_samples, seed=seed, chunk_size=chunk_size,
        singleton_index_draws=singleton_index_draws,
    ), quantiles, axis=0)


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
    bootstrap_samples: int = 10_000,
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
    bootstrap_samples: int = 10_000,
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
    n_samples: int = 10_000,
    seed: int = 13,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query-weighted means and conversation-bootstrap percentile intervals."""
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

    values = per_query[metric_columns].apply(pd.to_numeric, errors="raise").to_numpy()
    conversation_ids = conversation_ids_from_frame(per_query)
    draw_values = conversation_bootstrap_means(
        values, conversation_ids, n_samples=n_samples, seed=seed,
    )
    draws = pd.DataFrame(draw_values, columns=metric_columns)

    alpha = (1.0 - confidence) / 2.0
    rows = []
    for metric_index, metric in enumerate(metric_columns):
        metric_draws = draws[metric].to_numpy()
        rows.append(
            {
                "metric": metric,
                "bootstrap_samples": int(n_samples),
                "confidence": float(confidence),
                "mean": float(values[:, metric_index].mean()),
                "variance": float(metric_draws.var(ddof=1)) if metric_draws.shape[0] > 1 else 0.0,
                "ci95_low": float(np.quantile(metric_draws, alpha)),
                "ci95_high": float(np.quantile(metric_draws, 1.0 - alpha)),
            }
        )
    return pd.DataFrame(rows), draws


def paired_bootstrap_mean_ci(
    deltas: Sequence[float] | np.ndarray,
    *,
    conversation_ids: Sequence[Any],
    seed: int,
    replicates: int = 10_000,
    chunk_size: int = 500,
    quantiles: tuple[float, float] = (0.025, 0.975),
    progress: bool = True,
    progress_desc: str = "paired bootstrap",
    singleton_index_draws: bool = False,
) -> tuple[float, float]:
    """Conversation-bootstrap CI of paired query-level deltas.

    Conversation IDs must be aligned with deltas. Progress arguments remain
    accepted for notebook compatibility; the bounded-memory helper is quiet.
    """

    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("deltas muss ein nichtleerer 1D-Vektor sein.")
    bounds = conversation_bootstrap_ci(
        values, conversation_ids, n_samples=replicates, seed=seed,
        chunk_size=chunk_size, quantiles=quantiles,
        singleton_index_draws=singleton_index_draws,
    )
    return float(bounds[0, 0]), float(bounds[1, 0])


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
