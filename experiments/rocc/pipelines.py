"""Composable experiment pipelines built from the subpipeline APIs."""

from __future__ import annotations

import json
import sqlite3
import time
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .cached_itercqr import normalize_rewrite
from .progress import progress_iter
from .retrievers import normalize_retrieval_scores


@dataclass
class RewriteRetrievePipelineResult:
    rewrites: pd.DataFrame
    retrievals: pd.DataFrame
    hits: pd.DataFrame
    hits_by_sample_id: dict[str, list[dict[str, Any]]]
    rewrite_efficiency_per_batch: pd.DataFrame = field(default_factory=pd.DataFrame)
    rewrite_efficiency_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class RewriteEfficiencyBenchmarkResult:
    rewrites: pd.DataFrame
    rewrite_efficiency_per_batch: pd.DataFrame
    rewrite_efficiency_summary: pd.DataFrame


@dataclass
class QueryRetrievalResult:
    """Backend-neutral retrieval for already materialized query strings."""

    queries: pd.DataFrame
    hits: pd.DataFrame
    hits_by_query_id: dict[str, list[dict[str, Any]]]
    latency_per_batch: pd.DataFrame
    latency_summary: pd.DataFrame


@dataclass(frozen=True)
class PartialRetrievalDumpResult:
    output_file: Path
    shard_session: str
    top_k: int
    query_count: int
    hit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_file": str(self.output_file),
            "shard_session": self.shard_session,
            "top_k": self.top_k,
            "query_count": self.query_count,
            "hit_count": self.hit_count,
        }


@dataclass(frozen=True)
class PartialRetrievalMergeResult:
    hits_by_sample_id: dict[str, list[dict[str, Any]]]
    output_file: Path | None
    top_k: int
    query_count: int
    hit_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_file": str(self.output_file) if self.output_file else None,
            "top_k": self.top_k,
            "query_count": self.query_count,
            "hit_count": self.hit_count,
        }


class QueryRankingStore(Mapping[str, list[dict[str, Any]]]):
    """Disk-backed ranked hits keyed by stable query IDs."""

    def __init__(
        self,
        path: Path | str,
        *,
        read_only: bool = False,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.read_only = bool(read_only)
        if self.read_only:
            if not self.path.is_file():
                raise FileNotFoundError(self.path)
            self._connection = sqlite3.connect(
                f"file:{self.path}?mode=ro",
                uri=True,
            )
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(self.path)
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rankings (
                    query_id TEXT PRIMARY KEY,
                    hits_blob BLOB NOT NULL
                )
                """
            )
            self._connection.commit()
        self._cached_get = lru_cache(maxsize=128)(
            self._get_uncached
        )

    def __getitem__(self, query_id: str) -> list[dict[str, Any]]:
        return self._cached_get(str(query_id))

    def _get_uncached(
        self,
        query_id: str,
    ) -> list[dict[str, Any]]:
        row = self._connection.execute(
            "SELECT hits_blob FROM rankings WHERE query_id = ?",
            (str(query_id),),
        ).fetchone()
        if row is None:
            raise KeyError(str(query_id))
        return _unpack_query_hits(row[0])

    def __iter__(self):
        rows = self._connection.execute(
            "SELECT query_id FROM rankings ORDER BY query_id"
        )
        for (query_id,) in rows:
            yield str(query_id)

    def __len__(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM rankings"
        ).fetchone()
        return int(row[0])

    def put_many(
        self,
        hits_by_query_id: Mapping[
            str,
            Sequence[Mapping[str, Any]],
        ],
        *,
        top_k: int,
    ) -> None:
        if self.read_only:
            raise RuntimeError("Read-only ranking store cannot be written.")
        self._connection.executemany(
            """
            INSERT OR REPLACE INTO rankings (query_id, hits_blob)
            VALUES (?, ?)
            """,
            [
                (
                    str(query_id),
                    _pack_query_hits(hits[: int(top_k)]),
                )
                for query_id, hits in hits_by_query_id.items()
            ],
        )
        self._connection.commit()
        self._cached_get.cache_clear()

    def existing_ids(
        self,
        query_ids: Sequence[str] | None = None,
    ) -> set[str]:
        if query_ids is None:
            return set(iter(self))
        requested = [str(query_id) for query_id in query_ids]
        found: set[str] = set()
        for start in range(0, len(requested), 900):
            chunk = requested[start : start + 900]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._connection.execute(
                "SELECT query_id FROM rankings "
                f"WHERE query_id IN ({placeholders})",
                chunk,
            )
            found.update(str(row[0]) for row in rows)
        return found

    def close(self) -> None:
        self._cached_get.cache_clear()
        self._connection.close()

    def __enter__(self) -> "QueryRankingStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


@dataclass(frozen=True)
class QueryRetrievalStoreResult:
    store_path: Path
    requested_queries: int
    reused_queries: int
    retrieved_queries: int
    stored_queries: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "store_path": str(self.store_path),
            "requested_queries": self.requested_queries,
            "reused_queries": self.reused_queries,
            "retrieved_queries": self.retrieved_queries,
            "stored_queries": self.stored_queries,
        }


def run_query_retrieval_pipeline(
    *,
    queries: pd.DataFrame,
    retriever: Any,
    query_id_column: str = "query_id",
    query_column: str = "query",
    top_k: int = 1_000,
    batch_size: int = 64,
    warmup_queries: int = 0,
    measure_latency: bool = True,
    deduplication_mode: str = "casefold",
    progress: bool = True,
    progress_desc: str = "retrieve materialized queries",
) -> QueryRetrievalResult:
    """Retrieve fixed query artifacts through any search_batch backend.

    Setting ``measure_latency=False`` skips warm-up, synchronization,
    timers, and latency-table construction.
    """

    required = {str(query_id_column), str(query_column)}
    missing = required.difference(queries.columns)
    if missing:
        raise KeyError(
            "Materialized query columns missing: "
            + ", ".join(sorted(missing))
        )
    if int(top_k) < 1 or int(batch_size) < 1:
        raise ValueError("top_k and batch_size must be positive.")
    if deduplication_mode not in {"casefold", "exact"}:
        raise ValueError(
            "deduplication_mode must be 'casefold' or 'exact'."
        )
    materialized = queries.copy()
    materialized[query_id_column] = materialized[
        query_id_column
    ].astype(str)
    materialized[query_column] = (
        materialized[query_column].fillna("").astype(str)
    )
    if materialized[query_id_column].duplicated().any():
        raise ValueError("query_id values must be unique.")
    if deduplication_mode == "casefold":
        materialized["_query_norm"] = materialized[query_column].map(
            normalize_rewrite
        )
    else:
        materialized["_query_norm"] = materialized[query_column]
    canonical = (
        materialized[["_query_norm", query_column]]
        .drop_duplicates("_query_norm", keep="first")
        .sort_values("_query_norm", kind="mergesort")
        .reset_index(drop=True)
    )
    nonempty = canonical.loc[canonical["_query_norm"].ne("")].copy()

    warmup_count = (
        min(int(warmup_queries), len(nonempty))
        if measure_latency
        else 0
    )
    if warmup_count:
        warmup_texts = nonempty[query_column].iloc[:warmup_count].tolist()
        _synchronize_retriever(retriever)
        _search_retriever_batch(
            retriever,
            warmup_texts,
            top_k=min(int(top_k), 10),
            include_raw=False,
        )
        _synchronize_retriever(retriever)

    hits_by_norm: dict[str, list[dict[str, Any]]] = {"": []}
    latency_rows: list[dict[str, Any]] = []
    starts = range(0, len(nonempty), int(batch_size))
    for batch_index, start in progress_iter(
        enumerate(starts),
        total=(
            (len(nonempty) + int(batch_size) - 1) // int(batch_size)
        ),
        desc=progress_desc,
        unit="batch",
        enabled=progress,
    ):
        batch = nonempty.iloc[start : start + int(batch_size)]
        texts = batch[query_column].tolist()
        cuda_state = (
            _retrieval_cuda_state(retriever)
            if measure_latency
            else None
        )
        if cuda_state is not None:
            torch, device = cuda_state
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter() if measure_latency else None
        found = _search_retriever_batch(
            retriever,
            texts,
            top_k=int(top_k),
            include_raw=False,
        )
        if measure_latency:
            _synchronize_retriever(retriever)
            assert started is not None
            elapsed = time.perf_counter() - started
        if len(found) != len(batch):
            raise RuntimeError("Retriever returned a different batch size.")
        for norm, hits in zip(
            batch["_query_norm"].astype(str),
            found,
            strict=True,
        ):
            hits_by_norm[norm] = [dict(hit) for hit in hits]
        if measure_latency:
            latency_rows.append(
                {
                    "batch_index": int(batch_index),
                    "query_count": int(len(batch)),
                    "wall_time_seconds": float(elapsed),
                    "seconds_per_query": (
                        float(elapsed) / len(batch) if len(batch) else None
                    ),
                    "queries_per_second": (
                        len(batch) / float(elapsed) if elapsed > 0 else None
                    ),
                    "cuda_peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if cuda_state is not None
                        else None
                    ),
                    "cuda_peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(device))
                        if cuda_state is not None
                        else None
                    ),
                }
            )

    hits_by_query_id = {
        str(row[query_id_column]): hits_by_norm[str(row["_query_norm"])]
        for _, row in materialized.iterrows()
    }
    hit_rows = [
        {
            "query_id": query_id,
            "num_hits": len(hits),
            "top_docid": str(hits[0]["docid"]) if hits else None,
            "top_score": float(hits[0]["score"])
            if hits and hits[0].get("score") is not None
            else None,
        }
        for query_id, hits in hits_by_query_id.items()
    ]
    latency_per_batch = pd.DataFrame(latency_rows)
    latency_summary = (
        _summarize_retrieval_latency(
            latency_per_batch,
            requested_queries=len(materialized),
            unique_queries=len(canonical),
            warmup_queries=warmup_count,
        )
        if measure_latency
        else pd.DataFrame()
    )
    return QueryRetrievalResult(
        queries=materialized.drop(columns=["_query_norm"]),
        hits=pd.DataFrame(hit_rows),
        hits_by_query_id=hits_by_query_id,
        latency_per_batch=latency_per_batch,
        latency_summary=latency_summary,
    )


def run_query_retrieval_to_store(
    *,
    queries: pd.DataFrame,
    retriever: Any,
    store_path: Path | str,
    query_id_column: str = "query_id",
    query_column: str = "query",
    top_k: int = 1_000,
    retrieval_batch_size: int = 64,
    store_chunk_size: int = 512,
    measure_latency: bool = True,
    deduplication_mode: str = "casefold",
    progress: bool = True,
    progress_desc: str = "retrieve fixed queries",
) -> QueryRetrievalStoreResult:
    """Retrieve materialized queries incrementally into SQLite.

    ``measure_latency`` is forwarded to each retrieval chunk.
    """

    if deduplication_mode not in {"casefold", "exact"}:
        raise ValueError(
            "deduplication_mode must be 'casefold' or 'exact'."
        )
    if min(
        int(top_k),
        int(retrieval_batch_size),
        int(store_chunk_size),
    ) < 1:
        raise ValueError(
            "top_k, retrieval_batch_size, and store_chunk_size must be positive."
        )
    required = {str(query_id_column), str(query_column)}
    missing = required.difference(queries.columns)
    if missing:
        raise KeyError(
            "Materialized query columns missing: "
            + ", ".join(sorted(missing))
        )
    frame = queries.loc[:, [query_id_column, query_column]].copy()
    frame[query_id_column] = frame[query_id_column].astype(str)
    if frame[query_id_column].duplicated().any():
        raise ValueError("query_id values must be unique.")
    target = Path(store_path).expanduser().resolve()
    with QueryRankingStore(target) as store:
        existing = store.existing_ids(
            frame[query_id_column].astype(str).tolist()
        )
        pending = frame.loc[
            ~frame[query_id_column].isin(existing)
        ].reset_index(drop=True)
        iterator = progress_iter(
            range(0, len(pending), int(store_chunk_size)),
            total=(
                (len(pending) + int(store_chunk_size) - 1)
                // int(store_chunk_size)
            ),
            desc=progress_desc,
            unit="chunk",
            enabled=progress,
        )
        for start in iterator:
            chunk = pending.iloc[
                start : start + int(store_chunk_size)
            ]
            result = run_query_retrieval_pipeline(
                queries=chunk,
                retriever=retriever,
                query_id_column=query_id_column,
                query_column=query_column,
                top_k=int(top_k),
                batch_size=int(retrieval_batch_size),
                warmup_queries=0,
                measure_latency=measure_latency,
                deduplication_mode=deduplication_mode,
                progress=False,
            )
            store.put_many(
                result.hits_by_query_id,
                top_k=int(top_k),
            )
        stored_queries = len(store)
    return QueryRetrievalStoreResult(
        store_path=target,
        requested_queries=len(frame),
        reused_queries=len(existing),
        retrieved_queries=len(pending),
        stored_queries=stored_queries,
    )


def merge_query_ranking_stores(
    store_paths: Sequence[Path | str],
    *,
    output_path: Path | str,
    top_k: int,
    progress: bool = True,
) -> QueryRetrievalStoreResult:
    """Merge disjoint-shard ranking stores by raw retrieval score."""

    if len(store_paths) < 2:
        raise ValueError("At least two ranking stores are required.")
    inputs = [
        QueryRankingStore(path, read_only=True)
        for path in store_paths
    ]
    try:
        expected_ids = set(iter(inputs[0]))
        for store in inputs[1:]:
            if set(iter(store)) != expected_ids:
                raise ValueError(
                    "Partial ranking stores contain different query IDs."
                )
        target = Path(output_path).expanduser().resolve()
        temporary = target.with_name(f".{target.name}.tmp")
        if temporary.exists():
            temporary.unlink()
        with QueryRankingStore(temporary) as output:
            iterator = progress_iter(
                sorted(expected_ids),
                total=len(expected_ids),
                desc="merge partial query rankings",
                unit="query",
                enabled=progress,
            )
            batch: dict[str, list[dict[str, Any]]] = {}
            for query_id in iterator:
                merged = merge_partial_retrieval_hits(
                    [
                        {query_id: store[query_id]}
                        for store in inputs
                    ],
                    top_k=int(top_k),
                )[query_id]
                batch[query_id] = merged
                if len(batch) >= 256:
                    output.put_many(batch, top_k=int(top_k))
                    batch.clear()
            if batch:
                output.put_many(batch, top_k=int(top_k))
            stored_queries = len(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(target)
    finally:
        for store in inputs:
            store.close()
    return QueryRetrievalStoreResult(
        store_path=target,
        requested_queries=len(expected_ids),
        reused_queries=0,
        retrieved_queries=len(expected_ids),
        stored_queries=stored_queries,
    )


def _pack_query_hits(
    hits: Sequence[Mapping[str, Any]],
) -> bytes:
    payload = [
        [
            str(hit["docid"]),
            (
                float(hit["score"])
                if hit.get("score") is not None
                else None
            ),
        ]
        for hit in hits
    ]
    return zlib.compress(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8"),
        level=6,
    )


def _unpack_query_hits(blob: bytes) -> list[dict[str, Any]]:
    payload = json.loads(zlib.decompress(blob).decode("utf-8"))
    rows: list[dict[str, Any]] = []
    for rank, (docid, score) in enumerate(payload, start=1):
        row: dict[str, Any] = {
            "docid": str(docid),
            "rank": rank,
        }
        if score is not None:
            row["score"] = float(score)
        rows.append(row)
    return rows


def run_itercqr_retrieval_pipeline(
    *,
    dataloader: Any,
    rewriter: Any,
    retriever: Any,
    top_k: int = 100,
    max_batches: int | None = None,
    include_raw: bool = False,
    normalize_scores: bool = True,
    score_norm_method: str = "minmax",
    progress: bool = True,
    progress_desc: str = "IterCQR retrieval",
) -> RewriteRetrievePipelineResult:
    """Run dataloader -> IterCQR rewrite -> retrieval.

    This pipeline intentionally does not evaluate. Use
    ``evaluate_retrieval_pipeline`` on the returned result for MRR, nDCG, R@k,
    and bootstrap confidence intervals.
    """

    rewrite_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    hit_rows: list[dict[str, Any]] = []
    hits_by_sample_id: dict[str, list[dict[str, Any]]] = {}
    rewrite_efficiency_rows: list[dict[str, Any]] = []

    total_batches = _bounded_len(dataloader, max_batches)
    batches = progress_iter(
        enumerate(dataloader),
        total=total_batches,
        desc=progress_desc,
        unit="batch",
        enabled=progress,
    )
    for batch_index, batch in batches:
        if max_batches is not None and batch_index >= max_batches:
            break

        rewrites, efficiency_row = _rewrite_batch_with_efficiency_metrics(
            rewriter,
            batch,
            batch_index=batch_index,
        )
        rewrite_efficiency_rows.append(efficiency_row)
        hits_batch = _search_retriever_batch(
            retriever,
            rewrites,
            top_k=top_k,
            include_raw=include_raw,
        )

        for row_index, (rewrite, hits) in enumerate(zip(rewrites, hits_batch, strict=True)):
            sample_id = str(batch["bt_sample_ids"][row_index])
            metadata = batch["bt_metadata"][row_index]
            relevant_docids = [
                str(docid)
                for docid in metadata.get("positive_ctx_passage_ids", [])
            ]
            if normalize_scores:
                hits = normalize_retrieval_scores(
                    hits,
                    method=score_norm_method,
                )
            hits_by_sample_id[sample_id] = hits

            rewrite_rows.append(
                {
                    "sample_id": sample_id,
                    "split": metadata.get("split"),
                    "conv_id": metadata.get("conv_id"),
                    "turn_id": metadata.get("turn_id"),
                    "question": batch["bt_questions"][row_index],
                    "rewrite": rewrite,
                    "input_length": int(batch["bt_input_lengths"][row_index]),
                    "positive_ctx_passage_ids": relevant_docids,
                }
            )
            retrieval_rows.append(
                {
                    "sample_id": sample_id,
                    "split": metadata.get("split"),
                    "conv_id": metadata.get("conv_id"),
                    "turn_id": metadata.get("turn_id"),
                    "query": rewrite,
                    "top_k": top_k,
                    "num_hits": len(hits),
                    "top_docid": hits[0]["docid"] if hits else None,
                    "top_score": hits[0]["score"] if hits else None,
                    "top_score_norm": hits[0].get("score_norm") if hits else None,
                }
            )
            for hit in hits:
                hit_rows.append({"sample_id": sample_id, **hit})

    return RewriteRetrievePipelineResult(
        rewrites=pd.DataFrame(rewrite_rows),
        retrievals=pd.DataFrame(retrieval_rows),
        hits=pd.DataFrame(hit_rows),
        hits_by_sample_id=hits_by_sample_id,
        rewrite_efficiency_per_batch=pd.DataFrame(rewrite_efficiency_rows),
        rewrite_efficiency_summary=_summarize_rewrite_efficiency(rewrite_efficiency_rows),
    )


def run_itercqr_rewrite_efficiency_benchmark(
    *,
    dataloader: Any,
    rewriter: Any,
    max_batches: int | None = None,
    warmup_batches: int = 0,
    progress: bool = True,
    progress_desc: str = "IterCQR rewrite efficiency",
) -> RewriteEfficiencyBenchmarkResult:
    """Run only IterCQR rewriting and collect input/runtime efficiency metrics."""

    rewrite_rows: list[dict[str, Any]] = []
    rewrite_efficiency_rows: list[dict[str, Any]] = []

    total_batches = _bounded_len(dataloader, max_batches)
    batches = progress_iter(
        enumerate(dataloader),
        total=total_batches,
        desc=progress_desc,
        unit="batch",
        enabled=progress,
    )
    for batch_index, batch in batches:
        if max_batches is not None and batch_index >= max_batches:
            break

        rewrites, efficiency_row = _rewrite_batch_with_efficiency_metrics(
            rewriter,
            batch,
            batch_index=batch_index,
            is_warmup=batch_index < int(warmup_batches),
        )
        rewrite_efficiency_rows.append(efficiency_row)

        batch_lengths = _batch_input_lengths(batch)
        for row_index, rewrite in enumerate(rewrites):
            metadata = batch["bt_metadata"][row_index]
            rewrite_rows.append(
                {
                    "sample_id": str(batch["bt_sample_ids"][row_index]),
                    "split": metadata.get("split"),
                    "conv_id": metadata.get("conv_id"),
                    "turn_id": metadata.get("turn_id"),
                    "question": batch["bt_questions"][row_index],
                    "rewrite": rewrite,
                    "input_length": int(batch_lengths[row_index]),
                    "positive_ctx_passage_ids": [
                        str(docid)
                        for docid in metadata.get("positive_ctx_passage_ids", [])
                    ],
                }
            )

    return RewriteEfficiencyBenchmarkResult(
        rewrites=pd.DataFrame(rewrite_rows),
        rewrite_efficiency_per_batch=pd.DataFrame(rewrite_efficiency_rows),
        rewrite_efficiency_summary=_summarize_rewrite_efficiency(rewrite_efficiency_rows),
    )


def export_partial_retrieval_hits(
    retrieval_result: RewriteRetrievePipelineResult,
    output_file: Path | str,
    *,
    shard_session: str,
    top_k: int,
) -> PartialRetrievalDumpResult:
    """Write raw partial retrieval hits as JSONL for later score-based merging."""

    target = Path(output_file).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rewrites = retrieval_result.rewrites.set_index("sample_id", drop=False)
    hit_count = 0
    with target.open("w", encoding="utf-8") as handle:
        for sample_id, hits in retrieval_result.hits_by_sample_id.items():
            query = None
            if sample_id in rewrites.index:
                query = rewrites.loc[sample_id].get("rewrite")
            for hit in hits[:top_k]:
                row = {
                    "sample_id": str(sample_id),
                    "query": query,
                    "docid": str(hit["docid"]),
                    "score": float(hit["score"]),
                    "rank": int(hit.get("rank", hit_count + 1)),
                    "shard_session": shard_session,
                    "top_k": int(top_k),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                hit_count += 1
    return PartialRetrievalDumpResult(
        output_file=target,
        shard_session=shard_session,
        top_k=int(top_k),
        query_count=len(retrieval_result.hits_by_sample_id),
        hit_count=hit_count,
    )


def load_partial_retrieval_hits(path: Path | str) -> dict[str, list[dict[str, Any]]]:
    """Load a JSONL partial-hit dump grouped by sample id."""

    source = Path(path).expanduser().resolve()
    hits_by_sample_id: dict[str, list[dict[str, Any]]] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = str(row["sample_id"])
            hits_by_sample_id.setdefault(sample_id, []).append(
                {
                    "rank": int(row.get("rank", 0)),
                    "docid": str(row["docid"]),
                    "score": float(row["score"]),
                    "shard_session": row.get("shard_session"),
                }
            )
    return hits_by_sample_id


def merge_partial_retrieval_hits(
    partial_hits: list[dict[str, list[dict[str, Any]]]],
    *,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    """Merge partial hit maps by raw score and assign global ranks."""

    sample_ids = sorted({sample_id for hit_map in partial_hits for sample_id in hit_map})
    merged: dict[str, list[dict[str, Any]]] = {}
    for sample_id in sample_ids:
        by_docid: dict[str, dict[str, Any]] = {}
        for hit_map in partial_hits:
            for hit in hit_map.get(sample_id, []):
                docid = str(hit["docid"])
                score = float(hit["score"])
                previous = by_docid.get(docid)
                if previous is None or score > float(previous["score"]):
                    by_docid[docid] = {
                        "docid": docid,
                        "score": score,
                        "shard_session": hit.get("shard_session"),
                    }
        ranked = sorted(by_docid.values(), key=lambda row: (-float(row["score"]), row["docid"]))
        merged[sample_id] = [
            {**hit, "rank": rank}
            for rank, hit in enumerate(ranked[:top_k], start=1)
        ]
    return merged


def merge_partial_retrieval_dump_files(
    dump_files: list[Path | str],
    *,
    top_k: int,
    output_file: Path | str | None = None,
) -> PartialRetrievalMergeResult:
    partial_hits = [load_partial_retrieval_hits(path) for path in dump_files]
    merged = merge_partial_retrieval_hits(partial_hits, top_k=top_k)
    target = Path(output_file).expanduser().resolve() if output_file is not None else None
    hit_count = sum(len(hits) for hits in merged.values())
    if target is not None:
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            for sample_id, hits in merged.items():
                for hit in hits:
                    handle.write(
                        json.dumps({"sample_id": sample_id, **hit}, ensure_ascii=False)
                        + "\n"
                    )
    return PartialRetrievalMergeResult(
        hits_by_sample_id=merged,
        output_file=target,
        top_k=int(top_k),
        query_count=len(merged),
        hit_count=hit_count,
    )


def _rewrite_batch_with_efficiency_metrics(
    rewriter: Any,
    batch: dict[str, Any],
    *,
    batch_index: int,
    is_warmup: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    lengths = _batch_input_lengths(batch)
    stats = _input_length_stats(lengths)
    padded_input_length = _batch_padded_input_length(batch, lengths)
    batch_size = len(lengths)
    real_input_tokens = sum(lengths)
    padded_input_tokens = batch_size * padded_input_length
    attention_pair_work = batch_size * padded_input_length * padded_input_length

    cuda_state = _rewrite_cuda_state(rewriter)
    cuda_allocated_before_gb = None
    cuda_allocated_after_gb = None
    cuda_peak_allocated_gb = None
    cuda_peak_delta_gb = None
    cuda_peak_reserved_gb = None
    if cuda_state is not None:
        torch, device = cuda_state
        torch.cuda.synchronize(device)
        cuda_allocated_before = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        cuda_allocated_before_gb = _bytes_to_gb(cuda_allocated_before)

    started = time.perf_counter()
    rewrites = rewriter.rewrite_batch(batch)
    if cuda_state is not None:
        torch, device = cuda_state
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    if cuda_state is not None:
        torch, device = cuda_state
        cuda_allocated_after = torch.cuda.memory_allocated(device)
        cuda_peak_allocated = torch.cuda.max_memory_allocated(device)
        cuda_peak_reserved = torch.cuda.max_memory_reserved(device)
        cuda_allocated_after_gb = _bytes_to_gb(cuda_allocated_after)
        cuda_peak_allocated_gb = _bytes_to_gb(cuda_peak_allocated)
        cuda_peak_delta_gb = _bytes_to_gb(max(0, cuda_peak_allocated - cuda_allocated_before))
        cuda_peak_reserved_gb = _bytes_to_gb(cuda_peak_reserved)

    output_lengths = [
        len(
            rewriter.tokenizer.encode(
                str(rewrite),
                add_special_tokens=True,
            )
        )
        for rewrite in rewrites
    ]

    row = {
        "batch_index": int(batch_index),
        "is_warmup": bool(is_warmup),
        "batch_size": int(batch_size),
        "min_input_length": stats["min"],
        "mean_input_length": stats["mean"],
        "median_input_length": stats["median"],
        "max_input_length": stats["max"],
        "padded_input_length": int(padded_input_length),
        "real_input_tokens": int(real_input_tokens),
        "padded_input_tokens": int(padded_input_tokens),
        "padding_tokens": int(padded_input_tokens - real_input_tokens),
        "padding_fraction": (
            (padded_input_tokens - real_input_tokens) / padded_input_tokens
            if padded_input_tokens
            else 0.0
        ),
        "attention_pair_work": int(attention_pair_work),
        "output_tokens": int(sum(output_lengths)),
        "mean_output_tokens": (
            float(sum(output_lengths) / len(output_lengths))
            if output_lengths
            else 0.0
        ),
        "encoder_forward_calls": 1,
        "encoded_sequences": int(batch_size),
        "decoder_sequences": int(batch_size),
        "rewrite_wall_time_seconds": float(elapsed),
        "examples_per_second": batch_size / elapsed if elapsed > 0 else None,
        "padded_tokens_per_second": padded_input_tokens / elapsed if elapsed > 0 else None,
        "cuda_allocated_gb_before": cuda_allocated_before_gb,
        "cuda_allocated_gb_after": cuda_allocated_after_gb,
        "cuda_peak_allocated_gb": cuda_peak_allocated_gb,
        "cuda_peak_delta_gb": cuda_peak_delta_gb,
        "cuda_peak_reserved_gb": cuda_peak_reserved_gb,
    }
    return rewrites, row


def _batch_input_lengths(batch: dict[str, Any]) -> list[int]:
    lengths = batch["bt_input_lengths"]
    if hasattr(lengths, "detach"):
        lengths = lengths.detach().cpu().tolist()
    elif hasattr(lengths, "tolist"):
        lengths = lengths.tolist()
    return [int(length) for length in lengths]


def _batch_padded_input_length(batch: dict[str, Any], lengths: list[int]) -> int:
    input_ids = batch.get("bt_input_ids")
    shape = getattr(input_ids, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1])
    return max(lengths) if lengths else 0


def _input_length_stats(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    ordered = sorted(lengths)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median_value = float(ordered[midpoint])
    else:
        median_value = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {
        "min": int(ordered[0]),
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(median_value),
        "max": int(ordered[-1]),
    }


def _rewrite_cuda_state(rewriter: Any) -> tuple[Any, Any] | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None

    if not torch.cuda.is_available():
        return None

    device = getattr(rewriter, "device", None)
    if device is None:
        cuda_device = torch.device("cuda", torch.cuda.current_device())
    else:
        cuda_device = torch.device(device)
        if cuda_device.type != "cuda":
            return None
        if cuda_device.index is None:
            cuda_device = torch.device("cuda", torch.cuda.current_device())
    return torch, cuda_device


def _bytes_to_gb(num_bytes: int) -> float:
    return float(num_bytes) / (1000 ** 3)


def _summarize_rewrite_efficiency(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            [
                {
                    "num_examples": 0,
                    "num_batches": 0,
                    "total_real_input_tokens": 0,
                    "total_padded_input_tokens": 0,
                    "total_padding_tokens": 0,
                    "total_attention_pair_work": 0,
                    "total_output_tokens": 0,
                    "encoder_forward_calls": 0,
                    "decoder_sequences": 0,
                    "mean_final_input_length": 0.0,
                    "mean_padded_input_length": 0.0,
                    "padding_fraction": 0.0,
                    "total_rewrite_wall_time_seconds": 0.0,
                    "mean_examples_per_second": None,
                    "mean_padded_tokens_per_second": None,
                    "peak_cuda_allocated_gb": None,
                    "peak_cuda_delta_gb": None,
                    "peak_cuda_reserved_gb": None,
                    "median_seconds_per_query": None,
                    "p95_seconds_per_query": None,
                    "p99_seconds_per_query": None,
                }
            ]
        )

    productive = [row for row in rows if not bool(row.get("is_warmup", False))]
    if not productive:
        productive = list(rows)
    total_examples = sum(int(row["batch_size"]) for row in productive)
    total_real_input_tokens = sum(int(row["real_input_tokens"]) for row in productive)
    total_padded_input_tokens = sum(int(row["padded_input_tokens"]) for row in productive)
    total_padding_tokens = sum(int(row["padding_tokens"]) for row in productive)
    total_attention_pair_work = sum(int(row["attention_pair_work"]) for row in productive)
    total_output_tokens = sum(int(row.get("output_tokens", 0)) for row in productive)
    total_encoder_calls = sum(int(row.get("encoder_forward_calls", 1)) for row in productive)
    total_decoder_sequences = sum(
        int(row.get("decoder_sequences", row["batch_size"]))
        for row in productive
    )
    total_wall_time = sum(float(row["rewrite_wall_time_seconds"]) for row in productive)
    peak_cuda_allocated_gb = _max_optional_float(
        row["cuda_peak_allocated_gb"] for row in productive
    )
    peak_cuda_delta_gb = _max_optional_float(
        row["cuda_peak_delta_gb"] for row in productive
    )
    peak_cuda_reserved_gb = _max_optional_float(
        row.get("cuda_peak_reserved_gb") for row in productive
    )
    seconds_per_query = [
        float(row["rewrite_wall_time_seconds"]) / int(row["batch_size"])
        for row in productive
        if int(row["batch_size"]) > 0
    ]
    return pd.DataFrame(
        [
            {
                "num_examples": int(total_examples),
                "num_batches": int(len(productive)),
                "warmup_batches": int(len(rows) - len(productive)),
                "total_real_input_tokens": int(total_real_input_tokens),
                "total_padded_input_tokens": int(total_padded_input_tokens),
                "total_padding_tokens": int(total_padding_tokens),
                "total_attention_pair_work": int(total_attention_pair_work),
                "total_output_tokens": int(total_output_tokens),
                "encoder_forward_calls": int(total_encoder_calls),
                "decoder_sequences": int(total_decoder_sequences),
                "mean_final_input_length": (
                    total_real_input_tokens / total_examples if total_examples else 0.0
                ),
                "mean_padded_input_length": (
                    total_padded_input_tokens / total_examples if total_examples else 0.0
                ),
                "padding_fraction": (
                    total_padding_tokens / total_padded_input_tokens
                    if total_padded_input_tokens
                    else 0.0
                ),
                "total_rewrite_wall_time_seconds": float(total_wall_time),
                "mean_examples_per_second": (
                    total_examples / total_wall_time if total_wall_time > 0 else None
                ),
                "mean_padded_tokens_per_second": (
                    total_padded_input_tokens / total_wall_time
                    if total_wall_time > 0
                    else None
                ),
                "peak_cuda_allocated_gb": peak_cuda_allocated_gb,
                "peak_cuda_delta_gb": peak_cuda_delta_gb,
                "peak_cuda_reserved_gb": peak_cuda_reserved_gb,
                "median_seconds_per_query": _quantile_or_none(
                    seconds_per_query,
                    0.5,
                ),
                "p95_seconds_per_query": _quantile_or_none(
                    seconds_per_query,
                    0.95,
                ),
                "p99_seconds_per_query": _quantile_or_none(
                    seconds_per_query,
                    0.99,
                ),
            }
        ]
    )


def _max_optional_float(values: Any) -> float | None:
    cleaned = [float(value) for value in values if value is not None]
    return max(cleaned) if cleaned else None


def _quantile_or_none(
    values: Sequence[float],
    quantile: float,
) -> float | None:
    if not values:
        return None
    return float(pd.Series(values, dtype=float).quantile(quantile))


def _summarize_retrieval_latency(
    rows: pd.DataFrame,
    *,
    requested_queries: int,
    unique_queries: int,
    warmup_queries: int,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            [{
                "requested_queries": int(requested_queries),
                "unique_normalized_queries": int(unique_queries),
                "warmup_queries": int(warmup_queries),
                "batches": 0,
                "wall_time_seconds": 0.0,
                "mean_seconds_per_query": None,
                "median_seconds_per_query": None,
                "p95_seconds_per_query": None,
                "p99_seconds_per_query": None,
                "queries_per_second": None,
                "peak_cuda_allocated_bytes": None,
                "peak_cuda_reserved_bytes": None,
            }]
        )
    seconds = rows["seconds_per_query"].dropna().astype(float)
    total_seconds = float(rows["wall_time_seconds"].sum())
    timed_queries = int(rows["query_count"].sum())
    return pd.DataFrame(
        [{
            "requested_queries": int(requested_queries),
            "unique_normalized_queries": int(unique_queries),
            "warmup_queries": int(warmup_queries),
            "batches": int(len(rows)),
            "wall_time_seconds": total_seconds,
            "mean_seconds_per_query": (
                total_seconds / timed_queries if timed_queries else None
            ),
            "median_seconds_per_query": (
                float(seconds.median()) if not seconds.empty else None
            ),
            "p95_seconds_per_query": (
                float(seconds.quantile(0.95))
                if not seconds.empty
                else None
            ),
            "p99_seconds_per_query": (
                float(seconds.quantile(0.99))
                if not seconds.empty
                else None
            ),
            "queries_per_second": (
                timed_queries / total_seconds
                if total_seconds > 0
                else None
            ),
            "peak_cuda_allocated_bytes": _max_optional_float(
                rows["cuda_peak_allocated_bytes"]
            ),
            "peak_cuda_reserved_bytes": _max_optional_float(
                rows["cuda_peak_reserved_bytes"]
            ),
        }]
    )


def _retrieval_cuda_state(retriever: Any) -> tuple[Any, Any] | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    if not torch.cuda.is_available():
        return None
    raw_device = getattr(retriever, "device", None)
    if callable(raw_device):
        raw_device = raw_device()
    if raw_device is None:
        return None
    device = torch.device(raw_device)
    if device.type != "cuda":
        return None
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return torch, device


def _synchronize_retriever(retriever: Any) -> None:
    state = _retrieval_cuda_state(retriever)
    if state is not None:
        torch, device = state
        torch.cuda.synchronize(device)


def _search_retriever_batch(
    retriever: Any,
    queries: list[str],
    *,
    top_k: int,
    include_raw: bool,
) -> list[list[dict[str, Any]]]:
    if hasattr(retriever, "search_batch") and not include_raw:
        return retriever.search_batch(queries, top_k=top_k)
    return [
        retriever.search(query, top_k=top_k, include_raw=include_raw)
        for query in queries
    ]


def _bounded_len(dataloader: Any, max_batches: int | None) -> int | None:
    try:
        total = len(dataloader)
    except TypeError:
        total = None
    if max_batches is None:
        return total
    if total is None:
        return max_batches
    return min(total, max_batches)
