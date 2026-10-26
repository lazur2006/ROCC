"""Persistent IterCQR -> BM25 evaluation shared by ROCC notebooks."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .evaluation import retrieval_metrics
from .itercqr_components import load_itercqr_rewriter
from .progress import get_tqdm
from .retrievers import load_bm25_retriever


@dataclass(frozen=True)
class CachedIterCQRBM25Config:
    """Runtime and cache configuration for the fixed IterCQR/BM25 path."""

    cache_db: Path | str
    model_dir: Path | str
    bm25_index_dir: Path | str
    device: str
    rewrite_batch_size: int = 16
    retrieval_batch_size: int = 64
    retrieval_workers: int = 1
    retrieval_top_k: int = 1_000
    eval_ks: tuple[int, ...] = (3, 10, 100, 1000)
    k1: float = 0.9
    b: float = 0.4
    generation_max_length: int = 32
    num_beams: int = 1
    do_sample: bool = False
    progress: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_db", Path(self.cache_db))
        object.__setattr__(self, "model_dir", Path(self.model_dir))
        object.__setattr__(
            self,
            "bm25_index_dir",
            Path(self.bm25_index_dir),
        )
        object.__setattr__(
            self,
            "eval_ks",
            tuple(int(k) for k in self.eval_ks),
        )
        if self.rewrite_batch_size < 1:
            raise ValueError("rewrite_batch_size muss mindestens 1 sein.")
        if self.retrieval_batch_size < 1:
            raise ValueError(
                "retrieval_batch_size muss mindestens 1 sein."
            )
        if self.retrieval_top_k < 1:
            raise ValueError("retrieval_top_k muss mindestens 1 sein.")


@dataclass
class CachedIterCQRBM25Result:
    """Frames and cache statistics produced by one pipeline run."""

    serialized_with_rewrite: pd.DataFrame
    retrieval_metric_pairs: pd.DataFrame
    evaluated: pd.DataFrame
    unique_inputs: int
    rewrite_cache_misses: int
    unique_rewrites: int
    retrieval_cache_misses: int
    scored_query_rewrites: int
    rewriter_loaded: bool
    retriever_loaded: bool


@dataclass
class CachedIterCQRRewriteResult:
    """Rewrite-only cache result without a retriever dependency."""

    serialized_with_rewrite: pd.DataFrame
    unique_inputs: int
    rewrite_cache_misses: int
    unique_rewrites: int
    rewriter_loaded: bool


@dataclass
class CachedBM25Result:
    """Frames and cache statistics for direct BM25 queries."""

    evaluated: pd.DataFrame
    unique_queries: int
    retrieval_cache_misses: int
    scored_query_pairs: int
    retriever_loaded: bool


class CachedIterCQRBM25Pipeline:
    """Deduplicate and persist token inputs, rewrites, and BM25 rankings."""

    _TABLE_KEYS = {
        "rewrites": "input_key",
        "retrievals": "rewrite_key",
    }

    def __init__(
        self,
        *,
        config: CachedIterCQRBM25Config,
        tokenizer: Any | None = None,
        rewriter_loader: Callable[..., Any] = load_itercqr_rewriter,
        retriever_loader: Callable[..., Any] = load_bm25_retriever,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self._rewriter_loader = rewriter_loader
        self._retriever_loader = retriever_loader
        self._rewriter: Any | None = None
        self._retriever: Any | None = None
        self._rewriter_loaded = False
        self._retriever_loaded = False
        self.model_cache_signature = _model_cache_signature(config)
        self.bm25_cache_signature = _bm25_cache_signature(config)
        Path(config.cache_db).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connection()):
            pass

    def input_key_for(self, token_ids: Sequence[int]) -> str:
        """Return the legacy cache key for one tokenized IterCQR input."""

        digest = hashlib.sha256()
        digest.update(self.model_cache_signature.encode("utf-8"))
        digest.update(b"\0")
        digest.update(pack_token_ids(token_ids))
        return digest.hexdigest()

    def rewrite_key_for(self, rewrite_norm: str) -> str:
        """Return the legacy cache key for one normalized BM25 query."""

        digest = hashlib.sha256()
        digest.update(self.bm25_cache_signature.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(rewrite_norm).encode("utf-8"))
        return digest.hexdigest()

    def register_token_inputs(
        self,
        token_sequences: Sequence[Sequence[int]],
    ) -> list[str]:
        """Persist token sequences and return their keys in input order."""

        normalized = [
            [int(token_id) for token_id in sequence]
            for sequence in token_sequences
        ]
        input_keys = [
            self.input_key_for(sequence) for sequence in normalized
        ]
        with closing(self._connection()) as connection:
            connection.executemany(
                """
                INSERT OR IGNORE INTO token_inputs
                (input_key, token_blob, input_length)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        input_key,
                        pack_token_ids(sequence),
                        len(sequence),
                    )
                    for input_key, sequence in zip(
                        input_keys,
                        normalized,
                        strict=True,
                    )
                ],
            )
            connection.commit()
        return input_keys

    def run(
        self,
        serialized_inputs: pd.DataFrame,
        *,
        gold_by_sample: Mapping[str, Iterable[Any]],
        metric_columns: Sequence[str] | None = None,
        progress_prefix: str = "",
    ) -> CachedIterCQRBM25Result:
        """Resolve cached rewrites/rankings and score unique query pairs."""

        required = {"sample_id", "input_key"}
        missing_columns = required.difference(serialized_inputs.columns)
        if missing_columns:
            raise KeyError(
                "serialized_inputs fehlen Spalten: "
                + ", ".join(sorted(missing_columns))
            )
        if serialized_inputs.empty:
            empty = serialized_inputs.copy()
            return CachedIterCQRBM25Result(
                serialized_with_rewrite=empty,
                retrieval_metric_pairs=pd.DataFrame(),
                evaluated=empty,
                unique_inputs=0,
                rewrite_cache_misses=0,
                unique_rewrites=0,
                retrieval_cache_misses=0,
                scored_query_rewrites=0,
                rewriter_loaded=False,
                retriever_loaded=False,
            )

        retriever_loaded_before = self._retriever_loaded
        rewrite_result = self.materialize_rewrites(
            serialized_inputs,
            progress_prefix=progress_prefix,
        )
        serialized_with_rewrite = (
            rewrite_result.serialized_with_rewrite
        )

        unique_rewrites = (
            serialized_with_rewrite[
                ["rewrite_key", "rewrite_norm"]
            ]
            .drop_duplicates("rewrite_key")
            .sort_values("rewrite_key")
            .reset_index(drop=True)
        )
        available_retrievals = self._cached_keys(
            "retrievals",
            progress_prefix=progress_prefix,
        )
        retrieval_misses = unique_rewrites.loc[
            ~unique_rewrites["rewrite_key"].isin(
                available_retrievals
            )
        ].copy()
        self._materialize_retrievals(
            retrieval_misses,
            progress_prefix=progress_prefix,
        )

        retrieval_metric_pairs = self._score_rankings(
            serialized_with_rewrite,
            gold_by_sample=gold_by_sample,
            metric_columns=metric_columns,
            progress_prefix=progress_prefix,
        )
        evaluated = serialized_with_rewrite.merge(
            retrieval_metric_pairs,
            on=["sample_id", "rewrite_key"],
            how="left",
            validate="many_to_one",
        )
        return CachedIterCQRBM25Result(
            serialized_with_rewrite=serialized_with_rewrite,
            retrieval_metric_pairs=retrieval_metric_pairs,
            evaluated=evaluated,
            unique_inputs=rewrite_result.unique_inputs,
            rewrite_cache_misses=(
                rewrite_result.rewrite_cache_misses
            ),
            unique_rewrites=len(unique_rewrites),
            retrieval_cache_misses=len(retrieval_misses),
            scored_query_rewrites=len(retrieval_metric_pairs),
            rewriter_loaded=rewrite_result.rewriter_loaded,
            retriever_loaded=(
                self._retriever_loaded and not retriever_loaded_before
            ),
        )

    def materialize_rewrites(
        self,
        serialized_inputs: pd.DataFrame,
        *,
        progress_prefix: str = "",
    ) -> CachedIterCQRRewriteResult:
        """Resolve cached IterCQR rewrites without running retrieval."""

        required = {"sample_id", "input_key"}
        missing_columns = required.difference(serialized_inputs.columns)
        if missing_columns:
            raise KeyError(
                "serialized_inputs fehlen Spalten: "
                + ", ".join(sorted(missing_columns))
            )
        if serialized_inputs.empty:
            return CachedIterCQRRewriteResult(
                serialized_with_rewrite=serialized_inputs.copy(),
                unique_inputs=0,
                rewrite_cache_misses=0,
                unique_rewrites=0,
                rewriter_loaded=False,
            )

        rewriter_loaded_before = self._rewriter_loaded
        unique_input_keys = (
            serialized_inputs["input_key"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        available_rewrites = self._cached_keys(
            "rewrites",
            progress_prefix=progress_prefix,
        )
        missing_input_keys = [
            key
            for key in unique_input_keys
            if key not in available_rewrites
        ]
        self._materialize_rewrites(
            missing_input_keys,
            progress_prefix=progress_prefix,
        )
        rewrite_lookup = self._load_rewrites(
            unique_input_keys,
            progress_prefix=progress_prefix,
        )
        serialized_with_rewrite = serialized_inputs.merge(
            rewrite_lookup,
            on="input_key",
            how="left",
            validate="many_to_one",
        )
        serialized_with_rewrite["rewrite_key"] = (
            serialized_with_rewrite["rewrite_norm"].map(
                self.rewrite_key_for
            )
        )
        return CachedIterCQRRewriteResult(
            serialized_with_rewrite=serialized_with_rewrite,
            unique_inputs=len(unique_input_keys),
            rewrite_cache_misses=len(missing_input_keys),
            unique_rewrites=int(
                serialized_with_rewrite["rewrite_key"].nunique()
            ),
            rewriter_loaded=(
                self._rewriter_loaded and not rewriter_loaded_before
            ),
        )

    def run_bm25(
        self,
        queries: pd.DataFrame,
        *,
        gold_by_sample: Mapping[str, Iterable[Any]],
        query_column: str = "query",
        metric_columns: Sequence[str] | None = None,
        progress_prefix: str = "",
    ) -> CachedBM25Result:
        """Retrieve and score already materialized BM25 query strings."""

        required = {"sample_id", query_column}
        missing_columns = required.difference(queries.columns)
        if missing_columns:
            raise KeyError(
                "BM25-Queries fehlen Spalten: "
                + ", ".join(sorted(missing_columns))
            )
        materialized = queries.copy()
        materialized["rewrite_norm"] = materialized[
            query_column
        ].map(normalize_rewrite)
        materialized["rewrite_key"] = materialized[
            "rewrite_norm"
        ].map(self.rewrite_key_for)
        unique_queries = (
            materialized[["rewrite_key", "rewrite_norm"]]
            .drop_duplicates("rewrite_key")
            .sort_values("rewrite_key")
            .reset_index(drop=True)
        )
        available_retrievals = self._cached_keys(
            "retrievals",
            progress_prefix=progress_prefix,
        )
        retrieval_misses = unique_queries.loc[
            ~unique_queries["rewrite_key"].isin(available_retrievals)
        ].copy()
        retriever_loaded_before = self._retriever_loaded
        self._materialize_retrievals(
            retrieval_misses,
            progress_prefix=progress_prefix,
        )
        metric_pairs = self._score_rankings(
            materialized,
            gold_by_sample=gold_by_sample,
            metric_columns=metric_columns,
            progress_prefix=progress_prefix,
        )
        evaluated = materialized.merge(
            metric_pairs,
            on=["sample_id", "rewrite_key"],
            how="left",
            validate="many_to_one",
        )
        return CachedBM25Result(
            evaluated=evaluated,
            unique_queries=len(unique_queries),
            retrieval_cache_misses=len(retrieval_misses),
            scored_query_pairs=len(metric_pairs),
            retriever_loaded=(
                self._retriever_loaded and not retriever_loaded_before
            ),
        )

    def load_rankings(
        self,
        rewrite_keys: Iterable[str],
        *,
        progress_prefix: str = "",
    ) -> dict[str, list[str]]:
        """Load cached ranked document IDs for the requested rewrites."""

        keys = sorted({str(key) for key in rewrite_keys})
        if not keys:
            return {}

        rankings: dict[str, list[str]] = {}
        tqdm = get_tqdm()
        with closing(self._connection()) as connection, tqdm(
            total=len(keys),
            desc=_description(
                progress_prefix,
                "load cached rankings",
            ),
            unit="ranking",
            dynamic_ncols=True,
            disable=not self.config.progress,
        ) as progress:
            for key_batch in _chunks(keys, 500):
                placeholders = ",".join("?" for _ in key_batch)
                rows = connection.execute(
                    f"""
                    SELECT rewrite_key, docids_blob
                    FROM retrievals
                    WHERE rewrite_key IN ({placeholders})
                    """,
                    key_batch,
                ).fetchall()
                rankings.update(
                    {
                        str(key): unpack_docids(blob)
                        for key, blob in rows
                    }
                )
                progress.update(len(key_batch))

        missing = [key for key in keys if key not in rankings]
        if missing:
            raise KeyError(f"Retrievals fehlen: {missing[:3]}")
        return rankings

    def load_token_inputs(
        self,
        input_keys: Sequence[str],
    ) -> list[list[int]]:
        """Load registered token sequences in the requested order."""

        return self._fetch_token_sequences(
            [str(key) for key in input_keys]
        )

    def make_rewriter_batch(
        self,
        input_keys: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        """Build the canonical padded IterCQR batch for registered inputs."""

        return self._make_rewriter_batch(
            [str(key) for key in input_keys]
        )

    def close(self) -> None:
        """Close the BM25 runtime if this instance loaded it."""

        if self._retriever is not None:
            close = getattr(self._retriever, "close", None)
            if callable(close):
                close()
            self._retriever = None

    def _connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.config.cache_db,
            timeout=120,
        )
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS token_inputs (
                input_key TEXT PRIMARY KEY,
                token_blob BLOB NOT NULL,
                input_length INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rewrites (
                input_key TEXT PRIMARY KEY,
                rewrite TEXT NOT NULL,
                rewrite_norm TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrievals (
                rewrite_key TEXT PRIMARY KEY,
                rewrite_norm TEXT NOT NULL,
                docids_blob BLOB NOT NULL
            )
            """
        )
        connection.commit()
        return connection

    def _cached_keys(
        self,
        table: str,
        *,
        progress_prefix: str,
    ) -> set[str]:
        key_column = self._TABLE_KEYS.get(table)
        if key_column is None:
            raise ValueError(f"Unbekannte Cache-Tabelle: {table}")

        keys: set[str] = set()
        with closing(self._connection()) as connection:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            )
            cursor = connection.execute(
                f"SELECT {key_column} FROM {table}"
            )
            tqdm = get_tqdm()
            with tqdm(
                total=total,
                desc=_description(
                    progress_prefix,
                    f"load {table} cache keys",
                ),
                unit="key",
                dynamic_ncols=True,
                disable=not self.config.progress,
            ) as progress:
                while True:
                    rows = cursor.fetchmany(10_000)
                    if not rows:
                        break
                    keys.update(str(row[0]) for row in rows)
                    progress.update(len(rows))
        return keys

    def _fetch_token_sequences(
        self,
        input_keys: Sequence[str],
    ) -> list[list[int]]:
        rows: list[tuple[str, bytes]] = []
        with closing(self._connection()) as connection:
            for key_batch in _chunks(list(input_keys), 800):
                placeholders = ",".join("?" for _ in key_batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT input_key, token_blob
                        FROM token_inputs
                        WHERE input_key IN ({placeholders})
                        """,
                        key_batch,
                    ).fetchall()
                )
        token_by_key = {
            str(key): unpack_token_ids(blob) for key, blob in rows
        }
        missing = [
            str(key)
            for key in input_keys
            if str(key) not in token_by_key
        ]
        if missing:
            raise KeyError(f"Token-Inputs fehlen: {missing[:3]}")
        return [token_by_key[str(key)] for key in input_keys]

    def _make_rewriter_batch(
        self,
        input_keys: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError(
                "Für Rewrite-Cache-Misses wird ein Tokenizer benötigt."
            )
        sequences = self._fetch_token_sequences(input_keys)
        pad_token_id = int(self.tokenizer.pad_token_id)
        max_length = max(map(len, sequences))
        input_ids = [
            sequence
            + [pad_token_id] * (max_length - len(sequence))
            for sequence in sequences
        ]
        attention_mask = [
            [1] * len(sequence)
            + [0] * (max_length - len(sequence))
            for sequence in sequences
        ]
        return {
            "bt_input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "bt_attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
        }

    def _get_rewriter(self) -> Any:
        if self._rewriter is None:
            self._rewriter = self._rewriter_loader(
                model_dir=self.config.model_dir,
                device=self.config.device,
            )
            self._rewriter_loaded = True
        return self._rewriter

    def _get_retriever(self) -> Any:
        if self._retriever is None:
            self._retriever = self._retriever_loader(
                index_dir=self.config.bm25_index_dir,
                k1=self.config.k1,
                b=self.config.b,
                retrieval_workers=self.config.retrieval_workers,
            )
            self._retriever_loaded = True
        return self._retriever

    def _materialize_rewrites(
        self,
        missing_input_keys: Sequence[str],
        *,
        progress_prefix: str,
    ) -> None:
        if not missing_input_keys:
            return
        tqdm = get_tqdm()
        with tqdm(
            total=1,
            desc=_description(
                progress_prefix,
                "load IterCQR rewriter",
            ),
            unit="model",
            dynamic_ncols=True,
            disable=not self.config.progress,
        ) as progress:
            rewriter = self._get_rewriter()
            progress.update(1)

        with closing(self._connection()) as connection, tqdm(
            total=len(missing_input_keys),
            desc=_description(
                progress_prefix,
                "IterCQR rewrite cache misses",
            ),
            unit="input",
            dynamic_ncols=True,
            disable=not self.config.progress,
        ) as progress:
            for key_batch in _chunks(
                list(missing_input_keys),
                self.config.rewrite_batch_size,
            ):
                rewrites = rewriter.rewrite_batch(
                    self._make_rewriter_batch(key_batch)
                )
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO rewrites
                    (input_key, rewrite, rewrite_norm)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (
                            key,
                            str(rewrite),
                            normalize_rewrite(rewrite),
                        )
                        for key, rewrite in zip(
                            key_batch,
                            rewrites,
                            strict=True,
                        )
                    ],
                )
                connection.commit()
                progress.update(len(key_batch))

    def _load_rewrites(
        self,
        input_keys: Sequence[str],
        *,
        progress_prefix: str,
    ) -> pd.DataFrame:
        rows: list[tuple[str, str, str]] = []
        batches = list(_chunks(list(input_keys), 800))
        tqdm = get_tqdm()
        with closing(self._connection()) as connection:
            for key_batch in tqdm(
                batches,
                desc=_description(
                    progress_prefix,
                    "load rewrite cache",
                ),
                unit="batch",
                dynamic_ncols=True,
                disable=not self.config.progress,
            ):
                placeholders = ",".join("?" for _ in key_batch)
                rows.extend(
                    connection.execute(
                        f"""
                        SELECT input_key, rewrite, rewrite_norm
                        FROM rewrites
                        WHERE input_key IN ({placeholders})
                        """,
                        key_batch,
                    ).fetchall()
                )
        lookup = pd.DataFrame(
            rows,
            columns=["input_key", "rewrite", "rewrite_norm"],
        )
        found = set(lookup["input_key"].astype(str))
        missing = [key for key in input_keys if key not in found]
        if missing:
            raise KeyError(f"Rewrites fehlen: {missing[:3]}")
        return lookup

    def _materialize_retrievals(
        self,
        retrieval_misses: pd.DataFrame,
        *,
        progress_prefix: str,
    ) -> None:
        if retrieval_misses.empty:
            return
        tqdm = get_tqdm()
        with tqdm(
            total=1,
            desc=_description(
                progress_prefix,
                "load BM25 retriever",
            ),
            unit="index",
            dynamic_ncols=True,
            disable=not self.config.progress,
        ) as progress:
            retriever = self._get_retriever()
            progress.update(1)

        with closing(self._connection()) as connection, tqdm(
            total=len(retrieval_misses),
            desc=_description(
                progress_prefix,
                "BM25 retrieval cache misses",
            ),
            unit="rewrite",
            dynamic_ncols=True,
            disable=not self.config.progress,
        ) as progress:
            for start in range(
                0,
                len(retrieval_misses),
                self.config.retrieval_batch_size,
            ):
                batch = retrieval_misses.iloc[
                    start : start
                    + self.config.retrieval_batch_size
                ]
                queries = batch["rewrite_norm"].astype(str).tolist()
                hits_batch: list[list[dict[str, Any]]] = [
                    [] for _ in queries
                ]
                nonempty = [
                    index
                    for index, query in enumerate(queries)
                    if query
                ]
                if nonempty:
                    found = retriever.search_batch(
                        [queries[index] for index in nonempty],
                        top_k=self.config.retrieval_top_k,
                    )
                    for index, hits in zip(
                        nonempty,
                        found,
                        strict=True,
                    ):
                        hits_batch[index] = hits
                connection.executemany(
                    """
                    INSERT OR REPLACE INTO retrievals
                    (rewrite_key, rewrite_norm, docids_blob)
                    VALUES (?, ?, ?)
                    """,
                    [
                        (
                            str(row.rewrite_key),
                            str(row.rewrite_norm),
                            pack_docids(hits),
                        )
                        for row, hits in zip(
                            batch.itertuples(index=False),
                            hits_batch,
                            strict=True,
                        )
                    ],
                )
                connection.commit()
                progress.update(len(batch))

    def _score_rankings(
        self,
        serialized_with_rewrite: pd.DataFrame,
        *,
        gold_by_sample: Mapping[str, Iterable[Any]],
        metric_columns: Sequence[str] | None,
        progress_prefix: str,
    ) -> pd.DataFrame:
        metric_pairs = (
            serialized_with_rewrite[
                ["sample_id", "rewrite_key"]
            ]
            .drop_duplicates()
            .sort_values(["rewrite_key", "sample_id"])
            .reset_index(drop=True)
        )
        pairs_by_rewrite = {
            str(rewrite_key): group["sample_id"]
            .astype(str)
            .tolist()
            for rewrite_key, group in metric_pairs.groupby(
                "rewrite_key",
                sort=False,
            )
        }
        metric_rows: list[dict[str, Any]] = []
        rewrite_keys = list(pairs_by_rewrite)
        tqdm = get_tqdm()
        with closing(self._connection()) as connection, tqdm(
            total=len(metric_pairs),
            desc=_description(
                progress_prefix,
                "score retrieved rankings",
            ),
            unit="query-rewrite",
            dynamic_ncols=True,
            disable=not self.config.progress,
        ) as progress:
            for key_batch in _chunks(rewrite_keys, 500):
                placeholders = ",".join("?" for _ in key_batch)
                rows = connection.execute(
                    f"""
                    SELECT rewrite_key, docids_blob
                    FROM retrievals
                    WHERE rewrite_key IN ({placeholders})
                    """,
                    key_batch,
                ).fetchall()
                blob_by_key = {
                    str(key): blob for key, blob in rows
                }
                missing = [
                    key
                    for key in key_batch
                    if key not in blob_by_key
                ]
                if missing:
                    raise KeyError(
                        f"Retrievals fehlen: {missing[:3]}"
                    )
                for rewrite_key in key_batch:
                    docids = unpack_docids(
                        blob_by_key[rewrite_key]
                    )
                    hits = [
                        {"docid": docid, "rank": rank}
                        for rank, docid in enumerate(
                            docids,
                            start=1,
                        )
                    ]
                    sample_ids = pairs_by_rewrite[rewrite_key]
                    for sample_id in sample_ids:
                        metrics = retrieval_metrics(
                            hits,
                            gold_by_sample[str(sample_id)],
                            ks=self.config.eval_ks,
                        )
                        selected_metrics = (
                            metrics
                            if metric_columns is None
                            else {
                                metric: metrics[metric]
                                for metric in metric_columns
                            }
                        )
                        metric_rows.append(
                            {
                                "sample_id": str(sample_id),
                                "rewrite_key": rewrite_key,
                                **selected_metrics,
                            }
                        )
                    progress.update(len(sample_ids))
        return pd.DataFrame(metric_rows)


def normalize_rewrite(text: Any) -> str:
    """Normalize rewrites exactly as the shared NB03/NB04 cache."""

    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def pack_token_ids(token_ids: Sequence[int]) -> bytes:
    """Serialize token ids in the legacy little-endian int32 format."""

    return np.asarray(token_ids, dtype="<i4").tobytes()


def unpack_token_ids(blob: bytes) -> list[int]:
    """Deserialize the legacy token blob."""

    return np.frombuffer(
        blob,
        dtype="<i4",
    ).astype(np.int64).tolist()


def pack_docids(hits: Sequence[dict[str, Any]]) -> bytes:
    """Compress ranked document ids using the legacy JSON/zlib format."""

    payload = json.dumps(
        [str(hit["docid"]) for hit in hits],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return zlib.compress(payload, level=6)


def unpack_docids(blob: bytes) -> list[str]:
    """Decompress ranked document ids from the shared cache."""

    return list(
        json.loads(zlib.decompress(blob).decode("utf-8"))
    )


def _model_cache_signature(
    config: CachedIterCQRBM25Config,
) -> str:
    weights = Path(config.model_dir) / "pytorch_model.bin"
    stat = weights.stat()
    return json.dumps(
        {
            "model_dir": str(Path(config.model_dir).resolve()),
            "weights_size": stat.st_size,
            "weights_mtime_ns": stat.st_mtime_ns,
            "generation_max_length": (
                config.generation_max_length
            ),
            "num_beams": config.num_beams,
            "do_sample": config.do_sample,
        },
        sort_keys=True,
    )


def _bm25_cache_signature(
    config: CachedIterCQRBM25Config,
) -> str:
    return json.dumps(
        {
            "index_dir": str(
                Path(config.bm25_index_dir).resolve()
            ),
            "k1": config.k1,
            "b": config.b,
            "top_k": config.retrieval_top_k,
        },
        sort_keys=True,
    )


def _chunks(
    values: Sequence[str],
    size: int,
) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def _description(prefix: str, text: str) -> str:
    return f"{prefix} {text}".strip()


__all__ = [
    "CachedBM25Result",
    "CachedIterCQRBM25Config",
    "CachedIterCQRBM25Pipeline",
    "CachedIterCQRBM25Result",
    "CachedIterCQRRewriteResult",
    "normalize_rewrite",
    "pack_docids",
    "pack_token_ids",
    "unpack_docids",
    "unpack_token_ids",
]
