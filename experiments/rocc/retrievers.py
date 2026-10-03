"""Self-contained first-stage retriever wrappers."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from math import exp
from pathlib import Path
from typing import Any, Literal

from .paths import project_path


JAVA21_HOME = Path("/usr/lib/jvm/java-21-openjdk-amd64")
JAVA21_JVM_PATH = JAVA21_HOME / "lib" / "server" / "libjvm.so"
BM25_K1 = 0.9
BM25_B = 0.4
DEFAULT_RETRIEVAL_TOP_K = 1000
DEFAULT_ANCE_ENCODER = "castorini/ance-msmarco-passage"


def default_bm25_index_dir() -> Path:
    return project_path("experiments", "data", "bm25", "lucene_index")


def default_dense_faiss_index_dir() -> Path:
    return project_path(
        "experiments",
        "data",
        "pyserini_ance_faiss_topiocqa",
        "faiss_flat_index_full_sharded",
    )


def configure_java21(env: dict[str, str] | None = None) -> dict[str, str]:
    """Configure Java 21 for Pyserini/Pyjnius before importing pyserini."""

    target = os.environ if env is None else env
    if JAVA21_HOME.exists():
        target["JAVA_HOME"] = str(JAVA21_HOME)
        target["PATH"] = f"{JAVA21_HOME / 'bin'}:{target.get('PATH', '')}"
    if JAVA21_JVM_PATH.exists():
        target["JVM_PATH"] = str(JAVA21_JVM_PATH)
    return target


class BM25Retriever:
    def __init__(
        self,
        index_dir: Path | str | None = None,
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
        retrieval_workers: int = 1,
        parallel_mode: Literal["shared"] = "shared",
    ) -> None:
        os.environ.setdefault("OPENAI_API_KEY", "pyserini-bm25-rocc-no-openai-call")
        configure_java21()
        if retrieval_workers < 1:
            raise ValueError("retrieval_workers must be >= 1.")
        if parallel_mode != "shared":
            raise ValueError("parallel_mode must be 'shared'.")

        self.index_dir = Path(index_dir).expanduser().resolve() if index_dir else default_bm25_index_dir()
        if not self.index_dir.exists():
            raise FileNotFoundError(
                "BM25 index is not available inside experiments. "
                f"Expected: {self.index_dir}. Build/copy the index there or pass index_dir."
            )

        try:
            from pyserini.search.lucene import LuceneSearcher
        except ModuleNotFoundError as exc:
            raise RuntimeError("pyserini is required for BM25 retrieval.") from exc

        self.k1 = float(k1)
        self.b = float(b)
        self.retrieval_workers = int(retrieval_workers)
        self.parallel_mode = parallel_mode
        self._executor: ThreadPoolExecutor | None = None
        self.searcher = LuceneSearcher(str(self.index_dir))
        self.searcher.set_bm25(self.k1, self.b)

    def search(
        self,
        query: str,
        *,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        include_raw: bool = False,
    ) -> list[dict[str, Any]]:
        return self._search_with_searcher(
            self.searcher,
            query,
            top_k=top_k,
            include_raw=include_raw,
        )

    def search_batch(
        self,
        queries: list[str],
        *,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
        include_raw: bool = False,
    ) -> list[list[dict[str, Any]]]:
        if not queries:
            return []
        normalized_queries = [str(query) for query in queries]
        if include_raw or self.retrieval_workers <= 1 or len(normalized_queries) <= 1:
            return [
                self.search(query, top_k=top_k, include_raw=include_raw)
                for query in normalized_queries
            ]

        def worker(query: str) -> list[dict[str, Any]]:
            return self._search_with_searcher(
                self.searcher,
                query,
                top_k=top_k,
                include_raw=False,
            )

        return list(self._parallel_executor().map(worker, normalized_queries))

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _parallel_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.retrieval_workers)
        return self._executor

    def _search_with_searcher(
        self,
        searcher: Any,
        query: str,
        *,
        top_k: int,
        include_raw: bool,
    ) -> list[dict[str, Any]]:
        hits = searcher.search(str(query), k=top_k)
        rows = []
        for rank, hit in enumerate(hits, start=1):
            row: dict[str, Any] = {
                "rank": rank,
                "docid": str(hit.docid),
                "score": float(hit.score),
            }
            if include_raw:
                raw_doc = searcher.doc(hit.docid)
                raw = json.loads(raw_doc.raw()) if raw_doc is not None and raw_doc.raw() else {}
                row.update(
                    {
                        "title": raw.get("title"),
                        "text": raw.get("text") or raw.get("contents"),
                        "raw_passage_id": raw.get("raw_passage_id"),
                    }
                )
            rows.append(row)
        return rows


def load_bm25_retriever(
    index_dir: Path | str | None = None,
    *,
    k1: float = BM25_K1,
    b: float = BM25_B,
    retrieval_workers: int = 1,
    parallel_mode: Literal["shared"] = "shared",
) -> BM25Retriever:
    return BM25Retriever(
        index_dir=index_dir,
        k1=k1,
        b=b,
        retrieval_workers=retrieval_workers,
        parallel_mode=parallel_mode,
    )


def normalize_retrieval_scores(
    hits: list[dict[str, Any]],
    *,
    method: str = "minmax",
    score_key: str = "score",
    output_key: str = "score_norm",
) -> list[dict[str, Any]]:
    """Add query-local normalized scores to retrieval hits.

    BM25 scores are not globally calibrated across queries. These normalizers
    are therefore only meaningful within one result list for display, features,
    or local reranking diagnostics.
    """

    if method not in {"minmax", "top", "softmax"}:
        raise ValueError("method must be one of: minmax, top, softmax")
    if not hits:
        return []

    rows = [dict(hit) for hit in hits]
    scores = [float(row[score_key]) for row in rows]

    if method == "minmax":
        lo = min(scores)
        hi = max(scores)
        denom = hi - lo
        values = [(score - lo) / denom if denom else 1.0 for score in scores]
    elif method == "top":
        top = max(scores)
        values = [score / top if top else 0.0 for score in scores]
    else:
        max_score = max(scores)
        weights = [exp(score - max_score) for score in scores]
        denom = sum(weights)
        values = [weight / denom if denom else 0.0 for weight in weights]

    for row, value in zip(rows, values, strict=True):
        row[output_key] = float(value)
    return rows


class DenseFaissRetriever:
    """ANCE query encoder + local sharded FAISS IndexFlatIP search."""

    def __init__(
        self,
        index_dir: Path | str | None = None,
        *,
        encoder_name: str = DEFAULT_ANCE_ENCODER,
        device: str = "cpu",
        faiss_threads: int = 4,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser().resolve() if index_dir else default_dense_faiss_index_dir()
        self.encoder_name = encoder_name
        self.device = device
        self.faiss_threads = faiss_threads
        self._query_encoder = None
        self.shards = shard_dirs(self.index_dir)

    def encode_queries(self, queries: list[str]) -> Any:
        if self._query_encoder is None:
            os.environ.setdefault("OPENAI_API_KEY", "pyserini-ance-faiss-no-openai-call")
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            configure_java21()
            try:
                from pyserini.encode._ance import AnceEncoder, AnceQueryEncoder
            except ModuleNotFoundError as exc:
                raise RuntimeError("pyserini is required for ANCE query encoding.") from exc
            if not hasattr(AnceEncoder, "all_tied_weights_keys"):
                AnceEncoder.all_tied_weights_keys = {}
            self._query_encoder = AnceQueryEncoder(encoder_dir=self.encoder_name, device=self.device)

        import numpy as np

        vectors = [
            self._query_encoder.encode(str(query)).astype("float32", copy=False)
            for query in queries
        ]
        return np.ascontiguousarray(np.vstack(vectors).astype("float32", copy=False))

    def search(
        self,
        query: str,
        *,
        top_k: int = 100,
        include_raw: bool = False,
    ) -> list[dict[str, Any]]:
        if include_raw:
            raise NotImplementedError("DenseFaissRetriever does not load raw document text.")
        return self.search_batch([query], top_k=top_k)[0]

    def search_batch(
        self,
        queries: list[str],
        *,
        top_k: int = 100,
    ) -> list[list[dict[str, Any]]]:
        query_vectors = self.encode_queries(queries)
        return search_sharded_faiss(query_vectors, self.shards, top_k=top_k, faiss_threads=self.faiss_threads)

def load_dense_faiss_retriever(
    index_dir: Path | str | None = None,
    *,
    encoder_name: str = DEFAULT_ANCE_ENCODER,
    device: str = "cpu",
    faiss_threads: int = 4,
) -> DenseFaissRetriever:
    return DenseFaissRetriever(
        index_dir=index_dir,
        encoder_name=encoder_name,
        device=device,
        faiss_threads=faiss_threads,
    )


def shard_dirs(index_dir: Path) -> list[Path]:
    shards = [
        path
        for path in sorted(index_dir.glob("shard_*"))
        if path.is_dir() and (path / "index").exists() and (path / "docid").exists()
    ]
    if not shards:
        raise FileNotFoundError(f"No FAISS shard dirs found in {index_dir}")
    return shards


def search_sharded_faiss(
    query_vectors: Any,
    shards: list[Path],
    *,
    top_k: int,
    faiss_threads: int = 4,
) -> list[list[dict[str, Any]]]:
    import faiss
    import heapq

    if faiss_threads > 0:
        faiss.omp_set_num_threads(faiss_threads)

    heaps: list[list[tuple[float, str]]] = [[] for _ in range(query_vectors.shape[0])]
    for shard in shards:
        index = faiss.read_index(str(shard / "index"))
        docids = (shard / "docid").read_text(encoding="utf-8").splitlines()
        shard_k = min(top_k, index.ntotal)
        scores, offsets = index.search(query_vectors, shard_k)
        for query_index, (query_scores, query_offsets) in enumerate(zip(scores, offsets, strict=True)):
            heap = heaps[query_index]
            for score, offset in zip(query_scores, query_offsets, strict=True):
                if int(offset) < 0:
                    continue
                item = (float(score), str(docids[int(offset)]))
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)

    results = []
    for heap in heaps:
        ranked = sorted(heap, reverse=True)
        results.append(
            [
                {"rank": rank, "docid": docid, "score": score}
                for rank, (score, docid) in enumerate(ranked, start=1)
            ]
        )
    return results
