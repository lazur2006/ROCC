"""Persistent IterCQR current-query-only retrieval evaluation."""

from __future__ import annotations

import gc
import gzip
import hashlib
import inspect
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cached_itercqr import (
    CachedIterCQRBM25Config,
    CachedIterCQRBM25Pipeline,
)
from .colab import ensure_colab_retrieval_dependencies
from .datasets.qrecc import (
    qrecc_ance_session_shard_range,
    sync_qrecc_ance_shard_range_from_azure,
)
from .datasets.topiocqa import sync_topiocqa_ance_index_from_azure
from .dense_ance import (
    check_dense_gpu_capacity,
    load_dense_ance_torch_retriever,
    summarize_dense_faiss_shards,
)
from .evaluation import retrieval_metrics
from .itercqr_components import load_itercqr_tokenizer
from .pipelines import (
    QueryRankingStore,
    merge_query_ranking_stores,
    run_query_retrieval_to_store,
)
from .retrievers import load_bm25_retriever
from .teacher_evaluation import (
    materialize_history_arm_inputs,
    serialize_itercqr_input,
)


PROTOCOL = "itercqr_current_query_only_v1"
ARM = "CurrentQueryOnly"
DATASETS = {"topiocqa", "qrecc"}
BACKENDS = {"bm25", "ance"}
METRICS = ("MRR", "nDCG@3", "R@10", "R@100", "R@1000")


@dataclass(frozen=True)
class CurrentQueryOnlyResult:
    """Loaded or newly computed current-query-only evaluation."""

    summary: pd.DataFrame
    metrics_by_query: pd.DataFrame
    manifest: dict[str, Any]
    manifest_path: Path
    query_bundle_manifest_path: Path
    ranking_store_path: Path | None
    result_dir: Path
    backend: str
    ready: bool
    reused: bool

    @property
    def extension(self) -> dict[str, Any] | None:
        """Return the binding stored in an NB07 final manifest."""

        if not self.ready:
            return None
        return {
            "protocol": str(self.manifest["protocol"]),
            "scope": "result_dir",
            "manifest": str(
                self.manifest_path.relative_to(self.result_dir)
            ),
            "manifest_sha256": _sha256_file(self.manifest_path),
        }


def load_or_compute_current_query_only(
    *,
    result_dir: Path | str,
    dataset: str,
    samples: Sequence[Any],
    sample_ids: Sequence[str],
    gold_by_sample: Mapping[str, Sequence[Any]],
    population_sha256: str,
    model_dir: Path | str,
    expected_model_sha256: str,
    expected_serializer_sha256: str,
    pipeline_config: CachedIterCQRBM25Config,
    backend: str,
    bm25_index_dir: Path | str,
    bm25_k1: float,
    bm25_b: float,
    dense_index_dir: Path | str,
    azure_sas_file: Path | str | None = None,
    qrecc_ance_session_id: str | None = None,
    device: str = "cpu",
    budget: int = 64,
    top_k: int = 1_000,
    eval_ks: Sequence[int] = (3, 10, 100, 1000),
    retrieval_batch_size: int = 64,
    store_chunk_size: int = 256,
    retrieval_workers: int = 1,
    progress: bool = True,
) -> CurrentQueryOnlyResult:
    """Load or compute the B64 IterCQR run with an empty history.

    The query bundle is shared by BM25 and ANCE. Backend rankings and
    metrics are kept below ``current_query_only/<backend>``. QReCC ANCE
    writes one partial store per configured session and evaluates only
    after both stores are present. With ``azure_sas_file=None``, ANCE
    uses only the local index and never attempts an Azure download.
    """

    normalized_dataset = str(dataset).lower()
    normalized_backend = str(backend).lower()
    if normalized_dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if normalized_backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend!r}")
    if int(budget) != 64:
        raise ValueError("The current-query-only protocol is fixed at B64.")
    required_eval_ks = {3, 10, 100, 1000}
    if not required_eval_ks.issubset(map(int, eval_ks)):
        raise ValueError(
            "The current-query-only protocol requires evaluation at "
            "3, 10, 100, and 1000."
        )
    if int(top_k) < max(map(int, eval_ks)):
        raise ValueError("top_k must cover every evaluation cutoff.")

    resolved_result = Path(result_dir).expanduser().resolve()
    root = resolved_result / "current_query_only"
    root.mkdir(parents=True, exist_ok=True)
    ids = [str(sample_id) for sample_id in sample_ids]
    id_set = set(ids)
    if len(ids) != len(id_set):
        raise ValueError("sample_ids contains duplicates.")
    if set(map(str, gold_by_sample)) != id_set:
        raise ValueError("Gold labels and current-query-only population differ.")

    current_serializer_sha256 = _sha256_text(
        inspect.getsource(serialize_itercqr_input)
    )
    if current_serializer_sha256 != str(expected_serializer_sha256):
        raise RuntimeError(
            "The current IterCQR serializer differs from the frozen "
            "NB07 query bundle."
        )

    bundle, bundle_manifest, bundle_reused = _load_or_build_bundle(
        root=root,
        dataset=normalized_dataset,
        samples=samples,
        sample_ids=ids,
        population_sha256=str(population_sha256),
        model_dir=Path(model_dir),
        expected_model_sha256=str(expected_model_sha256),
        pipeline_config=pipeline_config,
        budget=int(budget),
        progress=progress,
    )
    query_bundle_sha256 = str(bundle_manifest["query_bundle_sha256"])
    bundle_manifest_path = root / "query_bundle/manifest.json"
    output_dir = root / normalized_backend
    final_manifest_path = output_dir / "final_manifest.json"

    complete = _load_complete_result(
        result_dir=resolved_result,
        root=root,
        output_dir=output_dir,
        final_manifest_path=final_manifest_path,
        dataset=normalized_dataset,
        backend=normalized_backend,
        sample_ids=ids,
        bundle=bundle,
        population_sha256=str(population_sha256),
        query_bundle_sha256=query_bundle_sha256,
        budget=int(budget),
        top_k=int(top_k),
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
    )
    if complete is not None:
        return complete

    queries = _unique_queries(bundle)
    if normalized_backend == "bm25":
        store_path = output_dir / "rankings.sqlite3"
        ranking_manifest_path = output_dir / "ranking_manifest.json"
        _ensure_bm25_store(
            store_path=store_path,
            manifest_path=ranking_manifest_path,
            queries=queries,
            dataset=normalized_dataset,
            population_sha256=str(population_sha256),
            query_bundle_sha256=query_bundle_sha256,
            index_dir=Path(bm25_index_dir),
            k1=float(bm25_k1),
            b=float(bm25_b),
            top_k=int(top_k),
            retrieval_batch_size=int(retrieval_batch_size),
            store_chunk_size=int(store_chunk_size),
            retrieval_workers=int(retrieval_workers),
            progress=progress,
        )
    elif normalized_dataset == "topiocqa":
        store_path = output_dir / "full/rankings.sqlite3"
        ranking_manifest_path = output_dir / "full/manifest.json"
        _ensure_dense_store(
            store_path=store_path,
            manifest_path=ranking_manifest_path,
            queries=queries,
            dataset=normalized_dataset,
            population_sha256=str(population_sha256),
            query_bundle_sha256=query_bundle_sha256,
            index_dir=Path(dense_index_dir),
            azure_sas_file=azure_sas_file,
            session_id=None,
            start_shard=None,
            end_shard=None,
            device=str(device),
            top_k=int(top_k),
            retrieval_batch_size=int(retrieval_batch_size),
            store_chunk_size=int(store_chunk_size),
            progress=progress,
        )
    else:
        if qrecc_ance_session_id not in {"session_1", "session_2"}:
            raise ValueError(
                "Missing QReCC ANCE result requires session_1 or session_2."
            )
        session_id = str(qrecc_ance_session_id)
        start_shard, end_shard = qrecc_ance_session_shard_range(session_id)
        session_dir = output_dir / session_id
        _ensure_dense_store(
            store_path=session_dir / "rankings.sqlite3",
            manifest_path=session_dir / "manifest.json",
            queries=queries,
            dataset=normalized_dataset,
            population_sha256=str(population_sha256),
            query_bundle_sha256=query_bundle_sha256,
            index_dir=Path(dense_index_dir),
            azure_sas_file=azure_sas_file,
            session_id=session_id,
            start_shard=start_shard,
            end_shard=end_shard,
            device=str(device),
            top_k=int(top_k),
            retrieval_batch_size=int(retrieval_batch_size),
            store_chunk_size=int(store_chunk_size),
            progress=progress,
        )
        session_stores: list[Path] = []
        for candidate in ("session_1", "session_2"):
            candidate_dir = output_dir / candidate
            candidate_start, candidate_end = (
                qrecc_ance_session_shard_range(candidate)
            )
            if not _ranking_artifact_valid(
                manifest_path=candidate_dir / "manifest.json",
                store_path=candidate_dir / "rankings.sqlite3",
                queries=queries,
                dataset=normalized_dataset,
                backend="ance_partial",
                population_sha256=str(population_sha256),
                query_bundle_sha256=query_bundle_sha256,
                top_k=int(top_k),
                session_id=candidate,
                start_shard=candidate_start,
                end_shard=candidate_end,
            ):
                return CurrentQueryOnlyResult(
                    summary=pd.DataFrame(),
                    metrics_by_query=pd.DataFrame(),
                    manifest={},
                    manifest_path=final_manifest_path,
                    query_bundle_manifest_path=bundle_manifest_path,
                    ranking_store_path=None,
                    result_dir=resolved_result,
                    backend=normalized_backend,
                    ready=False,
                    reused=bundle_reused,
                )
            session_stores.append(candidate_dir / "rankings.sqlite3")
        store_path = output_dir / "merged/rankings.sqlite3"
        if store_path.exists():
            _assert_store_ids(store_path, set(queries["query_id"]))
        else:
            merge_query_ranking_stores(
                session_stores,
                output_path=store_path,
                top_k=int(top_k),
                progress=progress,
            )

    _evaluate_store(
        root=root,
        output_dir=output_dir,
        dataset=normalized_dataset,
        backend=normalized_backend,
        bundle=bundle,
        gold_by_sample=gold_by_sample,
        population_sha256=str(population_sha256),
        query_bundle_sha256=query_bundle_sha256,
        store_path=store_path,
        budget=int(budget),
        top_k=int(top_k),
        eval_ks=tuple(map(int, eval_ks)),
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
    )
    computed = _load_complete_result(
        result_dir=resolved_result,
        root=root,
        output_dir=output_dir,
        final_manifest_path=final_manifest_path,
        dataset=normalized_dataset,
        backend=normalized_backend,
        sample_ids=ids,
        bundle=bundle,
        population_sha256=str(population_sha256),
        query_bundle_sha256=query_bundle_sha256,
        budget=int(budget),
        top_k=int(top_k),
        bm25_k1=float(bm25_k1),
        bm25_b=float(bm25_b),
    )
    if computed is None:
        raise RuntimeError("Current-query-only evaluation did not complete.")
    return CurrentQueryOnlyResult(
        **{
            **computed.__dict__,
            "reused": False,
        }
    )


def _load_or_build_bundle(
    *,
    root: Path,
    dataset: str,
    samples: Sequence[Any],
    sample_ids: Sequence[str],
    population_sha256: str,
    model_dir: Path,
    expected_model_sha256: str,
    pipeline_config: CachedIterCQRBM25Config,
    budget: int,
    progress: bool,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    bundle_path = root / "query_bundle/queries.jsonl.gz"
    manifest_path = root / "query_bundle/manifest.json"
    _require_complete_pair((bundle_path, manifest_path), "query bundle")
    if bundle_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        bundle = _read_bundle(bundle_path)
        _validate_bundle(
            bundle=bundle,
            bundle_path=bundle_path,
            manifest=manifest,
            dataset=dataset,
            sample_ids=sample_ids,
            population_sha256=population_sha256,
            expected_model_sha256=expected_model_sha256,
            budget=budget,
        )
        return bundle, manifest, True

    model_path = model_dir / "pytorch_model.bin"
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    if _sha256_file(model_path) != expected_model_sha256:
        raise RuntimeError("The IterCQR model differs from the frozen model.")
    sample_by_id = {str(sample.sample_id): sample for sample in samples}
    if set(sample_by_id) != set(sample_ids):
        raise ValueError("Samples and current-query-only population differ.")

    tokenizer = load_itercqr_tokenizer(model_dir)
    pipeline = CachedIterCQRBM25Pipeline(
        config=pipeline_config,
        tokenizer=tokenizer,
    )
    try:
        empty_histories = {sample_id: [] for sample_id in sample_ids}
        serialized, _ = materialize_history_arm_inputs(
            samples=[sample_by_id[sample_id] for sample_id in sample_ids],
            histories_by_arm={"CQ": empty_histories},
            arm_order=("CQ",),
            tokenizer=tokenizer,
            budgets=(budget,),
            pipeline=pipeline,
            progress=progress,
            progress_desc=f"serialize {dataset} current-query-only inputs",
        )
        rewritten = pipeline.materialize_rewrites(
            serialized,
            progress_prefix=f"{dataset} current query only",
        ).serialized_with_rewrite
    finally:
        pipeline.close()
        del pipeline, tokenizer
        gc.collect()
        _empty_cuda_cache()

    current_queries = {
        sample_id: str(sample_by_id[sample_id].current_query)
        for sample_id in sample_ids
    }
    rows: list[dict[str, Any]] = []
    for row in rewritten.sort_values(
        "sample_id", kind="mergesort"
    ).itertuples(index=False):
        query = str(row.rewrite)
        query_norm = re.sub(r"\s+", " ", query).strip()
        rows.append(
            {
                "sample_id": str(row.sample_id),
                "current_query": current_queries[str(row.sample_id)],
                "query_id": _sha256_text(f"{PROTOCOL}\0{query_norm}"),
                "query": query,
                "query_norm": query_norm,
                "budget": budget,
                "input_length": int(row.input_length),
                "raw_input_length": int(row.raw_input_length),
                "was_truncated": bool(row.was_truncated),
            }
        )
    bundle = pd.DataFrame(rows)
    _write_bundle(bundle_path, bundle.to_dict(orient="records"))
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "population_sha256": population_sha256,
        "model_sha256": expected_model_sha256,
        "budget": budget,
        "history": [],
        "rows": len(bundle),
        "unique_retrieval_queries": int(bundle["query_id"].nunique()),
        "max_raw_input_length": int(bundle["raw_input_length"].max()),
        "truncated_inputs": int(bundle["was_truncated"].sum()),
        "query_bundle_sha256": _sha256_file(bundle_path),
        "complete": True,
    }
    _write_json(manifest_path, manifest)
    _validate_bundle(
        bundle=bundle,
        bundle_path=bundle_path,
        manifest=manifest,
        dataset=dataset,
        sample_ids=sample_ids,
        population_sha256=population_sha256,
        expected_model_sha256=expected_model_sha256,
        budget=budget,
    )
    return bundle, manifest, False


def _validate_bundle(
    *,
    bundle: pd.DataFrame,
    bundle_path: Path,
    manifest: Mapping[str, Any],
    dataset: str,
    sample_ids: Sequence[str],
    population_sha256: str,
    expected_model_sha256: str,
    budget: int,
) -> None:
    expected = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "population_sha256": population_sha256,
        "model_sha256": expected_model_sha256,
        "budget": budget,
        "history": [],
        "rows": len(sample_ids),
        "query_bundle_sha256": _sha256_file(bundle_path),
        "complete": True,
    }
    _assert_identity(manifest, expected, "current-query-only query bundle")
    required = {
        "sample_id",
        "current_query",
        "query_id",
        "query",
        "query_norm",
        "budget",
        "input_length",
        "raw_input_length",
        "was_truncated",
    }
    if not required.issubset(bundle.columns):
        raise RuntimeError("Current-query-only query columns are missing.")
    bundle = bundle.copy()
    bundle["sample_id"] = bundle["sample_id"].astype(str)
    if (
        len(bundle) != len(sample_ids)
        or bundle["sample_id"].duplicated().any()
        or set(bundle["sample_id"]) != set(sample_ids)
        or not bundle["budget"].astype(int).eq(budget).all()
        or bundle["was_truncated"].astype(bool).any()
    ):
        raise RuntimeError("Current-query-only query bundle is incomplete.")
    collisions = bundle.groupby("query_id")["query_norm"].nunique()
    if collisions.gt(1).any():
        raise RuntimeError("Current-query-only query ID collision.")
    if int(manifest.get("unique_retrieval_queries", -1)) != int(
        bundle["query_id"].nunique()
    ):
        raise RuntimeError("Current-query-only unique-query count differs.")


def _load_complete_result(
    *,
    result_dir: Path,
    root: Path,
    output_dir: Path,
    final_manifest_path: Path,
    dataset: str,
    backend: str,
    sample_ids: Sequence[str],
    bundle: pd.DataFrame,
    population_sha256: str,
    query_bundle_sha256: str,
    budget: int,
    top_k: int,
    bm25_k1: float,
    bm25_b: float,
) -> CurrentQueryOnlyResult | None:
    if not final_manifest_path.exists():
        return None
    manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "backend": backend,
        "population_sha256": population_sha256,
        "query_bundle_sha256": query_bundle_sha256,
        "top_k": top_k,
        "budget": budget,
        "history": [],
        "complete": True,
        "selection_performed": False,
        "latency_measured": False,
    }
    if backend == "bm25":
        expected.update({"k1": bm25_k1, "b": bm25_b})
    _assert_identity(manifest, expected, "current-query-only result")

    store_path = _canonical_store_path(root, dataset, backend)
    if not store_path.is_file():
        raise FileNotFoundError(store_path)
    if manifest.get("ranking_store_sha256") != _sha256_file(store_path):
        raise RuntimeError("Current-query-only ranking store has drifted.")
    expected_query_ids = set(bundle["query_id"].astype(str))
    _assert_store_ids(store_path, expected_query_ids)

    files = manifest.get("files", {})
    expected_files = {"metrics_by_query.csv", "summary.csv"}
    if set(files) != expected_files:
        raise RuntimeError("Current-query-only result file set differs.")
    for filename, expected_hash in files.items():
        path = output_dir / str(filename)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise RuntimeError(
                f"Current-query-only result file has drifted: {filename}"
            )
    metrics = pd.read_csv(output_dir / "metrics_by_query.csv")
    summary = pd.read_csv(output_dir / "summary.csv")
    _validate_metrics(
        metrics=metrics,
        summary=summary,
        sample_ids=sample_ids,
        bundle=bundle,
        budget=budget,
    )
    return CurrentQueryOnlyResult(
        summary=summary,
        metrics_by_query=metrics,
        manifest=dict(manifest),
        manifest_path=final_manifest_path,
        query_bundle_manifest_path=root / "query_bundle/manifest.json",
        ranking_store_path=store_path,
        result_dir=result_dir,
        backend=backend,
        ready=True,
        reused=True,
    )


def _validate_metrics(
    *,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    sample_ids: Sequence[str],
    bundle: pd.DataFrame,
    budget: int,
) -> None:
    required_metrics = {
        "system",
        "arm",
        "budget",
        "sample_id",
        "query_id",
        "input_length",
        "raw_input_length",
        *METRICS,
    }
    required_summary = {
        "system",
        "arm",
        "budget",
        "n",
        *METRICS,
        "mean_input_length",
        "max_input_length",
    }
    if not required_metrics.issubset(metrics.columns):
        raise RuntimeError("Current-query-only metric columns are missing.")
    if not required_summary.issubset(summary.columns) or len(summary) != 1:
        raise RuntimeError("Current-query-only summary is incomplete.")
    metrics = metrics.copy()
    metrics["sample_id"] = metrics["sample_id"].astype(str)
    expected_queries = bundle.set_index("sample_id")["query_id"].astype(str)
    observed_queries = metrics.set_index("sample_id")["query_id"].astype(str)
    if (
        len(metrics) != len(sample_ids)
        or metrics["sample_id"].duplicated().any()
        or set(metrics["sample_id"]) != set(sample_ids)
        or not metrics["budget"].astype(int).eq(budget).all()
        or not observed_queries.sort_index().equals(
            expected_queries.sort_index()
        )
    ):
        raise RuntimeError("Current-query-only metrics are not aligned.")
    row = summary.iloc[0]
    if (
        str(row["system"]) != "itercqr"
        or str(row["arm"]) != ARM
        or int(row["budget"]) != budget
        or int(row["n"]) != len(sample_ids)
    ):
        raise RuntimeError("Current-query-only summary identity differs.")
    for metric in METRICS:
        if not np.isclose(
            float(row[metric]),
            float(metrics[metric].mean()),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"Current-query-only summary differs for {metric}."
            )


def _ensure_bm25_store(
    *,
    store_path: Path,
    manifest_path: Path,
    queries: pd.DataFrame,
    dataset: str,
    population_sha256: str,
    query_bundle_sha256: str,
    index_dir: Path,
    k1: float,
    b: float,
    top_k: int,
    retrieval_batch_size: int,
    store_chunk_size: int,
    retrieval_workers: int,
    progress: bool,
) -> None:
    if _ranking_artifact_valid(
        manifest_path=manifest_path,
        store_path=store_path,
        queries=queries,
        dataset=dataset,
        backend="bm25",
        population_sha256=population_sha256,
        query_bundle_sha256=query_bundle_sha256,
        top_k=top_k,
        k1=k1,
        b=b,
    ):
        return
    if manifest_path.exists():
        raise RuntimeError("Current-query-only BM25 manifest has drifted.")
    if store_path.exists():
        _assert_resumable_store(store_path, set(queries["query_id"]))
    retriever = load_bm25_retriever(
        index_dir=index_dir,
        k1=k1,
        b=b,
        retrieval_workers=retrieval_workers,
    )
    try:
        result = run_query_retrieval_to_store(
            queries=queries,
            retriever=retriever,
            store_path=store_path,
            top_k=top_k,
            retrieval_batch_size=retrieval_batch_size,
            store_chunk_size=store_chunk_size,
            measure_latency=False,
            deduplication_mode="exact",
            progress=progress,
            progress_desc=f"{dataset} current-query-only BM25",
        )
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
    if result.stored_queries != len(queries):
        raise RuntimeError("Current-query-only BM25 store is incomplete.")
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "dataset": dataset,
            "backend": "bm25",
            "population_sha256": population_sha256,
            "query_bundle_sha256": query_bundle_sha256,
            "queries": len(queries),
            "top_k": top_k,
            "k1": k1,
            "b": b,
            "store_sha256": _sha256_file(store_path),
            "complete": True,
            "latency_measured": False,
        },
    )


def _ensure_dense_store(
    *,
    store_path: Path,
    manifest_path: Path,
    queries: pd.DataFrame,
    dataset: str,
    population_sha256: str,
    query_bundle_sha256: str,
    index_dir: Path,
    azure_sas_file: Path | str | None,
    session_id: str | None,
    start_shard: int | None,
    end_shard: int | None,
    device: str,
    top_k: int,
    retrieval_batch_size: int,
    store_chunk_size: int,
    progress: bool,
) -> None:
    manifest_backend = "ance" if dataset == "topiocqa" else "ance_partial"
    if _ranking_artifact_valid(
        manifest_path=manifest_path,
        store_path=store_path,
        queries=queries,
        dataset=dataset,
        backend=manifest_backend,
        population_sha256=population_sha256,
        query_bundle_sha256=query_bundle_sha256,
        top_k=top_k,
        session_id=session_id,
        start_shard=start_shard,
        end_shard=end_shard,
    ):
        return
    if manifest_path.exists():
        raise RuntimeError("Current-query-only ANCE manifest has drifted.")
    if store_path.exists():
        _assert_resumable_store(store_path, set(queries["query_id"]))

    try:
        dense_summary = summarize_dense_faiss_shards(
            index_dir,
            start_shard=start_shard,
            end_shard=end_shard,
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as exc:
        if azure_sas_file is None:
            raise FileNotFoundError(
                f"Local ANCE index is missing or invalid: {index_dir}. "
                "Place the downloaded index at this path. Azure download "
                "is disabled unless azure_sas_file is explicitly provided."
            ) from exc
        if dataset == "topiocqa":
            synced = sync_topiocqa_ance_index_from_azure(
                local_index_dir=index_dir,
                sas_file=azure_sas_file,
                validate_azure=True,
                progress=progress,
            )
        else:
            synced = sync_qrecc_ance_shard_range_from_azure(
                local_index_dir=index_dir,
                session_id=session_id,
                sas_file=azure_sas_file,
                validate_azure=True,
                progress=progress,
            )
        dense_summary = summarize_dense_faiss_shards(
            synced.local_index_dir,
            start_shard=start_shard,
            end_shard=end_shard,
        )
    ensure_colab_retrieval_dependencies()
    capacity = check_dense_gpu_capacity(
        dense_summary,
        device=device,
        index_dtype="float32",
        reserve_gib=4.0,
    )
    if not capacity.fits:
        raise RuntimeError("Current-query-only ANCE index does not fit on GPU.")
    retriever = load_dense_ance_torch_retriever(
        index_dir=dense_summary.index_dir,
        device=device,
        index_dtype="float32",
        start_shard=start_shard,
        end_shard=end_shard,
        chunk_size=50_000,
        query_batch_size=64,
        progress=progress,
        status_log=(
            store_path.parent / f"current_query_only_{dataset}_{session_id or 'full'}.log"
        ),
    )
    try:
        result = run_query_retrieval_to_store(
            queries=queries,
            retriever=retriever,
            store_path=store_path,
            top_k=top_k,
            retrieval_batch_size=retrieval_batch_size,
            store_chunk_size=store_chunk_size,
            measure_latency=False,
            deduplication_mode="exact",
            progress=progress,
            progress_desc=(
                f"{dataset} current-query-only ANCE {session_id or 'full'}"
            ),
        )
    finally:
        close = getattr(retriever, "close", None)
        if callable(close):
            close()
        del retriever
        gc.collect()
        _empty_cuda_cache()
    if result.stored_queries != len(queries):
        raise RuntimeError("Current-query-only ANCE store is incomplete.")
    _write_json(
        manifest_path,
        {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "dataset": dataset,
            "backend": manifest_backend,
            "session_id": session_id,
            "start_shard": start_shard,
            "end_shard": end_shard,
            "population_sha256": population_sha256,
            "query_bundle_sha256": query_bundle_sha256,
            "queries": len(queries),
            "top_k": top_k,
            "store_sha256": _sha256_file(store_path),
            "complete": True,
            "latency_measured": False,
        },
    )


def _ranking_artifact_valid(
    *,
    manifest_path: Path,
    store_path: Path,
    queries: pd.DataFrame,
    dataset: str,
    backend: str,
    population_sha256: str,
    query_bundle_sha256: str,
    top_k: int,
    session_id: str | None = None,
    start_shard: int | None = None,
    end_shard: int | None = None,
    k1: float | None = None,
    b: float | None = None,
) -> bool:
    if manifest_path.is_file() and not store_path.is_file():
        raise RuntimeError(
            "The current-query-only ranking manifest has no store."
        )
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "backend": backend,
        "population_sha256": population_sha256,
        "query_bundle_sha256": query_bundle_sha256,
        "queries": len(queries),
        "top_k": top_k,
        "complete": True,
        "latency_measured": False,
        "store_sha256": _sha256_file(store_path),
    }
    if backend == "ance" or backend == "ance_partial":
        expected.update(
            {
                "session_id": session_id,
                "start_shard": start_shard,
                "end_shard": end_shard,
            }
        )
    if backend == "bm25":
        expected.update({"k1": k1, "b": b})
    if any(manifest.get(key) != value for key, value in expected.items()):
        return False
    try:
        _assert_store_ids(store_path, set(queries["query_id"]))
    except RuntimeError:
        return False
    return True


def _evaluate_store(
    *,
    root: Path,
    output_dir: Path,
    dataset: str,
    backend: str,
    bundle: pd.DataFrame,
    gold_by_sample: Mapping[str, Sequence[Any]],
    population_sha256: str,
    query_bundle_sha256: str,
    store_path: Path,
    budget: int,
    top_k: int,
    eval_ks: Sequence[int],
    bm25_k1: float,
    bm25_b: float,
) -> None:
    rows: list[dict[str, Any]] = []
    with QueryRankingStore(store_path, read_only=True) as store:
        for row in bundle.itertuples(index=False):
            sample_id = str(row.sample_id)
            scores = retrieval_metrics(
                store[str(row.query_id)],
                gold_by_sample[sample_id],
                ks=eval_ks,
            )
            rows.append(
                {
                    "system": "itercqr",
                    "arm": ARM,
                    "budget": budget,
                    "sample_id": sample_id,
                    "query_id": str(row.query_id),
                    "input_length": int(row.input_length),
                    "raw_input_length": int(row.raw_input_length),
                    **scores,
                }
            )
    metrics = pd.DataFrame(rows).sort_values(
        "sample_id", kind="mergesort"
    )
    summary = pd.DataFrame(
        [
            {
                "system": "itercqr",
                "arm": ARM,
                "budget": budget,
                "n": len(metrics),
                **{
                    metric: float(metrics[metric].mean())
                    for metric in METRICS
                },
                "mean_input_length": float(metrics["input_length"].mean()),
                "max_input_length": int(metrics["input_length"].max()),
            }
        ]
    )
    metrics_path = output_dir / "metrics_by_query.csv"
    summary_path = output_dir / "summary.csv"
    _write_frame(metrics_path, metrics)
    _write_frame(summary_path, summary)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": dataset,
        "backend": backend,
        "population_sha256": population_sha256,
        "query_bundle_sha256": query_bundle_sha256,
        "ranking_store_sha256": _sha256_file(store_path),
        "top_k": top_k,
        "budget": budget,
        "history": [],
        "files": {
            "metrics_by_query.csv": _sha256_file(metrics_path),
            "summary.csv": _sha256_file(summary_path),
        },
        "complete": True,
        "selection_performed": False,
        "latency_measured": False,
    }
    if backend == "bm25":
        manifest.update({"k1": bm25_k1, "b": bm25_b})
    _write_json(output_dir / "final_manifest.json", manifest)


def _canonical_store_path(root: Path, dataset: str, backend: str) -> Path:
    if backend == "bm25":
        return root / "bm25/rankings.sqlite3"
    if dataset == "topiocqa":
        return root / "ance/full/rankings.sqlite3"
    return root / "ance/merged/rankings.sqlite3"


def _unique_queries(bundle: pd.DataFrame) -> pd.DataFrame:
    conflicts = bundle.groupby("query_id")["query_norm"].nunique()
    if conflicts.gt(1).any():
        raise RuntimeError("A query ID maps to multiple exact query texts.")
    return (
        bundle[["query_id", "query"]]
        .drop_duplicates("query_id")
        .sort_values("query_id", kind="mergesort")
        .reset_index(drop=True)
    )


def _assert_resumable_store(store_path: Path, expected_ids: set[str]) -> None:
    with QueryRankingStore(store_path, read_only=True) as store:
        unexpected = set(iter(store)).difference(expected_ids)
    if unexpected:
        raise RuntimeError(
            "Incomplete current-query-only store belongs to another bundle."
        )


def _assert_store_ids(store_path: Path, expected_ids: set[str]) -> None:
    with QueryRankingStore(store_path, read_only=True) as store:
        observed = set(iter(store))
    if observed != expected_ids:
        raise RuntimeError("Current-query-only ranking store is incomplete.")


def _require_complete_pair(paths: Sequence[Path], label: str) -> None:
    present = [path.is_file() for path in paths]
    if any(present) and not all(present):
        raise RuntimeError(f"The {label} is only partially present.")


def _assert_identity(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    label: str,
) -> None:
    drift = {
        key: (observed.get(key), value)
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if drift:
        raise RuntimeError(f"The {label} has drifted: {drift!r}")


def _read_bundle(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return pd.DataFrame(
            json.loads(line) for line in handle if line.strip()
        )


def _write_bundle(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0
        ) as compressed:
            with io.TextIOWrapper(
                compressed, encoding="utf-8", newline="\n"
            ) as text:
                for row in rows:
                    text.write(
                        json.dumps(
                            dict(row),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        )
                        + "\n"
                    )
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


__all__ = [
    "ARM",
    "PROTOCOL",
    "CurrentQueryOnlyResult",
    "load_or_compute_current_query_only",
]
