"""Cached BM25 evaluation of token, Q/A, and turn ROCC projections."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .cached_itercqr import (
    CachedIterCQRBM25Config,
    CachedIterCQRBM25Pipeline,
)
from .evaluation import evaluate_fixed_rrf_views
from .history_selector import (
    expand_selected_histories,
    read_selected_histories_jsonl,
)
from .itercqr_components import load_itercqr_tokenizer
from .teacher_evaluation import (
    materialize_history_arm_inputs,
    serialize_itercqr_input,
)


PROTOCOL = "nb07a_pretrained_granularity_bm25_v2"
GRANULARITIES = ("token", "qa", "turn")
ARMS = ("I", "R", "D", "I+R+D")
VIEW_SETS = {
    "I": ("I",),
    "R": ("R",),
    "D": ("D",),
    "I+R+D": ("I", "R", "D"),
}
METRICS = ("MRR", "nDCG@3", "R@10", "R@100", "R@1000")
DATA_FILENAMES = (
    "metrics_by_query.csv",
    "summary.csv",
    "input_length_summary.csv",
    "serialization_summary.csv",
)


@dataclass(frozen=True)
class GranularityProjectionResult:
    """Persisted or newly computed granularity projection."""

    metrics_by_query: pd.DataFrame
    summary: pd.DataFrame
    input_length_summary: pd.DataFrame
    serialization_summary: pd.DataFrame
    manifest: dict[str, Any]
    manifest_path: Path
    data_paths: tuple[Path, ...]
    reused: bool

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        """Return data artifacts followed by their manifest."""

        return (*self.data_paths, self.manifest_path)


def load_or_compute_topiocqa_bm25_granularity_projection(
    *,
    result_dir: Path | str,
    backend_dir: Path | str,
    rows: Sequence[dict[str, Any]],
    samples: Sequence[Any],
    sample_ids: Sequence[str],
    depth_by_sample: Mapping[str, int],
    gold_by_sample: Mapping[str, Sequence[Any]],
    population_sha256: str,
    selector_checkpoint_sha256: str,
    query_bundle_manifest: Mapping[str, Any],
    backend_identity: Mapping[str, Any],
    pipeline_config: CachedIterCQRBM25Config,
    itercqr_model_dir: Path | str,
    budgets: Sequence[int],
    rrf_k: int,
    retrieval_top_k: int,
    eval_ks: Sequence[int],
    progress: bool = True,
) -> GranularityProjectionResult:
    """Load or compute the full-dev BM25 granularity projection.

    The canonical token route is copied from the frozen NB07a result.
    Only Q/A and turn projections are newly materialized when the
    granularity artifact is absent.
    """

    resolved_result = Path(result_dir)
    resolved_backend = Path(backend_dir)
    projection_dir = resolved_backend / "granularity_projection"
    manifest_path = projection_dir / "manifest.json"
    data_paths = tuple(
        projection_dir / filename for filename in DATA_FILENAMES
    )
    data_by_name = dict(zip(DATA_FILENAMES, data_paths, strict=True))

    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    sample_id_set = set(normalized_ids)
    normalized_budgets = tuple(int(budget) for budget in budgets)
    if len(sample_id_set) != len(normalized_ids):
        raise ValueError("sample_ids contains duplicates.")
    if len(set(normalized_budgets)) != len(normalized_budgets):
        raise ValueError("budgets contains duplicates.")
    if set(map(str, gold_by_sample)) != sample_id_set:
        raise ValueError("Gold and projection population differ.")
    if set(map(str, depth_by_sample)) != sample_id_set:
        raise ValueError("History depths and projection population differ.")
    if str(backend_identity.get("backend")) != "bm25":
        raise ValueError("Granularity projection requires BM25 identity.")

    viterbi_path = (
        resolved_result / "predictions/pretrained/viterbi.jsonl"
    )
    viterbi_manifest_path = viterbi_path.parent / "manifest.json"
    main_inputs_path = (
        resolved_result / "query_bundle/serialization/main_inputs.csv"
    )
    token_metrics_path = resolved_backend / "route_metrics_by_query.csv"
    token_summary_path = resolved_backend / "route_summary.csv"
    source_paths = (
        viterbi_path,
        viterbi_manifest_path,
        main_inputs_path,
        token_metrics_path,
        token_summary_path,
    )
    missing_sources = [str(path) for path in source_paths if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(
            "Granularity projection sources are missing: "
            + ", ".join(missing_sources)
        )

    viterbi_manifest = json.loads(
        viterbi_manifest_path.read_text(encoding="utf-8")
    )
    viterbi_sha256 = _sha256_file(viterbi_path)
    if (
        viterbi_manifest.get("system") != "pretrained"
        or viterbi_manifest.get("population_sha256")
        != str(population_sha256)
        or viterbi_manifest.get("checkpoint_sha256")
        != str(selector_checkpoint_sha256)
        or not viterbi_manifest.get("complete", False)
        or viterbi_manifest.get("files", {})
        .get("viterbi", {})
        .get("sha256")
        != viterbi_sha256
    ):
        raise RuntimeError(
            "Pretrained Viterbi artifact does not have the frozen "
            "NB07a identity."
        )

    required_bundle_fields = {
        "itercqr_model_sha256",
        "serializer_sha256",
        "query_bundle_sha256",
    }
    missing_bundle_fields = required_bundle_fields.difference(
        query_bundle_manifest
    )
    if missing_bundle_fields:
        raise KeyError(
            "Query-bundle manifest is missing: "
            + ", ".join(sorted(missing_bundle_fields))
        )

    current_serializer_sha256 = _sha256_text(
        inspect.getsource(serialize_itercqr_input)
    )
    current_model_sha256 = _sha256_file(
        Path(itercqr_model_dir) / "pytorch_model.bin"
    )
    if current_serializer_sha256 != str(
        query_bundle_manifest["serializer_sha256"]
    ):
        raise RuntimeError(
            "Current serializer differs from the frozen query bundle."
        )
    if current_model_sha256 != str(
        query_bundle_manifest["itercqr_model_sha256"]
    ):
        raise RuntimeError(
            "Current IterCQR model differs from the frozen query bundle."
        )

    identity = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "backend": "bm25",
        "population_sha256": str(population_sha256),
        "population_queries": len(normalized_ids),
        "selector_system": "pretrained",
        "selector_checkpoint_sha256": str(
            selector_checkpoint_sha256
        ),
        "selector_viterbi_sha256": viterbi_sha256,
        "main_inputs_sha256": _sha256_file(main_inputs_path),
        "token_metrics_sha256": _sha256_file(token_metrics_path),
        "token_summary_sha256": _sha256_file(token_summary_path),
        "rows_sha256": _canonical_sha256(list(rows)),
        "gold_sha256": _canonical_sha256(
            {
                str(sample_id): sorted(
                    str(docid) for docid in docids
                )
                for sample_id, docids in sorted(
                    gold_by_sample.items(),
                    key=lambda item: str(item[0]),
                )
            }
        ),
        "query_bundle_sha256": str(
            query_bundle_manifest["query_bundle_sha256"]
        ),
        "itercqr_model_sha256": current_model_sha256,
        "serializer_sha256": current_serializer_sha256,
        "projection_function_sha256": _sha256_text(
            inspect.getsource(expand_selected_histories)
        ),
        "evaluation_function_sha256": _sha256_text(
            inspect.getsource(evaluate_fixed_rrf_views)
        ),
        "budgets": list(normalized_budgets),
        "granularities": list(GRANULARITIES),
        "arms": list(ARMS),
        "rrf_k": int(rrf_k),
        "retrieval_top_k": int(retrieval_top_k),
        "eval_ks": [int(k) for k in eval_ks],
        "no_history_rule": "I_rewrite_for_I_R_dataset_current_query_for_D",
        "bm25_identity": dict(backend_identity),
    }

    artifact_presence = {
        "manifest.json": manifest_path.is_file(),
        **{
            filename: path.is_file()
            for filename, path in data_by_name.items()
        },
    }
    if any(artifact_presence.values()) and not all(
        artifact_presence.values()
    ):
        raise RuntimeError(
            "Granularity projection is only partially present: "
            + json.dumps(artifact_presence, sort_keys=True)
        )

    main_inputs = pd.read_csv(main_inputs_path)
    token_metrics_source = pd.read_csv(token_metrics_path)
    token_summary_source = pd.read_csv(token_summary_path)
    i_source, token_r_source, token_metrics = _validate_sources(
        main_inputs=main_inputs,
        token_metrics_source=token_metrics_source,
        sample_ids=sample_id_set,
        budgets=normalized_budgets,
    )

    if all(artifact_presence.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(
            manifest=manifest,
            identity=identity,
            data_by_name=data_by_name,
        )
        current_implementation = _sha256_file(Path(__file__))
        recorded_implementation = manifest.get(
            "implementation_v2_sha256"
        )
        if (
            recorded_implementation is not None
            and recorded_implementation != current_implementation
        ):
            raise RuntimeError(
                "Granularity implementation differs from its manifest."
            )
        metrics_by_query = pd.read_csv(
            data_by_name["metrics_by_query.csv"]
        )
        summary = pd.read_csv(data_by_name["summary.csv"])
        input_length_summary = pd.read_csv(
            data_by_name["input_length_summary.csv"]
        )
        serialization_summary = pd.read_csv(
            data_by_name["serialization_summary.csv"]
        )
        reused = True
    else:
        (
            metrics_by_query,
            summary,
            input_length_summary,
            serialization_summary,
        ) = _compute_projection(
            rows=rows,
            samples=samples,
            sample_ids=normalized_ids,
            depth_by_sample=depth_by_sample,
            gold_by_sample=gold_by_sample,
            i_source=i_source,
            token_r_source=token_r_source,
            token_metrics=token_metrics,
            pipeline_config=pipeline_config,
            itercqr_model_dir=Path(itercqr_model_dir),
            budgets=normalized_budgets,
            rrf_k=int(rrf_k),
            retrieval_top_k=int(retrieval_top_k),
            eval_ks=tuple(int(k) for k in eval_ks),
            viterbi_path=viterbi_path,
            progress=bool(progress),
        )
        projection_dir.mkdir(parents=True, exist_ok=True)
        _write_frame(
            data_by_name["metrics_by_query.csv"],
            metrics_by_query,
        )
        _write_frame(data_by_name["summary.csv"], summary)
        _write_frame(
            data_by_name["input_length_summary.csv"],
            input_length_summary,
        )
        _write_frame(
            data_by_name["serialization_summary.csv"],
            serialization_summary,
        )
        manifest = {
            **identity,
            "implementation_v2_sha256": _sha256_file(Path(__file__)),
            "metric_rows": len(metrics_by_query),
            "summary_rows": len(summary),
            "input_length_rows": len(input_length_summary),
            "serialization_rows": len(serialization_summary),
            "files": {
                filename: _sha256_file(path)
                for filename, path in data_by_name.items()
            },
            "complete": True,
            "selection_performed": False,
        }
        _write_json(manifest_path, manifest)
        reused = False

    _validate_outputs(
        metrics_by_query=metrics_by_query,
        summary=summary,
        input_length_summary=input_length_summary,
        serialization_summary=serialization_summary,
        token_metrics_source=token_metrics_source,
        token_summary_source=token_summary_source,
        sample_ids=sample_id_set,
        depth_by_sample=depth_by_sample,
        budgets=normalized_budgets,
    )
    return GranularityProjectionResult(
        metrics_by_query=metrics_by_query,
        summary=summary,
        input_length_summary=input_length_summary,
        serialization_summary=serialization_summary,
        manifest=dict(manifest),
        manifest_path=manifest_path,
        data_paths=data_paths,
        reused=reused,
    )


def _validate_sources(
    *,
    main_inputs: pd.DataFrame,
    token_metrics_source: pd.DataFrame,
    sample_ids: set[str],
    budgets: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_main = {
        "sample_id",
        "budget",
        "arm",
        "input_key",
        "input_length",
        "rewrite",
        "rewrite_norm",
        "rewrite_key",
    }
    missing_main = required_main.difference(main_inputs.columns)
    if missing_main:
        raise KeyError(
            "main_inputs.csv is missing: "
            + ", ".join(sorted(missing_main))
        )
    main_inputs = main_inputs.copy()
    main_inputs["sample_id"] = main_inputs["sample_id"].astype(str)
    i_source = main_inputs.loc[main_inputs["arm"].eq("I")].copy()
    token_r_source = main_inputs.loc[
        main_inputs["arm"].eq("pretrained_R")
    ].copy()
    expected_input_rows = len(sample_ids) * len(budgets)
    for name, frame in (("I", i_source), ("token_R", token_r_source)):
        if (
            len(frame) != expected_input_rows
            or set(frame["sample_id"]) != sample_ids
            or set(frame["budget"].astype(int)) != set(budgets)
            or frame.duplicated(["sample_id", "budget"]).any()
        ):
            raise RuntimeError(f"Persisted {name} inputs are incomplete.")

    required_metrics = {
        "system",
        "budget",
        "sample_id",
        "arm",
        "requested_views",
        "requested_view_count",
        "unique_view_count",
        "target_rank",
        *METRICS,
    }
    missing_metrics = required_metrics.difference(
        token_metrics_source.columns
    )
    if missing_metrics:
        raise KeyError(
            "Token metrics are missing: "
            + ", ".join(sorted(missing_metrics))
        )
    token_metrics_source = token_metrics_source.copy()
    token_metrics_source["sample_id"] = token_metrics_source[
        "sample_id"
    ].astype(str)
    token_metrics = token_metrics_source.loc[
        token_metrics_source["system"].eq("pretrained")
        & token_metrics_source["budget"].isin(budgets)
        & token_metrics_source["arm"].isin(ARMS)
    ].copy()
    expected_metric_rows = (
        len(sample_ids) * len(budgets) * len(ARMS)
    )
    if (
        len(token_metrics) != expected_metric_rows
        or token_metrics.duplicated(
            ["budget", "sample_id", "arm"]
        ).any()
        or set(token_metrics["sample_id"]) != sample_ids
    ):
        raise RuntimeError("Canonical token metrics are incomplete.")
    return i_source, token_r_source, token_metrics


def _compute_projection(
    *,
    rows: Sequence[dict[str, Any]],
    samples: Sequence[Any],
    sample_ids: Sequence[str],
    depth_by_sample: Mapping[str, int],
    gold_by_sample: Mapping[str, Sequence[Any]],
    i_source: pd.DataFrame,
    token_r_source: pd.DataFrame,
    token_metrics: pd.DataFrame,
    pipeline_config: CachedIterCQRBM25Config,
    itercqr_model_dir: Path,
    budgets: Sequence[int],
    rrf_k: int,
    retrieval_top_k: int,
    eval_ks: Sequence[int],
    viterbi_path: Path,
    progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    token_histories = read_selected_histories_jsonl(
        viterbi_path,
        progress=progress,
    )
    if set(token_histories) != set(sample_ids):
        raise RuntimeError(
            "Viterbi selections and full-dev population differ."
        )
    projected_histories = {
        "qa_R": expand_selected_histories(
            rows,
            token_histories,
            granularity="qa",
        ),
        "turn_R": expand_selected_histories(
            rows,
            token_histories,
            granularity="turn",
        ),
    }

    tokenizer = load_itercqr_tokenizer(itercqr_model_dir)
    pipeline = CachedIterCQRBM25Pipeline(
        config=pipeline_config,
        tokenizer=tokenizer,
    )
    try:
        r_serialized, serialization_summary = (
            materialize_history_arm_inputs(
                samples=samples,
                histories_by_arm=projected_histories,
                arm_order=("qa_R", "turn_R"),
                tokenizer=tokenizer,
                budgets=budgets,
                pipeline=pipeline,
                progress=progress,
                progress_desc="serialize Q/A and turn projections",
            )
        )
        r_result = pipeline.run(
            r_serialized,
            gold_by_sample=gold_by_sample,
            metric_columns=METRICS,
            progress_prefix="NB07a granularity R",
        )
        r_evaluated = r_result.evaluated.copy()

        i_result = pipeline.run_bm25(
            i_source,
            gold_by_sample=gold_by_sample,
            query_column="rewrite",
            metric_columns=METRICS,
            progress_prefix="NB07a granularity I",
        )
        i_evaluated = i_result.evaluated.copy()
        persisted_i_keys = i_source[
            ["sample_id", "budget", "rewrite_key"]
        ].sort_values(["budget", "sample_id"], kind="mergesort")
        computed_i_keys = i_evaluated[
            ["sample_id", "budget", "rewrite_key"]
        ].sort_values(["budget", "sample_id"], kind="mergesort")
        if not persisted_i_keys.reset_index(drop=True).equals(
            computed_i_keys.reset_index(drop=True)
        ):
            raise RuntimeError("I rewrite keys drift from the query bundle.")

        unique_inputs = (
            r_evaluated[["input_key"]]
            .drop_duplicates("input_key")
            .sort_values("input_key", kind="mergesort")
        )
        sequences = pipeline.load_token_inputs(
            unique_inputs["input_key"].astype(str).tolist()
        )
        direct_by_input = dict(
            zip(
                unique_inputs["input_key"].astype(str),
                [
                    str(
                        tokenizer.decode(
                            sequence,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True,
                        )
                    ).strip()
                    for sequence in sequences
                ],
                strict=True,
            )
        )
        d_sources = r_evaluated[
            ["sample_id", "budget", "arm", "input_key"]
        ].copy()
        d_sources["granularity"] = d_sources["arm"].map(
            {"qa_R": "qa", "turn_R": "turn"}
        )
        d_sources["direct_query"] = d_sources["input_key"].map(
            direct_by_input
        )
        current_query_by_sample = {
            str(row["sample_id"]): str(row["current_query"])
            for row in rows
        }
        no_history = d_sources["sample_id"].map(depth_by_sample).eq(0)
        d_sources.loc[no_history, "direct_query"] = d_sources.loc[
            no_history,
            "sample_id",
        ].map(current_query_by_sample)
        d_result = pipeline.run_bm25(
            d_sources,
            gold_by_sample=gold_by_sample,
            query_column="direct_query",
            metric_columns=METRICS,
            progress_prefix="NB07a granularity D",
        )
        d_evaluated = d_result.evaluated.copy()

        projected_query_views = _build_projected_query_views(
            i_evaluated=i_evaluated,
            r_evaluated=r_evaluated,
            d_evaluated=d_evaluated,
            sample_ids=sample_ids,
            depth_by_sample=depth_by_sample,
            budgets=budgets,
        )
        raw_rankings = pipeline.load_rankings(
            projected_query_views["query_id"].astype(str).tolist(),
            progress_prefix="NB07a granularity",
        )
        hits = {
            query_id: [
                {"docid": str(docid), "rank": rank}
                for rank, docid in enumerate(docids, start=1)
            ]
            for query_id, docids in raw_rankings.items()
        }
        projected_metrics = evaluate_fixed_rrf_views(
            query_views=projected_query_views,
            hits_by_query_id=hits,
            gold_by_sample=gold_by_sample,
            view_sets=VIEW_SETS,
            group_columns=("granularity", "budget"),
            rrf_k=rrf_k,
            top_k=retrieval_top_k,
            ks=eval_ks,
            measure_latency=False,
            progress=progress,
        )
    finally:
        pipeline.close()

    token_metrics = token_metrics.drop(
        columns=["system", "fusion_seconds"],
        errors="ignore",
    ).copy()
    token_metrics["granularity"] = "token"
    missing_projected_columns = set(projected_metrics.columns).difference(
        token_metrics.columns
    )
    if missing_projected_columns:
        raise RuntimeError(
            "Token metrics cannot match projected schema: "
            + ", ".join(sorted(missing_projected_columns))
        )
    token_metrics = token_metrics.loc[:, projected_metrics.columns]
    metrics_by_query = pd.concat(
        [token_metrics, projected_metrics],
        ignore_index=True,
    )
    summary = _frame_summary(
        metrics_by_query,
        ("granularity", "budget", "arm"),
    )
    input_length_summary = _input_length_summary(
        i_source=i_source,
        token_r_source=token_r_source,
        r_serialized=r_serialized,
        budgets=budgets,
    )
    metrics_by_query = _sort_metrics(metrics_by_query)
    summary = _sort_summary(summary)
    input_length_summary = _sort_lengths(input_length_summary)
    serialization_summary = serialization_summary.sort_values(
        ["budget", "arm"],
        kind="mergesort",
    ).reset_index(drop=True)
    return (
        metrics_by_query,
        summary,
        input_length_summary,
        serialization_summary,
    )


def _build_projected_query_views(
    *,
    i_evaluated: pd.DataFrame,
    r_evaluated: pd.DataFrame,
    d_evaluated: pd.DataFrame,
    sample_ids: Sequence[str],
    depth_by_sample: Mapping[str, int],
    budgets: Sequence[int],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    i_base = i_evaluated[
        ["sample_id", "budget", "rewrite_norm", "rewrite_key"]
    ].copy()
    for granularity, r_arm in (("qa", "qa_R"), ("turn", "turn_R")):
        i_view = i_base.rename(
            columns={
                "rewrite_norm": "query_norm",
                "rewrite_key": "query_id",
            }
        ).assign(granularity=granularity, view="I")

        r_view = r_evaluated.loc[
            r_evaluated["arm"].eq(r_arm),
            ["sample_id", "budget", "rewrite_norm", "rewrite_key"],
        ].merge(
            i_base.rename(
                columns={
                    "rewrite_norm": "i_query_norm",
                    "rewrite_key": "i_query_id",
                }
            ),
            on=["sample_id", "budget"],
            how="left",
            validate="one_to_one",
        )
        no_history = r_view["sample_id"].map(depth_by_sample).eq(0)
        r_view["query_norm"] = r_view["rewrite_norm"]
        r_view["query_id"] = r_view["rewrite_key"]
        r_view.loc[no_history, "query_norm"] = r_view.loc[
            no_history,
            "i_query_norm",
        ]
        r_view.loc[no_history, "query_id"] = r_view.loc[
            no_history,
            "i_query_id",
        ]
        r_view = r_view[
            ["sample_id", "budget", "query_norm", "query_id"]
        ].assign(granularity=granularity, view="R")

        d_view = d_evaluated.loc[
            d_evaluated["granularity"].eq(granularity),
            ["sample_id", "budget", "rewrite_norm", "rewrite_key"],
        ].rename(
            columns={
                "rewrite_norm": "query_norm",
                "rewrite_key": "query_id",
            }
        )
        d_view = d_view.assign(granularity=granularity, view="D")
        frames.extend([i_view, r_view, d_view])

    query_views = pd.concat(frames, ignore_index=True)[
        [
            "granularity",
            "budget",
            "sample_id",
            "view",
            "query_id",
            "query_norm",
        ]
    ]
    expected_rows = 2 * len(budgets) * len(sample_ids) * 3
    if (
        len(query_views) != expected_rows
        or query_views.duplicated(
            ["granularity", "budget", "sample_id", "view"]
        ).any()
    ):
        raise RuntimeError("Projected query views are incomplete.")
    return query_views


def _input_length_summary(
    *,
    i_source: pd.DataFrame,
    token_r_source: pd.DataFrame,
    r_serialized: pd.DataFrame,
    budgets: Sequence[int],
) -> pd.DataFrame:
    i_lengths = (
        i_source.groupby("budget", observed=True)
        .agg(
            n=("sample_id", "nunique"),
            mean_I_t5_input_tokens=("input_length", "mean"),
        )
        .reset_index()
    )
    sources = {
        "token": token_r_source,
        "qa": r_serialized.loc[r_serialized["arm"].eq("qa_R")],
        "turn": r_serialized.loc[r_serialized["arm"].eq("turn_R")],
    }
    rows: list[pd.DataFrame] = []
    for granularity, source in sources.items():
        r_lengths = (
            source.groupby("budget", observed=True)
            .agg(
                n_R=("sample_id", "nunique"),
                mean_R_t5_input_tokens=("input_length", "mean"),
            )
            .reset_index()
        )
        combined = i_lengths.merge(
            r_lengths,
            on="budget",
            how="inner",
            validate="one_to_one",
        )
        combined.insert(0, "granularity", granularity)
        rows.append(combined)
    result = pd.concat(rows, ignore_index=True)
    if set(result["budget"].astype(int)) != set(map(int, budgets)):
        raise RuntimeError("Input-length budgets are incomplete.")
    result["delta_R_minus_I_t5_input_tokens"] = (
        result["mean_R_t5_input_tokens"]
        - result["mean_I_t5_input_tokens"]
    )
    result["R_to_I_input_ratio"] = (
        result["mean_R_t5_input_tokens"]
        / result["mean_I_t5_input_tokens"]
    )
    result["relative_input_reduction"] = (
        1.0 - result["R_to_I_input_ratio"]
    )
    return result


def _validate_outputs(
    *,
    metrics_by_query: pd.DataFrame,
    summary: pd.DataFrame,
    input_length_summary: pd.DataFrame,
    serialization_summary: pd.DataFrame,
    token_metrics_source: pd.DataFrame,
    token_summary_source: pd.DataFrame,
    sample_ids: set[str],
    depth_by_sample: Mapping[str, int],
    budgets: Sequence[int],
) -> None:
    expected_metric_rows = (
        len(GRANULARITIES) * len(budgets) * len(ARMS) * len(sample_ids)
    )
    required_metric_columns = {
        "granularity",
        "budget",
        "sample_id",
        "arm",
        "requested_view_count",
        "unique_view_count",
        *METRICS,
    }
    if (
        not required_metric_columns.issubset(metrics_by_query.columns)
        or len(metrics_by_query) != expected_metric_rows
        or metrics_by_query.duplicated(
            ["granularity", "budget", "arm", "sample_id"]
        ).any()
        or set(metrics_by_query["granularity"]) != set(GRANULARITIES)
        or set(metrics_by_query["arm"]) != set(ARMS)
        or set(metrics_by_query["budget"].astype(int)) != set(budgets)
        or set(metrics_by_query["sample_id"].astype(str)) != sample_ids
    ):
        raise RuntimeError("Granularity metrics are incomplete.")

    expected_summary_rows = len(GRANULARITIES) * len(budgets) * len(ARMS)
    if (
        len(summary) != expected_summary_rows
        or set(summary["n"].astype(int)) != {len(sample_ids)}
        or summary.duplicated(["granularity", "budget", "arm"]).any()
    ):
        raise RuntimeError("Granularity summary is incomplete.")
    if (
        len(input_length_summary) != len(GRANULARITIES) * len(budgets)
        or set(input_length_summary["n"].astype(int)) != {len(sample_ids)}
        or set(input_length_summary["n_R"].astype(int))
        != {len(sample_ids)}
        or input_length_summary.duplicated(
            ["granularity", "budget"]
        ).any()
    ):
        raise RuntimeError("Input-length summary is incomplete.")
    if (
        len(serialization_summary) != 2 * len(budgets)
        or set(serialization_summary["queries"].astype(int))
        != {len(sample_ids)}
        or set(serialization_summary["arm"]) != {"qa_R", "turn_R"}
    ):
        raise RuntimeError("Serialization summary is incomplete.")

    recomputed = _frame_summary(
        metrics_by_query,
        ("granularity", "budget", "arm"),
    )
    summary_check = recomputed.merge(
        summary,
        on=["granularity", "budget", "arm"],
        suffixes=("_expected", "_observed"),
        validate="one_to_one",
    )
    if not (
        (
            summary_check["n_expected"]
            == summary_check["n_observed"]
        ).all()
        and all(
            np.allclose(
                summary_check[f"{metric}_expected"],
                summary_check[f"{metric}_observed"],
                rtol=0.0,
                atol=1e-12,
            )
            for metric in METRICS
        )
    ):
        raise RuntimeError("Granularity summary drifts from query metrics.")

    expected_token_from_queries = _frame_summary(
        token_metrics_source.loc[
            token_metrics_source["system"].eq("pretrained")
            & token_metrics_source["budget"].isin(budgets)
            & token_metrics_source["arm"].isin(ARMS)
        ],
        ("budget", "arm"),
    )
    expected_token = token_summary_source.loc[
        token_summary_source["system"].eq("pretrained")
        & token_summary_source["budget"].isin(budgets)
        & token_summary_source["arm"].isin(ARMS),
        ["budget", "arm", "n", *METRICS],
    ].copy()
    source_check = expected_token_from_queries.merge(
        expected_token,
        on=["budget", "arm"],
        suffixes=("_queries", "_summary"),
        validate="one_to_one",
    )
    if not (
        len(expected_token) == len(budgets) * len(ARMS)
        and (
            source_check["n_queries"]
            == source_check["n_summary"]
        ).all()
        and all(
            np.allclose(
                source_check[f"{metric}_queries"],
                source_check[f"{metric}_summary"],
                rtol=0.0,
                atol=1e-12,
            )
            for metric in METRICS
        )
    ):
        raise RuntimeError(
            "Canonical token route summary drifts from query metrics."
        )
    observed_token = summary.loc[
        summary["granularity"].eq("token")
    ].drop(columns="granularity")
    token_check = expected_token.merge(
        observed_token,
        on=["budget", "arm"],
        suffixes=("_expected", "_observed"),
        validate="one_to_one",
    )
    if not all(
        np.allclose(
            token_check[f"{metric}_expected"],
            token_check[f"{metric}_observed"],
            rtol=0.0,
            atol=1e-12,
        )
        for metric in METRICS
    ):
        raise RuntimeError("Token branch drifts from canonical NB07a.")

    no_history_ids = {
        str(sample_id)
        for sample_id, depth in depth_by_sample.items()
        if int(depth) == 0
    }
    no_history_ird = metrics_by_query.loc[
        metrics_by_query["sample_id"].astype(str).isin(no_history_ids)
        & metrics_by_query["arm"].eq("I+R+D")
    ]
    if (
        len(no_history_ird)
        != len(no_history_ids) * len(GRANULARITIES) * len(budgets)
        or not no_history_ird["requested_view_count"].eq(3).all()
        or not no_history_ird["unique_view_count"].isin((1, 2)).all()
    ):
        raise RuntimeError(
            "No-history I and R must collapse while D may add the direct query."
        )

    canonical_i = metrics_by_query.loc[
        metrics_by_query["granularity"].eq("token")
        & metrics_by_query["arm"].eq("I")
    ].set_index(["budget", "sample_id"])
    for granularity in ("qa", "turn"):
        projected_i = metrics_by_query.loc[
            metrics_by_query["granularity"].eq(granularity)
            & metrics_by_query["arm"].eq("I")
        ].set_index(["budget", "sample_id"])
        for metric in METRICS:
            if not np.allclose(
                canonical_i.loc[projected_i.index, metric],
                projected_i[metric],
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"I metrics drift for {granularity}: {metric}"
                )


def _frame_summary(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    return (
        frame.groupby(list(group_columns), observed=True)
        .agg(
            n=("sample_id", "nunique"),
            **{metric: (metric, "mean") for metric in METRICS},
        )
        .reset_index()
    )


def _sort_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    return _sort_with_order(
        frame,
        columns=("granularity", "budget", "arm", "sample_id"),
    )


def _sort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return _sort_with_order(
        frame,
        columns=("granularity", "budget", "arm"),
    )


def _sort_lengths(frame: pd.DataFrame) -> pd.DataFrame:
    return _sort_with_order(
        frame,
        columns=("granularity", "budget"),
    )


def _sort_with_order(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    result = frame.copy()
    result["_granularity_order"] = result["granularity"].map(
        {value: index for index, value in enumerate(GRANULARITIES)}
    )
    if "arm" in result:
        result["_arm_order"] = result["arm"].map(
            {value: index for index, value in enumerate(ARMS)}
        )
    sort_columns = [
        (
            "_granularity_order"
            if column == "granularity"
            else "_arm_order"
            if column == "arm"
            else column
        )
        for column in columns
    ]
    return (
        result.sort_values(sort_columns, kind="mergesort")
        .drop(
            columns=["_granularity_order", "_arm_order"],
            errors="ignore",
        )
        .reset_index(drop=True)
    )


def _validate_manifest(
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    data_by_name: Mapping[str, Path],
) -> None:
    for field, expected in identity.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"Granularity manifest drifts at {field!r}."
            )
    if not manifest.get("complete", False):
        raise RuntimeError("Granularity manifest is incomplete.")
    recorded = manifest.get("files", {})
    if set(recorded) != set(data_by_name):
        raise RuntimeError(
            "Granularity manifest contains the wrong artifact set."
        )
    for filename, path in data_by_name.items():
        if (
            not path.is_file()
            or _sha256_file(path) != recorded[filename]
        ):
            raise RuntimeError(f"Granularity artifact drifts: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
