"""Notebook-friendly end-to-end ROCC pipelines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .datasets.itercqr import IterCQRDataPipelineResult, load_itercqr_dataloaders
from .itercqr_components import load_itercqr_rewriter, load_itercqr_t5_for_inference
from .paths import project_path


RetrieverKind = Literal["sparse", "dense"]
DatasetName = Literal["topiocqa", "qrecc"]
SplitName = Literal["train", "dev", "test"]
QReCCDenseMode = Literal["run_partial", "merge_eval"]


@dataclass(frozen=True)
class HolisticPipelineConfig:
    dataset_name: str = "topiocqa"
    retriever_kind: RetrieverKind = "sparse"
    splits: tuple[str, ...] = ("dev",)
    batch_size: int = 32
    data_dir: Path | None = None
    model_dir: Path | None = None
    retriever_index_dir: Path | None = None
    progress: bool = True
    max_query_tokens: int = 32
    max_history_turn_tokens: int = 64
    max_history_answer_tokens: int = 32
    max_input_tokens: int = 512
    include_history: bool = True


@dataclass
class HolisticPipeline:
    config: HolisticPipelineConfig
    data: IterCQRDataPipelineResult | None = None
    model: Any | None = None
    tokenizer: Any | None = None
    retriever: Any | None = None

    def prepare_data(self) -> IterCQRDataPipelineResult:
        self.data = load_itercqr_dataloaders(
            dataset_name=self.config.dataset_name,  # type: ignore[arg-type]
            splits=self.config.splits,  # type: ignore[arg-type]
            batch_size=self.config.batch_size,
            data_dir=self.config.data_dir,
            model_dir=self.config.model_dir,
            progress=self.config.progress,
            max_query_tokens=self.config.max_query_tokens,
            max_history_turn_tokens=self.config.max_history_turn_tokens,
            max_history_answer_tokens=self.config.max_history_answer_tokens,
            max_input_tokens=self.config.max_input_tokens,
            include_history=self.config.include_history,
        )
        self.tokenizer = self.data.tokenizer
        return self.data

    def load_itercqr_model(self, *, device: str | None = None) -> Any:
        self.model, self.tokenizer, resolved_device = load_itercqr_t5_for_inference(
            model_dir=self.config.model_dir,
            device=device,
        )
        return self.model, self.tokenizer, resolved_device

    def load_first_stage_retriever(self) -> Any:
        if self.config.retriever_kind == "sparse":
            from .retrievers import load_bm25_retriever

            self.retriever = load_bm25_retriever(self.config.retriever_index_dir)
            return self.retriever
        if self.config.retriever_kind == "dense":
            from .retrievers import load_dense_faiss_retriever

            self.retriever = load_dense_faiss_retriever(self.config.retriever_index_dir)
            return self.retriever
        raise ValueError(f"Unsupported retriever_kind: {self.config.retriever_kind}")

    def prepare(
        self,
        *,
        load_model: bool = False,
        load_retriever: bool = False,
        device: str | None = None,
    ) -> "HolisticPipeline":
        self.prepare_data()
        if load_model:
            self.load_itercqr_model(device=device)
        if load_retriever:
            self.load_first_stage_retriever()
        return self


def build_holistic_pipeline(
    config: HolisticPipelineConfig | None = None,
    *,
    prepare: bool = False,
    load_model: bool = False,
    load_retriever: bool = False,
    device: str | None = None,
) -> HolisticPipeline:
    pipeline = HolisticPipeline(config or HolisticPipelineConfig())
    if prepare:
        pipeline.prepare(
            load_model=load_model,
            load_retriever=load_retriever,
            device=device,
        )
    return pipeline


@dataclass
class IterCQRFullPipelineResult:
    config: dict[str, Any]
    status: pd.DataFrame
    data_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    capacity_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    pipeline_result: Any | None = None
    evaluation: Any | None = None
    partial_dump: Any | None = None
    merge_result: Any | None = None
    eval_result_paths: dict[str, str | None] = field(default_factory=dict)


@dataclass
class IterCQRBudgetSweepResult:
    config: dict[str, Any]
    budget_results: dict[int, IterCQRFullPipelineResult]
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    status: pd.DataFrame = field(default_factory=pd.DataFrame)
    evaluation_aggregate: pd.DataFrame = field(default_factory=pd.DataFrame)
    bootstrap_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def run_itercqr_full_pipeline(
    *,
    dataset_name: DatasetName | str,
    split: SplitName | str,
    retriever_kind: RetrieverKind | str,
    data_dir: Path | str | None = None,
    bm25_index_dir: Path | str | None = None,
    dense_index_dir: Path | str | None = None,
    azure_sas_file: Path | str | None = None,
    qrecc_session_id: str = "session_1",
    qrecc_dense_mode: QReCCDenseMode | str = "run_partial",
    run_id: str | None = None,
    output_dir: Path | str | None = None,
    batch_size: int = 16,
    top_k: int = 100,
    eval_ks: tuple[int, ...] = (3, 10, 100),
    bootstrap_samples: int = 10_000,
    save_per_query: bool = True,
    max_batches: int | None = None,
    device: str | None = None,
    progress: bool = True,
    bm25_k1: float | None = None,
    bm25_b: float | None = None,
    max_query_tokens: int = 32,
    max_history_turn_tokens: int = 64,
    max_history_answer_tokens: int = 32,
    max_input_tokens: int = 512,
    include_history: bool = True,
) -> IterCQRFullPipelineResult:
    """Run a compact IterCQR baseline pipeline and persist eval summaries only."""

    normalized_dataset = str(dataset_name).lower()
    normalized_split = str(split).lower()
    normalized_retriever = str(retriever_kind).lower()
    normalized_qrecc_mode = str(qrecc_dense_mode).lower()
    if normalized_dataset not in {"topiocqa", "qrecc"}:
        raise ValueError("dataset_name must be 'topiocqa' or 'qrecc'.")
    if normalized_split not in {"train", "dev", "test"}:
        raise ValueError("split must be 'train', 'dev', or 'test'.")
    if normalized_retriever not in {"sparse", "dense"}:
        raise ValueError("retriever_kind must be 'sparse' or 'dense'.")
    if normalized_qrecc_mode not in {"run_partial", "merge_eval"}:
        raise ValueError("qrecc_dense_mode must be 'run_partial' or 'merge_eval'.")

    resolved_run_id = run_id or f"{normalized_dataset}_{normalized_split}_{normalized_retriever}"
    resolved_data_dir = _default_data_dir(normalized_dataset, data_dir)
    resolved_bm25_index_dir = _default_bm25_index_dir(normalized_dataset, bm25_index_dir)
    resolved_dense_index_dir = _default_dense_index_dir(normalized_dataset, dense_index_dir)
    resolved_output_dir = _default_output_dir(resolved_run_id, output_dir)
    resolved_sas_file = _resolve_existing_sas_file(azure_sas_file)
    resolved_device = device or _default_device()

    config = {
        "dataset_name": normalized_dataset,
        "split": normalized_split,
        "retriever_kind": normalized_retriever,
        "data_dir": str(resolved_data_dir),
        "bm25_index_dir": str(resolved_bm25_index_dir),
        "dense_index_dir": str(resolved_dense_index_dir),
        "azure_sas_file": str(resolved_sas_file) if resolved_sas_file else None,
        "qrecc_session_id": qrecc_session_id,
        "qrecc_dense_mode": normalized_qrecc_mode,
        "run_id": resolved_run_id,
        "output_dir": str(resolved_output_dir),
        "batch_size": int(batch_size),
        "top_k": int(top_k),
        "eval_ks": tuple(int(k) for k in eval_ks),
        "bootstrap_samples": int(bootstrap_samples),
        "save_per_query": bool(save_per_query),
        "max_batches": max_batches,
        "device": resolved_device,
        "bm25_k1": None if bm25_k1 is None else float(bm25_k1),
        "bm25_b": None if bm25_b is None else float(bm25_b),
        "max_query_tokens": int(max_query_tokens),
        "max_history_turn_tokens": int(max_history_turn_tokens),
        "max_history_answer_tokens": int(max_history_answer_tokens),
        "max_input_tokens": int(max_input_tokens),
        "include_history": bool(include_history),
    }

    status_rows: list[dict[str, Any]] = []
    if normalized_dataset == "qrecc" and normalized_retriever == "dense" and normalized_qrecc_mode == "merge_eval":
        result = _run_qrecc_dense_merge_eval(
            config=config,
            status_rows=status_rows,
            progress=progress,
        )
        _tag_full_pipeline_result_with_budget(
            result,
            int(config["max_input_tokens"]),
            include_history=bool(config["include_history"]),
        )
        return result

    result = _run_rewrite_retrieve_eval(
        config=config,
        status_rows=status_rows,
        progress=progress,
    )
    _tag_full_pipeline_result_with_budget(
        result,
        int(config["max_input_tokens"]),
        include_history=bool(config["include_history"]),
    )
    return result


def run_itercqr_budget_sweep(
    *,
    dataset_name: DatasetName | str,
    split: SplitName | str,
    retriever_kind: RetrieverKind | str,
    budgets: tuple[int, ...] = (512, 256, 128, 64),
    data_dir: Path | str | None = None,
    bm25_index_dir: Path | str | None = None,
    dense_index_dir: Path | str | None = None,
    azure_sas_file: Path | str | None = None,
    qrecc_session_id: str = "session_1",
    qrecc_dense_mode: QReCCDenseMode | str = "run_partial",
    run_id: str | None = None,
    output_dir: Path | str | None = None,
    batch_size: int = 16,
    top_k: int = 100,
    eval_ks: tuple[int, ...] = (3, 10, 100),
    bootstrap_samples: int = 10_000,
    save_per_query: bool = True,
    max_batches: int | None = None,
    device: str | None = None,
    progress: bool = True,
    rewriter: Any | None = None,
    retriever: Any | None = None,
    bm25_k1: float | None = None,
    bm25_b: float | None = None,
    max_query_tokens: int = 32,
    max_history_turn_tokens: int = 64,
    max_history_answer_tokens: int = 32,
    include_history: bool = True,
) -> IterCQRBudgetSweepResult:
    """Run the IterCQR baseline for several global tokenizer budgets."""

    normalized_dataset = str(dataset_name).lower()
    normalized_split = str(split).lower()
    normalized_retriever = str(retriever_kind).lower()
    normalized_qrecc_mode = str(qrecc_dense_mode).lower()
    if normalized_dataset not in {"topiocqa", "qrecc"}:
        raise ValueError("dataset_name must be 'topiocqa' or 'qrecc'.")
    if normalized_split not in {"train", "dev", "test"}:
        raise ValueError("split must be 'train', 'dev', or 'test'.")
    if normalized_retriever not in {"sparse", "dense"}:
        raise ValueError("retriever_kind must be 'sparse' or 'dense'.")
    if normalized_qrecc_mode not in {"run_partial", "merge_eval"}:
        raise ValueError("qrecc_dense_mode must be 'run_partial' or 'merge_eval'.")

    base_run_id = run_id or f"{normalized_dataset}_{normalized_split}_{normalized_retriever}"
    normalized_budgets = tuple(int(budget) for budget in budgets)
    if not normalized_budgets:
        raise ValueError("budgets must contain at least one value.")

    resolved_data_dir = _default_data_dir(normalized_dataset, data_dir)
    resolved_bm25_index_dir = _default_bm25_index_dir(normalized_dataset, bm25_index_dir)
    resolved_dense_index_dir = _default_dense_index_dir(normalized_dataset, dense_index_dir)
    resolved_sas_file = _resolve_existing_sas_file(azure_sas_file)
    resolved_device = device or _default_device()
    base_config = {
        "dataset_name": normalized_dataset,
        "split": normalized_split,
        "retriever_kind": normalized_retriever,
        "data_dir": str(resolved_data_dir),
        "bm25_index_dir": str(resolved_bm25_index_dir),
        "dense_index_dir": str(resolved_dense_index_dir),
        "azure_sas_file": str(resolved_sas_file) if resolved_sas_file else None,
        "qrecc_session_id": qrecc_session_id,
        "qrecc_dense_mode": normalized_qrecc_mode,
        "run_id": base_run_id,
        "output_dir": str(_default_output_dir(base_run_id, output_dir)),
        "batch_size": int(batch_size),
        "top_k": int(top_k),
        "eval_ks": tuple(int(k) for k in eval_ks),
        "bootstrap_samples": int(bootstrap_samples),
        "save_per_query": bool(save_per_query),
        "max_batches": max_batches,
        "device": resolved_device,
        "bm25_k1": None if bm25_k1 is None else float(bm25_k1),
        "bm25_b": None if bm25_b is None else float(bm25_b),
        "max_query_tokens": int(max_query_tokens),
        "max_history_turn_tokens": int(max_history_turn_tokens),
        "max_history_answer_tokens": int(max_history_answer_tokens),
        "max_input_tokens": normalized_budgets[0],
        "include_history": bool(include_history),
    }

    budget_results: dict[int, IterCQRFullPipelineResult] = {}
    if normalized_dataset == "qrecc" and normalized_retriever == "dense" and normalized_qrecc_mode == "merge_eval":
        for budget in normalized_budgets:
            budget_config = _budget_config(
                base_config,
                base_run_id=base_run_id,
                budget=budget,
                output_dir=output_dir,
            )
            status_rows: list[dict[str, Any]] = []
            result = _run_qrecc_dense_merge_eval(
                config=budget_config,
                status_rows=status_rows,
                progress=progress,
            )
            _tag_full_pipeline_result_with_budget(
                result,
                budget,
                include_history=bool(budget_config["include_history"]),
            )
            budget_results[budget] = result
        return _budget_sweep_result(
            base_config=base_config,
            budgets=normalized_budgets,
            budget_results=budget_results,
        )

    setup_status_rows: list[dict[str, Any]] = []
    resource_summary = _step(
        setup_status_rows,
        1,
        4,
        "Prepare shared dataset resources",
        progress,
        lambda: _prepare_dataset_resources(base_config, progress=progress),
    )
    resolved_rewriter = rewriter
    if resolved_rewriter is None:
        resolved_rewriter = _step(
            setup_status_rows,
            2,
            4,
            "Load shared IterCQR rewriter",
            progress,
            lambda: load_itercqr_rewriter(device=base_config["device"]),
        )
    elif progress:
        print("[2/4] Use shared IterCQR rewriter", flush=True)

    if retriever is None:
        retriever_index_result = _step(
            setup_status_rows,
            3,
            4,
            "Prepare shared retriever index",
            progress,
            lambda: _prepare_retriever_index(base_config),
        )
        resolved_retriever = _step(
            setup_status_rows,
            4,
            4,
            "Load shared retriever",
            progress,
            lambda: _load_retriever(base_config, retriever_index_result),
        )
    else:
        retriever_index_result = _preloaded_retriever_index_result(base_config, retriever)
        resolved_retriever = retriever
        if progress:
            print("[3/4] Use shared retriever index", flush=True)
            print("[4/4] Use shared retriever", flush=True)

    for budget in normalized_budgets:
        budget_config = _budget_config(
            base_config,
            base_run_id=base_run_id,
            budget=budget,
            output_dir=output_dir,
        )
        status_rows: list[dict[str, Any]] = []
        result = _run_budget_rewrite_retrieve_eval(
            config=budget_config,
            status_rows=status_rows,
            progress=progress,
            resource_summary=resource_summary,
            rewriter=resolved_rewriter,
            retriever=resolved_retriever,
            retriever_index_result=retriever_index_result,
        )
        _tag_full_pipeline_result_with_budget(
            result,
            budget,
            include_history=bool(budget_config["include_history"]),
        )
        budget_results[budget] = result

    return _budget_sweep_result(
        base_config=base_config,
        budgets=normalized_budgets,
        budget_results=budget_results,
    )


def _run_rewrite_retrieve_eval(
    *,
    config: dict[str, Any],
    status_rows: list[dict[str, Any]],
    progress: bool,
) -> IterCQRFullPipelineResult:
    total_steps = 8
    dataset_name = str(config["dataset_name"])
    split = str(config["split"])
    retriever_kind = str(config["retriever_kind"])
    qrecc_mode = str(config["qrecc_dense_mode"])
    qrecc_session_id = str(config["qrecc_session_id"])

    data_summary = pd.DataFrame()
    index_summary = pd.DataFrame()
    capacity_summary = pd.DataFrame()
    pipeline_result = None
    evaluation = None
    partial_dump = None
    merge_result = None
    eval_result_paths: dict[str, str | None] = {}

    _step(status_rows, 1, total_steps, "Resolve paths and config", progress, lambda: None)

    resource_summary: dict[str, Any] = {}

    def prepare_resources_and_load_data() -> IterCQRDataPipelineResult:
        nonlocal resource_summary
        resource_summary = _prepare_dataset_resources(config, progress=progress)
        dataloader_kwargs: dict[str, Any] = {}
        if dataset_name == "topiocqa":
            dataloader_kwargs["topiocqa_download_corpus"] = False
        return load_itercqr_dataloaders(
            dataset_name=dataset_name,
            splits=(split,),
            batch_size=int(config["batch_size"]),
            data_dir=Path(str(config["data_dir"])),
            progress=progress,
            max_query_tokens=int(config["max_query_tokens"]),
            max_history_turn_tokens=int(config["max_history_turn_tokens"]),
            max_history_answer_tokens=int(config["max_history_answer_tokens"]),
            max_input_tokens=int(config["max_input_tokens"]),
            include_history=bool(config["include_history"]),
            **dataloader_kwargs,
        )

    data = _step(
        status_rows,
        2,
        total_steps,
        "Prepare resources and load dataset split",
        progress,
        prepare_resources_and_load_data,
    )
    dataloader = data.dataloader(split)
    _validate_itercqr_budget_dataloader(
        dataloader,
        budget=int(config["max_input_tokens"]),
        max_query_tokens=int(config["max_query_tokens"]),
        max_history_answer_tokens=int(config["max_history_answer_tokens"]),
        include_history=bool(config["include_history"]),
    )
    data_summary = _data_summary(data, split, resource_summary)

    rewriter = _step(status_rows, 3, total_steps, "Load IterCQR rewriter", progress, lambda: load_itercqr_rewriter(
        device=config["device"],
    ))

    retriever_index_result = _step(
        status_rows,
        4,
        total_steps,
        "Prepare retriever index",
        progress,
        lambda: _prepare_retriever_index(config),
    )
    index_summary = retriever_index_result["index_summary"]
    capacity_summary = retriever_index_result["capacity_summary"]

    retriever = _step(
        status_rows,
        5,
        total_steps,
        "Load retriever",
        progress,
        lambda: _load_retriever(config, retriever_index_result),
    )

    normalize_scores = not (
        dataset_name == "qrecc"
        and retriever_kind == "dense"
        and qrecc_mode == "run_partial"
    )
    pipeline_result = _step(
        status_rows,
        6,
        total_steps,
        "Run rewrite + retrieval",
        progress,
        lambda: _run_pipeline(
            dataloader=dataloader,
            rewriter=rewriter,
            retriever=retriever,
            top_k=int(config["top_k"]),
            max_batches=config["max_batches"],
            normalize_scores=normalize_scores,
            progress=progress,
            progress_desc=_progress_desc(config),
        ),
    )
    _tag_pipeline_result_with_budget(
        pipeline_result,
        int(config["max_input_tokens"]),
        include_history=bool(config["include_history"]),
    )

    if dataset_name == "qrecc" and retriever_kind == "dense" and qrecc_mode == "run_partial":
        partial_dump = _step(
            status_rows,
            7,
            total_steps,
            "Export QReCC partial dump",
            progress,
            lambda: _export_qrecc_partial_dump(config, pipeline_result),
        )
        _step(
            status_rows,
            8,
            total_steps,
            "Upload QReCC partial dump when SAS is available",
            progress,
            lambda: _upload_qrecc_partial_dump_if_possible(config, partial_dump),
        )
    else:
        evaluation = _step(
            status_rows,
            7,
            total_steps,
            "Evaluate retrieval",
            progress,
            lambda: _evaluate_pipeline(config, pipeline_result),
        )
        _tag_evaluation_result_with_budget(
            evaluation,
            int(config["max_input_tokens"]),
            include_history=bool(config["include_history"]),
        )
        eval_result_paths = _step(
            status_rows,
            8,
            total_steps,
            "Save eval results",
            progress,
            lambda: _save_eval_results(
                evaluation,
                Path(str(config["output_dir"])),
                save_per_query=bool(config["save_per_query"]),
            ),
        )

    return IterCQRFullPipelineResult(
        config=config,
        status=pd.DataFrame(status_rows),
        data_summary=data_summary,
        index_summary=index_summary,
        capacity_summary=capacity_summary,
        pipeline_result=pipeline_result,
        evaluation=evaluation,
        partial_dump=partial_dump,
        merge_result=merge_result,
        eval_result_paths=eval_result_paths,
    )


def _run_qrecc_dense_merge_eval(
    *,
    config: dict[str, Any],
    status_rows: list[dict[str, Any]],
    progress: bool,
) -> IterCQRFullPipelineResult:
    if config["split"] != "test":
        raise ValueError("QReCC dense merge_eval currently evaluates the QReCC test split only.")

    total_steps = 5
    resource_summary = _step(
        status_rows,
        1,
        total_steps,
        "Prepare QReCC processed resources",
        progress,
        lambda: _prepare_dataset_resources(config, progress=progress),
    )
    azure_eval = _step(
        status_rows,
        2,
        total_steps,
        "Download, merge, and evaluate QReCC partial dumps",
        progress,
        lambda: _evaluate_qrecc_dense_from_azure(config, progress=progress),
    )
    _tag_evaluation_result_with_budget(
        azure_eval.evaluation,
        int(config["max_input_tokens"]),
        include_history=bool(config["include_history"]),
    )
    eval_result_paths = _step(
        status_rows,
        3,
        total_steps,
        "Save eval results",
        progress,
        lambda: _save_eval_results(
            azure_eval.evaluation,
            Path(str(config["output_dir"])),
            save_per_query=bool(config["save_per_query"]),
        ),
    )
    _step(status_rows, 4, total_steps, "Build display summaries", progress, lambda: None)
    _step(status_rows, 5, total_steps, "Finish merge/eval run", progress, lambda: None)

    return IterCQRFullPipelineResult(
        config=config,
        status=pd.DataFrame(status_rows),
        data_summary=pd.DataFrame([{
            "dataset_name": "qrecc",
            "split": "test",
            "evaluated_queries": azure_eval.evaluated_queries,
            "total_test_gold_queries": azure_eval.total_test_gold_queries,
            "max_query_tokens": int(config["max_query_tokens"]),
            "max_history_turn_tokens": int(config["max_history_turn_tokens"]),
            "max_history_answer_tokens": int(config["max_history_answer_tokens"]),
            "max_input_tokens": int(config["max_input_tokens"]),
            "include_history": bool(config["include_history"]),
            **resource_summary,
        }]),
        index_summary=pd.DataFrame([azure_eval.merge_result.to_dict()]),
        capacity_summary=pd.DataFrame(),
        pipeline_result=None,
        evaluation=azure_eval.evaluation,
        partial_dump=None,
        merge_result=azure_eval.merge_result,
        eval_result_paths=eval_result_paths,
    )


def _run_budget_rewrite_retrieve_eval(
    *,
    config: dict[str, Any],
    status_rows: list[dict[str, Any]],
    progress: bool,
    resource_summary: dict[str, Any],
    rewriter: Any,
    retriever: Any,
    retriever_index_result: dict[str, Any],
) -> IterCQRFullPipelineResult:
    total_steps = 4
    dataset_name = str(config["dataset_name"])
    split = str(config["split"])
    retriever_kind = str(config["retriever_kind"])
    qrecc_mode = str(config["qrecc_dense_mode"])

    data = _step(
        status_rows,
        1,
        total_steps,
        "Load dataset split for budget",
        progress,
        lambda: _load_itercqr_data_for_config(config, progress=progress),
    )
    dataloader = data.dataloader(split)
    _validate_itercqr_budget_dataloader(
        dataloader,
        budget=int(config["max_input_tokens"]),
        max_query_tokens=int(config["max_query_tokens"]),
        max_history_answer_tokens=int(config["max_history_answer_tokens"]),
        include_history=bool(config["include_history"]),
    )
    data_summary = _data_summary(data, split, resource_summary)

    pipeline_result = _step(
        status_rows,
        2,
        total_steps,
        "Run rewrite + retrieval",
        progress,
        lambda: _run_pipeline(
            dataloader=dataloader,
            rewriter=rewriter,
            retriever=retriever,
            top_k=int(config["top_k"]),
            max_batches=config["max_batches"],
            normalize_scores=not (
                dataset_name == "qrecc"
                and retriever_kind == "dense"
                and qrecc_mode == "run_partial"
            ),
            progress=progress,
            progress_desc=_progress_desc(config),
        ),
    )
    _tag_pipeline_result_with_budget(
        pipeline_result,
        int(config["max_input_tokens"]),
        include_history=bool(config["include_history"]),
    )

    evaluation = None
    partial_dump = None
    eval_result_paths: dict[str, str | None] = {}
    if dataset_name == "qrecc" and retriever_kind == "dense" and qrecc_mode == "run_partial":
        partial_dump = _step(
            status_rows,
            3,
            total_steps,
            "Export QReCC partial dump",
            progress,
            lambda: _export_qrecc_partial_dump(config, pipeline_result),
        )
        _step(
            status_rows,
            4,
            total_steps,
            "Upload QReCC partial dump when SAS is available",
            progress,
            lambda: _upload_qrecc_partial_dump_if_possible(config, partial_dump),
        )
    else:
        evaluation = _step(
            status_rows,
            3,
            total_steps,
            "Evaluate retrieval",
            progress,
            lambda: _evaluate_pipeline(config, pipeline_result),
        )
        _tag_evaluation_result_with_budget(
            evaluation,
            int(config["max_input_tokens"]),
            include_history=bool(config["include_history"]),
        )
        eval_result_paths = _step(
            status_rows,
            4,
            total_steps,
            "Save eval results",
            progress,
            lambda: _save_eval_results(
                evaluation,
                Path(str(config["output_dir"])),
                save_per_query=bool(config["save_per_query"]),
            ),
        )

    return IterCQRFullPipelineResult(
        config=config,
        status=pd.DataFrame(status_rows),
        data_summary=data_summary,
        index_summary=retriever_index_result["index_summary"].copy(),
        capacity_summary=retriever_index_result["capacity_summary"].copy(),
        pipeline_result=pipeline_result,
        evaluation=evaluation,
        partial_dump=partial_dump,
        merge_result=None,
        eval_result_paths=eval_result_paths,
    )


def _load_itercqr_data_for_config(
    config: dict[str, Any],
    *,
    progress: bool,
) -> IterCQRDataPipelineResult:
    dataloader_kwargs: dict[str, Any] = {}
    if str(config["dataset_name"]) == "topiocqa":
        dataloader_kwargs["topiocqa_download_corpus"] = False
    return load_itercqr_dataloaders(
        dataset_name=str(config["dataset_name"]),
        splits=(str(config["split"]),),
        batch_size=int(config["batch_size"]),
        data_dir=Path(str(config["data_dir"])),
        progress=progress,
        max_query_tokens=int(config["max_query_tokens"]),
        max_history_turn_tokens=int(config["max_history_turn_tokens"]),
        max_history_answer_tokens=int(config["max_history_answer_tokens"]),
        max_input_tokens=int(config["max_input_tokens"]),
        include_history=bool(config["include_history"]),
        **dataloader_kwargs,
    )


def _budget_config(
    base_config: dict[str, Any],
    *,
    base_run_id: str,
    budget: int,
    output_dir: Path | str | None,
) -> dict[str, Any]:
    budget_run_id = _budget_run_id(base_run_id, budget)
    config = dict(base_config)
    config["run_id"] = budget_run_id
    config["output_dir"] = str(_default_output_dir(budget_run_id, _budget_output_dir(output_dir, budget)))
    config["max_input_tokens"] = int(budget)
    return config


def _budget_sweep_result(
    *,
    base_config: dict[str, Any],
    budgets: tuple[int, ...],
    budget_results: dict[int, IterCQRFullPipelineResult],
) -> IterCQRBudgetSweepResult:
    config = {
        "dataset_name": base_config["dataset_name"],
        "split": base_config["split"],
        "retriever_kind": base_config["retriever_kind"],
        "budgets": budgets,
        "base_run_id": base_config["run_id"],
        "bm25_k1": base_config.get("bm25_k1"),
        "bm25_b": base_config.get("bm25_b"),
        "max_query_tokens": int(base_config["max_query_tokens"]),
        "max_history_turn_tokens": int(base_config["max_history_turn_tokens"]),
        "max_history_answer_tokens": int(base_config["max_history_answer_tokens"]),
        "include_history": bool(base_config["include_history"]),
    }
    return IterCQRBudgetSweepResult(
        config=config,
        budget_results=budget_results,
        summary=_budget_sweep_summary(budget_results),
        status=_concat_budget_frames(
            result.status for result in budget_results.values()
        ),
        evaluation_aggregate=_concat_budget_frames(
            result.evaluation.aggregate
            for result in budget_results.values()
            if result.evaluation is not None
        ),
        bootstrap_summary=_concat_budget_frames(
            result.evaluation.bootstrap_summary
            for result in budget_results.values()
            if result.evaluation is not None
        ),
    )


def _preloaded_retriever_index_result(config: dict[str, Any], retriever: Any) -> dict[str, Any]:
    fallback_index_dir = (
        config["bm25_index_dir"]
        if str(config["retriever_kind"]) == "sparse"
        else config["dense_index_dir"]
    )
    index_dir = Path(str(getattr(retriever, "index_dir", fallback_index_dir))).expanduser().resolve()
    return {
        "index_dir": index_dir,
        "index_summary": pd.DataFrame([{
            "retriever_kind": str(config["retriever_kind"]),
            "index_dir": str(index_dir),
            "preloaded": True,
        }]),
        "capacity_summary": pd.DataFrame(),
        "start_shard": getattr(retriever, "start_shard", None),
        "end_shard": getattr(retriever, "end_shard", None),
    }


def _budget_run_id(base_run_id: str, budget: int) -> str:
    return f"{base_run_id}_max{int(budget)}"


def _budget_output_dir(output_dir: Path | str | None, budget: int) -> Path | None:
    if output_dir is None:
        return None
    return Path(output_dir).expanduser().resolve() / f"max{int(budget)}"


def _validate_itercqr_budget_dataloader(
    dataloader: Any,
    *,
    budget: int,
    max_query_tokens: int,
    max_history_answer_tokens: int,
    include_history: bool,
) -> None:
    dataset = getattr(dataloader, "dataset", None)
    if dataset is None:
        raise AssertionError("Dataloader has no dataset.")
    dataset_config = getattr(dataset, "config", None)
    if dataset_config is None:
        raise AssertionError("Dataloader dataset has no IterCQR tokenization config.")
    if int(dataset_config.max_input_tokens) != int(budget):
        raise AssertionError(
            f"Expected max_input_tokens={budget}, got {dataset_config.max_input_tokens}."
        )
    if int(dataset_config.max_query_tokens) != int(max_query_tokens):
        raise AssertionError(
            f"Expected max_query_tokens={max_query_tokens}, got {dataset_config.max_query_tokens}."
        )
    if bool(dataset_config.include_history) != bool(include_history):
        raise AssertionError(
            f"Expected include_history={include_history}, got {dataset_config.include_history}."
        )
    if int(dataset_config.max_history_answer_tokens) != int(max_history_answer_tokens):
        raise AssertionError(
            "Expected max_history_answer_tokens="
            f"{max_history_answer_tokens}, got {dataset_config.max_history_answer_tokens}."
        )
    if int(dataset_config.max_history_turn_tokens) != int(max_query_tokens + max_history_answer_tokens):
        raise AssertionError(
            "Expected max_history_turn_tokens="
            f"{max_query_tokens + max_history_answer_tokens}, "
            f"got {dataset_config.max_history_turn_tokens}."
        )

    expected_segment_budgets = {
        "current_query": int(max_query_tokens),
        "history_question": int(max_query_tokens),
        "history_answer": int(max_history_answer_tokens),
    }
    for index, example in enumerate(getattr(dataset, "examples", [])):
        if int(example.input_length) > int(budget):
            raise AssertionError(
                f"Example {index} exceeds budget: input_length={example.input_length}, budget={budget}."
            )
        if not include_history and int(example.input_length) > int(max_query_tokens):
            raise AssertionError(
                f"Example {index} exceeds current-query budget: "
                f"input_length={example.input_length}, max_query_tokens={max_query_tokens}."
            )
        seen_current_query = False
        for stat in getattr(example, "segment_token_stats", []):
            segment_type = str(stat.get("segment_type"))
            if not include_history and segment_type != "current_query":
                raise AssertionError(
                    f"Example {index} has history segment despite include_history=False: "
                    f"{segment_type}."
                )
            if segment_type not in expected_segment_budgets:
                continue
            expected_budget = expected_segment_budgets[segment_type]
            actual_budget = int(stat.get("budget_tokens", -1))
            if actual_budget != expected_budget:
                raise AssertionError(
                    f"Example {index} {segment_type} budget is {actual_budget}, "
                    f"expected {expected_budget}."
                )
            if segment_type == "current_query":
                seen_current_query = True
        if not seen_current_query:
            raise AssertionError(f"Example {index} has no current_query segment.")


def _tag_full_pipeline_result_with_budget(
    result: IterCQRFullPipelineResult,
    budget: int,
    *,
    include_history: bool | None = None,
) -> None:
    result.status = _with_run_context_columns(result.status, budget, include_history)
    result.data_summary = _with_run_context_columns(result.data_summary, budget, include_history)
    result.index_summary = _with_run_context_columns(result.index_summary, budget, include_history)
    result.capacity_summary = _with_run_context_columns(result.capacity_summary, budget, include_history)
    if result.pipeline_result is not None:
        _tag_pipeline_result_with_budget(
            result.pipeline_result,
            budget,
            include_history=include_history,
        )
    if result.evaluation is not None:
        _tag_evaluation_result_with_budget(
            result.evaluation,
            budget,
            include_history=include_history,
        )


def _tag_pipeline_result_with_budget(
    pipeline_result: Any,
    budget: int,
    *,
    include_history: bool | None = None,
) -> None:
    for attribute in (
        "rewrites",
        "retrievals",
        "hits",
        "rewrite_efficiency_per_batch",
        "rewrite_efficiency_summary",
    ):
        frame = getattr(pipeline_result, attribute, None)
        if isinstance(frame, pd.DataFrame):
            setattr(
                pipeline_result,
                attribute,
                _with_run_context_columns(frame, budget, include_history),
            )


def _tag_evaluation_result_with_budget(
    evaluation: Any,
    budget: int,
    *,
    include_history: bool | None = None,
) -> None:
    for attribute in ("per_query", "aggregate", "bootstrap_summary", "bootstrap_samples"):
        frame = getattr(evaluation, attribute, None)
        if isinstance(frame, pd.DataFrame):
            setattr(evaluation, attribute, _with_run_context_columns(frame, budget, include_history))


def _with_run_context_columns(
    frame: pd.DataFrame,
    budget: int,
    include_history: bool | None,
) -> pd.DataFrame:
    if frame.empty and not frame.columns.empty and "budget_tokens" in frame.columns:
        tagged = frame.copy()
    else:
        tagged = frame.copy()
        if "budget_tokens" in tagged.columns:
            tagged["budget_tokens"] = int(budget)
        else:
            tagged.insert(0, "budget_tokens", int(budget))
    if include_history is not None:
        if "include_history" in tagged.columns:
            tagged["include_history"] = bool(include_history)
        else:
            insert_at = 1 if "budget_tokens" in tagged.columns else 0
            tagged.insert(insert_at, "include_history", bool(include_history))
    return tagged


def _with_budget_column(frame: pd.DataFrame, budget: int) -> pd.DataFrame:
    tagged = frame.copy()
    if "budget_tokens" in tagged.columns:
        tagged["budget_tokens"] = int(budget)
    else:
        tagged.insert(0, "budget_tokens", int(budget))
    return tagged


def _concat_budget_frames(frames: Any) -> pd.DataFrame:
    collected = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not collected:
        return pd.DataFrame()
    return pd.concat(collected, ignore_index=True)


def _budget_sweep_summary(
    budget_results: dict[int, IterCQRFullPipelineResult],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget, result in budget_results.items():
        row: dict[str, Any] = {
            "budget_tokens": int(budget),
            "include_history": bool(result.config.get("include_history", True)),
            "run_id": result.config.get("run_id"),
            "dataset_name": result.config.get("dataset_name"),
            "split": result.config.get("split"),
            "retriever_kind": result.config.get("retriever_kind"),
            "qrecc_dense_mode": result.config.get("qrecc_dense_mode"),
        }
        if result.evaluation is not None and not result.evaluation.aggregate.empty:
            for _, metric_row in result.evaluation.aggregate.iterrows():
                row[str(metric_row["metric"])] = metric_row.get("mean")
        if result.partial_dump is not None:
            row.update({
                "partial_dump_file": str(result.partial_dump.output_file),
                "partial_dump_queries": int(result.partial_dump.query_count),
                "partial_dump_hits": int(result.partial_dump.hit_count),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def _prepare_dataset_resources(config: dict[str, Any], *, progress: bool) -> dict[str, Any]:
    dataset_name = str(config["dataset_name"])
    data_dir = Path(str(config["data_dir"]))
    sas_file = config.get("azure_sas_file")

    if dataset_name == "topiocqa":
        from .datasets.topiocqa import prepare_topiocqa_resources_for_indexing

        setup = prepare_topiocqa_resources_for_indexing(
            data_dir=data_dir,
            check_azure_ance=False,
            azure_ance_sas_file=sas_file,
            progress=progress,
            timeout=1000,
        )
        return {
            "resource_root": str(setup.resources.root),
            "raw_download_needed": bool(setup.raw_download_needed),
            "corpus_download_needed": bool(setup.corpus_download_needed),
            "bm25_index_complete": bool(getattr(setup.bm25_status, "complete", False)),
            "local_ance_index_ready": bool(setup.artifacts.ance_index_ready),
            "azure_ance_index_ready": bool(setup.azure_ance_status and setup.azure_ance_status.ready),
            "collection_required": bool(setup.corpus_download_needed),
        }

    if dataset_name == "qrecc":
        from .datasets.qrecc import resolve_qrecc_resources

        resources = resolve_qrecc_resources(data_dir)
        ready_before = _qrecc_processed_splits_ready(resources)
        sync_result = None
        ready_after = _qrecc_processed_splits_ready(resources)
        if not ready_after:
            missing = [
                str(path)
                for path in (
                    resources.processed_train_json,
                    resources.processed_test_json,
                    resources.collection_manifest_json,
                )
                if not path.exists()
            ]
            raise FileNotFoundError(
                "QReCC processed splits are missing. Prepare them with notebook 00a "
                "or place the supplied processed splits and manifest locally: "
                f"{missing}"
            )
        return {
            "resource_root": str(resources.root),
            "processed_splits_ready_before": bool(ready_before),
            "processed_splits_synced": bool(sync_result and not sync_result.skipped_existing),
            "processed_splits_ready_after": bool(ready_after),
            "collection_required": False,
        }

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def _qrecc_processed_splits_ready(resources: Any) -> bool:
    return bool(
        resources.processed_train_json.exists()
        and resources.processed_test_json.exists()
        and resources.collection_manifest_json.exists()
    )


def _prepare_retriever_index(config: dict[str, Any]) -> dict[str, Any]:
    dataset_name = str(config["dataset_name"])
    retriever_kind = str(config["retriever_kind"])
    if retriever_kind == "sparse":
        index_dir = Path(str(config["bm25_index_dir"]))
        if not index_dir.exists():
            raise FileNotFoundError(f"BM25 Lucene index does not exist: {index_dir}")
        return {
            "index_dir": index_dir,
            "index_summary": pd.DataFrame([{
                "retriever_kind": "sparse",
                "index_dir": str(index_dir),
                "exists": index_dir.exists(),
            }]),
            "capacity_summary": pd.DataFrame(),
            "start_shard": None,
            "end_shard": None,
        }

    dense_index_dir = Path(str(config["dense_index_dir"]))
    if not str(config["device"]).startswith("cuda"):
        raise RuntimeError("Dense ANCE Torch retrieval requires a CUDA device.")

    if dataset_name == "topiocqa":
        summary = _topiocqa_dense_summary_or_sync(
            dense_index_dir,
            config.get("azure_sas_file"),
        )
        capacity = _dense_capacity(summary, str(config["device"]))
        if not bool(capacity.fits):
            raise RuntimeError(f"Dense index does not fit on GPU: {capacity.__dict__}")
        return {
            "index_dir": summary.index_dir,
            "index_summary": pd.DataFrame([summary.to_dict()]),
            "capacity_summary": pd.DataFrame([capacity.__dict__]),
            "start_shard": None,
            "end_shard": None,
        }

    if dataset_name == "qrecc":
        from .datasets.qrecc import qrecc_ance_session_shard_range

        start_shard, end_shard = qrecc_ance_session_shard_range(str(config["qrecc_session_id"]))
        summary = _qrecc_dense_summary_or_sync(
            dense_index_dir,
            config.get("azure_sas_file"),
            str(config["qrecc_session_id"]),
            start_shard,
            end_shard,
        )
        capacity = _dense_capacity(summary, str(config["device"]))
        if not bool(capacity.fits):
            raise RuntimeError(f"Dense shard range does not fit on GPU: {capacity.__dict__}")
        return {
            "index_dir": summary.index_dir,
            "index_summary": pd.DataFrame([summary.to_dict()]),
            "capacity_summary": pd.DataFrame([capacity.__dict__]),
            "start_shard": start_shard,
            "end_shard": end_shard,
        }

    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def _load_retriever(config: dict[str, Any], index_result: dict[str, Any]) -> Any:
    if config["retriever_kind"] == "sparse":
        from .retrievers import load_bm25_retriever

        bm25_kwargs: dict[str, Any] = {}
        if config.get("bm25_k1") is not None:
            bm25_kwargs["k1"] = float(config["bm25_k1"])
        if config.get("bm25_b") is not None:
            bm25_kwargs["b"] = float(config["bm25_b"])
        return load_bm25_retriever(
            index_dir=index_result["index_dir"],
            retrieval_workers=int(config.get("retrieval_workers", 1)),
            **bm25_kwargs,
        )

    from .dense_ance import load_dense_ance_torch_retriever

    return load_dense_ance_torch_retriever(
        index_dir=index_result["index_dir"],
        device=str(config["device"]),
        index_dtype="float32",
        start_shard=index_result.get("start_shard"),
        end_shard=index_result.get("end_shard"),
        chunk_size=50_000,
        query_batch_size=64,
        progress=True,
        status_log=None,
    )


def _run_pipeline(
    *,
    dataloader: Any,
    rewriter: Any,
    retriever: Any,
    top_k: int,
    max_batches: int | None,
    normalize_scores: bool,
    progress: bool,
    progress_desc: str,
) -> Any:
    from .pipelines import run_itercqr_retrieval_pipeline

    return run_itercqr_retrieval_pipeline(
        dataloader=dataloader,
        rewriter=rewriter,
        retriever=retriever,
        top_k=top_k,
        max_batches=max_batches,
        include_raw=False,
        normalize_scores=normalize_scores,
        progress=progress,
        progress_desc=progress_desc,
    )


def _evaluate_pipeline(config: dict[str, Any], pipeline_result: Any) -> Any:
    from .evaluation import evaluate_retrieval_pipeline

    return evaluate_retrieval_pipeline(
        pipeline_result,
        ks=tuple(config["eval_ks"]),
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=42,
    )


def _evaluate_qrecc_dense_from_azure(config: dict[str, Any], *, progress: bool) -> Any:
    from .datasets.qrecc import evaluate_qrecc_ance_partial_run_from_azure

    return evaluate_qrecc_ance_partial_run_from_azure(
        run_id=str(config["run_id"]),
        data_dir=Path(str(config["data_dir"])),
        local_run_dir=_qrecc_partial_run_dir(str(config["run_id"])),
        sas_file=config.get("azure_sas_file"),
        session_ids=("session_1", "session_2"),
        final_top_k=int(config["top_k"]),
        ks=tuple(config["eval_ks"]),
        bootstrap_samples=int(config["bootstrap_samples"]),
        seed=42,
        progress=progress,
        download_remote=False,
    )


def _export_qrecc_partial_dump(config: dict[str, Any], pipeline_result: Any) -> Any:
    from .pipelines import export_partial_retrieval_hits

    output_file = _qrecc_partial_run_dir(str(config["run_id"])) / f"{config['qrecc_session_id']}.jsonl"
    return export_partial_retrieval_hits(
        pipeline_result,
        output_file,
        shard_session=str(config["qrecc_session_id"]),
        top_k=int(config["top_k"]),
    )


def _upload_qrecc_partial_dump_if_possible(config: dict[str, Any], partial_dump: Any) -> Any:
    return {"uploaded": False, "reason": "partial dump retained locally"}


def _save_eval_results(
    evaluation: Any,
    output_dir: Path,
    *,
    save_per_query: bool,
) -> dict[str, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = output_dir / "evaluation_aggregate.csv"
    bootstrap_path = output_dir / "bootstrap_summary.csv"
    per_query_path = output_dir / "evaluation_per_query.csv"

    evaluation.aggregate.to_csv(aggregate_path, index=False)
    evaluation.bootstrap_summary.to_csv(bootstrap_path, index=False)
    saved_per_query: str | None = None
    if save_per_query:
        _per_query_eval_frame(evaluation.per_query).to_csv(per_query_path, index=False)
        saved_per_query = str(per_query_path)

    return {
        "evaluation_aggregate": str(aggregate_path),
        "bootstrap_summary": str(bootstrap_path),
        "evaluation_per_query": saved_per_query,
    }


def _per_query_eval_frame(per_query: pd.DataFrame) -> pd.DataFrame:
    prefix_columns = [
        "budget_tokens",
        "include_history",
        "sample_id",
        "split",
        "conv_id",
        "turn_id",
        "target_rank",
    ]
    metric_columns = [
        column
        for column in per_query.columns
        if column == "MRR" or column.startswith("nDCG@") or column.startswith("R@")
    ]
    columns = [column for column in [*prefix_columns, *metric_columns] if column in per_query.columns]
    return per_query.loc[:, columns].copy()


def _topiocqa_dense_summary_or_sync(index_dir: Path, sas_file: Any) -> Any:
    from .dense_ance import summarize_dense_faiss_shards
    from .datasets.topiocqa import TOPIOCQA_CORPUS_PASSAGES
    summary = summarize_dense_faiss_shards(index_dir)
    if not summary.contiguous or summary.num_passages != TOPIOCQA_CORPUS_PASSAGES:
        raise RuntimeError("TopiOCQA local ANCE index is incomplete; finish copying/building it first.")
    return summary


def _qrecc_dense_summary_or_sync(
    index_dir: Path,
    sas_file: Any,
    session_id: str,
    start_shard: int,
    end_shard: int,
) -> Any:
    from .dense_ance import summarize_dense_faiss_shards
    from .datasets.qrecc import QRECC_ANCE_SHARD_PASSAGES, QRECC_TOTAL_PASSAGES
    summary = summarize_dense_faiss_shards(index_dir, start_shard=start_shard, end_shard=end_shard)
    if (not summary.contiguous or summary.shard_count != end_shard - start_shard + 1
            or summary.first_passage_start != start_shard * QRECC_ANCE_SHARD_PASSAGES
            or summary.last_passage_end != min((end_shard + 1) * QRECC_ANCE_SHARD_PASSAGES, QRECC_TOTAL_PASSAGES)):
        raise RuntimeError("QReCC local ANCE shard range is incomplete; finish copying/building it first.")
    return summary


def _dense_capacity(summary: Any, device: str) -> Any:
    from .dense_ance import check_dense_gpu_capacity

    return check_dense_gpu_capacity(
        summary,
        device=device,
        index_dtype="float32",
        reserve_gib=4.0,
    )


def _data_summary(
    data: IterCQRDataPipelineResult,
    split: str,
    resource_summary: dict[str, Any] | None = None,
) -> pd.DataFrame:
    loader = data.dataloader(split)
    tokenization_config = getattr(loader.dataset, "config", data.tokenization_config)
    row = {
        "dataset_name": data.dataset_name,
        "split": split,
        "resource_root": str(data.resources.root),
        "turns": len(loader.dataset),
        "batches": len(loader),
        "max_query_tokens": int(tokenization_config.max_query_tokens),
        "max_history_turn_tokens": int(tokenization_config.max_history_turn_tokens),
        "max_history_answer_tokens": int(tokenization_config.max_history_answer_tokens),
        "max_input_tokens": int(tokenization_config.max_input_tokens),
        "include_history": bool(tokenization_config.include_history),
    }
    if resource_summary:
        row.update(resource_summary)
    return pd.DataFrame([row])


def _step(
    status_rows: list[dict[str, Any]],
    step_index: int,
    total_steps: int,
    label: str,
    progress: bool,
    func: Any,
) -> Any:
    if progress:
        print(f"[{step_index}/{total_steps}] {label}", flush=True)
    started = time.perf_counter()
    try:
        value = func()
    except Exception as exc:
        status_rows.append({
            "step": step_index,
            "total_steps": total_steps,
            "label": label,
            "status": "failed",
            "seconds": time.perf_counter() - started,
            "detail": repr(exc),
        })
        raise
    status_rows.append({
        "step": step_index,
        "total_steps": total_steps,
        "label": label,
        "status": "done",
        "seconds": time.perf_counter() - started,
        "detail": None,
    })
    return value


def _default_data_dir(dataset_name: str, data_dir: Path | str | None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    return project_path("experiments", "data", dataset_name)


def _default_bm25_index_dir(dataset_name: str, index_dir: Path | str | None) -> Path:
    if index_dir is not None:
        return Path(index_dir).expanduser().resolve()
    return project_path(
        "experiments",
        "data",
        f"pyserini_bm25_lucene_{dataset_name}",
        "lucene_index",
    )


def _default_dense_index_dir(dataset_name: str, index_dir: Path | str | None) -> Path:
    if index_dir is not None:
        return Path(index_dir).expanduser().resolve()
    return project_path(
        "experiments",
        "data",
        f"pyserini_ance_faiss_{dataset_name}",
        "faiss_flat_index_full_sharded",
    )


def _default_output_dir(run_id: str, output_dir: Path | str | None) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    return project_path("experiments", "results", "full_pipeline_runs", run_id)


def _qrecc_partial_run_dir(run_id: str) -> Path:
    return project_path("experiments", "results", "qrecc", "ance_partial_runs", run_id)


def _resolve_existing_sas_file(path: Path | str | None) -> Path | None:
    # Local/public-index runs never discover credentials implicitly.
    return Path(path).expanduser().resolve() if path is not None else None


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _progress_desc(config: dict[str, Any]) -> str:
    return (
        f"{config['retriever_kind']} {config['dataset_name']} "
        f"{config['split']} IterCQR retrieval max{config['max_input_tokens']}"
    )
