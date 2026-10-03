"""The five final turn-selection candidate families for IterCQR experiments."""

from __future__ import annotations

import json
import math
import random
import re
import time
from collections.abc import Collection
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .datasets.itercqr import IterCQRDataPipelineResult, load_itercqr_dataloaders
from .evaluation import RetrievalEvaluationResult
from .itercqr_components import IterCQRRoccConfig, TokenizedExample, load_itercqr_rewriter
from .progress import progress_iter

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:  # pragma: no cover
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        pass


TURN_SELECTION_GENERATOR_FAMILIES = (
    "recent_variants",
    "position_anchor",
    "lexical_overlap",
    "spacy_query_entity_overlap",
    "spacy_recent_entity",
)
DEFAULT_MAX_CANDIDATES_PER_GENERATOR = 8
DEFAULT_CONTROL_EXHAUSTIVE_MAX_DEPTH = 8
DEFAULT_CONTROL_SAMPLES_PER_CARDINALITY = 10
DEFAULT_SPACY_MODEL = "en_core_web_sm"
SPACY_EXCLUDED_ENTITY_LABELS = frozenset({"CARDINAL", "ORDINAL"})
SPACY_EXCLUDED_ENTITY_TEXTS = frozenset({"unanswerable"})
LEXICAL_STOPWORDS = frozenset(
    """
    a about above after again against ain all am an and any are aren aren't as at
    be because been before being below between both but by can couldn couldn't d
    did didn didn't do does doesn doesn't doing don don't down during each few
    for from further had hadn hadn't has hasn hasn't have haven haven't having he
    he'd he'll her here hers herself he's him himself his how i i'd if i'll i'm
    in into is isn isn't it it'd it'll it's its itself i've just ll m ma me
    mightn mightn't more most mustn mustn't my myself needn needn't no nor not
    now o of off on once only or other our ours ourselves out over own re s same
    shan shan't she she'd she'll she's should shouldn shouldn't should've so some
    such t than that that'll the their theirs them themselves then there these
    they they'd they'll they're they've this those through to too under until up
    ve very was wasn wasn't we we'd we'll we're were weren weren't we've what
    when where which while who whom why will with won won't wouldn wouldn't y
    you you'd you'll your you're yours yourself yourselves you've
    """.split()
)
_SPACY_NLP_BY_MODEL: dict[str, Any] = {}


def is_lexical_stopword(text: str) -> bool:
    """Return whether all lexical parts belong to the embedded NLTK list."""

    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    return bool(tokens) and all(token in LEXICAL_STOPWORDS for token in tokens)


@dataclass
class TurnUnit:
    question: str
    answer: str
    topic: str | None = None

    @property
    def text(self) -> str:
        return " ".join(part for part in (self.question, self.answer) if part).strip()


@dataclass
class TurnSelectionSample:
    sample_id: str
    split: str
    conv_id: Any
    turn_id: Any
    question: str
    history_units: list[TurnUnit]
    positive_ctx_passage_ids: list[str]
    positive_ctx_texts: list[str]
    topic: str | None = None
    conversation_token_length: int = 0
    history_token_lengths: list[int] = field(default_factory=list)
    history_question_token_lengths: list[int] = field(default_factory=list)
    history_answer_token_lengths: list[int] = field(default_factory=list)
    current_question_token_length: int = 0


@dataclass(frozen=True)
class TurnSelectionCandidateConfig:
    """Configuration for the five final turn-selection generators."""

    generator_families: tuple[str, ...] = TURN_SELECTION_GENERATOR_FAMILIES
    max_candidates_per_generator: int = DEFAULT_MAX_CANDIDATES_PER_GENERATOR
    include_references: bool = True
    include_control_space: bool = False
    control_exhaustive_max_depth: int = DEFAULT_CONTROL_EXHAUSTIVE_MAX_DEPTH
    control_samples_per_cardinality: int = DEFAULT_CONTROL_SAMPLES_PER_CARDINALITY
    spacy_model: str = DEFAULT_SPACY_MODEL
    seed: int = 13


@dataclass
class CandidatePoolResult:
    candidates: pd.DataFrame
    duplicate_summary: pd.DataFrame
    unit_summary: pd.DataFrame
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class IterCQRCandidatePipelineResult:
    config: dict[str, Any]
    status: pd.DataFrame
    data_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    index_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    capacity_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    pipeline_result: Any | None = None
    evaluation: RetrievalEvaluationResult | None = None
    eval_result_paths: dict[str, str | None] = field(default_factory=dict)


def generate_turn_selection_candidates(
    *,
    data: IterCQRDataPipelineResult,
    split: str,
    config: TurnSelectionCandidateConfig | None = None,
    sample_ids: Collection[str] | None = None,
    spacy_nlp: Any | None = None,
    progress: bool = True,
) -> CandidatePoolResult:
    """Generate masks from the five final turn-selection families.

    Reference and control-space masks can be included in the same pool. They
    are tagged by ``candidate_roles`` and never counted as generator families.
    """

    resolved_config = _validate_candidate_config(config or TurnSelectionCandidateConfig())
    requested_sample_ids = None if sample_ids is None else {str(sample_id) for sample_id in sample_ids}
    samples = _turn_selection_samples(data, split)
    if requested_sample_ids is not None:
        available_sample_ids = {sample.sample_id for sample in samples}
        missing_sample_ids = sorted(requested_sample_ids - available_sample_ids)
        if missing_sample_ids:
            preview = ", ".join(missing_sample_ids[:5])
            raise ValueError(
                f"{len(missing_sample_ids)} requested sample_ids are unavailable in split {split!r}: {preview}"
            )
        samples = [sample for sample in samples if sample.sample_id in requested_sample_ids]
    _populate_sample_token_lengths(
        samples,
        data.tokenizer,
        progress=progress,
    )

    active_spacy_families = {
        "spacy_query_entity_overlap",
        "spacy_recent_entity",
    }.intersection(resolved_config.generator_families)
    if active_spacy_families and spacy_nlp is None:
        spacy_nlp = _load_spacy_nlp(resolved_config.spacy_model)
    entity_cache: dict[str, frozenset[str]] = {}

    raw_rows: list[dict[str, Any]] = []
    iterator = progress_iter(
        samples,
        total=len(samples),
        desc="generate turn candidates",
        unit="sample",
        enabled=progress,
    )
    for sample in iterator:
        if resolved_config.include_references:
            _append_reference_candidates(raw_rows, sample)

        for family in resolved_config.generator_families:
            if family == "recent_variants":
                _append_recent_variants(
                    raw_rows,
                    sample,
                    max_candidates=resolved_config.max_candidates_per_generator,
                )
            elif family == "position_anchor":
                _append_position_anchor_candidates(
                    raw_rows,
                    sample,
                    max_candidates=resolved_config.max_candidates_per_generator,
                )
            elif family == "lexical_overlap":
                _append_lexical_overlap_candidates(
                    raw_rows,
                    sample,
                    max_candidates=resolved_config.max_candidates_per_generator,
                )
            elif family in active_spacy_families:
                query_entities, turn_entities = _sample_entity_sets(
                    sample,
                    spacy_nlp,
                    entity_cache,
                )
                if family == "spacy_query_entity_overlap":
                    _append_spacy_query_entity_overlap_candidates(
                        raw_rows,
                        sample,
                        query_entities=query_entities,
                        turn_entities=turn_entities,
                        max_candidates=resolved_config.max_candidates_per_generator,
                    )
                else:
                    _append_spacy_recent_entity_candidates(
                        raw_rows,
                        sample,
                        turn_entities=turn_entities,
                        max_candidates=resolved_config.max_candidates_per_generator,
                    )
            else:  # pragma: no cover - guarded by _validate_candidate_config
                raise AssertionError(f"Unhandled generator family: {family}")

        if resolved_config.include_control_space:
            _append_control_space_candidates(
                raw_rows,
                sample,
                exhaustive_max_depth=resolved_config.control_exhaustive_max_depth,
                samples_per_cardinality=resolved_config.control_samples_per_cardinality,
                seed=resolved_config.seed,
            )

    candidates = _dedupe_candidate_rows(raw_rows)
    return CandidatePoolResult(
        candidates=candidates,
        duplicate_summary=_duplicate_summary(raw_rows, candidates),
        unit_summary=_unit_summary(data, split, samples),
        config={
            "dataset_name": data.dataset_name,
            "split": split,
            "unit_granularity": "turn",
            "generator_families": list(resolved_config.generator_families),
            "max_candidates_per_generator": resolved_config.max_candidates_per_generator,
            "include_references": resolved_config.include_references,
            "include_control_space": resolved_config.include_control_space,
            "control_exhaustive_max_depth": resolved_config.control_exhaustive_max_depth,
            "control_samples_per_cardinality": resolved_config.control_samples_per_cardinality,
            "spacy_model": resolved_config.spacy_model,
            "seed": resolved_config.seed,
            "sample_filter_size": None if requested_sample_ids is None else len(requested_sample_ids),
        },
    )


def materialize_turn_selection_candidate(
    *,
    data: IterCQRDataPipelineResult,
    split: str,
    sample_id: str,
    mask: str | Sequence[bool],
) -> dict[str, Any]:
    """Return a readable materialization for one mask without saving it."""

    samples = {sample.sample_id: sample for sample in _turn_selection_samples(data, split)}
    sample = samples[str(sample_id)]
    bits = _mask_to_bools(mask)
    selected = [
        {"index": index, "question": unit.question, "answer": unit.answer}
        for index, unit in enumerate(sample.history_units)
        if index < len(bits) and bits[index]
    ]
    removed = [
        {"index": index, "question": unit.question, "answer": unit.answer}
        for index, unit in enumerate(sample.history_units)
        if index >= len(bits) or not bits[index]
    ]
    return {
        "sample_id": sample.sample_id,
        "split": sample.split,
        "conv_id": sample.conv_id,
        "turn_id": sample.turn_id,
        "question": sample.question,
        "mask": _mask_to_string(bits),
        "history": [
            {"index": index, "question": unit.question, "answer": unit.answer}
            for index, unit in enumerate(sample.history_units)
        ],
        "selected_history": selected,
        "removed_history": removed,
    }


def run_itercqr_candidate_pipeline(
    *,
    dataset_name: str,
    split: str,
    retriever_kind: str,
    candidate_pool: CandidatePoolResult,
    run_id: str | None = None,
    data_dir: Path | str | None = None,
    bm25_index_dir: Path | str | None = None,
    dense_index_dir: Path | str | None = None,
    azure_sas_file: Path | str | None = None,
    qrecc_session_id: str = "session_1",
    qrecc_dense_mode: str = "run_partial",
    batch_size: int = 16,
    top_k: int = 100,
    eval_ks: tuple[int, ...] = (3, 10, 100),
    bootstrap_samples: int = 10_000,
    save_per_query: bool = True,
    output_dir: Path | str | None = None,
    max_candidates: int | None = None,
    max_batches: int | None = None,
    retrieval_workers: int = 1,
    streaming_eval: bool = True,
    device: str | None = None,
    progress: bool = True,
) -> IterCQRCandidatePipelineResult:
    """Run masked candidate inputs through rewriter, retriever, and eval."""

    if dataset_name == "qrecc" and retriever_kind == "dense":
        raise NotImplementedError("QReCC dense candidate evaluation is not supported in V1.")

    from .holistic import (
        _default_bm25_index_dir,
        _default_data_dir,
        _default_dense_index_dir,
        _default_device,
        _default_output_dir,
        _load_retriever,
        _prepare_dataset_resources,
        _prepare_retriever_index,
        _resolve_existing_sas_file,
        _run_pipeline,
    )

    normalized_dataset = str(dataset_name).lower()
    normalized_split = str(split).lower()
    normalized_retriever = str(retriever_kind).lower()
    normalized_retrieval_workers = int(retrieval_workers)
    if normalized_retrieval_workers < 1:
        raise ValueError("retrieval_workers must be >= 1.")
    resolved_run_id = run_id or f"{normalized_dataset}_{normalized_split}_{normalized_retriever}_candidates"
    resolved_data_dir = _default_data_dir(normalized_dataset, data_dir)
    resolved_output_dir = _default_output_dir(resolved_run_id, output_dir)
    resolved_sas_file = _resolve_existing_sas_file(azure_sas_file)
    resolved_device = device or _default_device()
    config = {
        "dataset_name": normalized_dataset,
        "split": normalized_split,
        "retriever_kind": normalized_retriever,
        "data_dir": str(resolved_data_dir),
        "bm25_index_dir": str(_default_bm25_index_dir(normalized_dataset, bm25_index_dir)),
        "dense_index_dir": str(_default_dense_index_dir(normalized_dataset, dense_index_dir)),
        "azure_sas_file": str(resolved_sas_file) if resolved_sas_file else None,
        "qrecc_session_id": qrecc_session_id,
        "qrecc_dense_mode": qrecc_dense_mode,
        "run_id": resolved_run_id,
        "output_dir": str(resolved_output_dir),
        "batch_size": int(batch_size),
        "top_k": int(top_k),
        "eval_ks": tuple(int(k) for k in eval_ks),
        "bootstrap_samples": int(bootstrap_samples),
        "save_per_query": bool(save_per_query),
        "max_batches": max_batches,
        "retrieval_workers": normalized_retrieval_workers,
        "streaming_eval": bool(streaming_eval),
        "device": resolved_device,
    }

    status_rows: list[dict[str, Any]] = []
    total_steps = 8
    _candidate_step(status_rows, 1, total_steps, "Resolve paths and config", progress, lambda: None)
    resource_summary = _candidate_step(
        status_rows,
        2,
        total_steps,
        "Prepare resources and load dataset split",
        progress,
        lambda: _prepare_dataset_resources(config, progress=progress),
    )
    data = _candidate_step(
        status_rows,
        3,
        total_steps,
        "Load candidate dataset",
        progress,
        lambda: _load_candidate_data(config, normalized_dataset, normalized_split, int(batch_size)),
    )
    candidate_frame = candidate_pool.candidates
    if max_candidates is not None:
        candidate_frame = candidate_frame.head(int(max_candidates)).copy()
    dataloader = make_candidate_dataloader(
        data=data,
        split=normalized_split,
        candidates=candidate_frame,
        batch_size=int(batch_size),
    )
    data_summary = _candidate_data_summary(data, dataloader, resource_summary)
    candidate_summary = _candidate_run_summary(candidate_frame, candidate_pool)

    rewriter = _candidate_step(
        status_rows,
        4,
        total_steps,
        "Load IterCQR rewriter",
        progress,
        lambda: load_itercqr_rewriter(device=resolved_device),
    )
    retriever_index_result = _candidate_step(
        status_rows,
        5,
        total_steps,
        "Prepare retriever index",
        progress,
        lambda: _prepare_retriever_index(config),
    )
    retriever = _candidate_step(
        status_rows,
        6,
        total_steps,
        "Load retriever",
        progress,
        lambda: _load_retriever(config, retriever_index_result),
    )
    if streaming_eval:
        stream_result = _candidate_step(
            status_rows,
            7,
            total_steps,
            "Run candidate rewrite + retrieval + streaming eval",
            progress,
            lambda: _run_candidate_streaming_eval(
                dataloader=dataloader,
                rewriter=rewriter,
                retriever=retriever,
                top_k=int(top_k),
                eval_ks=tuple(int(k) for k in eval_ks),
                output_dir=resolved_output_dir,
                save_per_query=bool(save_per_query),
                max_batches=max_batches,
                normalize_scores=True,
                progress=progress,
                progress_desc=f"{normalized_dataset} {normalized_split} candidate retrieval",
            ),
        )
        pipeline_result = None
        evaluation, eval_result_paths = _candidate_step(
            status_rows,
            8,
            total_steps,
            "Finalize candidate evaluation",
            progress,
            lambda: _finalize_streaming_candidate_eval(
                stream_result=stream_result,
                bootstrap_samples=int(bootstrap_samples),
                save_per_query=bool(save_per_query),
            ),
        )
    else:
        pipeline_result = _candidate_step(
            status_rows,
            7,
            total_steps,
            "Run candidate rewrite + retrieval",
            progress,
            lambda: _run_pipeline(
                dataloader=dataloader,
                rewriter=rewriter,
                retriever=retriever,
                top_k=int(top_k),
                max_batches=max_batches,
                normalize_scores=True,
                progress=progress,
                progress_desc=f"{normalized_dataset} {normalized_split} candidate retrieval",
            ),
        )
        evaluation, eval_result_paths = _candidate_step(
            status_rows,
            8,
            total_steps,
            "Evaluate and save candidate results",
            progress,
            lambda: _evaluate_and_save_candidate_results(
                pipeline_result=pipeline_result,
                candidates=candidate_frame,
                output_dir=resolved_output_dir,
                eval_ks=tuple(int(k) for k in eval_ks),
                bootstrap_samples=int(bootstrap_samples),
                save_per_query=bool(save_per_query),
            ),
        )
    return IterCQRCandidatePipelineResult(
        config=config,
        status=pd.DataFrame(status_rows),
        data_summary=data_summary,
        candidate_summary=candidate_summary,
        index_summary=retriever_index_result["index_summary"],
        capacity_summary=retriever_index_result["capacity_summary"],
        pipeline_result=pipeline_result,
        evaluation=evaluation,
        eval_result_paths=eval_result_paths,
    )


class CandidateMaskedTurnDataset(Dataset):
    def __init__(
        self,
        *,
        data: IterCQRDataPipelineResult,
        split: str,
        candidates: pd.DataFrame,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required for candidate dataloading.")
        self.data = data
        self.split = split
        self.candidates = candidates.reset_index(drop=True)
        self.tokenizer = data.tokenizer
        self.config: IterCQRRoccConfig = data.tokenization_config
        self.pad_token_id = int(getattr(self.tokenizer, "pad_token_id"))
        self.samples = {sample.sample_id: sample for sample in _turn_selection_samples(data, split)}

    def __len__(self) -> int:
        return len(self.candidates)

    def __getitem__(self, index: int) -> TokenizedExample:
        row = self.candidates.iloc[index].to_dict()
        sample = self.samples[str(row["sample_id"])]
        mask = _mask_to_bools(row["mask"])
        return _build_masked_example(sample, row, mask, self.tokenizer, self.config)

    def collate_fn(self, batch: list[TokenizedExample]) -> dict[str, Any]:
        pad_length = max(example.input_length for example in batch) if batch else 0
        multiple = self.config.pad_to_multiple_of
        if multiple is not None and pad_length:
            pad_length = int(math.ceil(pad_length / multiple) * multiple)
        pad_length = min(pad_length, self.config.max_input_tokens)
        input_ids = []
        attention_mask = []
        for example in batch:
            pad_count = pad_length - example.input_length
            input_ids.append(example.input_ids + [self.pad_token_id] * pad_count)
            attention_mask.append([1] * example.input_length + [0] * pad_count)
        return {
            "bt_sample_ids": [example.sample_id for example in batch],
            "bt_input_ids": torch.tensor(input_ids, dtype=torch.long),
            "bt_attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "bt_input_lengths": torch.tensor([example.input_length for example in batch], dtype=torch.long),
            "bt_questions": [example.question for example in batch],
            "bt_metadata": [example.metadata for example in batch],
        }


def make_candidate_dataloader(
    *,
    data: IterCQRDataPipelineResult,
    split: str,
    candidates: pd.DataFrame,
    batch_size: int,
) -> Any:
    if DataLoader is None:
        raise RuntimeError("PyTorch is required for candidate dataloading.")
    dataset = CandidateMaskedTurnDataset(data=data, split=split, candidates=candidates)
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=dataset.collate_fn)


def _turn_selection_samples(data: IterCQRDataPipelineResult, split: str) -> list[TurnSelectionSample]:
    frame = data.frame
    split_frame = frame[frame["split"].eq(split)].copy()
    if data.dataset_name == "topiocqa":
        return _topiocqa_turn_selection_samples(split_frame)
    if data.dataset_name == "qrecc":
        return _qrecc_turn_selection_samples(split_frame)
    raise ValueError(f"Unsupported dataset_name: {data.dataset_name}")


def _topiocqa_turn_selection_samples(frame: pd.DataFrame) -> list[TurnSelectionSample]:
    rows = frame.sort_values(["split", "conv_id", "turn_id"]).to_dict("records")
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row.get("split"), row.get("conv_id")), []).append(row)
    samples: list[TurnSelectionSample] = []
    for group in grouped.values():
        history: list[TurnUnit] = []
        for row in group:
            samples.append(
                TurnSelectionSample(
                    sample_id=_sample_id(row),
                    split=str(row.get("split")),
                    conv_id=row.get("conv_id"),
                    turn_id=row.get("turn_id"),
                    question=_clean_text(row.get("question")),
                    history_units=list(history),
                    positive_ctx_passage_ids=_positive_passage_ids(row),
                    positive_ctx_texts=_positive_ctx_texts(row),
                    topic=_topiocqa_row_topic(row),
                )
            )
            history.append(
                TurnUnit(
                    question=_clean_text(row.get("question")),
                    answer=_normalise_answer(row.get("answers")),
                    topic=_topiocqa_row_topic(row),
                )
            )
    return samples


def _validate_candidate_config(config: TurnSelectionCandidateConfig) -> TurnSelectionCandidateConfig:
    families = tuple(str(family).strip().lower() for family in config.generator_families)
    if len(families) != len(set(families)):
        raise ValueError("generator_families must not contain duplicates.")
    unsupported = sorted(set(families) - set(TURN_SELECTION_GENERATOR_FAMILIES))
    if unsupported:
        raise ValueError(
            "Only the five final turn-selection generators are supported; "
            f"unsupported families: {unsupported}"
        )
    max_candidates = int(config.max_candidates_per_generator)
    if max_candidates < 1:
        raise ValueError("max_candidates_per_generator must be at least 1.")
    exhaustive_max_depth = int(config.control_exhaustive_max_depth)
    if exhaustive_max_depth < 1:
        raise ValueError("control_exhaustive_max_depth must be at least 1.")
    samples_per_cardinality = int(config.control_samples_per_cardinality)
    if samples_per_cardinality < 1:
        raise ValueError("control_samples_per_cardinality must be at least 1.")
    spacy_model = str(config.spacy_model).strip()
    if {"spacy_query_entity_overlap", "spacy_recent_entity"}.intersection(families) and not spacy_model:
        raise ValueError("spacy_model is required when a spaCy generator is active.")
    return TurnSelectionCandidateConfig(
        generator_families=families,
        max_candidates_per_generator=max_candidates,
        include_references=bool(config.include_references),
        include_control_space=bool(config.include_control_space),
        control_exhaustive_max_depth=exhaustive_max_depth,
        control_samples_per_cardinality=samples_per_cardinality,
        spacy_model=spacy_model,
        seed=int(config.seed),
    )


def _append_reference_candidates(
    rows: list[dict[str, Any]],
    sample: TurnSelectionSample,
) -> None:
    n = len(sample.history_units)
    _append_candidate(
        rows,
        sample,
        [True] * n,
        origin="full_history",
        family="recency",
        role="reference",
        rank=1,
    )
    _append_candidate(
        rows,
        sample,
        [False] * n,
        origin="no_history",
        family="query_only",
        role="reference",
        rank=1,
    )


def _append_recent_variants(
    rows: list[dict[str, Any]],
    sample: TurnSelectionSample,
    *,
    max_candidates: int,
) -> None:
    n = len(sample.history_units)
    specs = [
        (
            f"recent_last_{size}",
            tuple(index >= n - size for index in range(n)),
        )
        for size in range(1, min(max_candidates, n) + 1)
    ]
    _append_generator_specs(rows, sample, "recent_variants", specs, max_candidates=max_candidates)


def _append_position_anchor_candidates(
    rows: list[dict[str, Any]],
    sample: TurnSelectionSample,
    *,
    max_candidates: int,
) -> None:
    n = len(sample.history_units)
    specs: list[tuple[str, tuple[bool, ...]]] = []
    for size in (1, 2, 3, 5, 8):
        if size <= n:
            specs.append(
                (
                    f"oldest_{size}",
                    tuple(index < size for index in range(n)),
                )
            )
    for recent_count in (1, 2, 4):
        if n >= recent_count + 1:
            chosen = {0, *range(n - recent_count, n)}
            specs.append(
                (
                    f"first_plus_recent_{recent_count}",
                    tuple(index in chosen for index in range(n)),
                )
            )
    _append_generator_specs(rows, sample, "position_anchor", specs, max_candidates=max_candidates)


def _append_lexical_overlap_candidates(
    rows: list[dict[str, Any]],
    sample: TurnSelectionSample,
    *,
    max_candidates: int,
) -> None:
    query_tokens = _lexical_token_set(sample.question)
    scored: list[tuple[float, int, int]] = []
    for index, unit in enumerate(sample.history_units):
        turn_tokens = _lexical_token_set(unit.text)
        overlap = len(query_tokens.intersection(turn_tokens))
        if overlap < 1:
            continue
        union_size = len(query_tokens.union(turn_tokens))
        jaccard = overlap / union_size if union_size else 0.0
        scored.append((jaccard, overlap, index))
    scored.sort(key=lambda item: (-item[0], -item[1], -item[2]))
    ranked_indices = [index for _, _, index in scored]
    specs = _ranked_prefix_specs(
        history_size=len(sample.history_units),
        ranked_indices=ranked_indices,
        max_candidates=max_candidates,
        origin_prefix="lexical_top",
    )
    _append_generator_specs(rows, sample, "lexical_overlap", specs, max_candidates=max_candidates)


def _append_spacy_query_entity_overlap_candidates(
    rows: list[dict[str, Any]],
    sample: TurnSelectionSample,
    *,
    query_entities: frozenset[str],
    turn_entities: Sequence[frozenset[str]],
    max_candidates: int,
) -> None:
    if not query_entities:
        return
    scored: list[tuple[int, int]] = []
    for index, entities in enumerate(turn_entities):
        overlap_count = len(query_entities.intersection(entities))
        if overlap_count:
            scored.append((overlap_count, index))
    scored.sort(key=lambda item: (-item[0], -item[1]))
    specs = _ranked_prefix_specs(
        history_size=len(sample.history_units),
        ranked_indices=[index for _, index in scored],
        max_candidates=max_candidates,
        origin_prefix="spacy_query_entity_overlap_top",
    )
    _append_generator_specs(
        rows,
        sample,
        "spacy_query_entity_overlap",
        specs,
        max_candidates=max_candidates,
    )


def _append_spacy_recent_entity_candidates(
    rows: list[dict[str, Any]],
    sample: TurnSelectionSample,
    *,
    turn_entities: Sequence[frozenset[str]],
    max_candidates: int,
) -> None:
    ranked_indices = [
        index
        for index in range(len(sample.history_units) - 1, -1, -1)
        if turn_entities[index]
    ]
    specs = _ranked_prefix_specs(
        history_size=len(sample.history_units),
        ranked_indices=ranked_indices,
        max_candidates=max_candidates,
        origin_prefix="spacy_recent_entity_top",
    )
    _append_generator_specs(rows, sample, "spacy_recent_entity", specs, max_candidates=max_candidates)


def _append_generator_specs(
    rows: list[dict[str, Any]],
    sample: TurnSelectionSample,
    family: str,
    specs: Sequence[tuple[str, Sequence[bool]]],
    *,
    max_candidates: int,
) -> None:
    seen_masks: set[str] = set()
    rank = 0
    for origin, mask in specs:
        bitmask = _mask_to_string(mask)
        if bitmask in seen_masks:
            continue
        seen_masks.add(bitmask)
        rank += 1
        _append_candidate(
            rows,
            sample,
            mask,
            origin=origin,
            family=family,
            role="generator",
            rank=rank,
        )
        if rank >= max_candidates:
            break


def _ranked_prefix_specs(
    *,
    history_size: int,
    ranked_indices: Sequence[int],
    max_candidates: int,
    origin_prefix: str,
) -> list[tuple[str, tuple[bool, ...]]]:
    specs = []
    for size in range(1, min(max_candidates, len(ranked_indices)) + 1):
        chosen = set(ranked_indices[:size])
        specs.append(
            (
                f"{origin_prefix}_{size}",
                tuple(index in chosen for index in range(history_size)),
            )
        )
    return specs


def _append_control_space_candidates(
    rows: list[dict[str, Any]],
    sample: TurnSelectionSample,
    *,
    exhaustive_max_depth: int,
    samples_per_cardinality: int,
    seed: int,
) -> None:
    n = len(sample.history_units)
    if n < 1:
        return
    rank = 0
    exhaustive = n <= exhaustive_max_depth
    for cardinality in range(1, n + 1):
        if exhaustive:
            selected_combinations = list(combinations(range(n), cardinality))
            mode = "exhaustive"
        else:
            selected_combinations = _sample_control_combinations(
                n=n,
                cardinality=cardinality,
                sample_id=sample.sample_id,
                seed=seed,
                max_samples=samples_per_cardinality,
            )
            mode = "cardinality_stratified"
        for local_rank, selected in enumerate(selected_combinations, start=1):
            rank += 1
            chosen = set(selected)
            _append_candidate(
                rows,
                sample,
                tuple(index in chosen for index in range(n)),
                origin=f"{mode}_c{cardinality}_s{local_rank}",
                family="mask_space",
                role="control",
                rank=rank,
            )


def _sample_control_combinations(
    *,
    n: int,
    cardinality: int,
    sample_id: str,
    seed: int,
    max_samples: int,
) -> list[tuple[int, ...]]:
    total = math.comb(n, cardinality)
    if total <= max_samples:
        return list(combinations(range(n), cardinality))
    rng = random.Random(f"{seed}:{sample_id}:mask_space:{cardinality}")
    selected: set[tuple[int, ...]] = set()
    max_attempts = max_samples * 100
    for _ in range(max_attempts):
        selected.add(tuple(sorted(rng.sample(range(n), cardinality))))
        if len(selected) >= max_samples:
            break
    if len(selected) != max_samples:
        raise RuntimeError(
            f"Could not sample {max_samples} unique masks for {sample_id} "
            f"at cardinality {cardinality}."
        )
    return sorted(selected)


def _lexical_token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(text).lower())
        if len(token) > 1 and token not in LEXICAL_STOPWORDS
    }


def _load_spacy_nlp(model_name: str) -> Any:
    cached = _SPACY_NLP_BY_MODEL.get(model_name)
    if cached is not None:
        return cached
    try:
        import spacy
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The final candidate API requires spaCy for its two entity generators."
        ) from exc
    try:
        nlp = spacy.load(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load spaCy model {model_name!r}. "
            f"Install it with: python -m spacy download {model_name}"
        ) from exc
    _SPACY_NLP_BY_MODEL[model_name] = nlp
    return nlp


def _sample_entity_sets(
    sample: TurnSelectionSample,
    nlp: Any,
    cache: dict[str, frozenset[str]],
) -> tuple[frozenset[str], tuple[frozenset[str], ...]]:
    query_entities = _spacy_entity_set(nlp, sample.question, cache)
    turn_entities = tuple(
        _spacy_entity_set(nlp, unit.text, cache)
        for unit in sample.history_units
    )
    return query_entities, turn_entities


def _spacy_entity_set(
    nlp: Any,
    text: str,
    cache: dict[str, frozenset[str]],
) -> frozenset[str]:
    normalized_text = _clean_text(text)
    if normalized_text in cache:
        return cache[normalized_text]
    doc = nlp(normalized_text)
    entities = frozenset(
        normalized_entity
        for entity in getattr(doc, "ents", ())
        if str(getattr(entity, "label_", "")).upper() not in SPACY_EXCLUDED_ENTITY_LABELS
        and (normalized_entity := _normalize_entity(getattr(entity, "text", "")))
        and normalized_entity not in SPACY_EXCLUDED_ENTITY_TEXTS
    )
    cache[normalized_text] = entities
    return entities


def _normalize_entity(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean_text(value).lower()).strip()


def _qrecc_turn_selection_samples(frame: pd.DataFrame) -> list[TurnSelectionSample]:
    rows = frame.sort_values(["split", "conv_id", "turn_id"]).to_dict("records")
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row.get("split"), row.get("conv_id")), []).append(row)
    samples: list[TurnSelectionSample] = []
    for group in grouped.values():
        turn_id_to_utterance: dict[int, str] = {}
        for row in group:
            turn_id = _coerce_int(row.get("turn_id"))
            current = _qrecc_current_utterance(row)
            if turn_id is not None:
                turn_id_to_utterance[turn_id] = current
            context = row.get("context") if isinstance(row.get("context"), (list, tuple)) else []
            units: list[TurnUnit] = []
            for index in range(0, len(context), 2):
                history_turn_id = int(index / 2) + 1
                question = turn_id_to_utterance.get(history_turn_id, _clean_text(context[index]))
                answer = _clean_text(context[index + 1]) if index + 1 < len(context) else ""
                units.append(TurnUnit(question=question, answer=answer, topic=None))
            samples.append(
                TurnSelectionSample(
                    sample_id=_sample_id(row),
                    split=str(row.get("split")),
                    conv_id=row.get("conv_id"),
                    turn_id=row.get("turn_id"),
                    question=current,
                    history_units=units,
                    positive_ctx_passage_ids=_positive_passage_ids(row),
                    positive_ctx_texts=[],
                    topic=None,
                )
            )
    return samples


def qrecc_turn_selection_samples(
    frame: pd.DataFrame,
) -> list[TurnSelectionSample]:
    """Expose the established QReCC ``Context`` to Q/A-turn mapping."""

    return _qrecc_turn_selection_samples(frame)


def _append_candidate(
    rows: list[dict[str, Any]],
    sample: TurnSelectionSample,
    mask: Sequence[bool],
    *,
    origin: str,
    family: str,
    role: str,
    rank: int,
) -> None:
    mask_bits = tuple(bool(value) for value in mask)
    if len(mask_bits) != len(sample.history_units):
        raise ValueError(
            f"Mask length {len(mask_bits)} does not match history depth "
            f"{len(sample.history_units)} for {sample.sample_id}."
        )
    rows.append(
        {
            "sample_id": sample.sample_id,
            "split": sample.split,
            "conv_id": sample.conv_id,
            "turn_id": sample.turn_id,
            "history_turn_count": len(sample.history_units),
            "mask": _mask_to_string(mask_bits),
            "candidate_role": str(role),
            "generator_origin": origin,
            "candidate_family": family,
            "candidate_rank": int(rank),
            **_candidate_length_features(sample),
        }
    )


def _dedupe_candidate_rows(raw_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not raw_rows:
        return pd.DataFrame()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        grouped.setdefault((str(row["sample_id"]), str(row["mask"])), []).append(row)
    rows: list[dict[str, Any]] = []
    for _, duplicates in grouped.items():
        first = dict(duplicates[0])
        candidate_origins = sorted({str(row["generator_origin"]) for row in duplicates})
        generator_origins = sorted(
            {
                str(row["generator_origin"])
                for row in duplicates
                if str(row["candidate_role"]) == "generator"
            }
        )
        roles = sorted({str(row["candidate_role"]) for row in duplicates})
        families = sorted({str(row["candidate_family"]) for row in duplicates})
        provenance = sorted(
            (
                {
                    "role": str(row["candidate_role"]),
                    "family": str(row["candidate_family"]),
                    "origin": str(row["generator_origin"]),
                    "rank": int(row["candidate_rank"]),
                }
                for row in duplicates
            ),
            key=lambda item: (item["role"], item["family"], item["rank"], item["origin"]),
        )
        first.pop("generator_origin", None)
        first.pop("candidate_role", None)
        first.pop("candidate_rank", None)
        first["candidate_roles"] = roles
        first["candidate_origins"] = candidate_origins
        first["candidate_origin_count"] = len(candidate_origins)
        first["generator_origins"] = generator_origins
        first["generator_origin_count"] = len(generator_origins)
        first["candidate_family"] = families
        first["candidate_provenance"] = provenance
        first["budget_family"] = next(
            (role for role in ("reference", "generator", "control") if role in roles),
            "other",
        )
        first["duplicate_count"] = len(duplicates) - 1
        mask_bits = _mask_to_bools(first["mask"])
        first["mask_true_count"] = sum(1 for bit in mask_bits if bit)
        first["mask_false_count"] = sum(1 for bit in mask_bits if not bit)
        first["mask_ratio"] = (
            first["mask_true_count"] / len(mask_bits) if mask_bits else 0.0
        )
        rows.append(first)
    rows = sorted(rows, key=lambda row: (str(row["sample_id"]), str(row["mask"])))
    for index, row in enumerate(rows, start=1):
        row["candidate_id"] = f"{row['sample_id']}::cand_{index:06d}"
    columns = [
        "candidate_id",
        "sample_id",
        "split",
        "conv_id",
        "turn_id",
        "history_turn_count",
        "mask",
        "mask_true_count",
        "mask_false_count",
        "mask_ratio",
        "candidate_roles",
        "candidate_origins",
        "candidate_origin_count",
        "generator_origins",
        "generator_origin_count",
        "candidate_family",
        "candidate_provenance",
        "budget_family",
        "duplicate_count",
        "conversation_token_length",
        "history_token_lengths",
        "history_question_token_lengths",
        "history_answer_token_lengths",
        "current_question_token_length",
    ]
    return pd.DataFrame(rows).loc[:, columns]


def _duplicate_summary(
    raw_rows: list[dict[str, Any]],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"metric": "raw_candidates", "value": len(raw_rows)},
        {"metric": "unique_candidates", "value": len(candidates)},
        {"metric": "duplicate_candidates", "value": max(0, len(raw_rows) - len(candidates))},
    ]
    for role in ("generator", "reference", "control"):
        raw_count = sum(str(row["candidate_role"]) == role for row in raw_rows)
        unique_count = (
            int(candidates["candidate_roles"].map(lambda roles: role in roles).sum())
            if not candidates.empty
            else 0
        )
        rows.append({"metric": f"raw_role:{role}", "value": raw_count})
        rows.append({"metric": f"unique_role:{role}", "value": unique_count})
    for family in TURN_SELECTION_GENERATOR_FAMILIES:
        raw_count = sum(
            str(row["candidate_role"]) == "generator"
            and str(row["candidate_family"]) == family
            for row in raw_rows
        )
        unique_count = (
            int(candidates["candidate_family"].map(lambda families: family in families).sum())
            if not candidates.empty
            else 0
        )
        rows.append({"metric": f"raw_generator:{family}", "value": raw_count})
        rows.append({"metric": f"unique_generator:{family}", "value": unique_count})
    return pd.DataFrame(rows)


def _unit_summary(
    data: IterCQRDataPipelineResult,
    split: str,
    samples: list[TurnSelectionSample],
) -> pd.DataFrame:
    rows = []
    for sample in samples:
        rows.append(
            {
                "dataset_name": data.dataset_name,
                "split": split,
                "sample_id": sample.sample_id,
                "conv_id": sample.conv_id,
                "turn_id": sample.turn_id,
                "history_turn_count": len(sample.history_units),
                **_candidate_length_features(sample),
            }
        )
    return pd.DataFrame(rows)


def _populate_sample_token_lengths(
    samples: list[TurnSelectionSample],
    tokenizer: Any,
    *,
    progress: bool,
) -> None:
    iterator = progress_iter(
        samples,
        total=len(samples),
        desc="measure candidate token lengths",
        unit="sample",
        enabled=progress,
    )
    for sample in iterator:
        question_lengths = [_tokenizer_token_count(tokenizer, unit.question) for unit in sample.history_units]
        answer_lengths = [_tokenizer_token_count(tokenizer, unit.answer) for unit in sample.history_units]
        turn_lengths = [q + a for q, a in zip(question_lengths, answer_lengths, strict=True)]
        current_length = _tokenizer_token_count(tokenizer, sample.question)
        sample.history_question_token_lengths = question_lengths
        sample.history_answer_token_lengths = answer_lengths
        sample.history_token_lengths = turn_lengths
        sample.current_question_token_length = int(current_length)
        sample.conversation_token_length = int(current_length + sum(turn_lengths))


def _candidate_length_features(sample: TurnSelectionSample) -> dict[str, Any]:
    if sample.current_question_token_length or sample.history_token_lengths or not sample.history_units:
        return {
            "conversation_token_length": int(sample.conversation_token_length),
            "history_token_lengths": list(sample.history_token_lengths),
            "history_question_token_lengths": list(sample.history_question_token_lengths),
            "history_answer_token_lengths": list(sample.history_answer_token_lengths),
            "current_question_token_length": int(sample.current_question_token_length),
        }
    question_lengths = [_rough_token_count(unit.question) for unit in sample.history_units]
    answer_lengths = [_rough_token_count(unit.answer) for unit in sample.history_units]
    turn_lengths = [q + a for q, a in zip(question_lengths, answer_lengths, strict=True)]
    current_length = _rough_token_count(sample.question)
    return {
        "conversation_token_length": int(current_length + sum(turn_lengths)),
        "history_token_lengths": turn_lengths,
        "history_question_token_lengths": question_lengths,
        "history_answer_token_lengths": answer_lengths,
        "current_question_token_length": int(current_length),
    }


def _build_masked_example(
    sample: TurnSelectionSample,
    candidate: dict[str, Any],
    mask: list[bool],
    tokenizer: Any,
    config: IterCQRRoccConfig,
) -> TokenizedExample:
    current_query = f"question: {sample.question}" if config.use_prefix else sample.question
    raw_query_ids = _encode_untruncated(tokenizer, current_query)
    query_ids = _encode_segment(tokenizer, current_query, config.max_query_tokens)
    raw_input_ids = list(raw_query_ids)
    input_ids = list(query_ids)
    segment_stats = [
        _segment_stat("current_query", config.max_query_tokens, len(raw_query_ids), len(query_ids), len(query_ids))
    ]
    selected_units = [
        unit
        for index, unit in enumerate(sample.history_units)
        if index < len(mask) and mask[index]
    ]
    first_context = config.use_prefix
    for unit in reversed(selected_units):
        answer_text = unit.answer
        if config.use_prefix and first_context:
            answer_text = f"context: {answer_text}" if answer_text else "context:"
            first_context = False
        for segment_type, text, budget in (
            ("history_answer", answer_text, config.max_history_answer_tokens),
            ("history_question", unit.question, config.max_query_tokens),
        ):
            raw_ids = _encode_untruncated(tokenizer, text)
            bucket_ids = _encode_segment(tokenizer, text, budget)
            raw_input_ids.extend(raw_ids)
            remaining = config.max_input_tokens - len(input_ids)
            if remaining <= 0:
                included = []
            elif len(bucket_ids) > remaining:
                included = _truncate_keep_final_token(bucket_ids, remaining)
            else:
                included = bucket_ids
            input_ids.extend(included)
            segment_stats.append(_segment_stat(segment_type, budget, len(raw_ids), len(bucket_ids), len(included)))
    metadata = {
        "split": sample.split,
        "conv_id": sample.conv_id,
        "turn_id": sample.turn_id,
        "positive_ctx_passage_ids": sample.positive_ctx_passage_ids,
        "candidate_id": candidate["candidate_id"],
        "original_sample_id": sample.sample_id,
        "mask": candidate["mask"],
        "candidate_roles": candidate["candidate_roles"],
        "candidate_origins": candidate["candidate_origins"],
        "candidate_origin_count": candidate["candidate_origin_count"],
        "candidate_family": candidate["candidate_family"],
        "candidate_provenance": candidate["candidate_provenance"],
        "generator_origins": candidate["generator_origins"],
        "generator_origin_count": candidate["generator_origin_count"],
        "budget_family": candidate["budget_family"],
        "duplicate_count": candidate["duplicate_count"],
        "history_turn_count": candidate["history_turn_count"],
        "mask_true_count": candidate["mask_true_count"],
        "mask_false_count": candidate["mask_false_count"],
        "mask_ratio": candidate["mask_ratio"],
        "conversation_token_length": candidate["conversation_token_length"],
        "history_token_lengths": candidate["history_token_lengths"],
        "history_question_token_lengths": candidate["history_question_token_lengths"],
        "history_answer_token_lengths": candidate["history_answer_token_lengths"],
        "current_question_token_length": candidate["current_question_token_length"],
    }
    for column in _optional_candidate_eval_metadata_columns():
        if column in candidate:
            metadata[column] = candidate.get(column)
    return TokenizedExample(
        sample_id=str(candidate["candidate_id"]),
        input_ids=input_ids,
        input_length=len(input_ids),
        raw_input_length=len(raw_input_ids),
        truncated_tokens=max(0, len(raw_input_ids) - len(input_ids)),
        was_truncated=len(raw_input_ids) > len(input_ids),
        question=sample.question,
        metadata=metadata,
        segment_token_stats=segment_stats,
    )


def _run_candidate_streaming_eval(
    *,
    dataloader: Any,
    rewriter: Any,
    retriever: Any,
    top_k: int,
    eval_ks: tuple[int, ...],
    output_dir: Path,
    save_per_query: bool,
    max_batches: int | None,
    normalize_scores: bool,
    progress: bool,
    progress_desc: str,
    flush_rows: int = 5000,
) -> dict[str, Any]:
    from .evaluation import retrieval_metrics
    from .pipelines import _rewrite_batch_with_efficiency_metrics, _search_retriever_batch
    from .retrievers import normalize_retrieval_scores

    output_dir.mkdir(parents=True, exist_ok=True)
    per_query_path = output_dir / "evaluation_per_query.csv"
    temp_per_query_path = output_dir / "_evaluation_per_query_stream_tmp.csv"
    write_path = per_query_path if save_per_query else temp_per_query_path
    if write_path.exists():
        write_path.unlink()

    row_buffer: list[dict[str, Any]] = []
    row_count = 0
    header_written = False

    def flush() -> None:
        nonlocal row_count, header_written
        if not row_buffer:
            return
        frame = pd.DataFrame(row_buffer)
        for column in (
            "candidate_roles",
            "candidate_origins",
            "candidate_family",
            "candidate_provenance",
            "generator_origins",
            "history_token_lengths",
            "history_question_token_lengths",
            "history_answer_token_lengths",
        ):
            if column in frame.columns:
                frame[column] = frame[column].apply(_json_dumps_cell)
        frame = frame.loc[:, [column for column in _candidate_per_query_columns(frame) if column in frame.columns]]
        frame.to_csv(write_path, mode="a", header=not header_written, index=False)
        row_count += int(len(frame))
        header_written = True
        row_buffer.clear()

    try:
        total_batches = len(dataloader)
    except TypeError:
        total_batches = None
    if max_batches is not None and total_batches is not None:
        total_batches = min(total_batches, int(max_batches))
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
        rewrites, _ = _rewrite_batch_with_efficiency_metrics(
            rewriter,
            batch,
            batch_index=batch_index,
        )
        hits_batch = _search_retriever_batch(
            retriever,
            rewrites,
            top_k=top_k,
            include_raw=False,
        )
        for row_index, (rewrite, hits) in enumerate(zip(rewrites, hits_batch, strict=True)):
            sample_id = str(batch["bt_sample_ids"][row_index])
            metadata = batch["bt_metadata"][row_index]
            relevant_docids = [
                str(docid)
                for docid in metadata.get("positive_ctx_passage_ids", [])
            ]
            if normalize_scores:
                hits = normalize_retrieval_scores(hits, method="minmax")
            metrics = retrieval_metrics(hits, relevant_docids, ks=eval_ks)
            row_buffer.append(
                {
                    "sample_id": sample_id,
                    "candidate_id": metadata.get("candidate_id", sample_id),
                    "original_sample_id": metadata.get("original_sample_id"),
                    "split": metadata.get("split"),
                    "conv_id": metadata.get("conv_id"),
                    "turn_id": metadata.get("turn_id"),
                    **metrics,
                    "mask": metadata.get("mask"),
                    "candidate_roles": metadata.get("candidate_roles"),
                    "candidate_origins": metadata.get("candidate_origins"),
                    "candidate_origin_count": metadata.get("candidate_origin_count"),
                    "candidate_family": metadata.get("candidate_family"),
                    "candidate_provenance": metadata.get("candidate_provenance"),
                    "generator_origins": metadata.get("generator_origins"),
                    "generator_origin_count": metadata.get("generator_origin_count"),
                    "budget_family": metadata.get("budget_family"),
                    "duplicate_count": metadata.get("duplicate_count"),
                    "history_turn_count": metadata.get("history_turn_count"),
                    "mask_true_count": metadata.get("mask_true_count"),
                    "mask_false_count": metadata.get("mask_false_count"),
                    "mask_ratio": metadata.get("mask_ratio"),
                    "conversation_token_length": metadata.get("conversation_token_length"),
                    "history_token_lengths": metadata.get("history_token_lengths"),
                    "history_question_token_lengths": metadata.get("history_question_token_lengths"),
                    "history_answer_token_lengths": metadata.get("history_answer_token_lengths"),
                    "current_question_token_length": metadata.get("current_question_token_length"),
                    **{
                        column: metadata.get(column)
                        for column in _optional_candidate_eval_metadata_columns()
                        if column in metadata
                    },
                }
            )
        if len(row_buffer) >= int(flush_rows):
            flush()
    flush()
    if not header_written:
        empty_frame = pd.DataFrame(columns=_candidate_per_query_columns(pd.DataFrame()))
        empty_frame.to_csv(write_path, index=False)
    return {
        "output_dir": output_dir,
        "per_query_path": write_path,
        "saved_per_query_path": per_query_path if save_per_query else None,
        "temp_per_query_path": temp_per_query_path if not save_per_query else None,
        "aggregate_path": output_dir / "evaluation_aggregate.csv",
        "bootstrap_path": output_dir / "bootstrap_summary.csv",
        "row_count": int(row_count),
    }


def _finalize_streaming_candidate_eval(
    *,
    stream_result: dict[str, Any],
    bootstrap_samples: int,
    save_per_query: bool,
) -> tuple[RetrievalEvaluationResult, dict[str, str | None]]:
    from .evaluation import aggregate_metric_frame, bootstrap_metric_summary, metric_columns_from_frame

    per_query_path = Path(stream_result["per_query_path"])
    aggregate_path = Path(stream_result["aggregate_path"])
    bootstrap_path = Path(stream_result["bootstrap_path"])
    header = pd.read_csv(per_query_path, nrows=0)
    metric_columns = metric_columns_from_frame(header)
    metrics_frame = pd.read_csv(per_query_path, usecols=metric_columns) if metric_columns else pd.DataFrame()
    aggregate = aggregate_metric_frame(metrics_frame, metric_columns=metric_columns)
    if int(bootstrap_samples) > 0 and not metrics_frame.empty and metric_columns:
        bootstrap_summary, bootstrap_draws = bootstrap_metric_summary(
            metrics_frame,
            metric_columns=metric_columns,
            n_samples=int(bootstrap_samples),
            seed=42,
        )
    else:
        bootstrap_summary = pd.DataFrame(
            columns=["metric", "bootstrap_samples", "confidence", "mean", "variance", "ci95_low", "ci95_high"]
        )
        bootstrap_draws = pd.DataFrame()
    aggregate.to_csv(aggregate_path, index=False)
    bootstrap_summary.to_csv(bootstrap_path, index=False)
    temp_path = stream_result.get("temp_per_query_path")
    if temp_path:
        Path(temp_path).unlink(missing_ok=True)
    evaluation = RetrievalEvaluationResult(
        per_query=pd.DataFrame(),
        aggregate=aggregate,
        bootstrap_summary=bootstrap_summary,
        bootstrap_samples=bootstrap_draws,
    )
    return evaluation, {
        "evaluation_aggregate": str(aggregate_path),
        "bootstrap_summary": str(bootstrap_path),
        "evaluation_per_query": str(stream_result["saved_per_query_path"]) if save_per_query else None,
    }


def _evaluate_and_save_candidate_results(
    *,
    pipeline_result: Any,
    candidates: pd.DataFrame,
    output_dir: Path,
    eval_ks: tuple[int, ...],
    bootstrap_samples: int,
    save_per_query: bool,
) -> tuple[RetrievalEvaluationResult, dict[str, str | None]]:
    from .evaluation import evaluate_retrieval_pipeline

    evaluation = evaluate_retrieval_pipeline(
        pipeline_result,
        ks=eval_ks,
        bootstrap_samples=bootstrap_samples,
        seed=42,
    )
    candidate_meta = _candidate_eval_metadata(candidates)
    evaluation.per_query = evaluation.per_query.merge(
        candidate_meta,
        left_on="sample_id",
        right_on="candidate_id",
        how="left",
    )
    paths = _save_candidate_eval_results(evaluation, output_dir, save_per_query=save_per_query)
    return evaluation, paths


def _candidate_eval_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "sample_id",
        "mask",
        "candidate_roles",
        "candidate_origins",
        "candidate_origin_count",
        "candidate_family",
        "candidate_provenance",
        "generator_origins",
        "generator_origin_count",
        "budget_family",
        "duplicate_count",
        "history_turn_count",
        "mask_true_count",
        "mask_false_count",
        "mask_ratio",
        "conversation_token_length",
        "history_token_lengths",
        "history_question_token_lengths",
        "history_answer_token_lengths",
        "current_question_token_length",
    ]
    columns.extend([column for column in _optional_candidate_eval_metadata_columns() if column in candidates.columns])
    meta = candidates.loc[:, columns].copy()
    meta = meta.rename(columns={"sample_id": "original_sample_id"})
    return meta


def _save_candidate_eval_results(
    evaluation: RetrievalEvaluationResult,
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
    saved_per_query = None
    if save_per_query:
        frame = evaluation.per_query.copy()
        for column in (
            "candidate_roles",
            "candidate_origins",
            "candidate_family",
            "candidate_provenance",
            "generator_origins",
            "history_token_lengths",
            "history_question_token_lengths",
            "history_answer_token_lengths",
        ):
            if column in frame.columns:
                frame[column] = frame[column].apply(_json_dumps_cell)
        frame = frame.loc[:, [column for column in _candidate_per_query_columns(frame) if column in frame.columns]]
        frame.to_csv(per_query_path, index=False)
        saved_per_query = str(per_query_path)
    return {
        "evaluation_aggregate": str(aggregate_path),
        "bootstrap_summary": str(bootstrap_path),
        "evaluation_per_query": saved_per_query,
    }


def _json_dumps_cell(value: Any) -> str:
    return json.dumps(_jsonable_cell(value), ensure_ascii=False)


def _jsonable_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return None if pd.isna(value) else value
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return _jsonable_cell(value.tolist())
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable_cell(value.item())
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(key): _jsonable_cell(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_cell(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _candidate_per_query_columns(frame: pd.DataFrame) -> list[str]:
    metric_columns = [
        column
        for column in frame.columns
        if column == "MRR" or column.startswith("nDCG@") or column.startswith("R@")
    ]
    return [
        "sample_id",
        "candidate_id",
        "original_sample_id",
        "split",
        "conv_id",
        "turn_id",
        "target_rank",
        *metric_columns,
        "mask",
        "candidate_roles",
        "candidate_origins",
        "candidate_origin_count",
        "candidate_family",
        "candidate_provenance",
        "generator_origins",
        "generator_origin_count",
        "budget_family",
        "duplicate_count",
        "history_turn_count",
        "mask_true_count",
        "mask_false_count",
        "mask_ratio",
        "conversation_token_length",
        "history_token_lengths",
        "history_question_token_lengths",
        "history_answer_token_lengths",
        "current_question_token_length",
        *_optional_candidate_eval_metadata_columns(),
    ]


def _optional_candidate_eval_metadata_columns() -> list[str]:
    return [
        "proposal_budget",
        "proposal_strategy",
        "proposal_threshold",
        "proposal_variant",
        "budget_fair_role",
        "materialized_input_hash",
        "materialized_input_length",
        "materialized_raw_input_length",
        "materialized_truncated_tokens",
        "materialized_was_truncated",
    ]


def _load_candidate_data(
    config: dict[str, Any],
    dataset_name: str,
    split: str,
    batch_size: int,
) -> IterCQRDataPipelineResult:
    kwargs: dict[str, Any] = {}
    if dataset_name == "topiocqa":
        kwargs["topiocqa_download_corpus"] = False
    return load_itercqr_dataloaders(
        dataset_name=dataset_name,
        splits=(split,),
        data_dir=Path(str(config["data_dir"])),
        batch_size=batch_size,
        progress=False,
        **kwargs,
    )


def _candidate_data_summary(
    data: IterCQRDataPipelineResult,
    dataloader: Any,
    resource_summary: dict[str, Any],
) -> pd.DataFrame:
    row = {
        "dataset_name": data.dataset_name,
        "split": next(iter(data.dataloaders.keys())),
        "resource_root": str(data.resources.root),
        "candidate_examples": len(dataloader.dataset),
        "candidate_batches": len(dataloader),
    }
    row.update(resource_summary)
    return pd.DataFrame([row])


def _candidate_run_summary(candidates: pd.DataFrame, pool: CandidatePoolResult) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame([{"candidate_count": 0}])
    duplicate_summary = pool.duplicate_summary
    raw_candidates = len(candidates)
    duplicate_candidates = 0
    if (
        isinstance(duplicate_summary, pd.DataFrame)
        and {"metric", "value"}.issubset(duplicate_summary.columns)
        and not duplicate_summary.empty
    ):
        raw_match = duplicate_summary.loc[duplicate_summary["metric"].eq("raw_candidates"), "value"]
        duplicate_match = duplicate_summary.loc[duplicate_summary["metric"].eq("duplicate_candidates"), "value"]
        if not raw_match.empty:
            raw_candidates = int(raw_match.iloc[0])
        if not duplicate_match.empty:
            duplicate_candidates = int(duplicate_match.iloc[0])
    return pd.DataFrame(
        [
            {
                "candidate_count": int(len(candidates)),
                "original_sample_count": int(candidates["sample_id"].nunique()),
                "mean_candidates_per_sample": float(candidates.groupby("sample_id").size().mean()),
                "raw_candidates": int(raw_candidates),
                "duplicate_candidates": int(duplicate_candidates),
            }
        ]
    )


def _candidate_step(
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
        status_rows.append(
            {
                "step": step_index,
                "total_steps": total_steps,
                "label": label,
                "status": "failed",
                "seconds": time.perf_counter() - started,
                "detail": repr(exc),
            }
        )
        raise
    status_rows.append(
        {
            "step": step_index,
            "total_steps": total_steps,
            "label": label,
            "status": "done",
            "seconds": time.perf_counter() - started,
            "detail": None,
        }
    )
    return value


def _sample_id(row: dict[str, Any]) -> str:
    if row.get("id") is not None:
        return str(row["id"])
    return f"{row.get('split', 'unknown')}:{row.get('conv_id')}:{row.get('turn_id')}"


def _normalise_answer(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _clean_text(item)
            if text:
                return text
        return ""
    return _clean_text(value)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except TypeError:
        pass
    return str(value).strip()


def _positive_passage_ids(row: dict[str, Any]) -> list[str]:
    value = row.get("positive_ctx_passage_ids")
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    contexts = row.get("positive_ctxs") or []
    ids = []
    for ctx in contexts:
        passage_id = ctx.get("passage_id") if isinstance(ctx, dict) else None
        if passage_id is not None:
            ids.append(str(passage_id))
    return ids


def _positive_ctx_texts(row: dict[str, Any]) -> list[str]:
    texts = []
    for ctx in row.get("positive_ctxs") or []:
        if isinstance(ctx, dict):
            text = _clean_text(ctx.get("text"))
            if text:
                texts.append(text)
    return texts


def _topiocqa_row_topic(row: dict[str, Any]) -> str | None:
    for ctx in row.get("positive_ctxs") or []:
        title = ctx.get("title") if isinstance(ctx, dict) else None
        if title:
            return str(title).split(" [SEP] ", 1)[0]
    return None


def _qrecc_current_utterance(row: dict[str, Any]) -> str:
    if _coerce_int(row.get("turn_id")) == 1:
        rewrite = _clean_text(row.get("rewrite"))
        if rewrite:
            return rewrite
    return _clean_text(row.get("question"))


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mask_to_string(mask: str | Sequence[bool]) -> str:
    if isinstance(mask, str):
        return mask
    if not mask:
        return "[]"
    return "".join("1" if bool(item) else "0" for item in mask)


def _mask_to_bools(mask: str | Sequence[bool]) -> list[bool]:
    if isinstance(mask, str):
        if mask == "[]":
            return []
        return [char == "1" for char in mask]
    return [bool(item) for item in mask]


def _rough_token_count(text: str) -> int:
    return len(_tokenize(text))


def _tokenizer_token_count(tokenizer: Any, text: str) -> int:
    return len(_encode_untruncated(tokenizer, text))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", str(text).lower())


def _encode_segment(tokenizer: Any, text: str, max_length: int) -> list[int]:
    if max_length < 1:
        return []
    return tokenizer.encode(text, add_special_tokens=True, max_length=max_length, truncation=True)


def _encode_untruncated(tokenizer: Any, text: str) -> list[int]:
    try:
        return tokenizer.encode(text, add_special_tokens=True, truncation=False, verbose=False)
    except TypeError:
        return tokenizer.encode(text, add_special_tokens=True, truncation=False)


def _truncate_keep_final_token(token_ids: list[int], max_length: int) -> list[int]:
    if max_length <= 0:
        return []
    if len(token_ids) <= max_length:
        return token_ids
    if max_length == 1:
        return [token_ids[-1]]
    return token_ids[: max_length - 1] + [token_ids[-1]]


def _segment_stat(
    segment_type: str,
    budget_tokens: int,
    raw_tokens: int,
    bucket_tokens: int,
    included_tokens: int,
) -> dict[str, Any]:
    return {
        "segment_type": segment_type,
        "budget_tokens": int(budget_tokens),
        "raw_tokens": int(raw_tokens),
        "bucket_tokens": int(bucket_tokens),
        "included_tokens": int(included_tokens),
        "bucket_truncated_tokens": max(0, int(raw_tokens) - int(bucket_tokens)),
        "global_truncated_tokens": max(0, int(bucket_tokens) - int(included_tokens)),
    }
