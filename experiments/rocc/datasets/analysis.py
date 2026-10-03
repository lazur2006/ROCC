"""Notebook-friendly dataset diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .itercqr import IterCQRDataPipelineResult, load_itercqr_dataloaders
from .qrecc import load_qrecc_frame
from .topiocqa import (
    compare_positive_contexts_to_lookup,
    load_topiocqa_passage_lookup,
    positive_passage_ids,
    summarize_topiocqa_split,
)


@dataclass
class IterCQRDatasetAnalysis:
    split_summary: pd.DataFrame
    dataloader_summary: pd.DataFrame
    padding_report: pd.DataFrame
    truncation_report: pd.DataFrame
    unknown_token_report: pd.DataFrame
    formatted_input_samples: pd.DataFrame
    segment_bucket_report: pd.DataFrame
    segment_bucket_samples: pd.DataFrame
    context_match_summary: pd.DataFrame | None = None
    non_exact_contexts: pd.DataFrame | None = None

    def as_dict(self) -> dict[str, pd.DataFrame | None]:
        return {
            "split_summary": self.split_summary,
            "dataloader_summary": self.dataloader_summary,
            "padding_report": self.padding_report,
            "truncation_report": self.truncation_report,
            "unknown_token_report": self.unknown_token_report,
            "formatted_input_samples": self.formatted_input_samples,
            "segment_bucket_report": self.segment_bucket_report,
            "segment_bucket_samples": self.segment_bucket_samples,
            "context_match_summary": self.context_match_summary,
            "non_exact_contexts": self.non_exact_contexts,
        }


def analyze_itercqr_dataset(
    source: IterCQRDataPipelineResult | Callable[[], IterCQRDataPipelineResult] | None = None,
    *,
    include_corpus_text_check: bool = False,
    sample_formatted_inputs: bool = True,
    formatted_sample_size: int = 5,
    formatted_sample_split: str | None = None,
    progress: bool = True,
    **pipeline_kwargs: Any,
) -> IterCQRDatasetAnalysis:
    result = _resolve_source(source, pipeline_kwargs)
    frame = result.frame
    dataset_name = result.dataset_name.lower()
    split_summary_frame = _split_summary_frame(result, frame, dataset_name)

    context_matches = None
    if include_corpus_text_check:
        if dataset_name != "topiocqa":
            raise ValueError("Corpus text check is currently only implemented for TopiOCQA.")
        if result.resources is None:
            raise ValueError("Corpus text check requires dataset resources.")
        lookup = load_topiocqa_passage_lookup(
            result.resources.corpus_tsv,
            positive_passage_ids(frame),
            progress=progress,
        )
        context_matches = compare_positive_contexts_to_lookup(frame, lookup)

    split_rows = []
    split_rows.append(_summarize_split(dataset_name, "all", split_summary_frame, context_matches))
    for split_name, split_frame in split_summary_frame.groupby("split", sort=True):
        split_context_matches = None
        if context_matches is not None:
            split_context_matches = context_matches[context_matches["split"].eq(split_name)]
        split_rows.append(
            _summarize_split(dataset_name, split_name, split_frame, split_context_matches)
        )

    dataloader_summary = pd.DataFrame(
        [
            {
                "split": split,
                "num_turns": len(dataloader.dataset),
                "num_batches": len(dataloader),
                "batch_size": result.config.batch_size,
                "num_workers": result.config.num_workers,
            }
            for split, dataloader in result.dataloaders.items()
        ]
    )

    padding_report = pd.DataFrame(
        [
            {
                "split": split,
                **{
                    key: value
                    for key, value in dataloader.dataset.padding_report(
                        result.config.batch_size
                    ).items()
                    if key != "batches"
                },
            }
            for split, dataloader in result.dataloaders.items()
        ]
    )

    truncation_report = pd.DataFrame(
        [
            {"split": split, **dataloader.dataset.truncation_report()}
            for split, dataloader in result.dataloaders.items()
        ]
    )

    unknown_token_report = pd.DataFrame(
        [
            {"split": split, **dataloader.dataset.unknown_token_report()}
            for split, dataloader in result.dataloaders.items()
        ]
    )
    formatted_input_samples = (
        _formatted_input_samples(
            result,
            sample_size=formatted_sample_size,
            split_name=formatted_sample_split,
        )
        if sample_formatted_inputs
        else pd.DataFrame()
    )
    segment_bucket_report = _segment_bucket_report(result)
    segment_bucket_samples = _segment_bucket_samples(
        result,
        sample_size=formatted_sample_size,
        split_name=formatted_sample_split,
    )

    context_match_summary = None
    non_exact_contexts = None
    if context_matches is not None:
        context_match_summary = pd.DataFrame(split_rows)
        non_exact_contexts = context_matches[
            ~context_matches["exact_match"]
        ].reset_index(drop=True)

    return IterCQRDatasetAnalysis(
        split_summary=pd.DataFrame(split_rows),
        dataloader_summary=dataloader_summary,
        padding_report=padding_report,
        truncation_report=truncation_report,
        unknown_token_report=unknown_token_report,
        formatted_input_samples=formatted_input_samples,
        segment_bucket_report=segment_bucket_report,
        segment_bucket_samples=segment_bucket_samples,
        context_match_summary=context_match_summary,
        non_exact_contexts=non_exact_contexts,
    )


def _formatted_input_samples(
    result: IterCQRDataPipelineResult,
    *,
    sample_size: int,
    split_name: str | None,
) -> pd.DataFrame:
    if sample_size < 1:
        return pd.DataFrame()

    if split_name is None:
        try:
            split_name = next(iter(result.dataloaders))
        except StopIteration:
            return pd.DataFrame()
    if split_name not in result.dataloaders:
        available = ", ".join(result.dataloaders)
        raise ValueError(f"Unknown formatted_sample_split: {split_name}. Available: {available}")

    dataloader = result.dataloaders[split_name]
    dataset = dataloader.dataset
    tokenizer = dataset.tokenizer
    rows = []
    for example in dataset.examples[:sample_size]:
        metadata = example.metadata
        row = {
            "split": metadata.get("split", split_name),
            "sample_id": example.sample_id,
            "conv_id": metadata.get("conv_id"),
            "turn_id": metadata.get("turn_id"),
            "topic_document": metadata.get("topic_document"),
            "topic_id": metadata.get("topic_id"),
            "topic_switch": metadata.get("topic_switch"),
            "topic_switch_depth": metadata.get("topic_switch_depth"),
            "topics_in_conversation": metadata.get("topics_in_conversation"),
            "question": example.question,
            "input_length": example.input_length,
            "raw_input_length": example.raw_input_length,
            "final_input_length": example.input_length,
            "was_truncated": example.was_truncated,
            "truncated_tokens": example.truncated_tokens,
            **_history_segment_summary(example),
            "formatted_input_text": tokenizer.decode(
                example.input_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ),
        }
        if "positive_ctx_passage_ids" in metadata:
            row["positive_ctx_passage_ids"] = metadata["positive_ctx_passage_ids"]
        rows.append(row)

    return pd.DataFrame(rows)


def _segment_bucket_report(result: IterCQRDataPipelineResult) -> pd.DataFrame:
    rows = _segment_bucket_rows(result)
    columns = [
        "split",
        "segment_type",
        "budget_tokens",
        "count",
        "raw_tokens_min",
        "raw_tokens_mean",
        "raw_tokens_max",
        "bucket_tokens_min",
        "bucket_tokens_mean",
        "bucket_tokens_max",
        "included_tokens_min",
        "included_tokens_mean",
        "included_tokens_max",
        "bucket_utilization_mean",
        "bucket_utilization_max",
        "bucket_truncation_rate",
        "global_truncation_rate",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    report_rows = []
    for (split, segment_type, budget_tokens), group in frame.groupby(
        ["split", "segment_type", "budget_tokens"],
        sort=True,
    ):
        report_rows.append(
            {
                "split": split,
                "segment_type": segment_type,
                "budget_tokens": int(budget_tokens),
                "count": int(len(group)),
                "raw_tokens_min": int(group["raw_tokens"].min()),
                "raw_tokens_mean": float(group["raw_tokens"].mean()),
                "raw_tokens_max": int(group["raw_tokens"].max()),
                "bucket_tokens_min": int(group["bucket_tokens"].min()),
                "bucket_tokens_mean": float(group["bucket_tokens"].mean()),
                "bucket_tokens_max": int(group["bucket_tokens"].max()),
                "included_tokens_min": int(group["included_tokens"].min()),
                "included_tokens_mean": float(group["included_tokens"].mean()),
                "included_tokens_max": int(group["included_tokens"].max()),
                "bucket_utilization_mean": float(group["bucket_utilization"].mean()),
                "bucket_utilization_max": float(group["bucket_utilization"].max()),
                "bucket_truncation_rate": float(
                    (group["bucket_truncated_tokens"] > 0).mean()
                ),
                "global_truncation_rate": float(
                    (group["global_truncated_tokens"] > 0).mean()
                ),
            }
        )
    return pd.DataFrame(report_rows, columns=columns)


def _segment_bucket_samples(
    result: IterCQRDataPipelineResult,
    *,
    sample_size: int,
    split_name: str | None,
) -> pd.DataFrame:
    if sample_size < 1:
        return pd.DataFrame()

    if split_name is None:
        try:
            split_name = next(iter(result.dataloaders))
        except StopIteration:
            return pd.DataFrame()
    if split_name not in result.dataloaders:
        available = ", ".join(result.dataloaders)
        raise ValueError(f"Unknown formatted_sample_split: {split_name}. Available: {available}")

    rows = []
    for example in result.dataloaders[split_name].dataset.examples[:sample_size]:
        metadata = example.metadata
        for segment_index, segment in enumerate(example.segment_token_stats):
            rows.append(
                {
                    "split": metadata.get("split", split_name),
                    "sample_id": example.sample_id,
                    "conv_id": metadata.get("conv_id"),
                    "turn_id": metadata.get("turn_id"),
                    "segment_index": segment_index,
                    **segment,
                }
            )
    return pd.DataFrame(rows)


def _segment_bucket_rows(result: IterCQRDataPipelineResult) -> list[dict[str, Any]]:
    rows = []
    for split, dataloader in result.dataloaders.items():
        for example in dataloader.dataset.examples:
            metadata = example.metadata
            for segment in example.segment_token_stats:
                rows.append(
                    {
                        "split": metadata.get("split", split),
                        "sample_id": example.sample_id,
                        "conv_id": metadata.get("conv_id"),
                        "turn_id": metadata.get("turn_id"),
                        **segment,
                    }
                )
    return rows


def _history_segment_summary(example: Any) -> dict[str, Any]:
    stats = list(getattr(example, "segment_token_stats", []) or [])
    current_query = next(
        (segment for segment in stats if segment.get("segment_type") == "current_query"),
        {},
    )
    history_stats = [
        segment
        for segment in stats
        if segment.get("segment_type") != "current_query"
        and int(segment.get("included_tokens", 0)) > 0
    ]
    history_utilizations = [
        float(segment.get("bucket_utilization", 0.0))
        for segment in history_stats
    ]
    return {
        "current_query_raw_tokens": current_query.get("raw_tokens"),
        "current_query_bucket_tokens": current_query.get("bucket_tokens"),
        "history_segments_used": len(history_stats),
        "history_bucket_utilization_mean": (
            sum(history_utilizations) / len(history_utilizations)
            if history_utilizations
            else 0.0
        ),
        "history_bucket_utilization_max": (
            max(history_utilizations) if history_utilizations else 0.0
        ),
    }


def _resolve_source(
    source: IterCQRDataPipelineResult | Callable[[], IterCQRDataPipelineResult] | None,
    pipeline_kwargs: dict[str, Any],
) -> IterCQRDataPipelineResult:
    if source is None:
        return load_itercqr_dataloaders(**pipeline_kwargs)
    if callable(source):
        return source()
    return source


def _split_summary_frame(
    result: IterCQRDataPipelineResult,
    frame: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    if dataset_name == "qrecc" and result.resources is not None:
        return load_qrecc_frame(result.resources, prefer_processed=True, gold_only=False)
    return frame


def _summarize_split(
    dataset_name: str,
    label: str,
    frame: pd.DataFrame,
    context_matches: pd.DataFrame | None = None,
) -> dict[str, object]:
    if dataset_name == "topiocqa":
        return summarize_topiocqa_split(label, frame, context_matches)
    if dataset_name == "qrecc":
        if context_matches is not None:
            raise ValueError("Corpus text check is currently only implemented for TopiOCQA.")
        return _summarize_qrecc_split(label, frame)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def _summarize_qrecc_split(label: str, frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "split": label,
            "conversations": 0,
            "turns": 0,
            "longest_conversation_turns": 0,
            "shortest_conversation_turns": 0,
            "mean_turns_per_conversation": 0.0,
            "turns_without_positive_ctx_passage_ids": 0,
        }

    group_columns = [column for column in ("split", "conv_id") if column in frame.columns]
    conversation_turn_counts = frame.groupby(group_columns)["turn_id"].nunique()
    turns_without_positive_ctx_passage_ids = int(
        (frame["positive_ctx_passage_ids"].apply(_list_length) == 0).sum()
    )

    return {
        "split": label,
        "conversations": int(frame.groupby(group_columns).ngroups),
        "turns": int(len(frame)),
        "longest_conversation_turns": int(conversation_turn_counts.max()),
        "shortest_conversation_turns": int(conversation_turn_counts.min()),
        "mean_turns_per_conversation": round(float(conversation_turn_counts.mean()), 2),
        "turns_without_positive_ctx_passage_ids": turns_without_positive_ctx_passage_ids,
    }


def _list_length(value: Any) -> int:
    if isinstance(value, list | tuple | set):
        return len(value)
    return 0
