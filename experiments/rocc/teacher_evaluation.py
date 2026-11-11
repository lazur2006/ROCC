"""Teacher, random-token, and recency arms for cached ROCC evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .cached_itercqr import (
    CachedIterCQRBM25Pipeline,
    CachedIterCQRBM25Result,
)
from .progress import get_tqdm


TEACHER_METRIC_COLUMNS = (
    "MRR",
    "nDCG@3",
    "R@10",
    "R@100",
    "R@1000",
)


@dataclass
class TeacherArmEvaluationResult:
    """Materialized arms and their cached retrieval evaluation."""

    serialized_inputs: pd.DataFrame
    pipeline_results: pd.DataFrame
    serialization_summary: pd.DataFrame
    pipeline_summary: pd.DataFrame
    cached_pipeline_result: CachedIterCQRBM25Result

    @property
    def summary(self) -> pd.DataFrame:
        return self.pipeline_summary


def materialize_teacher_arm_inputs(
    *,
    samples: Sequence[Any],
    teacher_results: Mapping[str, dict[str, Any]],
    tokenizer: Any,
    budgets: Sequence[int],
    seed: int,
    pipeline: CachedIterCQRBM25Pipeline,
    progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and persist the exact NB04 Teacher/control inputs."""

    histories_by_arm: dict[
        str,
        dict[str, Sequence[dict[str, str]]],
    ] = {
        "teacher": {},
        "random_token": {},
        "recency": {},
    }
    row_metadata: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_id = str(sample.sample_id)
        validated = teacher_results[str(sample.sample_id)]
        selected_teacher = _teacher_positions(validated)
        keep_count = sum(map(len, selected_teacher.values()))
        selected_random = _random_positions(
            validated,
            seed=seed,
            sample_id=str(sample.sample_id),
            keep_count=keep_count,
        )
        histories_by_arm["teacher"][sample_id] = (
            compressed_teacher_history(validated)
        )
        histories_by_arm["random_token"][sample_id] = (
            _compressed_history(
                validated,
                selected_random,
            )
        )
        histories_by_arm["recency"][sample_id] = (
            _recency_history(sample)
        )
        row_metadata[sample_id] = {
            "teacher_keep_tokens": keep_count,
        }

    return materialize_history_arm_inputs(
        samples=samples,
        histories_by_arm=histories_by_arm,
        arm_order=("teacher", "random_token", "recency"),
        tokenizer=tokenizer,
        budgets=budgets,
        pipeline=pipeline,
        row_metadata=row_metadata,
        progress=progress,
        progress_desc="serialize teacher arms",
    )


def materialize_history_arm_inputs(
    *,
    samples: Sequence[Any],
    histories_by_arm: Mapping[
        str,
        Mapping[str, Sequence[dict[str, str]]],
    ],
    arm_order: Sequence[str],
    tokenizer: Any,
    budgets: Sequence[int],
    pipeline: CachedIterCQRBM25Pipeline,
    row_metadata: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
    progress: bool = True,
    progress_desc: str = "serialize history arms",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize arbitrary history arms through the canonical serializer."""

    ordered_arms = [str(arm) for arm in arm_order]
    if len(set(ordered_arms)) != len(ordered_arms):
        raise ValueError("arm_order contains duplicate arms.")
    if set(ordered_arms) != {
        str(arm) for arm in histories_by_arm
    }:
        raise ValueError(
            "arm_order and histories_by_arm must contain the same arms."
        )

    serialized_rows: list[dict[str, Any]] = []
    token_sequences: list[list[int]] = []
    tqdm = get_tqdm()
    for sample in tqdm(
        samples,
        desc=progress_desc,
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    ):
        sample_id = str(sample.sample_id)
        metadata = dict((row_metadata or {}).get(sample_id, {}))
        for budget in budgets:
            for arm in ordered_arms:
                histories = histories_by_arm[arm]
                if sample_id not in histories:
                    raise KeyError(
                        f"History for {sample_id!r}, arm {arm!r} is missing."
                    )
                token_ids, raw_length = serialize_itercqr_input(
                    tokenizer,
                    sample,
                    histories[sample_id],
                    budget=int(budget),
                )
                token_sequences.append(token_ids)
                serialized_rows.append(
                    {
                        "sample_id": sample_id,
                        "conv_id": int(sample.conv_id),
                        "turn_id": int(sample.turn_id),
                        "budget": int(budget),
                        "arm": arm,
                        "input_length": len(token_ids),
                        "raw_input_length": raw_length,
                        "was_truncated": (
                            raw_length > len(token_ids)
                        ),
                        **metadata,
                    }
                )

    input_keys = pipeline.register_token_inputs(token_sequences)
    serialized_inputs = pd.DataFrame(serialized_rows)
    serialized_inputs.insert(5, "input_key", input_keys)
    serialization_summary = (
        serialized_inputs.groupby(
            ["budget", "arm"],
            observed=True,
        )
        .agg(
            queries=("sample_id", "size"),
            unique_inputs=("input_key", "nunique"),
            mean_input_length=("input_length", "mean"),
            truncation_rate=("was_truncated", "mean"),
        )
        .reset_index()
    )
    return serialized_inputs, serialization_summary


def run_teacher_arm_evaluation(
    *,
    samples: Sequence[Any],
    teacher_results: Mapping[str, dict[str, Any]],
    tokenizer: Any,
    budgets: Sequence[int],
    seed: int,
    pipeline: CachedIterCQRBM25Pipeline,
    gold_by_sample: Mapping[str, Sequence[Any]],
    progress: bool = True,
) -> TeacherArmEvaluationResult:
    """Materialize the three arms and run the shared cached pipeline."""

    serialized_inputs, serialization_summary = (
        materialize_teacher_arm_inputs(
            samples=samples,
            teacher_results=teacher_results,
            tokenizer=tokenizer,
            budgets=budgets,
            seed=seed,
            pipeline=pipeline,
            progress=progress,
        )
    )
    return _run_materialized_history_arm_evaluation(
        serialized_inputs=serialized_inputs,
        serialization_summary=serialization_summary,
        pipeline=pipeline,
        gold_by_sample=gold_by_sample,
    )


def run_history_arm_evaluation(
    *,
    samples: Sequence[Any],
    histories_by_arm: Mapping[
        str,
        Mapping[str, Sequence[dict[str, str]]],
    ],
    arm_order: Sequence[str],
    tokenizer: Any,
    budgets: Sequence[int],
    pipeline: CachedIterCQRBM25Pipeline,
    gold_by_sample: Mapping[str, Sequence[Any]],
    row_metadata: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
    progress: bool = True,
    progress_desc: str = "serialize history arms",
) -> TeacherArmEvaluationResult:
    """Evaluate arbitrary history arms with the shared cached pipeline."""

    serialized_inputs, serialization_summary = (
        materialize_history_arm_inputs(
            samples=samples,
            histories_by_arm=histories_by_arm,
            arm_order=arm_order,
            tokenizer=tokenizer,
            budgets=budgets,
            pipeline=pipeline,
            row_metadata=row_metadata,
            progress=progress,
            progress_desc=progress_desc,
        )
    )
    return _run_materialized_history_arm_evaluation(
        serialized_inputs=serialized_inputs,
        serialization_summary=serialization_summary,
        pipeline=pipeline,
        gold_by_sample=gold_by_sample,
    )


def _run_materialized_history_arm_evaluation(
    *,
    serialized_inputs: pd.DataFrame,
    serialization_summary: pd.DataFrame,
    pipeline: CachedIterCQRBM25Pipeline,
    gold_by_sample: Mapping[str, Sequence[Any]],
) -> TeacherArmEvaluationResult:
    cached_result = pipeline.run(
        serialized_inputs,
        gold_by_sample=gold_by_sample,
        metric_columns=TEACHER_METRIC_COLUMNS,
    )
    pipeline_summary = pd.DataFrame(
        [
            {
                "rows": len(cached_result.evaluated),
                "unique_inputs": cached_result.unique_inputs,
                "rewrite_cache_misses": (
                    cached_result.rewrite_cache_misses
                ),
                "unique_rewrites": cached_result.unique_rewrites,
                "retrieval_cache_misses": (
                    cached_result.retrieval_cache_misses
                ),
            }
        ]
    )
    return TeacherArmEvaluationResult(
        serialized_inputs=serialized_inputs,
        pipeline_results=cached_result.evaluated,
        serialization_summary=serialization_summary,
        pipeline_summary=pipeline_summary,
        cached_pipeline_result=cached_result,
    )


def _teacher_positions(
    validated: dict[str, Any],
) -> dict[tuple[int, str], set[int]]:
    selected: dict[tuple[int, str], set[int]] = {}
    for turn in validated["history"]:
        turn_id = int(turn["turn_id"])
        for field in ("question", "answer"):
            tokens = turn[f"{field}_tokens"]
            positions = {
                index
                for index, token in enumerate(tokens)
                if token["labels"]
            }
            if positions:
                selected[(turn_id, field)] = positions
    return selected


def _token_chunks(
    tokens: Sequence[dict[str, Any]],
    selected: set[int],
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    previous: int | None = None
    for index in sorted(selected):
        if previous is None or index == previous + 1:
            current.append(str(tokens[index]["text"]))
        else:
            chunks.append(" ".join(current))
            current = [str(tokens[index]["text"])]
        previous = index
    if current:
        chunks.append(" ".join(current))
    return chunks


def _compressed_history(
    validated: dict[str, Any],
    selected: dict[tuple[int, str], set[int]],
) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for turn in validated["history"]:
        turn_id = int(turn["turn_id"])
        question = " ".join(
            _token_chunks(
                turn["question_tokens"],
                selected.get((turn_id, "question"), set()),
            )
        )
        answer = " ".join(
            _token_chunks(
                turn["answer_tokens"],
                selected.get((turn_id, "answer"), set()),
            )
        )
        if question or answer:
            history.append(
                {"question": question, "answer": answer}
            )
    return history


def compressed_teacher_history(
    validated: dict[str, Any],
    *,
    extra_positions: (
        Mapping[tuple[int, str], Sequence[int]] | None
    ) = None,
) -> list[dict[str, str]]:
    """Compress the complete Teacher history before IterCQR truncation."""

    selected = _teacher_positions(validated)
    for key, positions in (extra_positions or {}).items():
        selected.setdefault(
            (int(key[0]), str(key[1])),
            set(),
        ).update(int(position) for position in positions)
    return _compressed_history(validated, selected)


def _random_positions(
    validated: dict[str, Any],
    *,
    seed: int,
    sample_id: str,
    keep_count: int,
) -> dict[tuple[int, str], set[int]]:
    available: list[tuple[int, str, int]] = []
    for turn in validated["history"]:
        turn_id = int(turn["turn_id"])
        for field in ("question", "answer"):
            available.extend(
                (turn_id, field, index)
                for index in range(
                    len(turn[f"{field}_tokens"])
                )
            )
    if keep_count == 0:
        return {}

    seed_bytes = hashlib.sha256(
        (
            f"{seed}\0{sample_id}\0random_token_control"
        ).encode("utf-8")
    ).digest()
    rng = np.random.default_rng(
        int.from_bytes(seed_bytes[:8], byteorder="big")
    )
    chosen = rng.choice(
        len(available),
        size=keep_count,
        replace=False,
    )
    selected: dict[tuple[int, str], set[int]] = {}
    for position in chosen:
        turn_id, field, token_index = available[int(position)]
        selected.setdefault((turn_id, field), set()).add(
            token_index
        )
    return selected


def _recency_history(sample: Any) -> list[dict[str, str]]:
    return [
        {
            "question": str(turn.question),
            "answer": str(turn.answer),
        }
        for turn in sample.history
    ]


def _encode_untruncated(tokenizer: Any, text: str) -> list[int]:
    try:
        return tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
        )
    except TypeError:
        return tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False,
        )


def _encode_segment(
    tokenizer: Any,
    text: str,
    max_length: int,
) -> list[int]:
    return tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
    )


def _truncate_keep_final(
    token_ids: list[int],
    max_length: int,
) -> list[int]:
    if max_length <= 0:
        return []
    if len(token_ids) <= max_length:
        return token_ids
    if max_length == 1:
        return [token_ids[-1]]
    return token_ids[: max_length - 1] + [token_ids[-1]]


def serialize_itercqr_input(
    tokenizer: Any,
    sample: Any,
    history: Sequence[dict[str, str]],
    *,
    budget: int,
) -> tuple[list[int], int]:
    current_query = f"question: {sample.current_query}"
    raw_input_ids = _encode_untruncated(
        tokenizer,
        current_query,
    )
    input_ids = _encode_segment(tokenizer, current_query, 32)

    first_context = True
    for turn in reversed(history):
        answer = str(turn["answer"])
        if first_context:
            answer = f"context: {answer}" if answer else "context:"
            first_context = False
        for text in (answer, str(turn["question"])):
            raw_segment = _encode_untruncated(tokenizer, text)
            segment = _encode_segment(tokenizer, text, 32)
            raw_input_ids.extend(raw_segment)
            remaining = int(budget) - len(input_ids)
            input_ids.extend(
                _truncate_keep_final(segment, remaining)
            )

    return input_ids, len(raw_input_ids)


__all__ = [
    "TEACHER_METRIC_COLUMNS",
    "TeacherArmEvaluationResult",
    "compressed_teacher_history",
    "materialize_history_arm_inputs",
    "materialize_teacher_arm_inputs",
    "run_history_arm_evaluation",
    "run_teacher_arm_evaluation",
    "serialize_itercqr_input",
]
