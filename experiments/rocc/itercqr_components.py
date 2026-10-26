"""Self-contained IterCQR tokenizer, model, and dataloader components."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

from .paths import project_path
from .population import derive_query_seed

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError:
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[no-redef]
        pass


def default_itercqr_model_dir() -> Path:
    return project_path("experiments", "model", "IterCQR", "IterCQR Model")


def default_hf_t5_backbone_dir(repo_id: str = "t5-base") -> Path:
    return project_path("experiments", "model", "hf", repo_id.replace("/", "__"))


def ensure_hf_t5_backbone(
    repo_id: str = "t5-base",
    *,
    local_dir: Path | str | None = None,
    force: bool = False,
) -> Path:
    """Download a HuggingFace T5 backbone and store it inside experiments.

    This is a reference/backbone snapshot. IterCQR inference still uses the
    fine-tuned IterCQR checkpoint, including its tokenizer, from
    `experiments/model/IterCQR/IterCQR Model`.
    """

    target = Path(local_dir).expanduser().resolve() if local_dir else default_hf_t5_backbone_dir(repo_id)
    has_weights = (target / "pytorch_model.bin").exists() or (target / "model.safetensors").exists()
    if target.exists() and has_weights and not force:
        return target

    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers is required to download a T5 backbone.") from exc

    target.mkdir(parents=True, exist_ok=True)
    tokenizer = T5Tokenizer.from_pretrained(repo_id, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(repo_id)
    tokenizer.save_pretrained(target)
    model.save_pretrained(target)
    return target


def iter_checkpoint_files(model_dir: Path | str | None = None) -> list[str]:
    resolved = Path(model_dir).expanduser().resolve() if model_dir else default_itercqr_model_dir()
    if not resolved.exists():
        return []
    return sorted(path.name for path in resolved.iterdir() if path.is_file())


def validate_itercqr_checkpoint(model_dir: Path | str | None = None) -> Path:
    resolved = Path(model_dir).expanduser().resolve() if model_dir else default_itercqr_model_dir()
    required = {
        "config.json",
        "generation_config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer_config.json",
    }
    missing = sorted(required - set(iter_checkpoint_files(resolved)))
    if missing:
        raise FileNotFoundError(f"IterCQR checkpoint is incomplete at {resolved}: missing {missing}")
    return resolved


@dataclass(frozen=True)
class IterCQRRoccConfig:
    max_query_tokens: int = 32
    max_history_turn_tokens: int = 64
    max_input_tokens: int = 512
    use_prefix: bool = True
    pad_to_multiple_of: int | None = None
    max_history_answer_tokens: int = 32
    include_history: bool = True

    def __post_init__(self) -> None:
        if self.max_query_tokens < 2:
            raise ValueError("max_query_tokens must be at least 2.")
        if self.max_history_turn_tokens < 2:
            raise ValueError("max_history_turn_tokens must be at least 2.")
        if self.max_history_answer_tokens < 2:
            raise ValueError("max_history_answer_tokens must be at least 2.")
        expected_turn_tokens = self.max_query_tokens + self.max_history_answer_tokens
        if self.max_history_turn_tokens != expected_turn_tokens:
            raise ValueError(
                "max_history_turn_tokens must equal "
                "max_query_tokens + max_history_answer_tokens."
            )
        if self.max_input_tokens < self.max_query_tokens:
            raise ValueError("max_input_tokens must be >= max_query_tokens.")
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of < 1:
            raise ValueError("pad_to_multiple_of must be None or >= 1.")


@dataclass(frozen=True)
class HistoryTurn:
    question: str
    answer: str


@dataclass(frozen=True)
class TokenizedExample:
    sample_id: str
    input_ids: list[int]
    input_length: int
    raw_input_length: int
    truncated_tokens: int
    was_truncated: bool
    question: str
    metadata: dict[str, Any]
    segment_token_stats: list[dict[str, Any]] = field(default_factory=list)


class IterCQRTopiOCQARoccDataset(Dataset):
    required_columns = {"conv_id", "turn_id", "question", "answers"}

    def __init__(
        self,
        frame: Any,
        tokenizer: Any,
        config: IterCQRRoccConfig | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.config = config or IterCQRRoccConfig()
        self.pad_token_id = self._resolve_pad_token_id(tokenizer)
        self.unk_token_id = self._resolve_unk_token_id(tokenizer)
        self.examples = self._tokenize_frame(frame)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TokenizedExample:
        return self.examples[index]

    def collate_fn(self, batch: list[TokenizedExample]) -> dict[str, Any]:
        if torch is None:
            raise RuntimeError("PyTorch is required to collate batches.")

        pad_length = self._batch_pad_length(example.input_length for example in batch)
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
            "bt_input_lengths": torch.tensor(
                [example.input_length for example in batch],
                dtype=torch.long,
            ),
            "bt_questions": [example.question for example in batch],
            "bt_metadata": [example.metadata for example in batch],
        }

    def bucketed_batch_indices(self, batch_size: int) -> list[list[int]]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        indices = sorted(range(len(self.examples)), key=lambda idx: self.examples[idx].input_length)
        return [indices[start : start + batch_size] for start in range(0, len(indices), batch_size)]

    def make_dataloader(self, batch_size: int, *, num_workers: int = 0) -> DataLoader:
        if DataLoader is None:
            raise RuntimeError("PyTorch is required to create a DataLoader.")
        return DataLoader(
            self,
            batch_sampler=self.bucketed_batch_indices(batch_size),
            collate_fn=self.collate_fn,
            num_workers=num_workers,
        )

    def padding_report(self, batch_size: int) -> dict[str, Any]:
        if not self.examples:
            return {"num_turns": 0, "batch_size": batch_size, "message": "Dataset is empty."}

        lengths = [example.input_length for example in self.examples]
        real_tokens = sum(lengths)
        static_total_tokens = len(lengths) * self.config.max_input_tokens
        static_pad_tokens = static_total_tokens - real_tokens
        dynamic_total_tokens = 0
        batch_rows = []

        for batch_number, batch in enumerate(self.bucketed_batch_indices(batch_size), start=1):
            batch_lengths = [self.examples[index].input_length for index in batch]
            pad_length = self._batch_pad_length(batch_lengths)
            total_tokens = pad_length * len(batch)
            pad_tokens = total_tokens - sum(batch_lengths)
            dynamic_total_tokens += total_tokens
            batch_rows.append(
                {
                    "batch": batch_number,
                    "size": len(batch),
                    "min_length": min(batch_lengths),
                    "max_length": max(batch_lengths),
                    "pad_to_length": pad_length,
                    "pad_tokens": pad_tokens,
                    "pad_fraction": pad_tokens / total_tokens if total_tokens else 0.0,
                }
            )

        dynamic_pad_tokens = dynamic_total_tokens - real_tokens
        return {
            "num_turns": len(lengths),
            "batch_size": batch_size,
            "max_input_tokens": self.config.max_input_tokens,
            "length_min": min(lengths),
            "length_mean": mean(lengths),
            "length_median": median(lengths),
            "length_max": max(lengths),
            "real_tokens": real_tokens,
            "static_total_tokens": static_total_tokens,
            "static_pad_tokens": static_pad_tokens,
            "static_pad_fraction": static_pad_tokens / static_total_tokens,
            "bucketed_dynamic_total_tokens": dynamic_total_tokens,
            "bucketed_dynamic_pad_tokens": dynamic_pad_tokens,
            "bucketed_dynamic_pad_fraction": dynamic_pad_tokens / dynamic_total_tokens,
            "padding_fraction_saved_points": (
                static_pad_tokens / static_total_tokens
                - dynamic_pad_tokens / dynamic_total_tokens
            ),
            "padding_tokens_saved": static_pad_tokens - dynamic_pad_tokens,
            "batches": batch_rows,
        }

    def truncation_report(self) -> dict[str, Any]:
        if not self.examples:
            return {
                "num_turns": 0,
                "max_input_tokens": self.config.max_input_tokens,
                "message": "Dataset is empty.",
            }

        raw_lengths = [example.raw_input_length for example in self.examples]
        final_lengths = [example.input_length for example in self.examples]
        truncated_turns = sum(1 for example in self.examples if example.was_truncated)
        truncated_tokens_total = sum(example.truncated_tokens for example in self.examples)
        return {
            "num_turns": len(self.examples),
            "max_input_tokens": self.config.max_input_tokens,
            "raw_length_min": min(raw_lengths),
            "raw_length_mean": mean(raw_lengths),
            "raw_length_median": median(raw_lengths),
            "raw_length_max": max(raw_lengths),
            "final_length_min": min(final_lengths),
            "final_length_mean": mean(final_lengths),
            "final_length_median": median(final_lengths),
            "final_length_max": max(final_lengths),
            "truncated_turns": truncated_turns,
            "truncation_rate": truncated_turns / len(self.examples),
            "truncated_tokens_total": truncated_tokens_total,
            "mean_truncated_tokens_all": truncated_tokens_total / len(self.examples),
            "mean_truncated_tokens_truncated_only": (
                truncated_tokens_total / truncated_turns if truncated_turns else 0.0
            ),
        }

    def unknown_token_report(self) -> dict[str, Any]:
        total_tokens = sum(example.input_length for example in self.examples)
        unknown_tokens = sum(
            token_id == self.unk_token_id
            for example in self.examples
            for token_id in example.input_ids
        )
        examples_with_unknown = sum(
            any(token_id == self.unk_token_id for token_id in example.input_ids)
            for example in self.examples
        )
        return {
            "unk_token_id": self.unk_token_id,
            "unk_tokens": unknown_tokens,
            "unk_rate": unknown_tokens / total_tokens if total_tokens else 0.0,
            "examples_with_unk": examples_with_unknown,
            "examples_with_unk_fraction": (
                examples_with_unknown / len(self.examples) if self.examples else 0.0
            ),
        }

    def _tokenize_frame(self, frame: Any) -> list[TokenizedExample]:
        missing_columns = self.required_columns - set(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input frame is missing required columns: {missing}")

        sort_columns = [column for column in ("split", "conv_id", "turn_id") if column in frame.columns]
        rows = frame.sort_values(sort_columns).to_dict("records")
        group_keys = ["conv_id"]
        if "split" in frame.columns:
            group_keys.insert(0, "split")

        grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row[column] for column in group_keys)
            grouped_rows.setdefault(key, []).append(row)

        examples: list[TokenizedExample] = []
        for group in grouped_rows.values():
            history: list[HistoryTurn] = []
            for row in group:
                examples.append(self._build_example(row, history))
                history.append(
                    HistoryTurn(
                        question=self._normalise_text(row["question"]),
                        answer=self._normalise_answer(row["answers"]),
                    )
                )
        return examples

    def _build_example(
        self,
        row: dict[str, Any],
        history: list[HistoryTurn],
    ) -> TokenizedExample:
        question = self._normalise_text(row["question"])
        current_query = f"question: {question}" if self.config.use_prefix else question
        raw_query_ids = self._encode_segment_untruncated(current_query)
        query_ids = self._encode_segment(current_query, self.config.max_query_tokens)
        raw_input_ids = list(raw_query_ids)
        input_ids = list(query_ids)
        segment_token_stats = [
            self._segment_token_stat(
                segment_type="current_query",
                budget_tokens=self.config.max_query_tokens,
                raw_tokens=len(raw_query_ids),
                bucket_tokens=len(query_ids),
                included_tokens=len(query_ids),
            )
        ]
        if not self.config.include_history:
            return self._make_example(row, question, input_ids, raw_input_ids, segment_token_stats)

        first_context = self.config.use_prefix
        for turn in reversed(history):
            answer_text = turn.answer
            if self.config.use_prefix and first_context:
                answer_text = f"context: {answer_text}" if answer_text else "context:"
                first_context = False

            for segment_type, text, budget_tokens in (
                ("history_answer", answer_text, self.config.max_history_answer_tokens),
                ("history_question", turn.question, self.config.max_query_tokens),
            ):
                raw_segment_ids = self._encode_segment_untruncated(text)
                segment_ids = self._encode_segment(text, budget_tokens)
                raw_input_ids.extend(raw_segment_ids)
                remaining = self.config.max_input_tokens - len(input_ids)
                if remaining <= 0:
                    included_ids = []
                elif len(segment_ids) > remaining:
                    included_ids = self._truncate_keep_final_token(segment_ids, remaining)
                else:
                    included_ids = segment_ids
                input_ids.extend(included_ids)
                segment_token_stats.append(
                    self._segment_token_stat(
                        segment_type=segment_type,
                        budget_tokens=budget_tokens,
                        raw_tokens=len(raw_segment_ids),
                        bucket_tokens=len(segment_ids),
                        included_tokens=len(included_ids),
                    )
                )

        metadata = {
            key: row.get(key)
            for key in (
                "split",
                "conv_id",
                "turn_id",
                "topic_id",
                "topic_document",
                "topic_switch",
                "topic_switch_depth",
                "topics_in_conversation",
                "positive_ctx_passage_ids",
            )
            if key in row
        }
        metadata["conversation_turn_depth"] = self._derive_conversation_turn_depth(row)
        metadata["topic_switch_depth"] = self._derive_topic_switch_depth(row)
        return TokenizedExample(
            sample_id=self._sample_id(row),
            input_ids=input_ids,
            input_length=len(input_ids),
            raw_input_length=len(raw_input_ids),
            truncated_tokens=max(0, len(raw_input_ids) - len(input_ids)),
            was_truncated=len(raw_input_ids) > len(input_ids),
            question=question,
            metadata=metadata,
            segment_token_stats=segment_token_stats,
        )

    def _make_example(
        self,
        row: dict[str, Any],
        question: str,
        input_ids: list[int],
        raw_input_ids: list[int],
        segment_token_stats: list[dict[str, Any]],
    ) -> TokenizedExample:
        metadata = {
            key: row.get(key)
            for key in (
                "split",
                "conv_id",
                "turn_id",
                "topic_id",
                "topic_document",
                "topic_switch",
                "topic_switch_depth",
                "topics_in_conversation",
                "positive_ctx_passage_ids",
            )
            if key in row
        }
        metadata["conversation_turn_depth"] = self._derive_conversation_turn_depth(row)
        metadata["topic_switch_depth"] = self._derive_topic_switch_depth(row)
        return TokenizedExample(
            sample_id=self._sample_id(row),
            input_ids=input_ids,
            input_length=len(input_ids),
            raw_input_length=len(raw_input_ids),
            truncated_tokens=max(0, len(raw_input_ids) - len(input_ids)),
            was_truncated=len(raw_input_ids) > len(input_ids),
            question=question,
            metadata=metadata,
            segment_token_stats=segment_token_stats,
        )

    @staticmethod
    def _segment_token_stat(
        *,
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
            "bucket_utilization": (
                float(bucket_tokens) / float(budget_tokens) if budget_tokens else 0.0
            ),
            "included_utilization": (
                float(included_tokens) / float(budget_tokens) if budget_tokens else 0.0
            ),
        }

    def _encode_segment(self, text: str, max_length: int) -> list[int]:
        if max_length < 1:
            return []
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
        )

    def _encode_segment_untruncated(self, text: str) -> list[int]:
        try:
            return self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
                verbose=False,
            )
        except TypeError:
            return self.tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )

    def _batch_pad_length(self, lengths: Iterable[int]) -> int:
        pad_length = max(lengths)
        multiple = self.config.pad_to_multiple_of
        if multiple is not None:
            pad_length = int(ceil(pad_length / multiple) * multiple)
        return min(pad_length, self.config.max_input_tokens)

    @staticmethod
    def _truncate_keep_final_token(token_ids: list[int], max_length: int) -> list[int]:
        if max_length <= 0:
            return []
        if len(token_ids) <= max_length:
            return token_ids
        if max_length == 1:
            return [token_ids[-1]]
        return token_ids[: max_length - 1] + [token_ids[-1]]

    @staticmethod
    def _normalise_answer(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            for item in value:
                text = IterCQRTopiOCQARoccDataset._normalise_text(item)
                if text:
                    return text
            return ""
        return IterCQRTopiOCQARoccDataset._normalise_text(value)

    @staticmethod
    def _normalise_text(value: Any) -> str:
        if value is None:
            return ""
        try:
            if value != value:
                return ""
        except TypeError:
            pass
        return str(value).strip()

    @staticmethod
    def _sample_id(row: dict[str, Any]) -> str:
        if row.get("id") is not None:
            return str(row["id"])
        split = row.get("split", "unknown")
        return f"{split}:{row.get('conv_id')}:{row.get('turn_id')}"

    @classmethod
    def _derive_conversation_turn_depth(cls, row: dict[str, Any]) -> int | None:
        turn_id = cls._coerce_optional_int(row.get("turn_id"))
        return None if turn_id is None else turn_id - 1

    @classmethod
    def _derive_topic_switch_depth(cls, row: dict[str, Any]) -> int | None:
        explicit_depth = cls._coerce_optional_int(row.get("topic_switch_depth"))
        if explicit_depth is not None:
            return explicit_depth
        topic_id = cls._coerce_optional_int(row.get("topic_id"))
        return None if topic_id is None else topic_id - 1

    @staticmethod
    def _coerce_optional_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            if value != value:
                return None
        except TypeError:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _resolve_pad_token_id(tokenizer: Any) -> int:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id.")
        return int(pad_token_id)

    @staticmethod
    def _resolve_unk_token_id(tokenizer: Any) -> int:
        unk_token_id = getattr(tokenizer, "unk_token_id", None)
        if unk_token_id is None:
            raise ValueError("Tokenizer must define unk_token_id.")
        return int(unk_token_id)


class IterCQRQReCCRoccDataset(IterCQRTopiOCQARoccDataset):
    """IterCQR-compatible QReCC dataset using the dataset-provided context."""

    required_columns = {
        "conv_id",
        "turn_id",
        "question",
        "answers",
        "rewrite",
        "context",
        "positive_ctx_passage_ids",
    }

    def _tokenize_frame(self, frame: Any) -> list[TokenizedExample]:
        missing_columns = self.required_columns - set(frame.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Input frame is missing required columns: {missing}")

        sort_columns = [column for column in ("split", "conv_id", "turn_id") if column in frame.columns]
        rows = frame.sort_values(sort_columns).to_dict("records")
        group_keys = ["conv_id"]
        if "split" in frame.columns:
            group_keys.insert(0, "split")

        grouped_rows: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            key = tuple(row[column] for column in group_keys)
            grouped_rows.setdefault(key, []).append(row)

        examples: list[TokenizedExample] = []
        for group in grouped_rows.values():
            turn_id_to_utterance: dict[int, str] = {}
            for row in group:
                turn_id = self._coerce_optional_int(row.get("turn_id"))
                current_utterance = self._qrecc_current_utterance(row)
                if turn_id is not None:
                    turn_id_to_utterance[turn_id] = current_utterance

                if not self._has_positive_ctx_passage_ids(row):
                    continue

                context_utterances = self._qrecc_context_utterances(
                    row,
                    turn_id_to_utterance,
                )
                examples.append(
                    self._build_qrecc_example(
                        row,
                        current_utterance=current_utterance,
                        context_utterances=context_utterances,
                    )
                )
        return examples

    def _build_qrecc_example(
        self,
        row: dict[str, Any],
        *,
        current_utterance: str,
        context_utterances: list[str],
    ) -> TokenizedExample:
        question = self._normalise_text(current_utterance)
        current_query = f"question: {question}" if self.config.use_prefix else question
        raw_query_ids = self._encode_segment_untruncated(current_query)
        query_ids = self._encode_segment(current_query, self.config.max_query_tokens)
        raw_input_ids = list(raw_query_ids)
        input_ids = list(query_ids)
        segment_token_stats = [
            self._segment_token_stat(
                segment_type="current_query",
                budget_tokens=self.config.max_query_tokens,
                raw_tokens=len(raw_query_ids),
                bucket_tokens=len(query_ids),
                included_tokens=len(query_ids),
            )
        ]
        if not self.config.include_history:
            return self._make_example(row, question, input_ids, raw_input_ids, segment_token_stats)

        first_context = self.config.use_prefix
        for index in range(len(context_utterances) - 1, -1, -1):
            context_text = context_utterances[index]
            if self.config.use_prefix and first_context:
                context_text = f"context: {context_text}"
                first_context = False
            raw_context_ids = self._encode_segment_untruncated(context_text)
            raw_input_ids.extend(raw_context_ids)
            max_length = (
                self.config.max_history_answer_tokens
                if index % 2 == 1
                else self.config.max_query_tokens
            )
            context_ids = self._encode_segment(context_text, max_length)
            remaining = self.config.max_input_tokens - len(input_ids)
            if remaining <= 0:
                included_ids = []
            elif len(context_ids) > remaining:
                included_ids = self._truncate_keep_final_token(context_ids, remaining)
            else:
                included_ids = context_ids
            input_ids.extend(included_ids)
            segment_token_stats.append(
                self._segment_token_stat(
                    segment_type="history_answer" if index % 2 == 1 else "history_question",
                    budget_tokens=max_length,
                    raw_tokens=len(raw_context_ids),
                    bucket_tokens=len(context_ids),
                    included_tokens=len(included_ids),
                )
            )

        metadata = {
            key: row.get(key)
            for key in (
                "split",
                "conv_id",
                "turn_id",
                "topic_id",
                "topic_document",
                "topic_switch",
                "topic_switch_depth",
                "topics_in_conversation",
                "positive_ctx_passage_ids",
            )
            if key in row
        }
        metadata["conversation_turn_depth"] = self._derive_conversation_turn_depth(row)
        metadata["topic_switch_depth"] = self._derive_topic_switch_depth(row)
        return TokenizedExample(
            sample_id=self._sample_id(row),
            input_ids=input_ids,
            input_length=len(input_ids),
            raw_input_length=len(raw_input_ids),
            truncated_tokens=max(0, len(raw_input_ids) - len(input_ids)),
            was_truncated=len(raw_input_ids) > len(input_ids),
            question=question,
            metadata=metadata,
            segment_token_stats=segment_token_stats,
        )

    @classmethod
    def _qrecc_current_utterance(cls, row: dict[str, Any]) -> str:
        turn_id = cls._coerce_optional_int(row.get("turn_id"))
        if turn_id == 1:
            rewrite = cls._normalise_text(row.get("rewrite"))
            if rewrite:
                return rewrite
        return cls._normalise_text(row.get("question"))

    @classmethod
    def _qrecc_context_utterances(
        cls,
        row: dict[str, Any],
        turn_id_to_utterance: dict[int, str],
    ) -> list[str]:
        context = row.get("context") or []
        if not isinstance(context, (list, tuple)):
            return []

        utterances: list[str] = []
        for index, item in enumerate(context):
            text = cls._normalise_text(item)
            if index % 2 == 0:
                history_turn_id = int(index / 2) + 1
                text = turn_id_to_utterance.get(history_turn_id, text)
            utterances.append(text)
        return utterances

    @classmethod
    def _has_positive_ctx_passage_ids(cls, row: dict[str, Any]) -> bool:
        value = row.get("positive_ctx_passage_ids")
        if value is None:
            return False
        try:
            if value != value:
                return False
        except TypeError:
            pass
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, tuple, set)):
            return bool(value)
        return bool(value)


def load_itercqr_tokenizer(model_dir: Path | str | None = None) -> Any:
    try:
        from transformers import T5Tokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers is required to load the IterCQR tokenizer.") from exc

    resolved_model_dir = validate_itercqr_checkpoint(model_dir)
    return T5Tokenizer.from_pretrained(resolved_model_dir, local_files_only=True, legacy=True)


def load_itercqr_t5_for_inference(
    model_dir: Path | str | None = None,
    device: Any | None = None,
) -> tuple[Any, Any, Any]:
    if torch is None:
        raise RuntimeError("PyTorch is required to load the IterCQR model.")
    try:
        from transformers import T5ForConditionalGeneration
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers is required to load the IterCQR model.") from exc

    resolved_model_dir = validate_itercqr_checkpoint(model_dir)

    resolved_device = torch.device(device) if device is not None else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    tokenizer = load_itercqr_tokenizer(resolved_model_dir)
    model = T5ForConditionalGeneration.from_pretrained(resolved_model_dir, local_files_only=True)
    model.to(resolved_device)
    model.eval()
    model.generation_config.length_penalty = 0
    return model, tokenizer, resolved_device


@dataclass(frozen=True)
class IterCQRRewriteConfig:
    max_length: int = 32
    num_beams: int = 1
    do_sample: bool = False


class IterCQRRewritePipeline:
    """Small stackable IterCQR T5 subpipeline."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        device: Any,
        generation_config: IterCQRRewriteConfig | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.generation_config = generation_config or IterCQRRewriteConfig()

    @classmethod
    def from_checkpoint(
        cls,
        *,
        model_dir: Path | str | None = None,
        device: Any | None = None,
        generation_config: IterCQRRewriteConfig | None = None,
    ) -> "IterCQRRewritePipeline":
        model, tokenizer, resolved_device = load_itercqr_t5_for_inference(
            model_dir=model_dir,
            device=device,
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            generation_config=generation_config,
        )

    def rewrite_batch(self, batch: dict[str, Any]) -> list[str]:
        if torch is None:
            raise RuntimeError("PyTorch is required for IterCQR generation.")
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=batch["bt_input_ids"].to(self.device),
                attention_mask=batch["bt_attention_mask"].to(self.device),
                do_sample=self.generation_config.do_sample,
                max_length=self.generation_config.max_length,
                num_beams=self.generation_config.num_beams,
                num_return_sequences=1,
            )
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    def rewrite_greedy_and_samples_batch(
        self,
        batch: dict[str, Any],
        *,
        sample_ids: Sequence[str],
        num_samples: int = 2,
        base_sampling_seed: int = 13,
        sampling_temperature: float = 1.0,
        sampling_top_k: int = 50,
        sampling_top_p: float = 1.0,
        batch_metrics_reporter: (
            Callable[[Mapping[str, Any]], None] | None
        ) = None,
    ) -> dict[str, list[str]]:
        """Decode one greedy and query-deterministic sampled rewrites."""

        if torch is None:
            raise RuntimeError("PyTorch is required for IterCQR generation.")
        if int(num_samples) < 1:
            raise ValueError("num_samples must be positive.")
        if float(sampling_temperature) <= 0.0:
            raise ValueError("sampling_temperature must be positive.")
        if int(sampling_top_k) < 0:
            raise ValueError("sampling_top_k must be nonnegative.")
        if not 0.0 < float(sampling_top_p) <= 1.0:
            raise ValueError("sampling_top_p must lie in (0, 1].")

        input_ids = batch["bt_input_ids"].to(self.device)
        attention_mask = batch["bt_attention_mask"].to(self.device)
        ordered_ids = [str(sample_id) for sample_id in sample_ids]
        if len(ordered_ids) != int(input_ids.shape[0]):
            raise ValueError("sample_ids and IterCQR batch size differ.")
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError("sample_ids must be unique within a batch.")

        try:
            from transformers import LogitsProcessor, LogitsProcessorList
            from transformers.generation.logits_process import (
                TemperatureLogitsWarper,
                TopKLogitsWarper,
                TopPLogitsWarper,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "transformers is required for IterCQR generation."
            ) from exc

        device = torch.device(self.device)
        measure_batch_metrics = batch_metrics_reporter is not None

        class PerSequenceSeededSampler(LogitsProcessor):
            """Sample each expanded sequence from its own RNG stream."""

            def __init__(self, seeds: Sequence[int]) -> None:
                self.generators = []
                for seed in seeds:
                    generator = torch.Generator(device=device)
                    generator.manual_seed(int(seed))
                    self.generators.append(generator)
                self.warpers = LogitsProcessorList()
                if float(sampling_temperature) != 1.0:
                    self.warpers.append(
                        TemperatureLogitsWarper(
                            float(sampling_temperature)
                        )
                    )
                if int(sampling_top_k) > 0:
                    self.warpers.append(
                        TopKLogitsWarper(int(sampling_top_k))
                    )
                if float(sampling_top_p) < 1.0:
                    self.warpers.append(
                        TopPLogitsWarper(float(sampling_top_p))
                    )

            def __call__(
                self,
                generated_ids: Any,
                scores: Any,
            ) -> Any:
                if int(scores.shape[0]) != len(self.generators):
                    raise RuntimeError(
                        "Expanded generation batch and RNG count differ."
                    )
                warped = self.warpers(generated_ids, scores)
                probabilities = torch.softmax(warped, dim=-1)
                sampled = torch.cat(
                    [
                        torch.multinomial(
                            probabilities[index : index + 1],
                            num_samples=1,
                            generator=generator,
                        )
                        for index, generator in enumerate(
                            self.generators
                        )
                    ],
                    dim=0,
                )
                forced = torch.full_like(scores, -torch.inf)
                forced.scatter_(1, sampled, 0.0)
                return forced

        def synchronize() -> None:
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        with torch.no_grad():
            if measure_batch_metrics:
                synchronize()
            if measure_batch_metrics and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            encoder_started = (
                time.perf_counter() if measure_batch_metrics else None
            )
            encoder_outputs = self.model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            if measure_batch_metrics:
                synchronize()
            encoder_seconds = (
                time.perf_counter() - encoder_started
                if encoder_started is not None
                else None
            )

            greedy_started = (
                time.perf_counter() if measure_batch_metrics else None
            )
            greedy_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                do_sample=False,
                max_length=self.generation_config.max_length,
                num_beams=self.generation_config.num_beams,
                num_return_sequences=1,
            )
            if measure_batch_metrics:
                synchronize()
            greedy_seconds = (
                time.perf_counter() - greedy_started
                if greedy_started is not None
                else None
            )
            result = {
                "greedy": self.tokenizer.batch_decode(
                    greedy_ids,
                    skip_special_tokens=True,
                )
            }
            for candidate_index in range(1, int(num_samples) + 1):
                result[f"sample_{candidate_index}"] = []

            samples_started = (
                time.perf_counter() if measure_batch_metrics else None
            )
            sampling_seeds = [
                derive_query_seed(
                    int(base_sampling_seed),
                    sample_id,
                    f"itercqr_sample_{candidate_index}",
                )
                for sample_id in ordered_ids
                for candidate_index in range(
                    1,
                    int(num_samples) + 1,
                )
            ]
            sampled_ids = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=1.0,
                top_k=0,
                top_p=1.0,
                max_length=self.generation_config.max_length,
                num_beams=1,
                num_return_sequences=int(num_samples),
                logits_processor=LogitsProcessorList(
                    [PerSequenceSeededSampler(sampling_seeds)]
                ),
            )
            decoded_samples = self.tokenizer.batch_decode(
                sampled_ids,
                skip_special_tokens=True,
            )
            for candidate_index in range(int(num_samples)):
                result[f"sample_{candidate_index + 1}"] = (
                    decoded_samples[
                        candidate_index::int(num_samples)
                    ]
                )
            if measure_batch_metrics:
                synchronize()
            samples_seconds = (
                time.perf_counter() - samples_started
                if samples_started is not None
                else None
            )
            if batch_metrics_reporter is not None:
                assert encoder_seconds is not None
                assert greedy_seconds is not None
                assert samples_seconds is not None
                batch_metrics_reporter(
                    {
                        "batch_size": int(len(ordered_ids)),
                        "num_samples": int(num_samples),
                        "encoder_seconds": float(encoder_seconds),
                        "greedy_decode_seconds": float(
                            greedy_seconds
                        ),
                        "sample_decode_seconds": float(
                            samples_seconds
                        ),
                        "encoder_forward_calls": 1,
                        "decoder_sequences": int(
                            len(ordered_ids) * (1 + num_samples)
                        ),
                        "cuda_peak_allocated_bytes": (
                            int(torch.cuda.max_memory_allocated(device))
                            if device.type == "cuda"
                            else None
                        ),
                        "cuda_peak_reserved_bytes": (
                            int(torch.cuda.max_memory_reserved(device))
                            if device.type == "cuda"
                            else None
                        ),
                    }
                )
        return result

    def rewrite_dataloader(
        self,
        dataloader: Any,
        *,
        max_batches: int | None = None,
    ) -> Any:
        import pandas as pd

        rows = []
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            rewrites = self.rewrite_batch(batch)
            for row_index, rewrite in enumerate(rewrites):
                metadata = batch["bt_metadata"][row_index]
                rows.append(
                    {
                        "sample_id": batch["bt_sample_ids"][row_index],
                        "question": batch["bt_questions"][row_index],
                        "rewrite": rewrite,
                        "input_length": int(batch["bt_input_lengths"][row_index]),
                        "split": metadata.get("split"),
                        "conv_id": metadata.get("conv_id"),
                        "turn_id": metadata.get("turn_id"),
                        "positive_ctx_passage_ids": metadata.get("positive_ctx_passage_ids", []),
                    }
                )
        return pd.DataFrame(rows)


def load_itercqr_rewriter(
    *,
    model_dir: Path | str | None = None,
    device: Any | None = None,
    generation_config: IterCQRRewriteConfig | None = None,
) -> IterCQRRewritePipeline:
    return IterCQRRewritePipeline.from_checkpoint(
        model_dir=model_dir,
        device=device,
        generation_config=generation_config,
    )
