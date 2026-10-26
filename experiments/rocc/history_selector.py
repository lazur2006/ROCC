"""MiniLM encoder + CRF history selector training and inference."""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF
from transformers import (
    AutoModel,
    get_linear_schedule_with_warmup,
)

from .progress import get_tqdm, progress_iter


LABEL_TO_ID = {"O": 0, "B-KEEP": 1, "I-KEEP": 2}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}
TAXONOMY_LABEL_TO_ID = {
    "O": 0,
    "ENTITY": 1,
    "CONCEPT": 2,
    "KEY_TERM": 3,
    "DEFINITION": 4,
    "RELATION_CUE": 5,
}
TAXONOMY_ID_TO_LABEL = {
    value: key for key, value in TAXONOMY_LABEL_TO_ID.items()
}
KEEP_LABELS = {
    "ENTITY",
    "CONCEPT",
    "KEY_TERM",
    "DEFINITION",
    "RELATION_CUE",
}


@dataclass(frozen=True)
class HistorySelectorConfig:
    """Shared query/history encoding and optimization configuration."""

    model_name: str = "microsoft/MiniLM-L12-H384-uncased"
    max_length: int = 512
    history_order: str = "recent_first"
    epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    seed: int = 13
    num_workers: int = 0
    crf_loss_weight: float = 1.0
    token_loss_weight: float = 1.0
    positive_labels: tuple[str, ...] = tuple(sorted(KEEP_LABELS))

    def __post_init__(self) -> None:
        if self.history_order not in {"recent_first", "chronological"}:
            raise ValueError(f"Unknown history order: {self.history_order}")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")


@dataclass(frozen=True)
class EncodedSelectorExample:
    sample_id: str
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    history_mask: list[int]
    history_len: int


@dataclass(frozen=True)
class EncodedHistoryPair:
    """Shared tokenization of one current-query/history pair."""

    sample_id: str
    history_text: str
    field_offsets: dict[tuple[int, str], int]
    field_ranges: dict[tuple[int, str], tuple[int, int]]
    input_ids: list[int]
    attention_mask: list[int]
    offset_mapping: list[tuple[int, int]]
    sequence_ids: list[int | None]
    history_mask: list[int]
    history_offsets: list[tuple[int, int]]


@dataclass(frozen=True)
class SelectorTrainingResult:
    output_dir: Path
    checkpoint: Path
    metrics_path: Path
    best_metrics: dict[str, Any]
    history: list[dict[str, Any]]
    duration_seconds: float


class HistorySelectorDataset(Dataset[EncodedSelectorExample]):
    """Tokenized query/history pairs with BIO labels on history tokens."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        config: HistorySelectorConfig,
        progress: bool = True,
        progress_desc: str = "encode selector data",
    ) -> None:
        positive_labels = set(config.positive_labels)
        tqdm = get_tqdm()
        self.examples = [
            encode_selector_row(
                row,
                tokenizer,
                max_length=config.max_length,
                positive_labels=positive_labels,
                history_order=config.history_order,
            )
            for row in tqdm(
                rows,
                total=len(rows),
                desc=progress_desc,
                unit="query",
                dynamic_ncols=True,
                disable=not progress,
            )
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedSelectorExample:
        return self.examples[index]


class EncodedSelectorDataset(Dataset[EncodedSelectorExample]):
    """Dataset wrapper for already materialized selector encodings."""

    def __init__(
        self,
        examples: Sequence[EncodedSelectorExample],
    ) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedSelectorExample:
        return self.examples[index]


class TaxonomyHistorySelectorDataset(
    Dataset[EncodedSelectorExample]
):
    """Tokenized query/history pairs with six-way taxonomy labels."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        max_length: int = 512,
        history_order: str = "recent_first",
        progress: bool = True,
        progress_desc: str = "encode taxonomy data",
    ) -> None:
        tqdm = get_tqdm()
        self.examples = [
            encode_taxonomy_row(
                row,
                tokenizer,
                max_length=max_length,
                history_order=history_order,
            )
            for row in tqdm(
                rows,
                total=len(rows),
                desc=progress_desc,
                unit="query",
                dynamic_ncols=True,
                disable=not progress,
            )
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> EncodedSelectorExample:
        return self.examples[index]


def read_label_jsonl(
    path: Path | str,
    *,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Read a label JSONL with byte-level progress."""

    resolved = Path(path)
    rows: list[dict[str, Any]] = []
    total = resolved.stat().st_size
    tqdm = get_tqdm()
    with resolved.open("rb") as handle, tqdm(
        total=total,
        desc=f"load {resolved.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        disable=not progress,
    ) as bar:
        for line in handle:
            bar.update(len(line))
            if line.strip():
                rows.append(json.loads(line))
    return rows


def seed_selector_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_history_text(
    row: dict[str, Any],
    *,
    history_order: str,
) -> tuple[str, dict[tuple[int, str], int]]:
    """Render history exactly as used by the original selector."""

    parts: list[str] = []
    field_offsets: dict[tuple[int, str], int] = {}
    cursor = 0
    turns = list(row.get("history", []))
    if history_order == "recent_first":
        turns.reverse()
    for turn in turns:
        turn_id = int(turn["turn_id"])
        prefix = f"T{turn_id} Q: "
        parts.append(prefix)
        cursor += len(prefix)
        field_offsets[(turn_id, "question")] = cursor
        question = str(turn.get("question", ""))
        parts.append(question)
        cursor += len(question)

        middle = "\nA: "
        parts.append(middle)
        cursor += len(middle)
        field_offsets[(turn_id, "answer")] = cursor
        answer = str(turn.get("answer", ""))
        parts.append(answer)
        cursor += len(answer)

        parts.append("\n\n")
        cursor += 2
    return "".join(parts), field_offsets


def positive_char_spans(
    row: dict[str, Any],
    field_offsets: dict[tuple[int, str], int],
    positive_labels: set[str],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for span in row.get("spans", []):
        if str(span.get("label")) not in positive_labels:
            continue
        key = (int(span["turn_id"]), str(span["field"]))
        if key not in field_offsets:
            continue
        start = field_offsets[key] + int(span["char_start"])
        end = field_offsets[key] + int(span["char_end"])
        if end > start:
            spans.append((start, end))
    return sorted(spans)


def encode_selector_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    positive_labels: set[str],
    history_order: str,
) -> EncodedSelectorExample:
    pair = encode_history_pair(
        row,
        tokenizer,
        max_length=max_length,
        history_order=history_order,
    )
    return project_collapsed_selector_row(
        row,
        pair,
        positive_labels=positive_labels,
    )


def project_collapsed_selector_row(
    row: dict[str, Any],
    pair: EncodedHistoryPair,
    *,
    positive_labels: set[str],
) -> EncodedSelectorExample:
    """Project one cached pair to O/B-KEEP/I-KEEP labels."""

    if pair.sample_id != str(row["sample_id"]):
        raise ValueError("Pair and row sample IDs do not match.")
    selected_spans = positive_char_spans(
        row,
        pair.field_offsets,
        positive_labels,
    )
    labels: list[int] = []
    history_mask: list[int] = []
    previous_span_index: int | None = None
    for offset, sequence_id in zip(
        pair.offset_mapping,
        pair.sequence_ids,
        strict=True,
    ):
        if sequence_id != 1 or offset == (0, 0):
            labels.append(-100)
            history_mask.append(0)
            previous_span_index = None
            continue
        token_start, token_end = map(int, offset)
        matching_span = None
        for span_index, (span_start, span_end) in enumerate(
            selected_spans
        ):
            if token_start < span_end and token_end > span_start:
                matching_span = span_index
                break
        history_mask.append(1)
        if matching_span is None:
            labels.append(LABEL_TO_ID["O"])
            previous_span_index = None
        elif matching_span == previous_span_index:
            labels.append(LABEL_TO_ID["I-KEEP"])
        else:
            labels.append(LABEL_TO_ID["B-KEEP"])
            previous_span_index = matching_span
    return EncodedSelectorExample(
        sample_id=str(row["sample_id"]),
        input_ids=pair.input_ids,
        attention_mask=pair.attention_mask,
        labels=labels,
        history_mask=history_mask,
        history_len=int(row["history_len"]),
    )


def encode_history_pair(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    history_order: str,
) -> EncodedHistoryPair:
    """Tokenize the canonical query/history pair once for all label tasks."""

    history_text, field_offsets = build_history_text(
        row,
        history_order=history_order,
    )
    turns = {
        int(turn["turn_id"]): turn
        for turn in row.get("history", [])
    }
    field_ranges = {
        key: (
            start,
            start + len(str(turns[key[0]].get(key[1], ""))),
        )
        for key, start in field_offsets.items()
    }
    encoding = tokenizer(
        str(row["current_query"]),
        history_text,
        max_length=max_length,
        truncation="only_second",
        return_offsets_mapping=True,
    )
    sequence_ids = list(encoding.sequence_ids())
    offset_mapping = [
        (int(offset[0]), int(offset[1]))
        for offset in encoding["offset_mapping"]
    ]
    history_mask = [
        int(sequence_id == 1 and offset != (0, 0))
        for offset, sequence_id in zip(
            offset_mapping,
            sequence_ids,
            strict=True,
        )
    ]
    history_offsets = [
        offset
        for offset, active in zip(
            offset_mapping,
            history_mask,
            strict=True,
        )
        if active
    ]
    return EncodedHistoryPair(
        sample_id=str(row["sample_id"]),
        history_text=history_text,
        field_offsets=field_offsets,
        field_ranges=field_ranges,
        input_ids=list(encoding["input_ids"]),
        attention_mask=list(encoding["attention_mask"]),
        offset_mapping=offset_mapping,
        sequence_ids=sequence_ids,
        history_mask=history_mask,
        history_offsets=history_offsets,
    )


def encode_taxonomy_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    history_order: str,
) -> EncodedSelectorExample:
    """Project validated Teacher spans to six taxonomy token classes."""

    pair = encode_history_pair(
        row,
        tokenizer,
        max_length=max_length,
        history_order=history_order,
    )
    return project_taxonomy_selector_row(row, pair)


def project_taxonomy_selector_row(
    row: dict[str, Any],
    pair: EncodedHistoryPair,
) -> EncodedSelectorExample:
    """Project one cached pair to the six-class Teacher taxonomy."""

    if pair.sample_id != str(row["sample_id"]):
        raise ValueError("Pair and row sample IDs do not match.")
    spans: list[tuple[int, int, int, int]] = []
    for source_index, span in enumerate(row.get("spans", [])):
        label = str(span.get("label", ""))
        key = (int(span["turn_id"]), str(span["field"]))
        if (
            label not in TAXONOMY_LABEL_TO_ID
            or label == "O"
            or key not in pair.field_offsets
        ):
            continue
        start = pair.field_offsets[key] + int(span["char_start"])
        end = pair.field_offsets[key] + int(span["char_end"])
        if end <= start:
            continue
        spans.append(
            (
                int(span.get("span_id", source_index)),
                source_index,
                start,
                end,
            )
        )
    spans.sort(key=lambda value: (value[0], value[1]))

    labels: list[int] = []
    for offset, sequence_id in zip(
        pair.offset_mapping,
        pair.sequence_ids,
        strict=True,
    ):
        if sequence_id != 1 or offset == (0, 0):
            labels.append(-100)
            continue
        token_start, token_end = offset
        selected_label = "O"
        for _, source_index, span_start, span_end in spans:
            if token_start < span_end and token_end > span_start:
                selected_label = str(
                    row["spans"][source_index]["label"]
                )
                break
        labels.append(TAXONOMY_LABEL_TO_ID[selected_label])

    return EncodedSelectorExample(
        sample_id=str(row["sample_id"]),
        input_ids=pair.input_ids,
        attention_mask=pair.attention_mask,
        labels=labels,
        history_mask=pair.history_mask,
        history_len=int(row["history_len"]),
    )


def collate_selector_batch(
    batch: Sequence[EncodedSelectorExample],
    pad_token_id: int,
) -> dict[str, Any]:
    max_length = max(len(example.input_ids) for example in batch)

    def pad(values: list[int], pad_value: int) -> list[int]:
        return values + [pad_value] * (max_length - len(values))

    return {
        "sample_ids": [example.sample_id for example in batch],
        "input_ids": torch.tensor(
            [pad(example.input_ids, pad_token_id) for example in batch],
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            [
                pad(example.attention_mask, 0)
                for example in batch
            ],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [pad(example.labels, -100) for example in batch],
            dtype=torch.long,
        ),
        "history_mask": torch.tensor(
            [pad(example.history_mask, 0) for example in batch],
            dtype=torch.bool,
        ),
        "history_len": torch.tensor(
            [example.history_len for example in batch],
            dtype=torch.long,
        ),
    }


class EncoderLinearHistorySelector(nn.Module):
    """Query-conditioned encoder with a linear history-token decoder."""

    def __init__(
        self,
        model_name: str,
        *,
        num_labels: int,
        class_weights: Sequence[float],
        token_loss_weight: float = 1.0,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(model_name)
        )
        hidden_size = int(self.encoder.config.hidden_size)
        dropout_probability = float(
            getattr(
                self.encoder.config,
                "hidden_dropout_prob",
                0.1,
            )
        )
        if len(class_weights) != int(num_labels):
            raise ValueError(
                "class_weights length must match num_labels."
            )
        self.dropout = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(hidden_size, int(num_labels))
        self.token_loss_weight = float(token_loss_weight)
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        history_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        emissions = self.classifier(
            self.dropout(outputs.last_hidden_state)
        )
        sequence_emissions, sequence_mask, _ = (
            gather_history_sequences(
                emissions,
                history_mask,
                labels,
            )
        )
        decoded = sequence_emissions.argmax(dim=-1)
        result: dict[str, Any] = {
            "emissions": emissions,
            "decoded": [
                decoded[index, : int(mask.sum().item())]
                .detach()
                .cpu()
                .tolist()
                for index, mask in enumerate(sequence_mask)
            ],
            "sequence_mask": sequence_mask,
        }
        if labels is not None:
            token_loss = nn.functional.cross_entropy(
                emissions.reshape(-1, emissions.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                weight=self.class_weights.to(emissions.device),
            )
            result["loss"] = self.token_loss_weight * token_loss
            result["token_loss"] = token_loss
        return result


class EncoderCrfHistorySelector(nn.Module):
    """Query-conditioned MiniLM token classifier with CRF decoding."""

    def __init__(
        self,
        model_name: str,
        *,
        num_labels: int,
        class_weights: Sequence[float],
        crf_loss_weight: float,
        token_loss_weight: float,
        encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = (
            encoder
            if encoder is not None
            else AutoModel.from_pretrained(model_name)
        )
        hidden_size = int(self.encoder.config.hidden_size)
        dropout_probability = float(
            getattr(
                self.encoder.config,
                "hidden_dropout_prob",
                0.1,
            )
        )
        if int(num_labels) < 2:
            raise ValueError("num_labels must be at least 2.")
        if len(class_weights) != int(num_labels):
            raise ValueError(
                "class_weights length must match num_labels."
            )
        self.dropout = nn.Dropout(dropout_probability)
        self.classifier = nn.Linear(hidden_size, int(num_labels))
        self.crf = CRF(int(num_labels), batch_first=True)
        self.crf_loss_weight = float(crf_loss_weight)
        self.token_loss_weight = float(token_loss_weight)
        self.register_buffer(
            "class_weights",
            torch.tensor(class_weights, dtype=torch.float),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        history_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        emissions = self.classifier(
            self.dropout(outputs.last_hidden_state)
        )
        sequence_emissions, sequence_mask, sequence_labels = (
            gather_history_sequences(
                emissions,
                history_mask,
                labels,
            )
        )
        result: dict[str, Any] = {"emissions": emissions}
        if labels is not None:
            log_likelihood = self.crf(
                sequence_emissions,
                sequence_labels,
                mask=sequence_mask.bool(),
                reduction="none",
            )
            token_counts = sequence_mask.sum(dim=1).clamp_min(1)
            crf_loss = (-log_likelihood / token_counts).mean()
            token_loss = nn.functional.cross_entropy(
                emissions.reshape(-1, emissions.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                weight=self.class_weights.to(emissions.device),
            )
            result["loss"] = (
                self.crf_loss_weight * crf_loss
                + self.token_loss_weight * token_loss
            )
            result["imitation_loss"] = result["loss"]
            result["crf_loss"] = crf_loss
            result["token_loss"] = token_loss
        result["decoded"] = self.crf.decode(
            sequence_emissions,
            mask=sequence_mask.bool(),
        )
        result["sequence_mask"] = sequence_mask
        return result


def inject_lora_encoder(
    encoder: nn.Module,
    *,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
) -> tuple[nn.Module, tuple[str, ...]]:
    """Inject PEFT LoRA adapters into every encoder Linear module."""

    from peft import LoraConfig, TaskType, get_peft_model

    target_modules = tuple(
        name
        for name, module in encoder.named_modules()
        if name and isinstance(module, nn.Linear)
    )
    if not target_modules:
        raise ValueError("The encoder contains no Linear target modules.")
    configuration = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        target_modules=list(target_modules),
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    return get_peft_model(encoder, configuration), target_modules


def load_selector_encoder_state(
    encoder: nn.Module,
    checkpoint: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load only the encoder part of a selector checkpoint strictly."""

    state = torch.load(
        Path(checkpoint),
        map_location=map_location,
        weights_only=True,
    )
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise ValueError(
            f"Checkpoint contains no encoder state: {checkpoint}"
        )
    encoder.load_state_dict(encoder_state, strict=True)


def selector_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Return total and trainable selector parameter counts."""

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


def gather_history_sequences(
    emissions: torch.Tensor,
    history_mask: torch.Tensor,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_emissions: list[torch.Tensor] = []
    batch_labels: list[torch.Tensor] = []
    lengths: list[int] = []
    for batch_index in range(emissions.size(0)):
        active = history_mask[batch_index].bool()
        selected_emissions = emissions[batch_index][active]
        if selected_emissions.size(0) == 0:
            selected_emissions = emissions[batch_index, :1]
        batch_emissions.append(selected_emissions)
        lengths.append(selected_emissions.size(0))
        if labels is not None:
            selected_labels = labels[batch_index][active]
            if selected_labels.size(0) == 0:
                selected_labels = labels.new_zeros(1)
            batch_labels.append(torch.clamp(selected_labels, min=0))

    maximum_length = max(lengths)
    number_of_tags = emissions.size(-1)
    padded_emissions = emissions.new_zeros(
        len(batch_emissions),
        maximum_length,
        number_of_tags,
    )
    padded_mask = torch.zeros(
        len(batch_emissions),
        maximum_length,
        dtype=torch.bool,
        device=emissions.device,
    )
    padded_labels = torch.zeros(
        len(batch_emissions),
        maximum_length,
        dtype=torch.long,
        device=emissions.device,
    )
    for index, selected_emissions in enumerate(batch_emissions):
        length = selected_emissions.size(0)
        padded_emissions[index, :length] = selected_emissions
        padded_mask[index, :length] = True
        if labels is not None:
            padded_labels[index, :length] = batch_labels[index]
    return padded_emissions, padded_mask, padded_labels


def selector_label_stats(
    dataset: HistorySelectorDataset,
) -> dict[str, Any]:
    counts = {label: 0 for label in LABEL_TO_ID}
    history_tokens = 0
    examples_with_keep = 0
    for example in dataset.examples:
        has_keep = False
        for label in example.labels:
            if label < 0:
                continue
            counts[ID_TO_LABEL[int(label)]] += 1
            history_tokens += 1
            has_keep = has_keep or int(label) != LABEL_TO_ID["O"]
        examples_with_keep += int(has_keep)
    keep_tokens = counts["B-KEEP"] + counts["I-KEEP"]
    return {
        "examples": len(dataset),
        "examples_with_keep": examples_with_keep,
        "history_tokens": history_tokens,
        "label_counts": counts,
        "keep_tokens": keep_tokens,
        "keep_token_rate": (
            keep_tokens / history_tokens if history_tokens else 0.0
        ),
    }


def selector_class_stats(
    dataset: Dataset[EncodedSelectorExample],
    *,
    label_to_id: Mapping[str, int],
) -> dict[str, Any]:
    """Count labeled history tokens for an arbitrary selector label space."""

    id_to_label = {
        int(label_id): str(label)
        for label, label_id in label_to_id.items()
    }
    counts = {label: 0 for label in label_to_id}
    history_tokens = 0
    for example in dataset:
        for label_id in example.labels:
            if int(label_id) < 0:
                continue
            counts[id_to_label[int(label_id)]] += 1
            history_tokens += 1
    return {
        "examples": len(dataset),
        "history_tokens": history_tokens,
        "label_counts": counts,
    }


def balanced_selector_class_weights(
    dataset: Dataset[EncodedSelectorExample],
    *,
    label_to_id: Mapping[str, int],
) -> tuple[float, ...]:
    """Return N / (K * n_c) weights from the active training split."""

    stats = selector_class_stats(
        dataset,
        label_to_id=label_to_id,
    )
    counts = stats["label_counts"]
    missing = [
        label for label, count in counts.items() if int(count) == 0
    ]
    if missing:
        raise ValueError(
            "Cannot balance absent classes: "
            + ", ".join(missing)
        )
    total = int(stats["history_tokens"])
    number_of_classes = len(label_to_id)
    by_id = {
        int(label_id): total / (
            number_of_classes * int(counts[label])
        )
        for label, label_id in label_to_id.items()
    }
    return tuple(by_id[index] for index in range(number_of_classes))


def fit_selector_model(
    *,
    model: nn.Module,
    train_loader: DataLoader,
    output_dir: Path | str,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    evaluate_fn: (
        Callable[[nn.Module, int], dict[str, Any]] | None
    ) = None,
    checkpoint_policy: str = "best",
    selection_metric: str | None = "keep_f1",
    maximize_metric: bool = True,
    progress: bool = True,
    progress_desc: str = "train selector",
    gradient_clip_norm: float = 1.0,
    run_config: Mapping[str, Any] | None = None,
    train_stats: Mapping[str, Any] | None = None,
    dev_stats: Mapping[str, Any] | None = None,
    checkpoint_name: str = "model.pt",
    metrics_name: str = "metrics.json",
    epoch_checkpoint_pattern: str | None = None,
    record_learning_rate: bool = False,
    resume: bool = False,
    started_at: float | None = None,
    epoch_reporter: (
        Callable[
            [int, int, float, dict[str, Any], bool],
            str,
        ]
        | None
    ) = None,
    batch_metrics_reporter: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
    epoch_metrics_reporter: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
) -> SelectorTrainingResult:
    """Fit any selector model with the canonical optimizer and loop."""

    if epochs < 1:
        raise ValueError("epochs must be positive.")
    if checkpoint_policy not in {"best", "last"}:
        raise ValueError(
            "checkpoint_policy must be 'best' or 'last'."
        )
    if checkpoint_policy == "best" and (
        evaluate_fn is None or selection_metric is None
    ):
        raise ValueError(
            "Best-checkpoint selection requires evaluation and a metric."
        )

    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    checkpoint = resolved_output / checkpoint_name
    metrics_path = resolved_output / metrics_name
    resume_path = resolved_output / "training_state.pt"
    started = started_at if started_at is not None else time.time()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * warmup_ratio),
        total_steps,
    )

    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_value = -math.inf if maximize_metric else math.inf
    start_epoch = 1
    if resume and resume_path.exists():
        state = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = list(state.get("history", []))
        best_metrics = state.get("best_metrics")
        best_value = float(state.get("best_value", best_value))
        start_epoch = int(state["epoch"]) + 1
        if history:
            started = time.time() - float(
                history[-1].get("elapsed_seconds", 0.0)
            )
        if start_epoch > epochs:
            if best_metrics is None or not checkpoint.exists():
                raise RuntimeError(
                    "Completed training state has no usable checkpoint."
                )
            duration = float(
                json.loads(metrics_path.read_text(encoding="utf-8")).get(
                    "duration_seconds",
                    history[-1].get("elapsed_seconds", 0.0),
                )
                if metrics_path.exists()
                else history[-1].get("elapsed_seconds", 0.0)
            )
            return SelectorTrainingResult(
                output_dir=resolved_output,
                checkpoint=checkpoint,
                metrics_path=metrics_path,
                best_metrics=dict(best_metrics),
                history=history,
                duration_seconds=duration,
            )

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_selector_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch=epoch,
            progress=progress,
            progress_desc=progress_desc,
            gradient_clip_norm=gradient_clip_norm,
            global_step_offset=(epoch - 1) * len(train_loader),
            metrics_reporter=batch_metrics_reporter,
        )
        metrics = (
            dict(evaluate_fn(model, epoch))
            if evaluate_fn is not None
            else {}
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        if record_learning_rate:
            metrics["learning_rate"] = float(
                optimizer.param_groups[0]["lr"]
            )
        metrics["elapsed_seconds"] = time.time() - started
        history.append(metrics)

        if epoch_metrics_reporter is not None:
            epoch_metrics_reporter(dict(metrics))

        improved = checkpoint_policy == "last"
        if checkpoint_policy == "best":
            candidate = float(metrics[str(selection_metric)])
            improved = (
                candidate > best_value
                if maximize_metric
                else candidate < best_value
            )
            if improved:
                best_value = candidate
        if improved:
            best_metrics = dict(metrics)
            torch.save(model.state_dict(), checkpoint)
        if epoch_checkpoint_pattern is not None:
            epoch_checkpoint = resolved_output / (
                epoch_checkpoint_pattern.format(epoch=epoch)
            )
            torch.save(model.state_dict(), epoch_checkpoint)

        _write_training_metrics(
            path=metrics_path,
            config=dict(run_config or {}),
            train_stats=dict(train_stats or {}),
            dev_stats=dict(dev_stats or {}),
            history=history,
            best_metrics=best_metrics,
            duration_seconds=time.time() - started,
        )
        if resume:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "history": history,
                    "best_metrics": best_metrics,
                    "best_value": best_value,
                },
                resume_path,
            )

        report = (
            epoch_reporter(
                epoch,
                epochs,
                train_loss,
                metrics,
                improved,
            )
            if epoch_reporter is not None
            else (
                f"epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}"
            )
        )
        print(report, flush=True)

    duration = time.time() - started
    if best_metrics is None or not checkpoint.exists():
        raise RuntimeError("Training produced no checkpoint.")
    _write_training_metrics(
        path=metrics_path,
        config=dict(run_config or {}),
        train_stats=dict(train_stats or {}),
        dev_stats=dict(dev_stats or {}),
        history=history,
        best_metrics=best_metrics,
        duration_seconds=duration,
    )
    return SelectorTrainingResult(
        output_dir=resolved_output,
        checkpoint=checkpoint,
        metrics_path=metrics_path,
        best_metrics=best_metrics,
        history=history,
        duration_seconds=duration,
    )


def train_selector_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    *,
    epoch: int,
    progress: bool,
    progress_desc: str = "train selector",
    gradient_clip_norm: float = 1.0,
    global_step_offset: int = 0,
    metrics_reporter: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
) -> float:
    model.train()
    losses: list[float] = []
    tqdm = get_tqdm()
    bar = tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"{progress_desc} epoch {epoch}",
        unit="batch",
        dynamic_ncols=True,
        disable=not progress,
    )
    for batch_index, batch in enumerate(bar, start=1):
        optimizer.zero_grad(set_to_none=True)
        moved = _move_batch(batch, device)
        model_inputs = {
            "input_ids": moved["input_ids"],
            "attention_mask": moved["attention_mask"],
            "history_mask": moved["history_mask"],
            "labels": moved["labels"],
        }
        output = model(
            **model_inputs,
        )
        loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            gradient_clip_norm,
        )
        optimizer.step()
        scheduler.step()
        value = float(loss.detach().cpu())
        losses.append(value)
        running_mean = float(np.mean(losses))
        if metrics_reporter is not None:
            payload = {
                "epoch": int(epoch),
                "batch": int(batch_index),
                "global_step": int(
                    global_step_offset + batch_index
                ),
                "loss": value,
                "running_mean": running_mean,
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
            }
            metrics_reporter(payload)
        postfix = {
            "loss": f"{value:.4f}",
            "mean": f"{running_mean:.4f}",
        }
        bar.set_postfix(**postfix, refresh=False)
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate_history_selector(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    progress: bool = True,
    progress_desc: str = "evaluate selector",
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    gold_sequences: list[list[int]] = []
    predicted_sequences: list[list[int]] = []
    tqdm = get_tqdm()
    for batch in tqdm(
        dataloader,
        total=len(dataloader),
        desc=progress_desc,
        unit="batch",
        dynamic_ncols=True,
        disable=not progress,
    ):
        moved = _move_batch(batch, device)
        output = model(
            input_ids=moved["input_ids"],
            attention_mask=moved["attention_mask"],
            history_mask=moved["history_mask"],
            labels=moved["labels"],
        )
        losses.append(float(output["loss"].detach().cpu()))
        _, sequence_mask, sequence_labels = gather_history_sequences(
            output["emissions"],
            moved["history_mask"],
            moved["labels"],
        )
        for index, predicted in enumerate(output["decoded"]):
            length = int(sequence_mask[index].sum().item())
            gold_sequences.append(
                sequence_labels[index, :length]
                .detach()
                .cpu()
                .tolist()
            )
            predicted_sequences.append(predicted[:length])

    metrics = _token_metrics(gold_sequences, predicted_sequences)
    metrics.update(_span_metrics(gold_sequences, predicted_sequences))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


@torch.no_grad()
def evaluate_taxonomy_selector(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    *,
    progress: bool = True,
    progress_desc: str = "evaluate taxonomy selector",
) -> dict[str, float]:
    """Evaluate six-way taxonomy labels on compacted history sequences."""

    model.eval()
    losses: list[float] = []
    gold_sequences: list[list[int]] = []
    predicted_sequences: list[list[int]] = []
    tqdm = get_tqdm()
    for batch in tqdm(
        dataloader,
        total=len(dataloader),
        desc=progress_desc,
        unit="batch",
        dynamic_ncols=True,
        disable=not progress,
    ):
        moved = _move_batch(batch, device)
        output = model(
            input_ids=moved["input_ids"],
            attention_mask=moved["attention_mask"],
            history_mask=moved["history_mask"],
            labels=moved["labels"],
        )
        losses.append(float(output["loss"].detach().cpu()))
        _, sequence_mask, sequence_labels = gather_history_sequences(
            output["emissions"],
            moved["history_mask"],
            moved["labels"],
        )
        for index, predicted in enumerate(output["decoded"]):
            length = int(sequence_mask[index].sum().item())
            gold_sequences.append(
                sequence_labels[index, :length]
                .detach()
                .cpu()
                .tolist()
            )
            predicted_sequences.append(list(predicted[:length]))

    metrics: dict[str, float] = {}
    content_f1: list[float] = []
    for label, label_id in TAXONOMY_LABEL_TO_ID.items():
        true_positive = false_positive = false_negative = 0
        for gold, predicted in zip(
            gold_sequences,
            predicted_sequences,
            strict=True,
        ):
            for gold_id, predicted_id in zip(
                gold,
                predicted,
                strict=True,
            ):
                true_positive += int(
                    gold_id == label_id
                    and predicted_id == label_id
                )
                false_positive += int(
                    gold_id != label_id
                    and predicted_id == label_id
                )
                false_negative += int(
                    gold_id == label_id
                    and predicted_id != label_id
                )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        prefix = label.lower()
        metrics[f"{prefix}_precision"] = precision
        metrics[f"{prefix}_recall"] = recall
        metrics[f"{prefix}_f1"] = f1
        if label != "O":
            content_f1.append(f1)
    metrics["taxonomy_macro_f1"] = float(np.mean(content_f1))
    metrics.update(
        _binary_keep_metrics(
            gold_sequences,
            predicted_sequences,
        )
    )
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


@torch.no_grad()
def predict_selected_histories(
    rows: Sequence[dict[str, Any]],
    *,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    label_space: str,
    config: HistorySelectorConfig | None = None,
    batch_size: int = 16,
    progress: bool = True,
    progress_desc: str = "predict selector histories",
) -> dict[str, list[dict[str, Any]]]:
    """Decode exact history substrings selected by one checkpoint."""

    selected_config = config or HistorySelectorConfig()
    tqdm = get_tqdm()
    encoded = [
        _encode_inference_row(
            row,
            tokenizer,
            max_length=selected_config.max_length,
            history_order=selected_config.history_order,
        )
        for row in tqdm(
            rows,
            total=len(rows),
            desc=f"{progress_desc}: encode",
            unit="query",
            dynamic_ncols=True,
            disable=not progress,
        )
    ]
    histories: dict[str, list[dict[str, Any]]] = {}
    model.eval()
    starts = range(0, len(encoded), batch_size)
    for start in tqdm(
        starts,
        total=math.ceil(len(encoded) / batch_size),
        desc=progress_desc,
        unit="batch",
        dynamic_ncols=True,
        disable=not progress,
    ):
        encoded_batch = encoded[start : start + batch_size]
        batch = _collate_inference(
            encoded_batch,
            int(tokenizer.pad_token_id),
            device,
        )
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            history_mask=batch["history_mask"],
        )
        source_batch = rows[start : start + batch_size]
        for row, encoded_row, tags in zip(
            source_batch,
            batch["rows"],
            output["decoded"],
            strict=True,
        ):
            histories[str(row["sample_id"])] = (
                _predicted_turns_from_tags(
                    encoded_row,
                    tags,
                    row,
                    label_space=label_space,
                )
            )
    return histories


def write_selected_histories_jsonl(
    path: Path | str,
    selected_histories: Mapping[
        str,
        Sequence[dict[str, Any]],
    ],
    *,
    sample_order: Sequence[str] | None = None,
    progress: bool = True,
) -> Path:
    """Atomically persist selected histories in the established format."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    ordered_ids = (
        [str(sample_id) for sample_id in sample_order]
        if sample_order is not None
        else [str(sample_id) for sample_id in selected_histories]
    )
    if set(ordered_ids) != {
        str(sample_id) for sample_id in selected_histories
    } or len(ordered_ids) != len(selected_histories):
        raise ValueError(
            "sample_order must contain each selected sample exactly once."
        )

    temporary = resolved.with_name(f".{resolved.name}.tmp")
    rows = progress_iter(
        ordered_ids,
        total=len(ordered_ids),
        desc=f"write {resolved.name}",
        unit="query",
        enabled=progress,
    )
    with temporary.open("w", encoding="utf-8") as handle:
        for sample_id in rows:
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "history": list(
                            selected_histories[sample_id]
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(resolved)
    return resolved


def read_selected_histories_jsonl(
    path: Path | str,
    *,
    progress: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Load selected histories and reject duplicate sample IDs."""

    resolved = Path(path)
    rows = read_label_jsonl(resolved, progress=progress)
    histories: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in histories:
            raise ValueError(
                f"Doppelte sample_id in {resolved}: {sample_id}"
            )
        histories[sample_id] = list(row.get("history", []))
    return histories


def expand_selected_histories(
    rows: Sequence[dict[str, Any]],
    selected_histories: Mapping[
        str,
        Sequence[dict[str, Any]],
    ],
    *,
    granularity: str,
) -> dict[str, list[dict[str, Any]]]:
    """Expand span selections to complete Q/A fields or complete turns."""

    if granularity not in {"qa", "turn"}:
        raise ValueError(
            "granularity muss 'qa' oder 'turn' sein."
        )

    row_by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in row_by_id:
            raise ValueError(
                f"Doppelte sample_id in rows: {sample_id}"
            )
        row_by_id[sample_id] = row

    selected_ids = {str(sample_id) for sample_id in selected_histories}
    row_ids = set(row_by_id)
    missing = sorted(row_ids.difference(selected_ids))
    extra = sorted(selected_ids.difference(row_ids))
    if missing or extra:
        raise ValueError(
            "Sample-Grain stimmt nicht überein: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )

    expanded: dict[str, list[dict[str, Any]]] = {}
    for sample_id, row in row_by_id.items():
        source_turns: dict[int, dict[str, Any]] = {}
        for turn in row.get("history", []):
            turn_id = int(turn["turn_id"])
            if turn_id in source_turns:
                raise ValueError(
                    f"Doppelte turn_id für {sample_id}: {turn_id}"
                )
            source_turns[turn_id] = turn

        selected_by_turn: dict[int, dict[str, Any]] = {}
        for turn in selected_histories[sample_id]:
            turn_id = int(turn["turn_id"])
            if turn_id not in source_turns:
                raise ValueError(
                    f"Unbekannte turn_id für {sample_id}: {turn_id}"
                )
            if turn_id in selected_by_turn:
                raise ValueError(
                    f"Doppelte selektierte turn_id für "
                    f"{sample_id}: {turn_id}"
                )
            if not (
                str(turn.get("question", "")).strip()
                or str(turn.get("answer", "")).strip()
            ):
                raise ValueError(
                    f"Leere Selektion für {sample_id}, Turn {turn_id}"
                )
            selected_by_turn[turn_id] = turn

        expanded_turns: list[dict[str, Any]] = []
        for turn_id, source in source_turns.items():
            selected = selected_by_turn.get(turn_id)
            if selected is None:
                continue
            keep_question = bool(
                str(selected.get("question", "")).strip()
            )
            keep_answer = bool(
                str(selected.get("answer", "")).strip()
            )
            if granularity == "turn":
                keep_question = keep_answer = True

            question = (
                str(source.get("question", ""))
                if keep_question
                else ""
            )
            answer = (
                str(source.get("answer", ""))
                if keep_answer
                else ""
            )
            if question or answer:
                expanded_turns.append(
                    {
                        "turn_id": turn_id,
                        "question": question,
                        "answer": answer,
                    }
                )
        expanded[sample_id] = expanded_turns
    return expanded


def _encode_inference_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    history_order: str,
) -> dict[str, Any]:
    pair = encode_history_pair(
        row,
        tokenizer,
        max_length=max_length,
        history_order=history_order,
    )
    return {
        "sample_id": str(row["sample_id"]),
        "input_ids": pair.input_ids,
        "attention_mask": pair.attention_mask,
        "history_mask": pair.history_mask,
        "history_offsets": pair.history_offsets,
        "history_text": pair.history_text,
        "ranges": pair.field_ranges,
    }


def encode_selector_inference_row(
    row: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    history_order: str,
) -> dict[str, Any]:
    """Public wrapper around the established inference encoding."""

    return _encode_inference_row(
        row,
        tokenizer,
        max_length=max_length,
        history_order=history_order,
    )


def _collate_inference(
    batch: Sequence[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    maximum_length = max(len(row["input_ids"]) for row in batch)

    def pad(values: list[int], pad_value: int) -> list[int]:
        return values + [pad_value] * (maximum_length - len(values))

    return {
        "rows": batch,
        "input_ids": torch.tensor(
            [pad(row["input_ids"], pad_token_id) for row in batch],
            dtype=torch.long,
            device=device,
        ),
        "attention_mask": torch.tensor(
            [
                pad(row["attention_mask"], 0)
                for row in batch
            ],
            dtype=torch.long,
            device=device,
        ),
        "history_mask": torch.tensor(
            [pad(row["history_mask"], 0) for row in batch],
            dtype=torch.bool,
            device=device,
        ),
    }


def collate_selector_inference_batch(
    batch: Sequence[dict[str, Any]],
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    """Public wrapper around the established inference collate."""

    return _collate_inference(batch, pad_token_id, device)


def _tag_spans(
    offsets: Sequence[tuple[int, int]],
    tags: Sequence[int],
    *,
    label_space: str,
) -> list[tuple[int, int]]:
    if label_space not in {"taxonomy", "collapsed"}:
        raise ValueError(
            "label_space must be 'taxonomy' or 'collapsed'."
        )
    spans: list[tuple[int, int]] = []
    start: int | None = None
    end: int | None = None
    for (token_start, token_end), tag in zip(
        offsets,
        tags,
        strict=True,
    ):
        keep = int(tag) != 0
        begins = (
            label_space == "collapsed"
            and int(tag) == LABEL_TO_ID["B-KEEP"]
        )
        if keep:
            if (
                start is None
                or begins
                or token_start > (end or token_start) + 4
            ):
                if start is not None and end is not None:
                    spans.append((start, end))
                start, end = token_start, token_end
            else:
                end = token_end
        elif start is not None and end is not None:
            spans.append((start, end))
            start = end = None
    if start is not None and end is not None:
        spans.append((start, end))
    return spans


def _predicted_turns_from_tags(
    encoded: dict[str, Any],
    tags: Sequence[int],
    row: dict[str, Any],
    *,
    label_space: str,
) -> list[dict[str, Any]]:
    relevant_tags = list(tags[: len(encoded["history_offsets"])])
    spans = _tag_spans(
        encoded["history_offsets"],
        relevant_tags,
        label_space=label_space,
    )
    chunks_by_field: dict[
        tuple[int, str],
        list[tuple[int, str]],
    ] = defaultdict(list)
    for span_start, span_end in spans:
        for key, (field_start, field_end) in encoded[
            "ranges"
        ].items():
            overlap_start = max(span_start, field_start)
            overlap_end = min(span_end, field_end)
            if overlap_end <= overlap_start:
                continue
            text = re.sub(
                r"\s+",
                " ",
                encoded["history_text"][overlap_start:overlap_end],
            ).strip()
            if not text or re.fullmatch(r"[,:;.!?\-]+", text):
                continue
            chunks_by_field[key].append((overlap_start, text))

    turns: list[dict[str, Any]] = []
    for turn in row.get("history", []):
        turn_id = int(turn["turn_id"])
        question = _join_chunks(
            chunks_by_field.get((turn_id, "question"), [])
        )
        answer = _join_chunks(
            chunks_by_field.get((turn_id, "answer"), [])
        )
        if question or answer:
            turns.append(
                {
                    "turn_id": turn_id,
                    "question": question,
                    "answer": answer,
                }
            )
    return turns


def project_selector_tags_to_history(
    encoded: dict[str, Any],
    tags: Sequence[int],
    row: dict[str, Any],
    *,
    label_space: str,
) -> list[dict[str, Any]]:
    """Project decoded token tags back to exact history substrings."""

    return _predicted_turns_from_tags(
        encoded,
        tags,
        row,
        label_space=label_space,
    )


def _join_chunks(chunks: Sequence[tuple[int, str]]) -> str:
    seen: set[str] = set()
    parts: list[str] = []
    for _, text in sorted(chunks, key=lambda item: item[0]):
        key = text.casefold()
        if key in seen:
            continue
        parts.append(text)
        seen.add(key)
    return " ".join(parts)


def _move_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _token_metrics(
    gold_sequences: Sequence[Sequence[int]],
    predicted_sequences: Sequence[Sequence[int]],
) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    total = correct = 0
    for gold, predicted in zip(
        gold_sequences,
        predicted_sequences,
        strict=True,
    ):
        for gold_label, predicted_label in zip(
            gold,
            predicted,
            strict=True,
        ):
            gold_keep = gold_label != LABEL_TO_ID["O"]
            predicted_keep = predicted_label != LABEL_TO_ID["O"]
            true_positive += int(gold_keep and predicted_keep)
            false_positive += int(not gold_keep and predicted_keep)
            false_negative += int(gold_keep and not predicted_keep)
            correct += int(gold_label == predicted_label)
            total += 1
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "token_accuracy": correct / total if total else 0.0,
        "keep_precision": precision,
        "keep_recall": recall,
        "keep_f1": f1,
        "keep_tp": true_positive,
        "keep_fp": false_positive,
        "keep_fn": false_negative,
        "token_correct": correct,
        "token_total": total,
    }


def _binary_keep_metrics(
    gold_sequences: Sequence[Sequence[int]],
    predicted_sequences: Sequence[Sequence[int]],
) -> dict[str, float]:
    """Evaluate non-O tokens and contiguous non-O runs."""

    token_metrics = _token_metrics(
        gold_sequences,
        predicted_sequences,
    )
    true_positive = false_positive = false_negative = 0
    for gold, predicted in zip(
        gold_sequences,
        predicted_sequences,
        strict=True,
    ):
        gold_spans = _binary_keep_spans(gold)
        predicted_spans = _binary_keep_spans(predicted)
        true_positive += len(gold_spans & predicted_spans)
        false_positive += len(predicted_spans - gold_spans)
        false_negative += len(gold_spans - predicted_spans)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "keep_precision": token_metrics["keep_precision"],
        "keep_recall": token_metrics["keep_recall"],
        "keep_f1": token_metrics["keep_f1"],
        "keep_span_precision": precision,
        "keep_span_recall": recall,
        "keep_span_f1": f1,
    }


def _binary_keep_spans(
    sequence: Sequence[int],
) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    start: int | None = None
    for index, label in enumerate(sequence):
        if int(label) != 0 and start is None:
            start = index
        elif int(label) == 0 and start is not None:
            spans.add((start, index))
            start = None
    if start is not None:
        spans.add((start, len(sequence)))
    return spans


def _bio_spans(sequence: Sequence[int]) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    start: int | None = None
    for index, label in enumerate(sequence):
        if label == LABEL_TO_ID["B-KEEP"]:
            if start is not None:
                spans.add((start, index))
            start = index
        elif label == LABEL_TO_ID["I-KEEP"]:
            if start is None:
                start = index
        elif start is not None:
            spans.add((start, index))
            start = None
    if start is not None:
        spans.add((start, len(sequence)))
    return spans


def _span_metrics(
    gold_sequences: Sequence[Sequence[int]],
    predicted_sequences: Sequence[Sequence[int]],
) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    for gold, predicted in zip(
        gold_sequences,
        predicted_sequences,
        strict=True,
    ):
        gold_spans = _bio_spans(gold)
        predicted_spans = _bio_spans(predicted)
        true_positive += len(gold_spans & predicted_spans)
        false_positive += len(predicted_spans - gold_spans)
        false_negative += len(gold_spans - predicted_spans)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
    }


def _write_training_metrics(
    *,
    path: Path,
    config: dict[str, Any],
    train_stats: dict[str, Any],
    dev_stats: dict[str, Any],
    history: Sequence[dict[str, Any]],
    best_metrics: dict[str, Any] | None,
    duration_seconds: float,
) -> None:
    _write_json(
        path,
        {
            "config": config,
            "train_label_stats": train_stats,
            "dev_label_stats": dev_stats,
            "history": list(history),
            "best": best_metrics,
            "duration_seconds": duration_seconds,
        },
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "EncodedHistoryPair",
    "EncodedSelectorExample",
    "EncodedSelectorDataset",
    "EncoderCrfHistorySelector",
    "EncoderLinearHistorySelector",
    "HistorySelectorConfig",
    "HistorySelectorDataset",
    "LABEL_TO_ID",
    "SelectorTrainingResult",
    "TAXONOMY_ID_TO_LABEL",
    "TAXONOMY_LABEL_TO_ID",
    "TaxonomyHistorySelectorDataset",
    "balanced_selector_class_weights",
    "build_history_text",
    "collate_selector_batch",
    "collate_selector_inference_batch",
    "encode_history_pair",
    "encode_selector_inference_row",
    "encode_selector_row",
    "encode_taxonomy_row",
    "evaluate_history_selector",
    "evaluate_taxonomy_selector",
    "expand_selected_histories",
    "fit_selector_model",
    "gather_history_sequences",
    "inject_lora_encoder",
    "load_selector_encoder_state",
    "positive_char_spans",
    "predict_selected_histories",
    "project_collapsed_selector_row",
    "project_selector_tags_to_history",
    "project_taxonomy_selector_row",
    "read_label_jsonl",
    "read_selected_histories_jsonl",
    "seed_selector_training",
    "selector_class_stats",
    "selector_label_stats",
    "selector_parameter_counts",
    "train_selector_epoch",
    "write_selected_histories_jsonl",
]
