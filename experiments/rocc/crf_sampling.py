"""Exact sequence sampling and multi-candidate inference for ROCC CRFs."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torchcrf import CRF
from transformers import AutoModel

from .history_selector import (
    EncoderCrfHistorySelector,
    HistorySelectorConfig,
    _collate_inference,
    _encode_inference_row,
    _predicted_turns_from_tags,
    gather_history_sequences,
)
from .population import derive_query_seed
from .progress import get_tqdm


@torch.no_grad()
def sample_crf_sequences(
    crf: CRF,
    emissions: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_samples: int,
    generator: torch.Generator | None = None,
    query_generators: (
        Sequence[Sequence[torch.Generator]] | None
    ) = None,
) -> list[list[list[int]]]:
    """Draw exact joint sequences with forward-filter/backward-sampling.

    The outer list follows the draw index, the middle list the batch
    index, and the inner list contains the active tag sequence.
    """

    if not bool(getattr(crf, "batch_first", False)):
        raise ValueError("CRF sampling requires batch_first=True.")
    if emissions.ndim != 3:
        raise ValueError(
            "emissions must have shape (batch, length, labels)."
        )
    if mask.shape != emissions.shape[:2]:
        raise ValueError("mask shape does not match emissions.")
    if emissions.size(-1) != int(crf.num_tags):
        raise ValueError("emission label count does not match CRF.")
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if (generator is None) == (query_generators is None):
        raise ValueError(
            "Provide exactly one shared generator or query_generators."
        )

    active = mask.bool()
    if not active[:, 0].all():
        raise ValueError("Every CRF sequence must start active.")
    if (active[:, 1:] & ~active[:, :-1]).any():
        raise ValueError("CRF masks must be left-aligned without gaps.")

    batch_size, sequence_length, _ = emissions.shape
    lengths = active.sum(dim=1).long()
    alpha = crf.start_transitions + emissions[:, 0]
    alpha_rows = [alpha]
    for position in range(1, sequence_length):
        scores = (
            alpha.unsqueeze(2)
            + crf.transitions.unsqueeze(0)
            + emissions[:, position].unsqueeze(1)
        )
        next_alpha = torch.logsumexp(scores, dim=1)
        alpha = torch.where(
            active[:, position].unsqueeze(1),
            next_alpha,
            alpha,
        )
        alpha_rows.append(alpha)
    alphas = torch.stack(alpha_rows, dim=1)

    if query_generators is None:
        uniforms = torch.rand(
            (num_samples, batch_size, sequence_length),
            generator=generator,
            device="cpu",
            dtype=torch.float64,
        )
    else:
        if len(query_generators) != batch_size or any(
            len(row) != num_samples for row in query_generators
        ):
            raise ValueError(
                "query_generators must have shape batch x num_samples."
            )
        uniforms = torch.empty(
            (num_samples, batch_size, sequence_length),
            device="cpu",
            dtype=torch.float64,
        )
        for row_index, row_generators in enumerate(
            query_generators
        ):
            for draw_index, query_generator in enumerate(
                row_generators
            ):
                uniforms[draw_index, row_index] = torch.rand(
                    (sequence_length,),
                    generator=query_generator,
                    device="cpu",
                    dtype=torch.float64,
                )
    uniforms = uniforms.to(
        device=emissions.device,
        dtype=emissions.dtype,
    )
    batch_indices = torch.arange(
        batch_size,
        device=emissions.device,
    )
    last_positions = lengths - 1
    final_logits = (
        alphas[batch_indices, last_positions]
        + crf.end_transitions
    )
    current = _categorical_from_uniform(
        final_logits.unsqueeze(0).expand(num_samples, -1, -1),
        uniforms[:, :, -1],
    )

    sampled = torch.zeros(
        (num_samples, batch_size, sequence_length),
        dtype=torch.long,
        device=emissions.device,
    )
    draw_indices = torch.arange(
        num_samples,
        device=emissions.device,
    )[:, None].expand(-1, batch_size)
    expanded_batch_indices = batch_indices[None, :].expand(
        num_samples,
        -1,
    )
    sampled[
        draw_indices,
        expanded_batch_indices,
        last_positions[None, :].expand(num_samples, -1),
    ] = current

    transition_by_next = crf.transitions.transpose(0, 1)
    for position in range(sequence_length - 2, -1, -1):
        conditional_logits = (
            alphas[:, position].unsqueeze(0)
            + transition_by_next[current]
        )
        previous = _categorical_from_uniform(
            conditional_logits,
            uniforms[:, :, position],
        )
        position_is_active = (
            position < (lengths - 1)
        ).unsqueeze(0)
        current = torch.where(
            position_is_active,
            previous,
            current,
        )
        sampled[:, :, position] = torch.where(
            position_is_active,
            current,
            sampled[:, :, position],
        )

    sampled_cpu = sampled.cpu()
    lengths_cpu = lengths.cpu().tolist()
    return [
        [
            sampled_cpu[draw, row, :length].tolist()
            for row, length in enumerate(lengths_cpu)
        ]
        for draw in range(num_samples)
    ]


def load_crf_inference_model(
    checkpoint: Path | str,
    *,
    model_name: str,
    model_revision: str,
    num_labels: int,
    device: torch.device,
) -> EncoderCrfHistorySelector:
    """Load one collapsed MiniLM+PyTorch-CRF checkpoint for inference."""

    state = torch.load(
        Path(checkpoint),
        map_location=device,
        weights_only=True,
    )
    stored_weights = state.get("class_weights")
    if stored_weights is None:
        class_weights = [1.0] * int(num_labels)
    else:
        class_weights = (
            stored_weights.detach().cpu().to(torch.float).tolist()
        )

    encoder = AutoModel.from_pretrained(
        model_name,
        revision=model_revision,
        trust_remote_code=False,
    )
    model = EncoderCrfHistorySelector(
        model_name,
        num_labels=num_labels,
        class_weights=class_weights,
        crf_loss_weight=1.0,
        token_loss_weight=1.0,
        encoder=encoder,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "CRF checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def load_legacy_crf_inference_model(
    checkpoint: Path | str,
    *,
    model_name: str,
    model_revision: str,
    num_labels: int,
    device: torch.device,
) -> EncoderCrfHistorySelector:
    """Backward-compatible alias for the general CRF loader."""

    return load_crf_inference_model(
        checkpoint,
        model_name=model_name,
        model_revision=model_revision,
        num_labels=num_labels,
        device=device,
    )


@torch.no_grad()
def predict_crf_decoder_histories(
    rows: Sequence[dict[str, Any]],
    *,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    config: HistorySelectorConfig,
    batch_size: int,
    progress: bool = True,
    progress_desc: str = "predict CRF decoder ablation",
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Decode identical CRF emissions with Viterbi and local argmax."""

    if not isinstance(getattr(model, "crf", None), CRF):
        raise TypeError("model must expose a pytorch-crf CRF as model.crf.")
    tqdm = get_tqdm()
    encoded = [
        _encode_inference_row(
            row,
            tokenizer,
            max_length=config.max_length,
            history_order=config.history_order,
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
    histories = {
        "viterbi": {},
        "emission_argmax": {},
    }
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
        sequence_emissions, sequence_mask, _ = (
            gather_history_sequences(
                output["emissions"],
                batch["history_mask"],
                None,
            )
        )
        local_tags = sequence_emissions.argmax(dim=-1)
        local_sequences = [
            local_tags[index, : int(mask.sum().item())]
            .detach()
            .cpu()
            .tolist()
            for index, mask in enumerate(sequence_mask)
        ]
        source_batch = rows[start : start + batch_size]
        for row_index, (row, encoded_row) in enumerate(
            zip(source_batch, batch["rows"], strict=True)
        ):
            sample_id = str(row["sample_id"])
            histories["viterbi"][sample_id] = (
                _predicted_turns_from_tags(
                    encoded_row,
                    output["decoded"][row_index],
                    row,
                    label_space="collapsed",
                )
            )
            histories["emission_argmax"][sample_id] = (
                _predicted_turns_from_tags(
                    encoded_row,
                    local_sequences[row_index],
                    row,
                    label_space="collapsed",
                )
            )
    return histories


@torch.no_grad()
def predict_crf_history_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    model: nn.Module,
    tokenizer: Any,
    device: torch.device,
    config: HistorySelectorConfig,
    batch_size: int,
    num_samples: int,
    sampling_seed: int,
    sampling_seed_mode: Literal[
        "global_stream",
        "per_query_sha256",
    ] = "global_stream",
    batch_metrics_reporter: (
        Callable[[Mapping[str, Any]], None] | None
    ) = None,
    progress: bool = True,
    progress_desc: str = "predict CRF candidates",
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return Viterbi and exact sampled history selections."""

    if not isinstance(getattr(model, "crf", None), CRF):
        raise TypeError("model must expose a pytorch-crf CRF as model.crf.")
    if num_samples < 0:
        raise ValueError("num_samples must be nonnegative.")
    tqdm = get_tqdm()
    encoded = [
        _encode_inference_row(
            row,
            tokenizer,
            max_length=config.max_length,
            history_order=config.history_order,
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
    arm_names = ("viterbi",) + tuple(
        f"sample_{index}" for index in range(1, num_samples + 1)
    )
    histories = {arm: {} for arm in arm_names}
    if sampling_seed_mode not in {
        "global_stream",
        "per_query_sha256",
    }:
        raise ValueError("Unknown sampling_seed_mode.")
    generator: torch.Generator | None = None
    if sampling_seed_mode == "global_stream":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(sampling_seed))
    model.eval()
    measure_batch_metrics = batch_metrics_reporter is not None

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
        if measure_batch_metrics and device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        model_started = (
            time.perf_counter() if measure_batch_metrics else None
        )
        output = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            history_mask=batch["history_mask"],
        )
        if measure_batch_metrics and device.type == "cuda":
            torch.cuda.synchronize(device)
        model_seconds = (
            time.perf_counter() - model_started
            if model_started is not None
            else None
        )
        sequence_emissions, sequence_mask, _ = (
            gather_history_sequences(
                output["emissions"],
                batch["history_mask"],
                None,
            )
        )
        source_batch = rows[start : start + batch_size]
        query_generators = None
        if num_samples and sampling_seed_mode == "per_query_sha256":
            query_generators = []
            for row in source_batch:
                sample_id = str(row["sample_id"])
                row_generators: list[torch.Generator] = []
                for candidate_index in range(
                    1,
                    int(num_samples) + 1,
                ):
                    query_generator = torch.Generator(
                        device="cpu"
                    )
                    query_generator.manual_seed(
                        derive_query_seed(
                            int(sampling_seed),
                            sample_id,
                            f"ffbs_{candidate_index}",
                        )
                    )
                    row_generators.append(query_generator)
                query_generators.append(row_generators)
        if measure_batch_metrics and device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_started = (
            time.perf_counter() if measure_batch_metrics else None
        )
        sampled_tags = (
            sample_crf_sequences(
                model.crf,
                sequence_emissions,
                sequence_mask,
                num_samples=num_samples,
                generator=generator,
                query_generators=query_generators,
            )
            if num_samples
            else []
        )
        if measure_batch_metrics and device.type == "cuda":
            torch.cuda.synchronize(device)
        sampling_seconds = (
            time.perf_counter() - sampling_started
            if sampling_started is not None
            else None
        )
        if batch_metrics_reporter is not None:
            assert model_seconds is not None
            assert sampling_seconds is not None
            batch_metrics_reporter(
                {
                    "batch_index": int(start // batch_size),
                    "batch_size": int(len(source_batch)),
                    "model_viterbi_seconds": float(
                        model_seconds
                    ),
                    "ffbs_sampling_seconds": float(
                        sampling_seconds
                    ),
                    "sampling_seed_mode": sampling_seed_mode,
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
        candidate_tags = [output["decoded"], *sampled_tags]
        for row_index, (row, encoded_row) in enumerate(
            zip(source_batch, batch["rows"], strict=True)
        ):
            sample_id = str(row["sample_id"])
            for arm, tags_by_row in zip(
                arm_names,
                candidate_tags,
                strict=True,
            ):
                histories[arm][sample_id] = (
                    _predicted_turns_from_tags(
                        encoded_row,
                        tags_by_row[row_index],
                        row,
                        label_space="collapsed",
                    )
                )
    return histories


def _categorical_from_uniform(
    logits: torch.Tensor,
    uniforms: torch.Tensor,
) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    cumulative = probabilities.cumsum(dim=-1)
    sampled = (
        uniforms.unsqueeze(-1) > cumulative
    ).sum(dim=-1)
    return sampled.clamp_max(logits.size(-1) - 1).long()


__all__ = [
    "load_crf_inference_model",
    "load_legacy_crf_inference_model",
    "predict_crf_decoder_histories",
    "predict_crf_history_candidates",
    "sample_crf_sequences",
]
