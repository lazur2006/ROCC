"""High-level orchestration for the canonical history-selector experiment."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .cached_itercqr import (
    CachedIterCQRBM25Config,
    CachedIterCQRBM25Pipeline,
)
from .datasets.topiocqa import (
    build_topiocqa_conversation_samples,
    load_topiocqa_frame,
    resolve_topiocqa_resources,
)
from .evaluation import paired_bootstrap_mean_ci
from .history_selector import (
    LABEL_TO_ID,
    TAXONOMY_LABEL_TO_ID,
    EncodedHistoryPair,
    EncodedSelectorDataset,
    EncoderCrfHistorySelector,
    EncoderLinearHistorySelector,
    HistorySelectorConfig,
    SelectorTrainingResult,
    balanced_selector_class_weights,
    collate_selector_batch,
    encode_history_pair,
    evaluate_history_selector,
    evaluate_taxonomy_selector,
    fit_selector_model,
    inject_lora_encoder,
    predict_selected_histories,
    project_collapsed_selector_row,
    project_taxonomy_selector_row,
    read_label_jsonl,
    read_selected_histories_jsonl,
    seed_selector_training,
    selector_class_stats,
    selector_label_stats,
    selector_parameter_counts,
    write_selected_histories_jsonl,
)
from .itercqr_components import load_itercqr_tokenizer
from .population import (
    ConversationSplit,
    ConversationSplitBundle,
    add_conversation_columns,
    build_conversation_split_bundle,
    build_depth_eligible_population,
    select_depth_balanced_population,
)
from .progress import get_tqdm
from .teacher_evaluation import (
    TEACHER_METRIC_COLUMNS,
    run_history_arm_evaluation,
)


CANDIDATE_ORDER = (
    "taxonomy_linear",
    "taxonomy_crf",
    "collapsed_linear",
    "collapsed_crf",
)
FULL_CANDIDATE_ORDER = (
    "taxonomy_linear",
    "collapsed_linear",
    "taxonomy_crf",
    "collapsed_crf",
)
CANDIDATE_SPEC = {
    "taxonomy_linear": ("taxonomy", "linear"),
    "taxonomy_crf": ("taxonomy", "crf"),
    "collapsed_linear": ("collapsed", "linear"),
    "collapsed_crf": ("collapsed", "crf"),
}
DECODER_RANK = {"linear": 0, "crf": 1}
LABEL_SPACE_RANK = {"taxonomy": 0, "collapsed": 1}
DECISION_SCHEMA_VERSION = 3
TRAINING_REGIME = "independent_hf_baseline_v1"
FULL_CANDIDATE_RECIPE_VERSION = "independent_hf_full_v2"
FULL_ARCHITECTURE_BOOTSTRAP_SEEDS = {
    "taxonomy_linear": 13,
    "collapsed_linear": 14,
    "taxonomy_crf": 15,
    "collapsed_crf": 16,
}
ORACLE_GATE_DEPTH_BOUNDS = {
    "d02_04": (2, 4),
    "d05_06": (5, 6),
    "d07_08": (7, 8),
    "d09_10": (9, 10),
    "d11_14": (11, 14),
    "d15_plus": (15, None),
}
ORACLE_GATE_DEPTH_BIN_ORDER = tuple(ORACLE_GATE_DEPTH_BOUNDS)
ORACLE_GATE_SELECTION_ORDER = tuple(
    reversed(ORACLE_GATE_DEPTH_BIN_ORDER)
)
ORACLE_GATE_SAMPLES_PER_BIN = 100
ORACLE_GATE_EXPECTED_SIZE = 600
ORACLE_GATE_EXPECTED_SHA256 = (
    "147e62c8a7f071356afd7da51dc2b6154"
    "7837e321a43cb36e68b5a30f9fef0bb"
)


@dataclass(frozen=True)
class SelectorExperimentConfig:
    """Complete reproducible configuration for Notebook 05."""

    train_labels: Path | str
    train_manifest: Path | str
    topiocqa_data_dir: Path | str
    result_dir: Path | str
    encoding_cache_dir: Path | str
    itercqr_model_dir: Path | str
    bm25_index_dir: Path | str
    pipeline_cache_db: Path | str
    dev_labels: Path | str | None = None
    dev_manifest: Path | str | None = None
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2"
    model_revision: str = (
        "a50ef00143b4d5391434df20ae11632588ac25be"
    )
    max_length: int = 512
    history_order: str = "recent_first"
    seed: int = 13
    smoke_selection_size: int = ORACLE_GATE_EXPECTED_SIZE
    smoke_validation_size: int = 1_000
    smoke_train_size: int = 7_042
    smoke_epochs: int = 1
    full_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_clip_norm: float = 1.0
    num_workers: int = 0
    crf_loss_weight: float = 1.0
    token_loss_weight: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    budgets: tuple[int, ...] = (64, 128, 256, 512)
    rewrite_batch_size: int = 16
    retrieval_batch_size: int = 64
    retrieval_workers: int = 4
    retrieval_top_k: int = 1_000
    eval_ks: tuple[int, ...] = (3, 10, 100, 1000)
    bootstrap_replicates: int = 10_000
    expected_train_rows: int = 38_432
    expected_train_sha256: str = (
        "901f5c891f9e0582d2941b91651e5a42"
        "fe88746328f8fe4e476325b50173235b"
    )
    expected_dev_rows: int | None = None
    expected_dev_sha256: str | None = None
    expected_smoke_selection_sha256: str = (
        ORACLE_GATE_EXPECTED_SHA256
    )
    expected_teacher_protocol_hash: str = (
        "1bb7fbf0e48741db4ff349d075940041f"
        "7c471d0ca46ab99db90b1b273cadcc9"
    )
    expected_teacher_seed: int = 42

    def __post_init__(self) -> None:
        for field_name in (
            "train_labels",
            "train_manifest",
            "topiocqa_data_dir",
            "result_dir",
            "encoding_cache_dir",
            "itercqr_model_dir",
            "bm25_index_dir",
            "pipeline_cache_db",
        ):
            object.__setattr__(
                self,
                field_name,
                Path(getattr(self, field_name)),
            )
        for field_name in ("dev_labels", "dev_manifest"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, Path(value))
        budgets = tuple(int(value) for value in self.budgets)
        eval_ks = tuple(int(value) for value in self.eval_ks)
        object.__setattr__(self, "budgets", budgets)
        object.__setattr__(self, "eval_ks", eval_ks)
        if budgets != (64, 128, 256, 512):
            raise ValueError(
                "Notebook 05 requires budgets (64, 128, 256, 512)."
            )
        if self.history_order not in {"recent_first", "chronological"}:
            raise ValueError("Unknown history_order.")
        if self.batch_size < 1 or self.max_length < 2:
            raise ValueError("Invalid selector batch or sequence length.")
        if (
            self.smoke_selection_size < 1
            or self.smoke_validation_size < 1
            or self.smoke_train_size < 1
        ):
            raise ValueError("Smoke split sizes must be positive.")
        if self.smoke_selection_size != ORACLE_GATE_EXPECTED_SIZE:
            raise ValueError("Notebook 05 requires the 600-query gate.")
        if (
            self.expected_smoke_selection_sha256
            != ORACLE_GATE_EXPECTED_SHA256
        ):
            raise ValueError("Notebook 05 requires the canonical gate hash.")


@dataclass(frozen=True)
class SelectorCandidateChoice:
    """One pre-registered candidate and budget selected on train-only data."""

    arm: str
    label_space: str
    decoder: str
    budget: int
    mrr: float
    comparison: Mapping[str, Any]
    runner_up_arm: str
    runner_up_comparison: Mapping[str, Any]
    reason: str


@dataclass(frozen=True)
class SelectorDecision:
    """Complete fixed recipe released from the smoke experiment."""

    label_space: str
    decoder: str
    adaptation: str
    budget: int
    candidate_comparisons: tuple[Mapping[str, Any], ...]
    runner_up_comparison: Mapping[str, Any]
    adaptation_comparison: Mapping[str, Any]
    gate_comparison: Mapping[str, Any]
    go_full: bool
    candidate_reason: str
    adaptation_reason: str
    gate_reason: str

    @property
    def arm(self) -> str:
        return f"{self.label_space}_{self.decoder}"


@dataclass(frozen=True)
class FullSelectorArchitectureChoice:
    """Architecture fixed from the four full-trained candidates."""

    arm: str
    label_space: str
    decoder: str
    budget: int
    mrr: float
    checkpoint: Path
    checkpoint_sha256: str
    vehicle_sample_id_sha256: str


def load_selector_sources(
    config: SelectorExperimentConfig,
    *,
    include_positive_contexts: bool = False,
    progress: bool = True,
) -> dict[str, Any]:
    """Load only labeled TopiOCQA train data and retrieval gold."""

    train_rows, train_manifest = _load_label_source(
        labels_path=config.train_labels,
        manifest_path=config.train_manifest,
        expected_rows=config.expected_train_rows,
        expected_sha256=config.expected_train_sha256,
        expected_protocol=config.expected_teacher_protocol_hash,
        expected_seed=config.expected_teacher_seed,
        progress=progress,
    )
    resources = resolve_topiocqa_resources(config.topiocqa_data_dir)
    train_frame = add_conversation_columns(
        _load_topiocqa_split_frame(
            resources,
            split="train",
            progress=progress,
        )
    )
    train_samples = build_topiocqa_conversation_samples(
        train_frame,
        minimum_history_depth=2,
        progress=progress,
    )
    return _align_label_rows_with_samples(
        rows=train_rows,
        manifest=train_manifest,
        samples=train_samples,
        frame=train_frame,
        prefix="train",
        include_positive_contexts=include_positive_contexts,
    )


def build_selector_oracle_gate(
    sources: Mapping[str, Any],
    config: SelectorExperimentConfig,
    *,
    output_path: Path | str | None = None,
    progress: bool = True,
) -> ConversationSplit:
    """Build the exact 600-query population defined by Notebook 03."""

    manifest_path = Path(
        output_path
        or (
            config.result_dir
            / "train600"
            / "population"
            / "manifest.json"
        )
    )
    identity = {
        "schema_version": 1,
        "dataset_sha256": config.expected_train_sha256,
        "seed": config.seed,
        "algorithm": "nb03_depth_balanced_oracle600_v1",
        "depth_bounds": {
            label: [lower, upper]
            for label, (lower, upper) in (
                ORACLE_GATE_DEPTH_BOUNDS.items()
            )
        },
        "depth_bin_order": list(ORACLE_GATE_DEPTH_BIN_ORDER),
        "selection_order": list(ORACLE_GATE_SELECTION_ORDER),
        "samples_per_bin": ORACLE_GATE_SAMPLES_PER_BIN,
        "expected_size": ORACLE_GATE_EXPECTED_SIZE,
        "expected_sample_id_sha256": (
            ORACLE_GATE_EXPECTED_SHA256
        ),
        "evaluation_scope": "train_member",
    }
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("identity") != identity:
            raise ValueError(
                "Existing oracle-gate manifest does not match: "
                f"{manifest_path}"
            )
        split = ConversationSplit(
            name="train600",
            sample_ids=tuple(map(str, manifest["sample_ids"])),
            conversation_ids=tuple(manifest["conversation_ids"]),
            sample_id_sha256=str(manifest["sample_id_sha256"]),
        )
        _validate_selector_oracle_gate(split, config=config)
        return split

    split = _select_selector_oracle_gate(
        sources,
        config,
        name="train600",
        progress=progress,
    )
    _write_json(
        manifest_path,
        {
            "identity": identity,
            "sample_ids": list(split.sample_ids),
            "conversation_ids": [
                _to_builtin(value)
                for value in split.conversation_ids
            ],
            "queries": split.query_count,
            "conversations": split.conversation_count,
            "sample_id_sha256": split.sample_id_sha256,
            "complete": True,
        },
    )
    return split


def build_selector_smoke_splits(
    sources: Mapping[str, Any],
    config: SelectorExperimentConfig,
    *,
    progress: bool = True,
) -> ConversationSplitBundle:
    """Build or load the canonical train-only smoke splits."""

    manifest_path = config.result_dir / "smoke" / "splits.json"
    oracle_gate_identity = {
        "depth_bounds": {
            label: [lower, upper]
            for label, (lower, upper) in (
                ORACLE_GATE_DEPTH_BOUNDS.items()
            )
        },
        "depth_bin_order": list(ORACLE_GATE_DEPTH_BIN_ORDER),
        "selection_order": list(ORACLE_GATE_SELECTION_ORDER),
        "samples_per_bin": ORACLE_GATE_SAMPLES_PER_BIN,
        "expected_size": ORACLE_GATE_EXPECTED_SIZE,
        "expected_sample_id_sha256": (
            ORACLE_GATE_EXPECTED_SHA256
        ),
    }
    expected_header = {
        "dataset_sha256": config.expected_train_sha256,
        "seed": config.seed,
        "algorithm": (
            "nb03_oracle600_plus_stable_sha1_subset_sum_v1"
        ),
        "split_names": [
            "smoke_selection",
            "smoke_validation",
            "smoke_train",
        ],
        "hash_salts": {
            "smoke_validation": "smoke_dev",
            "smoke_train": "smoke_train",
        },
        "oracle_gate": oracle_gate_identity,
    }
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if all(
            manifest.get(key) == value
            for key, value in expected_header.items()
        ):
            bundle = _split_bundle_from_manifest(manifest)
            _validate_smoke_split_bundle(bundle, config=config)
            return bundle
        raise ValueError(
            f"Existing split manifest does not match: {manifest_path}"
        )

    gate_split = _select_selector_oracle_gate(
        sources,
        config,
        name="smoke_selection",
        progress=progress,
    )

    split_frame = pd.DataFrame(
        [
            {
                "sample_id": str(row["sample_id"]),
                "conv_id": int(row["conv_id"]),
                "history_depth": int(row["history_len"]),
            }
            for row in sources["train_rows"]
        ]
    )

    remaining_frame = split_frame.loc[
        ~split_frame["conv_id"].isin(gate_split.conversation_ids)
    ]
    remaining_bundle = build_conversation_split_bundle(
        remaining_frame,
        exact_sizes={
            "smoke_validation": config.smoke_validation_size,
            "smoke_train": config.smoke_train_size,
        },
        hash_salts={
            "smoke_validation": "smoke_dev",
            "smoke_train": "smoke_train",
        },
        seed=config.seed,
        progress=progress,
    )
    bundle = ConversationSplitBundle(
        splits={
            "smoke_selection": gate_split,
            "smoke_validation": remaining_bundle[
                "smoke_validation"
            ],
            "smoke_train": remaining_bundle["smoke_train"],
        },
        seed=config.seed,
        algorithm=expected_header["algorithm"],
    )
    _validate_smoke_split_bundle(bundle, config=config)
    manifest = {
        **expected_header,
        "splits": {
            name: {
                "sample_ids": list(split.sample_ids),
                "conversation_ids": [
                    _to_builtin(value)
                    for value in split.conversation_ids
                ],
                "queries": split.query_count,
                "conversations": split.conversation_count,
                "sample_id_sha256": split.sample_id_sha256,
            }
            for name, split in bundle.splits.items()
        },
    }
    _write_json(manifest_path, manifest)
    return bundle


def prepare_selector_datasets(
    rows: Sequence[dict[str, Any]],
    splits: ConversationSplitBundle | Mapping[str, Sequence[str]],
    config: SelectorExperimentConfig,
    *,
    class_weight_split: str,
    dataset_sha256: str,
    label_spaces: Sequence[str] = ("taxonomy", "collapsed"),
    progress: bool = True,
) -> dict[str, Any]:
    """Materialize shared pair encodings for both selector label spaces."""

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
            or tokenizer.sep_token
            or tokenizer.cls_token
        )
    selected_label_spaces = tuple(map(str, label_spaces))
    if (
        not selected_label_spaces
        or len(set(selected_label_spaces))
        != len(selected_label_spaces)
        or not set(selected_label_spaces).issubset(
            {"taxonomy", "collapsed"}
        )
    ):
        raise ValueError(
            "label_spaces must contain unique taxonomy/collapsed names."
        )
    split_ids = (
        {
            name: split.sample_ids
            for name, split in splits.splits.items()
        }
        if isinstance(splits, ConversationSplitBundle)
        else {
            str(name): tuple(map(str, sample_ids))
            for name, sample_ids in splits.items()
        }
    )
    row_by_id = {str(row["sample_id"]): row for row in rows}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    taxonomy: dict[str, EncodedSelectorDataset] = {}
    collapsed: dict[str, EncodedSelectorDataset] = {}
    for split_name, sample_ids in split_ids.items():
        missing = [
            sample_id
            for sample_id in sample_ids
            if sample_id not in row_by_id
        ]
        if missing:
            raise KeyError(
                f"Rows for {split_name} missing: {missing[:3]}"
            )
        split_rows = [row_by_id[sample_id] for sample_id in sample_ids]
        rows_by_split[split_name] = split_rows
        split_hash = _sample_id_sha256(sample_ids)
        pairs, pair_cache_key = _cached_history_pairs(
            rows=split_rows,
            split_hash=split_hash,
            dataset_sha256=dataset_sha256,
            tokenizer=tokenizer,
            config=config,
            progress=progress,
        )
        if "taxonomy" in selected_label_spaces:
            taxonomy[split_name] = _cached_projected_dataset(
                kind="taxonomy_v3",
                rows=split_rows,
                pairs=pairs,
                pair_cache_key=pair_cache_key,
                split_hash=split_hash,
                dataset_sha256=dataset_sha256,
                config=config,
                progress=progress,
            )
        if "collapsed" in selected_label_spaces:
            collapsed[split_name] = _cached_projected_dataset(
                kind="collapsed_bio_v3",
                rows=split_rows,
                pairs=pairs,
                pair_cache_key=pair_cache_key,
                split_hash=split_hash,
                dataset_sha256=dataset_sha256,
                config=config,
                progress=progress,
            )

    active_datasets = (
        taxonomy
        if "taxonomy" in selected_label_spaces
        else collapsed
    )
    if class_weight_split not in active_datasets:
        raise KeyError(
            f"Unknown class_weight_split: {class_weight_split}"
        )
    result: dict[str, Any] = {
        "tokenizer": tokenizer,
        "dataset_sha256": str(dataset_sha256),
        "rows_by_split": rows_by_split,
    }
    if "taxonomy" in selected_label_spaces:
        result.update(
            {
                "taxonomy": taxonomy,
                "taxonomy_class_weights": (
                    balanced_selector_class_weights(
                        taxonomy[class_weight_split],
                        label_to_id=TAXONOMY_LABEL_TO_ID,
                    )
                ),
                "taxonomy_stats": {
                    name: selector_class_stats(
                        dataset,
                        label_to_id=TAXONOMY_LABEL_TO_ID,
                    )
                    for name, dataset in taxonomy.items()
                },
            }
        )
    if "collapsed" in selected_label_spaces:
        result.update(
            {
                "collapsed": collapsed,
                "collapsed_class_weights": (
                    balanced_selector_class_weights(
                        collapsed[class_weight_split],
                        label_to_id=LABEL_TO_ID,
                    )
                ),
                "collapsed_stats": {
                    name: selector_label_stats(dataset)
                    for name, dataset in collapsed.items()
                },
            }
        )
    return result


def run_taxonomy_training(
    prepared: Mapping[str, Any],
    config: SelectorExperimentConfig,
    *,
    decoder: str,
    adaptation: str,
    train_split: str = "smoke_train",
    validation_split: str | None = "smoke_validation",
    monitor_split: str | None = None,
    epochs: int | None = None,
    output_dir: Path | str | None = None,
    device: str | torch.device | None = None,
    resume: bool = True,
    progress: bool = True,
    training_recipe_version: str | None = None,
    batch_metrics_reporter: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
    epoch_metrics_reporter: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
) -> SelectorTrainingResult:
    """Train one six-class taxonomy model from the pinned baseline."""

    return _run_selector_training_stage(
        prepared,
        config,
        label_space="taxonomy",
        decoder=decoder,
        adaptation=adaptation,
        train_split=train_split,
        validation_split=validation_split,
        monitor_split=monitor_split,
        epochs=epochs,
        output_dir=output_dir,
        device=device,
        resume=resume,
        progress=progress,
        training_recipe_version=training_recipe_version,
        batch_metrics_reporter=batch_metrics_reporter,
        epoch_metrics_reporter=epoch_metrics_reporter,
    )


def run_collapsed_training(
    prepared: Mapping[str, Any],
    config: SelectorExperimentConfig,
    *,
    decoder: str,
    adaptation: str,
    train_split: str = "smoke_train",
    validation_split: str | None = "smoke_validation",
    monitor_split: str | None = None,
    epochs: int | None = None,
    output_dir: Path | str | None = None,
    device: str | torch.device | None = None,
    resume: bool = True,
    progress: bool = True,
    training_recipe_version: str | None = None,
    batch_metrics_reporter: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
    epoch_metrics_reporter: (
        Callable[[dict[str, Any]], None] | None
    ) = None,
) -> SelectorTrainingResult:
    """Train one direct BIO-KEEP model from the pinned baseline."""

    return _run_selector_training_stage(
        prepared,
        config,
        label_space="collapsed",
        decoder=decoder,
        adaptation=adaptation,
        train_split=train_split,
        validation_split=validation_split,
        monitor_split=monitor_split,
        epochs=epochs,
        output_dir=output_dir,
        device=device,
        resume=resume,
        progress=progress,
        training_recipe_version=training_recipe_version,
        batch_metrics_reporter=batch_metrics_reporter,
        epoch_metrics_reporter=epoch_metrics_reporter,
    )


def predict_selector_histories(
    rows: Sequence[dict[str, Any]],
    config: SelectorExperimentConfig,
    *,
    checkpoint: Path | str,
    label_space: str,
    adaptation: str,
    decoder: str,
    class_weights: Sequence[float],
    output_path: Path | str,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Persist exact selected histories for one explicit model recipe."""

    _validate_recipe(
        label_space=label_space,
        decoder=decoder,
        adaptation=adaptation,
    )
    checkpoint_path = Path(checkpoint)
    _validate_checkpoint_recipe(
        checkpoint_path,
        config=config,
        label_space=label_space,
        decoder=decoder,
        adaptation=adaptation,
    )
    expected_ids = [str(row["sample_id"]) for row in rows]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Prediction rows contain duplicate sample IDs.")

    resolved_output = Path(output_path)
    identity = {
        "sample_id_sha256": _sample_id_sha256(expected_ids),
        "checkpoint_sha256": _sha256_file(
            checkpoint_path,
            progress=False,
        ),
        "label_space": label_space,
        "adaptation": adaptation,
        "decoder": decoder,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "max_length": config.max_length,
        "history_order": config.history_order,
    }
    manifest_path = resolved_output.with_suffix(".manifest.json")
    if resolved_output.exists() and manifest_path.exists():
        manifest = _read_json(manifest_path)
        reusable = (
            manifest.get("identity") == identity
            and int(manifest.get("rows", -1)) == len(expected_ids)
            and manifest.get("output_sha256")
            == _sha256_file(resolved_output, progress=False)
            and manifest.get("complete") is True
        )
        if reusable:
            persisted = read_selected_histories_jsonl(
                resolved_output,
                progress=progress,
            )
            if list(persisted) == expected_ids:
                return persisted

    runtime_device = _runtime_device(device)
    model = _load_selector_model(
        config,
        checkpoint=checkpoint_path,
        label_space=label_space,
        decoder=decoder,
        adaptation=adaptation,
        class_weights=class_weights,
        device=runtime_device,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
            or tokenizer.sep_token
            or tokenizer.cls_token
        )
    histories = predict_selected_histories(
        rows,
        model=model,
        tokenizer=tokenizer,
        device=runtime_device,
        label_space=label_space,
        config=_history_selector_config(config),
        batch_size=config.batch_size,
        progress=progress,
        progress_desc=(
            f"predict {adaptation}/{label_space}/{decoder}"
        ),
    )
    write_selected_histories_jsonl(
        resolved_output,
        histories,
        sample_order=expected_ids,
        progress=progress,
    )
    _write_json(
        manifest_path,
        {
            "identity": identity,
            "rows": len(expected_ids),
            "output_sha256": _sha256_file(
                resolved_output,
                progress=False,
            ),
            "complete": True,
        },
    )
    return histories


def run_selector_retrieval_evaluation(
    *,
    samples: Sequence[Any],
    selector_histories: Mapping[
        str,
        Mapping[str, Sequence[dict[str, str]]],
    ],
    gold_by_sample: Mapping[str, Sequence[Any]],
    config: SelectorExperimentConfig,
    output_dir: Path | str,
    budgets: Sequence[int] | None = None,
    include_recency: bool = True,
    include_query_only: bool = True,
    persist_runtime_cache_stats: bool = True,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate selector arms with optional equal-budget controls."""

    resolved_budgets = tuple(
        int(value)
        for value in (
            config.budgets if budgets is None else budgets
        )
    )
    if (
        not resolved_budgets
        or len(set(resolved_budgets)) != len(resolved_budgets)
        or not set(resolved_budgets).issubset(config.budgets)
    ):
        raise ValueError("Retrieval budgets must be a nonempty config subset.")
    sample_ids = [str(sample.sample_id) for sample in samples]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Retrieval samples contain duplicate sample IDs.")
    missing_gold = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in gold_by_sample
    ]
    if missing_gold:
        raise KeyError(f"Missing retrieval gold: {missing_gold[:3]}")

    histories_by_arm: dict[
        str,
        dict[str, Sequence[dict[str, str]]],
    ] = {}
    for arm, histories in selector_histories.items():
        missing = [
            sample_id
            for sample_id in sample_ids
            if sample_id not in histories
        ]
        if missing:
            raise KeyError(f"Missing histories for {arm}: {missing[:3]}")
        histories_by_arm[str(arm)] = {
            sample_id: list(histories[sample_id])
            for sample_id in sample_ids
        }
    if include_recency:
        if "recency" in histories_by_arm:
            raise ValueError("recency is a reserved control-arm name.")
        histories_by_arm["recency"] = {
            str(sample.sample_id): [
                {
                    "question": str(turn.question),
                    "answer": str(turn.answer),
                }
                for turn in sample.history
            ]
            for sample in samples
        }
    if include_query_only:
        if "query_only" in histories_by_arm:
            raise ValueError("query_only is a reserved control-arm name.")
        histories_by_arm["query_only"] = {
            sample_id: [] for sample_id in sample_ids
        }
    arm_order = tuple(histories_by_arm)

    runtime_device = _runtime_device(device)
    tokenizer = load_itercqr_tokenizer(config.itercqr_model_dir)
    pipeline = CachedIterCQRBM25Pipeline(
        config=CachedIterCQRBM25Config(
            cache_db=config.pipeline_cache_db,
            model_dir=config.itercqr_model_dir,
            bm25_index_dir=config.bm25_index_dir,
            device=str(runtime_device),
            rewrite_batch_size=config.rewrite_batch_size,
            retrieval_batch_size=config.retrieval_batch_size,
            retrieval_workers=config.retrieval_workers,
            retrieval_top_k=config.retrieval_top_k,
            eval_ks=config.eval_ks,
            progress=progress,
        ),
        tokenizer=tokenizer,
    )
    try:
        evaluation = run_history_arm_evaluation(
            samples=samples,
            histories_by_arm=histories_by_arm,
            arm_order=arm_order,
            tokenizer=tokenizer,
            budgets=resolved_budgets,
            pipeline=pipeline,
            gold_by_sample=gold_by_sample,
            progress=progress,
            progress_desc="serialize selector arms",
        )
    finally:
        pipeline.close()

    summary = (
        evaluation.pipeline_results.groupby(
            ["budget", "arm"],
            observed=True,
            sort=True,
        )
        .agg(
            n=("sample_id", "size"),
            **{
                metric: (metric, "mean")
                for metric in TEACHER_METRIC_COLUMNS
            },
        )
        .reset_index()
    )
    resolved_output = Path(output_dir)
    resolved_output.mkdir(parents=True, exist_ok=True)
    evaluation.pipeline_results.to_csv(
        resolved_output / "metrics_by_query.csv",
        index=False,
    )
    summary.to_csv(
        resolved_output / "summary.csv",
        index=False,
    )
    if persist_runtime_cache_stats:
        evaluation.pipeline_summary.to_csv(
            resolved_output / "cache_stats.csv",
            index=False,
        )
    return {
        "evaluation": evaluation,
        "summary": summary,
    }


def compare_selector_mrr(
    pipeline_results: pd.DataFrame,
    *,
    left_arm: str,
    right_arm: str,
    budget: int,
    seed: int,
    replicates: int = 10_000,
    output_path: Path | str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Compute one paired query-level MRR difference and percentile CI."""

    selected = pipeline_results.loc[
        pipeline_results["budget"].eq(int(budget))
        & pipeline_results["arm"].isin([left_arm, right_arm]),
        ["sample_id", "arm", "MRR"],
    ]
    if selected.duplicated(["sample_id", "arm"]).any():
        raise ValueError("Comparison contains duplicate sample/arm rows.")
    wide = selected.pivot(
        index="sample_id",
        columns="arm",
        values="MRR",
    )
    if (
        left_arm not in wide
        or right_arm not in wide
        or wide[[left_arm, right_arm]].isna().any().any()
    ):
        raise ValueError("Comparison arms are not paired completely.")
    identity = {
        "left_arm": left_arm,
        "right_arm": right_arm,
        "budget": int(budget),
        "seed": int(seed),
        "replicates": int(replicates),
        "paired_rows_sha256": _frame_sha256(
            selected.sort_values(
                ["sample_id", "arm"],
                kind="mergesort",
            )
        ),
    }
    resolved_output = (
        Path(output_path) if output_path is not None else None
    )
    if resolved_output is not None:
        cached = _load_cached_mapping(
            resolved_output,
            identity=identity,
        )
        if cached is not None:
            return cached
    deltas = (
        wide[left_arm].to_numpy(dtype=float)
        - wide[right_arm].to_numpy(dtype=float)
    )
    ci_low, ci_high = paired_bootstrap_mean_ci(
        deltas,
        seed=int(seed),
        replicates=int(replicates),
        progress=progress,
        progress_desc=f"bootstrap {left_arm} - {right_arm}",
    )
    result = {
        "budget": int(budget),
        "comparison": f"{left_arm} - {right_arm}",
        "left_arm": left_arm,
        "right_arm": right_arm,
        "n": int(len(deltas)),
        "mrr_delta": float(np.mean(deltas)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "seed": int(seed),
        "replicates": int(replicates),
    }
    if resolved_output is not None:
        _write_json(resolved_output, result)
        _write_json(
            resolved_output.with_suffix(".manifest.json"),
            {
                "identity": identity,
                "output_sha256": _sha256_file(
                    resolved_output,
                    progress=False,
                ),
                "complete": True,
            },
        )
    return result


def compare_selector_metric(
    pipeline_results: pd.DataFrame,
    *,
    left_arm: str,
    right_arm: str,
    budget: int,
    metric: str,
    seed: int,
    replicates: int = 10_000,
    progress: bool = True,
) -> dict[str, Any]:
    """Compute one deterministic paired difference for any metric."""

    required = {"sample_id", "arm", "budget", metric}
    missing = required.difference(pipeline_results.columns)
    if missing:
        raise KeyError(
            "Metric comparison columns missing: "
            + ", ".join(sorted(missing))
        )
    selected = pipeline_results.loc[
        pipeline_results["budget"].eq(int(budget))
        & pipeline_results["arm"].isin([left_arm, right_arm]),
        ["sample_id", "arm", metric],
    ].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    if selected.duplicated(["sample_id", "arm"]).any():
        raise ValueError("Comparison contains duplicate sample/arm rows.")
    wide = (
        selected.pivot(
            index="sample_id",
            columns="arm",
            values=metric,
        )
        .sort_index(kind="mergesort")
    )
    if (
        left_arm not in wide
        or right_arm not in wide
        or wide[[left_arm, right_arm]].isna().any().any()
    ):
        raise ValueError("Comparison arms are not paired completely.")
    deltas = (
        wide[left_arm].to_numpy(dtype=float)
        - wide[right_arm].to_numpy(dtype=float)
    )
    ci_low, ci_high = paired_bootstrap_mean_ci(
        deltas,
        seed=int(seed),
        replicates=int(replicates),
        progress=progress,
        progress_desc=(
            f"bootstrap {metric} {left_arm} - {right_arm}"
        ),
    )
    return {
        "budget": int(budget),
        "metric": str(metric),
        "comparison": f"{left_arm} - {right_arm}",
        "left_arm": left_arm,
        "right_arm": right_arm,
        "n": int(len(deltas)),
        "delta": float(np.mean(deltas)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "seed": int(seed),
        "replicates": int(replicates),
    }


def compare_selector_candidates(
    pipeline_results: pd.DataFrame,
    config: SelectorExperimentConfig,
    *,
    output_path: Path | str | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Compare all four candidate arms with equal-budget Recency."""

    relevant = pipeline_results.loc[
        pipeline_results["arm"].isin(
            [*CANDIDATE_ORDER, "recency"]
        ),
        ["sample_id", "arm", "budget", "MRR"],
    ].sort_values(
        ["budget", "arm", "sample_id"],
        kind="mergesort",
    )
    identity = {
        "candidate_order": list(CANDIDATE_ORDER),
        "budgets": list(config.budgets),
        "seed": config.seed,
        "replicates": config.bootstrap_replicates,
        "pipeline_rows_sha256": _frame_sha256(relevant),
    }
    resolved_output = (
        Path(output_path) if output_path is not None else None
    )
    if resolved_output is not None and resolved_output.exists():
        manifest_path = resolved_output.with_suffix(
            ".manifest.json"
        )
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            if (
                manifest.get("identity") == identity
                and manifest.get("complete") is True
                and manifest.get("output_sha256")
                == _sha256_file(resolved_output, progress=False)
            ):
                cached = pd.read_csv(
                    resolved_output,
                    float_precision="round_trip",
                )
                if len(cached) == (
                    len(CANDIDATE_ORDER) * len(config.budgets)
                ):
                    return cached
    records: list[dict[str, Any]] = []
    for candidate_index, arm in enumerate(CANDIDATE_ORDER):
        label_space, decoder = CANDIDATE_SPEC[arm]
        for budget_index, budget in enumerate(config.budgets):
            comparison = compare_selector_mrr(
                pipeline_results,
                left_arm=arm,
                right_arm="recency",
                budget=budget,
                seed=(
                    config.seed
                    + len(config.budgets) * candidate_index
                    + budget_index
                ),
                replicates=config.bootstrap_replicates,
                progress=progress,
            )
            records.append(
                {
                    "arm": arm,
                    "label_space": label_space,
                    "decoder": decoder,
                    "mrr": _arm_metric_mean(
                        pipeline_results,
                        arm=arm,
                        budget=budget,
                        metric="MRR",
                    ),
                    "recency_mrr": _arm_metric_mean(
                        pipeline_results,
                        arm="recency",
                        budget=budget,
                        metric="MRR",
                    ),
                    **comparison,
                    "admissible": (
                        float(comparison["ci95_low"]) > 0.0
                    ),
                }
            )
    result = pd.DataFrame(records)
    expected = len(CANDIDATE_ORDER) * len(config.budgets)
    if len(result) != expected:
        raise RuntimeError(
            f"Expected {expected} candidate comparisons, got {len(result)}."
        )
    if resolved_output is not None:
        resolved_output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(resolved_output, index=False)
        _write_json(
            resolved_output.with_suffix(".manifest.json"),
            {
                "identity": identity,
                "output_sha256": _sha256_file(
                    resolved_output,
                    progress=False,
                ),
                "complete": True,
            },
        )
    return result


def select_selector_candidate(
    candidate_comparisons: pd.DataFrame,
    pipeline_results: pd.DataFrame,
    config: SelectorExperimentConfig,
    *,
    runner_output_path: Path | str | None = None,
    progress: bool = True,
) -> SelectorCandidateChoice | None:
    """Apply the pre-registered candidate/budget selection rule."""

    required = {
        "arm",
        "label_space",
        "decoder",
        "budget",
        "mrr",
        "ci95_low",
        "admissible",
    }
    missing = required.difference(candidate_comparisons.columns)
    if missing:
        raise KeyError(
            "Candidate comparison columns missing: "
            + ", ".join(sorted(missing))
        )
    eligible = candidate_comparisons.loc[
        candidate_comparisons["admissible"].astype(bool)
        & candidate_comparisons["ci95_low"].gt(0.0)
    ].copy()
    if eligible.empty:
        return None
    eligible["decoder_rank"] = eligible["decoder"].map(DECODER_RANK)
    eligible["label_space_rank"] = eligible["label_space"].map(
        LABEL_SPACE_RANK
    )
    winner = (
        eligible.sort_values(
            [
                "mrr",
                "budget",
                "decoder_rank",
                "label_space_rank",
            ],
            ascending=[False, True, True, True],
            kind="mergesort",
        )
        .iloc[0]
    )
    winner_arm = str(winner["arm"])
    winner_budget = int(winner["budget"])
    other_arms = [
        arm for arm in CANDIDATE_ORDER if arm != winner_arm
    ]
    runner_up_arm = sorted(
        other_arms,
        key=lambda arm: (
            -_arm_metric_mean(
                pipeline_results,
                arm=arm,
                budget=winner_budget,
                metric="MRR",
            ),
            CANDIDATE_ORDER.index(arm),
        ),
    )[0]
    runner_up_comparison = compare_selector_mrr(
        pipeline_results,
        left_arm=winner_arm,
        right_arm=runner_up_arm,
        budget=winner_budget,
        seed=config.seed + 16,
        replicates=config.bootstrap_replicates,
        output_path=runner_output_path,
        progress=progress,
    )
    comparison = {
        key: _to_builtin(value)
        for key, value in winner.items()
        if key not in {"decoder_rank", "label_space_rank"}
    }
    reason = (
        f"{winner_arm} at B{winner_budget} has the highest unrounded "
        "MRR among candidates with a strictly positive lower CI "
        "against equal-budget Recency."
    )
    return SelectorCandidateChoice(
        arm=winner_arm,
        label_space=str(winner["label_space"]),
        decoder=str(winner["decoder"]),
        budget=winner_budget,
        mrr=float(winner["mrr"]),
        comparison=comparison,
        runner_up_arm=runner_up_arm,
        runner_up_comparison=runner_up_comparison,
        reason=reason,
    )


def save_selector_candidate_choice(
    choice: SelectorCandidateChoice | None,
    comparisons: pd.DataFrame,
    path: Path | str,
) -> Path:
    """Persist the complete architecture/budget decision."""

    resolved = Path(path)
    _write_json(
        resolved,
        {
            "schema_version": DECISION_SCHEMA_VERSION,
            "choice": (
                _json_payload(asdict(choice))
                if choice is not None
                else None
            ),
            "comparisons": _dataframe_records(comparisons),
            "go_lora": choice is not None,
        },
    )
    return resolved


def select_selector_adaptation(
    adaptation_comparison: Mapping[str, Any],
) -> tuple[str, str]:
    """Choose LoRA only when its paired interval is fully positive."""

    if float(adaptation_comparison["ci95_low"]) > 0.0:
        return "lora", "LoRA interval lies fully above zero."
    return (
        "full",
        "LoRA interval is not fully above zero; Full-FT retained.",
    )


def passes_selector_gate(
    gate_comparison: Mapping[str, Any],
) -> tuple[bool, str]:
    """Release full training only for a fully positive Recency interval."""

    go_full = float(gate_comparison["ci95_low"]) > 0.0
    return (
        go_full,
        (
            "Selected selector interval lies fully above zero."
            if go_full
            else "Selected selector interval is not fully above zero."
        ),
    )


def resolve_selector_decision(
    *,
    choice: SelectorCandidateChoice,
    candidate_comparisons: pd.DataFrame,
    adaptation_comparison: Mapping[str, Any],
    gate_comparison: Mapping[str, Any],
) -> SelectorDecision:
    """Resolve adaptation and full-run gate for one fixed candidate."""

    adaptation, adaptation_reason = select_selector_adaptation(
        adaptation_comparison
    )
    go_full, gate_reason = passes_selector_gate(gate_comparison)
    return SelectorDecision(
        label_space=choice.label_space,
        decoder=choice.decoder,
        adaptation=adaptation,
        budget=choice.budget,
        candidate_comparisons=tuple(
            _dataframe_records(candidate_comparisons)
        ),
        runner_up_comparison=dict(choice.runner_up_comparison),
        adaptation_comparison=dict(adaptation_comparison),
        gate_comparison=dict(gate_comparison),
        go_full=go_full,
        candidate_reason=choice.reason,
        adaptation_reason=adaptation_reason,
        gate_reason=gate_reason,
    )


def save_selector_decision(
    decision: SelectorDecision,
    path: Path | str,
    *,
    checkpoints: Mapping[str, Path | str] | None = None,
) -> Path:
    """Persist one versioned decision and its smoke checkpoint hashes."""

    resolved = Path(path)
    _write_json(
        resolved,
        {
            "schema_version": DECISION_SCHEMA_VERSION,
            "decision": _decision_payload(decision),
            "checkpoint_sha256": {
                str(name): _sha256_file(
                    Path(checkpoint),
                    progress=False,
                )
                for name, checkpoint in (checkpoints or {}).items()
            },
        },
    )
    return resolved


def load_selector_decision(
    path: Path | str,
) -> SelectorDecision:
    """Load only the current decision schema."""

    payload = _read_json(Path(path))
    if payload.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise ValueError("Unsupported selector decision schema.")
    value = payload.get("decision")
    if not isinstance(value, dict):
        raise ValueError("Selector decision payload is missing.")
    required = set(SelectorDecision.__dataclass_fields__)
    if set(value) != required:
        raise ValueError("Selector decision fields do not match schema.")
    return SelectorDecision(
        label_space=str(value["label_space"]),
        decoder=str(value["decoder"]),
        adaptation=str(value["adaptation"]),
        budget=int(value["budget"]),
        candidate_comparisons=tuple(
            dict(row) for row in value["candidate_comparisons"]
        ),
        runner_up_comparison=dict(value["runner_up_comparison"]),
        adaptation_comparison=dict(value["adaptation_comparison"]),
        gate_comparison=dict(value["gate_comparison"]),
        go_full=bool(value["go_full"]),
        candidate_reason=str(value["candidate_reason"]),
        adaptation_reason=str(value["adaptation_reason"]),
        gate_reason=str(value["gate_reason"]),
    )


def run_full_selector_candidates(
    prepared: Mapping[str, Any],
    config: SelectorExperimentConfig,
    *,
    monitor_split: str = "train600",
    output_dir: Path | str | None = None,
    device: str | torch.device | None = None,
    batch_metrics_reporter: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    epoch_metrics_reporter: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    history_reporter: (
        Callable[[str, pd.DataFrame, pd.DataFrame], None] | None
    ) = None,
    progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Train all four independent full candidates from the HF baseline."""

    for split_name in ("full", monitor_split):
        if split_name not in prepared.get("rows_by_split", {}):
            raise KeyError(f"Prepared split is missing: {split_name}")
    for label_space in ("taxonomy", "collapsed"):
        if label_space not in prepared:
            raise KeyError(f"Prepared label space is missing: {label_space}")

    resolved_output = Path(
        output_dir or (config.result_dir / "full_candidates")
    )
    resolved_output.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    combined_loss_frames: list[pd.DataFrame] = []
    combined_epoch_frames: list[pd.DataFrame] = []

    for arm in FULL_CANDIDATE_ORDER:
        label_space, decoder = CANDIDATE_SPEC[arm]
        candidate_dir = resolved_output / arm
        loss_path = candidate_dir / "loss_history.csv"
        epoch_path = candidate_dir / "epoch_metrics.csv"
        completed_epochs = _completed_training_epochs(candidate_dir)
        loss_records = _load_training_frame(loss_path)
        if not loss_records.empty:
            loss_records = loss_records.loc[
                pd.to_numeric(
                    loss_records["epoch"], errors="raise"
                ).le(completed_epochs)
            ].copy()
        epoch_records = _load_training_frame(epoch_path)
        if not epoch_records.empty:
            epoch_records = epoch_records.loc[
                pd.to_numeric(
                    epoch_records["epoch"], errors="raise"
                ).le(completed_epochs)
            ].copy()

        batch_rows = _dataframe_records(loss_records)
        epoch_rows = _dataframe_records(epoch_records)

        def persist_candidate_history() -> None:
            loss_frame = pd.DataFrame(batch_rows)
            epoch_frame = pd.DataFrame(epoch_rows)
            if not loss_frame.empty:
                loss_frame = (
                    loss_frame.drop_duplicates(
                        ["candidate", "global_step"],
                        keep="last",
                    )
                    .sort_values(
                        ["candidate", "global_step"],
                        kind="mergesort",
                    )
                    .reset_index(drop=True)
                )
                _write_csv_atomic(loss_path, loss_frame)
            if not epoch_frame.empty:
                epoch_frame = (
                    epoch_frame.drop_duplicates(
                        ["candidate", "epoch"],
                        keep="last",
                    )
                    .sort_values(
                        ["candidate", "epoch"],
                        kind="mergesort",
                    )
                    .reset_index(drop=True)
                )
                _write_csv_atomic(epoch_path, epoch_frame)

        def report_batch(row: dict[str, Any]) -> None:
            payload = {"candidate": arm, **dict(row)}
            batch_rows.append(payload)
            if batch_metrics_reporter is not None:
                batch_metrics_reporter(arm, payload)

        def report_epoch(row: dict[str, Any]) -> None:
            payload = {"candidate": arm, **dict(row)}
            epoch_rows.append(payload)
            persist_candidate_history()
            if epoch_metrics_reporter is not None:
                epoch_metrics_reporter(arm, payload)

        training_function = (
            run_taxonomy_training
            if label_space == "taxonomy"
            else run_collapsed_training
        )
        was_complete = _candidate_training_is_complete(
            candidate_dir,
            expected_epochs=config.full_epochs,
            recipe_version=FULL_CANDIDATE_RECIPE_VERSION,
        )
        run = training_function(
            prepared,
            config,
            decoder=decoder,
            adaptation="full",
            train_split="full",
            validation_split=None,
            monitor_split=monitor_split,
            epochs=config.full_epochs,
            output_dir=candidate_dir,
            device=device,
            resume=True,
            progress=progress,
            training_recipe_version=(
                FULL_CANDIDATE_RECIPE_VERSION
            ),
            batch_metrics_reporter=report_batch,
            epoch_metrics_reporter=report_epoch,
        )
        persist_candidate_history()

        loss_frame = _load_training_frame(loss_path)
        epoch_frame = _load_training_frame(epoch_path)
        if len(epoch_frame) != config.full_epochs:
            raise RuntimeError(
                f"{arm} has {len(epoch_frame)} epoch metrics; "
                f"expected {config.full_epochs}."
            )
        expected_steps = (
            len(prepared[label_space]["full"])
            + config.batch_size
            - 1
        ) // config.batch_size * config.full_epochs
        if len(loss_frame) != expected_steps:
            raise RuntimeError(
                f"{arm} has {len(loss_frame)} batch losses; "
                f"expected {expected_steps}."
            )
        checkpoint_sha256 = _sha256_file(
            run.checkpoint,
            progress=False,
        )
        candidate_result = {
            "arm": arm,
            "label_space": label_space,
            "decoder": decoder,
            "adaptation": "full",
            "run": run,
            "checkpoint": run.checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "class_weights": list(
                map(
                    float,
                    prepared[f"{label_space}_class_weights"],
                )
            ),
            "loss_history": loss_frame,
            "epoch_metrics": epoch_frame,
            "reused": bool(was_complete),
        }
        results[arm] = candidate_result
        combined_loss_frames.append(loss_frame)
        combined_epoch_frames.append(epoch_frame)
        if history_reporter is not None:
            history_reporter(arm, loss_frame, epoch_frame)

    combined_loss = pd.concat(
        combined_loss_frames,
        ignore_index=True,
    ).sort_values(
        ["candidate", "global_step"],
        kind="mergesort",
    )
    combined_epochs = pd.concat(
        combined_epoch_frames,
        ignore_index=True,
    ).sort_values(
        ["candidate", "epoch"],
        kind="mergesort",
    )
    _write_csv_atomic(
        resolved_output / "training_loss.csv",
        combined_loss,
    )
    _write_csv_atomic(
        resolved_output / "epoch_metrics.csv",
        combined_epochs,
    )
    _write_json(
        resolved_output / "training_manifest.json",
        {
            "schema_version": 1,
            "training_recipe_version": (
                FULL_CANDIDATE_RECIPE_VERSION
            ),
            "training_regime": TRAINING_REGIME,
            "initialization_source": "huggingface_baseline",
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "train_dataset_sha256": config.expected_train_sha256,
            "train_split_sha256": _sample_id_sha256(
                [
                    str(row["sample_id"])
                    for row in prepared["rows_by_split"]["full"]
                ]
            ),
            "monitor_split": monitor_split,
            "monitor_split_sha256": _sample_id_sha256(
                [
                    str(row["sample_id"])
                    for row in prepared["rows_by_split"][
                        monitor_split
                    ]
                ]
            ),
            "monitor_affects_checkpoint_selection": False,
            "epochs": config.full_epochs,
            "candidates": {
                arm: {
                    "label_space": results[arm]["label_space"],
                    "decoder": results[arm]["decoder"],
                    "checkpoint": _relative_path(
                        Path(results[arm]["checkpoint"]),
                        config.result_dir,
                    ),
                    "checkpoint_sha256": results[arm][
                        "checkpoint_sha256"
                    ],
                }
                for arm in FULL_CANDIDATE_ORDER
            },
            "training_loss_sha256": _sha256_file(
                resolved_output / "training_loss.csv",
                progress=False,
            ),
            "epoch_metrics_sha256": _sha256_file(
                resolved_output / "epoch_metrics.csv",
                progress=False,
            ),
            "complete": True,
        },
    )
    return results


def select_full_selector_architecture(
    pipeline_results: pd.DataFrame,
    candidate_runs: Mapping[str, Mapping[str, Any]],
    config: SelectorExperimentConfig,
    *,
    budget: int = 64,
    vehicle_sample_id_sha256: str = ORACLE_GATE_EXPECTED_SHA256,
    output_dir: Path | str | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Select the highest-MRR full candidate on the fixed train600 vehicle."""

    resolved_output = Path(
        output_dir
        or (config.result_dir / "train600" / "architecture")
    )
    selected = pipeline_results.loc[
        pipeline_results["budget"].eq(int(budget))
        & pipeline_results["arm"].isin(FULL_CANDIDATE_ORDER),
        ["sample_id", "arm", *TEACHER_METRIC_COLUMNS],
    ].copy()
    if selected.duplicated(["sample_id", "arm"]).any():
        raise ValueError("Architecture rows contain duplicates.")
    ids_by_arm = {
        arm: tuple(
            sorted(
                selected.loc[
                    selected["arm"].eq(arm), "sample_id"
                ].astype(str)
            )
        )
        for arm in FULL_CANDIDATE_ORDER
    }
    if any(len(ids) != ORACLE_GATE_EXPECTED_SIZE for ids in ids_by_arm.values()):
        raise ValueError("Architecture comparison is not complete for 600 queries.")
    if len(set(ids_by_arm.values())) != 1:
        raise ValueError("Architecture candidates use different query IDs.")
    if _sample_id_sha256(next(iter(ids_by_arm.values()))) != str(
        vehicle_sample_id_sha256
    ):
        raise ValueError("Architecture vehicle hash does not match NB03.")

    summary = (
        pipeline_results.loc[
            pipeline_results["arm"].isin(
                [*FULL_CANDIDATE_ORDER, "recency"]
            )
        ]
        .groupby(["budget", "arm"], observed=True, sort=True)
        .agg(
            n=("sample_id", "size"),
            **{
                metric: (metric, "mean")
                for metric in TEACHER_METRIC_COLUMNS
            },
        )
        .reset_index()
    )
    primary = summary.loc[
        summary["budget"].eq(int(budget))
        & summary["arm"].isin(FULL_CANDIDATE_ORDER)
    ].copy()
    primary["tie_rank"] = primary["arm"].map(
        {
            arm: index
            for index, arm in enumerate(FULL_CANDIDATE_ORDER)
        }
    )
    winner_row = primary.sort_values(
        ["MRR", "tie_rank"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    winner_arm = str(winner_row["arm"])
    comparisons = pd.DataFrame(
        [
            compare_selector_metric(
                pipeline_results,
                left_arm=winner_arm,
                right_arm=arm,
                budget=int(budget),
                metric="MRR",
                seed=FULL_ARCHITECTURE_BOOTSTRAP_SEEDS[arm],
                replicates=config.bootstrap_replicates,
                progress=progress,
            )
            for arm in FULL_CANDIDATE_ORDER
            if arm != winner_arm
        ]
    )
    checkpoint = Path(candidate_runs[winner_arm]["checkpoint"])
    checkpoint_sha256 = str(
        candidate_runs[winner_arm]["checkpoint_sha256"]
    )
    if checkpoint_sha256 != _sha256_file(
        checkpoint,
        progress=False,
    ):
        raise ValueError("Winner checkpoint hash changed.")
    label_space, decoder = CANDIDATE_SPEC[winner_arm]
    choice = FullSelectorArchitectureChoice(
        arm=winner_arm,
        label_space=label_space,
        decoder=decoder,
        budget=int(budget),
        mrr=float(winner_row["MRR"]),
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha256,
        vehicle_sample_id_sha256=str(vehicle_sample_id_sha256),
    )

    decision = {
        "schema_version": 1,
        "locked": True,
        "evaluation_scope": "train_member",
        "selection_rule": "highest_unrounded_MRR_then_fixed_order",
        "tie_break_order": list(FULL_CANDIDATE_ORDER),
        "budget": int(budget),
        "selected_arm": winner_arm,
        "label_space": label_space,
        "decoder": decoder,
        "adaptation": "full",
        "selected_mrr": float(winner_row["MRR"]),
        "vehicle_sample_id_sha256": str(
            vehicle_sample_id_sha256
        ),
        "selector_checkpoint": _relative_path(
            checkpoint,
            config.result_dir,
        ),
        "selector_checkpoint_sha256": checkpoint_sha256,
        "bootstrap_replicates": config.bootstrap_replicates,
        "bootstrap_seeds": dict(
            FULL_ARCHITECTURE_BOOTSTRAP_SEEDS
        ),
        "summary_sha256": _frame_sha256(summary),
        "paired_comparisons_sha256": _frame_sha256(comparisons),
    }
    decision_path = resolved_output / "decision.json"
    if decision_path.exists():
        _write_locked_json(decision_path, decision)
    _write_csv_atomic(resolved_output / "summary.csv", summary)
    _write_csv_atomic(
        resolved_output / "paired_comparisons.csv",
        comparisons,
    )
    _write_locked_json(decision_path, decision)
    return {
        "choice": choice,
        "summary": summary,
        "comparisons": comparisons,
        "decision": decision,
        "decision_path": decision_path,
    }


def run_full_selector_training(
    sources: Mapping[str, Any],
    config: SelectorExperimentConfig,
    *,
    decision: SelectorDecision,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train the fixed recipe freshly from the pinned MiniLM baseline."""

    _validate_decision(decision, config)
    if not decision.go_full:
        raise RuntimeError("Smoke decision does not release full training.")
    full_dir = config.result_dir / "full"
    reused = _completed_full_run(
        full_dir,
        decision=decision,
        config=config,
    )
    if reused is not None:
        return reused

    full_ids = [
        str(row["sample_id"]) for row in sources["train_rows"]
    ]
    prepared = prepare_selector_datasets(
        sources["train_rows"],
        {"full": full_ids},
        config,
        class_weight_split="full",
        dataset_sha256=config.expected_train_sha256,
        label_spaces=(decision.label_space,),
        progress=progress,
    )
    training_function = (
        run_taxonomy_training
        if decision.label_space == "taxonomy"
        else run_collapsed_training
    )
    final = training_function(
        prepared,
        config,
        decoder=decision.decoder,
        adaptation=decision.adaptation,
        train_split="full",
        validation_split=None,
        epochs=config.full_epochs,
        output_dir=full_dir / decision.label_space,
        device=device,
        resume=True,
        progress=progress,
    )
    final_class_weights = list(
        map(
            float,
            prepared[f"{decision.label_space}_class_weights"],
        )
    )
    manifest = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": _decision_payload(decision),
        "training_regime": TRAINING_REGIME,
        "train_dataset_sha256": config.expected_train_sha256,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "final_checkpoint": str(final.checkpoint.resolve()),
        "final_checkpoint_sha256": _sha256_file(
            final.checkpoint,
            progress=False,
        ),
        "final_class_weights": final_class_weights,
        "complete": True,
    }
    _write_json(full_dir / "training_manifest.json", manifest)
    return {
        "reused": False,
        "prepared": prepared,
        "final": final,
        "final_checkpoint": final.checkpoint,
        "final_class_weights": final_class_weights,
        "manifest": manifest,
    }


def run_final_selector_evaluation(
    config: SelectorExperimentConfig,
    *,
    decision: SelectorDecision,
    final_checkpoint: Path | str,
    device: str | torch.device | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Load Dev only here and evaluate the already fixed full recipe."""

    _validate_decision(decision, config)
    if not decision.go_full:
        raise RuntimeError("Decision does not release final evaluation.")
    _, dev_dataset_sha256 = _require_dev_config(config)
    final_checkpoint = Path(final_checkpoint)
    dev_dir = config.result_dir / "dev"
    completed = _completed_dev_run(
        dev_dir,
        decision=decision,
        config=config,
        final_checkpoint=final_checkpoint,
    )
    if completed is not None:
        return completed

    sources = _load_selector_dev_sources(config, progress=progress)
    dev_ids = [str(row["sample_id"]) for row in sources["dev_rows"]]
    prepared = prepare_selector_datasets(
        sources["dev_rows"],
        {"dev": dev_ids},
        config,
        class_weight_split="dev",
        dataset_sha256=dev_dataset_sha256,
        label_spaces=(decision.label_space,),
        progress=progress,
    )
    runtime_device = _runtime_device(device)
    checkpoint_manifest = _validate_checkpoint_recipe(
        final_checkpoint,
        config=config,
        label_space=decision.label_space,
        decoder=decision.decoder,
        adaptation=decision.adaptation,
    )
    final_weights = list(
        map(
            float,
            checkpoint_manifest["config"]["class_weights"],
        )
    )
    final_model = _load_selector_model(
        config,
        checkpoint=final_checkpoint,
        label_space=decision.label_space,
        decoder=decision.decoder,
        adaptation=decision.adaptation,
        class_weights=final_weights,
        device=runtime_device,
    )
    final_loader = _selector_loader(
        prepared[decision.label_space]["dev"],
        tokenizer=prepared["tokenizer"],
        config=config,
        shuffle=False,
    )
    evaluator = (
        evaluate_taxonomy_selector
        if decision.label_space == "taxonomy"
        else evaluate_history_selector
    )
    label_metrics = pd.DataFrame(
        [
            {
                "stage": decision.label_space,
                **evaluator(
                    final_model,
                    final_loader,
                    runtime_device,
                    progress=progress,
                    progress_desc=(
                        f"final dev {decision.label_space}"
                    ),
                ),
            }
        ]
    )
    predictions = predict_selector_histories(
        sources["dev_rows"],
        config,
        checkpoint=final_checkpoint,
        label_space=decision.label_space,
        adaptation=decision.adaptation,
        decoder=decision.decoder,
        class_weights=final_weights,
        output_path=dev_dir / "predictions.jsonl",
        device=runtime_device,
        progress=progress,
    )
    retrieval = run_selector_retrieval_evaluation(
        samples=sources["dev_samples"],
        selector_histories={"selector": predictions},
        gold_by_sample=sources["dev_gold_by_sample"],
        config=config,
        output_dir=dev_dir / "retrieval",
        budgets=config.budgets,
        device=runtime_device,
        progress=progress,
    )
    comparisons = pd.DataFrame(
        [
            compare_selector_mrr(
                retrieval["evaluation"].pipeline_results,
                left_arm="selector",
                right_arm="recency",
                budget=budget,
                seed=config.seed + budget_index,
                replicates=config.bootstrap_replicates,
                progress=progress,
            )
            for budget_index, budget in enumerate(config.budgets)
        ]
    )
    label_metrics.to_csv(
        dev_dir / "label_metrics.csv",
        index=False,
    )
    comparisons.to_csv(
        dev_dir / "bootstrap.csv",
        index=False,
    )
    retrieval["summary"].to_csv(
        dev_dir / "summary.csv",
        index=False,
    )
    manifest = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": _decision_payload(decision),
        "training_regime": TRAINING_REGIME,
        "dev_dataset_sha256": dev_dataset_sha256,
        "evaluated_budgets": list(config.budgets),
        "final_checkpoint_sha256": _sha256_file(
            final_checkpoint,
            progress=False,
        ),
        "complete": True,
    }
    _write_json(dev_dir / "manifest.json", manifest)
    return {
        "reused": False,
        "label_metrics": label_metrics,
        "retrieval": retrieval,
        "retrieval_summary": retrieval["summary"],
        "comparisons": comparisons,
        "manifest": manifest,
    }


def validate_selector_environment(
    config: SelectorExperimentConfig,
    *,
    require_lora: bool = True,
) -> dict[str, Any]:
    """Validate the fixed encoder, CRF, and LoRA dependencies."""

    resources = resolve_topiocqa_resources(config.topiocqa_data_dir)
    required_paths = (
        config.train_labels,
        config.train_manifest,
        resources.train_json,
        config.itercqr_model_dir,
        config.bm25_index_dir,
    )
    missing_paths = [
        str(path) for path in required_paths if not Path(path).exists()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Selector environment paths missing: "
            + ", ".join(missing_paths)
        )
    crf_version = importlib.metadata.version("pytorch-crf")
    if crf_version != "0.7.2":
        raise RuntimeError(
            f"pytorch-crf 0.7.2 required, found {crf_version}."
        )
    peft_version: str | None = None
    if require_lora:
        if importlib.util.find_spec("peft") is None:
            raise RuntimeError(
                "peft is required for the LoRA control arm."
            )
        peft_version = importlib.metadata.version("peft")
    model_config = AutoConfig.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        trust_remote_code=False,
    )
    maximum_positions = int(
        getattr(model_config, "max_position_embeddings", 0)
    )
    if maximum_positions < config.max_length:
        raise RuntimeError(
            f"Encoder supports only {maximum_positions} positions."
        )
    return {
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "max_position_embeddings": maximum_positions,
        "crf_backend": "pytorch-crf",
        "pytorch_crf_version": crf_version,
        "peft_version": peft_version,
    }


def _run_selector_training_stage(
    prepared: Mapping[str, Any],
    config: SelectorExperimentConfig,
    *,
    label_space: str,
    decoder: str,
    adaptation: str,
    train_split: str,
    validation_split: str | None,
    monitor_split: str | None,
    epochs: int | None,
    output_dir: Path | str | None,
    device: str | torch.device | None,
    resume: bool,
    progress: bool,
    training_recipe_version: str | None,
    batch_metrics_reporter: (
        Callable[[dict[str, Any]], None] | None
    ),
    epoch_metrics_reporter: (
        Callable[[dict[str, Any]], None] | None
    ),
) -> SelectorTrainingResult:
    _validate_recipe(
        label_space=label_space,
        decoder=decoder,
        adaptation=adaptation,
    )
    if validation_split is not None and monitor_split is not None:
        raise ValueError(
            "validation_split and monitor_split are mutually exclusive."
        )
    runtime_device = _runtime_device(device)
    selected_epochs = int(epochs or config.smoke_epochs)
    resolved_output = Path(
        output_dir
        or _default_smoke_stage_dir(
            config,
            label_space=label_space,
            decoder=decoder,
            adaptation=adaptation,
        )
    )
    seed_selector_training(config.seed)
    encoder, target_modules = _build_encoder(
        config,
        adaptation=adaptation,
    )

    class_weights = prepared[
        f"{label_space}_class_weights"
    ]
    seed_selector_training(config.seed)
    model = _new_selector_model(
        config,
        label_space=label_space,
        decoder=decoder,
        encoder=encoder,
        class_weights=class_weights,
    ).to(runtime_device)
    seed_selector_training(config.seed)

    train_loader = _selector_loader(
        prepared[label_space][train_split],
        tokenizer=prepared["tokenizer"],
        config=config,
        shuffle=True,
    )
    validation_loader = (
        _selector_loader(
            prepared[label_space][validation_split],
            tokenizer=prepared["tokenizer"],
            config=config,
            shuffle=False,
        )
        if validation_split is not None
        else None
    )
    monitor_loader = (
        _selector_loader(
            prepared[label_space][monitor_split],
            tokenizer=prepared["tokenizer"],
            config=config,
            shuffle=False,
        )
        if monitor_split is not None
        else None
    )
    run_config = _training_config(
        config,
        label_space=label_space,
        adaptation=adaptation,
        decoder=decoder,
        train_split=train_split,
        validation_split=validation_split,
        epochs=selected_epochs,
        class_weights=class_weights,
        target_modules=target_modules,
        model=model,
    )
    run_config.update(
        _split_training_identity(
            prepared,
            train_split=train_split,
            validation_split=validation_split,
        )
    )
    if monitor_split is not None:
        run_config.update(
            {
                "monitor_split": monitor_split,
                "monitor_split_sha256": _sample_id_sha256(
                    [
                        str(row["sample_id"])
                        for row in prepared["rows_by_split"][
                            monitor_split
                        ]
                    ]
                ),
                "monitor_affects_checkpoint_selection": False,
            }
        )
    if training_recipe_version is not None:
        run_config["training_recipe_version"] = str(
            training_recipe_version
        )
    run_config["run_id"] = _mapping_sha256(run_config)
    _ensure_run_config(resolved_output / "config.json", run_config)

    evaluator = (
        evaluate_taxonomy_selector
        if label_space == "taxonomy"
        else evaluate_history_selector
    )
    selection_metric = (
        "taxonomy_macro_f1"
        if label_space == "taxonomy"
        else "keep_f1"
    )
    evaluation_loader = validation_loader or monitor_loader
    evaluation_kind = (
        "validation"
        if validation_loader is not None
        else "monitor"
    )
    result = fit_selector_model(
        model=model,
        train_loader=train_loader,
        output_dir=resolved_output,
        device=runtime_device,
        epochs=selected_epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        evaluate_fn=(
            (
                lambda selected_model, epoch: evaluator(
                    selected_model,
                    evaluation_loader,
                    runtime_device,
                    progress=progress,
                    progress_desc=(
                        f"{label_space} {adaptation}/{decoder} "
                        f"{evaluation_kind} epoch {epoch}"
                    ),
                )
            )
            if evaluation_loader is not None
            else None
        ),
        checkpoint_policy=(
            "best" if validation_loader is not None else "last"
        ),
        selection_metric=(
            selection_metric
            if validation_loader is not None
            else None
        ),
        progress=progress,
        progress_desc=(
            f"train {label_space} {adaptation}/{decoder}"
        ),
        gradient_clip_norm=config.gradient_clip_norm,
        run_config=run_config,
        train_stats=prepared[f"{label_space}_stats"][train_split],
        dev_stats=(
            prepared[f"{label_space}_stats"][validation_split]
            if validation_split is not None
            else {}
        ),
        epoch_checkpoint_pattern="checkpoint_epoch_{epoch}.pt",
        record_learning_rate=True,
        resume=resume,
        batch_metrics_reporter=batch_metrics_reporter,
        epoch_metrics_reporter=epoch_metrics_reporter,
    )
    _finalize_training_artifacts(
        result,
        run_config=run_config,
        model=model,
        extra={
            "target_modules": list(target_modules),
            "crf_backend": (
                {
                    "distribution": "pytorch-crf",
                    "version": importlib.metadata.version(
                        "pytorch-crf"
                    ),
                    "num_tags": len(
                        _label_to_id(label_space)
                    ),
                    "batch_first": True,
                    "reduction": "none",
                    "mask_dtype": "bool",
                }
                if decoder == "crf"
                else None
            ),
        },
    )
    return result


def _require_dev_config(
    config: SelectorExperimentConfig,
) -> tuple[int, str]:
    if (
        config.dev_labels is None
        or config.dev_manifest is None
        or config.expected_dev_rows is None
        or config.expected_dev_sha256 is None
    ):
        raise RuntimeError(
            "Dev evaluation requires explicit Dev paths, row count, "
            "and dataset hash."
        )
    return int(config.expected_dev_rows), str(
        config.expected_dev_sha256
    )


def _load_selector_dev_sources(
    config: SelectorExperimentConfig,
    *,
    progress: bool,
) -> dict[str, Any]:
    expected_dev_rows, expected_dev_sha256 = _require_dev_config(
        config
    )
    if config.dev_labels is None or config.dev_manifest is None:
        raise RuntimeError("Dev paths are unavailable.")
    dev_rows, dev_manifest = _load_label_source(
        labels_path=config.dev_labels,
        manifest_path=config.dev_manifest,
        expected_rows=expected_dev_rows,
        expected_sha256=expected_dev_sha256,
        expected_protocol=config.expected_teacher_protocol_hash,
        expected_seed=config.expected_teacher_seed,
        progress=progress,
    )
    resources = resolve_topiocqa_resources(config.topiocqa_data_dir)
    dev_frame = add_conversation_columns(
        _load_topiocqa_split_frame(
            resources,
            split="dev",
            progress=progress,
        )
    )
    dev_samples = build_topiocqa_conversation_samples(
        dev_frame,
        minimum_history_depth=2,
        progress=progress,
    )
    aligned = _align_label_rows_with_samples(
        rows=dev_rows,
        manifest=dev_manifest,
        samples=dev_samples,
        frame=dev_frame,
        prefix="dev",
    )
    return {
        "dev_rows": aligned["train_rows"],
        "dev_manifest": aligned["train_manifest"],
        "dev_samples": aligned["train_samples"],
        "dev_gold_by_sample": aligned["train_gold_by_sample"],
    }


def _load_topiocqa_split_frame(
    resources: Any,
    *,
    split: str,
    progress: bool,
) -> pd.DataFrame:
    tqdm = get_tqdm()
    with tqdm(
        total=1,
        desc=f"load TopiOCQA {split}",
        unit="split",
        dynamic_ncols=True,
        disable=not progress,
    ) as bar:
        frame = load_topiocqa_frame(
            resources,
            splits=(split,),
        )
        bar.update(1)
    return frame


def _align_label_rows_with_samples(
    *,
    rows: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
    samples: Sequence[Any],
    frame: pd.DataFrame,
    prefix: str,
    include_positive_contexts: bool = False,
) -> dict[str, Any]:
    sample_by_id = {str(sample.sample_id): sample for sample in samples}
    sample_ids = [str(row["sample_id"]) for row in rows]
    if not all(
        sample_id.startswith(f"{prefix}:")
        for sample_id in sample_ids
    ):
        raise ValueError(f"{prefix} labels contain foreign sample IDs.")
    missing = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in sample_by_id
    ]
    if missing:
        raise ValueError(
            f"TopiOCQA {prefix} samples missing: {missing[:3]}"
        )
    gold = frame.set_index("sample_id")[
        "positive_ctx_passage_ids"
    ].to_dict()
    missing_gold = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in gold
    ]
    if missing_gold:
        raise ValueError(
            f"TopiOCQA {prefix} gold missing: {missing_gold[:3]}"
        )
    aligned = {
        "train_rows": list(rows),
        "train_manifest": dict(manifest),
        "train_samples": [
            sample_by_id[sample_id] for sample_id in sample_ids
        ],
        "train_gold_by_sample": {
            sample_id: gold[sample_id] for sample_id in sample_ids
        },
    }
    if include_positive_contexts:
        positive_contexts = frame.set_index("sample_id")[
            "positive_ctxs"
        ].to_dict()
        aligned["train_positive_ctxs_by_sample"] = {
            sample_id: list(positive_contexts[sample_id] or [])
            for sample_id in sample_ids
        }
    return aligned


def _load_label_source(
    *,
    labels_path: Path,
    manifest_path: Path,
    expected_rows: int,
    expected_sha256: str,
    expected_protocol: str,
    expected_seed: int,
    progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = _read_json(manifest_path)
    if (
        manifest.get("dataset_sha256") != expected_sha256
        or manifest.get("protocol_hash") != expected_protocol
        or manifest.get("seed") != int(expected_seed)
        or int(manifest.get("eligible_queries", -1)) != expected_rows
    ):
        raise ValueError(
            f"Manifest contract mismatch: {manifest_path}"
        )
    actual_sha256 = _sha256_file(labels_path, progress=progress)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Dataset hash mismatch for {labels_path}: {actual_sha256}"
        )
    rows = read_label_jsonl(labels_path, progress=progress)
    if any(
        row.get("teacher_metadata", {}).get("seed")
        != int(expected_seed)
        for row in rows
    ):
        raise ValueError(
            f"Teacher seed mismatch in {labels_path}."
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(rows) != expected_rows or len(set(sample_ids)) != len(rows):
        raise ValueError(
            f"Unexpected row or ID count in {labels_path}."
        )
    return rows, manifest


def _cached_history_pairs(
    *,
    rows: Sequence[dict[str, Any]],
    split_hash: str,
    dataset_sha256: str,
    tokenizer: Any,
    config: SelectorExperimentConfig,
    progress: bool,
) -> tuple[list[EncodedHistoryPair], str]:
    signature = {
        "kind": "history_pairs_v1",
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_hash,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "max_length": config.max_length,
        "history_order": config.history_order,
    }
    cache_key = _mapping_sha256(signature)
    config.encoding_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.encoding_cache_dir / f"{cache_key}.pt"
    expected_ids = [str(row["sample_id"]) for row in rows]
    if cache_path.exists():
        pairs = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
        if [pair.sample_id for pair in pairs] == expected_ids:
            return list(pairs), cache_key
        raise ValueError(f"Pair cache identity mismatch: {cache_path}")

    tqdm = get_tqdm()
    pairs = [
        encode_history_pair(
            row,
            tokenizer,
            max_length=config.max_length,
            history_order=config.history_order,
        )
        for row in tqdm(
            rows,
            total=len(rows),
            desc=f"encode shared pairs {len(rows)}",
            unit="query",
            dynamic_ncols=True,
            disable=not progress,
        )
    ]
    temporary = cache_path.with_suffix(".tmp")
    torch.save(pairs, temporary)
    temporary.replace(cache_path)
    return pairs, cache_key


def _cached_projected_dataset(
    *,
    kind: str,
    rows: Sequence[dict[str, Any]],
    pairs: Sequence[EncodedHistoryPair],
    pair_cache_key: str,
    split_hash: str,
    dataset_sha256: str,
    config: SelectorExperimentConfig,
    progress: bool,
) -> EncodedSelectorDataset:
    signature = {
        "kind": kind,
        "pair_cache_key": pair_cache_key,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_hash,
    }
    cache_key = _mapping_sha256(signature)
    config.encoding_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = config.encoding_cache_dir / f"{cache_key}.pt"
    expected_ids = [str(row["sample_id"]) for row in rows]
    if len(rows) != len(pairs):
        raise ValueError("Rows and cached pairs differ in length.")
    if cache_path.exists():
        examples = torch.load(
            cache_path,
            map_location="cpu",
            weights_only=False,
        )
        if [example.sample_id for example in examples] == expected_ids:
            return EncodedSelectorDataset(examples)
        raise ValueError(
            f"Projection cache identity mismatch: {cache_path}"
        )

    tqdm = get_tqdm()
    row_pairs = tqdm(
        zip(rows, pairs, strict=True),
        total=len(rows),
        desc=f"project {kind} {len(rows)}",
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    )
    if kind == "taxonomy_v3":
        examples = [
            project_taxonomy_selector_row(row, pair)
            for row, pair in row_pairs
        ]
    elif kind == "collapsed_bio_v3":
        positive_labels = set(TAXONOMY_LABEL_TO_ID).difference(
            {"O"}
        )
        examples = [
            project_collapsed_selector_row(
                row,
                pair,
                positive_labels=positive_labels,
            )
            for row, pair in row_pairs
        ]
    else:
        raise ValueError(f"Unknown projection kind: {kind}")
    temporary = cache_path.with_suffix(".tmp")
    torch.save(examples, temporary)
    temporary.replace(cache_path)
    return EncodedSelectorDataset(examples)


def _selector_loader(
    dataset: EncodedSelectorDataset,
    *,
    tokenizer: Any,
    config: SelectorExperimentConfig,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_selector_batch(
            batch,
            int(tokenizer.pad_token_id),
        ),
        num_workers=config.num_workers,
        generator=generator if shuffle else None,
    )


def _build_encoder(
    config: SelectorExperimentConfig,
    *,
    adaptation: str,
) -> tuple[nn.Module, tuple[str, ...]]:
    if adaptation not in {"full", "lora"}:
        raise ValueError("adaptation must be 'full' or 'lora'.")
    encoder = AutoModel.from_pretrained(
        config.model_name,
        revision=config.model_revision,
        trust_remote_code=False,
    )
    if adaptation == "full":
        return encoder, ()
    return inject_lora_encoder(
        encoder,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )


def _new_selector_model(
    config: SelectorExperimentConfig,
    *,
    label_space: str,
    decoder: str,
    encoder: nn.Module,
    class_weights: Sequence[float],
) -> nn.Module:
    num_labels = len(_label_to_id(label_space))
    if decoder == "linear":
        return EncoderLinearHistorySelector(
            config.model_name,
            num_labels=num_labels,
            class_weights=class_weights,
            token_loss_weight=config.token_loss_weight,
            encoder=encoder,
        )
    if decoder == "crf":
        return EncoderCrfHistorySelector(
            config.model_name,
            num_labels=num_labels,
            class_weights=class_weights,
            crf_loss_weight=config.crf_loss_weight,
            token_loss_weight=config.token_loss_weight,
            encoder=encoder,
        )
    raise ValueError("decoder must be 'linear' or 'crf'.")


def _load_selector_model(
    config: SelectorExperimentConfig,
    *,
    checkpoint: Path,
    label_space: str,
    decoder: str,
    adaptation: str,
    class_weights: Sequence[float],
    device: torch.device,
) -> nn.Module:
    _validate_checkpoint_recipe(
        checkpoint,
        config=config,
        label_space=label_space,
        decoder=decoder,
        adaptation=adaptation,
    )
    encoder, _ = _build_encoder(config, adaptation=adaptation)
    model = _new_selector_model(
        config,
        label_space=label_space,
        decoder=decoder,
        encoder=encoder,
        class_weights=class_weights,
    ).to(device)
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        ),
        strict=True,
    )
    model.eval()
    return model


def _training_config(
    config: SelectorExperimentConfig,
    *,
    label_space: str,
    adaptation: str,
    decoder: str,
    train_split: str,
    validation_split: str | None,
    epochs: int,
    class_weights: Sequence[float],
    target_modules: Sequence[str],
    model: nn.Module,
) -> dict[str, Any]:
    return {
        "label_space": label_space,
        "training_regime": TRAINING_REGIME,
        "initialization_source": "huggingface_baseline",
        "adaptation": adaptation,
        "decoder": decoder,
        "train_split": train_split,
        "validation_split": validation_split,
        "epochs": int(epochs),
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "max_length": config.max_length,
        "history_order": config.history_order,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "gradient_clip_norm": config.gradient_clip_norm,
        "token_loss_weight": config.token_loss_weight,
        "crf_loss_weight": (
            config.crf_loss_weight if decoder == "crf" else None
        ),
        "class_weights": list(map(float, class_weights)),
        "target_modules": list(target_modules),
        "parameters": selector_parameter_counts(model),
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version(
            "transformers"
        ),
        "pytorch_crf_version": importlib.metadata.version(
            "pytorch-crf"
        ),
        "peft_version": (
            importlib.metadata.version("peft")
            if adaptation == "lora"
            else None
        ),
    }


def _split_training_identity(
    prepared: Mapping[str, Any],
    *,
    train_split: str,
    validation_split: str | None,
) -> dict[str, Any]:
    return {
        "dataset_sha256": str(prepared["dataset_sha256"]),
        "train_split_sha256": _sample_id_sha256(
            [
                str(row["sample_id"])
                for row in prepared["rows_by_split"][train_split]
            ]
        ),
        "validation_split_sha256": (
            _sample_id_sha256(
                [
                    str(row["sample_id"])
                    for row in prepared["rows_by_split"][
                        validation_split
                    ]
                ]
            )
            if validation_split is not None
            else None
        ),
    }


def _finalize_training_artifacts(
    result: SelectorTrainingResult,
    *,
    run_config: Mapping[str, Any],
    model: nn.Module,
    extra: Mapping[str, Any],
) -> None:
    pd.DataFrame(result.history).to_csv(
        result.output_dir / "metrics.csv",
        index=False,
    )
    state_path = result.output_dir / "training_state.pt"
    ended_timestamp = max(
        result.checkpoint.stat().st_mtime,
        (
            state_path.stat().st_mtime
            if state_path.exists()
            else result.checkpoint.stat().st_mtime
        ),
    )
    ended_at = datetime.fromtimestamp(
        ended_timestamp,
        timezone.utc,
    )
    started_at = ended_at - timedelta(
        seconds=result.duration_seconds,
    )
    _write_json(
        result.output_dir / "manifest.json",
        {
            "config": dict(run_config),
            "parameters": selector_parameter_counts(model),
            "checkpoint": str(result.checkpoint.resolve()),
            "checkpoint_sha256": _sha256_file(
                result.checkpoint,
                progress=False,
            ),
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": result.duration_seconds,
            "best_metrics": result.best_metrics,
            "complete": True,
            **dict(extra),
        },
    )


def _validate_checkpoint_recipe(
    checkpoint: Path,
    *,
    config: SelectorExperimentConfig,
    label_space: str,
    decoder: str,
    adaptation: str,
) -> dict[str, Any]:
    manifest_path = checkpoint.parent / "manifest.json"
    if not checkpoint.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Checkpoint or manifest missing: {checkpoint}"
        )
    manifest = _read_json(manifest_path)
    run_config = manifest.get("config", {})
    expected = {
        "label_space": label_space,
        "decoder": decoder,
        "adaptation": adaptation,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
        "training_regime": TRAINING_REGIME,
    }
    if any(
        run_config.get(key) != value
        for key, value in expected.items()
    ):
        raise ValueError(
            f"Checkpoint recipe does not match: {checkpoint}"
        )
    if manifest.get("checkpoint_sha256") != _sha256_file(
        checkpoint,
        progress=False,
    ):
        raise ValueError(f"Checkpoint hash mismatch: {checkpoint}")
    if manifest.get("complete") is not True:
        raise ValueError(f"Checkpoint run is incomplete: {checkpoint}")
    return manifest


def _validate_recipe(
    *,
    label_space: str,
    decoder: str,
    adaptation: str,
) -> None:
    if label_space not in {"taxonomy", "collapsed"}:
        raise ValueError(
            "label_space must be 'taxonomy' or 'collapsed'."
        )
    if decoder not in {"linear", "crf"}:
        raise ValueError("decoder must be 'linear' or 'crf'.")
    if adaptation not in {"full", "lora"}:
        raise ValueError("adaptation must be 'full' or 'lora'.")


def _validate_decision(
    decision: SelectorDecision,
    config: SelectorExperimentConfig,
) -> None:
    _validate_recipe(
        label_space=decision.label_space,
        decoder=decision.decoder,
        adaptation=decision.adaptation,
    )
    if decision.budget not in config.budgets:
        raise ValueError("Decision budget is not configured.")
    if len(decision.candidate_comparisons) != (
        len(CANDIDATE_ORDER) * len(config.budgets)
    ):
        raise ValueError("Decision does not contain all candidate CIs.")


def _completed_full_run(
    full_dir: Path,
    *,
    decision: SelectorDecision,
    config: SelectorExperimentConfig,
) -> dict[str, Any] | None:
    manifest_path = full_dir / "training_manifest.json"
    if not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    expected = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": _decision_payload(decision),
        "training_regime": TRAINING_REGIME,
        "train_dataset_sha256": config.expected_train_sha256,
        "model_name": config.model_name,
        "model_revision": config.model_revision,
    }
    if any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            f"Existing full run does not match: {manifest_path}"
        )
    final_checkpoint = Path(manifest["final_checkpoint"])
    final_class_weights = list(
        map(float, manifest.get("final_class_weights", []))
    )
    if (
        manifest.get("complete") is not True
        or manifest.get("final_checkpoint_sha256")
        != _sha256_file(final_checkpoint, progress=False)
        or not final_class_weights
    ):
        raise ValueError("Completed full-run manifest is inconsistent.")
    return {
        "reused": True,
        "prepared": None,
        "final": None,
        "final_checkpoint": final_checkpoint,
        "final_class_weights": final_class_weights,
        "manifest": manifest,
    }


def _completed_dev_run(
    dev_dir: Path,
    *,
    decision: SelectorDecision,
    config: SelectorExperimentConfig,
    final_checkpoint: Path,
) -> dict[str, Any] | None:
    manifest_path = dev_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    _, dev_dataset_sha256 = _require_dev_config(config)
    manifest = _read_json(manifest_path)
    expected = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision": _decision_payload(decision),
        "training_regime": TRAINING_REGIME,
        "dev_dataset_sha256": dev_dataset_sha256,
        "final_checkpoint_sha256": _sha256_file(
            final_checkpoint,
            progress=False,
        ),
        "complete": True,
    }
    if any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(
            f"Existing Dev run does not match: {manifest_path}"
        )
    if manifest.get("evaluated_budgets") != list(config.budgets):
        return None
    required = (
        dev_dir / "label_metrics.csv",
        dev_dir / "summary.csv",
        dev_dir / "bootstrap.csv",
    )
    if not all(path.exists() for path in required):
        raise ValueError("Completed Dev manifest has missing tables.")
    return {
        "reused": True,
        "label_metrics": pd.read_csv(required[0]),
        "retrieval": None,
        "retrieval_summary": pd.read_csv(required[1]),
        "comparisons": pd.read_csv(required[2]),
        "manifest": manifest,
    }


def _default_smoke_stage_dir(
    config: SelectorExperimentConfig,
    *,
    label_space: str,
    decoder: str,
    adaptation: str,
) -> Path:
    base = config.result_dir / "smoke"
    if adaptation == "lora":
        base = base / "lora"
    return base / f"{label_space}_{decoder}"


def _history_selector_config(
    config: SelectorExperimentConfig,
) -> HistorySelectorConfig:
    return HistorySelectorConfig(
        model_name=config.model_name,
        max_length=config.max_length,
        history_order=config.history_order,
        epochs=config.full_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        seed=config.seed,
        num_workers=config.num_workers,
        crf_loss_weight=config.crf_loss_weight,
        token_loss_weight=config.token_loss_weight,
    )


def _label_to_id(label_space: str) -> Mapping[str, int]:
    if label_space == "taxonomy":
        return TAXONOMY_LABEL_TO_ID
    if label_space == "collapsed":
        return LABEL_TO_ID
    raise ValueError(f"Unknown label space: {label_space}")


def _arm_metric_mean(
    pipeline_results: pd.DataFrame,
    *,
    arm: str,
    budget: int,
    metric: str,
) -> float:
    selected = pipeline_results.loc[
        pipeline_results["arm"].eq(arm)
        & pipeline_results["budget"].eq(int(budget)),
        metric,
    ]
    if selected.empty:
        raise ValueError(f"No {arm}/B{budget} rows for {metric}.")
    return float(selected.mean())


def _load_cached_mapping(
    path: Path,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() or not manifest_path.exists():
        return None
    manifest = _read_json(manifest_path)
    if (
        manifest.get("identity") != dict(identity)
        or manifest.get("complete") is not True
        or manifest.get("output_sha256")
        != _sha256_file(path, progress=False)
    ):
        return None
    return _read_json(path)


def _completed_training_epochs(output_dir: Path) -> int:
    metrics_path = output_dir / "metrics.json"
    if not metrics_path.exists():
        return 0
    history = _read_json(metrics_path).get("history", [])
    return max(
        (int(row.get("epoch", 0)) for row in history),
        default=0,
    )


def _candidate_training_is_complete(
    output_dir: Path,
    *,
    expected_epochs: int,
    recipe_version: str,
) -> bool:
    manifest_path = output_dir / "manifest.json"
    state_path = output_dir / "training_state.pt"
    if not manifest_path.exists() or not state_path.exists():
        return False
    manifest = _read_json(manifest_path)
    run_config = manifest.get("config", {})
    return bool(
        manifest.get("complete") is True
        and run_config.get("training_recipe_version")
        == str(recipe_version)
        and int(run_config.get("epochs", -1))
        == int(expected_epochs)
        and _completed_training_epochs(output_dir)
        == int(expected_epochs)
        and (output_dir / "loss_history.csv").exists()
        and (output_dir / "epoch_metrics.csv").exists()
    )


def _load_training_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, float_precision="round_trip")


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _relative_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def _write_locked_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _json_payload(dict(value))
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise RuntimeError(
                f"Locked decision differs from recomputation: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _frame_sha256(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataframe_records(
    frame: pd.DataFrame,
) -> list[dict[str, Any]]:
    return [
        {
            str(key): _to_builtin(value)
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _decision_payload(
    decision: SelectorDecision,
) -> dict[str, Any]:
    return _json_payload(asdict(decision))


def _json_payload(value: Any) -> Any:
    """Normalize tuples and NumPy scalars to their JSON representation."""

    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            default=_to_builtin,
        )
    )


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _ensure_run_config(
    path: Path,
    run_config: Mapping[str, Any],
) -> None:
    expected = dict(run_config)
    if path.exists():
        if _read_json(path) != expected:
            raise ValueError(
                f"Existing run config does not match: {path}"
            )
        return
    _write_json(path, expected)


def _split_bundle_from_manifest(
    manifest: Mapping[str, Any],
) -> ConversationSplitBundle:
    splits = {
        str(name): ConversationSplit(
            name=str(name),
            sample_ids=tuple(map(str, value["sample_ids"])),
            conversation_ids=tuple(value["conversation_ids"]),
            sample_id_sha256=str(value["sample_id_sha256"]),
        )
        for name, value in manifest["splits"].items()
    }
    return ConversationSplitBundle(
        splits=splits,
        seed=int(manifest["seed"]),
        algorithm=str(manifest["algorithm"]),
    )


def _validate_smoke_split_bundle(
    bundle: ConversationSplitBundle,
    *,
    config: SelectorExperimentConfig,
) -> None:
    expected_sizes = {
        "smoke_selection": config.smoke_selection_size,
        "smoke_validation": config.smoke_validation_size,
        "smoke_train": config.smoke_train_size,
    }
    if set(bundle.splits) != set(expected_sizes):
        raise ValueError("Smoke split names do not match the contract.")
    for name, expected_size in expected_sizes.items():
        if bundle[name].query_count != expected_size:
            raise ValueError(
                f"{name} has {bundle[name].query_count} queries, "
                f"expected {expected_size}."
            )
    gate = bundle["smoke_selection"]
    if (
        gate.conversation_count != ORACLE_GATE_EXPECTED_SIZE
        or gate.sample_id_sha256
        != config.expected_smoke_selection_sha256
    ):
        raise ValueError("Smoke selection does not match the NB03 gate.")
    conversation_sets = [
        set(split.conversation_ids)
        for split in bundle.splits.values()
    ]
    if any(
        left.intersection(right)
        for index, left in enumerate(conversation_sets)
        for right in conversation_sets[index + 1 :]
    ):
        raise ValueError("Smoke split conversations overlap.")


def _select_selector_oracle_gate(
    sources: Mapping[str, Any],
    config: SelectorExperimentConfig,
    *,
    name: str,
    progress: bool,
) -> ConversationSplit:
    split_frame = pd.DataFrame(
        [
            {
                "sample_id": str(row["sample_id"]),
                "conv_id": int(row["conv_id"]),
                "history_depth": int(row["history_len"]),
            }
            for row in sources["train_rows"]
        ]
    )
    eligible_population = build_depth_eligible_population(
        split_frame,
        depth_bounds=ORACLE_GATE_DEPTH_BOUNDS,
        depth_bin_order=ORACLE_GATE_DEPTH_BIN_ORDER,
    )
    selected_population = select_depth_balanced_population(
        eligible_population,
        depth_bin_order=ORACLE_GATE_DEPTH_BIN_ORDER,
        selection_order=ORACLE_GATE_SELECTION_ORDER,
        samples_per_bin=ORACLE_GATE_SAMPLES_PER_BIN,
        seed=config.seed,
        progress=progress,
    )
    sample_ids = tuple(
        sorted(selected_population["sample_id"].astype(str))
    )
    conversation_ids = tuple(
        sorted(
            selected_population["conv_id"].astype(int).unique(),
            key=lambda value: str(value),
        )
    )
    split = ConversationSplit(
        name=str(name),
        sample_ids=sample_ids,
        conversation_ids=conversation_ids,
        sample_id_sha256=_sample_id_sha256(sample_ids),
    )
    _validate_selector_oracle_gate(split, config=config)
    return split


def _validate_selector_oracle_gate(
    split: ConversationSplit,
    *,
    config: SelectorExperimentConfig,
) -> None:
    if (
        split.query_count != ORACLE_GATE_EXPECTED_SIZE
        or split.conversation_count != ORACLE_GATE_EXPECTED_SIZE
        or split.sample_id_sha256
        != config.expected_smoke_selection_sha256
    ):
        raise ValueError(
            "Population does not match the canonical NB03 600-query gate."
        )


def _sample_id_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(map(str, sample_ids))).encode("utf-8")
    ).hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(
    path: Path,
    *,
    progress: bool,
) -> str:
    digest = hashlib.sha256()
    total = path.stat().st_size
    tqdm = get_tqdm()
    with path.open("rb") as handle, tqdm(
        total=total,
        desc=f"hash {path.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        disable=not progress,
    ) as bar:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            bar.update(len(chunk))
    return digest.hexdigest()


def _runtime_device(
    device: str | torch.device | None,
) -> torch.device:
    return torch.device(
        device
        or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


__all__ = [
    "CANDIDATE_ORDER",
    "FULL_ARCHITECTURE_BOOTSTRAP_SEEDS",
    "FULL_CANDIDATE_ORDER",
    "FULL_CANDIDATE_RECIPE_VERSION",
    "FullSelectorArchitectureChoice",
    "SelectorCandidateChoice",
    "SelectorDecision",
    "SelectorExperimentConfig",
    "build_selector_oracle_gate",
    "build_selector_smoke_splits",
    "compare_selector_candidates",
    "compare_selector_metric",
    "compare_selector_mrr",
    "load_selector_decision",
    "load_selector_sources",
    "passes_selector_gate",
    "predict_selector_histories",
    "prepare_selector_datasets",
    "resolve_selector_decision",
    "run_collapsed_training",
    "run_final_selector_evaluation",
    "run_full_selector_candidates",
    "run_full_selector_training",
    "run_selector_retrieval_evaluation",
    "run_taxonomy_training",
    "save_selector_candidate_choice",
    "save_selector_decision",
    "select_full_selector_architecture",
    "select_selector_adaptation",
    "select_selector_candidate",
    "validate_selector_environment",
]
