"""High-level IterCQR dataloader pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import pandas as pd

from ..itercqr_components import (
    IterCQRQReCCRoccDataset,
    IterCQRRoccConfig,
    IterCQRTopiOCQARoccDataset,
    load_itercqr_tokenizer,
)
from ..progress import progress_iter
from .qrecc import QReCCResources, load_qrecc_collection_build_result, load_qrecc_frame, resolve_qrecc_resources
from .topiocqa import TopiOCQAResources, ensure_topiocqa_resources, load_topiocqa_frame


DatasetName = Literal["topiocqa", "qrecc"]
SplitName = Literal["train", "dev", "test", "all"]


@dataclass(frozen=True)
class IterCQRDataPipelineConfig:
    dataset_name: DatasetName = "topiocqa"
    splits: tuple[SplitName, ...] = ("train", "dev")
    batch_size: int = 32
    num_workers: int = 0
    data_dir: Path | None = None
    model_dir: Path | None = None
    force_download: bool = False
    progress: bool = True
    topiocqa_download_corpus: bool = True
    max_query_tokens: int = 32
    max_history_turn_tokens: int = 64
    max_input_tokens: int = 512
    pad_to_multiple_of: int | None = None
    max_history_answer_tokens: int = 32
    include_history: bool = True


@dataclass
class IterCQRDataPipelineResult:
    dataset_name: str
    dataloaders: dict[str, Any]
    tokenizer: Any
    frame: pd.DataFrame
    resources: TopiOCQAResources | QReCCResources | None
    config: IterCQRDataPipelineConfig
    tokenization_config: Any

    def dataloader(self, split: SplitName = "dev") -> Any:
        return self.dataloaders[split]


def load_itercqr_dataloader(
    *,
    dataset_name: DatasetName = "topiocqa",
    split: SplitName = "dev",
    batch_size: int = 32,
    num_workers: int = 0,
    data_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    force_download: bool = False,
    progress: bool = True,
    topiocqa_download_corpus: bool = True,
    max_query_tokens: int = 32,
    max_history_turn_tokens: int = 64,
    max_history_answer_tokens: int = 32,
    max_input_tokens: int = 512,
    pad_to_multiple_of: int | None = None,
    include_history: bool = True,
) -> Any:
    result = load_itercqr_dataloaders(
        dataset_name=dataset_name,
        splits=(split,),
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=data_dir,
        model_dir=model_dir,
        force_download=force_download,
        progress=progress,
        topiocqa_download_corpus=topiocqa_download_corpus,
        max_query_tokens=max_query_tokens,
        max_history_turn_tokens=max_history_turn_tokens,
        max_history_answer_tokens=max_history_answer_tokens,
        max_input_tokens=max_input_tokens,
        pad_to_multiple_of=pad_to_multiple_of,
        include_history=include_history,
    )
    return result.dataloader(split)


def load_itercqr_dataloaders(
    *,
    dataset_name: DatasetName = "topiocqa",
    splits: Sequence[SplitName] = ("train", "dev"),
    batch_size: int = 32,
    num_workers: int = 0,
    data_dir: Path | str | None = None,
    model_dir: Path | str | None = None,
    force_download: bool = False,
    progress: bool = True,
    topiocqa_download_corpus: bool = True,
    max_query_tokens: int = 32,
    max_history_turn_tokens: int = 64,
    max_history_answer_tokens: int = 32,
    max_input_tokens: int = 512,
    pad_to_multiple_of: int | None = None,
    include_history: bool = True,
) -> IterCQRDataPipelineResult:
    normalized_splits = tuple(_normalize_split(split) for split in splits)
    if not normalized_splits:
        raise ValueError("At least one split must be requested.")
    config = IterCQRDataPipelineConfig(
        dataset_name=dataset_name,
        splits=normalized_splits,
        batch_size=batch_size,
        num_workers=num_workers,
        data_dir=Path(data_dir).expanduser().resolve() if data_dir is not None else None,
        model_dir=Path(model_dir).expanduser().resolve() if model_dir is not None else None,
        force_download=force_download,
        progress=progress,
        topiocqa_download_corpus=topiocqa_download_corpus,
        max_query_tokens=max_query_tokens,
        max_history_turn_tokens=max_history_turn_tokens,
        max_history_answer_tokens=max_history_answer_tokens,
        max_input_tokens=max_input_tokens,
        pad_to_multiple_of=pad_to_multiple_of,
        include_history=include_history,
    )

    if dataset_name == "qrecc":
        return _load_qrecc_itercqr_dataloaders(config)
    if dataset_name != "topiocqa":
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    return _load_topiocqa_itercqr_dataloaders(config)


def _load_topiocqa_itercqr_dataloaders(
    config: IterCQRDataPipelineConfig,
) -> IterCQRDataPipelineResult:
    steps = progress_iter(
        ["resources", "frame", "tokenizer", "dataloaders"],
        total=4,
        desc="IterCQR dataloading",
        unit="step",
        enabled=config.progress,
    )

    resources: TopiOCQAResources | None = None
    frame: pd.DataFrame | None = None
    tokenizer = None
    dataloaders: dict[str, Any] = {}

    for step in steps:
        if step == "resources":
            resources = ensure_topiocqa_resources(
                config.data_dir,
                force_download=config.force_download,
                download_corpus=config.topiocqa_download_corpus,
                progress=config.progress,
            )
        elif step == "frame":
            if resources is None:
                raise RuntimeError("TopiOCQA resources must be prepared before loading frame.")
            frame = load_topiocqa_frame(resources)
        elif step == "tokenizer":
            tokenizer = load_itercqr_tokenizer(config.model_dir)
        elif step == "dataloaders":
            if frame is None or tokenizer is None:
                raise RuntimeError("Frame and tokenizer must be loaded before dataloaders.")
            tokenization_config = IterCQRRoccConfig(
                max_query_tokens=config.max_query_tokens,
                max_history_turn_tokens=config.max_history_turn_tokens,
                max_history_answer_tokens=config.max_history_answer_tokens,
                max_input_tokens=config.max_input_tokens,
                pad_to_multiple_of=config.pad_to_multiple_of,
                include_history=config.include_history,
            )
            for split in config.splits:
                split_frame = frame if split == "all" else frame[frame["split"].eq(split)]
                dataset = IterCQRTopiOCQARoccDataset(
                    split_frame.reset_index(drop=True),
                    tokenizer,
                    tokenization_config,
                )
                dataloaders[split] = dataset.make_dataloader(
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                )

    if frame is None or tokenizer is None or resources is None:
        raise RuntimeError("IterCQR dataloader pipeline did not finish.")

    tokenization_config = next(iter(dataloaders.values())).dataset.config
    return IterCQRDataPipelineResult(
        dataset_name="topiocqa",
        dataloaders=dataloaders,
        tokenizer=tokenizer,
        frame=frame,
        resources=resources,
        config=config,
        tokenization_config=tokenization_config,
    )


def _load_qrecc_itercqr_dataloaders(
    config: IterCQRDataPipelineConfig,
) -> IterCQRDataPipelineResult:
    steps = progress_iter(
        ["resources", "frame", "tokenizer", "dataloaders"],
        total=4,
        desc="IterCQR QReCC dataloading",
        unit="step",
        enabled=config.progress,
    )

    resources: QReCCResources | None = None
    frame: pd.DataFrame | None = None
    all_frame: pd.DataFrame | None = None
    tokenizer = None
    dataloaders: dict[str, Any] = {}

    for step in steps:
        if step == "resources":
            resources = resolve_qrecc_resources(config.data_dir)
            load_qrecc_collection_build_result(resources, require_collection=False)
        elif step == "frame":
            if resources is None:
                raise RuntimeError("QReCC resources must be resolved before loading frame.")
            all_frame = load_qrecc_frame(resources, prefer_processed=True, gold_only=False)
            frame = all_frame[all_frame["positive_ctx_passage_ids"].apply(bool)].reset_index(drop=True)
        elif step == "tokenizer":
            tokenizer = load_itercqr_tokenizer(config.model_dir)
        elif step == "dataloaders":
            if all_frame is None or tokenizer is None:
                raise RuntimeError("Frame and tokenizer must be loaded before dataloaders.")
            tokenization_config = IterCQRRoccConfig(
                max_query_tokens=config.max_query_tokens,
                max_history_turn_tokens=config.max_history_turn_tokens,
                max_history_answer_tokens=config.max_history_answer_tokens,
                max_input_tokens=config.max_input_tokens,
                pad_to_multiple_of=config.pad_to_multiple_of,
                include_history=config.include_history,
            )
            for split in config.splits:
                split_frame = all_frame if split == "all" else all_frame[all_frame["split"].eq(split)]
                dataset = IterCQRQReCCRoccDataset(
                    split_frame.reset_index(drop=True),
                    tokenizer,
                    tokenization_config,
                )
                dataloaders[split] = dataset.make_dataloader(
                    batch_size=config.batch_size,
                    num_workers=config.num_workers,
                )

    if frame is None or tokenizer is None or resources is None:
        raise RuntimeError("QReCC IterCQR dataloader pipeline did not finish.")

    tokenization_config = next(iter(dataloaders.values())).dataset.config
    return IterCQRDataPipelineResult(
        dataset_name="qrecc",
        dataloaders=dataloaders,
        tokenizer=tokenizer,
        frame=frame,
        resources=resources,
        config=config,
        tokenization_config=tokenization_config,
    )


def _normalize_split(split: str) -> SplitName:
    if split not in {"train", "dev", "test", "all"}:
        raise ValueError(f"Unsupported split: {split}")
    return split  # type: ignore[return-value]
