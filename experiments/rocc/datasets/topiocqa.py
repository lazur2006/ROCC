"""TopiOCQA resource preparation and frame loading."""

from __future__ import annotations

import csv
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit
from urllib.request import urlopen

import pandas as pd

from ..paths import project_path
from ..progress import download_file, progress_iter


TOPIOCQA_CORPUS_PASSAGES = 25_700_592
TOPIOCQA_ANCE_SHARD_PASSAGES = 800_000
TOPIOCQA_AZURE_FILE_SHARE_URL = ""  # Legacy Azure helpers require an explicit user-supplied endpoint.
TOPIOCQA_AZURE_ANCE_SHARDS_PATH = ""  # Legacy Azure helpers require an explicit user-supplied endpoint.
TOPIOCQA_AZURE_ANCE_COLLECTION_PATH = ""  # Legacy Azure helpers require an explicit user-supplied endpoint.


@dataclass(frozen=True)
class DownloadSpec:
    url: str
    relative_path: Path
    label: str


@dataclass(frozen=True)
class TopiOCQAResources:
    root: Path
    train_json: Path
    dev_json: Path
    corpus_tsv: Path

    @property
    def data_root(self) -> Path:
        return self.root / "downloads" / "data"


@dataclass(frozen=True)
class TopiOCQATurn:
    turn_id: int
    question: str
    answer: str


@dataclass(frozen=True)
class TopiOCQAConversationSample:
    sample_id: str
    split: str
    conv_id: int
    turn_id: int
    current_query: str
    history: list[TopiOCQATurn]


@dataclass(frozen=True)
class TopiOCQAArtifactStatus:
    root: Path
    train_json: Path
    dev_json: Path
    corpus_tsv: Path
    bm25_collection_jsonl: Path
    bm25_index_dir: Path
    ance_collection_jsonl: Path
    ance_index_dir: Path
    raw_train_dev_ready: bool
    corpus_ready: bool
    bm25_collection_ready: bool
    bm25_index_ready: bool
    ance_collection_ready: bool
    ance_index_ready: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "raw_train_dev_ready": self.raw_train_dev_ready,
            "corpus_ready": self.corpus_ready,
            "bm25_collection_ready": self.bm25_collection_ready,
            "bm25_index_ready": self.bm25_index_ready,
            "ance_collection_ready": self.ance_collection_ready,
            "ance_index_ready": self.ance_index_ready,
            "train_json": str(self.train_json),
            "dev_json": str(self.dev_json),
            "corpus_tsv": str(self.corpus_tsv),
            "bm25_collection_jsonl": str(self.bm25_collection_jsonl),
            "bm25_index_dir": str(self.bm25_index_dir),
            "ance_collection_jsonl": str(self.ance_collection_jsonl),
            "ance_index_dir": str(self.ance_index_dir),
        }


@dataclass(frozen=True)
class TopiOCQAResourcePreparationResult:
    resources: TopiOCQAResources
    artifacts: TopiOCQAArtifactStatus
    artifacts_before: TopiOCQAArtifactStatus
    bm25_status: Any
    azure_ance_status: TopiOCQAAzureAnceIndexStatus | None
    bm25_log_file: Path
    azure_ance_sas_file: Path | None
    raw_download_needed: bool
    corpus_download_needed: bool
    downloaded: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.resources.root),
            "raw_train_dev_ready": self.artifacts.raw_train_dev_ready,
            "corpus_ready": self.artifacts.corpus_ready,
            "bm25_collection_ready": self.artifacts.bm25_collection_ready,
            "bm25_index_complete": bool(getattr(self.bm25_status, "complete", False)),
            "ance_collection_ready": self.artifacts.ance_collection_ready,
            "local_ance_index_ready": self.artifacts.ance_index_ready,
            "azure_ance_index_ready": bool(
                self.azure_ance_status and self.azure_ance_status.ready
            ),
            "raw_download_needed": self.raw_download_needed,
            "corpus_download_needed": self.corpus_download_needed,
            "downloaded": self.downloaded,
            "bm25_index_dir": str(self.artifacts.bm25_index_dir),
            "bm25_log_file": str(self.bm25_log_file),
            "ance_index_dir": str(self.artifacts.ance_index_dir),
            "azure_ance_sas_file": (
                str(self.azure_ance_sas_file) if self.azure_ance_sas_file is not None else None
            ),
        }


@dataclass(frozen=True)
class TopiOCQAAzureAnceIndexStatus:
    target_base_url: str
    checked: bool
    ready: bool
    expected_shards: int
    complete_shards: int
    first_shard_id: int | None
    last_shard_id: int | None
    indexed_passages: int | None
    sampled_manifest_complete: bool
    validated_manifests: int
    error: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "target_base_url": self.target_base_url,
            "checked": self.checked,
            "ready": self.ready,
            "expected_shards": self.expected_shards,
            "complete_shards": self.complete_shards,
            "first_shard_id": self.first_shard_id,
            "last_shard_id": self.last_shard_id,
            "indexed_passages": self.indexed_passages,
            "sampled_manifest_complete": self.sampled_manifest_complete,
            "validated_manifests": self.validated_manifests,
            "error": self.error,
        }


@dataclass(frozen=True)
class TopiOCQAAnceAzureIndexSyncResult:
    target_dir: Path
    local_index_dir: Path
    log_file: Path
    azure_target_url: str
    azure_ready: bool
    azure_complete_shards: int
    local_complete_shards: int
    local_indexed_passages: int
    downloaded: bool
    skipped_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "target_dir": str(self.target_dir),
            "local_index_dir": str(self.local_index_dir),
            "log_file": str(self.log_file),
            "azure_target_url": self.azure_target_url,
            "azure_ready": self.azure_ready,
            "azure_complete_shards": self.azure_complete_shards,
            "local_complete_shards": self.local_complete_shards,
            "local_indexed_passages": self.local_indexed_passages,
            "downloaded": self.downloaded,
            "skipped_existing": self.skipped_existing,
        }


@dataclass(frozen=True)
class TopiOCQAAnceCollectionAzureSyncResult:
    source_file: Path
    azure_target_url: str
    uploaded: bool
    skipped_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "source_file": str(self.source_file),
            "azure_target_url": self.azure_target_url,
            "uploaded": self.uploaded,
            "skipped_existing": self.skipped_existing,
        }


DOWNLOADS = [
    DownloadSpec(
        url="https://zenodo.org/records/6151011/files/data/retriever/original/train.json",
        relative_path=Path("downloads/data/retriever/original/train.json"),
        label="TopiOCQA train",
    ),
    DownloadSpec(
        url="https://zenodo.org/records/6151011/files/data/retriever/original/dev.json",
        relative_path=Path("downloads/data/retriever/original/dev.json"),
        label="TopiOCQA dev",
    ),
    DownloadSpec(
        url="https://zenodo.org/records/6149599/files/data/wikipedia_split/full_wiki_segments.tsv",
        relative_path=Path("downloads/data/wikipedia_split/full_wiki_segments.tsv"),
        label="TopiOCQA wiki corpus",
    ),
]


def default_topiocqa_root(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()

    return project_path("experiments", "data", "topiocqa")


def resolve_topiocqa_resources(data_dir: Path | str | None = None) -> TopiOCQAResources:
    """Resolve expected TopiOCQA resource paths without downloading anything."""

    return _resources_for_root(default_topiocqa_root(data_dir))


def inspect_topiocqa_artifacts(
    resources: TopiOCQAResources | Path | str | None = None,
    *,
    data_dir: Path | str | None = None,
) -> TopiOCQAArtifactStatus:
    """Report local TopiOCQA raw, sparse, and dense index artifacts."""

    if isinstance(resources, TopiOCQAResources):
        resolved_resources = resources
    elif resources is not None:
        resolved_resources = resolve_topiocqa_resources(resources)
    else:
        resolved_resources = resolve_topiocqa_resources(data_dir)

    bm25_collection_jsonl = project_path(
        "experiments",
        "data",
        "pyserini_bm25_lucene_topiocqa",
        "collection",
        "docs.jsonl",
    )
    bm25_index_dir = project_path(
        "experiments",
        "data",
        "pyserini_bm25_lucene_topiocqa",
        "lucene_index",
    )
    ance_collection_jsonl = project_path(
        "experiments",
        "data",
        "pyserini_ance_faiss_topiocqa",
        "collection",
        "full_wiki_segments_ance.jsonl",
    )
    ance_index_dir = project_path(
        "experiments",
        "data",
        "pyserini_ance_faiss_topiocqa",
        "faiss_flat_index_full_sharded",
    )
    return TopiOCQAArtifactStatus(
        root=resolved_resources.root,
        train_json=resolved_resources.train_json,
        dev_json=resolved_resources.dev_json,
        corpus_tsv=resolved_resources.corpus_tsv,
        bm25_collection_jsonl=bm25_collection_jsonl,
        bm25_index_dir=bm25_index_dir,
        ance_collection_jsonl=ance_collection_jsonl,
        ance_index_dir=ance_index_dir,
        raw_train_dev_ready=resolved_resources.train_json.exists() and resolved_resources.dev_json.exists(),
        corpus_ready=resolved_resources.corpus_tsv.exists(),
        bm25_collection_ready=bm25_collection_jsonl.exists(),
        bm25_index_ready=_dir_has_entries(bm25_index_dir),
        ance_collection_ready=ance_collection_jsonl.exists(),
        ance_index_ready=_is_topiocqa_dense_index_ready(ance_index_dir),
    )


def inspect_topiocqa_azure_ance_index(
    *,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = TOPIOCQA_AZURE_FILE_SHARE_URL,
    azure_shards_path: str = TOPIOCQA_AZURE_ANCE_SHARDS_PATH,
    expected_shards: int | None = None,
    validate_manifests: bool = False,
    timeout: float = 30.0,
) -> TopiOCQAAzureAnceIndexStatus:
    """Inspect the remote TopiOCQA ANCE shard directory in Azure File Share."""

    target_base_url = f"{azure_file_share_url.rstrip('/')}/{azure_shards_path.strip('/')}"
    expected = expected_shards or expected_topiocqa_ance_shards()
    try:
        sas_query = _topiocqa_sas_query_from_file(sas_file=sas_file, url_file=url_file)
        shard_names = _azure_list_directory_names(target_base_url, sas_query, timeout=timeout)
        shard_ids = sorted(
            int(name.split("_", 1)[1])
            for name in shard_names
            if name.startswith("shard_") and name.split("_", 1)[1].isdigit()
        )
        sample_ids = sorted({sid for sid in (0, expected - 1) if sid in shard_ids})
        sampled_complete = True
        for shard_id in sample_ids:
            manifest = _azure_read_json_file(
                f"{target_base_url}/shard_{shard_id:06d}/manifest.json",
                sas_query,
                timeout=timeout,
            )
            sampled_complete = sampled_complete and manifest.get("complete") is True

        indexed_passages: int | None = None
        validated_manifests = 0
        complete_shards = len(shard_ids)
        if validate_manifests:
            indexed_passages = 0
            complete_shards = 0
            for shard_id in shard_ids:
                manifest = _azure_read_json_file(
                    f"{target_base_url}/shard_{shard_id:06d}/manifest.json",
                    sas_query,
                    timeout=timeout,
                )
                validated_manifests += 1
                if manifest.get("complete") is True:
                    complete_shards += 1
                    indexed_passages += int(manifest.get("num_passages", 0))

        first_shard_id = shard_ids[0] if shard_ids else None
        last_shard_id = shard_ids[-1] if shard_ids else None
        ready = (
            complete_shards >= expected
            and first_shard_id == 0
            and last_shard_id is not None
            and last_shard_id >= expected - 1
            and sampled_complete
            and (indexed_passages is None or indexed_passages >= TOPIOCQA_CORPUS_PASSAGES)
        )
        return TopiOCQAAzureAnceIndexStatus(
            target_base_url=target_base_url,
            checked=True,
            ready=ready,
            expected_shards=expected,
            complete_shards=complete_shards,
            first_shard_id=first_shard_id,
            last_shard_id=last_shard_id,
            indexed_passages=indexed_passages,
            sampled_manifest_complete=sampled_complete and bool(sample_ids),
            validated_manifests=validated_manifests,
            error=None,
        )
    except Exception as exc:
        return TopiOCQAAzureAnceIndexStatus(
            target_base_url=target_base_url,
            checked=False,
            ready=False,
            expected_shards=expected,
            complete_shards=0,
            first_shard_id=None,
            last_shard_id=None,
            indexed_passages=None,
            sampled_manifest_complete=False,
            validated_manifests=0,
            error=str(exc),
        )


def sync_topiocqa_ance_index_from_azure(
    *,
    local_index_dir: Path | str | None = None,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = TOPIOCQA_AZURE_FILE_SHARE_URL,
    azure_shards_path: str = TOPIOCQA_AZURE_ANCE_SHARDS_PATH,
    validate_azure: bool = True,
    validate_azure_manifests: bool = False,
    log_file: Path | str = "/content/azcopy_topiocqa_ance_index_download.log",
    install_azcopy: bool = True,
    progress: bool = True,
    timeout: float = 30.0,
) -> TopiOCQAAnceAzureIndexSyncResult:
    """Download the finished TopiOCQA ANCE shard index from Azure.

    This function never deletes remote Azure files. It only reads the remote
    shard directory and copies it to ``local_index_dir`` with AzCopy.
    """

    target_dir = (
        Path(local_index_dir).expanduser().resolve()
        if local_index_dir is not None
        else project_path(
            "experiments",
            "data",
            "pyserini_ance_faiss_topiocqa",
            "faiss_flat_index_full_sharded",
        )
    )
    resolved_log_file = Path(log_file).expanduser().resolve()
    before_ready = _is_topiocqa_dense_index_ready(target_dir)
    before_summary = _safe_topiocqa_dense_summary(target_dir)

    azure_status = inspect_topiocqa_azure_ance_index(
        sas_file=sas_file,
        url_file=url_file,
        azure_file_share_url=azure_file_share_url,
        azure_shards_path=azure_shards_path,
        expected_shards=expected_topiocqa_ance_shards(),
        validate_manifests=validate_azure_manifests,
        timeout=timeout,
    )
    if validate_azure and not azure_status.ready:
        raise RuntimeError(f"TopiOCQA ANCE index is not complete on Azure: {azure_status.to_dict()}")

    target_base_url = f"{azure_file_share_url.rstrip('/')}/{azure_shards_path.strip('/')}"
    redacted_target_url = f"{target_base_url}?***"
    if before_ready:
        return TopiOCQAAnceAzureIndexSyncResult(
            target_dir=target_dir,
            local_index_dir=before_summary.index_dir if before_summary else target_dir,
            log_file=resolved_log_file,
            azure_target_url=redacted_target_url,
            azure_ready=azure_status.ready,
            azure_complete_shards=azure_status.complete_shards,
            local_complete_shards=before_summary.shard_count if before_summary else 0,
            local_indexed_passages=before_summary.num_passages if before_summary else 0,
            downloaded=False,
            skipped_existing=True,
        )

    sas_query = _topiocqa_sas_query_from_file(sas_file=sas_file, url_file=url_file)
    target_url = f"{target_base_url}?{sas_query}"
    from ..dense_ance import download_dense_ance_index_from_azure

    downloaded_index_dir = download_dense_ance_index_from_azure(
        azure_sharded_dir_url=target_url,
        local_sharded_dir=target_dir,
        log_path=resolved_log_file,
        install_azcopy=install_azcopy,
        progress=progress,
    )
    after_summary = _safe_topiocqa_dense_summary(downloaded_index_dir)
    if (
        after_summary is None
        or after_summary.shard_count < expected_topiocqa_ance_shards()
        or after_summary.num_passages < TOPIOCQA_CORPUS_PASSAGES
        or not after_summary.contiguous
    ):
        raise RuntimeError(
            "Downloaded TopiOCQA ANCE index is incomplete: "
            f"{after_summary.to_dict() if after_summary else None}"
        )

    return TopiOCQAAnceAzureIndexSyncResult(
        target_dir=target_dir,
        local_index_dir=downloaded_index_dir,
        log_file=resolved_log_file,
        azure_target_url=redacted_target_url,
        azure_ready=azure_status.ready,
        azure_complete_shards=azure_status.complete_shards,
        local_complete_shards=after_summary.shard_count,
        local_indexed_passages=after_summary.num_passages,
        downloaded=True,
        skipped_existing=False,
    )


def sync_topiocqa_ance_collection_to_azure(
    *,
    collection_file: Path | str | None = None,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = TOPIOCQA_AZURE_FILE_SHARE_URL,
    azure_collection_path: str = TOPIOCQA_AZURE_ANCE_COLLECTION_PATH,
    azcopy_path: Path | str | None = None,
    install_azcopy: bool = True,
    progress: bool = True,
) -> TopiOCQAAnceCollectionAzureSyncResult:
    """Upload the TopiOCQA ANCE collection JSONL to Azure Files.

    This writes one collection file only. It does not delete or modify the
    remote FAISS shard directory.
    """

    source = (
        Path(collection_file).expanduser().resolve()
        if collection_file is not None
        else project_path(
            "experiments",
            "data",
            "pyserini_ance_faiss_topiocqa",
            "collection",
            "full_wiki_segments_ance.jsonl",
        )
    )
    if not source.exists():
        raise FileNotFoundError(f"TopiOCQA ANCE collection is missing: {source}")

    sas_query = _topiocqa_sas_query_from_file(sas_file=sas_file, url_file=url_file)
    target_base_url = f"{azure_file_share_url.rstrip('/')}/{azure_collection_path.strip('/')}"
    target_url = f"{target_base_url}?{sas_query}"
    redacted_target_url = f"{target_base_url}?***"

    import subprocess
    import shutil

    from ..dense_ance import ensure_azcopy

    if azcopy_path is not None:
        azcopy = Path(azcopy_path).expanduser().resolve()
        if not azcopy.exists() and install_azcopy:
            azcopy = ensure_azcopy(azcopy)
    elif install_azcopy:
        azcopy = ensure_azcopy()
    else:
        existing = shutil.which("azcopy")
        azcopy = Path(existing) if existing else Path("azcopy")

    if not azcopy.exists() and shutil.which(str(azcopy)) is None:
        raise RuntimeError("azcopy is not available. Install azcopy or pass install_azcopy=True.")

    command = [
        str(azcopy),
        "copy",
        str(source),
        target_url,
        "--from-to=LocalFileSMB",
        "--overwrite=ifSourceNewer",
        "--check-length=true",
        "--output-type=text",
        "--log-level=INFO",
    ]
    if progress:
        print("upload TopiOCQA ANCE collection to Azure:", redacted_target_url, flush=True)
        subprocess.run(command, check=True)
    else:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

    return TopiOCQAAnceCollectionAzureSyncResult(
        source_file=source,
        azure_target_url=redacted_target_url,
        uploaded=True,
        skipped_existing=False,
    )


def expected_topiocqa_ance_shards() -> int:
    return math.ceil(TOPIOCQA_CORPUS_PASSAGES / TOPIOCQA_ANCE_SHARD_PASSAGES)


def prepare_topiocqa_resources_for_indexing(
    data_dir: Path | str | None = None,
    *,
    check_azure_ance: bool = False,
    azure_ance_sas_file: Path | str | None = None,
    bm25_log_file: Path | str | None = None,
    progress: bool = True,
    timeout: float = 60.0,
) -> TopiOCQAResourcePreparationResult:
    """Prepare only the TopiOCQA raw files still needed by the index notebook."""

    resources = resolve_topiocqa_resources(data_dir)
    artifacts_before = inspect_topiocqa_artifacts(resources)
    resolved_bm25_log_file = (
        Path(bm25_log_file).expanduser().resolve()
        if bm25_log_file is not None
        else project_path("experiments", "logs", "topiocqa_bm25_lucene_index.log")
    )

    from ..azure_indexing import inspect_bm25_lucene_index

    bm25_status = inspect_bm25_lucene_index(
        artifacts_before.bm25_index_dir,
        source_passages=TOPIOCQA_CORPUS_PASSAGES,
        log_file=resolved_bm25_log_file,
    )
    resolved_azure_sas_file = (
        Path(azure_ance_sas_file).expanduser().resolve()
        if azure_ance_sas_file is not None
        else _default_topiocqa_ance_sas_file()
    )
    azure_ance_status = (
        inspect_topiocqa_azure_ance_index(
            sas_file=resolved_azure_sas_file,
            validate_manifests=False,
            timeout=timeout,
        )
        if check_azure_ance
        else None
    )

    corpus_downstream_ready = (
        artifacts_before.bm25_collection_ready
        or bm25_status.complete
        or artifacts_before.ance_collection_ready
        or artifacts_before.ance_index_ready
        or bool(azure_ance_status and azure_ance_status.ready)
    )
    raw_download_needed = not artifacts_before.raw_train_dev_ready
    corpus_download_needed = not artifacts_before.corpus_ready and not corpus_downstream_ready

    downloaded = raw_download_needed or corpus_download_needed
    if downloaded:
        resources = ensure_topiocqa_resources(
            data_dir=resources.root,
            download_corpus=corpus_download_needed,
            progress=progress,
            timeout=timeout,
        )

    artifacts = inspect_topiocqa_artifacts(resources)
    return TopiOCQAResourcePreparationResult(
        resources=resources,
        artifacts=artifacts,
        artifacts_before=artifacts_before,
        bm25_status=bm25_status,
        azure_ance_status=azure_ance_status,
        bm25_log_file=resolved_bm25_log_file,
        azure_ance_sas_file=resolved_azure_sas_file,
        raw_download_needed=raw_download_needed,
        corpus_download_needed=corpus_download_needed,
        downloaded=downloaded,
    )


def ensure_topiocqa_resources(
    data_dir: Path | str | None = None,
    *,
    force_download: bool = False,
    download_corpus: bool = True,
    progress: bool = True,
    timeout: float = 60.0,
) -> TopiOCQAResources:
    resources = _resources_for_root(default_topiocqa_root(data_dir))
    downloads = DOWNLOADS if download_corpus else DOWNLOADS[:2]
    resource_bar = progress_iter(
        downloads,
        total=len(downloads),
        desc="prepare TopiOCQA",
        unit="file",
        enabled=progress,
    )
    for spec in resource_bar:
        target = resources.root / spec.relative_path
        download_file(
            spec.url,
            target,
            desc=spec.label,
            force=force_download,
            timeout=timeout,
            progress=progress,
        )

    missing = [
        str(path)
        for path in (
            (resources.train_json, resources.dev_json, resources.corpus_tsv)
            if download_corpus
            else (resources.train_json, resources.dev_json)
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"TopiOCQA resources missing after preparation: {missing}")
    return resources


def load_topiocqa_frame(
    resources: TopiOCQAResources | None = None,
    *,
    splits: Iterable[str] = ("train", "dev"),
) -> pd.DataFrame:
    """Load only the requested TopiOCQA split files."""

    resources = resources or ensure_topiocqa_resources()
    selected = tuple(str(split) for split in splits)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("splits must contain unique train/dev names.")
    unknown = set(selected).difference({"train", "dev"})
    if unknown:
        raise ValueError(
            "Unknown TopiOCQA split: " + ", ".join(sorted(unknown))
        )
    paths = {
        "train": resources.train_json,
        "dev": resources.dev_json,
    }
    frames = [
        _load_retriever_json(paths[split], split=split)
        for split in selected
    ]
    return _add_topic_switch_columns(
        pd.concat(frames, ignore_index=True)
    )


def build_topiocqa_conversation_samples(
    frame: pd.DataFrame,
    *,
    minimum_history_depth: int = 2,
    progress: bool = True,
) -> list[TopiOCQAConversationSample]:
    """Build ordered conversational samples with explicit history turn IDs."""

    samples: list[TopiOCQAConversationSample] = []
    ordered = frame.sort_values(["split", "conv_id", "turn_id"])
    conversations = ordered.groupby(["split", "conv_id"], sort=True)
    groups = progress_iter(
        conversations,
        total=conversations.ngroups,
        desc="build TopiOCQA conversation samples",
        unit="conversation",
        enabled=progress,
    )
    for (split, conv_id), conversation in groups:
        history: list[TopiOCQATurn] = []
        for row in conversation.itertuples(index=False):
            if len(history) >= minimum_history_depth:
                samples.append(
                    TopiOCQAConversationSample(
                        sample_id=str(row.sample_id),
                        split=str(split),
                        conv_id=int(conv_id),
                        turn_id=int(row.turn_id),
                        current_query=str(row.question),
                        history=list(history),
                    )
                )
            history.append(
                TopiOCQATurn(
                    turn_id=int(row.turn_id),
                    question=str(row.question),
                    answer=_first_topiocqa_answer(row.answers),
                )
            )
    return samples


def _first_topiocqa_answer(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else ""
    return "" if value is None else str(value)


def preview_topiocqa_corpus(
    resources: TopiOCQAResources | None = None,
    *,
    n: int = 3,
) -> pd.DataFrame:
    resources = resources or ensure_topiocqa_resources()
    rows = []
    for row in iter_topiocqa_corpus_rows(resources.corpus_tsv):
        rows.append(row)
        if len(rows) >= n:
            break
    return pd.DataFrame(rows)


def iter_topiocqa_corpus_rows(
    corpus_tsv: Path,
    *,
    max_passages: int | None = None,
) -> Iterable[dict[str, str]]:
    with corpus_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            if max_passages is not None and index >= max_passages:
                break
            yield row


def load_topiocqa_passage_lookup(
    corpus_tsv: Path,
    passage_ids: Iterable[str],
    *,
    progress: bool = True,
) -> dict[str, dict[str, str]]:
    wanted = {str(passage_id) for passage_id in passage_ids}
    found: dict[str, dict[str, str]] = {}
    if not wanted:
        return found

    rows = iter_topiocqa_corpus_rows(corpus_tsv)
    rows = progress_iter(
        rows,
        total=TOPIOCQA_CORPUS_PASSAGES,
        desc="scan TopiOCQA corpus",
        unit="passage",
        enabled=progress,
    )
    for row in rows:
        passage_id = str(row["id"])
        if passage_id in wanted:
            found[passage_id] = row
            if len(found) == len(wanted):
                break
    return found


def compare_positive_contexts_to_lookup(
    frame: pd.DataFrame,
    corpus_lookup: dict[str, dict[str, str]],
) -> pd.DataFrame:
    rows = []
    for _, turn in frame.sort_values(["split", "conv_id", "turn_id"]).iterrows():
        for ctx_index, ctx in enumerate(turn["positive_ctxs"] or []):
            passage_id = ctx.get("passage_id")
            passage_id = str(passage_id) if passage_id is not None else None
            wiki_row = corpus_lookup.get(passage_id) if passage_id is not None else None
            wiki_text = wiki_row["text"] if wiki_row else None
            positive_ctx_text = ctx.get("text")
            rows.append(
                {
                    "split": turn["split"],
                    "conv_id": turn["conv_id"],
                    "turn_id": turn["turn_id"],
                    "question": turn["question"],
                    "positive_ctx_index": ctx_index,
                    "passage_id": passage_id,
                    "topic_document": turn["topic_document"],
                    "topic_id": turn["topic_id"],
                    "wiki_passage_found": wiki_row is not None,
                    "exact_match": wiki_text == positive_ctx_text,
                    "wiki_text": wiki_text,
                    "positive_ctx_text": positive_ctx_text,
                }
            )
    return pd.DataFrame(rows)


def summarize_topiocqa_split(
    label: str,
    frame: pd.DataFrame,
    context_matches: pd.DataFrame | None = None,
) -> dict[str, object]:
    conversation_turn_counts = frame.groupby(["split", "conv_id"])["turn_id"].nunique()
    conversation_topic_stats = (
        frame.groupby(["split", "conv_id"])
        .agg(
            topics=("topic_document", lambda values: values.nunique(dropna=True)),
            topic_switches=("topic_switch", "sum"),
        )
    )
    turns_without_positive_ctx_passage_ids = int(
        (frame["positive_ctx_passage_ids"].apply(len) == 0).sum()
    )

    if context_matches is None or context_matches.empty:
        positive_ctxs_checked = 0
        missing_wiki_passages = 0
        exact_matches = 0
        non_exact_matches = 0
        all_positive_ctx_texts_exact_match = None
    else:
        positive_ctxs_checked = len(context_matches)
        missing_wiki_passages = int((~context_matches["wiki_passage_found"]).sum())
        exact_matches = int(context_matches["exact_match"].sum())
        non_exact_matches = int((~context_matches["exact_match"]).sum())
        all_positive_ctx_texts_exact_match = bool(context_matches["exact_match"].all())

    return {
        "split": label,
        "conversations": int(frame.groupby(["split", "conv_id"]).ngroups),
        "turns": int(len(frame)),
        "longest_conversation_turns": int(conversation_turn_counts.max()),
        "shortest_conversation_turns": int(conversation_turn_counts.min()),
        "mean_turns_per_conversation": round(float(conversation_turn_counts.mean()), 2),
        "min_topics_per_conversation": int(conversation_topic_stats["topics"].min()),
        "max_topics_per_conversation": int(conversation_topic_stats["topics"].max()),
        "mean_topics_per_conversation": round(float(conversation_topic_stats["topics"].mean()), 2),
        "min_topic_switches_per_conversation": int(conversation_topic_stats["topic_switches"].min()),
        "max_topic_switches_per_conversation": int(conversation_topic_stats["topic_switches"].max()),
        "mean_topic_switches_per_conversation": round(
            float(conversation_topic_stats["topic_switches"].mean()),
            2,
        ),
        "turns_without_positive_ctx_passage_ids": turns_without_positive_ctx_passage_ids,
        "positive_ctxs_checked": positive_ctxs_checked,
        "missing_wiki_passages": missing_wiki_passages,
        "exact_text_matches": exact_matches,
        "non_exact_text_matches": non_exact_matches,
        "all_positive_ctx_texts_exact_match": all_positive_ctx_texts_exact_match,
    }


def positive_passage_ids(frame: pd.DataFrame) -> list[str]:
    ids: list[str] = []
    for contexts in frame["positive_ctxs"]:
        for ctx in contexts or []:
            passage_id = ctx.get("passage_id")
            if passage_id is not None:
                ids.append(str(passage_id))
    return ids


def _resources_for_root(root: Path) -> TopiOCQAResources:
    data_root = root / "downloads" / "data"
    return TopiOCQAResources(
        root=root,
        train_json=data_root / "retriever" / "original" / "train.json",
        dev_json=data_root / "retriever" / "original" / "dev.json",
        corpus_tsv=data_root / "wikipedia_split" / "full_wiki_segments.tsv",
    )


def _load_retriever_json(path: Path, split: str) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    frame = pd.DataFrame(records)
    frame.insert(0, "split", split)
    return frame


def _extract_positive_ctx_passage_ids(positive_ctxs: list[dict[str, Any]]) -> list[str]:
    return [
        str(ctx["passage_id"])
        for ctx in positive_ctxs or []
        if ctx.get("passage_id") is not None
    ]


def _extract_positive_ctx_topics(positive_ctxs: list[dict[str, Any]]) -> list[str]:
    topics = []
    for ctx in positive_ctxs or []:
        title = ctx.get("title")
        if title:
            topics.append(str(title).split(" [SEP] ", 1)[0])
    return topics


def _add_topic_switch_columns(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = frame.sort_values(["split", "conv_id", "turn_id"]).reset_index(drop=True).copy()
    annotated["positive_ctx_passage_ids"] = annotated["positive_ctxs"].apply(
        _extract_positive_ctx_passage_ids
    )
    annotated["positive_ctx_topics"] = annotated["positive_ctxs"].apply(
        _extract_positive_ctx_topics
    )
    annotated["topic_document"] = annotated["positive_ctx_topics"].apply(
        lambda topics: topics[0] if topics else None
    )
    annotated["topic_id"] = None
    annotated["topic_switch"] = False

    for _, indices in annotated.groupby(["split", "conv_id"], sort=False).groups.items():
        topic_to_id = {}
        previous_topic = None
        for index in indices:
            topic = annotated.at[index, "topic_document"]
            if topic is None:
                continue
            if topic not in topic_to_id:
                topic_to_id[topic] = len(topic_to_id) + 1
            annotated.at[index, "topic_id"] = topic_to_id[topic]
            annotated.at[index, "topic_switch"] = (
                previous_topic is not None and topic != previous_topic
            )
            previous_topic = topic

    annotated["topic_id"] = annotated["topic_id"].astype("Int64")
    annotated["topic_switch_depth"] = (annotated["topic_id"] - 1).astype("Int64")
    annotated["topics_in_conversation"] = (
        annotated.groupby(["split", "conv_id"])["topic_document"]
        .transform(lambda values: values.nunique(dropna=True))
        .astype("Int64")
    )
    return annotated


def _is_topiocqa_dense_index_ready(index_dir: Path) -> bool:
    summary = _safe_topiocqa_dense_summary(index_dir)
    return (
        summary is not None
        and summary.shard_count >= expected_topiocqa_ance_shards()
        and summary.num_passages >= TOPIOCQA_CORPUS_PASSAGES
        and summary.contiguous
    )


def _safe_topiocqa_dense_summary(index_dir: Path):
    candidates = [
        index_dir,
        index_dir / "faiss_flat_index_full_sharded",
    ]
    try:
        from ..dense_ance import summarize_dense_faiss_shards

        for candidate in candidates:
            try:
                return summarize_dense_faiss_shards(candidate)
            except (FileNotFoundError, RuntimeError, ValueError, KeyError):
                pass
        return None
    except (ImportError, ModuleNotFoundError):
        return None


def _default_topiocqa_ance_sas_file() -> Path:
    candidates = (
        Path("/content/drive/MyDrive/secrets/qrecc_file_share_rw_sas.txt"),
        Path("/content/drive/MyDrive/secrets/ance_sharded_dir_sas.txt"),
        project_path("experiments", "secrets", "qrecc_file_share_rw_sas.txt"),
        project_path("experiments", "secrets", "ance_sharded_dir_sas.txt"),
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def _topiocqa_sas_query_from_file(
    *,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
) -> str:
    if url_file is not None:
        resolved_url_file = Path(url_file).expanduser().resolve()
        if not resolved_url_file.exists():
            raise FileNotFoundError(f"TopiOCQA ANCE Azure URL file does not exist: {resolved_url_file}")
        value = resolved_url_file.read_text(encoding="utf-8").strip()
        query = urlsplit(value).query or _extract_sas_query(value)
        if not query:
            raise ValueError(f"TopiOCQA ANCE Azure URL file has no SAS query: {resolved_url_file}")
        return query

    resolved_sas_file = (
        Path(sas_file).expanduser().resolve()
        if sas_file is not None
        else _default_topiocqa_ance_sas_file()
    )
    if not resolved_sas_file.exists():
        raise FileNotFoundError(f"TopiOCQA ANCE Azure SAS file does not exist: {resolved_sas_file}")
    query = _extract_sas_query(resolved_sas_file.read_text(encoding="utf-8").strip())
    if not query:
        raise ValueError(f"TopiOCQA ANCE Azure SAS file is empty: {resolved_sas_file}")
    return query


def _azure_list_directory_names(base_url: str, sas_query: str, *, timeout: float) -> list[str]:
    url = f"{base_url}?restype=directory&comp=list&{sas_query}"
    with urlopen(url, timeout=timeout) as response:
        root = ET.fromstring(response.read())
    names: list[str] = []
    for directory in root.iter():
        if not directory.tag.endswith("Directory"):
            continue
        for child in directory:
            if child.tag.endswith("Name") and child.text:
                names.append(child.text)
                break
    return names


def _azure_read_json_file(file_url: str, sas_query: str, *, timeout: float) -> dict[str, Any]:
    with urlopen(f"{file_url}?{sas_query}", timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_sas_query(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    parsed = urlsplit(stripped)
    if parsed.query:
        return parsed.query
    return stripped.lstrip("?")


def _dir_has_entries(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())
