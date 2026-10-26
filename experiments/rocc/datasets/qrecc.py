"""QReCC resource preparation, collection building, and index adapters."""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence, TYPE_CHECKING
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen, urlretrieve

import pandas as pd

from ..paths import project_path
from ..progress import copy_file, download_file, extract_zip, get_tqdm, progress_iter

if TYPE_CHECKING:
    from ..azure_indexing import CollectionWriteResult


ZENODO_RECORD = "https://zenodo.org/records/5543685/files"
QRECC_TRAIN_FILENAME = "scai-qrecc21-training-turns.json"
QRECC_TEST_FILENAME = "scai-qrecc21-test-turns.json"
QRECC_PASSAGES_ZIP = "passages.zip"
QRECC_PASSAGE_SUBDIRS = ("commoncrawl", "wayback", "wayback-backfill")
QRECC_OFFSET_TYPECODE = "Q"
QRECC_TOTAL_PASSAGES = 54_573_064
QRECC_ANCE_COLAB_TAIL_START = 39_700_000
QRECC_ANCE_SHARD_PASSAGES = 100_000
QRECC_AZURE_FILE_SHARE_URL = ""  # Legacy Azure helpers require an explicit user-supplied endpoint.
QRECC_AZURE_ANCE_SHARDS_PATH = ""  # Legacy Azure helpers require an explicit user-supplied endpoint.
QRECC_AZURE_COLLECTION_PATH = ""  # Legacy Azure helpers require an explicit user-supplied endpoint.
QRECC_AZURE_ANCE_RESULTS_PATH = ""  # Legacy Azure helpers require an explicit user-supplied endpoint.
QRECC_ANCE_INDEX_DIRNAME = "faiss_flat_index_full_sharded"
AZCOPY_DOWNLOAD_URL = "https://aka.ms/downloadazcopy-v10-linux"


@dataclass(frozen=True)
class DownloadSpec:
    url: str
    relative_path: Path
    label: str


@dataclass(frozen=True)
class QReCCResources:
    root: Path
    train_json: Path
    test_json: Path
    passages_zip: Path

    @property
    def data_root(self) -> Path:
        return self.root / "downloads"

    @property
    def passages_dir(self) -> Path:
        return self.data_root / "passages"

    @property
    def collection_dir(self) -> Path:
        return self.root / "collection"

    @property
    def collection_jsonl(self) -> Path:
        return self.collection_dir / "qrecc_collection.jsonl"

    @property
    def collection_offsets_bin(self) -> Path:
        return self.collection_dir / "qrecc_collection.offsets.u64"

    @property
    def processed_train_json(self) -> Path:
        return self.collection_dir / "qrecc-training.json"

    @property
    def processed_test_json(self) -> Path:
        return self.collection_dir / "qrecc-test.json"

    @property
    def collection_manifest_json(self) -> Path:
        return self.collection_dir / "manifest.json"


@dataclass(frozen=True)
class QReCCCollectionBuildResult:
    collection_jsonl: Path
    processed_train_json: Path
    processed_test_json: Path
    manifest_json: Path
    num_passages: int
    num_files: int
    train_turns: int
    test_turns: int
    train_turns_with_gold: int
    test_turns_with_gold: int
    missing_gold_passages: int
    skipped_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "collection_jsonl": str(self.collection_jsonl),
            "processed_train_json": str(self.processed_train_json),
            "processed_test_json": str(self.processed_test_json),
            "manifest_json": str(self.manifest_json),
            "num_passages": self.num_passages,
            "num_files": self.num_files,
            "train_turns": self.train_turns,
            "test_turns": self.test_turns,
            "train_turns_with_gold": self.train_turns_with_gold,
            "test_turns_with_gold": self.test_turns_with_gold,
            "missing_gold_passages": self.missing_gold_passages,
            "skipped_existing": self.skipped_existing,
        }


@dataclass(frozen=True)
class QReCCArtifactStatus:
    root: Path
    train_json: Path
    test_json: Path
    passages_zip: Path
    passages_dir: Path
    collection_jsonl: Path
    processed_train_json: Path
    processed_test_json: Path
    manifest_json: Path
    bm25_collection_jsonl: Path
    bm25_index_dir: Path
    ance_collection_jsonl: Path
    ance_index_dir: Path
    raw_train_test_ready: bool
    passages_zip_exists: bool
    passages_dir_exists: bool
    processed_collection_ready: bool
    bm25_collection_ready: bool
    bm25_index_ready: bool
    ance_collection_ready: bool
    ance_index_ready: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "raw_train_test_ready": self.raw_train_test_ready,
            "passages_zip_exists": self.passages_zip_exists,
            "passages_dir_exists": self.passages_dir_exists,
            "processed_collection_ready": self.processed_collection_ready,
            "bm25_collection_ready": self.bm25_collection_ready,
            "bm25_index_ready": self.bm25_index_ready,
            "ance_collection_ready": self.ance_collection_ready,
            "ance_index_ready": self.ance_index_ready,
            "train_json": str(self.train_json),
            "test_json": str(self.test_json),
            "passages_zip": str(self.passages_zip),
            "passages_dir": str(self.passages_dir),
            "collection_jsonl": str(self.collection_jsonl),
            "processed_train_json": str(self.processed_train_json),
            "processed_test_json": str(self.processed_test_json),
            "manifest_json": str(self.manifest_json),
            "bm25_collection_jsonl": str(self.bm25_collection_jsonl),
            "bm25_index_dir": str(self.bm25_index_dir),
            "ance_collection_jsonl": str(self.ance_collection_jsonl),
            "ance_index_dir": str(self.ance_index_dir),
        }


@dataclass(frozen=True)
class QReCCPassageStore:
    collection_jsonl: Path
    offsets_path: Path
    offsets: array

    def __len__(self) -> int:
        return len(self.offsets)

    def get(self, passage_id: str | int) -> dict[str, Any]:
        index = int(passage_id)
        if index < 0 or index >= len(self.offsets):
            raise IndexError(f"QReCC passage id out of range: {passage_id}")

        with self.collection_jsonl.open("rb") as handle:
            handle.seek(self.offsets[index])
            row = json.loads(handle.readline().decode("utf-8"), strict=False)

        if str(row.get("id")) != str(index):
            raise RuntimeError(f"Offset table mismatch at passage id {passage_id}: row id is {row.get('id')}")
        row["row_index"] = index
        return row

    def to_dict(self) -> dict[str, object]:
        return {
            "collection_jsonl": str(self.collection_jsonl),
            "offsets_path": str(self.offsets_path),
            "num_passages": len(self.offsets),
            "offsets_ram_mib": round((len(self.offsets) * self.offsets.itemsize) / 1024**2, 2),
        }


@dataclass(frozen=True)
class QReCCAnceInputSyncResult:
    source_collection_jsonl: Path
    source_offsets_bin: Path
    source_manifest_json: Path
    drive_dir: Path
    drive_collection_jsonl: Path
    drive_offsets_bin: Path
    drive_manifest_json: Path
    total_bytes: int
    skipped_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "source_collection_jsonl": str(self.source_collection_jsonl),
            "source_offsets_bin": str(self.source_offsets_bin),
            "source_manifest_json": str(self.source_manifest_json),
            "drive_dir": str(self.drive_dir),
            "drive_collection_jsonl": str(self.drive_collection_jsonl),
            "drive_offsets_bin": str(self.drive_offsets_bin),
            "drive_manifest_json": str(self.drive_manifest_json),
            "total_gib": round(self.total_bytes / 1024**3, 2),
            "skipped_existing": self.skipped_existing,
        }


@dataclass(frozen=True)
class QReCCAnceInputRestoreResult:
    drive_dir: Path
    drive_collection_jsonl: Path
    drive_offsets_bin: Path
    drive_manifest_json: Path
    local_dir: Path
    local_collection_jsonl: Path
    local_offsets_bin: Path
    local_manifest_json: Path
    copied: bool
    skipped_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "drive_dir": str(self.drive_dir),
            "drive_collection_jsonl": str(self.drive_collection_jsonl),
            "drive_offsets_bin": str(self.drive_offsets_bin),
            "drive_manifest_json": str(self.drive_manifest_json),
            "local_dir": str(self.local_dir),
            "local_collection_jsonl": str(self.local_collection_jsonl),
            "local_offsets_bin": str(self.local_offsets_bin),
            "local_manifest_json": str(self.local_manifest_json),
            "copied": self.copied,
            "skipped_existing": self.skipped_existing,
        }


@dataclass(frozen=True)
class QReCCAnceAzureInputRestoreResult:
    azure_collection_url: str
    azure_train_url: str
    azure_test_url: str
    drive_dir: Path
    drive_offsets_bin: Path
    drive_manifest_json: Path
    local_dir: Path
    local_collection_jsonl: Path
    local_train_json: Path
    local_test_json: Path
    local_offsets_bin: Path
    local_manifest_json: Path
    downloaded_collection: bool
    downloaded_processed_splits: bool
    copied_support_files: bool
    skipped_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "azure_collection_url": self.azure_collection_url,
            "azure_train_url": self.azure_train_url,
            "azure_test_url": self.azure_test_url,
            "drive_dir": str(self.drive_dir),
            "drive_offsets_bin": str(self.drive_offsets_bin),
            "drive_manifest_json": str(self.drive_manifest_json),
            "local_dir": str(self.local_dir),
            "local_collection_jsonl": str(self.local_collection_jsonl),
            "local_train_json": str(self.local_train_json),
            "local_test_json": str(self.local_test_json),
            "local_offsets_bin": str(self.local_offsets_bin),
            "local_manifest_json": str(self.local_manifest_json),
            "downloaded_collection": self.downloaded_collection,
            "downloaded_processed_splits": self.downloaded_processed_splits,
            "copied_support_files": self.copied_support_files,
            "skipped_existing": self.skipped_existing,
        }


@dataclass(frozen=True)
class QReCCProcessedSplitsAzureSyncResult:
    azure_train_url: str
    azure_test_url: str
    azure_manifest_url: str
    local_dir: Path
    local_train_json: Path
    local_test_json: Path
    local_manifest_json: Path
    downloaded_train: bool
    downloaded_test: bool
    downloaded_manifest: bool
    skipped_existing: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "azure_train_url": self.azure_train_url,
            "azure_test_url": self.azure_test_url,
            "azure_manifest_url": self.azure_manifest_url,
            "local_dir": str(self.local_dir),
            "local_train_json": str(self.local_train_json),
            "local_test_json": str(self.local_test_json),
            "local_manifest_json": str(self.local_manifest_json),
            "downloaded_train": self.downloaded_train,
            "downloaded_test": self.downloaded_test,
            "downloaded_manifest": self.downloaded_manifest,
            "skipped_existing": self.skipped_existing,
        }


@dataclass(frozen=True)
class QReCCAnceAzureUploadTarget:
    sas_file: Path
    url_file: Path
    target_base_url: str
    redacted_target_url: str

    def to_dict(self) -> dict[str, object]:
        return {
            "sas_file": str(self.sas_file),
            "url_file": str(self.url_file),
            "target_base_url": self.target_base_url,
            "redacted_target_url": self.redacted_target_url,
        }


@dataclass(frozen=True)
class QReCCAnceShardStatus:
    index_dir: Path
    complete_shards: int
    first_shard_id: int | None
    last_shard_id: int | None
    next_shard_id: int
    indexed_passages: int

    def to_dict(self) -> dict[str, object]:
        return {
            "index_dir": str(self.index_dir),
            "complete_shards": self.complete_shards,
            "first_shard_id": self.first_shard_id,
            "last_shard_id": self.last_shard_id,
            "next_shard_id": self.next_shard_id,
            "indexed_passages": self.indexed_passages,
        }


@dataclass(frozen=True)
class QReCCAzureAnceIndexStatus:
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
class QReCCAnceAzureIndexSyncResult:
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
class QReCCAnceShardRangeSyncResult:
    target_dir: Path
    local_index_dir: Path
    log_file: Path
    azure_target_url: str
    session_id: str | None
    start_shard: int
    end_shard: int
    expected_shards: int
    copied_shards: int
    skipped_existing_shards: int
    local_complete_shards: int
    local_indexed_passages: int

    def to_dict(self) -> dict[str, object]:
        return {
            "target_dir": str(self.target_dir),
            "local_index_dir": str(self.local_index_dir),
            "log_file": str(self.log_file),
            "azure_target_url": self.azure_target_url,
            "session_id": self.session_id,
            "start_shard": self.start_shard,
            "end_shard": self.end_shard,
            "expected_shards": self.expected_shards,
            "copied_shards": self.copied_shards,
            "skipped_existing_shards": self.skipped_existing_shards,
            "local_complete_shards": self.local_complete_shards,
            "local_indexed_passages": self.local_indexed_passages,
        }


@dataclass(frozen=True)
class QReCCAzureFileTransferResult:
    local_file: Path
    azure_url: str
    transferred: bool
    skipped_existing: bool
    exists_remote: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "local_file": str(self.local_file),
            "azure_url": self.azure_url,
            "transferred": self.transferred,
            "skipped_existing": self.skipped_existing,
            "exists_remote": self.exists_remote,
        }


@dataclass(frozen=True)
class QReCCPartialDenseAzureEvalResult:
    run_id: str
    session_ids: tuple[str, ...]
    local_run_dir: Path
    local_dump_files: tuple[Path, ...]
    merged_file: Path
    downloads: tuple[QReCCAzureFileTransferResult, ...]
    merge_result: Any
    evaluation: Any
    eval_rewrites: pd.DataFrame
    final_top_k: int
    total_test_gold_queries: int
    evaluated_queries: int

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "session_ids": self.session_ids,
            "local_run_dir": str(self.local_run_dir),
            "local_dump_files": [str(path) for path in self.local_dump_files],
            "merged_file": str(self.merged_file),
            "final_top_k": self.final_top_k,
            "downloads": len(self.downloads),
            "downloads_remote_exists": all(item.exists_remote for item in self.downloads),
            "merge_query_count": self.merge_result.query_count,
            "merge_hit_count": self.merge_result.hit_count,
            "total_test_gold_queries": self.total_test_gold_queries,
            "evaluated_queries": self.evaluated_queries,
        }


def default_qrecc_root(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        return Path(data_dir).expanduser().resolve()
    return project_path("experiments", "data", "qrecc")


def resolve_qrecc_resources(data_dir: Path | str | None = None) -> QReCCResources:
    return _resources_for_root(default_qrecc_root(data_dir))


def inspect_qrecc_artifacts(
    resources: QReCCResources | None = None,
    data_dir: Path | str | None = None,
) -> QReCCArtifactStatus:
    resources = resources or _resources_for_root(default_qrecc_root(data_dir))
    bm25_collection_jsonl = project_path(
        "experiments",
        "data",
        "pyserini_bm25_lucene_qrecc",
        "collection",
        "docs.jsonl",
    )
    bm25_index_dir = project_path(
        "experiments",
        "data",
        "pyserini_bm25_lucene_qrecc",
        "lucene_index",
    )
    ance_collection_jsonl = project_path(
        "experiments",
        "data",
        "pyserini_ance_faiss_qrecc",
        "collection",
        "qrecc_collection_ance.jsonl",
    )
    ance_index_dir = project_path(
        "experiments",
        "data",
        "pyserini_ance_faiss_qrecc",
        QRECC_ANCE_INDEX_DIRNAME,
    )
    ance_index_status = inspect_qrecc_ance_shards(ance_index_dir)
    return QReCCArtifactStatus(
        root=resources.root,
        train_json=resources.train_json,
        test_json=resources.test_json,
        passages_zip=resources.passages_zip,
        passages_dir=resources.passages_dir,
        collection_jsonl=resources.collection_jsonl,
        processed_train_json=resources.processed_train_json,
        processed_test_json=resources.processed_test_json,
        manifest_json=resources.collection_manifest_json,
        bm25_collection_jsonl=bm25_collection_jsonl,
        bm25_index_dir=bm25_index_dir,
        ance_collection_jsonl=ance_collection_jsonl,
        ance_index_dir=ance_index_dir,
        raw_train_test_ready=resources.train_json.exists() and resources.test_json.exists(),
        passages_zip_exists=resources.passages_zip.exists(),
        passages_dir_exists=resources.passages_dir.exists(),
        processed_collection_ready=(
            resources.collection_jsonl.exists()
            and resources.processed_train_json.exists()
            and resources.processed_test_json.exists()
            and resources.collection_manifest_json.exists()
        ),
        bm25_collection_ready=bm25_collection_jsonl.exists() or bm25_collection_jsonl.is_symlink(),
        bm25_index_ready=_dir_has_entries(bm25_index_dir),
        ance_collection_ready=ance_collection_jsonl.exists(),
        ance_index_ready=(
            ance_index_status.complete_shards >= _expected_qrecc_ance_shards()
            and ance_index_status.indexed_passages >= QRECC_TOTAL_PASSAGES
        ),
    )


def ensure_qrecc_resources(
    data_dir: Path | str | None = None,
    *,
    force_download: bool = False,
    progress: bool = True,
    download_passages: bool = False,
    extract_passages: bool = False,
    force_extract: bool = False,
    timeout: float = 60.0,
) -> QReCCResources:
    """Prepare SCAI QReCC train/test JSONs and optionally the passage corpus."""

    resources = _resources_for_root(default_qrecc_root(data_dir))
    downloads = list(_qrecc_downloads())
    resource_bar = progress_iter(
        downloads,
        total=len(downloads),
        desc="prepare QReCC",
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

    if download_passages:
        download_file(
            f"{ZENODO_RECORD}/{QRECC_PASSAGES_ZIP}",
            resources.passages_zip,
            desc="QReCC passages",
            force=force_download,
            timeout=timeout,
            progress=progress,
        )
        if extract_passages:
            extract_zip(
                resources.passages_zip,
                resources.passages_dir,
                desc="unzip QReCC passages",
                force=force_extract,
                progress=progress,
            )

    missing = [
        str(path)
        for path in (resources.train_json, resources.test_json)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"QReCC train/test resources missing after preparation: {missing}")
    return resources


def build_qrecc_collection_and_splits(
    resources: QReCCResources | None = None,
    data_dir: Path | str | None = None,
    *,
    force: bool = False,
    progress: bool = True,
) -> QReCCCollectionBuildResult:
    """Build numeric-id QReCC collection and processed train/test files.

    The output train/test files contain ``positive_ctx_passage_ids`` in the same
    numeric docid space as the consolidated Pyserini collection.
    """

    resources = resources or ensure_qrecc_resources(data_dir)
    if (
        not force
        and resources.collection_jsonl.exists()
        and resources.processed_train_json.exists()
        and resources.processed_test_json.exists()
        and resources.collection_manifest_json.exists()
        and _qrecc_manifest_matches_sources(resources.collection_manifest_json, resources)
    ):
        manifest = json.loads(resources.collection_manifest_json.read_text(encoding="utf-8"))
        return QReCCCollectionBuildResult(
            collection_jsonl=resources.collection_jsonl,
            processed_train_json=resources.processed_train_json,
            processed_test_json=resources.processed_test_json,
            manifest_json=resources.collection_manifest_json,
            num_passages=int(manifest["num_passages"]),
            num_files=int(manifest["num_files"]),
            train_turns=int(manifest["train_turns"]),
            test_turns=int(manifest["test_turns"]),
            train_turns_with_gold=int(manifest["train_turns_with_gold"]),
            test_turns_with_gold=int(manifest["test_turns_with_gold"]),
            missing_gold_passages=int(manifest["missing_gold_passages"]),
            skipped_existing=True,
        )

    train_rows = _read_qrecc_rows(resources.train_json)
    test_rows = _read_qrecc_rows(resources.test_json)
    reuse_existing_collection = resources.collection_jsonl.exists() and not force
    if not reuse_existing_collection and not resources.passages_dir.exists():
        raise FileNotFoundError(f"QReCC passages are not extracted: {resources.passages_dir}")

    gold_rawpids = set(_iter_gold_rawpids(train_rows)) | set(_iter_gold_rawpids(test_rows))
    rawpid2pid: dict[str, str] = {}

    resources.collection_dir.mkdir(parents=True, exist_ok=True)
    tmp_collection = resources.collection_jsonl.with_suffix(resources.collection_jsonl.suffix + ".tmp")
    tmp_train = resources.processed_train_json.with_suffix(resources.processed_train_json.suffix + ".tmp")
    tmp_test = resources.processed_test_json.with_suffix(resources.processed_test_json.suffix + ".tmp")
    for path in (tmp_collection, tmp_train, tmp_test):
        if path.exists():
            path.unlink()

    started = time.time()
    files = _qrecc_passage_files(resources.passages_dir) if resources.passages_dir.exists() else []
    if reuse_existing_collection:
        num_passages = _map_gold_rawpids_from_collection(
            resources.collection_jsonl,
            gold_rawpids,
            rawpid2pid,
            progress=progress,
        )
    else:
        file_iter = progress_iter(
            files,
            total=len(files),
            desc="write QReCC collection",
            unit="file",
            enabled=progress,
        )
        num_passages = 0
        with tmp_collection.open("w", encoding="utf-8") as output:
            for file_path in file_iter:
                with file_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        row = json.loads(line, strict=False)
                        raw_pid = str(row["id"])
                        numeric_pid = str(num_passages)
                        output.write(
                            json.dumps(
                                {
                                    "id": numeric_pid,
                                    "contents": str(row.get("contents", "")),
                                    "raw_passage_id": raw_pid,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        if raw_pid in gold_rawpids:
                            rawpid2pid[raw_pid] = numeric_pid
                        num_passages += 1
        tmp_collection.replace(resources.collection_jsonl)

    processed_train, train_missing = _add_numeric_positive_ctxs(train_rows, rawpid2pid)
    processed_test, test_missing = _add_numeric_positive_ctxs(test_rows, rawpid2pid)
    tmp_train.write_text(json.dumps(processed_train, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_test.write_text(json.dumps(processed_test, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tmp_train.replace(resources.processed_train_json)
    tmp_test.replace(resources.processed_test_json)

    manifest = {
        "train_source_json": str(resources.train_json),
        "test_source_json": str(resources.test_json),
        "train_source_name": resources.train_json.name,
        "test_source_name": resources.test_json.name,
        "num_passages": num_passages,
        "num_files": len(files),
        "train_turns": len(processed_train),
        "test_turns": len(processed_test),
        "train_turns_with_gold": sum(bool(row["positive_ctx_passage_ids"]) for row in processed_train),
        "test_turns_with_gold": sum(bool(row["positive_ctx_passage_ids"]) for row in processed_test),
        "missing_gold_passages": train_missing + test_missing,
        "elapsed_seconds": round(time.time() - started, 3),
        "collection_format": {"id": "numeric string", "contents": "passage text"},
        "dataset_gold_field": "positive_ctx_passage_ids",
    }
    resources.collection_manifest_json.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return QReCCCollectionBuildResult(
        collection_jsonl=resources.collection_jsonl,
        processed_train_json=resources.processed_train_json,
        processed_test_json=resources.processed_test_json,
        manifest_json=resources.collection_manifest_json,
        num_passages=num_passages,
        num_files=len(files),
        train_turns=len(processed_train),
        test_turns=len(processed_test),
        train_turns_with_gold=int(manifest["train_turns_with_gold"]),
        test_turns_with_gold=int(manifest["test_turns_with_gold"]),
        missing_gold_passages=train_missing + test_missing,
        skipped_existing=False,
    )


def load_qrecc_collection_build_result(
    resources: QReCCResources | None = None,
    data_dir: Path | str | None = None,
    require_collection: bool = True,
) -> QReCCCollectionBuildResult:
    """Load an existing processed QReCC collection manifest as a build result."""

    resources = resources or resolve_qrecc_resources(data_dir)
    required = [
        resources.processed_train_json,
        resources.processed_test_json,
        resources.collection_manifest_json,
    ]
    if require_collection:
        required.insert(0, resources.collection_jsonl)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"QReCC processed collection files missing: {missing}")
    manifest = json.loads(resources.collection_manifest_json.read_text(encoding="utf-8"))
    return QReCCCollectionBuildResult(
        collection_jsonl=resources.collection_jsonl,
        processed_train_json=resources.processed_train_json,
        processed_test_json=resources.processed_test_json,
        manifest_json=resources.collection_manifest_json,
        num_passages=int(manifest["num_passages"]),
        num_files=int(manifest.get("num_files", 0)),
        train_turns=int(manifest["train_turns"]),
        test_turns=int(manifest["test_turns"]),
        train_turns_with_gold=int(manifest["train_turns_with_gold"]),
        test_turns_with_gold=int(manifest["test_turns_with_gold"]),
        missing_gold_passages=int(manifest.get("missing_gold_passages", 0)),
        skipped_existing=True,
    )


def write_qrecc_bm25_collection(
    collection: QReCCCollectionBuildResult | Path | str | None = None,
    *,
    output_file: Path | str | None = None,
    data_dir: Path | str | None = None,
    force: bool = False,
    link: bool = True,
    progress: bool = True,
) -> "CollectionWriteResult":
    """Expose the processed QReCC corpus as a Pyserini BM25 JsonCollection.

    The processed corpus already has ``id`` and ``contents`` fields. By default
    this creates a lightweight ``docs.jsonl`` symlink instead of copying the
    large corpus file.
    """

    from ..azure_indexing import CollectionWriteResult

    source = _resolve_qrecc_collection_jsonl(collection, data_dir=data_dir)
    output_path = Path(output_file).expanduser().resolve() if output_file is not None else (
        project_path("experiments", "data", "pyserini_bm25_lucene_qrecc", "collection", "docs.jsonl")
    )
    if (output_path.exists() or output_path.is_symlink()) and not force:
        return CollectionWriteResult(
            output_file=output_path,
            written_passages=_qrecc_collection_passage_count(source),
            skipped_existing=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        output_path.unlink()

    if link:
        output_path.symlink_to(source)
        return CollectionWriteResult(
            output_file=output_path,
            written_passages=_qrecc_collection_passage_count(source),
            skipped_existing=False,
        )

    written = _copy_qrecc_collection_jsonl(
        source,
        output_path,
        desc="write QReCC BM25 JSONL",
        progress=progress,
    )
    return CollectionWriteResult(output_file=output_path, written_passages=written, skipped_existing=False)


def write_qrecc_ance_collection(
    collection: QReCCCollectionBuildResult | Path | str | None = None,
    *,
    output_file: Path | str | None = None,
    data_dir: Path | str | None = None,
    force: bool = False,
    progress: bool = True,
) -> "CollectionWriteResult":
    """Write the processed QReCC corpus in ANCE encoding format.

    Pyserini's ANCE encoder path expects rows with ``id`` and ``text``; QReCC's
    processed corpus stores passage text in ``contents``.
    """

    from ..azure_indexing import CollectionWriteResult

    source = _resolve_qrecc_collection_jsonl(collection, data_dir=data_dir)
    output_path = Path(output_file).expanduser().resolve() if output_file is not None else (
        project_path("experiments", "data", "pyserini_ance_faiss_qrecc", "collection", "qrecc_collection_ance.jsonl")
    )
    if output_path.exists() and not force:
        return CollectionWriteResult(
            output_file=output_path,
            written_passages=_count_jsonl_rows(output_path),
            skipped_existing=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    total = _qrecc_collection_passage_count(source)
    written = 0
    with source.open("r", encoding="utf-8") as source_handle:
        rows = progress_iter(
            source_handle,
            total=total,
            desc="write QReCC ANCE JSONL",
            unit="passage",
            enabled=progress,
        )
        with tmp_output.open("w", encoding="utf-8") as output:
            for line in rows:
                row = json.loads(line, strict=False)
                output.write(
                    json.dumps(
                        {
                            "id": str(row["id"]),
                            "text": str(row.get("contents") or row.get("text") or ""),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
        close = getattr(rows, "close", None)
        if close is not None:
            close()

    tmp_output.replace(output_path)
    return CollectionWriteResult(output_file=output_path, written_passages=written, skipped_existing=False)


def load_qrecc_frame(
    resources: QReCCResources | None = None,
    *,
    prefer_processed: bool = True,
    gold_only: bool = True,
) -> pd.DataFrame:
    """Load QReCC turns in the normalized ROCC schema.

    By default this returns the retrieval-ready subset only, i.e. turns with at
    least one mapped gold passage id.
    """

    resources = resources or ensure_qrecc_resources()
    use_processed = (
        prefer_processed
        and resources.processed_train_json.exists()
        and resources.processed_test_json.exists()
        and _qrecc_manifest_matches_sources(resources.collection_manifest_json, resources)
    )
    train_json = resources.processed_train_json if use_processed else resources.train_json
    test_json = resources.processed_test_json if use_processed else resources.test_json
    train_df = _load_qrecc_json(train_json, split="train")
    test_df = _load_qrecc_json(test_json, split="test")
    frame = pd.concat([train_df, test_df], ignore_index=True)
    if gold_only:
        frame = frame[frame["positive_ctx_passage_ids"].apply(bool)].reset_index(drop=True)
    return frame


def sample_qrecc_turns_by_source(
    frame: pd.DataFrame,
    *,
    per_source: int = 3,
    split: str | None = None,
    seed: int | None = None,
    source_column: str = "conversation_source",
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    if per_source < 1:
        raise ValueError("per_source must be >= 1.")
    if source_column not in frame.columns:
        raise ValueError(f"QReCC frame is missing source column: {source_column}")

    subset = frame
    if split is not None:
        if "split" not in subset.columns:
            raise ValueError("QReCC frame is missing split column.")
        subset = subset[subset["split"].eq(split)]
    if subset.empty:
        return subset.reset_index(drop=True)

    rng = random.Random(seed)
    sampled_groups = []
    for _, group in subset.groupby(source_column, dropna=False, sort=True):
        indices = group.index.tolist()
        if len(indices) > per_source:
            indices = rng.sample(indices, per_source)
        sampled_groups.append(subset.loc[indices])

    result = pd.concat(sampled_groups, ignore_index=False)
    sort_columns = [
        column
        for column in (source_column, "split", "conv_id", "turn_id")
        if column in result.columns
    ]
    if sort_columns:
        result = result.sort_values(sort_columns)

    selected_columns = list(columns) if columns is not None else [
        "conversation_source",
        "split",
        "conv_id",
        "turn_id",
        "question",
        "rewrite",
        "answers",
        "positive_ctx_passage_ids",
    ]
    existing_columns = [column for column in selected_columns if column in result.columns]
    return result[existing_columns].reset_index(drop=True)


def lookup_qrecc_passages_for_turns(
    turns: pd.DataFrame,
    passage_store: QReCCPassageStore,
    *,
    include_contents: bool = True,
) -> pd.DataFrame:
    rows = []
    for _, turn in turns.iterrows():
        passage_ids = turn.get("positive_ctx_passage_ids") or []
        for passage_rank, passage_id in enumerate(passage_ids, start=1):
            passage = passage_store.get(passage_id)
            row = {
                "conversation_source": turn.get("conversation_source"),
                "split": turn.get("split"),
                "conv_id": turn.get("conv_id"),
                "turn_id": turn.get("turn_id"),
                "question": turn.get("question"),
                "rewrite": turn.get("rewrite"),
                "positive_ctx_rank": passage_rank,
                "passage_id": str(passage_id),
                "row_index": passage["row_index"],
                "id": passage["id"],
                "raw_passage_id": passage.get("raw_passage_id"),
            }
            if include_contents:
                row["contents"] = passage.get("contents")
            rows.append(row)
    return pd.DataFrame(rows)


def load_qrecc_passage_store(
    collection_jsonl: Path | str,
    *,
    offsets_path: Path | str | None = None,
    force_rebuild: bool = False,
    progress: bool = True,
) -> QReCCPassageStore:
    collection_path = Path(collection_jsonl).expanduser().resolve()
    resolved_offsets_path = (
        Path(offsets_path).expanduser().resolve()
        if offsets_path is not None
        else collection_path.with_suffix(".offsets.u64")
    )
    build_qrecc_passage_offsets(
        collection_path,
        offsets_path=resolved_offsets_path,
        force=force_rebuild,
        progress=progress,
    )
    offsets = _read_qrecc_offsets(resolved_offsets_path)
    return QReCCPassageStore(
        collection_jsonl=collection_path,
        offsets_path=resolved_offsets_path,
        offsets=offsets,
    )


def build_qrecc_passage_offsets(
    collection_jsonl: Path | str,
    *,
    offsets_path: Path | str | None = None,
    force: bool = False,
    progress: bool = True,
) -> Path:
    collection_path = Path(collection_jsonl).expanduser().resolve()
    if not collection_path.exists():
        raise FileNotFoundError(f"QReCC collection does not exist: {collection_path}")

    resolved_offsets_path = (
        Path(offsets_path).expanduser().resolve()
        if offsets_path is not None
        else collection_path.with_suffix(".offsets.u64")
    )
    if _qrecc_offsets_are_fresh(collection_path, resolved_offsets_path) and not force:
        return resolved_offsets_path

    resolved_offsets_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_offsets_path = resolved_offsets_path.with_suffix(resolved_offsets_path.suffix + ".tmp")
    if tmp_offsets_path.exists():
        tmp_offsets_path.unlink()

    total_bytes = collection_path.stat().st_size
    bar = get_tqdm()(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc="build QReCC passage offsets",
        disable=not progress,
    )
    try:
        with collection_path.open("rb") as collection, tmp_offsets_path.open("wb") as output:
            batch = array(QRECC_OFFSET_TYPECODE)
            while True:
                offset = collection.tell()
                line = collection.readline()
                if not line:
                    break
                batch.append(offset)
                if len(batch) >= 1_000_000:
                    batch.tofile(output)
                    batch = array(QRECC_OFFSET_TYPECODE)
                bar.update(len(line))
            if batch:
                batch.tofile(output)
    finally:
        bar.close()

    tmp_offsets_path.replace(resolved_offsets_path)
    return resolved_offsets_path


def sync_qrecc_ance_inputs_to_drive(
    collection: QReCCCollectionBuildResult | Path | str | None = None,
    *,
    drive_dir: Path | str = "/content/drive/MyDrive/corpora/qrecc",
    data_dir: Path | str | None = None,
    mount_drive: bool = True,
    force: bool = False,
    progress: bool = True,
) -> QReCCAnceInputSyncResult:
    """Copy the minimal QReCC ANCE input set to Google Drive.

    QReCC already stores ANCE-ready rows as ``id`` + ``contents``. The offsets
    file is copied with it so range jobs can seek directly into the 82GB JSONL.
    """

    source_collection = _resolve_qrecc_collection_jsonl(collection, data_dir=data_dir)
    source_offsets = source_collection.with_name("qrecc_collection.offsets.u64")
    source_manifest = source_collection.parent / "manifest.json"
    if not source_manifest.exists():
        raise FileNotFoundError(f"QReCC collection manifest is missing: {source_manifest}")
    build_qrecc_passage_offsets(
        source_collection,
        offsets_path=source_offsets,
        force=False,
        progress=progress,
    )

    target_dir = Path(drive_dir).expanduser()
    if mount_drive and str(target_dir).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        from ..colab import mount_google_drive

        mount_google_drive()
    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    copies = [
        (source_collection, target_dir / source_collection.name, "copy QReCC collection to Drive"),
        (source_offsets, target_dir / source_offsets.name, "copy QReCC offsets to Drive"),
        (source_manifest, target_dir / source_manifest.name, "copy QReCC manifest to Drive"),
    ]
    skipped_existing = True
    total_bytes = 0
    for source, target, desc in copies:
        total_bytes += source.stat().st_size
        if _same_existing_file(source, target) and not force:
            continue
        skipped_existing = False
        copy_file(source, target, desc=desc, force=force, progress=progress)

    return QReCCAnceInputSyncResult(
        source_collection_jsonl=source_collection,
        source_offsets_bin=source_offsets,
        source_manifest_json=source_manifest,
        drive_dir=target_dir,
        drive_collection_jsonl=target_dir / source_collection.name,
        drive_offsets_bin=target_dir / source_offsets.name,
        drive_manifest_json=target_dir / source_manifest.name,
        total_bytes=total_bytes,
        skipped_existing=skipped_existing,
    )


def sync_qrecc_ance_inputs_from_drive(
    collection: QReCCCollectionBuildResult | Path | str | None = None,
    *,
    drive_dir: Path | str = "/content/drive/MyDrive/corpora/qrecc",
    data_dir: Path | str | None = None,
    mount_drive: bool = True,
    force: bool = False,
    progress: bool = True,
) -> QReCCAnceInputRestoreResult:
    """Copy the minimal QReCC ANCE input set from Google Drive to local disk."""

    if isinstance(collection, QReCCCollectionBuildResult) or hasattr(collection, "collection_jsonl"):
        local_collection = Path(getattr(collection, "collection_jsonl")).expanduser().resolve()
    elif collection is not None:
        local_collection = Path(collection).expanduser().resolve()
    else:
        local_collection = _resources_for_root(default_qrecc_root(data_dir)).collection_jsonl
    local_offsets = local_collection.with_name("qrecc_collection.offsets.u64")
    local_manifest = local_collection.parent / "manifest.json"

    source_dir = Path(drive_dir).expanduser()
    if mount_drive and str(source_dir).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        from ..colab import mount_google_drive

        mount_google_drive()
    source_dir = source_dir.resolve()

    drive_collection = source_dir / local_collection.name
    drive_offsets = source_dir / local_offsets.name
    drive_manifest = source_dir / local_manifest.name
    missing = [str(path) for path in (drive_collection, drive_offsets, drive_manifest) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"QReCC ANCE input files missing on Drive: {missing}")

    local_collection.parent.mkdir(parents=True, exist_ok=True)
    skipped_existing = True
    for source, target, desc in (
        (drive_collection, local_collection, "copy QReCC collection from Drive"),
        (drive_offsets, local_offsets, "copy QReCC offsets from Drive"),
        (drive_manifest, local_manifest, "copy QReCC manifest from Drive"),
    ):
        if _same_existing_file(source, target) and not force:
            continue
        skipped_existing = False
        copy_file(source, target, desc=desc, force=force, progress=progress)

    return QReCCAnceInputRestoreResult(
        drive_dir=source_dir,
        drive_collection_jsonl=drive_collection,
        drive_offsets_bin=drive_offsets,
        drive_manifest_json=drive_manifest,
        local_dir=local_collection.parent,
        local_collection_jsonl=local_collection,
        local_offsets_bin=local_offsets,
        local_manifest_json=local_manifest,
        copied=not skipped_existing,
        skipped_existing=skipped_existing,
    )


def sync_qrecc_ance_inputs_from_azure(
    *,
    data_dir: Path | str | None = None,
    drive_dir: Path | str = "/content/drive/MyDrive/corpora/qrecc",
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_collection_path: str = QRECC_AZURE_COLLECTION_PATH,
    mount_drive: bool = True,
    force: bool = False,
    download_collection: bool = True,
    download_processed_splits: bool = True,
    progress: bool = True,
    azcopy_path: Path | str | None = None,
    install_azcopy: bool = True,
) -> QReCCAnceAzureInputRestoreResult:
    """Restore QReCC processed files from Azure and optional Drive support files.

    The 82GB collection is only needed for local passage lookup or shard/index
    creation. Set ``download_collection=False`` when the dense index already
    exists and only the processed train/test splits are needed.
    """

    resources = _resources_for_root(default_qrecc_root(data_dir))
    local_collection = resources.collection_jsonl
    local_train = resources.processed_train_json
    local_test = resources.processed_test_json
    local_offsets = resources.collection_offsets_bin
    local_manifest = resources.collection_manifest_json

    source_dir = Path(drive_dir).expanduser()
    if mount_drive and str(source_dir).startswith("/content/drive") and not Path("/content/drive/MyDrive").exists():
        from ..colab import mount_google_drive

        mount_google_drive()
    source_dir = source_dir.resolve()
    drive_offsets = source_dir / local_offsets.name
    drive_manifest = source_dir / local_manifest.name
    missing_support = [
        str(drive_path)
        for drive_path, local_path in ((drive_offsets, local_offsets), (drive_manifest, local_manifest))
        if not drive_path.exists() and not local_path.exists()
    ]
    if missing_support:
        raise FileNotFoundError(f"QReCC ANCE support files missing on Drive: {missing_support}")

    sas_query = _qrecc_sas_query_from_file(sas_file=sas_file, url_file=url_file)
    azure_collection_root = azure_collection_path.strip("/").rsplit("/", 1)[0]

    def azure_file_url(filename: str) -> str:
        return f"{azure_file_share_url.rstrip('/')}/{azure_collection_root}/{filename}?{sas_query}"

    def redact(url: str) -> str:
        return url.split("?", 1)[0] + "?***"

    collection_url = azure_file_url(local_collection.name)
    train_url = azure_file_url(local_train.name)
    test_url = azure_file_url(local_test.name)
    redacted_collection_url = redact(collection_url)
    redacted_train_url = redact(train_url)
    redacted_test_url = redact(test_url)

    local_collection.parent.mkdir(parents=True, exist_ok=True)
    copied_support_files = False
    for source, target, desc in (
        (drive_offsets, local_offsets, "copy QReCC offsets from Drive"),
        (drive_manifest, local_manifest, "copy QReCC manifest from Drive"),
    ):
        if not source.exists() and target.exists():
            continue
        if _same_existing_file(source, target) and not force:
            continue
        copy_file(source, target, desc=desc, force=force, progress=progress)
        copied_support_files = True

    azcopy: Path | None = None

    def download_from_azure(url: str, target: Path, label: str) -> bool:
        nonlocal azcopy
        if target.exists() and not force:
            return False
        if azcopy is None:
            azcopy = _ensure_azcopy(Path(azcopy_path).expanduser() if azcopy_path else None, install=install_azcopy)
        command = [
            str(azcopy),
            "copy",
            url,
            str(target),
            "--from-to=FileSMBLocal",
            "--check-length=true",
            "--output-type=text",
            "--log-level=INFO",
        ]
        if progress:
            print(f"download {label} from Azure:", redact(url), flush=True)
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True

    downloaded_collection = False
    if download_collection:
        downloaded_collection = download_from_azure(
            collection_url,
            local_collection,
            "QReCC collection",
        )
    downloaded_train = False
    downloaded_test = False
    if download_processed_splits:
        downloaded_train = download_from_azure(
            train_url,
            local_train,
            "QReCC processed train split",
        )
        downloaded_test = download_from_azure(
            test_url,
            local_test,
            "QReCC processed test split",
        )
    downloaded_processed_splits = downloaded_train or downloaded_test

    skipped_existing = not downloaded_collection and not copied_support_files and not downloaded_processed_splits
    return QReCCAnceAzureInputRestoreResult(
        azure_collection_url=redacted_collection_url,
        azure_train_url=redacted_train_url,
        azure_test_url=redacted_test_url,
        drive_dir=source_dir,
        drive_offsets_bin=drive_offsets,
        drive_manifest_json=drive_manifest,
        local_dir=local_collection.parent,
        local_collection_jsonl=local_collection,
        local_train_json=local_train,
        local_test_json=local_test,
        local_offsets_bin=local_offsets,
        local_manifest_json=local_manifest,
        downloaded_collection=downloaded_collection,
        downloaded_processed_splits=downloaded_processed_splits,
        copied_support_files=copied_support_files,
        skipped_existing=skipped_existing,
    )


def sync_qrecc_processed_splits_from_azure(
    *,
    data_dir: Path | str | None = None,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_collection_path: str = QRECC_AZURE_COLLECTION_PATH,
    force: bool = False,
    progress: bool = True,
    azcopy_path: Path | str | None = None,
    install_azcopy: bool = True,
) -> QReCCProcessedSplitsAzureSyncResult:
    """Restore only QReCC processed splits and manifest from Azure.

    This is enough for IterCQR dataloading and retrieval evaluation. It does not
    download the 82GB processed collection.
    """

    resources = _resources_for_root(default_qrecc_root(data_dir))
    local_train = resources.processed_train_json
    local_test = resources.processed_test_json
    local_manifest = resources.collection_manifest_json

    sas_query = _qrecc_sas_query_from_file(sas_file=sas_file, url_file=url_file)
    azure_collection_root = azure_collection_path.strip("/").rsplit("/", 1)[0]

    def azure_file_url(filename: str) -> str:
        return f"{azure_file_share_url.rstrip('/')}/{azure_collection_root}/{filename}?{sas_query}"

    def redact(url: str) -> str:
        return url.split("?", 1)[0] + "?***"

    train_url = azure_file_url(local_train.name)
    test_url = azure_file_url(local_test.name)
    manifest_url = azure_file_url(local_manifest.name)
    local_train.parent.mkdir(parents=True, exist_ok=True)
    azcopy: Path | None = None

    def download_from_azure(url: str, target: Path, label: str) -> bool:
        nonlocal azcopy
        if target.exists() and not force:
            return False
        if azcopy is None:
            azcopy = _ensure_azcopy(Path(azcopy_path).expanduser() if azcopy_path else None, install=install_azcopy)
        command = [
            str(azcopy),
            "copy",
            url,
            str(target),
            "--from-to=FileSMBLocal",
            "--check-length=true",
            "--output-type=text",
            "--log-level=INFO",
        ]
        if progress:
            print(f"download {label} from Azure:", redact(url), flush=True)
            subprocess.run(command, check=True)
        else:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True

    downloaded_train = download_from_azure(train_url, local_train, "QReCC processed train split")
    downloaded_test = download_from_azure(test_url, local_test, "QReCC processed test split")
    downloaded_manifest = download_from_azure(manifest_url, local_manifest, "QReCC manifest")
    return QReCCProcessedSplitsAzureSyncResult(
        azure_train_url=redact(train_url),
        azure_test_url=redact(test_url),
        azure_manifest_url=redact(manifest_url),
        local_dir=local_train.parent,
        local_train_json=local_train,
        local_test_json=local_test,
        local_manifest_json=local_manifest,
        downloaded_train=downloaded_train,
        downloaded_test=downloaded_test,
        downloaded_manifest=downloaded_manifest,
        skipped_existing=not downloaded_train and not downloaded_test and not downloaded_manifest,
    )


def prepare_qrecc_ance_azure_upload_target(
    *,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_shards_path: str = QRECC_AZURE_ANCE_SHARDS_PATH,
) -> QReCCAnceAzureUploadTarget:
    """Write the full Azure upload URL used by the shard builder.

    ``sas_file`` may contain either a raw SAS query string or a full SAS URL.
    The query part is always combined with the QReCC shard target path so an
    old URL path in the secret cannot redirect the upload.
    """

    resolved_sas_file = (
        Path(sas_file).expanduser().resolve()
        if sas_file is not None
        else _default_qrecc_ance_sas_file()
    )
    if not resolved_sas_file.exists():
        raise FileNotFoundError(f"QReCC ANCE Azure SAS file does not exist: {resolved_sas_file}")

    sas_value = resolved_sas_file.read_text(encoding="utf-8").strip()
    sas_query = _extract_sas_query(sas_value)
    if not sas_query:
        raise ValueError(f"QReCC ANCE Azure SAS file is empty: {resolved_sas_file}")

    resolved_url_file = (
        Path(url_file).expanduser().resolve()
        if url_file is not None
        else _default_qrecc_ance_upload_url_file()
    )
    target_base_url = f"{azure_file_share_url.rstrip('/')}/{azure_shards_path.strip('/')}"
    target_url = f"{target_base_url}?{sas_query}"
    resolved_url_file.parent.mkdir(parents=True, exist_ok=True)
    resolved_url_file.write_text(target_url + "\n", encoding="utf-8")

    return QReCCAnceAzureUploadTarget(
        sas_file=resolved_sas_file,
        url_file=resolved_url_file,
        target_base_url=target_base_url,
        redacted_target_url=f"{target_base_url}?***",
    )


def inspect_qrecc_azure_ance_index(
    *,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_shards_path: str = QRECC_AZURE_ANCE_SHARDS_PATH,
    expected_shards: int | None = None,
    validate_manifests: bool = False,
    timeout: float = 30.0,
) -> QReCCAzureAnceIndexStatus:
    """Inspect the remote QReCC ANCE shard directory in Azure File Share.

    The default check lists shard directories and samples the first/last
    manifests. Set ``validate_manifests=True`` for a full manifest sweep.
    """

    target_base_url = f"{azure_file_share_url.rstrip('/')}/{azure_shards_path.strip('/')}"
    expected = expected_shards or _expected_qrecc_ance_shards()
    try:
        sas_query = _qrecc_sas_query_from_file(sas_file=sas_file, url_file=url_file)
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
            and (indexed_passages is None or indexed_passages >= QRECC_TOTAL_PASSAGES)
        )
        return QReCCAzureAnceIndexStatus(
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
        return QReCCAzureAnceIndexStatus(
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


def sync_qrecc_ance_index_from_azure(
    *,
    local_index_dir: Path | str | None = None,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_shards_path: str = QRECC_AZURE_ANCE_SHARDS_PATH,
    validate_azure: bool = True,
    validate_azure_manifests: bool = False,
    force: bool = False,
    log_file: Path | str = "/content/azcopy_qrecc_ance_index_download.log",
    install_azcopy: bool = True,
    progress: bool = True,
    timeout: float = 30.0,
) -> QReCCAnceAzureIndexSyncResult:
    """Download the finished QReCC ANCE shard index from Azure with AzCopy."""

    target_dir = (
        Path(local_index_dir).expanduser().resolve()
        if local_index_dir is not None
        else project_path(
            "experiments",
            "data",
            "pyserini_ance_faiss_qrecc",
            QRECC_ANCE_INDEX_DIRNAME,
        )
    )
    resolved_log_file = Path(log_file).expanduser().resolve()
    expected_shards = _expected_qrecc_ance_shards()
    before_status = inspect_qrecc_ance_shards(target_dir)
    local_ready = (
        before_status.complete_shards >= expected_shards
        and before_status.indexed_passages >= QRECC_TOTAL_PASSAGES
    )

    azure_status = inspect_qrecc_azure_ance_index(
        sas_file=sas_file,
        url_file=url_file,
        azure_file_share_url=azure_file_share_url,
        azure_shards_path=azure_shards_path,
        expected_shards=expected_shards,
        validate_manifests=validate_azure_manifests,
        timeout=timeout,
    )
    if validate_azure and not azure_status.ready:
        raise RuntimeError(f"QReCC ANCE index is not complete on Azure: {azure_status.to_dict()}")

    target_base_url = f"{azure_file_share_url.rstrip('/')}/{azure_shards_path.strip('/')}"
    redacted_target_url = f"{target_base_url}?***"
    if local_ready and not force:
        return QReCCAnceAzureIndexSyncResult(
            target_dir=target_dir,
            local_index_dir=before_status.index_dir,
            log_file=resolved_log_file,
            azure_target_url=redacted_target_url,
            azure_ready=azure_status.ready,
            azure_complete_shards=azure_status.complete_shards,
            local_complete_shards=before_status.complete_shards,
            local_indexed_passages=before_status.indexed_passages,
            downloaded=False,
            skipped_existing=True,
        )

    sas_query = _qrecc_sas_query_from_file(sas_file=sas_file, url_file=url_file)
    target_url = f"{target_base_url}?{sas_query}"
    from ..dense_ance import download_dense_ance_index_from_azure

    downloaded_index_dir = download_dense_ance_index_from_azure(
        azure_sharded_dir_url=target_url,
        local_sharded_dir=target_dir,
        log_path=resolved_log_file,
        install_azcopy=install_azcopy,
        progress=progress,
    )
    after_status = inspect_qrecc_ance_shards(downloaded_index_dir)
    if (
        after_status.complete_shards < expected_shards
        or after_status.indexed_passages < QRECC_TOTAL_PASSAGES
    ):
        raise RuntimeError(f"Downloaded QReCC ANCE index is incomplete: {after_status.to_dict()}")

    return QReCCAnceAzureIndexSyncResult(
        target_dir=target_dir,
        local_index_dir=downloaded_index_dir,
        log_file=resolved_log_file,
        azure_target_url=redacted_target_url,
        azure_ready=azure_status.ready,
        azure_complete_shards=azure_status.complete_shards,
        local_complete_shards=after_status.complete_shards,
        local_indexed_passages=after_status.indexed_passages,
        downloaded=True,
        skipped_existing=False,
    )


def qrecc_ance_session_shard_range(
    session_id: str,
    *,
    total_shards: int | None = None,
) -> tuple[int, int]:
    """Return the default inclusive QReCC ANCE shard range for a Colab session."""

    resolved_total = total_shards or _expected_qrecc_ance_shards()
    midpoint = resolved_total // 2
    if session_id == "session_1":
        return 0, midpoint - 1
    if session_id == "session_2":
        return midpoint, resolved_total - 1
    raise ValueError("session_id must be 'session_1' or 'session_2'.")


def sync_qrecc_ance_shard_range_from_azure(
    *,
    local_index_dir: Path | str | None = None,
    session_id: str | None = None,
    start_shard: int | None = None,
    end_shard: int | None = None,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_shards_path: str = QRECC_AZURE_ANCE_SHARDS_PATH,
    validate_azure: bool = True,
    validate_azure_manifests: bool = False,
    force: bool = False,
    log_file: Path | str = "/content/azcopy_qrecc_ance_shard_range_download.log",
    install_azcopy: bool = True,
    progress: bool = True,
    timeout: float = 30.0,
) -> QReCCAnceShardRangeSyncResult:
    """Download an inclusive QReCC ANCE shard range from Azure.

    The remote index is read-only from this function; no Azure shard is deleted
    or moved.
    """

    expected_total = _expected_qrecc_ance_shards()
    if session_id is not None and (start_shard is None or end_shard is None):
        start_shard, end_shard = qrecc_ance_session_shard_range(session_id, total_shards=expected_total)
    if start_shard is None or end_shard is None:
        raise ValueError("Pass either session_id or both start_shard and end_shard.")
    if start_shard < 0 or end_shard < start_shard or end_shard >= expected_total:
        raise ValueError(f"Invalid QReCC shard range: {start_shard}..{end_shard}")

    target_dir = (
        Path(local_index_dir).expanduser().resolve()
        if local_index_dir is not None
        else project_path(
            "experiments",
            "data",
            "pyserini_ance_faiss_qrecc",
            QRECC_ANCE_INDEX_DIRNAME,
        )
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved_log_file = Path(log_file).expanduser().resolve()
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)

    expected_shards = end_shard - start_shard + 1
    expected_start_passage = start_shard * QRECC_ANCE_SHARD_PASSAGES
    expected_end_passage = min((end_shard + 1) * QRECC_ANCE_SHARD_PASSAGES, QRECC_TOTAL_PASSAGES)
    try:
        from ..dense_ance import summarize_dense_faiss_shards

        before_summary = summarize_dense_faiss_shards(
            target_dir,
            start_shard=start_shard,
            end_shard=end_shard,
        )
    except (FileNotFoundError, RuntimeError, ValueError, KeyError):
        before_summary = None

    azure_status = inspect_qrecc_azure_ance_index(
        sas_file=sas_file,
        url_file=url_file,
        azure_file_share_url=azure_file_share_url,
        azure_shards_path=azure_shards_path,
        expected_shards=expected_total,
        validate_manifests=validate_azure_manifests,
        timeout=timeout,
    )
    if validate_azure and not azure_status.ready:
        raise RuntimeError(f"QReCC ANCE index is not complete on Azure: {azure_status.to_dict()}")

    sas_query = _qrecc_sas_query_from_file(sas_file=sas_file, url_file=url_file)
    target_base_url = f"{azure_file_share_url.rstrip('/')}/{azure_shards_path.strip('/')}"
    redacted_target_url = f"{target_base_url}?***"
    local_range_ready = (
        before_summary is not None
        and before_summary.shard_count == expected_shards
        and before_summary.first_passage_start == expected_start_passage
        and before_summary.last_passage_end == expected_end_passage
        and before_summary.contiguous
    )
    if local_range_ready and not force:
        return QReCCAnceShardRangeSyncResult(
            target_dir=target_dir,
            local_index_dir=before_summary.index_dir,
            log_file=resolved_log_file,
            azure_target_url=redacted_target_url,
            session_id=session_id,
            start_shard=start_shard,
            end_shard=end_shard,
            expected_shards=expected_shards,
            copied_shards=0,
            skipped_existing_shards=expected_shards,
            local_complete_shards=before_summary.shard_count,
            local_indexed_passages=before_summary.num_passages,
        )

    target_url = f"{target_base_url}?{sas_query}"
    include_path = ";".join(f"shard_{shard_id:06d}" for shard_id in range(start_shard, end_shard + 1))
    from ..dense_ance import download_dense_ance_index_from_azure

    download_dense_ance_index_from_azure(
        azure_sharded_dir_url=target_url,
        local_sharded_dir=target_dir.parent,
        log_path=resolved_log_file,
        install_azcopy=install_azcopy,
        progress=progress,
        include_path=include_path,
        from_to="FileSMBLocal",
    )
    after_summary = summarize_dense_faiss_shards(
        target_dir,
        start_shard=start_shard,
        end_shard=end_shard,
    )
    if (
        after_summary.shard_count != expected_shards
        or after_summary.first_passage_start != expected_start_passage
        or after_summary.last_passage_end != expected_end_passage
        or not after_summary.contiguous
    ):
        raise RuntimeError(f"Downloaded QReCC ANCE shard range is incomplete: {after_summary.to_dict()}")

    return QReCCAnceShardRangeSyncResult(
        target_dir=target_dir,
        local_index_dir=after_summary.index_dir,
        log_file=resolved_log_file,
        azure_target_url=redacted_target_url,
        session_id=session_id,
        start_shard=start_shard,
        end_shard=end_shard,
        expected_shards=expected_shards,
        copied_shards=expected_shards,
        skipped_existing_shards=0,
        local_complete_shards=after_summary.shard_count,
        local_indexed_passages=after_summary.num_passages,
    )


def qrecc_partial_dense_dump_azure_path(
    *,
    run_id: str,
    session_id: str,
    azure_results_path: str = QRECC_AZURE_ANCE_RESULTS_PATH,
) -> str:
    return f"{azure_results_path.strip('/')}/{run_id.strip('/')}/{session_id}.jsonl"


def qrecc_partial_dense_dump_exists_on_azure(
    *,
    run_id: str,
    session_id: str,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_results_path: str = QRECC_AZURE_ANCE_RESULTS_PATH,
    timeout: float = 30.0,
) -> bool:
    sas_query = _qrecc_sas_query_from_file(sas_file=sas_file, url_file=url_file)
    azure_path = qrecc_partial_dense_dump_azure_path(
        run_id=run_id,
        session_id=session_id,
        azure_results_path=azure_results_path,
    )
    url = f"{azure_file_share_url.rstrip('/')}/{azure_path}?{sas_query}"
    return _azure_file_exists(url, timeout=timeout)


def upload_qrecc_partial_dense_dump_to_azure(
    local_file: Path | str,
    *,
    run_id: str,
    session_id: str,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_results_path: str = QRECC_AZURE_ANCE_RESULTS_PATH,
    install_azcopy: bool = True,
    progress: bool = True,
) -> QReCCAzureFileTransferResult:
    source = Path(local_file).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Partial dense dump does not exist: {source}")
    sas_query = _qrecc_sas_query_from_file(sas_file=sas_file, url_file=url_file)
    azure_path = qrecc_partial_dense_dump_azure_path(
        run_id=run_id,
        session_id=session_id,
        azure_results_path=azure_results_path,
    )
    target_url = f"{azure_file_share_url.rstrip('/')}/{azure_path}?{sas_query}"
    redacted_url = target_url.split("?", 1)[0] + "?***"
    azcopy = _ensure_azcopy(None, install=install_azcopy)
    command = [
        str(azcopy),
        "copy",
        str(source),
        target_url,
        "--from-to=LocalFileSMB",
        "--overwrite=true",
        "--check-length=true",
        "--output-type=text",
        "--log-level=INFO",
    ]
    if progress:
        print("upload QReCC partial dense dump to Azure:", redacted_url, flush=True)
        subprocess.run(command, check=True)
    else:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return QReCCAzureFileTransferResult(
        local_file=source,
        azure_url=redacted_url,
        transferred=True,
        skipped_existing=False,
        exists_remote=True,
    )


def download_qrecc_partial_dense_dump_from_azure(
    *,
    run_id: str,
    session_id: str,
    local_file: Path | str,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_results_path: str = QRECC_AZURE_ANCE_RESULTS_PATH,
    install_azcopy: bool = True,
    progress: bool = True,
    timeout: float = 30.0,
) -> QReCCAzureFileTransferResult:
    target = Path(local_file).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    sas_query = _qrecc_sas_query_from_file(sas_file=sas_file, url_file=url_file)
    azure_path = qrecc_partial_dense_dump_azure_path(
        run_id=run_id,
        session_id=session_id,
        azure_results_path=azure_results_path,
    )
    source_url = f"{azure_file_share_url.rstrip('/')}/{azure_path}?{sas_query}"
    redacted_url = source_url.split("?", 1)[0] + "?***"
    if not _azure_file_exists(source_url, timeout=timeout):
        return QReCCAzureFileTransferResult(
            local_file=target,
            azure_url=redacted_url,
            transferred=False,
            skipped_existing=False,
            exists_remote=False,
        )
    azcopy = _ensure_azcopy(None, install=install_azcopy)
    command = [
        str(azcopy),
        "copy",
        source_url,
        str(target),
        "--from-to=FileSMBLocal",
        "--overwrite=ifSourceNewer",
        "--check-length=true",
        "--output-type=text",
        "--log-level=INFO",
    ]
    if progress:
        print("download QReCC partial dense dump from Azure:", redacted_url, flush=True)
        subprocess.run(command, check=True)
    else:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return QReCCAzureFileTransferResult(
        local_file=target,
        azure_url=redacted_url,
        transferred=True,
        skipped_existing=False,
        exists_remote=True,
    )


def evaluate_qrecc_ance_partial_run_from_azure(
    *,
    run_id: str,
    data_dir: Path | str | None = None,
    local_run_dir: Path | str | None = None,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
    session_ids: Sequence[str] = ("session_1", "session_2"),
    final_top_k: int = 100,
    ks: Sequence[int] = (3, 10, 100),
    bootstrap_samples: int = 10_000,
    seed: int = 42,
    azure_file_share_url: str = QRECC_AZURE_FILE_SHARE_URL,
    azure_results_path: str = QRECC_AZURE_ANCE_RESULTS_PATH,
    install_azcopy: bool = True,
    progress: bool = True,
    download_remote: bool = True,
) -> QReCCPartialDenseAzureEvalResult:
    """Download QReCC ANCE partial dumps from Azure, merge, and evaluate locally."""

    from ..evaluation import evaluate_retrieval_run
    from ..pipelines import (
        load_partial_retrieval_hits,
        merge_partial_retrieval_dump_files,
    )

    normalized_sessions = tuple(str(session_id) for session_id in session_ids)
    if len(normalized_sessions) < 2:
        raise ValueError("At least two session_ids are required for score-based partial merge.")
    if final_top_k < 1:
        raise ValueError("final_top_k must be >= 1.")

    resources = resolve_qrecc_resources(data_dir)
    resolved_local_run_dir = (
        Path(local_run_dir).expanduser().resolve()
        if local_run_dir is not None
        else project_path("experiments", "results", "qrecc", "ance_partial_runs", run_id)
    )
    resolved_local_run_dir.mkdir(parents=True, exist_ok=True)

    downloads: list[QReCCAzureFileTransferResult] = []
    local_dump_files: list[Path] = []
    missing_sessions: list[str] = []
    for session_id in normalized_sessions:
        local_file = resolved_local_run_dir / f"{session_id}.jsonl"
        if not download_remote:
            if not local_file.is_file():
                raise FileNotFoundError(
                    f"Missing local partial run: {local_file}. Run both QReCC "
                    "shard sessions with the same run_id before merge/evaluation."
                )
            local_dump_files.append(local_file)
            continue
        download = download_qrecc_partial_dense_dump_from_azure(
            run_id=run_id,
            session_id=session_id,
            local_file=local_file,
            sas_file=sas_file,
            url_file=url_file,
            azure_file_share_url=azure_file_share_url,
            azure_results_path=azure_results_path,
            install_azcopy=install_azcopy,
            progress=progress,
        )
        downloads.append(download)
        if not download.exists_remote:
            missing_sessions.append(session_id)
        else:
            local_dump_files.append(local_file)

    if missing_sessions:
        raise FileNotFoundError(
            "Missing QReCC ANCE partial dump(s) on Azure for "
            f"run_id={run_id}: {missing_sessions}"
        )

    hit_maps = [load_partial_retrieval_hits(path) for path in local_dump_files]
    sample_id_sets = [set(hit_map) for hit_map in hit_maps]
    if any(not sample_ids for sample_ids in sample_id_sets):
        empty_sessions = [
            session_id
            for session_id, sample_ids in zip(normalized_sessions, sample_id_sets, strict=True)
            if not sample_ids
        ]
        raise ValueError(f"Empty partial dump(s): {empty_sessions}")

    first_sample_ids = sample_id_sets[0]
    mismatched_sessions = [
        session_id
        for session_id, sample_ids in zip(normalized_sessions[1:], sample_id_sets[1:], strict=True)
        if sample_ids != first_sample_ids
    ]
    if mismatched_sessions:
        raise ValueError(
            "QReCC ANCE partial dumps do not contain identical sample_id sets. "
            f"Reference={normalized_sessions[0]}, mismatched={mismatched_sessions}"
        )

    merged_file = resolved_local_run_dir / "merged.jsonl"
    merge_result = merge_partial_retrieval_dump_files(
        local_dump_files,
        top_k=final_top_k,
        output_file=merged_file,
    )

    eval_rewrites = _qrecc_test_eval_frame(resources, sample_ids=first_sample_ids)
    missing_gold_sample_ids = sorted(first_sample_ids - set(eval_rewrites["sample_id"]))
    if missing_gold_sample_ids:
        raise ValueError(
            "Partial dump sample_id(s) missing from local QReCC test gold: "
            f"{missing_gold_sample_ids[:10]}"
        )

    evaluation = evaluate_retrieval_run(
        rewrites=eval_rewrites,
        hits_by_sample_id=merge_result.hits_by_sample_id,
        ks=ks,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )

    total_test_gold_queries = len(_qrecc_test_eval_frame(resources))
    return QReCCPartialDenseAzureEvalResult(
        run_id=run_id,
        session_ids=normalized_sessions,
        local_run_dir=resolved_local_run_dir,
        local_dump_files=tuple(local_dump_files),
        merged_file=merged_file,
        downloads=tuple(downloads),
        merge_result=merge_result,
        evaluation=evaluation,
        eval_rewrites=eval_rewrites,
        final_top_k=final_top_k,
        total_test_gold_queries=total_test_gold_queries,
        evaluated_queries=len(evaluation.per_query),
    )


def _qrecc_test_eval_frame(
    resources: QReCCResources,
    *,
    sample_ids: set[str] | None = None,
) -> pd.DataFrame:
    frame = load_qrecc_frame(resources, prefer_processed=True, gold_only=True)
    frame = frame[frame["split"].eq("test")].copy()
    frame["sample_id"] = frame.apply(_qrecc_eval_sample_id, axis=1)
    rewrite_text = frame["rewrite"].fillna("").astype(str)
    question_text = frame["question"].fillna("").astype(str)
    frame["query"] = rewrite_text.where(rewrite_text.str.len() > 0, question_text)
    frame["rewrite"] = frame["query"]
    if sample_ids is not None:
        frame = frame[frame["sample_id"].isin(sample_ids)].copy()
    sort_columns = [column for column in ("conv_id", "turn_id") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns)
    return frame.reset_index(drop=True)


def _qrecc_eval_sample_id(row: pd.Series) -> str:
    return (
        f"{_format_sample_id_component(row.get('split', 'test'))}:"
        f"{_format_sample_id_component(row.get('conv_id'))}:"
        f"{_format_sample_id_component(row.get('turn_id'))}"
    )


def _format_sample_id_component(value: Any) -> str:
    try:
        if pd.isna(value):
            return "None"
    except TypeError:
        pass
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_qrecc_ance_faiss_shards(
    *,
    collection: QReCCCollectionBuildResult | Path | str | None = None,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    start_passage: int,
    end_passage: int | None = None,
    output_shard_offset: int | None = None,
    offsets_file: Path | str | None = None,
    batch_size: int = 160,
    batches_per_shard: int = 625,
    device: str = "cuda:0",
    max_length: int = 384,
    dimension: int = 768,
    total_passages: int | None = None,
    script_path: Path | str | None = None,
    run_manifest_name: str = "run_manifest_qrecc_range.json",
    upload_completed_shards_url: str | None = None,
    upload_completed_shards_url_file: Path | str | None = None,
    upload_completed_shards_env: str = "AZURE_QRECC_ANCE_SHARDS_SAS_URL",
    upload_completed_shards_subdir: str | None = None,
    upload_from_to: str = "LocalFileSMB",
    delete_local_after_upload: bool = False,
    max_pending_uploads: int = 2,
    shard_order: str = "forward",
    azcopy_path: Path | str | None = None,
    install_azcopy: bool = True,
    progress: bool = True,
    dry_run: bool = False,
) -> "IndexBuildResult":
    """Build a local QReCC range with the shared ANCE passage encoder."""
    from ..azure_indexing import build_ance_faiss_shards
    if upload_completed_shards_url or upload_completed_shards_url_file or delete_local_after_upload:
        raise ValueError("This build is local-only; copy completed indices separately.")
    if shard_order != "forward":
        raise ValueError("Local builds use forward passage ordering; resume matching completed shards.")
    source = _resolve_qrecc_collection_jsonl(collection, data_dir=data_dir)
    output = Path(output_dir) if output_dir is not None else project_path(
        "experiments", "data", "pyserini_ance_faiss_qrecc", "faiss_flat_index_full_sharded")
    return build_ance_faiss_shards(
        collection_file=source, output_dir=output, batch_size=batch_size,
        batches_per_shard=batches_per_shard, device=device, max_length=max_length,
        dimension=dimension, start_passage=start_passage,
        end_passage=end_passage or total_passages or _manifest_num_passages(source) or QRECC_TOTAL_PASSAGES,
        output_shard_offset=output_shard_offset, text_field="contents",
        progress=progress, dry_run=dry_run)


def build_qrecc_ance_faiss_shards_reverse(
    *,
    collection: QReCCCollectionBuildResult | Path | str | None = None,
    data_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
    start_shard: int,
    stop_shard: int,
    batch_size: int = 160,
    batches_per_shard: int = 625,
    device: str = "cuda:0",
    max_length: int = 384,
    dimension: int = 768,
    total_passages: int | None = None,
    upload_completed_shards_url: str | None = None,
    upload_completed_shards_url_file: Path | str | None = None,
    upload_completed_shards_env: str = "AZURE_QRECC_ANCE_SHARDS_SAS_URL",
    upload_completed_shards_subdir: str | None = None,
    upload_from_to: str = "LocalFileSMB",
    delete_local_after_upload: bool = False,
    max_pending_uploads: int = 2,
    azcopy_path: Path | str | None = None,
    install_azcopy: bool = True,
    progress: bool = True,
    dry_run: bool = False,
) -> "IndexBuildResult":
    """Build a shard interval in descending shard-id order.

    ``start_shard`` is inclusive and must be >= ``stop_shard``. With the
    default QReCC shard size, ``start_shard=396, stop_shard=350`` encodes
    shard_000396 down to shard_000350.
    """

    if start_shard < stop_shard:
        raise ValueError("start_shard must be >= stop_shard for reverse shard building.")

    shard_passages = batch_size * batches_per_shard
    resolved_total_passages = total_passages or QRECC_TOTAL_PASSAGES
    start_passage = stop_shard * shard_passages
    end_passage = min((start_shard + 1) * shard_passages, resolved_total_passages)

    return build_qrecc_ance_faiss_shards(
        collection=collection,
        data_dir=data_dir,
        output_dir=output_dir,
        start_passage=start_passage,
        end_passage=end_passage,
        output_shard_offset=stop_shard,
        batch_size=batch_size,
        batches_per_shard=batches_per_shard,
        device=device,
        max_length=max_length,
        dimension=dimension,
        total_passages=resolved_total_passages,
        run_manifest_name="run_manifest_qrecc_reverse_range.json",
        upload_completed_shards_url=upload_completed_shards_url,
        upload_completed_shards_url_file=upload_completed_shards_url_file,
        upload_completed_shards_env=upload_completed_shards_env,
        upload_completed_shards_subdir=upload_completed_shards_subdir,
        upload_from_to=upload_from_to,
        delete_local_after_upload=delete_local_after_upload,
        max_pending_uploads=max_pending_uploads,
        shard_order="reverse",
        azcopy_path=azcopy_path,
        install_azcopy=install_azcopy,
        progress=progress,
        dry_run=dry_run,
    )


def build_qrecc_ance_colab_tail(
    *,
    collection: QReCCCollectionBuildResult | Path | str | None = None,
    output_dir: Path | str | None = None,
    start_passage: int = QRECC_ANCE_COLAB_TAIL_START,
    end_passage: int | None = None,
    device: str = "cuda:0",
    batch_size: int = 160,
    batches_per_shard: int = 625,
    upload_completed_shards_url: str | None = None,
    upload_completed_shards_url_file: Path | str | None = None,
    upload_completed_shards_env: str = "AZURE_QRECC_ANCE_SHARDS_SAS_URL",
    upload_completed_shards_subdir: str | None = None,
    delete_local_after_upload: bool = False,
    max_pending_uploads: int = 2,
    progress: bool = True,
    dry_run: bool = False,
) -> "IndexBuildResult":
    """Convenience wrapper for the current Colab tail split."""

    return build_qrecc_ance_faiss_shards(
        collection=collection,
        output_dir=output_dir,
        start_passage=start_passage,
        end_passage=end_passage,
        output_shard_offset=start_passage // (batch_size * batches_per_shard),
        batch_size=batch_size,
        batches_per_shard=batches_per_shard,
        device=device,
        run_manifest_name="run_manifest_qrecc_colab_tail.json",
        upload_completed_shards_url=upload_completed_shards_url,
        upload_completed_shards_url_file=upload_completed_shards_url_file,
        upload_completed_shards_env=upload_completed_shards_env,
        upload_completed_shards_subdir=upload_completed_shards_subdir,
        delete_local_after_upload=delete_local_after_upload,
        max_pending_uploads=max_pending_uploads,
        progress=progress,
        dry_run=dry_run,
    )


def inspect_qrecc_ance_shards(
    index_dir: Path | str | None = None,
) -> QReCCAnceShardStatus:
    resolved_index_dir = (
        Path(index_dir).expanduser().resolve()
        if index_dir is not None
        else project_path(
            "experiments",
            "data",
            "pyserini_ance_faiss_qrecc",
            "faiss_flat_index_full_sharded",
        )
    )
    if not any(resolved_index_dir.glob("shard_*/manifest.json")):
        nested = resolved_index_dir / QRECC_ANCE_INDEX_DIRNAME
        if any(nested.glob("shard_*/manifest.json")):
            resolved_index_dir = nested
    manifests = []
    if resolved_index_dir.exists():
        for manifest_path in sorted(resolved_index_dir.glob("shard_*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            shard = manifest_path.parent
            count = int(manifest.get("num_passages", 0))
            dimension = int(manifest.get("dimension", 0))
            if (manifest.get("complete") is True and count > 0 and dimension > 0
                    and (shard / "index").is_file()
                    and (shard / "index").stat().st_size == count * dimension * 4 + 45
                    and (shard / "docid").is_file() and (shard / "docid").stat().st_size > 0):
                manifests.append(manifest)

    shard_ids = sorted(int(manifest["shard_id"]) for manifest in manifests)
    indexed_passages = sum(int(manifest.get("num_passages", 0)) for manifest in manifests)
    return QReCCAnceShardStatus(
        index_dir=resolved_index_dir,
        complete_shards=len(shard_ids),
        first_shard_id=shard_ids[0] if shard_ids else None,
        last_shard_id=shard_ids[-1] if shard_ids else None,
        next_shard_id=(shard_ids[-1] + 1) if shard_ids else 0,
        indexed_passages=indexed_passages,
    )


def _load_qrecc_json(path: Path, *, split: str) -> pd.DataFrame:
    rows = _read_qrecc_rows(path)

    records = []
    for row in rows:
        conv_id = row["Conversation_no"]
        turn_id = row["Turn_no"]
        records.append(
            {
                "split": split,
                "conv_id": conv_id,
                "turn_id": turn_id,
                "question": row.get("Question", ""),
                "answers": row.get("Truth_answer") or row.get("Answer", ""),
                "rewrite": row.get("Truth_rewrite") or row.get("Rewrite", ""),
                "context": row.get("Context", []),
                "conversation_source": row.get("Conversation_source"),
                "positive_ctxs": _qrecc_positive_contexts(row),
                "positive_ctx_passage_ids": _qrecc_positive_passage_ids(row),
                "topic_id": row.get("Conversation_source"),
                "topic_switch": False,
                "topics_in_conversation": 1,
            }
        )
    frame = pd.DataFrame(records)
    if frame.shape[0] != len(rows):
        raise RuntimeError(f"Loaded {frame.shape[0]} QReCC rows from {path}, expected {len(rows)}")
    return frame


def _qrecc_positive_contexts(row: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(row.get("positive_ctxs"), list):
        return row["positive_ctxs"]

    contexts = []
    raw_ids = _raw_qrecc_passage_ids(row)
    for index, passage_id in enumerate(_qrecc_positive_passage_ids(row)):
        raw_id = raw_ids[index] if index < len(raw_ids) else None
        contexts.append(
            {
                "passage_id": passage_id,
                "raw_passage_id": raw_id,
                "text": None,
                "title": None,
            }
        )
    return contexts


def _qrecc_positive_passage_ids(row: dict[str, Any]) -> list[str]:
    ids = row.get("positive_ctx_passage_ids")
    if isinstance(ids, str):
        return [ids]
    if isinstance(ids, list):
        return [str(passage_id) for passage_id in ids if passage_id is not None]

    passages = row.get("Truth_passages") or row.get("Passages") or row.get("passages") or []
    if isinstance(passages, str):
        return [passages]
    return [str(passage_id) for passage_id in passages if passage_id is not None]


def _read_qrecc_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected QReCC JSON list at {path}")
    return rows


def _raw_qrecc_passage_ids(row: dict[str, Any]) -> list[str]:
    passages = row.get("Passages") or row.get("Truth_passages") or row.get("passages") or []
    if isinstance(passages, str):
        return [passages]
    return [str(passage_id) for passage_id in passages if passage_id is not None]


def _iter_gold_rawpids(rows: list[dict[str, Any]]) -> Iterable[str]:
    for row in rows:
        yield from _raw_qrecc_passage_ids(row)


def _qrecc_manifest_matches_sources(manifest_json: Path, resources: QReCCResources) -> bool:
    try:
        manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("train_source_name") == resources.train_json.name
        and manifest.get("test_source_name") == resources.test_json.name
    )


def _map_gold_rawpids_from_collection(
    collection_jsonl: Path,
    gold_rawpids: set[str],
    rawpid2pid: dict[str, str],
    *,
    progress: bool,
) -> int:
    num_passages = 0
    total = _manifest_num_passages(collection_jsonl)
    with collection_jsonl.open("r", encoding="utf-8") as handle:
        rows = progress_iter(
            handle,
            total=total,
            desc="map QReCC gold ids",
            unit="passage",
            enabled=progress,
        )
        for line in rows:
            row = json.loads(line, strict=False)
            raw_pid = row.get("raw_passage_id")
            if raw_pid is not None and raw_pid in gold_rawpids:
                rawpid2pid[str(raw_pid)] = str(row["id"])
                if len(rawpid2pid) >= len(gold_rawpids):
                    num_passages += 1
                    break
            num_passages += 1
        close = getattr(rows, "close", None)
        if close is not None:
            close()
    if total is not None:
        return total
    return num_passages


def _add_numeric_positive_ctxs(
    rows: list[dict[str, Any]],
    rawpid2pid: dict[str, str],
) -> tuple[list[dict[str, Any]], int]:
    processed = []
    missing = 0
    for row in rows:
        record = dict(row)
        positive_ctxs = []
        positive_ctx_passage_ids = []
        for raw_pid in _raw_qrecc_passage_ids(row):
            pid = rawpid2pid.get(raw_pid)
            if pid is None:
                missing += 1
                continue
            positive_ctx_passage_ids.append(pid)
            positive_ctxs.append(
                {
                    "passage_id": pid,
                    "raw_passage_id": raw_pid,
                    "text": None,
                    "title": None,
                }
            )
        record["positive_ctxs"] = positive_ctxs
        record["positive_ctx_passage_ids"] = positive_ctx_passage_ids
        processed.append(record)
    return processed, missing


def _qrecc_passage_files(passages_dir: Path) -> list[Path]:
    root = passages_dir / "collection-paragraph"
    if not root.exists():
        root = passages_dir

    files = []
    for subdir in QRECC_PASSAGE_SUBDIRS:
        directory = root / subdir
        if directory.exists():
            files.extend(sorted(directory.glob("*.jsonl")))
    if files:
        return files
    return sorted(root.rglob("*.jsonl"))


def _manifest_num_passages(collection_jsonl: Path) -> int | None:
    manifest = collection_jsonl.parent / "manifest.json"
    if not manifest.exists():
        return None
    try:
        return int(json.loads(manifest.read_text(encoding="utf-8"))["num_passages"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _resolve_qrecc_collection_jsonl(
    collection: QReCCCollectionBuildResult | Path | str | None,
    *,
    data_dir: Path | str | None,
) -> Path:
    if isinstance(collection, QReCCCollectionBuildResult) or hasattr(collection, "collection_jsonl"):
        source = Path(getattr(collection, "collection_jsonl")).expanduser().resolve()
    elif collection is not None:
        source = Path(collection).expanduser().resolve()
    else:
        source = _resources_for_root(default_qrecc_root(data_dir)).collection_jsonl
    if not source.exists():
        raise FileNotFoundError(f"QReCC processed collection is missing: {source}")
    return source


def _default_qrecc_ance_sas_file() -> Path:
    candidates = (
        Path("/content/drive/MyDrive/secrets/ance_sharded_dir_sas.txt"),
        project_path("experiments", "secrets", "ance_sharded_dir_sas.txt"),
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def _default_qrecc_ance_upload_url_file() -> Path:
    if Path("/content").exists():
        return Path("/content/qrecc_ance_shards_upload_url.txt")
    return Path("/tmp/qrecc_ance_shards_upload_url.txt")


def _qrecc_sas_query_from_file(
    *,
    sas_file: Path | str | None = None,
    url_file: Path | str | None = None,
) -> str:
    if url_file is not None:
        resolved_url_file = Path(url_file).expanduser().resolve()
        if not resolved_url_file.exists():
            raise FileNotFoundError(f"QReCC ANCE Azure URL file does not exist: {resolved_url_file}")
        value = resolved_url_file.read_text(encoding="utf-8").strip()
        query = urlsplit(value).query or _extract_sas_query(value)
        if not query:
            raise ValueError(f"QReCC ANCE Azure URL file has no SAS query: {resolved_url_file}")
        return query

    default_url_file = _default_qrecc_ance_upload_url_file()
    if default_url_file.exists():
        value = default_url_file.read_text(encoding="utf-8").strip()
        query = urlsplit(value).query or _extract_sas_query(value)
        if query:
            return query

    resolved_sas_file = (
        Path(sas_file).expanduser().resolve()
        if sas_file is not None
        else _default_qrecc_ance_sas_file()
    )
    if not resolved_sas_file.exists():
        raise FileNotFoundError(f"QReCC ANCE Azure SAS file does not exist: {resolved_sas_file}")
    query = _extract_sas_query(resolved_sas_file.read_text(encoding="utf-8").strip())
    if not query:
        raise ValueError(f"QReCC ANCE Azure SAS file is empty: {resolved_sas_file}")
    return query


def _expected_qrecc_ance_shards() -> int:
    return (QRECC_TOTAL_PASSAGES + QRECC_ANCE_SHARD_PASSAGES - 1) // QRECC_ANCE_SHARD_PASSAGES


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


def _azure_file_exists(file_url: str, *, timeout: float) -> bool:
    try:
        request = Request(file_url, method="HEAD")
        with urlopen(request, timeout=timeout):
            return True
    except HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def _qrecc_local_shard_complete(shard_dir: Path) -> bool:
    manifest_path = shard_dir / "manifest.json"
    if not (manifest_path.exists() and (shard_dir / "index").exists() and (shard_dir / "docid").exists()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("complete") is True


def _extract_sas_query(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    parsed = urlsplit(stripped)
    if parsed.query:
        return parsed.query
    return stripped.lstrip("?")


def _ensure_azcopy(binary_path: Path | None = None, *, install: bool = True) -> Path:
    if binary_path is not None and binary_path.exists():
        return binary_path
    existing = shutil.which("azcopy")
    if existing:
        return Path(existing)
    if not install:
        raise RuntimeError("azcopy is not available. Install azcopy or pass install_azcopy=True.")

    binary = binary_path or Path("/usr/local/bin/azcopy")
    if binary.exists():
        return binary

    archive = Path("/tmp/azcopy.tar.gz")
    extract_dir = Path("/tmp/azcopy_extract")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    urlretrieve(AZCOPY_DOWNLOAD_URL, archive)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(extract_dir)
    candidates = list(extract_dir.glob("*/azcopy")) + list(extract_dir.glob("azcopy"))
    if not candidates:
        raise RuntimeError("Downloaded AzCopy archive did not contain an azcopy binary.")
    binary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], binary)
    binary.chmod(0o755)
    return binary


def _qrecc_collection_passage_count(collection_jsonl: Path) -> int:
    manifest_count = _manifest_num_passages(collection_jsonl)
    if manifest_count is not None:
        return manifest_count
    return _count_jsonl_rows(collection_jsonl)


def _count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _dir_has_entries(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return any(path.iterdir())


def _same_existing_file(source: Path, target: Path) -> bool:
    if not target.exists():
        return False
    try:
        if source.resolve() == target.resolve():
            return True
    except OSError:
        pass
    return target.stat().st_size == source.stat().st_size


def _copy_qrecc_collection_jsonl(
    source: Path,
    output_path: Path,
    *,
    desc: str,
    progress: bool,
) -> int:
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    total = _qrecc_collection_passage_count(source)
    written = 0
    with source.open("r", encoding="utf-8") as source_handle:
        rows = progress_iter(
            source_handle,
            total=total,
            desc=desc,
            unit="passage",
            enabled=progress,
        )
        with tmp_output.open("w", encoding="utf-8") as output:
            for line in rows:
                output.write(line)
                written += 1
        close = getattr(rows, "close", None)
        if close is not None:
            close()

    tmp_output.replace(output_path)
    return written


def _qrecc_offsets_are_fresh(collection_jsonl: Path, offsets_path: Path) -> bool:
    return (
        offsets_path.exists()
        and offsets_path.stat().st_size % array(QRECC_OFFSET_TYPECODE).itemsize == 0
        and offsets_path.stat().st_mtime >= collection_jsonl.stat().st_mtime
    )


def _read_qrecc_offsets(offsets_path: Path) -> array:
    offsets = array(QRECC_OFFSET_TYPECODE)
    itemsize = offsets.itemsize
    file_size = offsets_path.stat().st_size
    if file_size % itemsize != 0:
        raise ValueError(f"Invalid QReCC offset file size: {offsets_path}")
    with offsets_path.open("rb") as handle:
        offsets.fromfile(handle, file_size // itemsize)
    return offsets


def _qrecc_downloads() -> list[DownloadSpec]:
    return [
        DownloadSpec(
            url=f"{ZENODO_RECORD}/{QRECC_TRAIN_FILENAME}",
            relative_path=Path("downloads") / QRECC_TRAIN_FILENAME,
            label="SCAI QReCC train turns",
        ),
        DownloadSpec(
            url=f"{ZENODO_RECORD}/{QRECC_TEST_FILENAME}",
            relative_path=Path("downloads") / QRECC_TEST_FILENAME,
            label="SCAI QReCC test turns",
        ),
    ]


def _resources_for_root(root: Path) -> QReCCResources:
    data_root = root / "downloads"
    return QReCCResources(
        root=root,
        train_json=data_root / QRECC_TRAIN_FILENAME,
        test_json=data_root / QRECC_TEST_FILENAME,
        passages_zip=data_root / "passages.zip",
    )
