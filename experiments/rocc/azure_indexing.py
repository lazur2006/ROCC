"""Corpus dataloaders and JSONL collection writers for dense index encoding."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from .datasets.topiocqa import (
    TOPIOCQA_CORPUS_PASSAGES,
    ensure_topiocqa_resources,
    iter_topiocqa_corpus_rows,
)
from .paths import project_path
from .progress import progress_iter
from .retrievers import DEFAULT_ANCE_ENCODER, configure_java21

try:
    from torch.utils.data import IterableDataset
except ModuleNotFoundError:
    class IterableDataset:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class TopiOCQAIndexEncodingConfig:
    batch_size: int = 160
    max_passages: int | None = None
    include_title: bool = False
    num_workers: int = 0
    input_shard_id: int = 0
    input_shard_num: int = 1
    data_dir: Path | None = None
    force_download: bool = False
    progress: bool = True


@dataclass(frozen=True)
class CollectionWriteResult:
    output_file: Path
    written_passages: int
    skipped_existing: bool


@dataclass(frozen=True)
class IndexBuildResult:
    index_dir: Path
    command: list[str] | None
    built: bool
    log_file: Path | None = None


@dataclass(frozen=True)
class LuceneIndexStatus:
    index_dir: Path
    exists: bool
    loadable: bool
    complete: bool
    num_docs: int | None = None
    source_passages: int | None = None
    log_file: Path | None = None
    log_exit_code: int | None = None
    log_indexing_complete: bool = False
    log_indexed_docs: int | None = None
    log_empty_docs: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "index_dir": str(self.index_dir),
            "exists": self.exists,
            "loadable": self.loadable,
            "complete": self.complete,
            "num_docs": self.num_docs,
            "source_passages": self.source_passages,
            "log_file": str(self.log_file) if self.log_file is not None else None,
            "log_exit_code": self.log_exit_code,
            "log_indexing_complete": self.log_indexing_complete,
            "log_indexed_docs": self.log_indexed_docs,
            "log_empty_docs": self.log_empty_docs,
            "error": self.error,
        }


class TopiOCQACorpusIterableDataset(IterableDataset):
    """Streaming TopiOCQA corpus dataset for PyTorch DataLoader."""

    def __init__(
        self,
        corpus_tsv: Path,
        *,
        max_passages: int | None = None,
        include_title: bool = False,
        input_shard_id: int = 0,
        input_shard_num: int = 1,
    ) -> None:
        if input_shard_num < 1:
            raise ValueError("input_shard_num must be >= 1")
        if input_shard_id < 0 or input_shard_id >= input_shard_num:
            raise ValueError("input_shard_id must be in [0, input_shard_num)")
        self.corpus_tsv = corpus_tsv
        self.max_passages = max_passages
        self.include_title = include_title
        self.input_shard_id = input_shard_id
        self.input_shard_num = input_shard_num

    def __iter__(self) -> Iterator[dict[str, str]]:
        worker_id = 0
        worker_count = 1
        try:
            from torch.utils.data import get_worker_info

            worker_info = get_worker_info()
        except ModuleNotFoundError:
            worker_info = None
        if worker_info is not None:
            worker_id = worker_info.id
            worker_count = worker_info.num_workers

        emitted = 0
        local_index = 0
        for global_index, row in enumerate(iter_topiocqa_corpus_rows(self.corpus_tsv)):
            if global_index % self.input_shard_num != self.input_shard_id:
                continue
            if local_index % worker_count != worker_id:
                local_index += 1
                continue
            local_index += 1
            if self.max_passages is not None and emitted >= self.max_passages:
                break
            text = row["text"]
            if self.include_title and row.get("title"):
                text = f"{row['title']} {text}"
            emitted += 1
            yield {"docid": str(row["id"]), "text": text}


def build_topiocqa_index_dataloader(
    *,
    batch_size: int = 160,
    max_passages: int | None = None,
    include_title: bool = False,
    num_workers: int = 0,
    input_shard_id: int = 0,
    input_shard_num: int = 1,
    data_dir: Path | str | None = None,
    force_download: bool = False,
    progress: bool = True,
) -> Any:
    try:
        from torch.utils.data import DataLoader
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for the index encoding dataloader.") from exc

    resources = ensure_topiocqa_resources(
        data_dir,
        force_download=force_download,
        progress=progress,
    )
    dataset = TopiOCQACorpusIterableDataset(
        resources.corpus_tsv,
        max_passages=max_passages,
        include_title=include_title,
        input_shard_id=input_shard_id,
        input_shard_num=input_shard_num,
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def write_topiocqa_ance_collection(
    *,
    output_file: Path | str | None = None,
    max_passages: int | None = None,
    include_title: bool = False,
    force: bool = False,
    data_dir: Path | str | None = None,
    force_download: bool = False,
    progress: bool = True,
) -> CollectionWriteResult:
    resources = ensure_topiocqa_resources(
        data_dir,
        force_download=force_download,
        progress=progress,
    )
    output_path = Path(output_file).expanduser().resolve() if output_file is not None else (
        project_path(
            "experiments",
            "data",
            "pyserini_ance_faiss_topiocqa",
            "collection",
            "full_wiki_segments_ance.jsonl",
        )
    )

    if output_path.exists() and not force:
        return CollectionWriteResult(
            output_file=output_path,
            written_passages=count_jsonl_rows(output_path),
            skipped_existing=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    target_total = TOPIOCQA_CORPUS_PASSAGES
    if max_passages is not None:
        target_total = min(max_passages, TOPIOCQA_CORPUS_PASSAGES)

    rows = iter_topiocqa_corpus_rows(resources.corpus_tsv, max_passages=max_passages)
    rows = progress_iter(
        rows,
        total=target_total,
        desc="write ANCE JSONL",
        unit="passage",
        enabled=progress,
    )

    written = 0
    with tmp_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = row["text"]
            if include_title and row.get("title"):
                text = f"{row['title']} {text}"
            handle.write(
                json.dumps({"id": str(row["id"]), "text": text}, ensure_ascii=False) + "\n"
            )
            written += 1

    tmp_output.replace(output_path)
    return CollectionWriteResult(
        output_file=output_path,
        written_passages=written,
        skipped_existing=False,
    )


def write_topiocqa_bm25_collection(
    *,
    output_file: Path | str | None = None,
    max_passages: int | None = None,
    force: bool = False,
    data_dir: Path | str | None = None,
    force_download: bool = False,
    progress: bool = True,
) -> CollectionWriteResult:
    resources = ensure_topiocqa_resources(
        data_dir,
        force_download=force_download,
        progress=progress,
    )
    output_path = Path(output_file).expanduser().resolve() if output_file is not None else (
        project_path("experiments", "data", "pyserini_bm25_lucene_topiocqa", "collection", "docs.jsonl")
    )
    if output_path.exists() and not force:
        return CollectionWriteResult(
            output_file=output_path,
            written_passages=count_jsonl_rows(output_path),
            skipped_existing=True,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    target_total = TOPIOCQA_CORPUS_PASSAGES if max_passages is None else min(max_passages, TOPIOCQA_CORPUS_PASSAGES)
    rows = progress_iter(
        iter_topiocqa_corpus_rows(resources.corpus_tsv, max_passages=max_passages),
        total=target_total,
        desc="write BM25 JSONL",
        unit="passage",
        enabled=progress,
    )

    written = 0
    with tmp_output.open("w", encoding="utf-8") as handle:
        for row in rows:
            title = str(row.get("title") or "")
            normalized_title = title.replace(" [SEP] ", " ")
            text = str(row.get("text") or "")
            handle.write(
                json.dumps(
                    {
                        "id": str(row["id"]),
                        "contents": f"{normalized_title} {text}".strip(),
                        "title": title,
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    tmp_output.replace(output_path)
    return CollectionWriteResult(output_file=output_path, written_passages=written, skipped_existing=False)


def build_bm25_lucene_index(
    *,
    collection_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
    threads: int = 8,
    force: bool = False,
    progress: bool = True,
    log_file: Path | str | None = None,
) -> IndexBuildResult:
    collection = Path(collection_dir).expanduser().resolve() if collection_dir else (
        project_path("experiments", "data", "pyserini_bm25_lucene_topiocqa", "collection")
    )
    index = Path(index_dir).expanduser().resolve() if index_dir else (
        project_path("experiments", "data", "pyserini_bm25_lucene_topiocqa", "lucene_index")
    )
    if index.exists() and not force:
        return IndexBuildResult(index_dir=index, command=None, built=False)
    if force and index.exists():
        shutil.rmtree(index)
    if not collection.exists():
        raise FileNotFoundError(f"BM25 collection dir missing: {collection}")

    os.environ.setdefault("OPENAI_API_KEY", "pyserini-bm25-index-no-openai-call")
    env = os.environ.copy()
    configure_java21(env)

    command = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection),
        "--index",
        str(index),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    resolved_log_file = Path(log_file).expanduser().resolve() if log_file else (
        project_path("experiments", "logs", "bm25_lucene_index.log")
    )
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
    if progress:
        print(f"BM25 Lucene indexing log: {resolved_log_file}", flush=True)
    returncode, tail = _run_streaming_command(
        command,
        env=env,
        log_file=resolved_log_file,
        progress=progress,
    )
    if returncode != 0:
        raise RuntimeError(
            "Pyserini Lucene indexing failed.\n"
            f"log_file: {resolved_log_file}\n"
            f"last_log_lines:\n{''.join(tail)}"
        )
    return IndexBuildResult(index_dir=index, command=command, built=True, log_file=resolved_log_file)


def inspect_bm25_lucene_index(
    index_dir: Path | str,
    *,
    source_passages: int | None = None,
    log_file: Path | str | None = None,
) -> LuceneIndexStatus:
    index = Path(index_dir).expanduser().resolve()
    resolved_log_file = Path(log_file).expanduser().resolve() if log_file is not None else None
    if not index.exists():
        return LuceneIndexStatus(
            index_dir=index,
            exists=False,
            loadable=False,
            complete=False,
            source_passages=source_passages,
            log_file=resolved_log_file,
            error="index directory does not exist",
        )

    log_info = _parse_bm25_lucene_log(resolved_log_file) if resolved_log_file is not None else {}
    loaded_docs, load_error = _load_lucene_num_docs(index)
    loadable = loaded_docs is not None

    complete = False
    if loadable:
        log_indexed_docs = log_info.get("log_indexed_docs")
        log_empty_docs = log_info.get("log_empty_docs")
        log_exit_code = log_info.get("log_exit_code")
        log_indexing_complete = bool(log_info.get("log_indexing_complete", False))
        if resolved_log_file is None or not resolved_log_file.exists():
            complete = loaded_docs > 0
        elif (
            log_indexing_complete
            and log_exit_code == 0
            and log_indexed_docs == loaded_docs
        ):
            complete = (
                source_passages is None
                or log_empty_docs is None
                or log_indexed_docs + log_empty_docs == source_passages
            )

    return LuceneIndexStatus(
        index_dir=index,
        exists=True,
        loadable=loadable,
        complete=complete,
        num_docs=loaded_docs,
        source_passages=source_passages,
        log_file=resolved_log_file,
        log_exit_code=log_info.get("log_exit_code"),
        log_indexing_complete=bool(log_info.get("log_indexing_complete", False)),
        log_indexed_docs=log_info.get("log_indexed_docs"),
        log_empty_docs=log_info.get("log_empty_docs"),
        error=load_error,
    )


def _run_streaming_command(
    command: list[str],
    *,
    env: dict[str, str],
    log_file: Path,
    progress: bool,
    tail_lines: int = 80,
) -> tuple[int, list[str]]:
    tail: deque[str] = deque(maxlen=tail_lines)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"[{started_at}] command: {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        if process.stdout is None:
            raise RuntimeError("Failed to open subprocess stdout stream.")
        with process.stdout:
            for line in process.stdout:
                tail.append(line)
                log.write(line)
                log.flush()
                if progress:
                    print(line, end="", flush=True)
        returncode = process.wait()
        finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        final_line = f"[{finished_at}] exit_code: {returncode}\n"
        tail.append(final_line)
        log.write(final_line)
        log.flush()
    return returncode, list(tail)


def _load_lucene_num_docs(index: Path) -> tuple[int | None, str | None]:
    env = os.environ.copy()
    env.setdefault("OPENAI_API_KEY", "pyserini-bm25-rocc-no-openai-call")
    configure_java21(env)

    code = """
import json
import sys
from pyserini.search.lucene import LuceneSearcher
searcher = LuceneSearcher(sys.argv[1])
print(json.dumps({"num_docs": searcher.num_docs}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(index)],
        capture_output=True,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip()
        return None, error[-2000:] if error else f"Lucene load failed with exit code {result.returncode}"
    for line in reversed(result.stdout.splitlines()):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "num_docs" in data:
            return int(data["num_docs"]), None
    return None, f"Could not parse Lucene num_docs from stdout: {result.stdout[-2000:]}"


def _parse_bm25_lucene_log(log_file: Path | None) -> dict[str, int | bool]:
    if log_file is None or not log_file.exists():
        return {}

    result: dict[str, int | bool] = {}
    indexed_pattern = re.compile(r"Indexing Complete!\s+([0-9,]+) documents indexed")
    counter_pattern = re.compile(r"-\s+(indexed|empty|errors|skipped|unindexable):\s+([0-9,]+)")
    exit_pattern = re.compile(r"exit_code:\s+(-?\d+)")
    with log_file.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            indexed_match = indexed_pattern.search(line)
            if indexed_match:
                result["log_indexing_complete"] = True
                result["log_indexed_docs"] = int(indexed_match.group(1).replace(",", ""))

            counter_match = counter_pattern.search(line)
            if counter_match:
                key = counter_match.group(1)
                value = int(counter_match.group(2).replace(",", ""))
                if key == "indexed":
                    result["log_indexed_docs"] = value
                elif key == "empty":
                    result["log_empty_docs"] = value

            exit_match = exit_pattern.search(line)
            if exit_match:
                result["log_exit_code"] = int(exit_match.group(1))
    return result


def build_ance_faiss_shards(
    *,
    collection_file: Path | str,
    output_dir: Path | str,
    encoder_name: str = DEFAULT_ANCE_ENCODER,
    batch_size: int = 32,
    batches_per_shard: int = 100,
    max_length: int = 384,
    dimension: int = 768,
    device: str = "cpu",
    force: bool = False,
    progress: bool = True,
    start_passage: int = 0,
    end_passage: int | None = None,
    output_shard_offset: int | None = None,
    text_field: str = "text",
    id_field: str = "id",
    dry_run: bool = False,
) -> IndexBuildResult:
    """Build local float32 IndexFlatIP shards, publishing each shard atomically.

    Uses the same exact ANCE checkpoint loader as retrieval, with the original
    document encoder's ID-derived padding mask and passage length 384.
    Existing complete, matching shards are
    retained; an interrupted temporary shard is never mistaken for a complete one.
    """
    import hashlib
    import itertools
    import tempfile

    collection = Path(collection_file).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not collection.is_file():
        raise FileNotFoundError(collection)
    if min(batch_size, batches_per_shard, max_length, dimension) < 1 or start_passage < 0:
        raise ValueError("Positive encoder/shard sizes and nonnegative start required.")
    per_shard = batch_size * batches_per_shard
    if start_passage % per_shard:
        raise ValueError("start_passage must align with batch_size * batches_per_shard.")
    first_shard = start_passage // per_shard
    if output_shard_offset is not None and output_shard_offset != first_shard:
        raise ValueError("Shard offset must preserve the global passage ordering.")
    if end_passage is not None and end_passage <= start_passage:
        raise ValueError("end_passage must exceed start_passage.")
    if dry_run:
        return IndexBuildResult(index_dir=output, command=None, built=False)

    import faiss
    import numpy as np
    import torch
    from transformers import RobertaTokenizer
    from .dense_ance import load_ance_encoder_exact

    if end_passage is None:
        with collection.open(encoding="utf-8") as handle:
            end_passage = sum(1 for _ in handle)
    if force and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    encoder = None
    tokenizer = None
    built = 0
    with collection.open(encoding="utf-8") as rows:
        if sum(1 for _ in itertools.islice(rows, start_passage)) != start_passage:
            raise ValueError("Collection ends before start_passage.")
        for passage_start in range(start_passage, end_passage, per_shard):
            passage_end = min(passage_start + per_shard, end_passage)
            count = passage_end - passage_start
            shard_id = passage_start // per_shard
            shard = output / f"shard_{shard_id:06d}"
            expected = dict(complete=True, shard_id=shard_id,
                passage_start=passage_start, passage_end=passage_end,
                num_passages=count, dimension=dimension, encoder=encoder_name,
                max_length=max_length, batch_size=batch_size,
                text_field=text_field, id_field=id_field,
                attention_mask="input_ids_ne_pad_token_id")
            source_digest = hashlib.sha256()
            if shard.exists():
                try:
                    manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
                    with (shard / "docid").open(encoding="utf-8") as handle:
                        docids = sum(1 for _ in handle)
                    valid = (all(manifest.get(k) == v for k, v in expected.items())
                        and isinstance(manifest.get("source_sha256"), str)
                        and (shard / "index").stat().st_size == count * dimension * 4 + 45
                        and docids == count)
                except (OSError, ValueError):
                    valid = False
                if not valid:
                    raise RuntimeError(f"Existing shard is incomplete or incompatible: {shard}. "
                                       "Use a fresh output directory; never merge incompatible builds.")
                resumed_rows = 0
                for line in itertools.islice(rows, count):
                    source_digest.update(line.encode("utf-8"))
                    resumed_rows += 1
                if resumed_rows != count:
                    raise ValueError("Collection is shorter than the requested range.")
                if source_digest.hexdigest() != manifest["source_sha256"]:
                    raise RuntimeError(f"Collection content/order changed for {shard}; "
                                       "use a fresh output directory.")
                continue
            if encoder is None:
                encoder = load_ance_encoder_exact(encoder_name).to(device).eval()
                tokenizer = RobertaTokenizer.from_pretrained(
                    encoder_name, clean_up_tokenization_spaces=True)
            with tempfile.TemporaryDirectory(prefix=f".{shard.name}-", dir=output) as temp:
                staging = Path(temp)
                index = faiss.IndexFlatIP(dimension)
                batches = 0
                with (staging / "docid").open("w", encoding="utf-8") as ids:
                    remaining = count
                    while remaining:
                        batch = list(itertools.islice(rows, min(batch_size, remaining)))
                        if not batch:
                            raise ValueError("Collection is shorter than the requested range.")
                        for line in batch:
                            source_digest.update(line.encode("utf-8"))
                        records = [json.loads(line) for line in batch]
                        texts = [str(row[text_field]) for row in records]
                        inputs = tokenizer(texts, max_length=max_length, padding="longest",
                            truncation=True, add_special_tokens=True, return_tensors="pt")
                        # Original AnceDocumentEncoder calls model(input_ids), letting
                        # ANCE mask every pad ID, including literal <pad> in source text.
                        with torch.inference_mode():
                            vectors = encoder(inputs["input_ids"].to(device)).detach().cpu().numpy()
                        vectors = np.ascontiguousarray(vectors, dtype="float32")
                        if vectors.shape != (len(records), dimension) or not np.isfinite(vectors).all():
                            raise ValueError("Invalid ANCE embedding batch.")
                        index.add(vectors)
                        ids.writelines(str(row[id_field]) + "\n" for row in records)
                        remaining -= len(records)
                        batches += 1
                faiss.write_index(index, str(staging / "index"))
                manifest = {**expected, "source_sha256": source_digest.hexdigest(),
                            "num_batches": batches, "written_at_unix": time.time()}
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                staging.rename(shard)
            built += 1
            if progress:
                print(f"complete {shard.name}: {count} passages", flush=True)
    return IndexBuildResult(index_dir=output, command=None, built=bool(built))


def preview_jsonl_collection(path: Path | str, *, n: int = 5) -> pd.DataFrame:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for _, line in zip(range(n), handle, strict=False):
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def count_jsonl_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)
