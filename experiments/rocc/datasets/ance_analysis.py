"""ANCE passage-tokenization diagnostics."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, as_completed, wait
import csv
import json
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import pandas as pd

from ..progress import get_tqdm
from .qrecc import QRECC_TOTAL_PASSAGES, resolve_qrecc_resources
from .topiocqa import TOPIOCQA_CORPUS_PASSAGES, resolve_topiocqa_resources


AnceAnalysisDatasetName = Literal["topiocqa", "qrecc"]
DEFAULT_ANCE_ENCODER = "castorini/ance-msmarco-passage"
_THREAD_LOCAL = threading.local()


@dataclass
class AncePassageEncodingAnalysis:
    corpus_summary: pd.DataFrame
    token_length_summary: pd.DataFrame
    unk_summary: pd.DataFrame
    truncation_summary: pd.DataFrame
    padding_summary: pd.DataFrame
    run_metadata: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "corpus_summary": self.corpus_summary,
            "token_length_summary": self.token_length_summary,
            "unk_summary": self.unk_summary,
            "truncation_summary": self.truncation_summary,
            "padding_summary": self.padding_summary,
        }


@dataclass
class _RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: int | None = None
    max_value: int | None = None

    def update(self, value: int) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    def merge(self, other: "_RunningStats") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean
            self.m2 = other.m2
            self.min_value = other.min_value
            self.max_value = other.max_value
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.mean += delta * other.count / total
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.count = total
        if other.min_value is not None:
            self.min_value = (
                other.min_value
                if self.min_value is None
                else min(self.min_value, other.min_value)
            )
        if other.max_value is not None:
            self.max_value = (
                other.max_value
                if self.max_value is None
                else max(self.max_value, other.max_value)
            )

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return self.m2 / (self.count - 1)

    def to_summary(self) -> dict[str, float | int | None]:
        variance = self.variance
        ci_delta = 1.96 * math.sqrt(variance / self.count) if self.count else 0.0
        return {
            "mean": self.mean,
            "variance": variance,
            "ci95_low": self.mean - ci_delta if self.count else None,
            "ci95_high": self.mean + ci_delta if self.count else None,
            "min": self.min_value,
            "max": self.max_value,
        }


def analyze_ance_passage_encoding(
    dataset_name: AnceAnalysisDatasetName,
    *,
    data_dir: Path | str | None = None,
    collection_file: Path | str | None = None,
    batch_size: int = 160,
    max_length: int = 384,
    encoder_name: str = DEFAULT_ANCE_ENCODER,
    include_title: bool = False,
    max_passages: int | None = None,
    workers: int = 1,
    chunk_batches: int = 64,
    progress: bool = True,
) -> AncePassageEncodingAnalysis:
    """Analyze ANCE passage-tokenization length, truncation, and padding overhead."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    if chunk_batches <= 0:
        raise ValueError("chunk_batches must be positive.")

    resolved_dataset_name = _normalize_dataset_name(dataset_name)
    if workers > 1:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    source_file, expected_passages, rows = _resolve_passage_rows(
        resolved_dataset_name,
        data_dir=data_dir,
        collection_file=collection_file,
        include_title=include_title,
        max_passages=max_passages,
    )
    if not source_file.exists():
        raise FileNotFoundError(f"ANCE passage source is missing: {source_file}")

    # Load once in the parent process to fail fast if the tokenizer is missing.
    _load_ance_tokenizer(encoder_name)

    target_total = expected_passages
    if max_passages is not None:
        target_total = min(expected_passages, max_passages)

    bar = get_tqdm()(
        total=target_total,
        desc=f"analyze {resolved_dataset_name} ANCE passages",
        unit="passage",
        disable=not progress,
    )

    token_stats = _RunningStats()
    padded_length_stats = _RunningStats()
    actual_visible_tokens_total = 0
    padded_tokens_total = 0
    raw_token_total = 0
    unk_tokens = 0
    passages_with_unk = 0
    truncated_passages = 0
    truncated_tokens_total = 0

    def accumulate(chunk_result: dict[str, Any]) -> None:
        nonlocal actual_visible_tokens_total
        nonlocal padded_tokens_total
        nonlocal raw_token_total
        nonlocal unk_tokens
        nonlocal passages_with_unk
        nonlocal truncated_passages
        nonlocal truncated_tokens_total
        token_stats.merge(chunk_result["token_stats"])
        padded_length_stats.merge(chunk_result["padded_length_stats"])
        actual_visible_tokens_total += chunk_result["actual_visible_tokens"]
        padded_tokens_total += chunk_result["padded_tokens"]
        raw_token_total += chunk_result["raw_tokens"]
        unk_tokens += chunk_result["unk_tokens"]
        passages_with_unk += chunk_result["passages_with_unk"]
        truncated_passages += chunk_result["truncated_passages"]
        truncated_tokens_total += chunk_result["truncated_tokens"]
        bar.update(chunk_result["num_passages"])

    chunk_size = batch_size * chunk_batches
    try:
        if workers == 1:
            tokenizer = _load_ance_tokenizer(encoder_name)
            for texts in _iter_text_chunks(rows, chunk_size=chunk_size):
                accumulate(
                    _analyze_text_chunk(
                        texts,
                        batch_size=batch_size,
                        max_length=max_length,
                        tokenizer=tokenizer,
                    )
                )
        else:
            max_pending = max(1, workers * 2)
            pending: set[Future[dict[str, Any]]] = set()
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for texts in _iter_text_chunks(rows, chunk_size=chunk_size):
                    pending.add(
                        executor.submit(
                            _analyze_text_chunk,
                            texts,
                            batch_size=batch_size,
                            max_length=max_length,
                            encoder_name=encoder_name,
                        )
                    )
                    if len(pending) >= max_pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            accumulate(future.result())
                for future in as_completed(pending):
                    accumulate(future.result())
    finally:
        bar.close()

    num_passages = token_stats.count
    padding_overhead_tokens_total = padded_tokens_total - actual_visible_tokens_total
    mean_truncated_tokens_all = (
        truncated_tokens_total / num_passages if num_passages else 0.0
    )
    mean_truncated_tokens_truncated_only = (
        truncated_tokens_total / truncated_passages if truncated_passages else 0.0
    )
    padding_stats_summary = padded_length_stats.to_summary()

    return AncePassageEncodingAnalysis(
        corpus_summary=pd.DataFrame(
            [
                {
                    "dataset_name": resolved_dataset_name,
                    "num_passages": num_passages,
                    "expected_total_passages": expected_passages,
                    "max_passages": max_passages,
                    "full_scan": max_passages is None,
                    "batch_size": batch_size,
                    "max_length_setting": max_length,
                    "workers": workers,
                    "chunk_batches": chunk_batches,
                    "encoder_name": encoder_name,
                    "source_file": str(source_file),
                }
            ]
        ),
        token_length_summary=pd.DataFrame([token_stats.to_summary()]),
        unk_summary=pd.DataFrame(
            [
                {
                    "unk_tokens": unk_tokens,
                    "visible_tokens": raw_token_total,
                    "unk_rate": unk_tokens / raw_token_total if raw_token_total else 0.0,
                    "passages_with_unk": passages_with_unk,
                    "passages_with_unk_fraction": (
                        passages_with_unk / num_passages if num_passages else 0.0
                    ),
                }
            ]
        ),
        truncation_summary=pd.DataFrame(
            [
                {
                    "truncated_passages": truncated_passages,
                    "truncation_rate": (
                        truncated_passages / num_passages if num_passages else 0.0
                    ),
                    "truncated_tokens_total": truncated_tokens_total,
                    "mean_truncated_tokens_all": mean_truncated_tokens_all,
                    "mean_truncated_tokens_truncated_only": (
                        mean_truncated_tokens_truncated_only
                    ),
                }
            ]
        ),
        padding_summary=pd.DataFrame(
            [
                {
                    "batch_padded_length_mean": padded_length_stats.mean,
                    "batch_padded_length_variance": padded_length_stats.variance,
                    "batch_padded_length_ci95_low": padding_stats_summary["ci95_low"],
                    "batch_padded_length_ci95_high": padding_stats_summary["ci95_high"],
                    "batch_padded_length_min": padded_length_stats.min_value,
                    "batch_padded_length_max": padded_length_stats.max_value,
                    "actual_visible_tokens_total": actual_visible_tokens_total,
                    "padded_tokens_total": padded_tokens_total,
                    "padding_overhead_tokens_total": padding_overhead_tokens_total,
                    "padding_overhead_fraction": (
                        padding_overhead_tokens_total / padded_tokens_total
                        if padded_tokens_total
                        else 0.0
                    ),
                    "effective_token_utilization": (
                        actual_visible_tokens_total / padded_tokens_total
                        if padded_tokens_total
                        else 0.0
                    ),
                }
            ]
        ),
    )


def save_ance_passage_encoding_analysis(
    analysis: AncePassageEncodingAnalysis,
    output_file: Path | str,
    *,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(analysis.run_metadata or {})
    if run_metadata:
        metadata.update(run_metadata)
    payload = {
        "run_metadata": metadata,
        "corpus_summary": _dataframe_records(analysis.corpus_summary),
        "token_length_summary": _dataframe_records(analysis.token_length_summary),
        "unk_summary": _dataframe_records(analysis.unk_summary),
        "truncation_summary": _dataframe_records(analysis.truncation_summary),
        "padding_summary": _dataframe_records(analysis.padding_summary),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def load_ance_passage_encoding_analysis(
    input_file: Path | str,
) -> AncePassageEncodingAnalysis:
    input_path = Path(input_file).expanduser().resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    return AncePassageEncodingAnalysis(
        corpus_summary=pd.DataFrame(payload.get("corpus_summary", [])),
        token_length_summary=pd.DataFrame(payload.get("token_length_summary", [])),
        unk_summary=pd.DataFrame(payload.get("unk_summary", [])),
        truncation_summary=pd.DataFrame(payload.get("truncation_summary", [])),
        padding_summary=pd.DataFrame(payload.get("padding_summary", [])),
        run_metadata=payload.get("run_metadata", {}),
    )


def _iter_text_chunks(
    rows: Iterable[dict[str, str]],
    *,
    chunk_size: int,
) -> Iterable[list[str]]:
    texts: list[str] = []
    for passage in rows:
        texts.append(passage["text"])
        if len(texts) >= chunk_size:
            yield texts
            texts = []
    if texts:
        yield texts


def _analyze_text_chunk(
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    tokenizer: Any | None = None,
    encoder_name: str | None = None,
) -> dict[str, Any]:
    resolved_tokenizer = tokenizer or _thread_tokenizer(encoder_name or DEFAULT_ANCE_ENCODER)
    chunk_token_stats = _RunningStats()
    chunk_padded_length_stats = _RunningStats()
    counters = {
        "num_passages": 0,
        "actual_visible_tokens": 0,
        "padded_tokens": 0,
        "raw_tokens": 0,
        "unk_tokens": 0,
        "passages_with_unk": 0,
        "truncated_passages": 0,
        "truncated_tokens": 0,
    }
    unk_token_id = resolved_tokenizer.unk_token_id
    for start in range(0, len(texts), batch_size):
        batch_result = _analyze_text_batch(
            resolved_tokenizer,
            texts[start : start + batch_size],
            max_length=max_length,
            unk_token_id=unk_token_id,
        )
        _update_batch_stats(
            chunk_token_stats,
            chunk_padded_length_stats,
            batch_result,
            counters,
        )
    counters["token_stats"] = chunk_token_stats
    counters["padded_length_stats"] = chunk_padded_length_stats
    return counters


def _normalize_dataset_name(dataset_name: str) -> AnceAnalysisDatasetName:
    normalized = dataset_name.lower()
    if normalized not in {"topiocqa", "qrecc"}:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")
    return normalized  # type: ignore[return-value]


def _resolve_passage_rows(
    dataset_name: AnceAnalysisDatasetName,
    *,
    data_dir: Path | str | None,
    collection_file: Path | str | None,
    include_title: bool,
    max_passages: int | None,
) -> tuple[Path, int, Iterable[dict[str, str]]]:
    if collection_file is not None:
        source_file = Path(collection_file).expanduser().resolve()
        if dataset_name == "topiocqa" and source_file.suffix.lower() == ".tsv":
            return source_file, _expected_total(dataset_name), _iter_topiocqa_passages(
                source_file,
                include_title=include_title,
                max_passages=max_passages,
            )
        return source_file, _expected_total(dataset_name), _iter_jsonl_passages(
            source_file,
            max_passages=max_passages,
        )

    if dataset_name == "topiocqa":
        resources = resolve_topiocqa_resources(data_dir)
        return (
            resources.corpus_tsv,
            TOPIOCQA_CORPUS_PASSAGES,
            _iter_topiocqa_passages(
                resources.corpus_tsv,
                include_title=include_title,
                max_passages=max_passages,
            ),
        )

    resources = resolve_qrecc_resources(data_dir)
    return (
        resources.collection_jsonl,
        QRECC_TOTAL_PASSAGES,
        _iter_jsonl_passages(resources.collection_jsonl, max_passages=max_passages),
    )


def _expected_total(dataset_name: AnceAnalysisDatasetName) -> int:
    if dataset_name == "topiocqa":
        return TOPIOCQA_CORPUS_PASSAGES
    return QRECC_TOTAL_PASSAGES


def _iter_topiocqa_passages(
    corpus_tsv: Path,
    *,
    include_title: bool,
    max_passages: int | None,
) -> Iterable[dict[str, str]]:
    with corpus_tsv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for index, row in enumerate(reader):
            if max_passages is not None and index >= max_passages:
                break
            text = str(row["text"])
            if include_title and row.get("title"):
                text = f"{row['title']} {text}"
            yield {"id": str(row["id"]), "text": text}


def _iter_jsonl_passages(
    collection_jsonl: Path,
    *,
    max_passages: int | None,
) -> Iterable[dict[str, str]]:
    with collection_jsonl.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_passages is not None and index >= max_passages:
                break
            row = json.loads(line, strict=False)
            yield {
                "id": str(row["id"]),
                "text": str(row.get("text") or row.get("contents") or ""),
            }


def _flush_batch(batch_lengths: list[int], stats: _RunningStats) -> int:
    padded_length = max(batch_lengths)
    stats.update(padded_length)
    return padded_length * len(batch_lengths)


def _analyze_text_batch(
    tokenizer: Any,
    texts: list[str],
    *,
    max_length: int,
    unk_token_id: int | None,
) -> dict[str, list[int]]:
    tokenized = tokenizer(
        texts,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )
    input_ids = tokenized["input_ids"]
    raw_lengths = [len(token_ids) for token_ids in input_ids]
    effective_lengths = [min(length, max_length) for length in raw_lengths]
    unk_counts = [
        sum(1 for token_id in token_ids if token_id == unk_token_id)
        if unk_token_id is not None
        else 0
        for token_ids in input_ids
    ]
    return {
        "raw_lengths": raw_lengths,
        "effective_lengths": effective_lengths,
        "unk_counts": unk_counts,
    }


def _update_batch_stats(
    token_stats: _RunningStats,
    padded_length_stats: _RunningStats,
    batch_result: dict[str, list[int]],
    counters: dict[str, int],
) -> None:
    raw_lengths = batch_result["raw_lengths"]
    effective_lengths = batch_result["effective_lengths"]
    unk_counts = batch_result["unk_counts"]
    for raw_length in raw_lengths:
        token_stats.update(raw_length)
    padded_tokens = _flush_batch(effective_lengths, padded_length_stats)
    truncated_tokens = sum(
        raw_length - effective_length
        for raw_length, effective_length in zip(raw_lengths, effective_lengths)
    )
    counters["num_passages"] += len(raw_lengths)
    counters["actual_visible_tokens"] += sum(effective_lengths)
    counters["padded_tokens"] += padded_tokens
    counters["raw_tokens"] += sum(raw_lengths)
    counters["unk_tokens"] += sum(unk_counts)
    counters["passages_with_unk"] += sum(1 for count in unk_counts if count)
    counters["truncated_passages"] += sum(
        1
        for raw_length, effective_length in zip(raw_lengths, effective_lengths)
        if raw_length > effective_length
    )
    counters["truncated_tokens"] += truncated_tokens


def _load_ance_tokenizer(encoder_name: str) -> Any:
    try:
        from transformers import RobertaTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("transformers is required for ANCE passage analysis.") from exc
    return RobertaTokenizer.from_pretrained(
        encoder_name,
        clean_up_tokenization_spaces=True,
    )


def _thread_tokenizer(encoder_name: str) -> Any:
    tokenizers = getattr(_THREAD_LOCAL, "tokenizers", None)
    if tokenizers is None:
        tokenizers = {}
        _THREAD_LOCAL.tokenizers = tokenizers
    if encoder_name not in tokenizers:
        tokenizers[encoder_name] = _load_ance_tokenizer(encoder_name)
    return tokenizers[encoder_name]


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.where(pd.notna(frame), None)
    records = clean.to_dict("records")
    return json.loads(json.dumps(records, default=_json_default))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
