"""Gold-BM25 evidence preparation for controlled ROCC post-training."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from .candidates import is_lexical_stopword
from .history_selector import (
    EncodedSelectorDataset,
    EncodedSelectorExample,
    encode_history_pair,
)
from .progress import get_tqdm
from .retrievers import BM25_B, BM25_K1, configure_java21
from .selector_experiment import SelectorExperimentConfig


POST_TRAINING_SCORE_SCHEMA_VERSION = 1


class _GoldBM25ScoringConfig(Protocol):
    """Structural configuration required by Gold-BM25 materialization."""

    k1: float
    b: float

    @property
    def score_path(self) -> Path: ...


@dataclass(frozen=True)
class GoldBM25TokenScoreResult:
    path: Path
    manifest_path: Path
    identity: Mapping[str, Any]
    stats: Mapping[str, Any]
    dataset_sha256: str
    reused: bool


@dataclass(frozen=True)
class GoldScoreSelectorExample:
    selector: EncodedSelectorExample
    gold_scores: torch.Tensor


class GoldScoreSelectorDataset(Dataset[GoldScoreSelectorExample]):
    """Collapsed selector examples plus one BM25 score per model token."""

    def __init__(
        self,
        examples: Sequence[GoldScoreSelectorExample],
        *,
        stats: Mapping[str, Any],
    ) -> None:
        self.examples = list(examples)
        self.stats = dict(stats)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> GoldScoreSelectorExample:
        return self.examples[index]


class _GoldPassageBM25Scorer:
    def __init__(
        self,
        index_dir: Path,
        *,
        k1: float,
        b: float,
        document_cache_size: int = 512,
        analyzer_cache_size: int = 250_000,
    ) -> None:
        configure_java21()
        from pyserini.index.lucene import LuceneIndexReader

        self.reader = LuceneIndexReader(str(index_dir.resolve()))
        self.k1 = float(k1)
        self.b = float(b)
        self.document_cache_size = int(document_cache_size)
        self.analyzer_cache_size = int(analyzer_cache_size)
        self.documents: OrderedDict[
            str,
            tuple[dict[str, int], dict[str, float]],
        ] = OrderedDict()
        self.analyzed: OrderedDict[str, tuple[str, ...]] = OrderedDict()

    def analyze(self, token: str) -> tuple[str, ...]:
        key = str(token)
        cached = self.analyzed.get(key)
        if cached is not None:
            self.analyzed.move_to_end(key)
            return cached
        value = tuple(str(term) for term in self.reader.analyze(key))
        self.analyzed[key] = value
        if len(self.analyzed) > self.analyzer_cache_size:
            self.analyzed.popitem(last=False)
        return value

    def token_weight(self, docid: str, token: str) -> float:
        vector, weights = self._document(docid)
        total = 0.0
        for term in self.analyze(token):
            if term not in vector:
                continue
            if term not in weights:
                weights[term] = float(
                    self.reader.compute_bm25_term_weight(
                        str(docid),
                        term,
                        analyzer=None,
                        k1=self.k1,
                        b=self.b,
                    )
                )
            total += weights[term]
        return float(total)

    def _document(
        self,
        docid: str,
    ) -> tuple[dict[str, int], dict[str, float]]:
        key = str(docid)
        cached = self.documents.get(key)
        if cached is not None:
            self.documents.move_to_end(key)
            return cached
        vector = self.reader.get_document_vector(key)
        if vector is None:
            raise KeyError(f"Gold passage is absent from BM25 index: {key}")
        value = (
            {str(term): int(tf) for term, tf in vector.items()},
            {},
        )
        self.documents[key] = value
        if len(self.documents) > self.document_cache_size:
            self.documents.popitem(last=False)
        return value


def materialize_gold_bm25_token_scores(
    *,
    rows: Sequence[dict[str, Any]],
    gold_by_sample: Mapping[str, Sequence[Any]],
    positive_contexts_by_sample: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
    selector_config: SelectorExperimentConfig,
    config: _GoldBM25ScoringConfig,
    progress: bool = True,
) -> GoldBM25TokenScoreResult:
    """Create or resume the durable train-wide Gold-BM25 score JSONL."""

    if len(rows) != int(selector_config.expected_train_rows):
        raise ValueError(
            "Gold-BM25 scoring requires the complete labeled train set: "
            f"{len(rows)}/{selector_config.expected_train_rows}."
        )
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Gold-score rows contain duplicate sample IDs.")
    sample_id_sha256 = _sample_id_sha256(sample_ids)
    index_signature = _bm25_index_signature(
        selector_config.bm25_index_dir,
        k1=config.k1,
        b=config.b,
    )
    identity = {
        "schema_version": POST_TRAINING_SCORE_SCHEMA_VERSION,
        "dataset_sha256": selector_config.expected_train_sha256,
        "sample_id_sha256": sample_id_sha256,
        "rows": len(sample_ids),
        "bm25_index_signature": index_signature,
        "k1": config.k1,
        "b": config.b,
        "source_tokenization": r"regex_\S+_v1",
        "analyzer": "pyserini_default_lucene_analyzer",
        "multi_term_reduction": "sum",
        "normalization": "per_query_positive_max_v1",
    }
    path = config.score_path
    config_path = path.with_suffix(".config.json")
    manifest_path = path.with_suffix(".manifest.json")
    _ensure_locked_mapping(config_path, identity)

    if path.exists() and manifest_path.exists():
        manifest = _read_json(manifest_path)
        dataset_sha256 = _sha256_file(path, progress=progress)
        if manifest.get("complete") is True:
            if (
                manifest.get("identity") != identity
                or int(manifest.get("rows", -1)) != len(sample_ids)
                or manifest.get("dataset_sha256") != dataset_sha256
            ):
                raise RuntimeError(
                    "Completed Gold-BM25 score artifact failed its "
                    "manifest check."
                )
            return GoldBM25TokenScoreResult(
                path=path,
                manifest_path=manifest_path,
                identity=identity,
                stats=dict(manifest["stats"]),
                dataset_sha256=dataset_sha256,
                reused=True,
            )

    persisted = _scan_score_jsonl(
        path,
        expected_ids=set(sample_ids),
        repair_trailing=True,
        progress=progress,
    )
    missing = [sample_id for sample_id in sample_ids if sample_id not in persisted]
    if missing:
        scorer = _GoldPassageBM25Scorer(
            selector_config.bm25_index_dir,
            k1=config.k1,
            b=config.b,
        )
        row_by_id = {str(row["sample_id"]): row for row in rows}
        tqdm = get_tqdm()
        for sample_id in tqdm(
            missing,
            total=len(missing),
            desc="Gold-BM25 history-token scores",
            unit="query",
            dynamic_ncols=True,
            disable=not progress,
        ):
            gold_ids = tuple(
                dict.fromkeys(
                    str(value) for value in gold_by_sample[sample_id]
                )
            )
            if len(gold_ids) != 1:
                raise ValueError(
                    f"Expected exactly one gold passage for {sample_id}."
                )
            gold_id = gold_ids[0]
            contexts = [
                context
                for context in positive_contexts_by_sample[sample_id]
                if str(context.get("passage_id")) == gold_id
            ]
            if len(contexts) != 1:
                raise ValueError(
                    f"Gold context does not resolve uniquely for {sample_id}."
                )
            scored_tokens: list[dict[str, Any]] = []
            for token in _history_whitespace_tokens(row_by_id[sample_id]):
                raw_score = scorer.token_weight(gold_id, token["text"])
                scored_tokens.append(
                    {
                        **token,
                        "score_raw": raw_score,
                    }
                )
            maximum = max(
                (float(token["score_raw"]) for token in scored_tokens),
                default=0.0,
            )
            for token in scored_tokens:
                raw_score = float(token["score_raw"])
                token["score_norm"] = (
                    raw_score / maximum if maximum > 0.0 else 0.0
                )
            context_text = str(contexts[0].get("text") or "")
            record = {
                "sample_id": sample_id,
                "gold_passage_id": gold_id,
                "gold_passage_text_sha256": hashlib.sha256(
                    context_text.encode("utf-8")
                ).hexdigest(),
                "tokens": scored_tokens,
            }
            _append_jsonl_durable(path, record)

    final_ids = _scan_score_jsonl(
        path,
        expected_ids=set(sample_ids),
        repair_trailing=False,
        progress=progress,
    )
    if final_ids != set(sample_ids):
        missing_final = sorted(set(sample_ids).difference(final_ids))
        raise RuntimeError(
            "Gold-BM25 score JSONL is incomplete: "
            f"{len(final_ids)}/{len(sample_ids)}, missing {missing_final[:3]}"
        )
    stats = _gold_score_statistics(path, progress=progress)
    dataset_sha256 = _sha256_file(path, progress=progress)
    manifest = {
        "identity": identity,
        "rows": len(sample_ids),
        "dataset_sha256": dataset_sha256,
        "stats": stats,
        "complete": True,
    }
    _write_json(manifest_path, manifest)
    return GoldBM25TokenScoreResult(
        path=path,
        manifest_path=manifest_path,
        identity=identity,
        stats=stats,
        dataset_sha256=dataset_sha256,
        reused=False,
    )


def materialize_gold_bm25_token_scores_for_rows(
    *,
    rows: Sequence[dict[str, Any]],
    gold_by_sample: Mapping[str, Sequence[Any]],
    positive_contexts_by_sample: Mapping[
        str,
        Sequence[Mapping[str, Any]],
    ],
    bm25_index_dir: Path | str,
    output_path: Path | str,
    source_dataset_sha256: str,
    k1: float = BM25_K1,
    b: float = BM25_B,
    progress: bool = True,
) -> GoldBM25TokenScoreResult:
    """Materialize Gold-BM25 token scores for an explicit row cohort.

    This is the cohort-generic adapter for held-out diagnostics.  The
    canonical train-wide function above remains unchanged.
    """

    sample_ids = [str(row["sample_id"]) for row in rows]
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Gold-score cohort must have unique sample IDs.")
    expected_ids = set(sample_ids)
    if set(map(str, gold_by_sample)) != expected_ids:
        raise ValueError("Gold passages do not match the score cohort.")
    if set(map(str, positive_contexts_by_sample)) != expected_ids:
        raise ValueError("Positive contexts do not match the score cohort.")

    index_dir = Path(bm25_index_dir)
    path = Path(output_path)
    identity = {
        "schema_version": POST_TRAINING_SCORE_SCHEMA_VERSION,
        "dataset_sha256": str(source_dataset_sha256),
        "sample_id_sha256": _sample_id_sha256(sample_ids),
        "rows": len(sample_ids),
        "bm25_index_signature": _bm25_index_signature(
            index_dir,
            k1=float(k1),
            b=float(b),
        ),
        "k1": float(k1),
        "b": float(b),
        "source_tokenization": r"regex_\S+_v1",
        "analyzer": "pyserini_default_lucene_analyzer",
        "multi_term_reduction": "sum",
        "normalization": "per_query_positive_max_v1",
    }
    config_path = path.with_suffix(".config.json")
    manifest_path = path.with_suffix(".manifest.json")
    _ensure_locked_mapping(config_path, identity)

    if path.exists() and manifest_path.exists():
        manifest = _read_json(manifest_path)
        dataset_sha256 = _sha256_file(path, progress=progress)
        if manifest.get("complete") is True:
            if (
                manifest.get("identity") != identity
                or int(manifest.get("rows", -1)) != len(sample_ids)
                or manifest.get("dataset_sha256") != dataset_sha256
            ):
                raise RuntimeError(
                    "Completed cohort Gold-BM25 artifact failed validation."
                )
            return GoldBM25TokenScoreResult(
                path=path,
                manifest_path=manifest_path,
                identity=identity,
                stats=dict(manifest["stats"]),
                dataset_sha256=dataset_sha256,
                reused=True,
            )

    persisted = _scan_score_jsonl(
        path,
        expected_ids=expected_ids,
        repair_trailing=True,
        progress=progress,
    )
    missing = [sample_id for sample_id in sample_ids if sample_id not in persisted]
    if missing:
        scorer = _GoldPassageBM25Scorer(
            index_dir,
            k1=float(k1),
            b=float(b),
        )
        row_by_id = {str(row["sample_id"]): row for row in rows}
        tqdm = get_tqdm()
        for sample_id in tqdm(
            missing,
            total=len(missing),
            desc="Dev Gold-BM25 history-token scores",
            unit="query",
            dynamic_ncols=True,
            disable=not progress,
        ):
            gold_ids = tuple(
                dict.fromkeys(str(value) for value in gold_by_sample[sample_id])
            )
            if len(gold_ids) != 1:
                raise ValueError(
                    f"Expected one gold passage for {sample_id}."
                )
            gold_id = gold_ids[0]
            contexts = [
                context
                for context in positive_contexts_by_sample[sample_id]
                if str(context.get("passage_id")) == gold_id
            ]
            if len(contexts) != 1:
                raise ValueError(
                    f"Gold context is not unique for {sample_id}."
                )
            tokens = []
            for token in _history_whitespace_tokens(row_by_id[sample_id]):
                tokens.append(
                    {
                        **token,
                        "score_raw": scorer.token_weight(
                            gold_id,
                            str(token["text"]),
                        ),
                    }
                )
            maximum = max(
                (float(token["score_raw"]) for token in tokens),
                default=0.0,
            )
            for token in tokens:
                raw = float(token["score_raw"])
                token["score_norm"] = raw / maximum if maximum > 0.0 else 0.0
            context_text = str(contexts[0].get("text") or "")
            _append_jsonl_durable(
                path,
                {
                    "sample_id": sample_id,
                    "gold_passage_id": gold_id,
                    "gold_passage_text_sha256": hashlib.sha256(
                        context_text.encode("utf-8")
                    ).hexdigest(),
                    "tokens": tokens,
                },
            )

    final_ids = _scan_score_jsonl(
        path,
        expected_ids=expected_ids,
        repair_trailing=False,
        progress=progress,
    )
    if final_ids != expected_ids:
        raise RuntimeError(
            "Cohort Gold-BM25 score JSONL is incomplete: "
            f"{len(final_ids)}/{len(expected_ids)}."
        )
    stats = _gold_score_statistics(path, progress=progress)
    dataset_sha256 = _sha256_file(path, progress=progress)
    manifest = {
        "identity": identity,
        "rows": len(sample_ids),
        "dataset_sha256": dataset_sha256,
        "stats": stats,
        "complete": True,
    }
    _write_json(manifest_path, manifest)
    return GoldBM25TokenScoreResult(
        path=path,
        manifest_path=manifest_path,
        identity=identity,
        stats=stats,
        dataset_sha256=dataset_sha256,
        reused=False,
    )


def prepare_gold_score_selector_dataset(
    *,
    rows: Sequence[dict[str, Any]],
    base_dataset: EncodedSelectorDataset,
    tokenizer: Any,
    score_result: GoldBM25TokenScoreResult,
    selector_config: SelectorExperimentConfig,
    exclude_stopwords: bool = False,
    require_complete_train: bool = True,
    progress: bool = True,
) -> GoldScoreSelectorDataset:
    """Project persisted whitespace-token scores to MiniLM WordPieces."""

    if require_complete_train and len(rows) != int(
        selector_config.expected_train_rows
    ):
        raise ValueError(
            "Gold-score projection requires the complete train set."
        )
    row_by_id = {str(row["sample_id"]): row for row in rows}
    base_by_id = {
        str(example.sample_id): example
        for example in base_dataset.examples
    }
    if set(row_by_id) != set(base_by_id):
        raise ValueError("Rows and collapsed training examples differ.")
    score_by_id: dict[str, torch.Tensor] = {}
    source_positive_tokens = 0
    retained_positive_source_tokens = 0
    excluded_stopword_source_tokens = 0
    visible_positive_wordpieces = 0
    total_history_wordpieces = 0
    tqdm = get_tqdm()
    iterator = _iter_jsonl(score_result.path)
    for record in tqdm(
        iterator,
        total=len(row_by_id),
        desc="project Gold-BM25 scores to WordPieces",
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    ):
        sample_id = str(record["sample_id"])
        if sample_id not in row_by_id or sample_id in score_by_id:
            raise ValueError(f"Unexpected or duplicate score row: {sample_id}")
        pair = encode_history_pair(
            row_by_id[sample_id],
            tokenizer,
            max_length=selector_config.max_length,
            history_order=selector_config.history_order,
        )
        base = base_by_id[sample_id]
        if (
            pair.input_ids != base.input_ids
            or pair.attention_mask != base.attention_mask
            or pair.history_mask != base.history_mask
        ):
            raise RuntimeError(
                f"Cached selector encoding drifted for {sample_id}."
            )
        spans: list[tuple[int, int, float]] = []
        for token in record.get("tokens", []):
            score = float(token.get("score_norm", 0.0))
            if score <= 0.0:
                continue
            source_positive_tokens += 1
            if exclude_stopwords and is_lexical_stopword(token["text"]):
                excluded_stopword_source_tokens += 1
                continue
            retained_positive_source_tokens += 1
            key = (int(token["turn_id"]), str(token["field"]))
            if key not in pair.field_offsets:
                raise KeyError(f"Unknown history field in {sample_id}: {key}")
            start = pair.field_offsets[key] + int(token["start"])
            end = pair.field_offsets[key] + int(token["end"])
            spans.append((start, end, score))
        spans.sort(key=lambda item: (item[0], item[1]))

        projected = torch.zeros(len(pair.input_ids), dtype=torch.float32)
        span_index = 0
        for token_index, (offset, sequence_id) in enumerate(
            zip(pair.offset_mapping, pair.sequence_ids, strict=True)
        ):
            if sequence_id != 1 or offset == (0, 0):
                continue
            total_history_wordpieces += 1
            token_start, token_end = map(int, offset)
            while (
                span_index < len(spans)
                and spans[span_index][1] <= token_start
            ):
                span_index += 1
            candidate_index = span_index
            value = 0.0
            while (
                candidate_index < len(spans)
                and spans[candidate_index][0] < token_end
            ):
                span_start, span_end, score = spans[candidate_index]
                if token_start < span_end and token_end > span_start:
                    value = max(value, score)
                candidate_index += 1
            if value > 0.0:
                projected[token_index] = value
                visible_positive_wordpieces += 1
        score_by_id[sample_id] = projected

    if set(score_by_id) != set(base_by_id):
        raise RuntimeError("Projected Gold-BM25 scores are incomplete.")
    examples = [
        GoldScoreSelectorExample(
            selector=example,
            gold_scores=score_by_id[str(example.sample_id)],
        )
        for example in base_dataset.examples
    ]
    stats = {
        "queries": len(examples),
        "history_wordpieces": total_history_wordpieces,
        "positive_source_tokens": source_positive_tokens,
        "retained_positive_source_tokens": retained_positive_source_tokens,
        "excluded_stopword_source_tokens": (
            excluded_stopword_source_tokens
        ),
        "stopword_filter": (
            "embedded_nltk_english" if exclude_stopwords else "none"
        ),
        "positive_visible_wordpieces": visible_positive_wordpieces,
        "positive_visible_wordpiece_share": (
            visible_positive_wordpieces / total_history_wordpieces
            if total_history_wordpieces
            else 0.0
        ),
        "queries_without_visible_signal": sum(
            int(example.gold_scores.max().item() <= 0.0)
            for example in examples
        ),
    }
    return GoldScoreSelectorDataset(examples, stats=stats)


def _history_whitespace_tokens(
    row: Mapping[str, Any],
) -> Iterable[dict[str, Any]]:
    for turn in row.get("history", []):
        turn_id = int(turn["turn_id"])
        for field in ("question", "answer"):
            text = str(turn.get(field, ""))
            for match in re.finditer(r"\S+", text):
                start, end = match.span()
                yield {
                    "turn_id": turn_id,
                    "field": field,
                    "start": start,
                    "end": end,
                    "text": match.group(0),
                }


def _bm25_index_signature(
    index_dir: Path,
    *,
    k1: float,
    b: float,
) -> str:
    resolved = Path(index_dir).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"BM25 index is missing: {resolved}")
    files = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(resolved.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "write.lock"
    ]
    return _mapping_sha256(
        {
            "index_dir": str(resolved),
            "files": files,
            "k1": float(k1),
            "b": float(b),
            "pyserini_version": importlib.metadata.version("pyserini"),
        }
    )


def _gold_score_statistics(
    path: Path,
    *,
    progress: bool,
) -> dict[str, Any]:
    total_tokens = 0
    positive_tokens = 0
    queries = 0
    queries_with_positive = 0
    positive_values: list[float] = []
    tqdm = get_tqdm()
    with path.open("rb") as handle, tqdm(
        total=path.stat().st_size,
        desc=f"summarize {path.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        disable=not progress,
    ) as bar:
        for line in handle:
            bar.update(len(line))
            if not line.strip():
                continue
            record = json.loads(line)
            values = [
                float(token.get("score_norm", 0.0))
                for token in record.get("tokens", [])
            ]
            positives = [value for value in values if value > 0.0]
            queries += 1
            total_tokens += len(values)
            positive_tokens += len(positives)
            queries_with_positive += int(bool(positives))
            positive_values.extend(positives)
    array = np.asarray(positive_values, dtype=np.float64)
    return {
        "queries": queries,
        "history_tokens": total_tokens,
        "positive_history_tokens": positive_tokens,
        "positive_history_token_share": (
            positive_tokens / total_tokens if total_tokens else 0.0
        ),
        "queries_with_positive_signal": queries_with_positive,
        "queries_without_positive_signal": queries - queries_with_positive,
        "positive_score_mean": float(array.mean()) if array.size else 0.0,
        "positive_score_median": float(np.median(array)) if array.size else 0.0,
        "positive_score_p90": float(np.quantile(array, 0.9)) if array.size else 0.0,
        "positive_score_max": float(array.max()) if array.size else 0.0,
    }


def _scan_score_jsonl(
    path: Path,
    *,
    expected_ids: set[str],
    repair_trailing: bool,
    progress: bool,
) -> set[str]:
    if not path.exists():
        return set()
    found: set[str] = set()
    file_size = path.stat().st_size
    valid_end = 0
    tqdm = get_tqdm()
    with path.open("rb") as handle, tqdm(
        total=file_size,
        desc=f"load {path.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        dynamic_ncols=True,
        disable=not progress,
    ) as bar:
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                valid_end = handle.tell()
                break
            bar.update(len(line))
            line_end = handle.tell()
            if not line.endswith(b"\n"):
                break
            try:
                record = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if line_end < file_size or not repair_trailing:
                    raise RuntimeError(
                        f"Corrupt JSONL line at byte {line_start}: {path}"
                    ) from exc
                break
            sample_id = str(record.get("sample_id", ""))
            if sample_id not in expected_ids or sample_id in found:
                raise ValueError(
                    f"Unexpected or duplicate score sample: {sample_id}"
                )
            found.add(sample_id)
            _validate_score_record(record, sample_id=sample_id)
            valid_end = line_end
    if valid_end < file_size:
        if not repair_trailing:
            raise RuntimeError(f"Incomplete JSONL tail: {path}")
        with path.open("r+b") as handle:
            handle.truncate(valid_end)
            handle.flush()
            os.fsync(handle.fileno())
    return found


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _validate_score_record(
    record: Mapping[str, Any],
    *,
    sample_id: str,
) -> None:
    tokens = list(record.get("tokens", []))
    raw = np.asarray(
        [float(token.get("score_raw", 0.0)) for token in tokens],
        dtype=np.float64,
    )
    normalized = np.asarray(
        [float(token.get("score_norm", 0.0)) for token in tokens],
        dtype=np.float64,
    )
    if (
        np.any(~np.isfinite(raw))
        or np.any(~np.isfinite(normalized))
        or np.any(raw < 0.0)
        or np.any(normalized < 0.0)
        or np.any(normalized > 1.0 + 1e-7)
        or not np.array_equal(raw > 0.0, normalized > 0.0)
    ):
        raise ValueError(f"Invalid Gold-BM25 scores for {sample_id}.")
    maximum = float(raw.max(initial=0.0))
    expected = raw / maximum if maximum > 0.0 else np.zeros_like(raw)
    if not np.allclose(
        normalized,
        expected,
        atol=1e-7,
        rtol=1e-7,
    ):
        raise ValueError(
            f"Gold-BM25 normalization drifted for {sample_id}."
        )
    if maximum > 0.0:
        if not np.isclose(normalized.max(), 1.0, atol=1e-7, rtol=0.0):
            raise ValueError(
                f"Gold-BM25 max normalization failed for {sample_id}."
            )


def _append_jsonl_durable(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o600,
    )
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(
                f"Incomplete JSONL write: {written}/{len(payload)} bytes"
            )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sample_id_sha256(sample_ids: Iterable[str]) -> str:
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


def _sha256_file(path: Path, *, progress: bool = False) -> str:
    digest = hashlib.sha256()
    tqdm = get_tqdm()
    with path.open("rb") as handle, tqdm(
        total=path.stat().st_size,
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _ensure_locked_mapping(path: Path, value: Mapping[str, Any]) -> None:
    expected = json.loads(json.dumps(dict(value), ensure_ascii=False))
    if path.exists():
        if _read_json(path) != expected:
            raise RuntimeError(f"Locked configuration differs: {path}")
        return
    _write_json(path, expected)


__all__ = [
    "GoldBM25TokenScoreResult",
    "GoldScoreSelectorDataset",
    "GoldScoreSelectorExample",
    "materialize_gold_bm25_token_scores",
    "materialize_gold_bm25_token_scores_for_rows",
    "prepare_gold_score_selector_dataset",
]
