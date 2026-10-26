"""Reusable extractive Teacher labeling with durable resume support."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
from collections.abc import Callable, Collection, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from .progress import get_tqdm


@dataclass(frozen=True)
class TeacherProtocolConfig:
    """Notebook-owned Teacher protocol passed to the reusable runtime."""

    model: str
    api_surface: str
    protocol_hash: str
    system_prompt: str
    prompt_builder: Callable[[Any], str]
    output_schema: dict[str, Any]
    allowed_labels: Collection[str]
    seed: int | None
    temperature: float
    max_output_tokens: int
    expected_system_prompt_sha256: str | None = None
    schema_name: str = "rocc_history_labels"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_labels",
            frozenset(str(label) for label in self.allowed_labels),
        )
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer or None.")
        if self.expected_system_prompt_sha256 is None:
            return
        actual = hashlib.sha256(
            self.system_prompt.encode("utf-8")
        ).hexdigest()
        if actual != self.expected_system_prompt_sha256:
            raise ValueError(
                "System-Prompt stimmt nicht mit dem erwarteten SHA-256 "
                f"überein: {actual}"
            )


@dataclass(frozen=True)
class TeacherDatasetFinalization:
    """Completed Teacher dataset and its manifest."""

    dataset_path: Path
    manifest_path: Path
    manifest: dict[str, Any]


class TeacherRunner:
    """Run, persist, and resume one extractive Teacher protocol."""

    def __init__(
        self,
        *,
        protocol: TeacherProtocolConfig,
        cache_db: Path | str,
        concurrency: int = 8,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout: float = 240,
        progress: bool = True,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if concurrency < 1:
            raise ValueError("concurrency muss mindestens 1 sein.")
        self.protocol = protocol
        self.cache_db = Path(cache_db)
        self.concurrency = int(concurrency)
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = float(timeout)
        self.progress = bool(progress)
        self._client_factory = client_factory
        self._client_instance: Any | None = None
        self._jsonl_write_lock = Lock()
        self.cache_db.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._cache_connection()):
            pass

    def identity(self, sample: Any) -> tuple[str, str, str]:
        """Return rendered prompt, prompt hash, and stable cache key."""

        prompt = self.protocol.prompt_builder(sample)
        prompt_hash = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()
        cache_key = hashlib.sha256(
            (
                f"{sample.sample_id}\0{self.protocol.model}\0"
                f"{self.protocol.api_surface}\0"
                f"{self.protocol.protocol_hash}\0"
                f"{self.protocol.seed}\0"
                f"{prompt_hash}"
            ).encode("utf-8")
        ).hexdigest()
        return prompt, prompt_hash, cache_key

    def validate(
        self,
        sample: Any,
        raw: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate and materialize one extractive Teacher response."""

        return _validate_and_label(
            sample,
            raw,
            allowed_labels=self.protocol.allowed_labels,
        )

    def resolve(
        self,
        samples: Sequence[Any],
        *,
        allow_api: bool,
        dataset_path: Path | str,
        seed_paths: Sequence[Path | str] = (),
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Resolve samples from JSONL, seed JSONL, SQLite, then the API."""

        target = Path(dataset_path)
        seeds = tuple(Path(path) for path in seed_paths)
        persisted = self._load_jsonl(target)
        seeded: dict[tuple[str, str], dict[str, Any]] = {}
        for seed_path in seeds:
            if seed_path != target:
                seeded.update(self._load_jsonl(seed_path))
        cached = self._cached_rows()
        results: dict[str, dict[str, Any]] = {}
        pending: list[Any] = []
        persisted_count = 0
        seeded_count = 0
        sqlite_count = 0

        tqdm = get_tqdm()
        with tqdm(
            samples,
            desc="resolve teacher state",
            unit="query",
            dynamic_ncols=True,
            disable=not self.progress,
        ) as progress:
            for sample in progress:
                prompt, prompt_hash, _ = self.identity(sample)
                del prompt
                identity = (str(sample.sample_id), prompt_hash)
                if identity in persisted:
                    results[str(sample.sample_id)] = persisted[identity]
                    persisted_count += 1
                elif identity in seeded:
                    record = seeded[identity]
                    self._append_jsonl(target, record)
                    persisted[identity] = record
                    results[str(sample.sample_id)] = record
                    seeded_count += 1
                elif identity in cached:
                    (
                        validated,
                        prompt_tokens,
                        completion_tokens,
                    ) = cached[identity]
                    record = self._record(
                        validated,
                        prompt_hash=prompt_hash,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                    )
                    self._append_jsonl(target, record)
                    persisted[identity] = record
                    results[str(sample.sample_id)] = record
                    sqlite_count += 1
                else:
                    pending.append(sample)
                progress.set_postfix(
                    jsonl=persisted_count,
                    seeded=seeded_count,
                    sqlite=sqlite_count,
                    missing=len(pending),
                    refresh=False,
                )

        if not allow_api or not pending:
            return results, [
                str(sample.sample_id) for sample in pending
            ]

        self._client()
        batch_size = max(32, self.concurrency * 4)
        api_successes = 0
        api_failures = 0
        with tqdm(
            total=len(pending),
            desc=f"{self.protocol.model.upper()} teacher",
            unit="query",
            dynamic_ncols=True,
            disable=not self.progress,
        ) as progress:
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                batch_success = 0
                with ThreadPoolExecutor(
                    max_workers=min(self.concurrency, len(batch))
                ) as executor:
                    future_to_sample = {
                        executor.submit(
                            self._request_one,
                            sample,
                            target,
                        ): sample
                        for sample in batch
                    }
                    for future in as_completed(future_to_sample):
                        sample = future_to_sample[future]
                        try:
                            results[str(sample.sample_id)] = (
                                future.result()
                            )
                            batch_success += 1
                            api_successes += 1
                        except Exception:
                            api_failures += 1
                        finally:
                            progress.update(1)
                            progress.set_postfix(
                                ok=api_successes,
                                failed=api_failures,
                                refresh=False,
                            )
                if batch_success == 0:
                    progress.set_postfix(
                        ok=api_successes,
                        failed=api_failures,
                        stopped="failed batch",
                        refresh=True,
                    )
                    break

        missing = [
            str(sample.sample_id)
            for sample in samples
            if str(sample.sample_id) not in results
        ]
        return results, missing

    def usage_summary(
        self,
        rows: Iterable[dict[str, Any]],
    ) -> dict[str, int]:
        """Summarize token usage with the runner's progress setting."""

        return summarize_teacher_usage(
            rows,
            progress=self.progress,
        )

    def finalize_dataset(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        dataset_path: Path | str,
        dataset_name: str,
        eligible_queries: int,
        error_count: int = 0,
    ) -> TeacherDatasetFinalization:
        """Write the legacy manifest for one complete Teacher JSONL."""

        records = list(rows)
        if len(records) != int(eligible_queries):
            raise ValueError(
                "Dataset ist unvollständig: "
                f"{len(records)}/{int(eligible_queries)}"
            )

        usage = self.usage_summary(records)
        tqdm = get_tqdm()
        invalid_spans = sum(
            len(row["invalid_spans"])
            for row in tqdm(
                records,
                total=len(records),
                desc="summarize full labels",
                unit="query",
                dynamic_ncols=True,
                disable=not self.progress,
            )
        )
        target = Path(dataset_path)
        manifest = {
            "dataset": str(dataset_name),
            "eligible_queries": int(eligible_queries),
            "model": self.protocol.model,
            "api_surface": self.protocol.api_surface,
            "seed": self.protocol.seed,
            "temperature": self.protocol.temperature,
            "protocol_hash": self.protocol.protocol_hash,
            "labels": sorted(self.protocol.allowed_labels),
            "error_count": int(error_count),
            "invalid_spans": int(invalid_spans),
            **usage,
            "dataset_sha256": _file_sha256(
                target,
                progress=self.progress,
            ),
        }
        manifest_path = target.with_suffix(".manifest.json")
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return TeacherDatasetFinalization(
            dataset_path=target,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    def _cache_connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.cache_db, timeout=120)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS teacher_labels (
                cache_key TEXT PRIMARY KEY,
                sample_id TEXT NOT NULL,
                model TEXT NOT NULL,
                api_surface TEXT,
                protocol_hash TEXT,
                seed INTEGER,
                prompt_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                raw_json TEXT,
                validated_json TEXT,
                error TEXT,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        columns = {
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(teacher_labels)"
            )
        }
        for column, sql_type in (
            ("api_surface", "TEXT"),
            ("protocol_hash", "TEXT"),
            ("seed", "INTEGER"),
        ):
            if column not in columns:
                connection.execute(
                    "ALTER TABLE teacher_labels "
                    f"ADD COLUMN {column} {sql_type}"
                )
        connection.commit()
        return connection

    def _client(self) -> Any:
        if self._client_instance is not None:
            return self._client_instance

        endpoint = self.endpoint or os.environ.get(
            "AZURE_OPENAI_ENDPOINT"
        )
        api_key = self.api_key or os.environ.get(
            "AZURE_OPENAI_API_KEY"
        )
        if not endpoint or not api_key:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT und AZURE_OPENAI_API_KEY fehlen."
            )

        factory = self._client_factory
        if factory is None:
            from openai import OpenAI

            factory = OpenAI
        self._client_instance = factory(
            api_key=api_key,
            base_url=f"{endpoint.rstrip('/')}/openai/v1/",
            timeout=self.timeout,
        )
        return self._client_instance

    def _store_row(
        self,
        *,
        cache_key: str,
        sample_id: str,
        prompt_hash: str,
        status: str,
        raw: dict[str, Any] | None = None,
        validated: dict[str, Any] | None = None,
        error: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        with closing(self._cache_connection()) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO teacher_labels (
                    cache_key, sample_id, model, api_surface,
                    protocol_hash, seed, prompt_hash, status,
                    raw_json, validated_json, error,
                    prompt_tokens, completion_tokens, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        CURRENT_TIMESTAMP)
                """,
                (
                    cache_key,
                    sample_id,
                    self.protocol.model,
                    self.protocol.api_surface,
                    self.protocol.protocol_hash,
                    self.protocol.seed,
                    prompt_hash,
                    status,
                    (
                        json.dumps(raw, ensure_ascii=False)
                        if raw is not None
                        else None
                    ),
                    (
                        json.dumps(validated, ensure_ascii=False)
                        if validated is not None
                        else None
                    ),
                    error,
                    int(prompt_tokens),
                    int(completion_tokens),
                ),
            )
            connection.commit()

    def _record(
        self,
        validated: dict[str, Any],
        *,
        prompt_hash: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict[str, Any]:
        return {
            **validated,
            "teacher_metadata": {
                "model": self.protocol.model,
                "api_surface": self.protocol.api_surface,
                "protocol_hash": self.protocol.protocol_hash,
                "seed": self.protocol.seed,
                "prompt_hash": prompt_hash,
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": (
                    int(prompt_tokens) + int(completion_tokens)
                ),
            },
        }

    def _append_jsonl(
        self,
        path: Path,
        row: dict[str, Any],
    ) -> None:
        payload = (
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._jsonl_write_lock:
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_APPEND,
                0o600,
            )
            try:
                written = os.write(descriptor, payload)
                if written != len(payload):
                    raise OSError(
                        "Unvollständiger JSONL-Write: "
                        f"{written}/{len(payload)} Bytes"
                    )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _load_jsonl(
        self,
        path: Path,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if not path.exists():
            return {}

        rows: dict[tuple[str, str], dict[str, Any]] = {}
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
            disable=not self.progress,
        ) as progress:
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    valid_end = handle.tell()
                    break
                progress.update(len(line))
                line_end = handle.tell()
                if not line.endswith(b"\n"):
                    break
                try:
                    row = json.loads(line.decode("utf-8"))
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                ) as exc:
                    if line_end < file_size:
                        raise RuntimeError(
                            "Defekte JSONL-Zeile ab Byte "
                            f"{line_start}: {path}"
                        ) from exc
                    break
                if not isinstance(row, dict):
                    if line_end < file_size:
                        raise RuntimeError(
                            "Ungültige JSONL-Zeile ab Byte "
                            f"{line_start}: {path}"
                        )
                    break

                valid_end = line_end
                metadata = row.get("teacher_metadata", {})
                if (
                    metadata.get("model") == self.protocol.model
                    and metadata.get("api_surface")
                    == self.protocol.api_surface
                    and metadata.get("protocol_hash")
                    == self.protocol.protocol_hash
                    and metadata.get("seed")
                    == self.protocol.seed
                ):
                    sample_id = str(row.get("sample_id", ""))
                    prompt_hash = str(
                        metadata.get("prompt_hash", "")
                    )
                    if sample_id and prompt_hash:
                        rows[(sample_id, prompt_hash)] = row

        if valid_end < file_size:
            with self._jsonl_write_lock:
                with path.open("r+b") as handle:
                    handle.truncate(valid_end)
                    handle.flush()
                    os.fsync(handle.fileno())
            print("Unvollständiges JSONL-Ende entfernt:", path)
        return rows

    def _cached_rows(
        self,
    ) -> dict[tuple[str, str], tuple[dict[str, Any], int, int]]:
        with closing(self._cache_connection()) as connection:
            rows = connection.execute(
                """
                SELECT sample_id, prompt_hash, validated_json,
                       prompt_tokens, completion_tokens
                FROM teacher_labels
                WHERE model = ?
                  AND api_surface = ?
                  AND protocol_hash = ?
                  AND seed IS ?
                  AND status = 'ok'
                """,
                (
                    self.protocol.model,
                    self.protocol.api_surface,
                    self.protocol.protocol_hash,
                    self.protocol.seed,
                ),
            ).fetchall()
        tqdm = get_tqdm()
        return {
            (str(sample_id), str(prompt_hash)): (
                json.loads(validated_json),
                int(prompt_tokens),
                int(completion_tokens),
            )
            for (
                sample_id,
                prompt_hash,
                validated_json,
                prompt_tokens,
                completion_tokens,
            ) in tqdm(
                rows,
                desc="load teacher cache",
                unit="row",
                dynamic_ncols=True,
                disable=not self.progress,
            )
        }

    def _request_one(
        self,
        sample: Any,
        dataset_path: Path,
    ) -> dict[str, Any]:
        prompt, prompt_hash, cache_key = self.identity(sample)
        raw: dict[str, Any] | None = None
        try:
            response = self._client().chat.completions.create(
                model=self.protocol.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.protocol.system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=(
                    self.protocol.max_output_tokens
                ),
                seed=self.protocol.seed,
                temperature=self.protocol.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.protocol.schema_name,
                        "schema": self.protocol.output_schema,
                        "strict": True,
                    },
                },
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError(
                    "Azure lieferte eine leere Antwort."
                )
            raw = json.loads(content)
            if str(raw.get("sample_id", "")) != str(
                sample.sample_id
            ):
                raise ValueError(
                    "sample_id der Teacher-Antwort ist falsch."
                )

            validated = self.validate(sample, raw)
            if raw.get("spans") and not validated["spans"]:
                raise ValueError(
                    "Die Antwort enthielt keine verwendbaren Spans."
                )

            usage = response.usage
            prompt_tokens = int(
                getattr(usage, "prompt_tokens", 0) or 0
            )
            completion_tokens = int(
                getattr(usage, "completion_tokens", 0) or 0
            )
        except Exception as exc:
            self._store_row(
                cache_key=cache_key,
                sample_id=str(sample.sample_id),
                prompt_hash=prompt_hash,
                status="error",
                raw=raw,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        record = self._record(
            validated,
            prompt_hash=prompt_hash,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        self._append_jsonl(dataset_path, record)
        self._store_row(
            cache_key=cache_key,
            sample_id=str(sample.sample_id),
            prompt_hash=prompt_hash,
            status="ok",
            raw=raw,
            validated=record,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        return record


def summarize_teacher_usage(
    rows: Iterable[dict[str, Any]],
    *,
    progress: bool = True,
) -> dict[str, int]:
    """Sum prompt and completion tokens from Teacher records."""

    records = list(rows)
    prompt_tokens = 0
    completion_tokens = 0
    tqdm = get_tqdm()
    for row in tqdm(
        records,
        desc="summarize teacher usage",
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    ):
        metadata = row.get("teacher_metadata", {})
        prompt_tokens += int(
            metadata.get("prompt_tokens", 0) or 0
        )
        completion_tokens += int(
            metadata.get("completion_tokens", 0) or 0
        )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _file_sha256(
    path: Path,
    *,
    progress: bool,
) -> str:
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
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
            bar.update(len(block))
    return digest.hexdigest()


def _find_exact_span_locations(
    sample: Any,
    text: str,
) -> list[tuple[int, str, int, str, str]]:
    locations = []
    for turn in sample.history:
        for field in ("question", "answer"):
            source_text = getattr(turn, field)
            start = source_text.find(text)
            if start >= 0:
                locations.append(
                    (
                        int(turn.turn_id),
                        field,
                        start,
                        source_text,
                        text,
                    )
                )
    return locations


def _find_case_insensitive_span_locations(
    sample: Any,
    text: str,
) -> list[tuple[int, str, int, str, str]]:
    locations = []
    folded_text = text.casefold()
    if not folded_text:
        return locations
    for turn in sample.history:
        for field in ("question", "answer"):
            source_text = getattr(turn, field)
            start = source_text.casefold().find(folded_text)
            if start >= 0:
                locations.append(
                    (
                        int(turn.turn_id),
                        field,
                        start,
                        source_text,
                        source_text[start : start + len(text)],
                    )
                )
    return locations


def _token_labels(
    text: str,
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tokens = []
    for match in re.finditer(r"\S+", text):
        start, end = match.span()
        labels = sorted(
            {
                str(span["label"])
                for span in spans
                if (
                    start < int(span["char_end"])
                    and end > int(span["char_start"])
                )
            }
        )
        tokens.append(
            {
                "text": match.group(0),
                "start": start,
                "end": end,
                "labels": labels,
            }
        )
    return tokens


def _history_with_token_labels(
    sample: Any,
    spans: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    spans_by_key: dict[
        tuple[int, str],
        list[dict[str, Any]],
    ] = {}
    for span in spans:
        key = (int(span["turn_id"]), str(span["field"]))
        spans_by_key.setdefault(key, []).append(span)

    return [
        {
            "turn_id": int(turn.turn_id),
            "question": turn.question,
            "answer": turn.answer,
            "question_tokens": _token_labels(
                turn.question,
                spans_by_key.get(
                    (int(turn.turn_id), "question"),
                    [],
                ),
            ),
            "answer_tokens": _token_labels(
                turn.answer,
                spans_by_key.get(
                    (int(turn.turn_id), "answer"),
                    [],
                ),
            ),
        }
        for turn in sample.history
    ]


def _validate_and_label(
    sample: Any,
    raw: dict[str, Any],
    *,
    allowed_labels: Collection[str],
) -> dict[str, Any]:
    source_by_turn = {
        int(turn.turn_id): turn for turn in sample.history
    }
    valid_spans = []
    invalid_spans = []
    raw_sample_id = str(raw.get("sample_id", ""))
    raw_spans = raw.get("spans", [])
    if raw_sample_id != str(sample.sample_id):
        invalid_spans.append(
            {
                "sample_id": raw_sample_id,
                "error": (
                    "sample_id mismatch; expected "
                    f"{sample.sample_id}"
                ),
            }
        )
        raw_spans = []

    for index, span in enumerate(raw_spans):
        label = str(span.get("label", ""))
        field = str(span.get("field", ""))
        text = str(span.get("text", ""))
        turn_id = int(span.get("turn_id", -1))
        source_turn = source_by_turn.get(turn_id)
        source_text = (
            getattr(source_turn, field, None)
            if source_turn
            and field in {"question", "answer"}
            else None
        )
        if not text or label not in allowed_labels:
            invalid_spans.append(
                {**span, "error": "bad label or empty text"}
            )
            continue

        repair: dict[str, Any] | None = None
        start = (
            source_text.find(text)
            if source_text is not None
            else -1
        )
        if start < 0:
            locations = _find_exact_span_locations(sample, text)
            if len(locations) == 1:
                (
                    repaired_turn_id,
                    repaired_field,
                    repaired_start,
                    repaired_source_text,
                    repaired_text,
                ) = locations[0]
                repair = {
                    "turn_id": turn_id,
                    "field": field,
                    "text": text,
                }
            else:
                locations = (
                    _find_case_insensitive_span_locations(
                        sample,
                        text,
                    )
                )
                if len(locations) != 1:
                    invalid_spans.append(
                        {
                            **span,
                            "error": (
                                "text is not an exact substring"
                            ),
                        }
                    )
                    continue
                (
                    repaired_turn_id,
                    repaired_field,
                    repaired_start,
                    repaired_source_text,
                    repaired_text,
                ) = locations[0]
                repair = {
                    "turn_id": turn_id,
                    "field": field,
                }
            turn_id = repaired_turn_id
            field = repaired_field
            text = repaired_text
            start = repaired_start
            source_text = repaired_source_text

        valid_span = {
            "span_id": index,
            "turn_id": turn_id,
            "field": field,
            "label": label,
            "text": text,
            "char_start": start,
            "char_end": start + len(text),
            "reason": str(span.get("reason", "")),
        }
        if repair is not None:
            valid_span["repaired_from"] = repair
        valid_spans.append(valid_span)

    return {
        "sample_id": str(sample.sample_id),
        "split": str(sample.split),
        "conv_id": int(sample.conv_id),
        "turn_id": int(sample.turn_id),
        "history_len": len(sample.history),
        "current_query": sample.current_query,
        "history": _history_with_token_labels(
            sample,
            valid_spans,
        ),
        "spans": valid_spans,
        "invalid_spans": invalid_spans,
    }


__all__ = [
    "TeacherDatasetFinalization",
    "TeacherProtocolConfig",
    "TeacherRunner",
    "summarize_teacher_usage",
]
