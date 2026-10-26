"""Adapters for evaluating published external retrieval runs."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .progress import get_tqdm


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_external_run_file(
    path: Path | str,
    *,
    url: str,
    expected_sha256: str,
    progress: bool = True,
) -> Path:
    """Download a published run once and enforce its registered digest."""

    target = Path(path).expanduser().resolve()
    expected = str(expected_sha256).lower()
    if target.is_file():
        observed = sha256_file(target)
        if observed != expected:
            raise RuntimeError(
                f"External run hash mismatch for {target}: {observed}"
            )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    if temporary.exists():
        temporary.unlink()

    request = urllib.request.Request(
        str(url),
        headers={"User-Agent": "ROCC-thesis-experiment/1.0"},
    )
    try:
        with urllib.request.urlopen(request) as response, temporary.open(
            "wb"
        ) as output:
            total = int(response.headers.get("Content-Length") or 0)
            bar = get_tqdm()(
                total=total or None,
                desc=f"download {target.name}",
                unit="B",
                unit_scale=True,
                disable=not progress,
            )
            try:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    bar.update(len(chunk))
            finally:
                bar.close()
        observed = sha256_file(temporary)
        if observed != expected:
            raise RuntimeError(
                f"Downloaded run hash mismatch for {target}: {observed}"
            )
        shutil.move(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def unique_ranked_docids(
    docids: Iterable[Any],
    *,
    top_k: int,
) -> list[str]:
    """Keep the first occurrence of each document up to ``top_k``."""

    if int(top_k) < 1:
        raise ValueError("top_k must be positive.")
    result: list[str] = []
    seen: set[str] = set()
    for raw_docid in docids:
        docid = str(raw_docid)
        if docid in seen:
            continue
        seen.add(docid)
        result.append(docid)
        if len(result) >= int(top_k):
            break
    return result


def load_ranked_run_file(
    path: Path | str,
    *,
    top_k: int,
) -> dict[str, list[str]]:
    """Load a six-column ranked run and retain at most ``top_k`` hits."""

    resolved = Path(path).expanduser().resolve()
    rows: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    ranks_seen: set[tuple[str, int]] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(
                    f"Malformed six-column run row {resolved}:{line_number}."
                )
            qid, _q0, docid, rank_text, _score, _tag = parts
            rank = int(rank_text)
            if rank < 1:
                raise ValueError(
                    f"Non-positive rank at {resolved}:{line_number}."
                )
            rank_key = (str(qid), rank)
            if rank_key in ranks_seen:
                raise ValueError(
                    f"Duplicate rank for query {qid!r} in {resolved}."
                )
            ranks_seen.add(rank_key)
            rows[str(qid)].append((rank, line_number, str(docid)))
    if not rows:
        raise ValueError(f"External run is empty: {resolved}")
    return {
        qid: unique_ranked_docids(
            (
                docid
                for _rank, _line, docid in sorted(
                    values,
                    key=lambda value: (value[0], value[1]),
                )
            ),
            top_k=int(top_k),
        )
        for qid, values in rows.items()
    }


def load_jsonl_rankings(
    path: Path | str,
    *,
    query_ids: Iterable[str],
    top_k: int,
) -> dict[str, list[str]]:
    """Select rankings by query ID from a JSONL or JSONL.GZ export."""

    resolved = Path(path).expanduser().resolve()
    requested = {str(query_id) for query_id in query_ids}
    opener = gzip.open if resolved.suffix == ".gz" else open
    result: dict[str, list[str]] = {}
    with opener(resolved, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id not in requested:
                continue
            if query_id in result:
                raise ValueError(
                    f"Duplicate query_id {query_id!r} at line {line_number}."
                )
            result[query_id] = unique_ranked_docids(
                row["docids"],
                top_k=int(top_k),
            )
    missing = requested.difference(result)
    if missing:
        raise RuntimeError(
            f"{resolved} is missing {len(missing)} requested rankings."
        )
    return result


def topiocqa_dqcis_qid_map(
    sample_ids: Iterable[str],
    dqcis_qids: Iterable[str],
) -> dict[str, str]:
    """Map TopiOCQA ``conv_turn`` QIDs to canonical dev sample IDs."""

    sample_set = {str(sample_id) for sample_id in sample_ids}
    result: dict[str, str] = {}
    for raw_qid in dqcis_qids:
        qid = str(raw_qid)
        conv_id, turn_id = _parse_dqcis_qid(qid)
        sample_id = f"dev:{conv_id}:{turn_id}"
        if sample_id not in sample_set:
            raise RuntimeError(
                f"DQ-CIS TopiOCQA QID has no dev sample: {qid}"
            )
        result[qid] = sample_id
    if set(result.values()) != sample_set:
        raise RuntimeError(
            "DQ-CIS TopiOCQA QIDs do not cover the complete population."
        )
    return result


def qrecc_dqcis_qid_map(
    raw_test_frame: pd.DataFrame,
    dqcis_qids: Iterable[str],
) -> pd.DataFrame:
    """Align DQ-CIS QReCC QIDs by conversation and turn order.

    DQ-CIS and the canonical QReCC test split contain the same ordered turns
    per conversation, but some published QID suffixes differ from the local
    ``Turn_no`` values. Alignment therefore uses the within-conversation
    ordinal after validating equal group sizes.
    """

    required = {"conv_id", "turn_id"}
    missing = required.difference(raw_test_frame.columns)
    if missing:
        raise KeyError(
            "QReCC mapping columns missing: " + ", ".join(sorted(missing))
        )
    frame = raw_test_frame.copy()
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).eq("test")].copy()
    if frame.duplicated(["conv_id", "turn_id"]).any():
        raise ValueError("QReCC test frame contains duplicate turns.")

    external_by_conversation: dict[int, list[tuple[int, str]]] = defaultdict(
        list
    )
    for raw_qid in dqcis_qids:
        qid = str(raw_qid)
        conv_id, turn_suffix = _parse_dqcis_qid(qid)
        external_by_conversation[conv_id].append((turn_suffix, qid))

    local_conversations = {
        int(conv_id)
        for conv_id in frame["conv_id"].astype(int).unique()
    }
    if set(external_by_conversation) != local_conversations:
        raise RuntimeError(
            "DQ-CIS and QReCC test conversations do not match."
        )

    rows: list[dict[str, Any]] = []
    for conv_id, local_group in frame.groupby(
        "conv_id",
        observed=True,
        sort=True,
    ):
        local_rows = local_group.sort_values(
            "turn_id",
            kind="mergesort",
        )
        external_rows = sorted(external_by_conversation[int(conv_id)])
        if len(local_rows) != len(external_rows):
            raise RuntimeError(
                f"Turn-count mismatch for QReCC conversation {conv_id}: "
                f"local={len(local_rows)}, DQ-CIS={len(external_rows)}."
            )
        for ordinal, (local, external) in enumerate(
            zip(local_rows.itertuples(index=False), external_rows, strict=True),
            start=1,
        ):
            external_suffix, qid = external
            rows.append(
                {
                    "dqcis_qid": qid,
                    "sample_id": (
                        f"test:{int(local.conv_id)}:{int(local.turn_id)}"
                    ),
                    "conv_id": int(local.conv_id),
                    "turn_id": int(local.turn_id),
                    "turn_ordinal": int(ordinal),
                    "dqcis_turn_suffix": int(external_suffix),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["conv_id", "turn_ordinal"],
        kind="mergesort",
    ).reset_index(drop=True)
    if len(result) != len(frame):
        raise RuntimeError("Incomplete QReCC DQ-CIS QID alignment.")
    if result["dqcis_qid"].duplicated().any():
        raise RuntimeError("Duplicate DQ-CIS QID after QReCC alignment.")
    if result["sample_id"].duplicated().any():
        raise RuntimeError("Duplicate QReCC sample ID after alignment.")
    return result


def load_qrecc_numeric_to_raw_docids(
    collection_jsonl: Path | str,
    offsets_path: Path | str,
    passage_ids: Iterable[str | int],
    *,
    progress: bool = True,
) -> dict[str, str]:
    """Resolve numeric QReCC collection IDs with one shared file handle."""

    collection = Path(collection_jsonl).expanduser().resolve()
    offsets_file = Path(offsets_path).expanduser().resolve()
    if not collection.is_file():
        raise FileNotFoundError(collection)
    if not offsets_file.is_file():
        raise FileNotFoundError(offsets_file)
    if offsets_file.stat().st_size % 8:
        raise RuntimeError("QReCC offset table is not uint64-aligned.")

    indices = sorted({int(passage_id) for passage_id in passage_ids})
    offsets = np.memmap(offsets_file, mode="r", dtype=np.uint64)
    if indices and (indices[0] < 0 or indices[-1] >= len(offsets)):
        raise IndexError("QReCC passage ID lies outside the offset table.")

    result: dict[str, str] = {}
    rows = get_tqdm()(
        indices,
        total=len(indices),
        desc="map QReCC ANCE IDs to raw IDs",
        unit="passage",
        dynamic_ncols=True,
        disable=not progress,
    )
    with collection.open("rb") as handle:
        for index in rows:
            handle.seek(int(offsets[index]))
            row = json.loads(handle.readline().decode("utf-8"), strict=False)
            if str(row.get("id")) != str(index):
                raise RuntimeError(
                    f"QReCC offset mismatch at passage {index}."
                )
            raw_passage_id = row.get("raw_passage_id")
            if raw_passage_id is None:
                raise RuntimeError(
                    f"QReCC passage {index} has no raw_passage_id."
                )
            result[str(index)] = str(raw_passage_id)
    return result


def remap_rankings(
    rankings: Mapping[str, Sequence[Any]],
    docid_mapping: Mapping[str, str],
    *,
    top_k: int,
) -> dict[str, list[str]]:
    """Translate document IDs while retaining deterministic ranking order."""

    result: dict[str, list[str]] = {}
    for query_id, docids in rankings.items():
        translated: list[str] = []
        for raw_docid in docids:
            docid = str(raw_docid)
            if docid not in docid_mapping:
                raise KeyError(f"Document-ID mapping missing for {docid}.")
            translated.append(str(docid_mapping[docid]))
        result[str(query_id)] = unique_ranked_docids(
            translated,
            top_k=int(top_k),
        )
    return result


def _parse_dqcis_qid(qid: str) -> tuple[int, int]:
    try:
        conv_text, turn_text = str(qid).rsplit("_", 1)
        return int(conv_text), int(turn_text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid DQ-CIS query ID: {qid!r}") from error
