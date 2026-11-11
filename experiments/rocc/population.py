"""Reusable helpers for deterministic conversational-query populations."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .progress import get_tqdm, progress_iter


DepthBounds = Mapping[str, tuple[int, int | None]]


@dataclass(frozen=True)
class ConversationSplit:
    """One deterministic split made of complete conversations."""

    name: str
    sample_ids: tuple[str, ...]
    conversation_ids: tuple[Any, ...]
    sample_id_sha256: str

    @property
    def query_count(self) -> int:
        return len(self.sample_ids)

    @property
    def conversation_count(self) -> int:
        return len(self.conversation_ids)


@dataclass(frozen=True)
class ConversationSplitBundle:
    """Named, conversation-disjoint deterministic splits."""

    splits: Mapping[str, ConversationSplit]
    seed: int
    algorithm: str

    def __getitem__(self, name: str) -> ConversationSplit:
        return self.splits[name]

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "split": split.name,
                    "queries": split.query_count,
                    "conversations": split.conversation_count,
                    "sample_id_sha256": split.sample_id_sha256,
                }
                for split in self.splits.values()
            ]
        )


def stable_sha1(value: str) -> str:
    """Return the deterministic SHA-1 key used for population tie-breaking."""

    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def derive_query_seed(
    base_seed: int,
    sample_id: str,
    candidate: str,
) -> int:
    """Derive one query-local RNG seed with the established SHA-256 rule."""

    payload = (
        f"{int(base_seed)}\0{str(sample_id)}\0{str(candidate)}"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], byteorder="big")


def add_conversation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the canonical ROCC sample ID and conversational history depth."""

    result = frame.copy()
    split = result["split"].astype(str)
    conv_id = pd.to_numeric(result["conv_id"], errors="raise").astype(int)
    turn_id = pd.to_numeric(result["turn_id"], errors="raise").astype(int)
    sample_id = (
        split
        + ":"
        + conv_id.astype(str)
        + ":"
        + turn_id.astype(str)
    )
    if "sample_id" in result:
        existing = result["sample_id"].astype(str)
        if not existing.equals(sample_id):
            raise ValueError("Existing sample_id values are not canonical.")
    result["sample_id"] = sample_id
    result["history_depth"] = turn_id - 1
    return result


def build_depth_eligible_population(
    frame: pd.DataFrame,
    *,
    depth_bounds: DepthBounds,
    depth_bin_order: Sequence[str],
) -> pd.DataFrame:
    """Assign depth bins and return rows covered by those bins."""

    def assign_depth_bin(depth: int) -> str | None:
        for label, (lower, upper) in depth_bounds.items():
            if depth >= lower and (upper is None or depth <= upper):
                return label
        return None

    result = frame.copy()
    result["depth_bin"] = result["history_depth"].map(assign_depth_bin)
    result = result.loc[result["depth_bin"].notna()].copy()
    result["depth_bin"] = pd.Categorical(
        result["depth_bin"],
        categories=tuple(depth_bin_order),
        ordered=True,
    )
    return result


def select_depth_balanced_population(
    eligible_population: pd.DataFrame,
    *,
    depth_bin_order: Sequence[str],
    selection_order: Sequence[str],
    samples_per_bin: int,
    seed: int,
    progress: bool = True,
) -> pd.DataFrame:
    """Select one query per conversation while balancing exact depths."""

    used_conversations: set[Any] = set()
    selected_parts: list[pd.DataFrame] = []
    bins = progress_iter(
        selection_order,
        total=len(selection_order),
        desc="select depth-balanced population",
        unit="bin",
        enabled=progress,
    )
    for depth_bin in bins:
        pool = eligible_population.loc[
            eligible_population["depth_bin"].eq(depth_bin)
            & ~eligible_population["conv_id"].isin(used_conversations)
        ]
        selected = _select_depth_balanced_queries(
            pool,
            depth_bin=str(depth_bin),
            target_size=samples_per_bin,
            seed=seed,
            progress=progress,
        )
        selected_parts.append(selected)
        used_conversations.update(selected["conv_id"].tolist())

    population = pd.concat(selected_parts, ignore_index=True)
    population["depth_bin"] = pd.Categorical(
        population["depth_bin"],
        categories=tuple(depth_bin_order),
        ordered=True,
    )
    return population.sort_values(
        ["depth_bin", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_conversation_split_bundle(
    frame: pd.DataFrame,
    *,
    exact_sizes: Mapping[str, int],
    fractional_splits: Mapping[str, float] | None = None,
    hash_salts: Mapping[str, str] | None = None,
    seed: int,
    progress: bool = True,
) -> ConversationSplitBundle:
    """Build exact and fractional splits from complete conversations."""

    required = {"sample_id", "conv_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            "Split frame is missing columns: "
            + ", ".join(sorted(missing))
        )
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id values must be unique.")
    if any(int(size) < 1 for size in exact_sizes.values()):
        raise ValueError("Exact split sizes must be positive.")
    fractional = dict(fractional_splits or {})
    if any(not 0.0 < float(value) <= 1.0 for value in fractional.values()):
        raise ValueError("Fractional split sizes must be in (0, 1].")
    duplicate_names = set(exact_sizes).intersection(fractional)
    if duplicate_names:
        raise ValueError(
            "Split names occur twice: "
            + ", ".join(sorted(duplicate_names))
        )
    salts = {
        str(name): str(value)
        for name, value in (hash_salts or {}).items()
    }
    unknown_salts = set(salts).difference(
        set(exact_sizes).union(fractional)
    )
    if unknown_salts:
        raise ValueError(
            "Hash salts reference unknown splits: "
            + ", ".join(sorted(unknown_salts))
        )

    rows_by_conversation = {
        conv_id: group["sample_id"].astype(str).tolist()
        for conv_id, group in frame.groupby("conv_id", sort=False)
    }
    remaining = set(rows_by_conversation)
    selected_by_name: dict[str, set[Any]] = {}
    for split_name, target_size in exact_sizes.items():
        split_salt = salts.get(str(split_name), str(split_name))
        ranked = sorted(
            remaining,
            key=lambda conv_id: (
                stable_sha1(f"{seed}:{split_salt}:{conv_id}"),
                str(conv_id),
            ),
        )
        weights = {
            conv_id: len(rows_by_conversation[conv_id])
            for conv_id in ranked
        }
        selected = _exact_conversation_subset(
            ranked,
            weights=weights,
            target_size=int(target_size),
            progress=progress,
            progress_desc=f"build {split_name}",
        )
        selected_by_name[str(split_name)] = selected
        remaining.difference_update(selected)

    for split_name, fraction in fractional.items():
        split_salt = salts.get(str(split_name), str(split_name))
        ranked = sorted(
            remaining,
            key=lambda conv_id: (
                stable_sha1(f"{seed}:{split_salt}:{conv_id}"),
                str(conv_id),
            ),
        )
        count = math.ceil(float(fraction) * len(ranked))
        selected = set(ranked[:count])
        selected_by_name[str(split_name)] = selected
        remaining.difference_update(selected)

    splits: dict[str, ConversationSplit] = {}
    seen_conversations: set[Any] = set()
    for split_name, selected in selected_by_name.items():
        overlap = seen_conversations.intersection(selected)
        if overlap:
            raise RuntimeError(
                f"Conversation leakage in {split_name}: "
                f"{sorted(map(str, overlap))[:3]}"
            )
        seen_conversations.update(selected)
        sample_ids = tuple(
            sorted(
                sample_id
                for conv_id in selected
                for sample_id in rows_by_conversation[conv_id]
            )
        )
        splits[split_name] = ConversationSplit(
            name=split_name,
            sample_ids=sample_ids,
            conversation_ids=tuple(
                sorted(selected, key=lambda value: str(value))
            ),
            sample_id_sha256=hashlib.sha256(
                "\n".join(sample_ids).encode("utf-8")
            ).hexdigest(),
        )

    return ConversationSplitBundle(
        splits=splits,
        seed=int(seed),
        algorithm="stable_sha1_subset_sum_v1",
    )


def _exact_conversation_subset(
    conversation_ids: Sequence[Any],
    *,
    weights: Mapping[Any, int],
    target_size: int,
    progress: bool,
    progress_desc: str,
) -> set[Any]:
    reachable = [False] * (target_size + 1)
    previous_total = [-1] * (target_size + 1)
    previous_conversation: list[Any | None] = [
        None
    ] * (target_size + 1)
    reachable[0] = True

    ranked = progress_iter(
        conversation_ids,
        total=len(conversation_ids),
        desc=progress_desc,
        unit="conversation",
        enabled=progress,
    )
    for conv_id in ranked:
        weight = int(weights[conv_id])
        if weight < 1:
            raise ValueError(
                f"Conversation {conv_id!r} has no rows."
            )
        for current in range(target_size - weight, -1, -1):
            destination = current + weight
            if reachable[current] and not reachable[destination]:
                reachable[destination] = True
                previous_total[destination] = current
                previous_conversation[destination] = conv_id
        if reachable[target_size]:
            break

    if not reachable[target_size]:
        raise ValueError(
            "No exact complete-conversation subset reaches "
            f"target_size={target_size}."
        )

    selected: set[Any] = set()
    current = target_size
    while current:
        conv_id = previous_conversation[current]
        if conv_id is None:
            raise RuntimeError(
                "Broken subset-sum predecessor chain."
            )
        selected.add(conv_id)
        current = previous_total[current]
    return selected


def _select_depth_balanced_queries(
    pool: pd.DataFrame,
    *,
    depth_bin: str,
    target_size: int,
    seed: int,
    progress: bool,
) -> pd.DataFrame:
    ranked = pool.copy()
    ranked["conversation_max_depth_in_bin"] = ranked.groupby(
        "conv_id",
        observed=True,
    )["history_depth"].transform("max")
    ranked["selection_hash"] = ranked.apply(
        lambda row: stable_sha1(
            f"{seed}:{depth_bin}:{row['history_depth']}:"
            f"{row['conv_id']}:{row['sample_id']}"
        ),
        axis=1,
    )
    ranked["conv_id_sort"] = ranked["conv_id"].astype(str)

    available_conversations = ranked["conv_id"].nunique()
    if available_conversations < target_size:
        raise ValueError(
            f"{depth_bin} contains only {available_conversations} "
            f"available conversations for target_size={target_size}."
        )

    exact_depths = sorted(
        ranked["history_depth"].astype(int).unique().tolist()
    )
    selected_counts = {depth: 0 for depth in exact_depths}
    selected_indices: list[int] = []
    selected_conversations: set[Any] = set()
    bar = get_tqdm()(
        total=target_size,
        desc=f"balance exact depths in {depth_bin}",
        unit="query",
        disable=not progress,
        leave=False,
        dynamic_ncols=True,
    )
    try:
        while len(selected_indices) < target_size:
            available = ranked.loc[
                ~ranked["conv_id"].isin(selected_conversations)
            ]
            remaining = (
                available.groupby("history_depth", observed=True)["conv_id"]
                .nunique()
                .astype(int)
            )
            active_depths = [
                depth
                for depth in exact_depths
                if int(remaining.get(depth, 0)) > 0
            ]
            if not active_depths:
                raise RuntimeError(
                    "Could not complete depth-balanced sampling for "
                    f"{depth_bin}."
                )
            minimum = min(
                selected_counts[depth] for depth in active_depths
            )
            chosen_depth = min(
                [
                    depth
                    for depth in active_depths
                    if selected_counts[depth] == minimum
                ],
                key=lambda depth: (int(remaining.loc[depth]), -depth),
            )
            row = (
                available.loc[
                    available["history_depth"].eq(chosen_depth)
                ]
                .sort_values(
                    [
                        "conversation_max_depth_in_bin",
                        "selection_hash",
                        "conv_id_sort",
                        "sample_id",
                    ],
                    kind="mergesort",
                )
                .iloc[0]
            )
            selected_indices.append(int(row.name))
            selected_conversations.add(row["conv_id"])
            selected_counts[chosen_depth] += 1
            bar.update(1)
    finally:
        bar.close()

    return ranked.loc[selected_indices].drop(
        columns=[
            "conversation_max_depth_in_bin",
            "selection_hash",
            "conv_id_sort",
        ]
    )
