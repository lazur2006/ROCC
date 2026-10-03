"""Cached Teacher-agreement evaluation on labeled TopiOCQA Dev."""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .crf_sampling import load_crf_inference_model
from .history_selector import (
    LABEL_TO_ID,
    collate_selector_batch,
    evaluate_history_selector,
    read_label_jsonl,
)
from .selector_experiment import (
    SelectorExperimentConfig,
    prepare_selector_datasets,
)


PROTOCOL = "nb07a_topiocqa_teacher_dev_agreement_v1"
SYSTEM_ORDER = (
    "pretrained",
    "imitation_control_e6",
    "bm25_treatment_e6",
)
TEACHER_LABELS = (
    "CONCEPT",
    "DEFINITION",
    "ENTITY",
    "KEY_TERM",
    "RELATION_CUE",
)
METRIC_COLUMNS = (
    "token_accuracy",
    "keep_precision",
    "keep_recall",
    "keep_f1",
    "keep_tp",
    "keep_fp",
    "keep_fn",
    "token_correct",
    "token_total",
    "span_precision",
    "span_recall",
    "span_f1",
)
DATA_FILENAMES = ("metrics.csv", "population.csv")


@dataclass(frozen=True)
class TeacherDevAgreementResult:
    """Persisted or newly computed Teacher-agreement result."""

    metrics: pd.DataFrame
    population: pd.DataFrame
    manifest: dict[str, Any]
    manifest_path: Path
    data_paths: tuple[Path, ...]
    reused: bool

    @property
    def artifact_paths(self) -> tuple[Path, ...]:
        """Return data artifacts followed by their manifest."""

        return (*self.data_paths, self.manifest_path)


def load_or_compute_topiocqa_teacher_dev_agreement(
    *,
    result_dir: Path | str,
    full_dev_rows: Sequence[Mapping[str, Any]],
    full_dev_population_sha256: str,
    checkpoints: Mapping[str, Path | str],
    expected_checkpoint_sha256: Mapping[str, str],
    checkpoint_manifests: Mapping[str, Path | str],
    teacher_labels_path: Path | str,
    teacher_manifest_path: Path | str,
    expected_teacher_dataset_sha256: str,
    expected_teacher_manifest_sha256: str,
    expected_teacher_protocol_hash: str,
    expected_teacher_seed: int,
    experiment_config: SelectorExperimentConfig,
    encoding_cache_dir: Path | str,
    device: torch.device | str,
    batch_size: int = 32,
    progress: bool = True,
) -> TeacherDevAgreementResult:
    """Load or compute canonical collapsed-KEEP metrics on Teacher Dev.

    The denominator contains only visible history WordPieces of the 2,104
    Teacher-labeled TopiOCQA-Dev queries with at least two history turns.
    """

    resolved_result = Path(result_dir)
    output_dir = resolved_result / "teacher_dev_agreement"
    manifest_path = output_dir / "manifest.json"
    data_paths = tuple(output_dir / name for name in DATA_FILENAMES)
    data_by_name = dict(zip(DATA_FILENAMES, data_paths, strict=True))

    if int(batch_size) < 1:
        raise ValueError("batch_size must be positive.")
    normalized_rows = [dict(row) for row in full_dev_rows]
    full_dev_identity = _validate_full_dev_rows(
        normalized_rows,
        population_sha256=str(full_dev_population_sha256),
    )
    normalized_checkpoints = _normalize_system_paths(
        checkpoints,
        label="checkpoints",
    )
    normalized_manifests = _normalize_system_paths(
        checkpoint_manifests,
        label="checkpoint_manifests",
    )
    normalized_hashes = {
        str(system): str(digest)
        for system, digest in expected_checkpoint_sha256.items()
    }
    if tuple(normalized_checkpoints) != SYSTEM_ORDER:
        raise ValueError("Checkpoint order or system set is not canonical.")
    if tuple(normalized_manifests) != SYSTEM_ORDER:
        raise ValueError(
            "Checkpoint-manifest order or system set is not canonical."
        )
    if tuple(normalized_hashes) != SYSTEM_ORDER:
        raise ValueError(
            "Expected checkpoint order or system set is not canonical."
        )

    labels_path = Path(teacher_labels_path)
    source_manifest_path = Path(teacher_manifest_path)
    if labels_path.is_file() != source_manifest_path.is_file():
        raise RuntimeError(
            "Teacher labels and their source manifest are partially present."
        )
    source_available = labels_path.is_file()
    expected_model_contract = {
        "model_name": "sentence-transformers/all-MiniLM-L12-v2",
        "model_revision": (
            "a50ef00143b4d5391434df20ae11632588ac25be"
        ),
        "max_length": 512,
        "history_order": "recent_first",
    }
    observed_model_contract = {
        "model_name": str(experiment_config.model_name),
        "model_revision": str(experiment_config.model_revision),
        "max_length": int(experiment_config.max_length),
        "history_order": str(experiment_config.history_order),
    }
    if observed_model_contract != expected_model_contract:
        raise RuntimeError(
            "Teacher-Dev evaluator configuration is not canonical: "
            f"{observed_model_contract}"
        )

    identity = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": "TopiOCQA dev",
        "evaluation_scope": "teacher_labeled_history_depth_ge_2",
        "full_dev_population_sha256": str(
            full_dev_population_sha256
        ),
        **full_dev_identity,
        "teacher_dataset_sha256": str(
            expected_teacher_dataset_sha256
        ),
        "teacher_manifest_sha256": str(
            expected_teacher_manifest_sha256
        ),
        "teacher_protocol_hash": str(expected_teacher_protocol_hash),
        "teacher_seed": int(expected_teacher_seed),
        "teacher_labels": list(TEACHER_LABELS),
        "label_space": "collapsed",
        "label_ids": dict(LABEL_TO_ID),
        "checkpoint_sha256": normalized_hashes,
        "systems": list(SYSTEM_ORDER),
        "model_name": str(experiment_config.model_name),
        "model_revision": str(experiment_config.model_revision),
        "max_length": int(experiment_config.max_length),
        "history_order": str(experiment_config.history_order),
        "batch_size": int(batch_size),
        "visible_token_rule": (
            "history_wordpieces_only; current_query_special_padding_and_"
            "truncated_history_excluded"
        ),
        "keep_rule": "gold_or_prediction_label_is_not_O",
        "span_rule": "exact_BIO_wordpiece_span_match",
        "prepare_function_sha256": _sha256_text(
            inspect.getsource(prepare_selector_datasets)
        ),
        "evaluate_function_sha256": _sha256_text(
            inspect.getsource(evaluate_history_selector)
        ),
        "collate_function_sha256": _sha256_text(
            inspect.getsource(collate_selector_batch)
        ),
        "model_loader_sha256": _sha256_text(
            inspect.getsource(load_crf_inference_model)
        ),
        "implementation_sha256": _sha256_file(Path(__file__)),
    }

    artifact_presence = {
        "manifest.json": manifest_path.is_file(),
        **{
            filename: path.is_file()
            for filename, path in data_by_name.items()
        },
    }
    if any(artifact_presence.values()) and not all(
        artifact_presence.values()
    ):
        raise RuntimeError(
            "Teacher-Dev agreement artifacts are partially present: "
            + json.dumps(artifact_presence, sort_keys=True)
        )

    if all(artifact_presence.values()):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _validate_manifest(
            manifest=manifest,
            identity=identity,
            data_by_name=data_by_name,
        )
        metrics = pd.read_csv(data_by_name["metrics.csv"])
        population = pd.read_csv(data_by_name["population.csv"])
        reused = True
    else:
        if not source_available:
            raise FileNotFoundError(
                "Teacher-Dev source files are required for the first "
                "agreement computation."
            )
        observed_dataset_sha256 = _sha256_file(labels_path)
        observed_manifest_sha256 = _sha256_file(source_manifest_path)
        if observed_dataset_sha256 != str(
            expected_teacher_dataset_sha256
        ):
            raise RuntimeError("Teacher-Dev dataset hash drift.")
        if observed_manifest_sha256 != str(
            expected_teacher_manifest_sha256
        ):
            raise RuntimeError("Teacher-Dev source-manifest hash drift.")
        source_manifest = json.loads(
            source_manifest_path.read_text(encoding="utf-8")
        )
        _validate_teacher_manifest(
            source_manifest,
            expected_dataset_sha256=str(
                expected_teacher_dataset_sha256
            ),
            expected_protocol_hash=str(expected_teacher_protocol_hash),
            expected_seed=int(expected_teacher_seed),
        )
        checkpoint_manifest_hashes: dict[str, str] = {}
        for system in SYSTEM_ORDER:
            checkpoint = normalized_checkpoints[system]
            checkpoint_manifest = normalized_manifests[system]
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            if not checkpoint_manifest.is_file():
                raise FileNotFoundError(checkpoint_manifest)
            observed_checkpoint_hash = _sha256_file(checkpoint)
            if observed_checkpoint_hash != normalized_hashes[system]:
                raise RuntimeError(
                    f"Checkpoint hash drift for {system}: "
                    f"{observed_checkpoint_hash}"
                )
            checkpoint_payload = json.loads(
                checkpoint_manifest.read_text(encoding="utf-8")
            )
            _validate_checkpoint_manifest(
                system=system,
                manifest=checkpoint_payload,
                expected_checkpoint_sha256=normalized_hashes[system],
                model_name=experiment_config.model_name,
                model_revision=experiment_config.model_revision,
            )
            checkpoint_manifest_hashes[system] = _sha256_file(
                checkpoint_manifest
            )
        teacher_rows = read_label_jsonl(labels_path, progress=progress)
        if any(
            row.get("teacher_metadata", {}).get("seed")
            != int(expected_teacher_seed)
            for row in teacher_rows
        ):
            raise RuntimeError("Teacher-Dev row seed drift.")
        population = _validate_teacher_rows(
            teacher_rows=teacher_rows,
            full_dev_rows=normalized_rows,
        )
        evaluation_config = replace(
            experiment_config,
            dev_labels=labels_path,
            dev_manifest=source_manifest_path,
            expected_dev_rows=2_104,
            expected_dev_sha256=str(expected_teacher_dataset_sha256),
            expected_teacher_seed=int(expected_teacher_seed),
            encoding_cache_dir=Path(encoding_cache_dir),
            batch_size=int(batch_size),
            num_workers=0,
        )
        prepared = prepare_selector_datasets(
            teacher_rows,
            {"dev": tuple(str(row["sample_id"]) for row in teacher_rows)},
            evaluation_config,
            class_weight_split="dev",
            dataset_sha256=str(expected_teacher_dataset_sha256),
            label_spaces=("collapsed",),
            progress=progress,
        )
        dataset = prepared["collapsed"]["dev"]
        tokenizer = prepared["tokenizer"]
        loader = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=False,
            collate_fn=lambda batch: collate_selector_batch(
                batch,
                int(tokenizer.pad_token_id),
            ),
            num_workers=0,
        )
        runtime_device = torch.device(device)
        metric_rows: list[dict[str, Any]] = []
        for system in SYSTEM_ORDER:
            model = load_crf_inference_model(
                normalized_checkpoints[system],
                model_name=evaluation_config.model_name,
                model_revision=evaluation_config.model_revision,
                num_labels=len(LABEL_TO_ID),
                device=runtime_device,
            )
            try:
                observed = evaluate_history_selector(
                    model,
                    loader,
                    runtime_device,
                    progress=progress,
                    progress_desc=f"Teacher Dev {system}",
                )
            finally:
                del model
                gc.collect()
                if runtime_device.type == "cuda":
                    torch.cuda.empty_cache()
            row = {
                "system": system,
                "checkpoint_sha256": normalized_hashes[system],
                "n": len(teacher_rows),
                **{column: observed[column] for column in METRIC_COLUMNS},
            }
            row["visible_gold_keep_wordpieces"] = int(
                row["keep_tp"] + row["keep_fn"]
            )
            row["visible_gold_drop_wordpieces"] = int(
                row["token_total"]
                - row["visible_gold_keep_wordpieces"]
            )
            row["visible_gold_keep_share"] = float(
                row["visible_gold_keep_wordpieces"]
                / row["token_total"]
            )
            metric_rows.append(row)
        metrics = pd.DataFrame(metric_rows)
        _validate_outputs(
            metrics=metrics,
            population=population,
            expected_checkpoint_sha256=normalized_hashes,
        )
        _write_frame(data_by_name["metrics.csv"], metrics)
        _write_frame(data_by_name["population.csv"], population)
        manifest = {
            **identity,
            "checkpoint_manifest_sha256": checkpoint_manifest_hashes,
            "metrics_rows": len(metrics),
            "population_rows": len(population),
            "files": {
                filename: _sha256_file(path)
                for filename, path in data_by_name.items()
            },
            "complete": True,
            "teacher_api_called": False,
            "training_performed": False,
            "retrieval_performed": False,
        }
        _write_json(manifest_path, manifest)
        reused = False

    _validate_outputs(
        metrics=metrics,
        population=population,
        expected_checkpoint_sha256=normalized_hashes,
    )
    return TeacherDevAgreementResult(
        metrics=metrics,
        population=population,
        manifest=dict(manifest),
        manifest_path=manifest_path,
        data_paths=data_paths,
        reused=reused,
    )


def _normalize_system_paths(
    values: Mapping[str, Path | str],
    *,
    label: str,
) -> dict[str, Path]:
    normalized = {
        str(system): Path(path) for system, path in values.items()
    }
    if set(normalized) != set(SYSTEM_ORDER):
        raise ValueError(f"{label} must contain exactly {SYSTEM_ORDER}.")
    return {system: normalized[system] for system in SYSTEM_ORDER}


def _validate_full_dev_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    population_sha256: str,
) -> dict[str, Any]:
    ids = [str(row["sample_id"]) for row in rows]
    if len(rows) != 2_514 or len(set(ids)) != len(ids):
        raise RuntimeError("Full TopiOCQA-Dev population is not canonical.")
    observed_population_hash = _sha256_text("\n".join(sorted(ids)))
    if observed_population_hash != population_sha256:
        raise RuntimeError("Full TopiOCQA-Dev population hash drift.")
    depth_by_id = {
        str(row["sample_id"]): int(row["history_len"]) for row in rows
    }
    counts = {
        "full_dev_queries": len(rows),
        "history_depth_0_queries": sum(
            depth == 0 for depth in depth_by_id.values()
        ),
        "history_depth_1_queries": sum(
            depth == 1 for depth in depth_by_id.values()
        ),
        "teacher_eligible_queries": sum(
            depth >= 2 for depth in depth_by_id.values()
        ),
    }
    if counts != {
        "full_dev_queries": 2_514,
        "history_depth_0_queries": 205,
        "history_depth_1_queries": 205,
        "teacher_eligible_queries": 2_104,
    }:
        raise RuntimeError(f"Unexpected TopiOCQA-Dev depths: {counts}")
    eligible_ids = sorted(
        sample_id
        for sample_id, depth in depth_by_id.items()
        if depth >= 2
    )
    return {
        **counts,
        "teacher_eligible_sample_id_sha256": _sha256_text(
            "\n".join(eligible_ids)
        ),
        "full_dev_rows_sha256": _canonical_sha256(list(rows)),
    }


def _validate_teacher_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_dataset_sha256: str,
    expected_protocol_hash: str,
    expected_seed: int,
) -> None:
    expected = {
        "dataset": "TopiOCQA dev",
        "eligible_queries": 2_104,
        "model": "gpt-5.4",
        "temperature": 0.0,
        "seed": int(expected_seed),
        "protocol_hash": expected_protocol_hash,
        "dataset_sha256": expected_dataset_sha256,
        "error_count": 0,
        "labels": list(TEACHER_LABELS),
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise RuntimeError(
                f"Teacher-Dev manifest drift at {field}: "
                f"{manifest.get(field)!r}"
            )


def _validate_checkpoint_manifest(
    *,
    system: str,
    manifest: Mapping[str, Any],
    expected_checkpoint_sha256: str,
    model_name: str,
    model_revision: str,
) -> None:
    if manifest.get("complete") is not True:
        raise RuntimeError(f"Incomplete checkpoint run: {system}")
    config = manifest.get("config", {})
    expected_common = {
        "label_space": "collapsed",
        "adaptation": "full",
        "model_name": model_name,
        "model_revision": model_revision,
    }
    for field, value in expected_common.items():
        if config.get(field) != value:
            raise RuntimeError(
                f"Checkpoint recipe drift for {system}/{field}."
            )
    if system == "pretrained":
        if (
            config.get("decoder") != "crf"
            or manifest.get("checkpoint_sha256")
            != expected_checkpoint_sha256
        ):
            raise RuntimeError("Pretrained checkpoint contract drift.")
    else:
        expected_arm = {
            "imitation_control_e6": "imitation_control",
            "bm25_treatment_e6": "bm25_treatment",
        }[system]
        epoch_six = manifest.get("epoch_checkpoints", {}).get("6", {})
        if (
            config.get("arm") != expected_arm
            or config.get("decoder") != "pytorch_crf"
            or int(config.get("epochs", -1)) != 6
            or epoch_six.get("sha256") != expected_checkpoint_sha256
        ):
            raise RuntimeError(f"Epoch-6 checkpoint contract drift: {system}")


def _validate_teacher_rows(
    *,
    teacher_rows: Sequence[Mapping[str, Any]],
    full_dev_rows: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    row_by_id = {str(row["sample_id"]): row for row in full_dev_rows}
    teacher_ids = [str(row["sample_id"]) for row in teacher_rows]
    eligible_ids = {
        sample_id
        for sample_id, row in row_by_id.items()
        if int(row["history_len"]) >= 2
    }
    if (
        len(teacher_rows) != 2_104
        or len(set(teacher_ids)) != len(teacher_ids)
        or set(teacher_ids) != eligible_ids
    ):
        raise RuntimeError(
            "Teacher rows do not equal the depth>=2 Full-Dev subset."
        )

    history_turns = source_tokens = source_keep_tokens = 0
    multi_label_tokens = valid_spans = invalid_spans = 0
    observed_labels: set[str] = set()
    depths: list[int] = []
    for teacher_row in teacher_rows:
        sample_id = str(teacher_row["sample_id"])
        source_row = row_by_id[sample_id]
        for field in (
            "conv_id",
            "turn_id",
            "history_len",
            "current_query",
        ):
            if teacher_row[field] != source_row[field]:
                raise RuntimeError(
                    f"Teacher/full-dev alignment drift: {sample_id}/{field}"
                )
        teacher_history = list(teacher_row.get("history", []))
        source_history = list(source_row.get("history", []))
        if len(teacher_history) != len(source_history):
            raise RuntimeError(f"History length drift for {sample_id}.")
        depths.append(len(teacher_history))
        history_turns += len(teacher_history)
        for teacher_turn, source_turn in zip(
            teacher_history,
            source_history,
            strict=True,
        ):
            for field in ("turn_id", "question", "answer"):
                if teacher_turn[field] != source_turn[field]:
                    raise RuntimeError(
                        f"History alignment drift: {sample_id}/{field}"
                    )
            for token_field in ("question_tokens", "answer_tokens"):
                for token in teacher_turn.get(token_field, []):
                    labels = [str(label) for label in token.get("labels", [])]
                    source_tokens += 1
                    source_keep_tokens += int(bool(labels))
                    multi_label_tokens += int(len(labels) > 1)
                    observed_labels.update(labels)
        valid_spans += len(teacher_row.get("spans", []))
        invalid_spans += len(teacher_row.get("invalid_spans", []))

    if observed_labels != set(TEACHER_LABELS):
        raise RuntimeError(
            f"Unexpected Teacher label set: {sorted(observed_labels)}"
        )
    return pd.DataFrame(
        [
            {
                "full_dev_queries": len(full_dev_rows),
                "teacher_labeled_queries": len(teacher_rows),
                "full_dev_coverage": len(teacher_rows)
                / len(full_dev_rows),
                "history_depth_min": min(depths),
                "history_depth_max": max(depths),
                "history_turns": history_turns,
                "source_tokens": source_tokens,
                "source_keep_tokens": source_keep_tokens,
                "source_keep_share": source_keep_tokens
                / source_tokens,
                "multi_label_tokens": multi_label_tokens,
                "valid_spans": valid_spans,
                "invalid_spans": invalid_spans,
            }
        ]
    )


def _validate_manifest(
    *,
    manifest: Mapping[str, Any],
    identity: Mapping[str, Any],
    data_by_name: Mapping[str, Path],
) -> None:
    for field, expected in identity.items():
        if manifest.get(field) != expected:
            raise RuntimeError(
                f"Teacher-agreement manifest drift at {field}."
            )
    if manifest.get("complete") is not True:
        raise RuntimeError("Teacher-agreement manifest is incomplete.")
    recorded_files = manifest.get("files", {})
    if set(recorded_files) != set(data_by_name):
        raise RuntimeError("Teacher-agreement file set drift.")
    for filename, path in data_by_name.items():
        if (
            not path.is_file()
            or _sha256_file(path) != recorded_files[filename]
        ):
            raise RuntimeError(f"Teacher-agreement artifact drift: {path}")


def _validate_outputs(
    *,
    metrics: pd.DataFrame,
    population: pd.DataFrame,
    expected_checkpoint_sha256: Mapping[str, str],
) -> None:
    required_metrics = {
        "system",
        "checkpoint_sha256",
        "n",
        *METRIC_COLUMNS,
        "visible_gold_keep_wordpieces",
        "visible_gold_drop_wordpieces",
        "visible_gold_keep_share",
    }
    if not required_metrics.issubset(metrics.columns):
        raise RuntimeError("Teacher-agreement metrics schema is incomplete.")
    numeric_columns = (
        *METRIC_COLUMNS,
        "visible_gold_keep_wordpieces",
        "visible_gold_drop_wordpieces",
        "visible_gold_keep_share",
    )
    if any(
        not math.isfinite(float(value))
        for column in numeric_columns
        for value in metrics[column]
    ):
        raise RuntimeError("Teacher-agreement metrics contain NaN or Inf.")
    if (
        len(metrics) != len(SYSTEM_ORDER)
        or tuple(metrics["system"].astype(str)) != SYSTEM_ORDER
        or set(metrics["n"].astype(int)) != {2_104}
        or len(population) != 1
    ):
        raise RuntimeError("Teacher-agreement output shape is invalid.")
    observed_checkpoints = dict(
        zip(
            metrics["system"].astype(str),
            metrics["checkpoint_sha256"].astype(str),
            strict=True,
        )
    )
    if observed_checkpoints != dict(expected_checkpoint_sha256):
        raise RuntimeError("Teacher-agreement checkpoint column drift.")
    integer_columns = (
        "keep_tp",
        "keep_fp",
        "keep_fn",
        "token_correct",
        "token_total",
        "visible_gold_keep_wordpieces",
        "visible_gold_drop_wordpieces",
    )
    if any((metrics[column] < 0).any() for column in integer_columns):
        raise RuntimeError("Teacher-agreement counts must be non-negative.")
    if (metrics["token_total"] <= 0).any():
        raise RuntimeError("Teacher-agreement token denominator is empty.")
    expected_visible_gold = metrics["keep_tp"] + metrics["keep_fn"]
    if not (
        metrics["visible_gold_keep_wordpieces"].astype(int)
        == expected_visible_gold.astype(int)
    ).all():
        raise RuntimeError("Visible Teacher KEEP count drift.")
    if (
        metrics["token_total"].nunique() != 1
        or metrics["visible_gold_keep_wordpieces"].nunique() != 1
    ):
        raise RuntimeError("Gold denominator differs between checkpoints.")
    for row in metrics.itertuples(index=False):
        tp = int(row.keep_tp)
        fp = int(row.keep_fp)
        fn = int(row.keep_fn)
        token_total = int(row.token_total)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        expected_accuracy = int(row.token_correct) / token_total
        expected_keep_share = (tp + fn) / token_total
        observed_values = (
            float(row.keep_precision),
            float(row.keep_recall),
            float(row.keep_f1),
            float(row.token_accuracy),
            float(row.visible_gold_keep_share),
        )
        expected_values = (
            precision,
            recall,
            f1,
            expected_accuracy,
            expected_keep_share,
        )
        if any(
            abs(observed - expected) > 1e-12
            for observed, expected in zip(
                observed_values,
                expected_values,
                strict=True,
            )
        ):
            raise RuntimeError(
                f"Teacher-agreement metric arithmetic drift: {row.system}"
            )
    probability_columns = (
        "token_accuracy",
        "keep_precision",
        "keep_recall",
        "keep_f1",
        "span_precision",
        "span_recall",
        "span_f1",
        "visible_gold_keep_share",
    )
    if any(
        ((metrics[column] < 0.0) | (metrics[column] > 1.0)).any()
        for column in probability_columns
    ):
        raise RuntimeError("Teacher-agreement probability outside [0, 1].")
    population_row = population.iloc[0]
    expected_population = {
        "full_dev_queries": 2_514,
        "teacher_labeled_queries": 2_104,
        "history_depth_min": 2,
        "history_depth_max": 15,
        "history_turns": 14_351,
        "source_tokens": 248_329,
        "source_keep_tokens": 28_842,
        "multi_label_tokens": 545,
        "valid_spans": 7_110,
        "invalid_spans": 43,
    }
    for field, expected in expected_population.items():
        if int(population_row[field]) != expected:
            raise RuntimeError(
                f"Teacher-agreement population drift at {field}."
            )


def _canonical_sha256(value: Any) -> str:
    return _sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
    )


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str)
        + "\n",
        encoding="utf-8",
    )
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(path)


__all__ = [
    "PROTOCOL",
    "SYSTEM_ORDER",
    "TeacherDevAgreementResult",
    "load_or_compute_topiocqa_teacher_dev_agreement",
]
