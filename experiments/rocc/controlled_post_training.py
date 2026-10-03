"""Controlled hard-relabel post-training for the ROCC CRF selector.

This module implements the canonical Notebook 06 experiment.  Both
training arms start from the same Notebook 05 checkpoint and use the
same weighted token CE plus CRF NLL.  The sole treatment difference is
whether positive Gold-BM25 evidence from non-stopwords may turn Teacher
DROP labels into KEEP labels.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel

from .cached_itercqr import (
    CachedIterCQRBM25Config,
    CachedIterCQRBM25Pipeline,
)
from .candidates import is_lexical_stopword
from .crf_sampling import predict_crf_history_candidates
from .evaluation import (
    normalize_docids,
    reciprocal_rank_fusion,
    retrieval_metrics,
)
from .history_selector import (
    LABEL_TO_ID,
    EncodedSelectorDataset,
    EncodedSelectorExample,
    EncoderCrfHistorySelector,
    HistorySelectorConfig,
    collate_selector_batch,
    evaluate_history_selector,
    fit_selector_model,
    seed_selector_training,
    selector_class_stats,
    selector_parameter_counts,
    write_selected_histories_jsonl,
)
from .post_training import (
    GoldBM25TokenScoreResult,
    GoldScoreSelectorDataset,
)
from .progress import get_tqdm
from .selector_experiment import (
    SelectorExperimentConfig,
    compare_selector_metric,
    run_selector_retrieval_evaluation,
)
from .teacher_evaluation import (
    TEACHER_METRIC_COLUMNS,
    compressed_teacher_history,
)


CONTROLLED_POST_TRAINING_ARMS = (
    "imitation_control",
    "bm25_treatment",
)
CONTROLLED_POST_TRAINING_RECIPE_VERSION = (
    "hard_relabel_nltk_nostop_controlled_ce_crf_v2"
)
CONTROLLED_POST_TRAINING_SCHEMA_VERSION = 2
CONTROLLED_TARGET_HEADROOM_SCHEMA_VERSION = 3
CONTROLLED_TEACHER_B64_REFERENCE = {
    "MRR": 0.17035112014443654,
    "nDCG@3": 0.16071315434523956,
    "R@10": 0.2916666666666667,
    "R@100": 0.5033333333333333,
    "R@1000": 0.6666666666666666,
}
CONTROLLED_POST_TRAINING_ROUTE_ORDER = (
    "viterbi",
    "ffbs2_rrf10",
    "viterbi_ffbs2_rrf10",
)
CONTROLLED_POST_TRAINING_BOOTSTRAP_SEEDS = {
    (
        comparison,
        route,
    ): 13 + comparison_index * len(CONTROLLED_POST_TRAINING_ROUTE_ORDER)
    + route_index
    for comparison_index, comparison in enumerate(
        (
            "treatment_minus_control",
            "control_minus_pretrained",
            "treatment_minus_pretrained",
        )
    )
    for route_index, route in enumerate(
        CONTROLLED_POST_TRAINING_ROUTE_ORDER
    )
}


@dataclass(frozen=True)
class ControlledPostTrainingConfig:
    """Frozen configuration of the controlled Notebook 06 experiment."""

    start_checkpoint: Path | str
    architecture_decision_path: Path | str
    fusion_decision_path: Path | str
    result_dir: Path | str
    expected_start_checkpoint_sha256: str
    expected_vehicle_sample_id_sha256: str
    expected_architecture_decision_sha256: str
    expected_fusion_decision_sha256: str
    epochs: int = 6
    batch_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_clip_norm: float = 1.0
    seed: int = 13
    budget: int = 64
    num_ffbs_samples: int = 2
    sampling_seed: int = 13
    rrf_k: int = 10
    k1: float = 0.9
    b: float = 0.4
    history_flush_interval: int = 50

    def __post_init__(self) -> None:
        for name in (
            "start_checkpoint",
            "architecture_decision_path",
            "fusion_decision_path",
            "result_dir",
        ):
            object.__setattr__(self, name, Path(getattr(self, name)))
        if self.epochs != 6:
            raise ValueError("Canonical Notebook 06 requires six epochs.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.budget != 64:
            raise ValueError("Canonical Notebook 06 requires budget 64.")
        if self.num_ffbs_samples != 2:
            raise ValueError("Canonical Notebook 06 requires two FFBS samples.")
        if self.rrf_k != 10:
            raise ValueError("Canonical Notebook 06 requires RRF10.")
        if self.history_flush_interval < 1:
            raise ValueError("history_flush_interval must be positive.")

    @property
    def score_path(self) -> Path:
        """Reuse the durable score artifact already created by NB06."""

        return self.result_dir / "gold_bm25_token_scores.jsonl"

    @property
    def experiment_dir(self) -> Path:
        """Keep the new recipe isolated from obsolete NB06 artifacts."""

        return self.result_dir / "controlled_hard_relabel_nostop_e6"


@dataclass(frozen=True)
class ControlledPostTrainingDatasets:
    """Original and hard-corrected collapsed-BIO datasets."""

    control_full: EncodedSelectorDataset
    treatment_full: EncodedSelectorDataset
    control_monitor: EncodedSelectorDataset
    treatment_monitor: EncodedSelectorDataset
    class_weights: tuple[float, ...]
    manifest: Mapping[str, Any]
    manifest_path: Path

    def training_dataset(self, arm: str) -> EncodedSelectorDataset:
        if arm == "imitation_control":
            return self.control_full
        if arm == "bm25_treatment":
            return self.treatment_full
        raise ValueError(f"Unknown controlled post-training arm: {arm}")


def validate_controlled_post_training_inputs(
    selector_config: SelectorExperimentConfig,
    config: ControlledPostTrainingConfig,
) -> dict[str, Any]:
    """Validate the frozen Notebook 05 hand-off and NB06 protocol."""

    checkpoint_sha256 = _sha256_file(config.start_checkpoint)
    architecture_sha256 = _sha256_file(
        config.architecture_decision_path
    )
    fusion_sha256 = _sha256_file(config.fusion_decision_path)
    if checkpoint_sha256 != config.expected_start_checkpoint_sha256:
        raise ValueError("NB05 start-checkpoint SHA-256 does not match.")
    if architecture_sha256 != config.expected_architecture_decision_sha256:
        raise ValueError("NB05 architecture decision is not byte-identical.")
    if fusion_sha256 != config.expected_fusion_decision_sha256:
        raise ValueError("NB05 fusion decision is not byte-identical.")

    architecture = _read_json(config.architecture_decision_path)
    fusion = _read_json(config.fusion_decision_path)
    if (
        architecture.get("locked") is not True
        or architecture.get("selected_arm") != "collapsed_crf"
        or architecture.get("label_space") != "collapsed"
        or architecture.get("decoder") != "crf"
        or int(architecture.get("budget", -1)) != config.budget
        or architecture.get("selector_checkpoint_sha256")
        != checkpoint_sha256
        or architecture.get("vehicle_sample_id_sha256")
        != config.expected_vehicle_sample_id_sha256
    ):
        raise ValueError("NB05 architecture decision is not the frozen target.")
    selected_parameters = fusion.get("selected_parameters", {})
    if (
        fusion.get("locked") is not True
        or fusion.get("selected_method") != "rrf10"
        or int(fusion.get("budget", -1)) != config.budget
        or int(selected_parameters.get("rrf_k", -1)) != config.rrf_k
        or fusion.get("views") != ["I", "R", "D"]
        or fusion.get("selector_checkpoint_sha256")
        != checkpoint_sha256
        or fusion.get("architecture_decision_sha256")
        != architecture_sha256
        or fusion.get("vehicle_sample_id_sha256")
        != config.expected_vehicle_sample_id_sha256
    ):
        raise ValueError("NB05 fusion decision is not the frozen RRF10 rule.")

    start_manifest = _read_json(
        config.start_checkpoint.parent / "manifest.json"
    )
    start_recipe = start_manifest.get("config", {})
    if (
        start_manifest.get("complete") is not True
        or start_manifest.get("checkpoint_sha256") != checkpoint_sha256
        or start_recipe.get("label_space") != "collapsed"
        or start_recipe.get("decoder") != "crf"
        or start_recipe.get("adaptation") != "full"
        or start_recipe.get("training_recipe_version")
        != "independent_hf_full_v2"
        or float(start_recipe.get("token_loss_weight", -1.0)) != 1.0
        or float(start_recipe.get("crf_loss_weight", -1.0)) != 1.0
        or start_recipe.get("model_name") != selector_config.model_name
        or start_recipe.get("model_revision")
        != selector_config.model_revision
        or start_recipe.get("dataset_sha256")
        != selector_config.expected_train_sha256
    ):
        raise ValueError("NB05 checkpoint manifest does not match NB06.")
    if selector_config.dev_labels is not None or (
        selector_config.dev_manifest is not None
    ):
        raise ValueError("Notebook 06 must not configure a Dev source.")

    return {
        "start_checkpoint": str(config.start_checkpoint.resolve()),
        "start_checkpoint_sha256": checkpoint_sha256,
        "architecture_decision_sha256": architecture_sha256,
        "fusion_decision_sha256": fusion_sha256,
        "vehicle_sample_id_sha256": (
            config.expected_vehicle_sample_id_sha256
        ),
        "reference_fusion_mrr": float(fusion["selected_mrr"]),
        "training_recipe": CONTROLLED_POST_TRAINING_RECIPE_VERSION,
        "epochs": config.epochs,
        "budget": config.budget,
        "ffbs_samples": config.num_ffbs_samples,
        "candidate_rrf_k": config.rrf_k,
        "pytorch_crf_version": importlib.metadata.version("pytorch-crf"),
    }


def prepare_controlled_post_training_datasets(
    *,
    base_full: EncodedSelectorDataset,
    base_monitor: EncodedSelectorDataset,
    gold_dataset: GoldScoreSelectorDataset,
    class_weights: Sequence[float],
    score_result: GoldBM25TokenScoreResult,
    config: ControlledPostTrainingConfig,
    progress: bool = True,
) -> ControlledPostTrainingDatasets:
    """Build original targets and monotonic BM25-corrected targets."""

    if len(base_full) != len(gold_dataset):
        raise ValueError("Full selector and Gold-score datasets differ.")
    base_by_id = {
        str(example.sample_id): example for example in base_full.examples
    }
    gold_by_id = {
        str(example.selector.sample_id): example
        for example in gold_dataset.examples
    }
    if set(base_by_id) != set(gold_by_id):
        raise ValueError("Full selector and Gold-score sample IDs differ.")
    if len(class_weights) != len(LABEL_TO_ID):
        raise ValueError("Collapsed class weights have the wrong size.")
    if gold_dataset.stats.get("stopword_filter") != "embedded_nltk_english":
        raise ValueError(
            "Controlled BM25 treatment requires the embedded NLTK "
            "stopword filter."
        )

    treatment_examples: list[EncodedSelectorExample] = []
    changed_queries = 0
    flips = 0
    positive_wordpieces = 0
    tqdm = get_tqdm()
    for original in tqdm(
        base_full.examples,
        total=len(base_full),
        desc="derive BM25-corrected Teacher labels",
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    ):
        gold_example = gold_by_id[str(original.sample_id)]
        converted, stats = _hard_relabel_example(
            original,
            gold_example.gold_scores,
        )
        treatment_examples.append(converted)
        changed_queries += int(stats["changed"])
        flips += int(stats["flips"])
        positive_wordpieces += int(stats["positive_wordpieces"])

    treatment_full = EncodedSelectorDataset(treatment_examples)
    treatment_by_id = {
        str(example.sample_id): example
        for example in treatment_examples
    }
    monitor_ids = [
        str(example.sample_id) for example in base_monitor.examples
    ]
    treatment_monitor = EncodedSelectorDataset(
        [treatment_by_id[sample_id] for sample_id in monitor_ids]
    )
    fixed_weights = tuple(float(value) for value in class_weights)
    original_stats = selector_class_stats(
        base_full,
        label_to_id=LABEL_TO_ID,
    )
    treatment_stats = selector_class_stats(
        treatment_full,
        label_to_id=LABEL_TO_ID,
    )
    original_keep = sum(
        int(original_stats["label_counts"][label])
        for label in ("B-KEEP", "I-KEEP")
    )
    treatment_keep = sum(
        int(treatment_stats["label_counts"][label])
        for label in ("B-KEEP", "I-KEEP")
    )
    history_wordpieces = int(original_stats["history_tokens"])
    control_target_sha256 = _label_dataset_sha256(base_full)
    treatment_target_sha256 = _label_dataset_sha256(treatment_full)
    manifest = {
        "schema_version": CONTROLLED_POST_TRAINING_SCHEMA_VERSION,
        "complete": True,
        "recipe": CONTROLLED_POST_TRAINING_RECIPE_VERSION,
        "rule": (
            "Teacher_KEEP_OR_(BM25_score_raw_gt_0_AND_not_NLTK_stopword)"
        ),
        "stopword_source": "experiments.rocc.candidates.LEXICAL_STOPWORDS",
        "normalization_relevance": "none_at_zero_threshold",
        "teacher_keep_is_monotonic": True,
        "teacher_bio_tags_are_preserved": True,
        "new_keep_tag_rule": (
            "I-KEEP_if_previous_visible_output_tag_is_KEEP_else_B-KEEP"
        ),
        "queries": len(base_full),
        "monitor_queries": len(base_monitor),
        "monitor_sample_id_sha256": _sample_id_sha256(monitor_ids),
        "score_dataset_sha256": score_result.dataset_sha256,
        "control_target_sha256": control_target_sha256,
        "treatment_target_sha256": treatment_target_sha256,
        "changed_queries": changed_queries,
        "changed_query_share": changed_queries / len(base_full),
        "drop_to_keep_wordpiece_flips": flips,
        "positive_bm25_wordpieces": positive_wordpieces,
        "positive_bm25_source_tokens": int(
            gold_dataset.stats["positive_source_tokens"]
        ),
        "retained_positive_bm25_source_tokens": int(
            gold_dataset.stats["retained_positive_source_tokens"]
        ),
        "excluded_stopword_source_tokens": int(
            gold_dataset.stats["excluded_stopword_source_tokens"]
        ),
        "visible_history_wordpieces": history_wordpieces,
        "teacher_keep_wordpieces": original_keep,
        "teacher_keep_share": original_keep / history_wordpieces,
        "treatment_keep_wordpieces": treatment_keep,
        "treatment_keep_share": treatment_keep / history_wordpieces,
        "control_class_stats": original_stats,
        "treatment_class_stats": treatment_stats,
        "fixed_class_weights": list(fixed_weights),
        "class_weight_source": "unchanged_NB05_collapsed_full_train",
        "only_target_labels_differ_between_training_arms": True,
    }
    manifest_path = config.experiment_dir / "targets" / "manifest.json"
    _ensure_locked_json(manifest_path, manifest)
    if manifest["monitor_sample_id_sha256"] != (
        config.expected_vehicle_sample_id_sha256
    ):
        raise ValueError("Hard-relabel monitor is not the canonical gate.")
    return ControlledPostTrainingDatasets(
        control_full=base_full,
        treatment_full=treatment_full,
        control_monitor=base_monitor,
        treatment_monitor=treatment_monitor,
        class_weights=fixed_weights,
        manifest=manifest,
        manifest_path=manifest_path,
    )


def evaluate_controlled_teacher_targets(
    *,
    rows: Sequence[dict[str, Any]],
    samples: Sequence[Any],
    gold_by_sample: Mapping[str, Sequence[Any]],
    score_result: GoldBM25TokenScoreResult,
    selector_config: SelectorExperimentConfig,
    config: ControlledPostTrainingConfig,
    device: str | torch.device,
    progress: bool = True,
) -> dict[str, Any]:
    """Measure retrieval headroom in original versus corrected targets."""

    _validate_vehicle(rows, samples, gold_by_sample, config)
    output_dir = config.experiment_dir / "target_headroom"
    histories, selection_by_query = _build_raw_target_histories(
        rows=rows,
        score_result=score_result,
        progress=progress,
    )
    sample_order = [str(row["sample_id"]) for row in rows]
    for arm, selected in histories.items():
        write_selected_histories_jsonl(
            output_dir / "histories" / f"{arm}.jsonl",
            selected,
            sample_order=sample_order,
            progress=progress,
        )
    retrieval = run_selector_retrieval_evaluation(
        samples=samples,
        selector_histories=histories,
        gold_by_sample=gold_by_sample,
        config=selector_config,
        output_dir=output_dir / "R_itercqr",
        budgets=(config.budget,),
        include_recency=False,
        include_query_only=False,
        persist_runtime_cache_stats=True,
        device=device,
        progress=progress,
    )
    results = retrieval["evaluation"].pipeline_results.copy()
    original_row = retrieval["summary"].loc[
        retrieval["summary"]["arm"].eq("teacher_original")
        & retrieval["summary"]["budget"].eq(config.budget)
    ]
    if len(original_row) != 1:
        raise RuntimeError("Teacher target reference row is missing.")
    original = original_row.iloc[0]
    for metric, expected in CONTROLLED_TEACHER_B64_REFERENCE.items():
        if not np.isclose(
            float(original[metric]),
            expected,
            atol=1e-15,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Raw Teacher target no longer reproduces Notebook 04: "
                f"{metric}={float(original[metric])}, expected={expected}."
            )
    comparison_rows: list[dict[str, Any]] = []
    for index, metric in enumerate(TEACHER_METRIC_COLUMNS):
        comparison = compare_selector_metric(
            results,
            left_arm="teacher_bm25_corrected",
            right_arm="teacher_original",
            budget=config.budget,
            metric=metric,
            seed=30 + index,
            replicates=selector_config.bootstrap_replicates,
            progress=progress,
        )
        comparison_rows.append(comparison)
    comparisons = pd.DataFrame(comparison_rows)
    selection_summary = (
        selection_by_query.groupby("arm", observed=True, sort=False)
        .agg(
            queries=("sample_id", "size"),
            history_tokens=("history_tokens", "sum"),
            keep_tokens=("keep_tokens", "sum"),
            mean_selected_words=("selected_words", "mean"),
            empty_selections=("empty_selection", "sum"),
        )
        .reset_index()
    )
    selection_summary["keep_share"] = (
        selection_summary["keep_tokens"]
        / selection_summary["history_tokens"]
    )
    _write_frame(output_dir / "selection_by_query.csv", selection_by_query)
    _write_frame(output_dir / "selection_summary.csv", selection_summary)
    _write_frame(output_dir / "paired_comparisons.csv", comparisons)
    manifest = {
        "schema_version": CONTROLLED_TARGET_HEADROOM_SCHEMA_VERSION,
        "complete": True,
        "scope": "train_member_600_target_oracle",
        "budget": config.budget,
        "route": "full_raw_history_KEEP_to_IterCQR_to_BM25",
        "teacher_keep_rule": "any_Teacher_token_label",
        "treatment_rule": (
            "Teacher_KEEP_OR_(BM25_score_raw_gt_0_AND_not_NLTK_stopword)"
        ),
        "stopword_source": "experiments.rocc.candidates.LEXICAL_STOPWORDS",
        "preselection_model_projection": "none",
        "preselection_history_truncation": "none",
        "score_dataset_sha256": score_result.dataset_sha256,
        "teacher_b64_reference": CONTROLLED_TEACHER_B64_REFERENCE,
        "vehicle_sample_id_sha256": (
            config.expected_vehicle_sample_id_sha256
        ),
        "privileged_target_diagnostic": True,
        "model_evaluation": False,
        "summary_sha256": _sha256_file(
            output_dir / "R_itercqr" / "summary.csv"
        ),
        "comparisons_sha256": _sha256_file(
            output_dir / "paired_comparisons.csv"
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "histories": histories,
        "selection_by_query": selection_by_query,
        "selection_summary": selection_summary,
        "pipeline_results": results,
        "summary": retrieval["summary"],
        "comparisons": comparisons,
        "manifest": manifest,
    }


def evaluate_controlled_post_training_baseline(
    *,
    rows: Sequence[dict[str, Any]],
    samples: Sequence[Any],
    gold_by_sample: Mapping[str, Sequence[Any]],
    monitor_dataset: EncodedSelectorDataset,
    tokenizer: Any,
    class_weights: Sequence[float],
    selector_config: SelectorExperimentConfig,
    config: ControlledPostTrainingConfig,
    device: str | torch.device,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate the shared Notebook 05 start checkpoint as epoch zero."""

    runtime_device = torch.device(device)
    model = _load_controlled_model(
        checkpoint=config.start_checkpoint,
        class_weights=class_weights,
        selector_config=selector_config,
        config=config,
    ).to(runtime_device)
    loader = _selector_loader(
        monitor_dataset,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_workers=selector_config.num_workers,
        shuffle=False,
        seed=config.seed,
    )
    label_metrics = evaluate_history_selector(
        model,
        loader,
        runtime_device,
        progress=progress,
        progress_desc="pretrained train600 Teacher-label metrics",
    )
    epoch_evaluation = evaluate_controlled_post_training_epoch(
        system="pretrained",
        epoch=0,
        model=model,
        rows=rows,
        samples=samples,
        gold_by_sample=gold_by_sample,
        tokenizer=tokenizer,
        selector_config=selector_config,
        config=config,
        device=runtime_device,
        progress=progress,
    )
    del model
    if runtime_device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "system": "pretrained",
        "epoch": 0,
        "checkpoint": config.start_checkpoint,
        "checkpoint_sha256": config.expected_start_checkpoint_sha256,
        "label_metrics": label_metrics,
        "evaluation": epoch_evaluation,
    }


def run_controlled_post_training_arm(
    *,
    arm: str,
    datasets: ControlledPostTrainingDatasets,
    rows: Sequence[dict[str, Any]],
    samples: Sequence[Any],
    gold_by_sample: Mapping[str, Sequence[Any]],
    tokenizer: Any,
    selector_config: SelectorExperimentConfig,
    config: ControlledPostTrainingConfig,
    device: str | torch.device,
    batch_metrics_reporter: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    epoch_metrics_reporter: (
        Callable[[str, Mapping[str, Any]], None] | None
    ) = None,
    history_reporter: (
        Callable[[str, pd.DataFrame, pd.DataFrame], None] | None
    ) = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Train one six-epoch arm and evaluate all fixed routes per epoch."""

    if arm not in CONTROLLED_POST_TRAINING_ARMS:
        raise ValueError(f"Unknown controlled post-training arm: {arm}")
    _validate_vehicle(rows, samples, gold_by_sample, config)
    train_dataset = datasets.training_dataset(arm)
    if len(train_dataset) != int(selector_config.expected_train_rows):
        raise ValueError("Controlled post-training requires full train.")
    runtime_device = torch.device(device)
    output_dir = config.experiment_dir / "training" / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    target_sha256 = str(
        datasets.manifest[
            (
                "control_target_sha256"
                if arm == "imitation_control"
                else "treatment_target_sha256"
            )
        ]
    )
    run_config = {
        "schema_version": CONTROLLED_POST_TRAINING_SCHEMA_VERSION,
        "training_recipe_version": (
            CONTROLLED_POST_TRAINING_RECIPE_VERSION
        ),
        "arm": arm,
        "initialization_source": "NB05_collapsed_crf_checkpoint",
        "start_checkpoint_sha256": (
            config.expected_start_checkpoint_sha256
        ),
        "teacher_dataset_sha256": selector_config.expected_train_sha256,
        "score_dataset_sha256": datasets.manifest[
            "score_dataset_sha256"
        ],
        "target_sha256": target_sha256,
        "target_rule": (
            "unchanged_Teacher_labels"
            if arm == "imitation_control"
            else (
                "Teacher_KEEP_OR_(BM25_score_raw_gt_0_AND_not_NLTK_stopword)"
            )
        ),
        "model_name": selector_config.model_name,
        "model_revision": selector_config.model_revision,
        "label_space": "collapsed",
        "decoder": "pytorch_crf",
        "adaptation": "full",
        "loss": "weighted_token_CE_plus_token_normalized_CRF_NLL",
        "token_loss_weight": 1.0,
        "crf_loss_weight": 1.0,
        "gold_loss_weight": 0.0,
        "ce_conflict_mask": None,
        "class_weights": list(datasets.class_weights),
        "class_weight_source": "unchanged_NB05_collapsed_full_train",
        "epochs": config.epochs,
        "checkpoint_policy": "last_fixed_epoch_6",
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "gradient_clip_norm": config.gradient_clip_norm,
        "seed": config.seed,
        "monitor_scope": "train_member_600_descriptive_only",
        "monitor_budget": config.budget,
        "monitor_routes": list(CONTROLLED_POST_TRAINING_ROUTE_ORDER),
        "num_ffbs_samples": config.num_ffbs_samples,
        "sampling_seed": config.sampling_seed,
        "candidate_fusion": f"RRF{config.rrf_k}",
        "monitor_sample_id_sha256": (
            config.expected_vehicle_sample_id_sha256
        ),
        "monitor_affects_checkpoint_selection": False,
        "only_treatment_factor": "target_labels",
    }
    run_config["run_id"] = _mapping_sha256(run_config)
    _ensure_locked_json(output_dir / "config.json", run_config)
    reused = _completed_training_run(
        output_dir,
        run_config=run_config,
        epochs=config.epochs,
    )

    model = _load_controlled_model(
        checkpoint=config.start_checkpoint,
        class_weights=datasets.class_weights,
        selector_config=selector_config,
        config=config,
    ).to(runtime_device)
    train_loader = _selector_loader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_workers=selector_config.num_workers,
        shuffle=True,
        seed=config.seed,
    )
    monitor_loader = _selector_loader(
        datasets.control_monitor,
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        num_workers=selector_config.num_workers,
        shuffle=False,
        seed=config.seed,
    )

    loss_path = output_dir / "loss_history.csv"
    epoch_path = output_dir / "epoch_metrics.csv"
    batch_rows = _frame_records(_read_frame(loss_path))
    epoch_rows = _frame_records(_read_frame(epoch_path))

    def persist_history() -> tuple[pd.DataFrame, pd.DataFrame]:
        loss_frame = pd.DataFrame(batch_rows)
        epoch_frame = pd.DataFrame(epoch_rows)
        if not loss_frame.empty:
            loss_frame = (
                loss_frame.drop_duplicates("global_step", keep="last")
                .sort_values("global_step", kind="mergesort")
                .reset_index(drop=True)
            )
            _write_frame(loss_path, loss_frame)
        if not epoch_frame.empty:
            epoch_frame = (
                epoch_frame.drop_duplicates("epoch", keep="last")
                .sort_values("epoch", kind="mergesort")
                .reset_index(drop=True)
            )
            _write_frame(epoch_path, epoch_frame)
        return loss_frame, epoch_frame

    def report_batch(row: dict[str, Any]) -> None:
        payload = {"arm": arm, **dict(row)}
        batch_rows.append(payload)
        if int(row["batch"]) % config.history_flush_interval == 0:
            persist_history()
        if batch_metrics_reporter is not None:
            batch_metrics_reporter(arm, payload)

    def report_epoch(row: dict[str, Any]) -> None:
        payload = {"arm": arm, **dict(row)}
        epoch_rows.append(payload)
        persist_history()
        if epoch_metrics_reporter is not None:
            epoch_metrics_reporter(arm, payload)

    def evaluate_epoch(
        selected_model: torch.nn.Module,
        epoch: int,
    ) -> dict[str, Any]:
        label_metrics = evaluate_history_selector(
            selected_model,
            monitor_loader,
            runtime_device,
            progress=progress,
            progress_desc=(
                f"{arm} train600 Teacher-label metrics epoch {epoch}"
            ),
        )
        route_evaluation = evaluate_controlled_post_training_epoch(
            system=arm,
            epoch=epoch,
            model=selected_model,
            rows=rows,
            samples=samples,
            gold_by_sample=gold_by_sample,
            tokenizer=tokenizer,
            selector_config=selector_config,
            config=config,
            device=runtime_device,
            progress=progress,
        )
        summary = route_evaluation["summary"].set_index("route")
        return {
            **label_metrics,
            **{
                f"train600_{route}_{metric}": float(
                    summary.loc[route, metric]
                )
                for route in CONTROLLED_POST_TRAINING_ROUTE_ORDER
                for metric in TEACHER_METRIC_COLUMNS
            },
        }

    run = fit_selector_model(
        model=model,
        train_loader=train_loader,
        output_dir=output_dir,
        device=runtime_device,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        evaluate_fn=evaluate_epoch,
        checkpoint_policy="last",
        selection_metric=None,
        progress=progress,
        progress_desc=f"train {arm} CE+CRF",
        gradient_clip_norm=config.gradient_clip_norm,
        run_config=run_config,
        train_stats=(
            datasets.manifest[
                (
                    "control_class_stats"
                    if arm == "imitation_control"
                    else "treatment_class_stats"
                )
            ]
        ),
        dev_stats={},
        epoch_checkpoint_pattern="checkpoint_epoch_{epoch}.pt",
        record_learning_rate=True,
        resume=True,
        epoch_reporter=(
            lambda epoch, epochs, train_loss, metrics, improved: (
                f"epoch {epoch}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"Teacher_KEEP_F1={float(metrics['keep_f1']):.4f}, "
                f"Viterbi_MRR="
                f"{float(metrics['train600_viterbi_MRR']):.4f}, "
                f"FFBS2_MRR="
                f"{float(metrics['train600_ffbs2_rrf10_MRR']):.4f}, "
                f"K3_MRR="
                f"{float(metrics['train600_viterbi_ffbs2_rrf10_MRR']):.4f}"
            )
        ),
        batch_metrics_reporter=report_batch,
        epoch_metrics_reporter=report_epoch,
    )
    loss_frame, epoch_frame = persist_history()
    expected_steps = math.ceil(len(train_dataset) / config.batch_size) * (
        config.epochs
    )
    if len(loss_frame) != expected_steps or len(epoch_frame) != config.epochs:
        raise RuntimeError(
            f"Incomplete {arm} training history: "
            f"steps={len(loss_frame)}/{expected_steps}, "
            f"epochs={len(epoch_frame)}/{config.epochs}."
        )
    checkpoint_rows: dict[str, dict[str, str]] = {}
    for epoch in range(1, config.epochs + 1):
        checkpoint = output_dir / f"checkpoint_epoch_{epoch}.pt"
        if not checkpoint.exists():
            raise RuntimeError(f"Missing epoch checkpoint: {checkpoint}")
        checkpoint_rows[str(epoch)] = {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256_file(checkpoint),
        }
    final_epoch_checkpoint = (
        output_dir / f"checkpoint_epoch_{config.epochs}.pt"
    )
    final_checkpoint_sha256 = _sha256_file(run.checkpoint)
    if not _checkpoint_state_dicts_equal(
        run.checkpoint,
        final_epoch_checkpoint,
    ):
        raise RuntimeError("Final model is not the fixed epoch-6 checkpoint.")
    manifest = {
        "schema_version": CONTROLLED_POST_TRAINING_SCHEMA_VERSION,
        "complete": True,
        "config": run_config,
        "checkpoint": str(run.checkpoint.resolve()),
        "checkpoint_sha256": final_checkpoint_sha256,
        "epoch_checkpoints": checkpoint_rows,
        "duration_seconds": run.duration_seconds,
        "parameters": selector_parameter_counts(model),
        "loss_history_sha256": _sha256_file(loss_path),
        "epoch_metrics_sha256": _sha256_file(epoch_path),
    }
    _write_json(output_dir / "manifest.json", manifest)
    if history_reporter is not None:
        history_reporter(arm, loss_frame, epoch_frame)
    del model
    if runtime_device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "arm": arm,
        "run": run,
        "checkpoint": run.checkpoint,
        "checkpoint_sha256": final_checkpoint_sha256,
        "epoch_checkpoints": checkpoint_rows,
        "class_weights": list(datasets.class_weights),
        "loss_history": loss_frame,
        "epoch_metrics": epoch_frame,
        "manifest": manifest,
        "reused": reused,
    }


def evaluate_controlled_post_training_epoch(
    *,
    system: str,
    epoch: int,
    model: torch.nn.Module,
    rows: Sequence[dict[str, Any]],
    samples: Sequence[Any],
    gold_by_sample: Mapping[str, Sequence[Any]],
    tokenizer: Any,
    selector_config: SelectorExperimentConfig,
    config: ControlledPostTrainingConfig,
    device: torch.device,
    progress: bool = True,
) -> dict[str, Any]:
    """Evaluate Viterbi, two FFBS samples, and their fixed RRF10 views."""

    _validate_vehicle(rows, samples, gold_by_sample, config)
    output_dir = (
        config.experiment_dir
        / "train600"
        / str(system)
        / f"epoch_{int(epoch)}"
    )
    candidates = predict_crf_history_candidates(
        rows,
        model=model,
        tokenizer=tokenizer,
        device=device,
        config=HistorySelectorConfig(
            model_name=selector_config.model_name,
            max_length=selector_config.max_length,
            history_order=selector_config.history_order,
            batch_size=config.batch_size,
        ),
        batch_size=config.batch_size,
        num_samples=config.num_ffbs_samples,
        sampling_seed=config.sampling_seed,
        progress=progress,
        progress_desc=f"{system} epoch {epoch} Viterbi + 2 FFBS",
    )
    sample_order = [str(row["sample_id"]) for row in rows]
    for candidate, histories in candidates.items():
        write_selected_histories_jsonl(
            output_dir / "predictions" / f"{candidate}.jsonl",
            histories,
            sample_order=sample_order,
            progress=progress,
        )
    retrieval = run_selector_retrieval_evaluation(
        samples=samples,
        selector_histories=candidates,
        gold_by_sample=gold_by_sample,
        config=selector_config,
        output_dir=output_dir / "candidate_retrieval",
        budgets=(config.budget,),
        include_recency=False,
        include_query_only=False,
        persist_runtime_cache_stats=True,
        device=device,
        progress=progress,
    )
    pipeline_results = retrieval["evaluation"].pipeline_results.copy()
    candidate_evaluation = _evaluate_candidate_fusions(
        pipeline_results=pipeline_results,
        gold_by_sample=gold_by_sample,
        selector_config=selector_config,
        config=config,
        device=device,
        system=system,
        epoch=epoch,
        progress=progress,
    )
    by_query = candidate_evaluation["by_query"]
    union_by_query = candidate_evaluation["union_by_query"]
    union_summary = candidate_evaluation["union_summary"]
    union_patterns = candidate_evaluation["union_patterns"]
    summary = (
        by_query.groupby("route", observed=True, sort=False)
        .agg(
            n=("sample_id", "size"),
            **{
                metric: (metric, "mean")
                for metric in TEACHER_METRIC_COLUMNS
            },
        )
        .reset_index()
    )
    _write_frame(output_dir / "route_metrics_by_query.csv", by_query)
    _write_frame(output_dir / "route_summary.csv", summary)
    _write_frame(output_dir / "candidate_union_by_query.csv", union_by_query)
    _write_frame(output_dir / "candidate_union_summary.csv", union_summary)
    _write_frame(output_dir / "candidate_union_patterns.csv", union_patterns)
    manifest = {
        "schema_version": CONTROLLED_POST_TRAINING_SCHEMA_VERSION,
        "complete": True,
        "system": str(system),
        "epoch": int(epoch),
        "evaluation_scope": "train_member_600_descriptive_only",
        "budget": config.budget,
        "candidate_decoder": "pytorch_crf_exact_ffbs",
        "num_ffbs_samples": config.num_ffbs_samples,
        "sampling_seed": config.sampling_seed,
        "common_random_numbers": True,
        "fusion": f"RRF{config.rrf_k}",
        "routes": list(CONTROLLED_POST_TRAINING_ROUTE_ORDER),
        "vehicle_sample_id_sha256": (
            config.expected_vehicle_sample_id_sha256
        ),
        "route_metrics_sha256": _sha256_file(
            output_dir / "route_metrics_by_query.csv"
        ),
        "route_summary_sha256": _sha256_file(
            output_dir / "route_summary.csv"
        ),
        "candidate_union_summary_sha256": _sha256_file(
            output_dir / "candidate_union_summary.csv"
        ),
        "candidate_union_is_oracle_diagnostic": True,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return {
        "system": system,
        "epoch": int(epoch),
        "candidates": candidates,
        "pipeline_results": pipeline_results,
        "by_query": by_query,
        "summary": summary,
        "union_by_query": union_by_query,
        "union_summary": union_summary,
        "union_patterns": union_patterns,
        "manifest": manifest,
        "output_dir": output_dir,
    }


def compare_controlled_post_training_runs(
    *,
    baseline: Mapping[str, Any],
    runs: Mapping[str, Mapping[str, Any]],
    datasets: ControlledPostTrainingDatasets,
    selector_config: SelectorExperimentConfig,
    config: ControlledPostTrainingConfig,
    progress: bool = True,
) -> dict[str, Any]:
    """Create fixed epoch trajectories, endpoint CIs, and a hand-off."""

    if set(runs) != set(CONTROLLED_POST_TRAINING_ARMS):
        raise ValueError("Both controlled post-training arms are required.")
    by_query_frames: list[pd.DataFrame] = []
    baseline_frame = baseline["evaluation"]["by_query"].copy()
    baseline_frame["system"] = "pretrained"
    baseline_frame["epoch"] = 0
    by_query_frames.append(baseline_frame)
    for arm in CONTROLLED_POST_TRAINING_ARMS:
        for epoch in range(1, config.epochs + 1):
            path = (
                config.experiment_dir
                / "train600"
                / arm
                / f"epoch_{epoch}"
                / "route_metrics_by_query.csv"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing epoch evaluation: {path}")
            frame = pd.read_csv(path, float_precision="round_trip")
            frame["system"] = arm
            frame["epoch"] = epoch
            by_query_frames.append(frame)
    all_by_query = pd.concat(
        by_query_frames,
        ignore_index=True,
        sort=False,
    )
    trajectory = (
        all_by_query.groupby(
            ["system", "epoch", "route"],
            observed=True,
            sort=False,
        )
        .agg(
            n=("sample_id", "size"),
            **{
                metric: (metric, "mean")
                for metric in TEACHER_METRIC_COLUMNS
            },
        )
        .reset_index()
    )

    comparison_specs = (
        (
            "treatment_minus_control",
            "bm25_treatment",
            "imitation_control",
        ),
        (
            "control_minus_pretrained",
            "imitation_control",
            "pretrained",
        ),
        (
            "treatment_minus_pretrained",
            "bm25_treatment",
            "pretrained",
        ),
    )
    comparison_rows: list[dict[str, Any]] = []
    for comparison, left, right in comparison_specs:
        for route in CONTROLLED_POST_TRAINING_ROUTE_ORDER:
            frames = []
            for system in (left, right):
                epoch = 0 if system == "pretrained" else config.epochs
                selected = all_by_query.loc[
                    all_by_query["system"].eq(system)
                    & all_by_query["epoch"].eq(epoch)
                    & all_by_query["route"].eq(route)
                ].copy()
                selected["arm"] = system
                frames.append(selected)
            paired = pd.concat(frames, ignore_index=True, sort=False)
            result = compare_selector_metric(
                paired,
                left_arm=left,
                right_arm=right,
                budget=config.budget,
                metric="MRR",
                seed=CONTROLLED_POST_TRAINING_BOOTSTRAP_SEEDS[
                    (comparison, route)
                ],
                replicates=selector_config.bootstrap_replicates,
                progress=progress,
            )
            comparison_rows.append(
                {
                    "registered_comparison": comparison,
                    "route": route,
                    **result,
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    output_dir = config.experiment_dir / "comparison"
    _write_frame(output_dir / "metrics_by_query.csv", all_by_query)
    _write_frame(output_dir / "trajectory.csv", trajectory)
    _write_frame(output_dir / "paired_endpoint_comparisons.csv", comparisons)

    primary = comparisons.loc[
        comparisons["registered_comparison"].eq(
            "treatment_minus_control"
        )
        & comparisons["route"].eq("ffbs2_rrf10")
    ].iloc[0]
    endpoint_checkpoints = {
        arm: {
            "path": str(Path(runs[arm]["checkpoint"]).resolve()),
            "sha256": str(runs[arm]["checkpoint_sha256"]),
            "epoch": config.epochs,
        }
        for arm in CONTROLLED_POST_TRAINING_ARMS
    }
    handoff = {
        "schema_version": CONTROLLED_POST_TRAINING_SCHEMA_VERSION,
        "locked_protocol": True,
        "selection_performed": False,
        "independent_evaluation_status": "pending",
        "training_recipe": CONTROLLED_POST_TRAINING_RECIPE_VERSION,
        "start_checkpoint_sha256": (
            config.expected_start_checkpoint_sha256
        ),
        "target_manifest_sha256": _sha256_file(datasets.manifest_path),
        "vehicle_sample_id_sha256": (
            config.expected_vehicle_sample_id_sha256
        ),
        "vehicle_scope": "train_member_600_descriptive_only",
        "fixed_endpoint_epoch": config.epochs,
        "endpoint_checkpoints": endpoint_checkpoints,
        "primary_mechanism_contrast": {
            "comparison": "bm25_treatment_minus_imitation_control",
            "route": "two_FFBS_R_rankings_fused_with_RRF10",
            "metric": "MRR",
            "delta": float(primary["delta"]),
            "ci95_low": float(primary["ci95_low"]),
            "ci95_high": float(primary["ci95_high"]),
            "seed": int(primary["seed"]),
            "replicates": int(primary["replicates"]),
        },
        "later_independent_system_protocol": {
            "I": "frozen_NB05_recency_to_IterCQR_to_BM25",
            "R": "Viterbi_plus_two_FFBS_rankings_fused_with_RRF10",
            "D": "frozen_NB05_Viterbi_direct_BM25",
            "outer_fusion": "RRF10_I_R_D",
            "promotion_rule": (
                "treatment_must_significantly_beat_control_on_the_"
                "independent_full_system"
            ),
        },
        "bootstrap_seeds": {
            f"{comparison}:{route}": seed
            for (comparison, route), seed in (
                CONTROLLED_POST_TRAINING_BOOTSTRAP_SEEDS.items()
            )
        },
        "trajectory_sha256": _sha256_file(output_dir / "trajectory.csv"),
        "comparisons_sha256": _sha256_file(
            output_dir / "paired_endpoint_comparisons.csv"
        ),
    }
    _write_json(output_dir / "independent_evaluation_handoff.json", handoff)
    return {
        "metrics_by_query": all_by_query,
        "trajectory": trajectory,
        "comparisons": comparisons,
        "handoff": handoff,
        "handoff_path": output_dir / "independent_evaluation_handoff.json",
    }


def _hard_relabel_example(
    example: EncodedSelectorExample,
    gold_scores: torch.Tensor,
) -> tuple[EncodedSelectorExample, dict[str, int]]:
    if len(example.labels) != int(gold_scores.numel()):
        raise ValueError("Selector labels and Gold scores differ in length.")
    labels = [int(value) for value in example.labels]
    original = list(labels)
    previous_keep = False
    flips = 0
    positive_wordpieces = 0
    for index, (active, score) in enumerate(
        zip(example.history_mask, gold_scores.tolist(), strict=True)
    ):
        if not int(active):
            previous_keep = False
            continue
        label = int(labels[index])
        if label < 0:
            raise ValueError("Visible history WordPiece has an ignored label.")
        positive = float(score) > 0.0
        positive_wordpieces += int(positive)
        if label != LABEL_TO_ID["O"]:
            previous_keep = True
            continue
        if positive:
            labels[index] = LABEL_TO_ID[
                "I-KEEP" if previous_keep else "B-KEEP"
            ]
            previous_keep = True
            flips += 1
        else:
            previous_keep = False
    if any(
        before != LABEL_TO_ID["O"] and after != before
        for before, after in zip(original, labels, strict=True)
        if before >= 0
    ):
        raise RuntimeError("Hard relabeling changed a Teacher KEEP tag.")
    return (
        EncodedSelectorExample(
            sample_id=str(example.sample_id),
            input_ids=list(example.input_ids),
            attention_mask=list(example.attention_mask),
            labels=labels,
            history_mask=list(example.history_mask),
            history_len=int(example.history_len),
        ),
        {
            "flips": flips,
            "positive_wordpieces": positive_wordpieces,
            "changed": int(flips > 0),
        },
    )


def hard_relabel_gold_score_dataset(
    dataset: GoldScoreSelectorDataset,
) -> EncodedSelectorDataset:
    """Apply the canonical NB06 hard-relabel rule to any scored cohort."""

    return EncodedSelectorDataset(
        [
            _hard_relabel_example(example.selector, example.gold_scores)[0]
            for example in dataset
        ]
    )


def _build_raw_target_histories(
    *,
    rows: Sequence[dict[str, Any]],
    score_result: GoldBM25TokenScoreResult,
    progress: bool,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    pd.DataFrame,
]:
    expected_ids = {str(row["sample_id"]) for row in rows}
    positive_by_id = _load_positive_score_tokens(
        score_result.path,
        expected_ids=expected_ids,
        progress=progress,
    )
    histories: dict[str, dict[str, list[dict[str, Any]]]] = {
        "teacher_original": {},
        "teacher_bm25_corrected": {},
    }
    stats: list[dict[str, Any]] = []
    tqdm = get_tqdm()
    for row in tqdm(
        rows,
        total=len(rows),
        desc="build full-history Teacher targets",
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    ):
        sample_id = str(row["sample_id"])
        positive_keys = positive_by_id[sample_id]
        extra_positions: dict[tuple[int, str], set[int]] = {}
        source_keys: set[tuple[int, str, int, int, str]] = set()
        teacher_keys: set[tuple[int, str, int, int, str]] = set()
        history_tokens = 0
        for turn in row.get("history", []):
            turn_id = int(turn["turn_id"])
            for field in ("question", "answer"):
                source_text = str(turn.get(field, ""))
                tokens = turn.get(f"{field}_tokens", [])
                for index, token in enumerate(tokens):
                    key = _score_token_key(token, turn_id, field)
                    if source_text[key[2] : key[3]] != key[4]:
                        raise RuntimeError(
                            f"Teacher token offsets drifted for {sample_id}."
                        )
                    if key in source_keys:
                        raise RuntimeError(
                            f"Duplicate Teacher token in {sample_id}: {key}."
                        )
                    source_keys.add(key)
                    history_tokens += 1
                    if token.get("labels"):
                        teacher_keys.add(key)
                    if key in positive_keys:
                        extra_positions.setdefault(
                            (turn_id, field),
                            set(),
                        ).add(index)
        missing = positive_keys.difference(source_keys)
        if missing:
            raise RuntimeError(
                f"Gold-BM25 tokens do not match Teacher history for "
                f"{sample_id}: {next(iter(missing))}."
            )
        corrected_keys = teacher_keys.union(positive_keys)
        original_history = compressed_teacher_history(row)
        corrected_history = compressed_teacher_history(
            row,
            extra_positions=extra_positions,
        )
        histories["teacher_original"][sample_id] = original_history
        histories["teacher_bm25_corrected"][sample_id] = (
            corrected_history
        )
        for arm, selected, history in (
            ("teacher_original", teacher_keys, original_history),
            (
                "teacher_bm25_corrected",
                corrected_keys,
                corrected_history,
            ),
        ):
            stats.append(
                {
                    "sample_id": sample_id,
                    "arm": arm,
                    "history_tokens": history_tokens,
                    "keep_tokens": len(selected),
                    "selected_words": len(selected),
                    "empty_selection": int(not history),
                }
            )
    return histories, pd.DataFrame(stats)


def build_raw_teacher_target_histories(
    *,
    rows: Sequence[dict[str, Any]],
    score_result: GoldBM25TokenScoreResult,
    progress: bool = True,
) -> tuple[
    dict[str, dict[str, list[dict[str, Any]]]],
    pd.DataFrame,
]:
    """Expose canonical original/corrected Teacher histories by cohort."""

    return _build_raw_target_histories(
        rows=rows,
        score_result=score_result,
        progress=progress,
    )


def _load_positive_score_tokens(
    path: Path | str,
    *,
    expected_ids: set[str],
    progress: bool,
) -> dict[str, set[tuple[int, str, int, int, str]]]:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Gold-BM25 score file is missing: {resolved}")
    selected: dict[
        str,
        set[tuple[int, str, int, int, str]],
    ] = {}
    tqdm = get_tqdm()
    with resolved.open("rb") as handle, tqdm(
        total=resolved.stat().st_size,
        desc="load full-history Gold-BM25 tokens",
        unit="B",
        unit_scale=True,
        dynamic_ncols=True,
        disable=not progress,
    ) as bar:
        for line in handle:
            bar.update(len(line))
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id not in expected_ids:
                continue
            if sample_id in selected:
                raise RuntimeError(
                    f"Duplicate Gold-BM25 score row: {sample_id}."
                )
            selected[sample_id] = {
                _score_token_key(
                    token,
                    int(token["turn_id"]),
                    str(token["field"]),
                )
                for token in record.get("tokens", [])
                if float(token.get("score_raw", 0.0)) > 0.0
                and not is_lexical_stopword(str(token.get("text", "")))
            }
    if set(selected) != expected_ids:
        missing = sorted(expected_ids.difference(selected))
        raise RuntimeError(
            "Gold-BM25 score rows are incomplete for the target gate: "
            f"{len(selected)}/{len(expected_ids)}, missing={missing[:3]}."
        )
    return selected


def _score_token_key(
    token: Mapping[str, Any],
    turn_id: int,
    field: str,
) -> tuple[int, str, int, int, str]:
    return (
        int(turn_id),
        str(field),
        int(token["start"]),
        int(token["end"]),
        str(token["text"]),
    )


def _load_controlled_model(
    *,
    checkpoint: Path | str,
    class_weights: Sequence[float],
    selector_config: SelectorExperimentConfig,
    config: ControlledPostTrainingConfig,
) -> EncoderCrfHistorySelector:
    seed_selector_training(config.seed)
    encoder = AutoModel.from_pretrained(
        selector_config.model_name,
        revision=selector_config.model_revision,
        trust_remote_code=False,
    )
    model = EncoderCrfHistorySelector(
        selector_config.model_name,
        num_labels=len(LABEL_TO_ID),
        class_weights=class_weights,
        crf_loss_weight=1.0,
        token_loss_weight=1.0,
        encoder=encoder,
    )
    state = torch.load(
        Path(checkpoint),
        map_location="cpu",
        weights_only=True,
    )
    expected_weights = torch.tensor(
        class_weights,
        dtype=state["class_weights"].dtype,
    )
    if not torch.allclose(
        state["class_weights"].cpu(),
        expected_weights,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("Pinned NB05 class weights differ from prepared data.")
    model.load_state_dict(state, strict=True)
    if not all(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Controlled post-training must use full FT.")
    seed_selector_training(config.seed)
    return model


def _evaluate_candidate_fusions(
    *,
    pipeline_results: pd.DataFrame,
    gold_by_sample: Mapping[str, Sequence[Any]],
    selector_config: SelectorExperimentConfig,
    config: ControlledPostTrainingConfig,
    device: torch.device,
    system: str,
    epoch: int,
    progress: bool,
) -> dict[str, pd.DataFrame]:
    selected = pipeline_results.loc[
        pipeline_results["budget"].eq(config.budget)
        & pipeline_results["arm"].isin(
            ["viterbi", "sample_1", "sample_2"]
        )
    ].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    if selected.duplicated(["sample_id", "arm"]).any():
        raise ValueError("Candidate retrieval rows are duplicated.")
    keys = selected.pivot(
        index="sample_id",
        columns="arm",
        values="rewrite_key",
    ).sort_index(kind="mergesort")
    if (
        list(keys.columns) != ["sample_1", "sample_2", "viterbi"]
        or keys.isna().any().any()
    ):
        raise ValueError("Candidate retrieval rankings are incomplete.")
    pipeline = CachedIterCQRBM25Pipeline(
        config=CachedIterCQRBM25Config(
            cache_db=selector_config.pipeline_cache_db,
            model_dir=selector_config.itercqr_model_dir,
            bm25_index_dir=selector_config.bm25_index_dir,
            device=str(device),
            rewrite_batch_size=selector_config.rewrite_batch_size,
            retrieval_batch_size=selector_config.retrieval_batch_size,
            retrieval_workers=selector_config.retrieval_workers,
            retrieval_top_k=selector_config.retrieval_top_k,
            eval_ks=selector_config.eval_ks,
            progress=progress,
        ),
        tokenizer=None,
    )
    try:
        rankings = pipeline.load_rankings(
            keys.to_numpy().ravel(),
            progress_prefix=f"{system} epoch {epoch} candidate fusion",
        )
    finally:
        pipeline.close()

    viterbi = selected.loc[
        selected["arm"].eq("viterbi"),
        ["sample_id", *TEACHER_METRIC_COLUMNS],
    ].copy()
    viterbi["route"] = "viterbi"
    rows = [viterbi]
    fused_rows: list[dict[str, Any]] = []
    union_rows: list[dict[str, Any]] = []
    tqdm = get_tqdm()
    for sample_id, row in tqdm(
        keys.iterrows(),
        total=len(keys),
        desc=f"{system} epoch {epoch} fixed candidate RRF10",
        unit="query",
        dynamic_ncols=True,
        disable=not progress,
    ):
        route_lists = {
            "ffbs2_rrf10": [
                rankings[str(row["sample_1"])],
                rankings[str(row["sample_2"])],
            ],
            "viterbi_ffbs2_rrf10": [
                rankings[str(row["viterbi"])],
                rankings[str(row["sample_1"])],
                rankings[str(row["sample_2"])],
            ],
        }
        for route, rank_lists in route_lists.items():
            hits = reciprocal_rank_fusion(
                rank_lists,
                rrf_k=config.rrf_k,
                top_k=selector_config.retrieval_top_k,
            )
            metrics = retrieval_metrics(
                hits,
                gold_by_sample[str(sample_id)],
                ks=selector_config.eval_ks,
            )
            fused_rows.append(
                {
                    "sample_id": str(sample_id),
                    "route": route,
                    **{
                        metric: metrics[metric]
                        for metric in TEACHER_METRIC_COLUMNS
                    },
                }
            )
        relevant = set(normalize_docids(gold_by_sample[str(sample_id)]))
        for cutoff in (10, 100, 1000):
            hits = {
                candidate: int(
                    bool(
                        relevant.intersection(
                            rankings[str(row[candidate])][:cutoff]
                        )
                    )
                )
                for candidate in ("viterbi", "sample_1", "sample_2")
            }
            pattern = "+".join(
                candidate
                for candidate in ("viterbi", "sample_1", "sample_2")
                if hits[candidate]
            ) or "neither"
            union_rows.append(
                {
                    "sample_id": str(sample_id),
                    "cutoff": cutoff,
                    "viterbi_hit": hits["viterbi"],
                    "sample_1_hit": hits["sample_1"],
                    "sample_2_hit": hits["sample_2"],
                    "union_hit": int(any(hits.values())),
                    "hit_pattern": pattern,
                }
            )
    rows.append(pd.DataFrame(fused_rows))
    result = pd.concat(rows, ignore_index=True, sort=False)
    result["budget"] = config.budget
    result["system"] = str(system)
    result["epoch"] = int(epoch)
    result = result.sort_values(
        ["route", "sample_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    union_by_query = pd.DataFrame(union_rows).sort_values(
        ["cutoff", "sample_id"],
        kind="mergesort",
    )
    union_summary = (
        union_by_query.groupby("cutoff", observed=True, sort=True)
        .agg(
            n=("sample_id", "size"),
            viterbi_recall=("viterbi_hit", "mean"),
            sample_1_recall=("sample_1_hit", "mean"),
            sample_2_recall=("sample_2_hit", "mean"),
            union_recall=("union_hit", "mean"),
        )
        .reset_index()
    )
    union_summary["union_gain_vs_best_single"] = (
        union_summary["union_recall"]
        - union_summary[
            ["viterbi_recall", "sample_1_recall", "sample_2_recall"]
        ].max(axis=1)
    )
    union_patterns = (
        union_by_query.groupby(
            ["cutoff", "hit_pattern"],
            observed=True,
            sort=True,
        )
        .size()
        .rename("count")
        .reset_index()
    )
    union_patterns["share"] = union_patterns["count"] / len(keys)
    return {
        "by_query": result,
        "union_by_query": union_by_query,
        "union_summary": union_summary,
        "union_patterns": union_patterns,
    }


def _selector_loader(
    dataset: EncodedSelectorDataset,
    *,
    tokenizer: Any,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        collate_fn=lambda batch: collate_selector_batch(
            batch,
            int(tokenizer.pad_token_id),
        ),
        num_workers=int(num_workers),
        generator=generator if shuffle else None,
    )


def _validate_vehicle(
    rows: Sequence[dict[str, Any]],
    samples: Sequence[Any],
    gold_by_sample: Mapping[str, Sequence[Any]],
    config: ControlledPostTrainingConfig,
) -> None:
    row_ids = [str(row["sample_id"]) for row in rows]
    sample_ids = [str(sample.sample_id) for sample in samples]
    if (
        len(row_ids) != 600
        or row_ids != sample_ids
        or set(row_ids) != set(map(str, gold_by_sample))
        or _sample_id_sha256(row_ids)
        != config.expected_vehicle_sample_id_sha256
    ):
        raise ValueError("Evaluation data is not the canonical 600 gate.")


def _label_dataset_sha256(dataset: EncodedSelectorDataset) -> str:
    digest = hashlib.sha256()
    for example in dataset.examples:
        digest.update(str(example.sample_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(
                [int(value) for value in example.labels],
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _checkpoint_state_dicts_equal(
    left: Path | str,
    right: Path | str,
) -> bool:
    """Compare checkpoint parameters, independent of torch ZIP bytes."""

    left_state = torch.load(
        Path(left),
        map_location="cpu",
        weights_only=True,
    )
    right_state = torch.load(
        Path(right),
        map_location="cpu",
        weights_only=True,
    )
    if left_state.keys() != right_state.keys():
        return False
    return all(
        torch.equal(left_state[key], right_state[key])
        for key in left_state
    )


def _completed_training_run(
    output_dir: Path,
    *,
    run_config: Mapping[str, Any],
    epochs: int,
) -> bool:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return False
    manifest = _read_json(manifest_path)
    return bool(
        manifest.get("complete") is True
        and manifest.get("config") == dict(run_config)
        and manifest.get("checkpoint_sha256")
        == _sha256_file(output_dir / "model.pt")
        and len(manifest.get("epoch_checkpoints", {})) == int(epochs)
    )


def _sample_id_sha256(sample_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(map(str, sample_ids))).encode("utf-8")
    ).hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path | str, value: Any) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def _ensure_locked_json(path: Path | str, value: Mapping[str, Any]) -> None:
    resolved = Path(path)
    expected = json.loads(json.dumps(dict(value), ensure_ascii=False))
    if resolved.exists():
        if _read_json(resolved) != expected:
            raise RuntimeError(f"Locked configuration differs: {resolved}")
        return
    _write_json(resolved, expected)


def _read_frame(path: Path | str) -> pd.DataFrame:
    resolved = Path(path)
    if not resolved.exists():
        return pd.DataFrame()
    return pd.read_csv(resolved, float_precision="round_trip")


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return [
        {
            str(key): (
                value.item() if isinstance(value, np.generic) else value
            )
            for key, value in row.items()
        }
        for row in frame.to_dict(orient="records")
    ]


def _write_frame(path: Path | str, frame: pd.DataFrame) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        lineterminator="\n",
    )
    temporary.replace(resolved)


__all__ = [
    "CONTROLLED_POST_TRAINING_ARMS",
    "CONTROLLED_POST_TRAINING_BOOTSTRAP_SEEDS",
    "CONTROLLED_POST_TRAINING_RECIPE_VERSION",
    "CONTROLLED_POST_TRAINING_ROUTE_ORDER",
    "ControlledPostTrainingConfig",
    "ControlledPostTrainingDatasets",
    "build_raw_teacher_target_histories",
    "compare_controlled_post_training_runs",
    "evaluate_controlled_post_training_baseline",
    "evaluate_controlled_post_training_epoch",
    "evaluate_controlled_teacher_targets",
    "hard_relabel_gold_score_dataset",
    "prepare_controlled_post_training_datasets",
    "run_controlled_post_training_arm",
    "validate_controlled_post_training_inputs",
]
