#!/usr/bin/env python3
"""Export final checkpoint and decoder comparisons without rerunning inference.

Checkpoint CIs are the saved 10,000-query paired bootstrap intervals. Decoder
within-checkpoint CIs are recomputed from stored per-query RR with the same
bounded-memory percentile procedure as rocc.evaluation.paired_bootstrap_mean_ci.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results"
OUT = RESULTS / "10_efficiency_analysis/figures/data"
SYSTEMS = ("pretrained", "imitation_control_e6", "bm25_treatment_e6")
ARMS = ("R", "D", "I+R+D")
sources = []


def read(path: Path) -> pd.DataFrame:
    sources.append({"path": str(Path("experiments/results") / path.relative_to(RESULTS)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return pd.read_csv(path)


def one(rows: pd.DataFrame) -> pd.Series:
    assert len(rows) == 1, f"Expected one row, found {len(rows)}"
    return rows.iloc[0]


def bootstrap(delta: np.ndarray, seed: int) -> tuple[float, float]:
    assert delta.ndim == 1 and len(delta) == 2514 and np.isfinite(delta).all()
    rng = np.random.default_rng(seed)
    draws = np.empty(10_000)
    for start in range(0, 10_000, 500):
        indices = rng.integers(0, len(delta), size=(500, len(delta)))
        draws[start:start + 500] = delta[indices].mean(axis=1)
    return tuple(map(float, np.percentile(draws, [2.5, 97.5])))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints, ffbs = [], []
    for dataset, directory, comparison_file, expected_n in (
        ("topiocqa", "07a_topiocqa_final", "primary_comparisons.csv", 2514),
        ("qrecc", "07b_qrecc_ood", "paired_comparisons.csv", 8209),
    ):
        for backend in ("bm25", "ance"):
            base = RESULTS / directory / "backends" / backend
            comparisons = read(base / comparison_file)
            summary = read(base / "route_summary.csv")
            rows = comparisons.loc[
                comparisons.metric.eq("MRR") & comparisons.right_system.eq("pretrained")
                & comparisons.left_system.isin(SYSTEMS[1:])
                & comparisons.left_budget.eq(64) & comparisons.right_budget.eq(64)
            ]
            assert len(rows) == 6 and set(rows.left_arm) == set(ARMS)
            for system, short, offset in ((SYSTEMS[1], "control", .13),
                                         (SYSTEMS[2], "treatment", -.13)):
                for index, arm in enumerate(ARMS):
                    row = one(rows.loc[rows.left_system.eq(system) & rows.left_arm.eq(arm)])
                    assert row.right_arm == arm and row.n == expected_n and row.replicates == 10000
                    left = one(summary.loc[summary.system.eq(system) & summary.budget.eq(64) & summary.arm.eq(arm)])
                    right = one(summary.loc[summary.system.eq("pretrained") & summary.budget.eq(64) & summary.arm.eq(arm)])
                    assert left.n == right.n == expected_n
                    assert np.isclose(left.MRR - right.MRR, row.delta, atol=1e-12)
                    checkpoints.append(dict(dataset=dataset, backend=backend, checkpoint=short,
                        system=system, arm=arm, n=expected_n, mean=left.MRR, base_mean=right.MRR,
                        y=3-index+offset, delta=row.delta, low=row.ci95_low, high=row.ci95_high,
                        minus=row.delta-row.ci95_low, plus=row.ci95_high-row.delta,
                        seed=int(row.seed), replicates=10000))
            if dataset != "topiocqa":
                continue
            metrics = read(base / "crf_decoder_metrics_by_query.csv")
            decoder_summary = read(base / "crf_decoder_summary.csv")
            assert metrics.budget.eq(64).all()
            assert not metrics.duplicated(["system", "arm", "sample_id"]).any()
            for system_index, system in enumerate(SYSTEMS):
                frame = metrics.loc[metrics.system.eq(system)].pivot(index="sample_id", columns="arm", values="MRR")
                frame = frame.sort_index()
                assert len(frame) == expected_n and frame.notna().all().all()
                for arm_index, (arm, short, offset) in enumerate((
                    ("2-FFBS-RRF10", "ffbs2", .13),
                    ("Viterbi+2-FFBS-RRF10", "viterbi_ffbs2", -.13),
                )):
                    values = (frame[arm] - frame["Viterbi"]).to_numpy()
                    seed = 9300 + (100 if backend == "ance" else 0) + 10*system_index + arm_index
                    low, high = bootstrap(values, seed)
                    mean, baseline = frame[arm].mean(), frame["Viterbi"].mean()
                    for check_arm, check_mean in ((arm, mean), ("Viterbi", baseline)):
                        stored = one(decoder_summary.loc[decoder_summary.system.eq(system) & decoder_summary.arm.eq(check_arm)])
                        assert stored.n == expected_n and np.isclose(stored.MRR, check_mean, atol=1e-12)
                    delta = values.mean()
                    ffbs.append(dict(backend=backend, system=system, arm=arm, decoder=short,
                        n=expected_n, mean=mean, base_mean=baseline, y=3-system_index+offset,
                        delta=delta, low=low, high=high, minus=delta-low, plus=high-delta,
                        seed=seed, replicates=10000))
    checkpoint_frame, ffbs_frame = pd.DataFrame(checkpoints), pd.DataFrame(ffbs)
    checkpoint_frame.to_csv(OUT / "results_final_checkpoints.csv", index=False, float_format="%.17g")
    ffbs_frame.to_csv(OUT / "results_final_ffbs.csv", index=False, float_format="%.17g")
    for (dataset, backend, checkpoint), frame in checkpoint_frame.groupby(["dataset", "backend", "checkpoint"]):
        frame.to_csv(OUT / f"results_final_checkpoints_{dataset}_{backend}_{checkpoint}.csv", index=False, float_format="%.17g")
    for (backend, decoder), frame in ffbs_frame.groupby(["backend", "decoder"]):
        frame.to_csv(OUT / f"results_final_ffbs_{backend}_{decoder}.csv", index=False, float_format="%.17g")
    pd.DataFrame(sources).drop_duplicates().to_csv(OUT / "results_final_sources.csv", index=False)
    teacher_matched_export()
    print("Verified 24 saved checkpoint deltas and computed 12 paired decoder intervals.")
    print(ffbs_frame[["backend", "system", "decoder", "mean", "base_mean", "delta", "low", "high"]].to_string(index=False))


def teacher_matched_export() -> None:
    """Matched-cohort point estimates only; no CI or significance is inferred."""
    cohort_source = RESULTS / "10_efficiency_analysis/teacher_eval/cohort.csv"
    student_source = RESULTS / "07a_topiocqa_final/backends/bm25/crf_decoder_metrics_by_query.csv"
    teacher_source = RESULTS / "10_efficiency_analysis/teacher_eval/retrieval_summary.csv"
    cohort = read(cohort_source)
    assert len(cohort) == 2104 and not cohort.sample_id.duplicated().any()
    assert cohort.history_depth.ge(2).all() and cohort.teacher_labeled.eq(1).all()
    ids = set(cohort.sample_id.astype(str))
    students, teachers = read(student_source), read(teacher_source)
    teacher = one(teachers.loc[teachers.system.eq("bm25_teacher") & teachers.budget.eq(64)])
    assert teacher.n == 2104
    rows = []
    for decoder, rankings in (("Viterbi", 1), ("2-FFBS-RRF10", 2), ("Viterbi+2-FFBS-RRF10", 3)):
        selected = students.loc[students.system.eq("bm25_treatment_e6") & students.budget.eq(64)
            & students.arm.eq(decoder) & students.sample_id.astype(str).isin(ids)]
        assert len(selected) == 2104 and not selected.sample_id.duplicated().any()
        assert set(selected.sample_id.astype(str)) == ids
        mean = float(selected.MRR.mean())
        rows.append(dict(dataset="TopiOCQA Dev", backend="BM25", student_system="bm25_treatment_e6",
            budget=64, student_decoder=decoder, nominal_ranking_count=rankings, n=2104,
            student_mrr=mean, teacher_system="bm25_teacher", teacher_mrr=float(teacher.MRR),
            teacher_minus_student_mrr=float(teacher.MRR)-mean,
            comparison_type="matched-cohort point-estimate difference only",
            cohort_rule="teacher_labeled_history_depth_ge_2", cohort_identifier="sample_id",
            cohort_source=str(Path("experiments/results") / cohort_source.relative_to(RESULTS)),
            student_source=str(Path("experiments/results") / student_source.relative_to(RESULTS)),
            teacher_source=str(Path("experiments/results") / teacher_source.relative_to(RESULTS))))
    pd.DataFrame(rows).to_csv(OUT / "results_ffbs_teacher_matched.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS,
                        help="Saved NB07a/b results and 10_efficiency_analysis/teacher_eval.")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: RESULTS_DIR/10_efficiency_analysis/figures/data.")
    args = parser.parse_args()
    RESULTS = args.results_dir.expanduser().resolve()
    OUT = (args.output_dir or RESULTS / "10_efficiency_analysis/figures/data").expanduser().resolve()
    main()
