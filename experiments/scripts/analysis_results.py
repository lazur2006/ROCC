#!/usr/bin/env python3
"""Display additional experiment analyses from locally generated results.

This script contains the derived statistics used alongside NB03--NB10:
development comparisons, retrieval/compression summaries, exact Shapley checks,
and latency/encoder calculations. It neither writes a second result format nor
starts training, retrieval, model downloads or API requests.

Notebook use: display_results(experiments_dir, notebook_stem).
CLI use: python experiments/scripts/analysis_results.py 10_efficiency_analysis
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import zlib

import numpy as np
import pandas as pd
from scipy.stats import t

Table = tuple[str, pd.DataFrame, list[Path]]


def _json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def _record(path: Path, sample_id: str) -> dict:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["sample_id"] == sample_id:
                return row
    raise ValueError(f"Missing {sample_id} in {path}")


def _local_tokenizer(experiments_dir: Path):
    """Load only tokenizer.json from the recorded, locally cached revision."""
    from tokenizers import Tokenizer

    manifest_path = experiments_dir / "results/05_history_selector/full_candidates/collapsed_crf/manifest.json"
    config = _json(manifest_path)["config"]
    hub = Path(os.environ.get("HF_HUB_CACHE", Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"))
    path = hub / ("models--" + config["model_name"].replace("/", "--")) / "snapshots" / config["model_revision"] / "tokenizer.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cached tokenizer required; no download attempted: {path}")
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return tokenizer, config, [manifest_path, path]


def _history_text(row: dict, order: str) -> str:
    """Canonical rendering from rocc/history_selector.py:build_history_text."""
    if order not in {"recent_first", "chronological"}:
        raise ValueError(f"Unknown history order: {order}")
    turns = row["history"][::-1] if order == "recent_first" else row["history"]
    return "".join(f"T{int(turn['turn_id'])} Q: {turn.get('question', '')}\nA: {turn.get('answer', '')}\n\n" for turn in turns)


def _truncation(experiments_dir: Path) -> Table:
    tokenizer, config, sources = _local_tokenizer(experiments_dir)
    base = experiments_dir / "results/04_teacher"
    rows = []
    for population, name in [("eligible training", "train"), ("training-member development", "train_600")]:
        path = base / f"rocc_history_labels_topiocqa_{name}.jsonl"
        count = truncated = 0
        seen = set()
        batch = []

        def consume(pairs):
            encoded = tokenizer.encode_batch(pairs)
            return sum(len(item.ids) > config["max_length"] for item in encoded)

        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["sample_id"] in seen:
                    raise ValueError(f"Duplicate annotation: {row['sample_id']}")
                seen.add(row["sample_id"])
                count += 1
                batch.append((str(row["current_query"]), _history_text(row, config["history_order"])))
                if len(batch) == 256:
                    truncated += consume(batch)
                    batch = []
        if batch:
            truncated += consume(batch)
        rows.append(dict(population=population, queries=count, max_wordpieces=config["max_length"], truncated_queries=truncated, truncated_percent=100 * truncated / count))
        sources.append(path)
    sources.append(experiments_dir / "rocc/history_selector.py")
    return "Student-window truncation (local tokenizer; no model inference)", pd.DataFrame(rows), sources


def _union(base: Path) -> Table:
    path = base / "fusion/standalone_metrics_by_query.csv"
    frame = _csv(path)
    if frame.duplicated(["sample_id", "arm"]).any():
        raise ValueError("Duplicate standalone sample/route")
    pivot = frame.pivot(index="sample_id", columns="arm", values="R@10")
    if pivot[["I", "R", "D"]].isna().any().any() or not pivot.isin([0, 1]).all().all():
        raise ValueError("Expected complete binary relevant-hit indicators")
    hits = pivot.astype(bool)
    rows = []
    for arm, others in [("I", ["R"]), ("R", ["I"]), ("D", ["I", "R"])]:
        rows.append(dict(route=arm, queries=len(hits), relevant_hits=int(hits[arm].sum()), recall=hits[arm].mean(), additional_hits=int((hits[arm] & ~hits[others].any(axis=1)).sum()), beyond="+".join(others)))
    for arms in [["I", "R"], ["I", "R", "D"]]:
        union = hits[arms].any(axis=1)
        rows.append(dict(route="union " + "+".join(arms), queries=len(hits), relevant_hits=int(union.sum()), recall=union.mean(), additional_hits=np.nan, beyond=""))
    return "Top-10 route hits and unions (same query population)", pd.DataFrame(rows), [path]


def _fusion_replay(experiments_dir: Path) -> Table:
    """Re-rank stored document IDs only; fail rather than retrieve a cache miss."""
    base = experiments_dir / "results/05_history_selector/train600"
    population_path = base / "population/manifest.json"
    architecture_path = base / "architecture/decision.json"
    decision_path = base / "fusion/decision.json"
    iterative_path = base / "retrieval/metrics_by_query.csv"
    direct_path = base / "fusion/direct_metrics_by_query.csv"
    saved_path = base / "fusion/standalone_metrics_by_query.csv"
    fused_path = base / "fusion/metrics_by_query.csv"
    gold_path = experiments_dir / "data/topiocqa/downloads/data/retriever/original/train.json"
    cache = Path(os.environ.get("ROCC_RANKING_CACHE", os.environ.get(
        "ROCC_THESIS_RANKING_CACHE",
        experiments_dir.parent / ".cache/master_thesis/03_oracle_headroom_analysis/runtime_cache.sqlite3",
    ))).expanduser().resolve()
    if not cache.is_file():
        raise FileNotFoundError(2, "NB05 analysis requires its saved ranking cache; set ROCC_RANKING_CACHE", str(cache))
    decision = _json(decision_path)
    selected = _json(architecture_path)["selected_arm"]
    ids = sorted(_json(population_path)["sample_ids"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate population IDs")
    iterative = _csv(iterative_path)
    iterative = iterative[(iterative.budget == decision["budget"]) & iterative.arm.isin(["recency", selected])]
    keys = defaultdict(dict)
    for row in iterative.itertuples(index=False):
        arm = "I" if row.arm == "recency" else "R"
        if arm in keys[row.sample_id]:
            raise ValueError("Duplicate cached route")
        keys[row.sample_id][arm] = row.rewrite_key
    for row in _csv(direct_path).itertuples(index=False):
        if row.budget != decision["budget"] or "D" in keys[row.sample_id]:
            raise ValueError("Unexpected direct-route scope")
        keys[row.sample_id]["D"] = row.rewrite_key
    if sorted(keys) != ids or any(set(value) != set("IRD") for value in keys.values()):
        raise ValueError("Cached route population does not match manifest")
    gold = {f"train:{row['conv_id']}:{row['turn_id']}": {str(ctx["passage_id"]) for ctx in row["positive_ctxs"]} for row in _json(gold_path)}
    if any(len(gold[sid]) != 1 for sid in ids):
        raise ValueError("Expected one relevant passage per development query")
    rankings = {}
    with sqlite3.connect(cache.as_uri() + "?mode=ro", uri=True) as connection:
        for key in sorted({key for value in keys.values() for key in value.values()}):
            record = connection.execute("SELECT docids_blob FROM retrievals WHERE rewrite_key=?", (key,)).fetchone()
            if record is None:
                raise ValueError(f"Ranking not cached: {key}; retrieval is disabled")
            rankings[key] = [str(item) for item in json.loads(zlib.decompress(record[0]))]
    depths = {len(value) for value in rankings.values()}
    if len(depths) != 1:
        raise ValueError("Cached ranking depths differ")
    depth = depths.pop()
    kappa = decision["selected_parameters"]["rrf_k"]

    def fuse(lists):
        scores = defaultdict(float)
        for ranking in lists:
            seen = set()
            for rank, docid in enumerate(ranking, 1):
                if docid not in seen:
                    scores[docid] += 1 / (kappa + rank)
                    seen.add(docid)
        return [docid for docid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:depth]]

    rows = []
    for sid in ids:
        lists = {arm: rankings[keys[sid][arm]] for arm in "IRD"}
        for arm in ["I", "R", "D", "IR", "IRD"]:
            ranking = lists[arm] if len(arm) == 1 else fuse([lists[item] for item in arm])
            relevant = [rank for rank, docid in enumerate(ranking, 1) if docid in gold[sid]]
            rows.append(dict(sample_id=sid, arm=arm, MRR=1 / min(relevant) if relevant else 0.0, **{"R@10": len(set(ranking[:10]) & gold[sid]) / len(gold[sid])}))
    replay = pd.DataFrame(rows).set_index(["sample_id", "arm"])
    saved = _csv(saved_path)
    fused = _csv(fused_path)
    fused = fused[fused.arm == decision["selected_method"]].copy()
    fused["arm"] = "IRD"
    expected = pd.concat([saved, fused]).set_index(["sample_id", "arm"])
    for metric in ["MRR", "R@10"]:
        np.testing.assert_allclose(replay.loc[expected.index, metric], expected[metric], rtol=0, atol=1e-14)
    summary = replay.reset_index().groupby("arm", sort=False).agg(n=("sample_id", "size"), MRR=("MRR", "mean"), **{"R@10": ("R@10", "mean")}).reset_index()
    summary["rrf_k"] = [np.nan if len(arm) == 1 else kappa for arm in summary.arm]
    return "Stored-ranking replay: I, R, D, IR and IRD (no retrieval)", summary, [population_path, architecture_path, decision_path, iterative_path, direct_path, saved_path, fused_path, gold_path, cache, experiments_dir / "rocc/evaluation.py"]


def _target_intervals(experiments_dir: Path, base: Path) -> Table:
    target_path = base / "target_headroom/R_itercqr/metrics_by_query.csv"
    baseline_path = experiments_dir / "results/05_history_selector/train600/fusion/standalone_metrics_by_query.csv"
    paired_path = base / "target_headroom/paired_comparisons.csv"
    target = _csv(target_path).pivot(index="sample_id", columns="arm", values="MRR").sort_index()
    baseline = _csv(baseline_path).query("arm == 'I'").set_index("sample_id")["MRR"]
    if set(baseline.index) != set(target.index) or target.isna().any().any():
        raise ValueError("Target/baseline population mismatch")
    target.insert(0, "I", baseline.reindex(target.index))
    paired = _csv(paired_path).query("metric == 'MRR'").iloc[0]
    n, replicates, seed = len(target), int(paired.replicates), int(paired.seed)
    rng = np.random.default_rng(seed)
    values = target.to_numpy()
    means = np.empty((replicates, len(target.columns)))
    for start in range(0, replicates, 250):
        draw = rng.integers(0, n, size=(min(250, replicates - start), n))
        means[start:start + len(draw)] = values[draw].mean(axis=1)
    left = target.columns.get_loc(paired.left_arm)
    right = target.columns.get_loc(paired.right_arm)
    ci = np.quantile(means[:, left] - means[:, right], [.025, .975])
    np.testing.assert_allclose(ci, [paired.ci95_low, paired.ci95_high], rtol=0, atol=1e-14)
    rows = []
    for column, arm in enumerate(target.columns):
        low, high = np.quantile(means[:, column], [.025, .975])
        rows.append(dict(arm=arm, n=n, MRR=values[:, column].mean(), ci95_low=low, ci95_high=high, seed=seed, replicates=replicates))
    return "Original/corrected target means and marginal 95% bootstrap intervals", pd.DataFrame(rows), [target_path, baseline_path, paired_path]


def _sesame(experiments_dir: Path) -> Table:
    path = experiments_dir / "results/06_post_training/gold_bm25_token_scores.jsonl"
    row = _record(path, "train:1340:5")
    frame = pd.DataFrame(row["tokens"])
    frame = frame[(frame.turn_id == 1) & (frame.field == "question")].copy()
    frame.insert(0, "sample_id", row["sample_id"])
    frame.insert(1, "gold_passage_id", row["gold_passage_id"])
    return "Sesame Street example: all first-question raw BM25 word scores", frame[["sample_id", "gold_passage_id", "text", "score_raw", "score_norm"]], [path]


COHORT = "TopiOCQA training-member development, BM25"


MODELS = {
    "taxonomy_linear": "Taxonomy + independent labels",
    "collapsed_linear": "BIO + independent labels",
    "taxonomy_crf": "Taxonomy + CRF",
    "collapsed_crf": "BIO + CRF (R)",
}


GENERATORS = {
    "recent_variants": "Recent turn",
    "position_anchor": "Position",
    "lexical_overlap": "Lexical overlap",
    "spacy_recent_entity": "Recent entity",
}


ARMS = {"imitation_control": "Continued imitation", "bm25_treatment": "BM25-corrected labels"}


def _table(title, frame, paths):
    return title, frame.reset_index(drop=True), paths


def _paired(frame):
    return frame[["comparison", "delta", "ci95_low", "ci95_high"]].rename(
        columns={"delta": "MRR_difference"}
    )


def _settings(paths):
    names = ["epochs", "batch_size", "learning_rate", "weight_decay", "warmup_ratio", "gradient_clip_norm"]
    rows = [{key: _json(path)["config"][key] for key in names} for path in paths]
    if any(row != rows[0] for row in rows[1:]):
        raise ValueError("The compared models no longer share their optimization settings")
    return _table("Shared optimization settings", pd.DataFrame(rows[:1]), paths)


def _oracle(exp):
    base = exp / "results/03_oracle_headroom_analysis"
    out = []
    path = base / "sampling_summary.csv"
    population = _csv(path).pivot(index="depth_bin", columns="cohort", values="n")
    population = population.rename(columns={"eligible_train_population": "eligible_training_queries", "depth_balanced_sample": "development_queries"}).reset_index()
    out.append(_table("TopiOCQA development-sample depth bins", population, [path]))

    budget_path, ci_path = base / "budget_degradation.csv", base / "bootstrap_summary.csv"
    intervals = _csv(ci_path)
    budget = _csv(budget_path).merge(
        intervals.query("metric_type == 'recency_delta'")[["budget", "ci_low", "ci_high"]],
        on="budget", validate="one_to_one",
    ).rename(columns={"recency_mrr": "I_MRR", "delta_vs_B512": "MRR_difference_vs_B512", "relative_delta_vs_B512_pct": "relative_difference_percent", "ci_low": "ci95_low", "ci_high": "ci95_high"})
    out.append(_table(f"{COHORT}: recency budget curve", budget, [budget_path, ci_path]))

    path = base / "space_oracle_summary.csv"
    oracle = _csv(path)[["budget", "control_mode", "queries", "recency_mrr", "space_oracle_mrr"]]
    oracle = oracle.rename(columns={"control_mode": "search", "recency_mrr": "I_MRR", "space_oracle_mrr": "oracle_MRR"})
    oracle["search"] = oracle["search"].replace({"approximated": "Sampled", "exhaustive": "Full enumeration"})
    out.append(_table(f"{COHORT}: V1 selection-space oracle", oracle, [path]))

    path = base / "control_relation_summary.csv"
    out.append(_table(f"{COHORT}: mean within-query outcome shares", _csv(path)[["budget", "relation_to_recency", "share"]], [path]))
    path = base / "entity_anchor_summary.csv"
    exact = _csv(path)[["population", "queries", "queries_with_exact_entity_anchor", "entity_anchor_coverage"]]
    exact = exact.rename(columns={"queries_with_exact_entity_anchor": "covered_queries", "entity_anchor_coverage": "coverage"})
    out.append(_table("Exact-entity coverage", exact, [path]))

    generator_path, random_path = [base / name for name in ["generator_summary.csv", "mc_summary_all_budgets.csv"]]
    random = _csv(random_path)
    generators = _csv(generator_path)[["generator_family", "coverage", "recency_mrr", "generator_oracle_mrr"]].merge(
        random.query("budget == 64")[["generator_family", "null_headroom_mean", "excess", "p_holm"]],
        on="generator_family", validate="one_to_one",
    ).merge(intervals.query("metric_type == 'excess' and budget == 64")[["generator_family", "ci_low", "ci_high"]], on="generator_family", validate="one_to_one")
    generators["random_oracle_MRR"] = generators.recency_mrr + generators.null_headroom_mean
    generators = generators.rename(columns={"generator_family": "generator", "generator_oracle_mrr": "generator_oracle_MRR", "excess": "MRR_excess", "ci_low": "ci95_low", "ci_high": "ci95_high", "p_holm": "Holm_p"})
    generators["generator"] = generators.generator.replace(GENERATORS)
    generators = generators[["generator", "coverage", "generator_oracle_MRR", "random_oracle_MRR", "MRR_excess", "ci95_low", "ci95_high", "Holm_p"]]
    out.append(_table(f"{COHORT}, B64: generator coverage and matched controls", generators, [generator_path, random_path, ci_path]))
    larger = random.query("budget != 64")[["budget", "generator_family", "excess", "p_holm"]].copy()
    # Larger-budget significance is discussed only for the recent-turn rule.
    larger.loc[~((larger.generator_family == "recent_variants") & larger.budget.isin([256, 512])), "p_holm"] = np.nan
    larger = larger.rename(columns={"generator_family": "generator", "excess": "MRR_excess", "p_holm": "Holm_p"})
    larger["generator"] = larger.generator.replace(GENERATORS)
    out.append(_table(f"{COHORT}: generator excess at larger budgets", larger, [random_path]))
    return out


def _teacher(exp):
    base = exp / "results/04_teacher"
    path = base / "teacher_600_metrics.csv"
    frame = _csv(path)
    arms = frame.query("row_type == 'arm'")[["budget", "arm", "MRR"]].copy()
    arms["arm"] = arms.arm.replace({"recency": "I", "random_token": "Word-count-matched random", "teacher": "Teacher"})
    out = [_table(f"{COHORT}: teacher budget curve", arms, [path])]
    paired = frame.query("row_type != 'arm' and budget == 64")[["comparison", "MRR_delta", "ci95_low", "ci95_high"]].rename(columns={"MRR_delta": "MRR_difference"})
    paired["comparison"] = paired.comparison.replace({"Teacher − Recency": "Teacher - I", "Teacher − Zufall": "Teacher - word-count-matched random"})
    out.append(_table(f"{COHORT}, B64: paired teacher MRR differences", paired, [path]))
    points = frame.query("row_type == 'arm'").set_index(["budget", "arm"]).MRR
    recovery = 100 * (points[64, "teacher"] - points[64, "recency"]) / (points[512, "recency"] - points[64, "recency"])
    out.append(_table("Teacher recovery of I512-to-I64 MRR loss", pd.DataFrame([{"recovered_percent": recovery}]), [path]))
    paths = [base / f"rocc_history_labels_topiocqa_{split}.manifest.json" for split in ["train", "dev"]]
    counts = pd.DataFrame([{"split": split, **{key: _json(path)[key] for key in ["eligible_queries", "model", "temperature"]}} for split, path in zip(["Training", "Development"], paths)])
    out.append(_table("Teacher annotation populations and settings", counts, paths))
    return out


def _students(exp):
    base = exp / "results/05_history_selector"
    path = base / "full_candidates/epoch_metrics.csv"
    epochs = _csv(path)
    epochs["exact_span_F1"] = epochs.span_f1.fillna(epochs.keep_span_f1)
    curve = epochs[["candidate", "epoch", "train_loss", "keep_f1", "exact_span_F1"]].rename(columns={"candidate": "model", "train_loss": "mean_training_loss", "keep_f1": "development_KEEP_F1", "exact_span_F1": "development_exact_span_F1"})
    curve["model"] = curve.model.replace(MODELS)
    out = [_table("Student epoch means and training-member development F1", curve, [path])]
    out.append(_settings([base / "full_candidates" / arm / "manifest.json" for arm in epochs.candidate.unique()]))
    paths, labels = [], []
    for arm, representation in [("taxonomy_linear", "Full taxonomy"), ("collapsed_crf", "BIO")]:
        path = base / "full_candidates" / arm / "metrics.json"
        stats = _json(path)["train_label_stats"]
        if sum(stats["label_counts"].values()) != stats["history_tokens"]:
            raise ValueError("Label counts do not sum to the supervised history-token count")
        labels.extend({"representation": representation, "label": label, "share_percent": 100 * count / stats["history_tokens"]} for label, count in stats["label_counts"].items())
        paths.append(path)
    out.append(_table("Training-label distribution over visible history WordPieces", pd.DataFrame(labels), paths))

    run = base / "train600"
    architecture_path, fusion_path = run / "architecture/summary.csv", run / "fusion/summary.csv"
    architecture = _csv(architecture_path).query("budget == 64").set_index("arm")
    _, routes, paths = _fusion_replay(exp)
    routes["arm"] = routes.arm.replace({"R": MODELS["collapsed_crf"], "IRD": "IRD (RRF k=10)", "IR": "IR (RRF k=10)"})
    scores = routes[["arm", "MRR", "R@10"]].copy()
    for arm in ["taxonomy_linear", "collapsed_linear", "taxonomy_crf"]:
        scores.loc[len(scores)] = [MODELS[arm], architecture.loc[arm, "MRR"], np.nan]
    fusion = _csv(fusion_path).set_index("arm")
    np.testing.assert_allclose(scores.loc[scores.arm == "IRD (RRF k=10)", "MRR"], fusion.loc["rrf10", "MRR"], rtol=0, atol=1e-14)
    scores.loc[len(scores)] = ["IRD (RRF k=60)", fusion.loc["rrf60", "MRR"], np.nan]
    scores = scores.rename(columns={"arm": "model_or_route"})
    out.append(_table(f"{COHORT}, B64: student and fusion retrieval", scores, [architecture_path, fusion_path, *paths]))
    paired_paths = [run / "architecture/paired_comparisons.csv", run / "fusion/paired_comparisons.csv"]
    paired = pd.concat([_csv(paired_paths[0]), _csv(paired_paths[1]).query("left_arm == 'rrf10' and right_arm in ['rrf60', 'R']")], ignore_index=True)
    paired = _paired(paired)
    for old, new in {**MODELS, "rrf10": "IRD (RRF k=10)", "rrf60": "IRD (RRF k=60)"}.items():
        paired["comparison"] = paired.comparison.str.replace(old, new, regex=False)
    out.append(_table(f"{COHORT}, B64: paired student/fusion MRR differences", paired, paired_paths))
    _, union, paths = _union(run)
    overlap = int(union.loc[union.route.isin(["I", "R"]), "relevant_hits"].sum() - union.loc[union.route == "union I+R", "relevant_hits"].iloc[0])
    union = union[["route", "relevant_hits", "additional_hits", "beyond"]]
    union.loc[len(union)] = ["Both I and R", overlap, np.nan, ""]
    out.append(_table(f"{COHORT}, B64: relevant-passage hits within top 10", union, paths))
    # The tokenizer settings are recorded by NB05, so this analysis belongs
    # here rather than creating a dependency from NB04 onto a later notebook.
    out.append(_truncation(exp))
    return out


def _post_training(exp):
    base = exp / "results/06_post_training/controlled_hard_relabel_nostop_e6"
    paths = [base / "training" / arm / "epoch_metrics.csv" for arm in ARMS]
    epochs = pd.concat([_csv(path) for path in paths], ignore_index=True)
    epochs["arm"] = epochs.arm.replace(ARMS)
    curve = epochs[["arm", "epoch", "train_loss", "keep_f1", "span_f1"]].rename(columns={"train_loss": "mean_training_loss", "keep_f1": "development_KEEP_F1", "span_f1": "development_exact_span_F1"})
    out = [_table("Post-training epoch means and development F1 against original teacher labels", curve, paths)]
    out.append(_settings([path.with_name("manifest.json") for path in paths]))
    routes = {"train600_viterbi_MRR": "Viterbi", "train600_ffbs2_rrf10_MRR": "Two FFBS selections, RRF k=10", "train600_viterbi_ffbs2_rrf10_MRR": "Viterbi + two FFBS selections, RRF k=10"}
    endpoint = epochs.loc[epochs.groupby("arm").epoch.idxmax(), ["arm", *routes]].melt(id_vars="arm", var_name="route", value_name="MRR")
    endpoint["route"] = endpoint.route.replace(routes)
    out.append(_table(f"{COHORT}, B64: final post-training endpoints", endpoint, paths))
    _, targets, paths = _target_intervals(exp, base)
    targets = targets[["arm", "MRR", "ci95_low", "ci95_high"]]
    targets["arm"] = targets.arm.replace({"teacher_original": "Original teacher", "teacher_bm25_corrected": "BM25-corrected teacher"})
    out.append(_table(f"{COHORT}, B64: target MRR and marginal 95% intervals", targets, paths))
    paths = [base / "target_headroom/paired_comparisons.csv", base / "comparison/paired_endpoint_comparisons.csv"]
    paired = pd.concat([_csv(paths[0]).query("metric == 'MRR'"), _csv(paths[1]).query("registered_comparison == 'treatment_minus_control' and route == 'viterbi'")], ignore_index=True)
    paired = _paired(paired)
    paired["comparison"] = ["BM25-corrected teacher - original teacher", "BM25-corrected student - continued imitation (Viterbi)"]
    out.append(_table(f"{COHORT}, B64: paired correction MRR differences", paired, paths))
    _, sesame, paths = _sesame(exp)
    sesame = sesame.loc[sesame.text.isin(["what", "show", "sesame", "street", "researching", "about"]), ["text", "score_raw"]].rename(columns={"text": "word", "score_raw": "BM25_score"})
    if len(sesame) != 6:
        raise ValueError("The six illustrated Sesame Street word scores are incomplete")
    out.append(_table("Sesame Street example train:1340:5, first question", sesame, paths))
    return out


def _one(frame: pd.DataFrame) -> pd.Series:
    if len(frame) != 1:
        raise ValueError(f'Expected one source row, found {len(frame)}')
    return frame.iloc[0]


def _ci(values) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError('Student-t interval requires at least two observations')
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))
    return mean, mean-half, mean+half


def _dqcis_details(exp: Path) -> list:
    root = exp/'results/08_dqcis_external_ablation'
    sources = [root/'summary.csv', root/'metrics_by_query.csv.gz', root/'paired_comparisons.csv']
    summary, metrics, paired = [pd.read_csv(p) for p in sources]
    # Same pointwise bootstrap as plot_topiocqa_efficiency.py, computed from
    # NB08's own per-query results without requiring later figure files.
    with sources[1].open('rb') as stream:
        source_key = hashlib.file_digest(stream, 'sha256').hexdigest()
    seed = int.from_bytes(hashlib.sha256(
        f'dqcis_fusion_bar_ci_v1\0{source_key}'.encode('utf-8')
    ).digest()[:8], 'big')
    values = metrics.pivot(index='sample_id', columns='system', values='MRR').sort_index()
    if values.empty or values.isna().any().any():
        raise ValueError('DQ-CIS systems must contain the same nonempty query population')
    rng, replicates = np.random.default_rng(seed), 10_000
    draws = {system: np.empty(replicates) for system in values}
    for start in range(0, replicates, 500):
        indices = rng.integers(0, len(values), size=(min(500, replicates-start), len(values)))
        for system in values:
            draws[system][start:start+len(indices)] = values[system].to_numpy()[indices].mean(axis=1)
    checked = []
    original_ids = None
    for _, row in summary.iterrows():
        group = metrics.loc[metrics.system.eq(row.system)].sort_values('sample_id')
        assert len(group) == row.n and not group.sample_id.duplicated().any()
        if original_ids is None:
            original_ids = group.sample_id.to_numpy()
        else:
            np.testing.assert_array_equal(original_ids, group.sample_id.to_numpy())
        mean = group.MRR.mean()
        np.testing.assert_allclose(mean, row.MRR, atol=1e-12, rtol=0)
        low, high = np.quantile(draws[row.system], (.025, .975))
        checked.append({'system': row.system, 'queries': int(row.n), 'ranking_depth': int(row.mean_depth),
                        'MRR': mean, 'ci95_low': low, 'ci95_high': high,
                        'nDCG@3': row['nDCG@3'], 'Recall@10': row['R@10'], 'Recall@100': row['R@100']})
    for _, row in paired.iterrows():
        left = metrics.loc[metrics.system.eq(row.left)].set_index('sample_id').MRR
        right = metrics.loc[metrics.system.eq(row.right)].set_index('sample_id').MRR
        differences = left.sort_index() - right.sort_index()
        assert differences.notna().all() and len(differences) == row.n_queries
        np.testing.assert_allclose(differences.mean(), row.delta_mrr, atol=1e-12, rtol=0)
    return [('DQ-CIS and added query views at depth 100', pd.DataFrame(checked), sources[:2]),
            ('Paired MRR differences, with saved 95% bootstrap intervals', paired, sources)]


def _shapley(level: dict) -> tuple[dict, dict]:
    players = level['players']
    n = len(players)
    assert len(level['coalitions']) == 2**n
    values = {}
    for mask, route in level['coalitions'].items():
        rank = route['target_rank']
        rr = 0.0 if rank is None else 1.0/rank
        assert math.isclose(rr, route['RR'], abs_tol=1e-12)
        values[int(mask, 2)] = rr
    result = {}
    for index, name in enumerate(players):
        bit = 1 << (n-index-1)
        result[name] = math.fsum((values[mask | bit]-value)/(n*math.comb(n-1, mask.bit_count()))
                                for mask, value in values.items() if not mask & bit)
        assert math.isclose(result[name], level['shap'][name], abs_tol=1e-11)
    assert math.isclose(sum(result.values()), values[2**n-1]-values[0], abs_tol=1e-11)
    return result, {'players': n, 'coalitions': len(values), 'empty_RR': values[0],
                    'full_RR': values[2**n-1], 'sum_shapley': sum(result.values())}


def _red_cross_details(exp: Path) -> list:
    root = exp/'results/09_explainability'
    source = root/'games/dev_203_8.json'
    game = json.loads(source.read_text())
    token = game['token']
    if len(token['players']) != 13:
        raise ValueError(
            "This additional NB09 comparison describes the historical Red Cross selection of 13 words. "
            "The current saved game uses a different selection; inspect the original NB09 "
            "case results instead of treating this run as the same historical word-ablation case."
        )
    attribution, checks = [], []
    for name in ['turn', 'qa_all', 'qa', 'token']:
        values, check = _shapley(game[name])
        checks.append({'game': name, **check})
        for player, value in values.items():
            attribution.append({'game': name, 'player': player, 'Shapley_value': value})
    metrics_path = exp/'results/07a_topiocqa_final/backends/bm25/route_metrics_by_query.csv'
    metric = pd.read_csv(metrics_path, usecols=['system', 'arm', 'budget', 'sample_id', 'MRR', 'target_rank'])
    metric = metric.loc[metric.sample_id.eq(game['sample_id']) & metric.system.eq('pretrained')]
    full = _one(metric.loc[metric.arm.eq('I') & metric.budget.eq(512)])
    inputs_path = exp/'results/07a_topiocqa_final/query_bundle/serialization/main_inputs.csv'
    inputs = pd.read_csv(inputs_path, usecols=['sample_id', 'budget', 'arm', 'input_length'])
    selected = _one(inputs.loc[inputs.sample_id.eq(game['sample_id']) & inputs.budget.eq(64) & inputs.arm.eq('pretrained_R')])
    routes = [dict(condition='I512', RR=full.MRR, target_rank=full.target_rank),
              dict(condition='I64', RR=game['route_I']['RR'], target_rank=game['route_I']['target_rank']),
              dict(condition='R64', RR=game['route_R']['RR'], target_rank=game['route_R']['target_rank']),
              dict(condition='V1 projection, turns 4 and 5', **{k:game['turn']['coalitions']['0001100'][k] for k in ['RR','target_rank']}),
              dict(condition='V2 projection, answer fields 4 and 5', **{k:game['qa_all']['coalitions']['00000001010000'][k] for k in ['RR','target_rank']})]
    for word in ['humanitarian', 'based']:
        matching = [name for name in token['players'] if name.split(':',1)[-1] == word]
        if len(matching) != 1:
            raise ValueError(
                f"NB09's historical word-ablation comparison requires exactly one selected {word!r}. "
                "The current saved selection differs; its word-ablation comparison is not "
                "the historical thesis case. Inspect the original NB09 case results."
            )
        mask = ''.join('0' if name == matching[0] else '1' for name in token['players'])
        route = token['coalitions'][mask]
        routes.append({'condition':f'R64 without {word}', 'RR':route['RR'], 'target_rank':route['target_rank']})
    context = pd.DataFrame([{'sample_id':game['sample_id'], 'previous_turns':len(game['history']),
                             'selected_words':len(token['players']), 'rewriter_input_tokens':int(selected.input_length),
                             'current_query':game['current_query']}])
    terms_path = root/'bm25_terms.json'
    terms = json.loads(terms_path.read_text())['samples'][game['sample_id']]
    term_rows = [{'route':route, 'word':r['text'], 'gold_document_BM25_weight':r['weight']}
                 for route in ['route_I','route_R'] for r in terms[route]['tokens']
                 if r['weight'] > 0 or r['text'].lower() == 'auschwitz']
    frames = [('Red Cross case and serialized input', context, [source, inputs_path]),
              ('Reciprocal rank and the single-word ablations', pd.DataFrame(routes), [source, metrics_path]),
              ('Exact game sizes and Shapley efficiency checks', pd.DataFrame(checks), [source]),
              ('Turn and query/answer-field attributions', pd.DataFrame(attribution).query("game in ['turn', 'qa_all']"), [source]),
              ('Word attributions for the thirteen selected words', pd.DataFrame(attribution).query("game == 'token'"), [source]),
              ('Positive lexical contributions and the Auschwitz contrast', pd.DataFrame(term_rows), [terms_path])]
    return frames


def _efficiency_details(exp: Path) -> list:
    root = exp/'results/10_efficiency_analysis'
    runs_path = root/'latency/reproducibility_route_components_runs.csv'
    summary_path = root/'latency/reproducibility_route_components_summary.csv'
    meta_path = root/'latency/reproducibility_manifest.json'
    runs, stored = pd.read_csv(runs_path), pd.read_csv(summary_path)
    meta = json.loads(meta_path.read_text())
    keys = [('I',64),('I',512),('R',64),('D',64),('IRD',64)]
    reference = runs.loc[runs.route.eq('I') & runs.budget.eq(512)].sort_values('independent_run')
    summaries, paired, protocols, manifest_paths = [], [], [], []
    for route,budget in keys:
        group = runs.loc[runs.route.eq(route) & runs.budget.eq(budget)].sort_values('independent_run')
        assert len(group) == meta['independent_script_invocations']
        np.testing.assert_array_equal(group.independent_run, reference.independent_run)
        mean,lo,hi = _ci(group.total_ms_per_query)
        original = _one(stored.loc[stored.route.eq(route) & stored.budget.eq(budget)])
        np.testing.assert_allclose([mean,lo,hi], original[['mean_total_ms','ci95_low_total_ms','ci95_high_total_ms']].to_numpy(dtype=float), atol=1e-9, rtol=0)
        components = [group[c].mean() for c in ['rocc_ms_per_query','t5_encoder_ms_per_query','t5_decoder_ms_per_query']]
        np.testing.assert_allclose(sum(components), mean, atol=1e-10, rtol=0)
        summaries.append(dict(route=route,budget=budget,queries=int(group.queries.iloc[0]),runs=len(group),
                              selection_ms=components[0],encoder_ms=components[1],decoder_ms=components[2],
                              total_ms=mean,ci95_low_ms=lo,ci95_high_ms=hi))
        delta,dlo,dhi = _ci(group.total_ms_per_query.to_numpy()-reference.total_ms_per_query.to_numpy())
        paired.append(dict(route=route,budget=budget,delta_ms=delta,ci95_low_ms=dlo,ci95_high_ms=dhi,
                           change_percent=100*delta/reference.total_ms_per_query.mean()))
    for run in sorted(runs.independent_run.unique()):
        path = root/f'latency/runs/run_{run}/manifest.json'
        manifest_paths.append(path)
        manifest = json.loads(path.read_text())
        protocols.append(dict(run=int(run),queries=manifest['population_queries'],selector_queries=manifest['rocc_processed_queries'],
                              zero_history_queries=manifest['history_zero_queries'],batch_size=manifest['batch_size'],
                              warmup_batches=manifest['warmup_batches_per_component'],shared_I_R_inputs=manifest['ird_b64']['identical_i_r_input_pairs'],
                              unique_IRD_rewriter_inputs=manifest['ird_b64']['unique_materialized_t5_inputs'],gpu=manifest['gpu']['name']))
    measured_path = root/'nsight/topiocqa_t5_encoder_summary.csv'
    component_path = root/'nsight/topiocqa_t5_encoder_component_metrics.csv'
    measured, components = pd.read_csv(measured_path), pd.read_csv(component_path)
    resource, allocations = {}, []
    for route,budget in [('I',64),('I',512),('R',64)]:
        row = _one(measured.loc[measured.route.eq(route) & measured.budget_label.eq(f'B{budget}')])
        sub = components.loc[components.route.eq(route) & components.budget_label.eq(f'B{budget}')]
        np.testing.assert_allclose([sub.fp32_flops.sum(),sub['dram__bytes.sum'].sum()],
                                  [row.fp32_flops,row.kernel_replay_dram_bytes],atol=1e-4,rtol=1e-12)
        resource[(route,budget)] = dict(route=route,budget=budget,GFLOP=row.fp32_flops/1e9,
                                        DRAM_GB=row.range_replay_dram_bytes/1e9,kind='measured encoder profile')
        for _,comp in sub.iterrows():
            allocations.append(dict(route=route,budget=budget,component=comp.component,
                                    measured_GFLOP=comp.fp32_flops/1e9,cold_kernel_DRAM_GB=comp['dram__bytes.sum']/1e9,
                                    estimated_range_DRAM_GB=row.range_replay_dram_bytes/1e9*comp['dram__bytes.sum']/row.kernel_replay_dram_bytes))
    resource[('D',64)] = dict(route='D',budget=64,GFLOP=0.,DRAM_GB=0.,kind='no T5 call; selector excluded')
    resource[('IRD',64)] = dict(route='IRD',budget=64,GFLOP=sum(resource[k]['GFLOP'] for k in [('I',64),('R',64)]),
                                DRAM_GB=sum(resource[k]['DRAM_GB'] for k in [('I',64),('R',64)]),kind='sum of I64 and R64 encoder proxies')
    for row in resource.values():
        row['GFLOP_change_percent'] = 100*(row['GFLOP']/resource[('I',512)]['GFLOP']-1)
        row['DRAM_change_percent'] = 100*(row['DRAM_GB']/resource[('I',512)]['DRAM_GB']-1)
    config_path = exp/'model/IterCQR/IterCQR Model/config.json'
    config = json.loads(config_path.read_text())
    theory = []
    for route,budget in [('I',512),('R',64)]:
        row = _one(measured.loc[measured.route.eq(route) & measured.budget_label.eq(f'B{budget}')])
        n,batch,d,ff,layers = int(row.sequence_length),int(row.batch_size),config['d_model'],config['d_ff'],config['num_layers']
        estimate = batch*layers*(8*n*d*d+4*n*n*d+4*n*d*ff)/1e9
        theory.append(dict(route=route,budget=budget,n=n,batch_size=batch,d_model=d,d_ff=ff,encoder_layers=layers,analytic_GFLOP=estimate,measured_GFLOP=row.fp32_flops/1e9))
    return [
        ('Five saved independent invocations and shared-input counts',pd.DataFrame(protocols),[meta_path,*manifest_paths]),
        ('Observed model-processing times, milliseconds per query',runs,[runs_path]),
        ('Mean component times and 95% Student-t intervals',pd.DataFrame(summaries),[runs_path,summary_path]),
        ('Paired differences relative to I512',pd.DataFrame(paired),[runs_path]),
        ('Encoder work and DRAM traffic per batch of 16',pd.DataFrame([resource[k] for k in keys]),[measured_path]),
        ('Component work and explicitly estimated DRAM allocation',pd.DataFrame(allocations),[measured_path,component_path]),
        ('Analytical encoder estimates from the saved model configuration',pd.DataFrame(theory),[config_path,measured_path]),
    ]


def _dqcis(exp: Path, stem: str) -> list:
    saved = _dqcis_details(exp)
    return [("MRR", saved[0][1][["system", "queries", "ranking_depth", "MRR", "ci95_low", "ci95_high"]], saved[0][2]),
            ("paired_MRR_differences", saved[1][1][["left", "right", "n_queries", "delta_mrr", "ci95_low", "ci95_high"]], saved[1][2])]


def _red_cross(exp: Path, stem: str) -> list:
    saved = _red_cross_details(exp)
    inputs_path = exp/"results/07a_topiocqa_final/query_bundle/serialization/main_inputs.csv"
    inputs = pd.read_csv(inputs_path, usecols=["sample_id", "budget", "arm", "input_length"])
    sample_id = saved[0][1].iloc[0].sample_id
    case_inputs = inputs.loc[inputs.sample_id.eq(sample_id)]
    routes = saved[1][1].copy()
    for label, arm, budget in [("I512", "I", 512), ("I64", "I", 64), ("R64", "pretrained_R", 64)]:
        row = _one(case_inputs.loc[case_inputs.arm.eq(arm) & case_inputs.budget.eq(budget)])
        routes.loc[routes.condition.eq(label), "rewriter_input_tokens"] = int(row.input_length)
    result = [("case", saved[0][1].drop(columns=["current_query", "rewriter_input_tokens"]), saved[0][2]),
              ("reciprocal_rank", routes, [*saved[1][2], inputs_path])]
    games = saved[2][1].loc[saved[2][1].game.isin(["turn", "qa_all", "token"]),
                             ["game", "players", "empty_RR", "full_RR"]]
    result.extend([("games", games, saved[2][2]),
                   ("turn_and_field_Shapley", saved[3][1], saved[3][2]),
                   ("word_Shapley", saved[4][1].drop(columns="game"), saved[4][2])])
    terms = saved[5][1]
    result.append(("Auschwitz_gold_document_BM25_weight", terms.loc[terms.word.str.lower().eq("auschwitz")], saved[5][2]))
    return result


def _efficiency(exp: Path, stem: str) -> list:
    saved = _efficiency_details(exp)
    protocol = saved[0][1].drop(columns=["run", "gpu"]).drop_duplicates()
    if len(protocol) != 1:
        raise ValueError("Latency runs do not share one protocol")
    protocol.insert(0, "independent_runs", len(saved[0][1]))
    paired = saved[3][1]
    paired = paired.loc[~(paired.route.eq("I") & paired.budget.eq(512))].copy()
    paired.insert(2, "reference", "I512")
    allocations = saved[5][1].drop(columns="cold_kernel_DRAM_GB").copy()
    allocations["component"] = allocations.component.replace({"embedding": "other", "layernorm_residual_other": "other"})
    allocations = allocations.groupby(["route", "budget", "component"], as_index=False).sum(numeric_only=True)
    np.testing.assert_allclose(allocations.measured_GFLOP.sum(), saved[5][1].measured_GFLOP.sum(), atol=1e-10)
    allocations["kind"] = "measured encoder work; estimated DRAM allocation"
    for component in allocations.component.unique():
        parts = allocations.loc[allocations.component.eq(component) & allocations.budget.eq(64)]
        allocations.loc[len(allocations)] = ["IRD", 64, component, parts.measured_GFLOP.sum(),
                                             parts.estimated_range_DRAM_GB.sum(), "sum of I64 and R64 proxies"]
        allocations.loc[len(allocations)] = ["D", 64, component, 0., 0., "no T5 call; selector excluded"]
    allocations = allocations.rename(columns={"measured_GFLOP": "GFLOP"})
    theory = saved[6][1].drop(columns="measured_GFLOP")
    lengths = theory.set_index("route")["n"]
    ratios = pd.DataFrame([{"reference": "I512", "selection": "R64", "linear_length_ratio": float(lengths["I"]/lengths["R"]),
                            "quadratic_length_ratio": float((lengths["I"]/lengths["R"])**2)}])
    return [("measurement_setup", protocol, saved[0][2]),
            ("model_processing_ms_per_query", saved[2][1].drop(columns=["queries", "runs"]), saved[2][2]),
            ("paired_time_differences_vs_I512", paired, saved[3][2]),
            ("encoder_per_batch_16", saved[4][1], saved[4][2]),
            ("encoder_components_per_batch_16", allocations, saved[5][2]),
            ("analytical_encoder_work", theory, saved[6][2]),
            ("input_length_ratios", ratios, saved[6][2])]


def _read(path: Path, **kwargs) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(2, "Required local result is missing", str(path))
    return pd.read_csv(path, **kwargs)


def _close(actual, expected, context: str) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=1e-12, equal_nan=False):
        raise ValueError(f"Local result mismatch: {context}")


_CONFIG = {
    "07a_topiocqa_final_retrieval": ("07a_topiocqa_final", "TopiOCQA Dev", "topiocqa"),
    "07b_qrecc_ood_retrieval": ("07b_qrecc_ood", "QReCC Test", "qrecc"),
}


_SYSTEM = {"pretrained": "base", "imitation_control_e6": "imitation", "bm25_treatment_e6": "corrected-label"}


_METRICS = ["MRR", "nDCG@3", "R@10", "R@100", "R@1000"]


def _retrieval_details(exp: Path, stem: str) -> list[tuple[str, pd.DataFrame, list[Path]]]:
    if stem not in _CONFIG:
        raise ValueError(f"Unsupported retrieval notebook: {stem}")
    exp = Path(exp).resolve()
    folder, dataset, slug = _CONFIG[stem]
    root = exp / "results" / folder
    overview = exp / "results/10_efficiency_analysis/figures"
    thesis = exp / "results/10_efficiency_analysis/figures/data"
    records, sources = defaultdict(list), defaultdict(list)

    def add(name, frame, *paths):
        frame = pd.DataFrame(frame).copy()
        if frame.empty:
            return
        if "dataset" not in frame:
            frame.insert(0, "dataset", dataset)
        records[name].append(frame)
        sources[name].extend(paths)

    quality_path = overview / "quality_bar_bootstrap_ci.csv"
    quality = _read(quality_path)
    quality = quality.loc[quality.dataset.eq(dataset) & quality.system.eq("pretrained")]
    for backend in ("bm25", "ance"):
        retriever = backend.upper()
        base = root / "backends" / backend
        summary_path = base / "route_summary.csv"
        summary = _read(summary_path)

        def mean(arm, budget=64, metric="MRR", system="pretrained"):
            return _one(summary.loc[summary.system.eq(system) & summary.arm.eq(arm) & summary.budget.eq(budget)])[metric]

        score_rows = []
        for _, row in quality.loc[quality.backend.eq(retriever)].iterrows():
            arm = row.route.replace("IRD", "I+R+D")
            for metric in _METRICS:
                # The budget figure plots only MRR. Other metrics use the
                # four B64 routes and the I512 reference, not all budgets.
                if metric != "MRR" and not (row.budget == 64 or (row.budget == 512 and row.route == "I")):
                    continue
                _close(row[metric], mean(arm, row.budget, metric), "compact route mean")
                score_rows.append(dict(retriever=retriever, checkpoint="base", route=row.route,
                    budget=int(row.budget), n=int(row.n), metric=metric, mean=row[metric],
                    ci95_low=row[metric+"_ci95_low"], ci95_high=row[metric+"_ci95_high"]))
        query_path = overview / "current_query_only_mrr_bootstrap_ci.csv"
        queries = _read(query_path)
        query = _one(queries.loc[queries.dataset.str.startswith(dataset) & queries.backend.eq(retriever)])
        score_rows.append(dict(retriever=retriever, checkpoint="base", route="query-only", budget=None,
            n=int(query.n), metric="MRR", mean=query.MRR, ci95_low=query.MRR_ci95_low, ci95_high=query.MRR_ci95_high))
        # G+2S is only used at B64 in the thesis; its other budgets/metrics
        # are deliberately excluded. Post-trained R means occur in the text
        # and also supply the Viterbi references for the FFBS figure.
        g2s = _one(summary.loc[summary.system.eq("itercqr") & summary.arm.eq("IterCQR-G+2S-RRF10") & summary.budget.eq(64)])
        score_rows.append(dict(retriever=retriever, checkpoint="IterCQR", route="G+2S", budget=64,
            n=int(g2s.n), metric="MRR", mean=g2s.MRR))
        checkpoints = ["imitation_control_e6", "bm25_treatment_e6"] if slug == "topiocqa" else ["bm25_treatment_e6"]
        for checkpoint in checkpoints:
            row = _one(summary.loc[summary.system.eq(checkpoint) & summary.arm.eq("R") & summary.budget.eq(64)])
            score_rows.append(dict(retriever=retriever, checkpoint=_SYSTEM[checkpoint], route="R", budget=64,
                n=int(row.n), metric="MRR", mean=row.MRR))
        add("retrieval_scores", score_rows, quality_path, summary_path, query_path)

        relative = []
        for route in ["I", "R", "D", "IRD", "query-only"]:
            value = query.MRR if route == "query-only" else mean(route.replace("IRD", "I+R+D"))
            relative.append(dict(retriever=retriever, candidate=route+("64" if route!="query-only" else ""),
                reference="I512", metric="MRR", relative_percent=100*(value/mean("I",512)-1)))
        if slug == "topiocqa":
            relative.append(dict(retriever=retriever, candidate="I128", reference="I256", metric="MRR",
                relative_percent=100*(mean("I",128)/mean("I",256)-1)))
        for metric in _METRICS[1:]:
            for route in ["I", "R", "D", "IRD"]:
                relative.append(dict(retriever=retriever, candidate=route+"64", reference="I512", metric=metric,
                    relative_percent=100*(mean(route.replace("IRD", "I+R+D"),metric=metric)/mean("I",512,metric)-1)))
        add("relative_retrieval_changes", relative, summary_path, query_path)
        recovery = []
        for metric in (["MRR", "R@10", "R@100"] if slug == "topiocqa" else ["R@10", "R@100"]):
            row = dict(retriever=retriever, route="R64", metric=metric,
                retained_I512_percent=100*mean("R",metric=metric)/mean("I",512,metric))
            if slug == "topiocqa":
                row["recovered_I64_loss_percent"] = 100*(mean("R",metric=metric)-mean("I",metric=metric))/(mean("I",512,metric)-mean("I",metric=metric))
            recovery.append(row)
        add("selected_route_recovery", recovery, summary_path)

        paired_path = thesis / "results_retrieval_paired_comparisons.csv"
        paired = _read(paired_path)
        paired = paired.loc[paired.panel.eq(slug+"_"+backend)]
        add("retrieval_paired_comparisons", paired[["candidate", "reference", "metric", "n", "delta", "ci95_low", "ci95_high"]].assign(retriever=retriever), paired_path)
        path = base / ("primary_comparisons.csv" if slug == "topiocqa" else "paired_comparisons.csv")
        comparisons = _read(path)
        g2s_pair = comparisons.loc[comparisons.metric.eq("MRR") & comparisons.left_system.eq("pretrained") & comparisons.left_arm.eq("I+R+D") & comparisons.right_arm.eq("IterCQR-G+2S-RRF10") & comparisons.right_budget.eq(64)]
        add("g2s_fusion_comparison", g2s_pair[["n", "delta", "ci95_low", "ci95_high"]].assign(retriever=retriever, candidate="IRD64", reference="G+2S64", metric="MRR"), path)
        post = comparisons.loc[comparisons.metric.eq("MRR") & comparisons.left_system.isin(["imitation_control_e6", "bm25_treatment_e6"]) & comparisons.left_budget.eq(64) & comparisons.right_budget.eq(64)]
        post = post.loc[post.right_system.eq("pretrained") | ((slug == "qrecc") & post.right_system.eq("imitation_control_e6"))]
        post = post[["left_system", "right_system", "left_arm", "n", "delta", "ci95_low", "ci95_high"]].rename(columns={"left_system":"candidate_checkpoint", "right_system":"reference_checkpoint", "left_arm":"route"})
        for col in ["candidate_checkpoint", "reference_checkpoint"]:
            post[col] = post[col].map(_SYSTEM)
        add("posttraining_mrr_changes", post.assign(retriever=retriever, budget=64), path)

        depth_path = overview / "depth_strata_mrr_bootstrap_ci.csv"
        depth = _read(depth_path)
        depth = depth.loc[depth.dataset.str.startswith(dataset) & depth.backend.eq(retriever)]
        omitted_bin = "d15_plus" if slug == "topiocqa" else "d11_14"
        shown_depth = depth.loc[~depth.depth_bin.eq(omitted_bin), ["depth_bin", "system_key", "n", "MRR", "MRR_ci95_low", "MRR_ci95_high"]].rename(columns={"system_key":"route"})
        shown_depth["route"] = shown_depth.route.replace({"R":"R64", "IRD":"IRD64"})
        add("history_depth_mrr", shown_depth.assign(retriever=retriever), depth_path)
        if backend == "bm25":
            omitted = depth.loc[depth.depth_bin.eq(omitted_bin), ["depth_bin", "n"]].drop_duplicates()
            add("omitted_depth_group", omitted, depth_path)

        saturation_path = thesis / "results_input_compression_512_vs_64.csv"
        saturation = _read(saturation_path)
        saturation = saturation.loc[saturation.dataset.eq(dataset) & saturation.backend.eq(retriever)]
        _close(_one(saturation).delta, mean("R",512)-mean("R",64), "R512 minus R64")
        add("selected_input_budget_mrr_change", saturation[["n", "delta", "ci95_low", "ci95_high"]].assign(retriever=retriever, candidate="R512", reference="R64"), saturation_path, summary_path)
        if slug == "topiocqa":
            _topiocqa_retrieval(add, exp, base, backend, thesis)

    _compression(add, exp, root, slug)
    if slug == "topiocqa":
        _teacher_granularity(add, exp, thesis)
    return [(name, pd.concat(frames, ignore_index=True), list(dict.fromkeys(sources[name]))) for name, frames in records.items()]


def _topiocqa_retrieval(add, exp, base, backend, thesis):
    retriever = backend.upper()
    path = thesis / f"results_retrieval_topic_switch_{backend}.csv"
    rows = []
    for _, group in _read(path).iterrows():
        for route in ["I512", "I64", "R64", "D64", "IRD64"]:
            rows.append(dict(retriever=retriever, topic_switch=bool(group.current_topic_switch), route=route,
                n=int(group.n), MRR=group[route], ci95_low=group[route+"low"], ci95_high=group[route+"high"]))
    add("topic_switch_mrr", rows, path)
    path = thesis / "results_retrieval_topic_switch_paired_comparisons.csv"
    frame = _read(path)
    add("topic_switch_paired_mrr", frame.loc[frame.backend.eq(backend), ["current_topic_switch", "candidate", "reference", "n", "delta", "ci95_low", "ci95_high"]].assign(retriever=retriever), path)
    # The plotted quantity is a paired change from the same checkpoint's
    # Viterbi R64. Its base mean is already in retrieval_scores, not repeated.
    path = thesis / "results_final_ffbs.csv"
    frame = _read(path)
    frame = frame.loc[frame.backend.eq(backend), ["system", "arm", "n", "mean", "delta", "low", "high"]].rename(columns={"system":"checkpoint", "arm":"decoder", "mean":"MRR", "delta":"delta_vs_own_Viterbi", "low":"ci95_low", "high":"ci95_high"})
    frame["checkpoint"] = frame.checkpoint.map(_SYSTEM)
    add("ffbs_mrr", frame.assign(retriever=retriever, route="R64"), path)
    path = base / "crf_decoder_comparisons.csv"
    frame = _read(path)
    frame = frame.loc[frame.metric.eq("MRR") & frame.left_system.eq("bm25_treatment_e6") & frame.arm.isin(["2-FFBS-RRF10", "Viterbi+2-FFBS-RRF10"])]
    frame = frame.loc[frame.right_system.eq("imitation_control_e6") | (frame.right_system.eq("pretrained") & frame.arm.eq("Viterbi+2-FFBS-RRF10"))]
    # The ANCE three-way treatment/control interval is not reported in thesis.
    if backend == "ance":
        frame = frame.loc[~(frame.right_system.eq("imitation_control_e6") & frame.arm.eq("Viterbi+2-FFBS-RRF10"))]
    frame = frame[["arm", "right_system", "n", "delta", "ci95_low", "ci95_high"]].rename(columns={"arm":"decoder", "right_system":"reference_checkpoint"})
    frame["reference_checkpoint"] = frame.reference_checkpoint.map(_SYSTEM)
    add("ffbs_checkpoint_mrr_changes", frame.assign(retriever=retriever, candidate_checkpoint="corrected-label", route="R64"), path)


def _compression(add, exp, root, slug):
    if slug == "topiocqa":
        path = root / "query_bundle/serialization/main_inputs.csv"
        raw = _read(path, usecols=["sample_id", "arm", "budget", "input_length"])
        selected = "pretrained_R"
    else:
        path = root / "query_bundle/query_bundle.jsonl.gz"
        raw = pd.read_json(path, lines=True, compression="gzip")
        raw = raw.loc[raw.family.eq("rocc") & raw.system.eq("pretrained")].rename(columns={"view":"arm"})
        selected = "R"
    arrays, summaries = {}, []
    for route, arm, budget in [("I512","I",512), ("I64","I",64), ("R512",selected,512), ("R64",selected,64)]:
        frame = raw.loc[raw.arm.eq(arm) & raw.budget.eq(budget), ["sample_id", "input_length"]].set_index("sample_id").sort_index()
        if frame.index.duplicated().any():
            raise ValueError("Duplicate context-length query")
        arrays[route] = frame.input_length
        summaries.append(dict(route=route, n=len(frame), mean_tokens=frame.input_length.mean(),
            maximum_tokens=int(frame.input_length.max()) if route=="R512" else None))
    if not all(arrays["I512"].index.equals(v.index) for v in arrays.values()):
        raise ValueError("Unpaired input-length population")
    add("input_length_means", summaries, path)
    n = len(arrays["I512"])
    counts = [("I64 shorter than I512", (arrays["I64"] < arrays["I512"]).sum()),
              ("R512 above64 through128", ((arrays["R512"]>64)&(arrays["R512"]<=128)).sum()),
              ("R512 above128", (arrays["R512"]>128).sum()),
              ("R512 above64", (arrays["R512"]>64).sum())]
    add("input_length_counts", [dict(condition=label, n=n, queries=int(count), percent=100*count/n) for label,count in counts], path)
    histogram = pd.DataFrame({"tokens_from":np.arange(1,513,8), "tokens_through":np.arange(8,513,8)})
    for route in ["I512", "I64", "R512"]:
        values = arrays[route].to_numpy(dtype=int)
        histogram[route] = np.histogram(values, bins=np.arange(0.5,513.5,8))[0]
        if histogram[route].sum() != n:
            raise ValueError("Incomplete input histogram")
    add("input_length_histogram", histogram, path)


def _teacher_granularity(add, exp, thesis):
    teacher = exp / "results/10_efficiency_analysis/teacher_eval"
    path = teacher / "retrieval_summary.csv"
    frame = _read(path)
    frame = frame.loc[frame.budget.eq(64), ["system", "n", "MRR", "ci95_low", "ci95_high"]]
    add("teacher_student_mrr", frame.assign(retriever="BM25", route="R64", cohort="teacher-labelled histories with at least 2 turns"), path)
    path = teacher / "paired_comparisons.csv"
    frame = _read(path)
    wanted = ["student_minus_gpt54_teacher", "bm25_teacher_minus_gpt54_teacher", "bm25_student_minus_bm25_teacher"]
    add("teacher_student_paired_mrr", frame.loc[frame.budget.eq(64)&frame.comparison.isin(wanted), ["left_system", "right_system", "n", "delta_mrr", "ci95_low", "ci95_high"]].assign(retriever="BM25", cohort="teacher-labelled histories with at least 2 turns"), path)
    path = thesis / "results_ffbs_teacher_matched.csv"
    frame = _read(path)
    frame = frame.loc[frame.student_decoder.eq("Viterbi+2-FFBS-RRF10"), ["student_system", "student_decoder", "n", "student_mrr"]]
    add("teacher_cohort_ffbs_mrr", frame.assign(retriever="BM25", route="R64", cohort="teacher-labelled histories with at least 2 turns"), path)
    # V3 is the base R64 score already in retrieval_scores. Only its distinct
    # pointwise interval used by the granularity plot is retained separately.
    path = thesis / "results_context_granularity_mrr.csv"
    frame = _read(path)
    add("granularity_mrr", frame.loc[frame.label.isin(["V1","V2"]), ["label", "n", "mean", "ci95_low", "ci95_high"]].rename(columns={"label":"granularity", "mean":"MRR"}).assign(retriever="BM25", route="R64"), path)
    add("granularity_v3_mrr_interval", frame.loc[frame.label.eq("V3"), ["n", "ci95_low", "ci95_high"]].assign(retriever="BM25", route="R64", granularity="V3"), path)
    path = thesis / "results_context_granularity_lengths.csv"
    frame = _read(path)
    add("granularity_input_length_changes", frame[["granularity", "n", "delta", "ci95_low", "ci95_high"]].assign(reference="I512"), path)
    # The V3 absolute length is already in input_length_means.
    add("granularity_input_length_means", frame.loc[frame.granularity.ne("token"), ["granularity", "n", "selected_mean"]].rename(columns={"selected_mean":"mean_tokens"}), path)
    path = exp / "results/07a_topiocqa_final/backends/bm25/granularity_mrr_comparisons.csv"
    frame = _read(path)
    frame = frame.loc[frame.budget.eq(64) & frame.arm.eq("R"), ["left_granularity", "right_granularity", "n", "delta_mrr", "ci95_low", "ci95_high"]]
    add("granularity_paired_mrr", frame.assign(retriever="BM25", route="R64"), path)
    path = exp / "results/10_efficiency_analysis/figures/granularity_projection_vs_i512_comparisons.csv"
    frame = _read(path)
    add("granularity_vs_i512_mrr", frame.loc[frame.candidate.isin(["v1 Turn", "v2 Q/A"]), ["candidate", "reference", "n", "delta_mrr", "ci95_low", "ci95_high"]].assign(retriever="BM25"), path)


def _retrieval(exp: Path, stem: str) -> list[Table]:
    """Show the completed backends even before the later figure analyses exist."""
    folder, dataset, slug = _CONFIG[stem]
    root = exp / "results" / folder
    backends = [backend for backend in ("bm25", "ance")
                if (root / "backends" / backend / "route_summary.csv").is_file()]
    if not backends:
        raise FileNotFoundError(2, "Run the notebook retrieval cells first", str(root / "backends"))
    if len(backends) == 2:
        try:
            return _retrieval_details(exp, stem)
        except FileNotFoundError as exc:
            # Only the additional, later analyses are optional. Missing own
            # experiment outputs still fail rather than masquerading as success.
            extra = exp / "results/10_efficiency_analysis"
            if not exc.filename or not Path(exc.filename).is_relative_to(extra):
                raise
            print(f"Additional retrieval comparisons are not available yet: {exc.filename}. "
                  "Run the README's teacher/figure analysis scripts and rerun this cell. "
                  "Showing the completed notebook results below.")
    else:
        print(f"Showing {backends[0].upper()}; combined comparisons also require the other "
              "retriever and the README's teacher/figure analysis scripts.")
    out = []
    for backend in backends:
        base = root / "backends" / backend
        path = base / "route_summary.csv"
        out.append((f"{dataset} / {backend.upper()}: route means", _read(path), [path]))
        path = base / ("primary_comparisons.csv" if slug == "topiocqa" else "paired_comparisons.csv")
        out.append((f"{dataset} / {backend.upper()}: paired comparisons", _read(path), [path]))

    def add(title, frame, *paths):
        out.append((title, pd.DataFrame(frame), list(paths)))

    _compression(add, exp, root, slug)
    return out


def result_tables(experiments_dir: Path, notebook_stem: str) -> list[Table]:
    """Compute display tables and retain their source paths for inspection."""
    exp = Path(experiments_dir).expanduser().resolve()
    builders = {
        "03_oracle_headroom_analysis": _oracle,
        "04_teacher": _teacher,
        "05_history_selector": _students,
        "06_post_training": _post_training,
    }
    try:
        if notebook_stem in builders:
            tables = builders[notebook_stem](exp)
        elif notebook_stem in _CONFIG:
            tables = _retrieval(exp, notebook_stem)
        elif notebook_stem == "08_dqcis_external_ablation":
            tables = _dqcis(exp, notebook_stem)
        elif notebook_stem == "09_explainability":
            tables = _red_cross(exp, notebook_stem)
        elif notebook_stem == "10_efficiency_analysis":
            tables = _efficiency(exp, notebook_stem)
        else:
            raise ValueError(f"No additional analysis for {notebook_stem}")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing input for {notebook_stem}: {exc}. Run its experiment cells first. "
            "NB10 additionally requires the README's latency and encoder-profile scripts. "
            "No model or measurement was run by this display helper."
        ) from exc
    names = set()
    for title, frame, _ in tables:
        if title in names or frame.empty or frame.columns.duplicated().any():
            raise ValueError(f"Empty or ambiguous analysis table: {title}")
        names.add(title)
    return tables


def display_results(experiments_dir: Path, notebook_stem: str) -> None:
    """Display analyses directly in a notebook; never write result files."""
    from IPython.display import Markdown, display

    for title, frame, _ in result_tables(experiments_dir, notebook_stem):
        display(Markdown(f"**{title.replace('_', ' ')}**"))
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(frame)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", help="Notebook stem, for example 10_efficiency_analysis")
    parser.add_argument("--experiments-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for title, frame, _ in result_tables(args.experiments_dir, args.notebook):
        print(f"\n{title}\n{frame.to_string(index=False)}")


if __name__ == "__main__":
    main()
