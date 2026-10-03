#!/usr/bin/env python3
"""Export saved retrieval results for native thesis plots, without model runs.

Full-population means and pointwise CIs are copied from frozen overview CSVs.
Their paired comparisons reproduce only the bootstrap calculation in
experiments/scripts/plot_topiocqa_efficiency.py::load_quality, with its exact
seeds, 10,000 resamples and 500-resample chunks. Current-topic-switch pointwise
and paired intervals are computed within each group from saved per-query RR,
using separate stable SHA256 seeds and the same resample and chunk counts.
Existing result files are read-only. Outputs are confined to
experiments/results/10_efficiency_analysis/figures/data/. Also regenerates the
saved paired R512-minus-R64 input-compression comparison from per-query RR.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/results"
FIGURES = RESULTS / "10_efficiency_analysis/figures"
OUT = FIGURES / "data"
TOPIOCQA_DEV = ROOT / "experiments/data/topiocqa/downloads/data/retriever/original/dev.json"
PANELS = (
    ("topiocqa_bm25", "TopiOCQA Dev", "BM25", "07a_topiocqa_final", 2514),
    ("topiocqa_ance", "TopiOCQA Dev", "ANCE", "07a_topiocqa_final", 2514),
    ("qrecc_bm25", "QReCC Test", "BM25", "07b_qrecc_ood", 8209),
    ("qrecc_ance", "QReCC Test", "ANCE", "07b_qrecc_ood", 8209),
)
SYSTEMS = (("I64", "I", 64), ("I512", "I", 512), ("R64", "R", 64),
           ("D64", "D", 64), ("IRD64", "IRD", 64))
METRICS = ("MRR", "nDCG@3", "R@10", "R@100", "R@1000")


def read(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def save(name, rows):
    path = OUT / f"results_retrieval_{name}.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def estimate(row, metric, prefix):
    mean = float(row[metric])
    low, high = float(row[metric + "_ci95_low"]), float(row[metric + "_ci95_high"])
    assert 0 <= low <= mean <= high <= 1
    return {prefix: mean, prefix + "minus": mean - low,
            prefix + "plus": high - mean, prefix + "low": low,
            prefix + "high": high}


def seed_for(key, metric):
    if metric == "MRR":
        return {"I64": 1062, "R64": 1061, "D64": 1063, "IRD64": 1060}[key]
    base = {"I64": 2000, "R64": 2100, "D64": 2200, "IRD64": 2300}[key]
    return base + {"nDCG@3": 500, "R@10": 0, "R@100": 1, "R@1000": 2}[metric]


def paired_rows(path, expected_n):
    arms = {"I", "R", "D", "I+R+D"}
    selected = {}
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["system"] != "pretrained" or row["arm"] not in arms:
                continue
            budget = int(row["budget"])
            if budget not in (64, 512) or (budget == 512 and row["arm"] != "I"):
                continue
            arm = row["arm"].replace("I+R+D", "IRD")
            key = f"{arm}{budget}"
            records = selected.setdefault(key, {})
            assert row["sample_id"] not in records
            records[row["sample_id"]] = [float(row[m]) for m in METRICS]
    assert set(selected) == {s[0] for s in SYSTEMS}
    reference = selected["I512"]
    assert len(reference) == expected_n
    result = {}
    for key, values in selected.items():
        assert set(values) == set(reference)
        if key == "I512":
            continue
        # Original merge preserves the candidate file order.
        delta = np.array([np.subtract(value, reference[sid]) for sid, value in values.items()])
        for column, metric in enumerate(METRICS):
            seed = seed_for(key, metric)
            rng = np.random.default_rng(seed)
            draws = np.concatenate([
                delta[:, column][rng.integers(0, expected_n, (500, expected_n))].mean(axis=1)
                for _ in range(20)
            ])
            low, high = np.quantile(draws, (0.025, 0.975))
            result[key, metric] = dict(delta=float(delta[:, column].mean()),
                low=float(low), high=float(high), seed=seed,
                marker="dagger" if low > 0 or high < 0 else "ddagger")
    return result


def main():
    global RESULTS, FIGURES, OUT, TOPIOCQA_DEV
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--figures-dir", type=Path, help="Default: RESULTS_DIR/10_efficiency_analysis/figures, produced by plot_topiocqa_efficiency.py.")
    parser.add_argument("--output-dir", type=Path, help="Default: FIGURES_DIR/data.")
    parser.add_argument("--topiocqa-dev", type=Path, default=TOPIOCQA_DEV,
                        help="Local original TopiOCQA Dev JSON from NB00b.")
    parser.add_argument("--reuse-paired", action="store_true",
                        help="Reuse the previously verified paired bootstrap export.")
    parser.add_argument("--topic-switch-only", action="store_true",
                        help="Export only current-topic-switch estimates and paired intervals.")
    args = parser.parse_args()
    RESULTS = args.results_dir.expanduser().resolve()
    FIGURES = (args.figures_dir or RESULTS / "10_efficiency_analysis/figures").expanduser().resolve()
    OUT = (args.output_dir or FIGURES / "data").expanduser().resolve()
    TOPIOCQA_DEV = args.topiocqa_dev.expanduser().resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.topic_switch_only:
        topic_switch_export()
        return
    quality = read(FIGURES / "quality_bar_bootstrap_ci.csv")
    depth = read(FIGURES / "depth_strata_mrr_bootstrap_ci.csv")
    query = read(FIGURES / "current_query_only_mrr_bootstrap_ci.csv")
    paired_export = []
    old_paired = read(OUT / "results_retrieval_paired_comparisons.csv") if args.reuse_paired else []
    depth_omissions = []
    for panel, dataset, backend, folder, n in PANELS:
        q = [r for r in quality if r["dataset"] == dataset and r["backend"] == backend
             and r["system"] == "pretrained"]
        assert len(q) == 16 and all(int(r["n"]) == n for r in q)
        lookup = {(r["route"], int(r["budget"])): r for r in q}
        source = RESULTS / folder / "backends" / backend.lower()
        summaries = read(source / "route_summary.csv")
        for (route, budget), row in lookup.items():
            original = next(r for r in summaries if r["system"] == "pretrained"
                and int(r["budget"]) == budget and r["arm"] == route.replace("IRD", "I+R+D"))
            assert all(abs(float(row[m]) - float(original[m])) < 1e-12 for m in METRICS)

        current = next(r for r in query if r["dataset"].startswith(dataset) and r["backend"] == backend)
        budget_rows = []
        for x, budget in enumerate((0, 64, 128, 256, 512)):
            out = dict(x=x, budget=budget)
            for route in ("I", "R", "D", "IRD", "query"):
                if (route == "query") == (budget == 0):
                    out.update(estimate(current if budget == 0 else lookup[route, budget], "MRR", route))
                else:
                    out.update({route + suffix: "nan" for suffix in ("", "minus", "plus", "low", "high")})
            budget_rows.append(out)
        save(f"budget_{panel}", budget_rows)

        d = [r for r in depth if r["dataset"].startswith(dataset) and r["backend"] == backend]
        bins = list(dict.fromkeys(r["depth_bin"] for r in d))
        depth_rows = []
        for x, group in enumerate(bins, 1):
            rows = [r for r in d if r["depth_bin"] == group]
            label = rows[0]["depth_label"].replace("–", "-").replace("−", "-")
            count = int(rows[0]["n"])
            out = dict(x=x, depth_bin=group, depth_label=label, n=count)
            for key in ("I512", "I64", "R", "IRD"):
                out.update(estimate(next(r for r in rows if r["system_key"] == key), "MRR", key))
            depth_rows.append(out)
        assert sum(r["n"] for r in depth_rows) == n
        omitted = depth_rows.pop()
        depth_omissions.append(dict(panel=panel, depth_bin=omitted["depth_bin"],
                                   n=omitted["n"], population_n=n))
        save(f"depth_{panel}", depth_rows)

        if args.reuse_paired:
            reused = [r for r in old_paired if r["panel"] == panel]
            assert len(reused) == 20 and all(int(r["n"]) == n for r in reused)
            paired = {(r["candidate"], r["metric"]): dict(delta=float(r["delta"]),
                low=float(r["ci95_low"]), high=float(r["ci95_high"]),
                seed=int(r["seed"]), marker=r["marker"]) for r in reused}
        else:
            paired = paired_rows(source / "route_metrics_by_query.csv", n)
        for (key, metric), p in paired.items():
            paired_export.append(dict(panel=panel, metric=metric, candidate=key,
                reference="I512", n=n, delta=p["delta"], ci95_low=p["low"],
                ci95_high=p["high"], seed=p["seed"], replicates=10000,
                marker=p["marker"], source=str(Path("experiments/results") / (source / "route_metrics_by_query.csv").relative_to(RESULTS))))
        for metric, slug in (("MRR", "mrr"), ("nDCG@3", "ndcg")):
            exported = []
            reference = float(lookup["I", 512][metric])
            for x, (key, route, budget) in enumerate(SYSTEMS, 1):
                row = lookup[route, budget]
                out = dict(x=x, system=key, n=n, reference=reference)
                out.update(estimate(row, metric, "value"))
                relative = 100 * (out["value"] / reference - 1)
                marker = "" if key == "I512" else "^{\\" + paired[key, metric]["marker"] + "}"
                out.update(relative_pct=relative, label_y=out["valuehigh"],
                    label=f"\\shortstack{{\\({out['value']:.3f}{marker}\\)\\\\\\({relative:+.1f}\\%\\)}}")
                exported.append(out)
            save(f"{slug}_{panel}", exported)

        recall_rows = []
        for x, metric in enumerate(("R@10", "R@100", "R@1000"), 1):
            out = dict(x=x, cutoff=metric.split("@")[1])
            reference = float(lookup["I", 512][metric])
            for key, route, budget in SYSTEMS:
                out.update(estimate(lookup[route, budget], metric, key))
                out[key + "relative"] = 100 * (out[key] / reference - 1)
                out[key + "labely"] = out[key + "high"]
                out[key + "label"] = "" if key == "I512" else "\\(\\" + paired[key, metric]["marker"] + "\\)"
            recall_rows.append(out)
        save(f"recall_{panel}", recall_rows)
        save(f"recall_main_{panel}", recall_rows[:2])
        # The appendix keeps the same saved estimates and intervals. Only its
        # categorical x position is reset for the single-cutoff layout.
        save(f"recall1000_{panel}", [dict(recall_rows[2], x=1)])
        print(f"Exported {panel}; all saved means reconciled, n={n}.", flush=True)
    save("paired_comparisons", paired_export)
    save("depth_omissions", depth_omissions)
    topic_switch_export()
    input_compression_export()
    for filename in ("quality_bar_bootstrap_ci.csv", "depth_strata_mrr_bootstrap_ci.csv", "current_query_only_mrr_bootstrap_ci.csv"):
        print(filename, hashlib.sha256((FIGURES / filename).read_bytes()).hexdigest())


def topic_switch_export():
    """Partition on the current sequential topic change, never future turns.

    This reproduces datasets/topiocqa.py::_add_topic_switch_columns from the
    raw dev split. Topic is the first positive-context Wikipedia title before
    `` [SEP] ``. The first query has no predecessor and belongs to no-switch.
    Only stored route-level RR is resampled. No model or retriever is invoked.
    """
    raw_path = TOPIOCQA_DEV
    raw = json.loads(raw_path.read_text())
    membership = {}
    previous = {}
    for row in sorted(raw, key=lambda r: (r["conv_id"], r["turn_id"])):
        topics = [str(c["title"]).split(" [SEP] ", 1)[0]
                  for c in row["positive_ctxs"] if c.get("title")]
        assert topics, "Missing topic must not be silently coded as no switch."
        topic = topics[0]
        conv = row["conv_id"]
        sid = f"dev:{conv}:{row['turn_id']}"
        assert sid not in membership
        membership[sid] = int(conv in previous and topic != previous[conv])
        previous[conv] = topic
    assert len(membership) == 2514
    counts = {flag: sum(v == flag for v in membership.values()) for flag in (0, 1)}
    assert counts == {0: 1842, 1: 672}
    paired_export = []
    for backend in ("bm25", "ance"):
        source = RESULTS / "07a_topiocqa_final/backends" / backend / "route_metrics_by_query.csv"
        selected = {}
        for row in read(source):
            route = row["arm"].replace("I+R+D", "IRD")
            key = f"{route}{row['budget']}"
            if row["system"] == "pretrained" and key in ("I512", "I64", "R64", "D64", "IRD64"):
                records = selected.setdefault(key, {})
                assert row["sample_id"] not in records
                records[row["sample_id"]] = float(row["MRR"])
        assert set(selected) == {"I512", "I64", "R64", "D64", "IRD64"}
        output = []
        for flag in (0, 1):
            out = dict(x=flag + 1, current_topic_switch=flag, n=counts[flag])
            ids = [sid for sid in sorted(membership) if membership[sid] == flag]
            reference = np.array([selected["I512"][sid] for sid in ids])
            for key, records in selected.items():
                assert set(records) == set(membership)
                values = np.array([records[sid] for sid in ids])
                seed_key = f"thesis-current-topic-switch|{backend}|{flag}|{key}|MRR"
                seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big")
                rng = np.random.default_rng(seed)
                draws = np.concatenate([values[rng.integers(0, len(values), (500, len(values)))].mean(axis=1)
                                        for _ in range(20)])
                low, high = np.quantile(draws, (0.025, 0.975))
                mean = float(values.mean())
                out.update({key: mean, key + "minus": mean - low, key + "plus": high - mean,
                            key + "low": low, key + "high": high, key + "seed": seed})
                out[key + "label"] = ""
                if key != "I512":
                    # Resample matched per-query differences within this group.
                    # Pointwise CIs and full-population paired CIs do not apply.
                    delta = values - reference
                    paired_seed_key = f"thesis-current-topic-switch-paired|{backend}|{flag}|{key}|I512|MRR"
                    paired_seed = int.from_bytes(hashlib.sha256(paired_seed_key.encode()).digest()[:8], "big")
                    paired_rng = np.random.default_rng(paired_seed)
                    paired_draws = np.concatenate([
                        delta[paired_rng.integers(0, len(delta), (500, len(delta)))].mean(axis=1)
                        for _ in range(20)
                    ])
                    paired_low, paired_high = np.quantile(paired_draws, (0.025, 0.975))
                    marker = "dagger" if paired_low > 0 or paired_high < 0 else "ddagger"
                    out[key + "label"] = "\\(\\" + marker + "\\)"
                    paired_export.append(dict(
                        backend=backend, current_topic_switch=flag, metric="MRR",
                        candidate=key, reference="I512", n=len(delta),
                        delta=float(delta.mean()), ci95_low=float(paired_low),
                        ci95_high=float(paired_high), seed=paired_seed,
                        replicates=10000, marker=marker,
                        source=str(Path("experiments/results") / source.relative_to(RESULTS)),
                        membership_source=str(raw_path.relative_to(RESULTS.parent.parent)) if raw_path.is_relative_to(RESULTS.parent.parent) else str(raw_path),
                    ))
            output.append(out)
        save(f"topic_switch_{backend}", output)
        print(f"Current topic switch {backend}: no switch n={counts[0]}, switch n={counts[1]}")
    save("topic_switch_membership", [dict(sample_id=sid, current_topic_switch=flag)
                                      for sid, flag in sorted(membership.items())])
    assert len(paired_export) == 16
    save("topic_switch_paired_comparisons", paired_export)


def input_compression_export() -> None:
    """Reproduce the saved R512-R64 paired bootstrap (seed 2000, 10k/500)."""
    rows = []
    for _, dataset, backend, folder, expected_n in PANELS:
        source = RESULTS / folder / "backends" / backend.lower() / "route_metrics_by_query.csv"
        selected = {64: {}, 512: {}}
        for row in read(source):
            budget = int(row["budget"])
            if row["system"] == "pretrained" and row["arm"] == "R" and budget in selected:
                assert row["sample_id"] not in selected[budget]
                selected[budget][row["sample_id"]] = float(row["MRR"])
        assert len(selected[64]) == expected_n and set(selected[64]) == set(selected[512])
        ids = sorted(selected[64])
        candidate = np.array([selected[512][sid] for sid in ids])
        reference = np.array([selected[64][sid] for sid in ids])
        delta = candidate - reference
        rng = np.random.default_rng(2000)
        draws = np.concatenate([
            delta[rng.integers(0, expected_n, (500, expected_n))].mean(axis=1)
            for _ in range(20)
        ])
        low, high = np.quantile(draws, (0.025, 0.975))
        rows.append(dict(dataset=dataset, backend=backend, system="pretrained", arm="R",
            candidate_budget=512, reference_budget=64, n=expected_n,
            candidate_mrr=float(candidate.mean()), reference_mrr=float(reference.mean()),
            delta=float(delta.mean()), ci95_low=float(low), ci95_high=float(high),
            seed=2000, replicates=10000, chunk_size=500,
            source=str(Path("experiments/results") / source.relative_to(RESULTS))))
    with (OUT / "results_input_compression_512_vs_64.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
