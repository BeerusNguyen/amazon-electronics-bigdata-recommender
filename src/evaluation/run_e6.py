"""E6 command line. Run from the repository root: ``python -m src.evaluation.run_e6 ACTION``.

Actions, in the order a full run uses them
  evaluate --phase P   T25-T28 for one phase; writes or refreshes that phase's rows
  diagnose             ALS rating error, item support (both phases) and the E5 tuning grid
  seed-stability       locked ALS config retrained with several seeds (Validation only)
  ablation             locked config with larger regParam and implicitPrefs (Validation only)
  compare              model_comparison.csv, runtime.csv and the cross-phase reliability checks
  figures              every chart in results/e6/figures
  report               execute the notebook and export results/e6/E6_evaluation_report.html

Output layout (one file per topic, phase is a column):
  results/e6/model_comparison.csv  metrics.csv  significance.csv  coverage.csv  fallback.csv
             cohorts.csv  beyond_accuracy.csv  reliability_checks.csv  protocol.json  runtime.csv
             diagnostics/  figures/  E6_evaluation_report.html
  outputs/e6/per_user/{phase}_{method}.parquet   per-user tables (gitignored)

Nothing here changes E4 data or E5 outputs, and Test is never used to choose a configuration.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import als_diagnostics as diag
from src.evaluation.beyond_accuracy import beyond_accuracy, category_level
from src.evaluation.coverage import coverage_by_segment, fallback_effectiveness
from src.evaluation.per_user_metrics import QUALITY_COLUMNS, per_user_ranking_metrics
from src.evaluation.protocol import (
    build_ground_truth,
    load_snapshot,
    protocol_summary,
    recommendation_integrity,
    restrict_ground_truth,
    temporal_integrity,
)
from src.evaluation.significance import bootstrap_ci, holm, paired_bootstrap, sign_flip_test

METHODS = ["random", "popularity", "als", "als_plus_popularity"]
E5_FILES = {
    "popularity": ("popularity-{phase}", "popularity-{phase}.json"),
    "als": ("als-only-{phase}", "als-only-{phase}.json"),
    "als_plus_popularity": ("als-{phase}", "als-{phase}.json"),
}
COMPARISONS = [("popularity", "random"), ("als", "random"), ("als", "popularity"),
               ("als_plus_popularity", "popularity"), ("als_plus_popularity", "als")]
SENSITIVITY_K = [1, 3, 5]


def _save_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10g")


def _upsert(frame, path, phase):
    """Replace this phase's rows in a shared file, keep the other phase."""
    if path.is_file():
        old = pd.read_csv(path)
        frame = pd.concat([old[old["phase"] != phase], frame], ignore_index=True)
    _save_csv(frame.sort_values("phase", kind="stable", key=lambda s: s != "validation"), path)


def _upsert_json(path, phase, value):
    data = json.loads(path.read_text()) if path.is_file() else {}
    data[phase] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def random_recommendations(users, history, k=10, seed=42):
    """Seeded uniform sample of unseen catalog items: the floor any useful model must beat."""
    rng = np.random.default_rng(seed)
    catalog = np.sort(history["product_id"].unique())
    users = np.sort(pd.unique(pd.Series(list(users), dtype="object")))
    draws = 3 * k
    frame = pd.DataFrame({"user_id": np.repeat(users, draws),
                          "product_id": catalog[rng.integers(0, catalog.size, users.size * draws)],
                          "order": np.tile(np.arange(draws), users.size)})
    seen = history[["user_id", "product_id"]].drop_duplicates().assign(_seen=True)
    frame = frame.merge(seen, on=["user_id", "product_id"], how="left")
    frame = (frame[frame["_seen"].isna()].drop_duplicates(["user_id", "product_id"])
             .sort_values(["user_id", "order"], kind="stable"))
    frame["rank"] = frame.groupby("user_id").cumcount() + 1
    return frame.loc[frame["rank"] <= k, ["user_id", "product_id", "rank"]].reset_index(drop=True)


def _per_user_quality(recs, ground_truth, users, k, args, phase, method, scope, relevance):
    """Metric rows for one setting; CI only for the headline setting (K = 10, rating >= 4)."""
    table = per_user_ranking_metrics(recs, ground_truth, users, k)
    headline = k == args.k and relevance == f">={args.relevance_threshold}"
    rows = []
    for metric in QUALITY_COLUMNS:
        values = table[metric].to_numpy()
        mean, low, high = (bootstrap_ci(values, args.bootstrap, seed=args.seed) if headline
                           else (float(values.mean()) if len(values) else 0.0, np.nan, np.nan))
        rows.append({"phase": phase, "method": method, "scope": scope, "relevance": relevance,
                     "k": k, "metric": metric, "mean": mean, "ci95_low": low, "ci95_high": high,
                     "scored_users": len(table)})
    return table, rows


def _cohorts(recs, ground_truth, users, history, k, phase, method):
    counts = history.groupby("user_id").size()
    gt = ground_truth.assign(history_count=ground_truth["user_id"].map(counts).fillna(0))
    cohorts = {
        "all": gt,
        "warm_user/warm_item": gt[gt.warm_user & gt.warm_item],
        "cold_user/warm_item": gt[~gt.warm_user & gt.warm_item],
        "warm_user/cold_item": gt[gt.warm_user & ~gt.warm_item],
        "cold_user/cold_item": gt[~gt.warm_user & ~gt.warm_item],
        "history_1_4": gt[gt.history_count.between(1, 4)],
        "history_ge_5": gt[gt.history_count >= 5],
    }
    rows = []
    for name, part in cohorts.items():
        table = per_user_ranking_metrics(recs, part, users, k)
        rows.append({"phase": phase, "method": method, "cohort": name,
                     "scored_users": len(table), "targets": len(part),
                     **{m: float(table[m].mean()) if len(table) else 0.0
                        for m in ("recall", "ndcg", "hit")}})
    return rows


def _significance(per_user, phase, args):
    rows = []
    for left, right in COMPARISONS:
        a = per_user[left].set_index("user_id")
        b = per_user[right].set_index("user_id").reindex(a.index)
        for metric in ("recall", "ndcg"):
            diff, low, high = paired_bootstrap(a[metric], b[metric], args.bootstrap, seed=args.seed)
            rows.append({"phase": phase, "method_a": left, "method_b": right, "metric": metric,
                         "mean_a": float(a[metric].mean()), "mean_b": float(b[metric].mean()),
                         "mean_diff": diff, "diff_ci95_low": low, "diff_ci95_high": high,
                         "users_a_better": int((a[metric] > b[metric]).sum()),
                         "users_b_better": int((a[metric] < b[metric]).sum()),
                         "p_value": sign_flip_test(a[metric], b[metric], args.permutations,
                                                   seed=args.seed)})
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm(frame["p_value"].to_numpy())
    frame["significant"] = frame["p_holm"] < 0.05
    return frame


def evaluate(args):
    """T25-T28 for one phase."""
    phase, k, threshold = args.phase, args.k, args.relevance_threshold
    out = args.results
    history, held_out = load_snapshot(args.data_dir, phase)
    users = held_out["user_id"].unique()
    logging.info("%s: history=%d held_out=%d target_users=%d", phase, len(history),
                 len(held_out), len(users))
    ground_truth = build_ground_truth(held_out, history, threshold)
    strict = build_ground_truth(held_out, history, threshold + 1)
    temporal = temporal_integrity(history, held_out)
    _upsert_json(out / "protocol.json", phase, {
        **protocol_summary(held_out, history, ground_truth, threshold, k), "temporal": temporal})

    tables = {"random": random_recommendations(users, history, k, args.seed)}
    e5_metrics = {}
    for method, (rec_dir, metric_file) in E5_FILES.items():
        tables[method] = diag.read_parquet_dir(
            args.e5_dir / "recommendations" / rec_dir.format(phase=phase))
        metric_path = args.e5_dir / "metrics" / metric_file.format(phase=phase)
        if metric_path.is_file():
            e5_metrics[method] = json.loads(metric_path.read_text())

    categories = None
    if args.metadata_dir and Path(args.metadata_dir).exists():
        categories = category_level(diag.read_parquet_dir(args.metadata_dir), level=1)

    checks = [
        {"phase": phase, "check": "temporal_order", "subject": "history vs held-out",
         "value": float(temporal["ordered"]), "status": "PASS" if temporal["ordered"] else "FAIL",
         "detail": f"history ends {temporal['history_end']}, held-out starts {temporal['held_out_start']}"},
        {"phase": phase, "check": "temporal_shared_pairs", "subject": "history vs held-out",
         "value": temporal["shared_pairs"], "status": "PASS" if temporal["shared_pairs"] == 0 else "FAIL",
         "detail": "user-item pairs present in both periods"},
    ]
    metric_rows, coverage_rows, beyond_rows, cohort_rows, per_user = [], [], [], [], {}
    for method in METHODS:
        recs = tables[method]
        violations = recommendation_integrity(recs, history, held_out, users, k)
        checks += [{"phase": phase, "check": f"integrity_{name}", "subject": method, "value": value,
                    "status": "PASS" if value == 0 else "FAIL", "detail": "violation count"}
                   for name, value in violations.items()]
        if any(violations.values()):
            _upsert(pd.DataFrame(checks), out / "reliability_checks.csv", phase)
            raise ValueError(f"{method} failed integrity checks: {violations}")

        for scope in ("all", "warm_item"):
            table, rows = _per_user_quality(recs, restrict_ground_truth(ground_truth, scope), users,
                                            k, args, phase, method, scope, f">={threshold}")
            metric_rows += rows
            if scope == "all":
                per_user[method] = table
        for cutoff in SENSITIVITY_K:
            metric_rows += _per_user_quality(recs, ground_truth, users, cutoff, args, phase,
                                             method, "all", f">={threshold}")[1]
        metric_rows += _per_user_quality(recs, strict, users, k, args, phase, method, "all",
                                         f">={threshold + 1}")[1]

        coverage = coverage_by_segment(recs, users, history, k)
        coverage.insert(0, "method", method)
        coverage.insert(0, "phase", phase)
        coverage_rows.append(coverage)
        beyond_rows.append({"phase": phase, "method": method,
                            **beyond_accuracy(recs, history, k, categories, threshold, args.seed)})
        cohort_rows += _cohorts(recs, ground_truth, users, history, k, phase, method)

        if method in e5_metrics:
            e5 = e5_metrics[method]
            cover = coverage.set_index("segment").loc["all"]
            for name, mine, theirs in (
                ("recall_at_k", per_user[method]["recall"].mean(), e5["recall_at_k"]),
                ("ndcg_at_k", per_user[method]["ndcg"].mean(), e5["ndcg_at_k"]),
                ("relevant_users", len(per_user[method]), e5["relevant_users"]),
                ("user_coverage", cover["coverage_ge1"], e5["user_coverage"]),
                ("full_k_coverage", cover["coverage_full_k"], e5["full_k_coverage"]),
            ):
                diff = abs(float(mine) - float(theirs))
                checks.append({"phase": phase, "check": f"e5_cross_check_{name}", "subject": method,
                               "value": diff, "status": "PASS" if diff < 1e-9 else "FAIL",
                               "detail": f"E6 {float(mine):.10g} vs E5 {float(theirs):.10g}"})

    args.outputs.mkdir(parents=True, exist_ok=True)
    for method, table in per_user.items():
        table.to_parquet(args.outputs / f"{phase}_{method}.parquet", index=False)

    fallback = fallback_effectiveness(tables["als"], tables["als_plus_popularity"], users,
                                      history, k, per_user["als_plus_popularity"])
    fallback.insert(0, "phase", phase)

    _upsert(pd.DataFrame(metric_rows), out / "metrics.csv", phase)
    _upsert(_significance(per_user, phase, args), out / "significance.csv", phase)
    _upsert(pd.concat(coverage_rows, ignore_index=True), out / "coverage.csv", phase)
    _upsert(fallback, out / "fallback.csv", phase)
    _upsert(pd.DataFrame(cohort_rows), out / "cohorts.csv", phase)
    _upsert(pd.DataFrame(beyond_rows), out / "beyond_accuracy.csv", phase)
    _upsert(pd.DataFrame(checks), out / "reliability_checks.csv", phase)
    logging.info("E6 %s evaluation written to %s", phase, out)


def diagnose(args):
    out = args.results / "diagnostics"
    _save_csv(pd.concat([diag.rating_prediction(args.data_dir, args.e5_dir, p)
                         for p in ("validation", "test")]), out / "als_rating_prediction.csv")
    _save_csv(pd.concat([diag.item_support(args.data_dir, args.e5_dir, p, args.k,
                                           args.relevance_threshold)
                         for p in ("validation", "test")]), out / "als_item_support.csv")
    _save_csv(diag.tuning_table(args.e5_dir), out / "als_tuning.csv")


def seed_stability(args):
    frame = diag.evaluate_variants(args.data_dir, args.config, diag.seed_variants(args.e5_dir, args.seeds),
                                   args.outputs / "scratch", args.k, args.relevance_threshold)
    _save_csv(frame, args.results / "diagnostics" / "seed_stability.csv")


def ablation(args):
    frame = diag.evaluate_variants(args.data_dir, args.config, diag.ablation_variants(args.e5_dir),
                                   args.outputs / "scratch", args.k, args.relevance_threshold)
    _save_csv(frame, args.results / "diagnostics" / "als_ablation.csv")


def _runtime(args):
    """Measured times only: ALS fit (seed run), E5 stage timers, CLI wall-clock (--timings)."""
    wall = {}
    if args.timings and Path(args.timings).is_file():
        timings = pd.read_csv(args.timings)
        wall = dict(zip(timings["step"], timings["seconds"]))
    fit = np.nan
    seeds = args.results / "diagnostics" / "seed_stability.csv"
    if seeds.is_file():
        table = pd.read_csv(seeds)
        match = table[(table["seed"] == diag.locked_parameters(args.e5_dir)["seed"])
                      & (table["method"] == "als")]
        fit = float(match["train_seconds"].iloc[0]) if len(match) else np.nan
    rows = []
    for phase in ("validation", "test"):
        path = args.e5_dir / "metrics" / f"als-{phase}.json"
        stages = json.loads(path.read_text()).get("runtime_seconds", {}) if path.is_file() else {}
        als_inf = stages.get("als_candidate_generation", np.nan) + \
            stages.get("seen_filtering_and_ranking", np.nan)
        hybrid_inf = als_inf + stages.get("popularity_fallback", np.nan) + \
            stages.get("final_assembly", np.nan)
        als_wall = wall.get("generate_validation_best" if phase == "validation" else "test_final",
                            np.nan)
        rows += [
            {"phase": phase, "method": "popularity", "train_seconds": np.nan,
             "inference_seconds": np.nan, "end_to_end_seconds": wall.get(f"popularity_{phase}", np.nan),
             "note": "fit, Top-10 and metrics in one E5 call"},
            {"phase": phase, "method": "als", "train_seconds": fit if phase == "validation" else np.nan,
             "inference_seconds": als_inf, "end_to_end_seconds": als_wall,
             "note": "Validation: generate (model reloaded); Test: test-final incl. refit"},
            {"phase": phase, "method": "als_plus_popularity",
             "train_seconds": fit if phase == "validation" else np.nan,
             "inference_seconds": hybrid_inf, "end_to_end_seconds": als_wall,
             "note": "same run as ALS only plus fallback and assembly"},
        ]
    frame = pd.DataFrame(rows)
    frame["hardware"] = args.hardware
    return frame


def _cross_phase_checks(args, comparison):
    rows = []
    sig = pd.read_csv(args.results / "significance.csv")
    sig = sig[sig["metric"] == "ndcg"]
    verdict = {p: {(r.method_a, r.method_b): (np.sign(r.mean_diff) if r.significant else 0)
                   for r in g.itertuples()} for p, g in sig.groupby("phase")}
    if {"validation", "test"} <= set(verdict):
        agree = sum(verdict["validation"][key] == verdict["test"].get(key) for key in verdict["validation"])
        total = len(verdict["validation"])
        ties = [f"{a} = {b}" for (a, b), v in verdict["test"].items() if v == 0]
        rows.append({"phase": "both", "check": "same_paired_conclusions", "subject": "NDCG@10",
                     "value": agree / total, "status": "PASS" if agree == total else "FAIL",
                     "detail": f"{agree}/{total} agree; ties: {', '.join(ties) or 'none'}"})
    seeds = args.results / "diagnostics" / "seed_stability.csv"
    if seeds.is_file():
        seed = pd.read_csv(seeds)
        mine = seed[(seed["seed"] == 42) & (seed["method"] == "als_plus_popularity")]["ndcg"]
        ref = comparison[(comparison["phase"] == "validation")
                         & (comparison["method"] == "als_plus_popularity")]["ndcg"]
        if len(mine) and len(ref):
            diff = abs(float(mine.iloc[0]) - float(ref.iloc[0]))
            rows.append({"phase": "validation", "check": "retrain_reproduces_e5",
                         "subject": "seed 42", "value": diff,
                         "status": "PASS" if diff < 1e-12 else "FAIL",
                         "detail": f"retrained NDCG {float(mine.iloc[0]):.10f} vs E5 {float(ref.iloc[0]):.10f}"})
    return pd.DataFrame(rows)


def compare(args):
    """T28 headline table: one row per phase and method."""
    metrics = pd.read_csv(args.results / "metrics.csv")
    headline = metrics[(metrics["k"] == args.k)
                       & (metrics["relevance"] == f">={args.relevance_threshold}")]
    coverage = pd.read_csv(args.results / "coverage.csv")
    coverage = coverage[coverage["segment"] == "all"].set_index(["phase", "method"])
    beyond = pd.read_csv(args.results / "beyond_accuracy.csv").set_index(["phase", "method"])
    runtime = _runtime(args)
    _save_csv(runtime, args.results / "runtime.csv")
    runtime = runtime.set_index(["phase", "method"])
    rows = []
    for (phase, method), part in headline.groupby(["phase", "method"], sort=False):
        row = {"phase": phase, "method": method}
        for scope, suffix in (("all", ""), ("warm_item", "_warm_item")):
            for r in part[part["scope"] == scope].itertuples():
                row[f"{r.metric}{suffix}"] = r.mean
                if r.metric in ("recall", "ndcg"):
                    row[f"{r.metric}{suffix}_ci95_low"] = r.ci95_low
                    row[f"{r.metric}{suffix}_ci95_high"] = r.ci95_high
            row[f"scored_users{suffix}"] = int(part[part["scope"] == scope]["scored_users"].iloc[0])
        row["target_users"] = int(coverage.loc[(phase, method), "users"])
        row["user_coverage_ge1"] = coverage.loc[(phase, method), "coverage_ge1"]
        row["user_coverage_full_k"] = coverage.loc[(phase, method), "coverage_full_k"]
        for column in ("distinct_items", "catalog_coverage", "gini_exposure", "head_share",
                       "novelty_bits", "personalization"):
            row[column] = beyond.loc[(phase, method), column]
        for column in ("train_seconds", "inference_seconds", "end_to_end_seconds"):
            row[column] = runtime[column].get((phase, method), np.nan)
        rows.append(row)
    comparison = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(METHODS)}
    comparison = comparison.sort_values(["phase", "method"], key=lambda s: s.map(order)
                                        if s.name == "method" else s != "validation")
    _save_csv(comparison, args.results / "model_comparison.csv")
    checks = pd.read_csv(args.results / "reliability_checks.csv")
    checks = checks[~checks["check"].isin(["same_paired_conclusions", "retrain_reproduces_e5"])]
    _save_csv(pd.concat([checks, _cross_phase_checks(args, comparison)], ignore_index=True),
              args.results / "reliability_checks.csv")
    logging.info("Wrote %s", args.results / "model_comparison.csv")


def report(args):
    """Execute the notebook in place, then export a code-free single-file HTML report."""
    import nbformat
    from nbconvert import HTMLExporter
    from nbconvert.preprocessors import ExecutePreprocessor

    path = Path(__file__).parent / "notebooks" / "E6_evaluation.ipynb"
    notebook = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=900, kernel_name=args.kernel).preprocess(
        notebook, {"metadata": {"path": str(path.parent)}})
    notebook.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python",
                                       "name": "python3"}
    nbformat.write(notebook, path)
    exporter = HTMLExporter(exclude_input=True, embed_images=True)
    html, _ = exporter.from_notebook_node(notebook)
    (args.results / "E6_evaluation_report.html").write_text(html, encoding="utf-8")
    logging.info("Wrote %s", args.results / "E6_evaluation_report.html")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("evaluate", "diagnose", "seed-stability", "ablation",
                                           "compare", "figures", "report"))
    parser.add_argument("--phase", choices=("validation", "test"), default="validation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/e4"))
    parser.add_argument("--e5-dir", type=Path, default=Path("outputs/e5"))
    parser.add_argument("--results", type=Path, default=Path("results/e6"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/e6/per_user"))
    parser.add_argument("--config", type=Path, default=Path("configs/e5.json"))
    parser.add_argument("--metadata-dir", type=Path, default=None,
                        help="optional E3 metadata_clean folder, enables category diversity")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--relevance-threshold", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2026])
    parser.add_argument("--timings", type=Path, default=None,
                        help="CSV step,seconds of measured E5 CLI wall-clock times")
    parser.add_argument("--hardware", default="local[2], 2 vCPU, 8 GB RAM")
    parser.add_argument("--kernel", default="python3", help="Jupyter kernel for the report action")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.action == "figures":
        from src.evaluation.e6_figures import draw_all
        draw_all(args.results)
        return
    {"evaluate": evaluate, "diagnose": diagnose, "seed-stability": seed_stability,
     "ablation": ablation, "compare": compare, "report": report}[args.action](args)


if __name__ == "__main__":
    main()
