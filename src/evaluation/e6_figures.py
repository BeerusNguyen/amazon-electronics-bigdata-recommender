"""E6 figures, drawn only from the CSV/JSON files in results/e6 so every chart is traceable.

Colour follows the method, never its rank: Popularity slot-1 blue, ALS only slot-2 orange,
ALS + Popularity slot-3 aqua (validated all-pairs for colour-vision deficiency); Random is
neutral grey because it is a reference floor. Two hues are below 3:1 contrast on white, so
every mark carries a direct value label and every chart has a CSV table behind it.

Files (numbered in reading order):
  fig01_quality_ci            fig06_beyond_accuracy
  fig02_significance          fig07_als_item_support
  fig03_coverage_fallback     fig08_als_ablation
  fig04_cohorts               fig09_tuning_vs_seed_noise
  fig05_k_sensitivity         fig10_recall_ceiling
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"popularity": "#257de0", "als": "#f47230", "als_plus_popularity": "#00b894",
          "random": "#9c9c9c"}
LABELS = {"random": "Random", "popularity": "Popularity", "als": "ALS only",
          "als_plus_popularity": "ALS + Popularity"}
ORDER = list(LABELS)
TEXT, TEXT_2, GRID = "#191919", "#666666", "#e6e6e6"
PCT = matplotlib.ticker.PercentFormatter(1.0)


def _fmt(value):
    """Readable label for values spanning 1e-5 .. 1."""
    if value == 0:
        return "0"
    return f"{value:.4f}" if abs(value) >= 0.001 else f"{value:.1e}"


def _style():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.titlecolor": TEXT,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
    })


def _save(fig, path, note, note_y=-0.02):
    fig.text(0.01, note_y, note, ha="left", va="top", fontsize=8, color=TEXT_2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _hbars(ax, labels, values, colors, pct=False, height=0.55):
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, height=height)
    for yy, value in zip(y, values):
        text = (f"{value:.3%}" if value < 0.01 else f"{value:.1%}") if pct else _fmt(value)
        ax.text(value, yy, f" {text}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(max(values) * 1.45, 1e-9))
    ax.grid(axis="y", visible=False)
    if pct:
        ax.xaxis.set_major_formatter(PCT)


def _headline(results):
    metrics = pd.read_csv(results / "metrics.csv")
    return metrics[(metrics["k"] == 10) & (metrics["relevance"] == ">=4")]


def fig_quality(results, out):
    data = _headline(results)
    data = data[data["scope"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, metric, title in ((axes[0], "recall", "Recall@10"), (axes[1], "ndcg", "NDCG@10")):
        part = data[data["metric"] == metric]
        phases = [p for p in ("validation", "test") if p in set(part["phase"])]
        for i, method in enumerate(ORDER):
            rows = part[part["method"] == method].set_index("phase").reindex(phases)
            y = np.arange(len(phases)) + (i - 1.5) * 0.2
            err = [rows["mean"] - rows["ci95_low"], rows["ci95_high"] - rows["mean"]]
            ax.barh(y, rows["mean"], height=0.18, color=COLORS[method], label=LABELS[method],
                    xerr=err, error_kw={"elinewidth": 0.8, "ecolor": TEXT_2, "capsize": 2})
            for yy, value, high in zip(y, rows["mean"], rows["ci95_high"]):
                ax.text(high, yy, f"  {_fmt(value)}", va="center", fontsize=7.5, color=TEXT)
        ax.set_yticks(np.arange(len(phases)))
        ax.set_yticklabels([p.capitalize() for p in phases])
        ax.invert_yaxis()
        ax.set_title(title, loc="left")
        ax.set_xlim(0, part["ci95_high"].max() * 1.35)
        ax.grid(axis="y", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=8.5)
    fig.suptitle("Ranking quality at K = 10 (all relevant users, error bar = 95% bootstrap CI)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=TEXT)
    _save(fig, out / "fig01_quality_ci.png",
          "Source: results/e6/metrics.csv. Denominator: users with at least one held-out rating >= 4; "
          "users without a list score 0.", note_y=-0.06)


def fig_significance(results, out):
    data = pd.read_csv(results / "significance.csv")
    data = data[data["metric"] == "ndcg"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(data) + 1.2))
    for i, row in data.iterrows():
        color = COLORS[row.method_a] if row.significant else "#bdbdbd"
        ax.plot([row.diff_ci95_low, row.diff_ci95_high], [i, i], color=color, linewidth=2)
        ax.plot(row.mean_diff, i, "o", color=color, markersize=6)
        p_text = "<= 0.001" if row.p_holm <= 0.001 + 1e-12 else f"= {row.p_holm:.2f}"
        ax.text(max(row.diff_ci95_high, 0), i, f"  {row.mean_diff:+.5f}  p_holm {p_text}",
                va="center", fontsize=7.5, color=TEXT)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels([f"{'Val' if r.phase == 'validation' else 'Test'}: "
                        f"{LABELS[r.method_a]} - {LABELS[r.method_b]}" for r in data.itertuples()],
                       fontsize=8)
    ax.invert_yaxis()
    lo, hi = data["diff_ci95_low"].min(), data["diff_ci95_high"].max()
    ax.set_xlim(lo - (hi - lo) * 0.05, hi + (hi - lo) * 0.6)
    ax.set_xlabel("Paired difference in NDCG@10 (A - B)")
    ax.set_title("Paired comparisons on the same users (grey = not significant after Holm)", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, out / "fig02_significance.png",
          "Source: results/e6/significance.csv. CI = paired bootstrap (2,000), p = sign-flip test "
          "(10,000), Holm over the 10 tests of each phase; <= 0.001 is the smallest reportable value.")


def fig_coverage(results, out, phase="test"):
    data = pd.read_csv(results / "fallback.csv")
    data = data[data["phase"] == phase]
    status = data[data["als_status"] != "all"].set_index("als_status")
    names = {"full_list": "ALS full list (10)", "short_list": "ALS short list (1-9)",
             "known_unserved": "Known user, no ALS list", "unknown_user": "Unknown (cold) user"}
    shades = {"full_list": "#f47230", "short_list": "#f8a77d", "known_unserved": "#cfcfcf",
              "unknown_user": "#9c9c9c"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    fig.subplots_adjust(wspace=0.35)
    ax, left = axes[0], 0.0
    for key in names:
        share = status.loc[key, "share_of_targets"]
        ax.barh([0], [share], left=left, height=0.5, color=shades[key], edgecolor="white",
                linewidth=2, label=f"{names[key]}: {share:.1%}")
        left += share
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(PCT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    ax.set_title("Target users by ALS serving status (before fallback)", loc="left")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    total = data[data["als_status"] == "all"].iloc[0]
    y = np.arange(2)
    before = [total["coverage_ge1_before"], total["coverage_full_before"]]
    after = [total["coverage_ge1_after"], total["coverage_full_after"]]
    ax.barh(y - 0.17, before, height=0.3, color=COLORS["als"], label="ALS only (before)")
    ax.barh(y + 0.17, after, height=0.3, color=COLORS["als_plus_popularity"],
            label="ALS + Popularity (after)")
    for yy, b, a in zip(y, before, after):
        ax.text(b, yy - 0.17, f" {b:.1%}", va="center", fontsize=8, color=TEXT)
        ax.text(a, yy + 0.17, f" {a:.1%}", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels(["Coverage >= 1 item", "Coverage = 10 items"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.15)
    ax.xaxis.set_major_formatter(PCT)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    ax.set_title(f"User coverage, all {int(total['users']):,} target users", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, out / "fig03_coverage_fallback.png",
          f"{phase.capitalize()}. Source: results/e6/fallback.csv. Denominator = every distinct "
          "held-out user, including users with no relevant item.", note_y=-0.2)


def fig_cohorts(results, out, phase="test"):
    data = pd.read_csv(results / "cohorts.csv")
    data = data[data["phase"] == phase]
    pretty = {"all": "All", "warm_user/warm_item": "warm user\nwarm item",
              "cold_user/warm_item": "cold user\nwarm item", "warm_user/cold_item": "warm user\ncold item",
              "cold_user/cold_item": "cold user\ncold item", "history_1_4": "history\n1-4 ratings",
              "history_ge_5": "history\n>= 5 ratings"}
    cohorts = list(pretty)
    methods = ["popularity", "als", "als_plus_popularity"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(cohorts))
    for i, method in enumerate(methods):
        rows = data[data["method"] == method].set_index("cohort").reindex(cohorts)
        ax.bar(x + (i - 1) * 0.28, rows["recall"], width=0.26, color=COLORS[method],
               label=LABELS[method])
    users = data[data["method"] == "popularity"].set_index("cohort").reindex(cohorts)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pretty[c]}\n(n={int(n):,})" for c, n in zip(cohorts, users["scored_users"])],
                       fontsize=8)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Recall@10")
    ax.legend(fontsize=8)
    ax.set_title(f"Recall@10 by cold-start cohort ({phase.capitalize()})", loc="left")
    _save(fig, out / "fig04_cohorts.png",
          "Source: results/e6/cohorts.csv. Cold-item cohorts are 0 for every method by construction: "
          "the item is absent from the history catalog.", note_y=-0.08)


def fig_k_sensitivity(results, out, phase="test"):
    metrics = pd.read_csv(results / "metrics.csv")
    data = metrics[(metrics["phase"] == phase) & (metrics["scope"] == "all")
                   & (metrics["relevance"] == ">=4")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, metric, title in ((axes[0], "recall", "Recall@K"), (axes[1], "ndcg", "NDCG@K")):
        for method in ORDER:
            rows = data[(data["method"] == method) & (data["metric"] == metric)].sort_values("k")
            ax.plot(rows["k"], rows["mean"], color=COLORS[method], linewidth=2, marker="o",
                    markersize=5, label=LABELS[method])
            last = rows.iloc[-1]
            ax.annotate(_fmt(last["mean"]), (last["k"], last["mean"]),
                        xytext=(5, {"random": -7, "als": 5}.get(method, 0)),
                        textcoords="offset points", fontsize=7.5, va="center", color=TEXT)
        ax.set_xticks([1, 3, 5, 10])
        ax.set_xlabel("K")
        ax.set_title(f"{title} ({phase.capitalize()})", loc="left")
    axes[0].legend(fontsize=8)
    _save(fig, out / "fig05_k_sensitivity.png",
          "Source: results/e6/metrics.csv. K < 10 uses the first K ranks of the same lists; "
          "Random and ALS only overlap near 0.")


def fig_beyond(results, out, phase="test"):
    data = pd.read_csv(results / "beyond_accuracy.csv")
    data = data[data["phase"] == phase].set_index("method").reindex(ORDER)
    panels = [("catalog_coverage", "Catalog coverage", True),
              ("head_share", "Share of slots from head items", True),
              ("novelty_bits", "Novelty (bits)", False), ("personalization", "Personalization", False)]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3))
    for ax, (column, title, pct) in zip(axes, panels):
        _hbars(ax, [LABELS[m] for m in ORDER] if ax is axes[0] else [""] * len(ORDER),
               data[column].tolist(), [COLORS[m] for m in ORDER], pct=pct, height=0.6)
        ax.set_title(title, loc="left", fontsize=9.5)
    _save(fig, out / "fig06_beyond_accuracy.png",
          f"{phase.capitalize()}. Source: results/e6/beyond_accuracy.csv. Head items = most popular items "
          "holding 20% of history positives. Novelty = mean -log2(item popularity share). "
          "Personalization = 1 - mean list overlap of random user pairs.")


def fig_item_support(results, out, phase="test"):
    data = pd.read_csv(results / "diagnostics" / "als_item_support.csv")
    data = data[data["phase"] == phase].set_index("method").reindex(["popularity", "als"])
    panels = [("median_item_history_ratings", "Median history ratings\nof recommended item", False),
              ("share_items_with_le_3_ratings", "Share of slots whose item\nhas <= 3 history ratings", True),
              ("mean_item_positive_count", "Mean positive count\nof recommended item", False)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
    for ax, (column, title, pct) in zip(axes, panels):
        _hbars(ax, ["Popularity", "ALS only"] if ax is axes[0] else ["", ""],
               data[column].tolist(), [COLORS["popularity"], COLORS["als"]], pct=pct)
        ax.set_title(title, loc="left", fontsize=9.5)
    _save(fig, out / "fig07_als_item_support.png",
          f"{phase.capitalize()}. Source: results/e6/diagnostics/als_item_support.csv. Spark ALS has no "
          "bias term, so with 1-4 ratings per user predicted ratings are unreliable and rarely rated "
          "items reach the top.")


def fig_ablation(results, out):
    data = pd.read_csv(results / "diagnostics" / "als_ablation.csv")
    head = _headline(results)
    pop = float(head[(head["phase"] == "validation") & (head["method"] == "popularity")
                     & (head["scope"] == "all") & (head["metric"] == "ndcg")]["mean"].iloc[0])
    variants = list(dict.fromkeys(data["variant"]))
    fig, ax = plt.subplots(figsize=(11, 3.8))
    y = np.arange(len(variants))
    for offset, method in ((-0.18, "als"), (0.18, "als_plus_popularity")):
        rows = data[data["method"] == method].set_index("variant").reindex(variants)
        ax.barh(y + offset, rows["ndcg"], height=0.32, color=COLORS[method], label=LABELS[method])
        for yy, value in zip(y + offset, rows["ndcg"]):
            ax.text(value, yy, f" {_fmt(value)}", va="center", fontsize=7.5, color=TEXT)
    ax.axvline(pop, color=COLORS["popularity"], linewidth=1.5)
    ax.text(pop, -0.75, f" Popularity {pop:.4f}", color=TEXT, fontsize=8, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(variants, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(data["ndcg"].max(), pop) * 1.3)
    ax.set_xlabel("NDCG@10 (Validation, all relevant users)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Diagnostic ablation: is the objective or the regularisation to blame?", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, out / "fig08_als_ablation.png",
          "Source: results/e6/diagnostics/als_ablation.csv. Validation only, locked rank/maxIter, "
          "never used to choose the Test model.")


def fig_tuning(results, out):
    tuning = pd.read_csv(results / "diagnostics" / "als_tuning.csv")
    seed = pd.read_csv(results / "diagnostics" / "seed_stability.csv")
    seed = seed[seed["method"] == "als_plus_popularity"]
    low, high = seed["ndcg"].min(), seed["ndcg"].max()
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.axvspan(low, high, color=GRID, zorder=0)
    ax.text((low + high) / 2, -0.7, "range over seeds 42 / 7 / 2026\n(selected config)",
            ha="center", va="bottom", fontsize=8, color=TEXT_2)
    for yy, row in enumerate(tuning.itertuples()):
        ax.plot(row.ndcg_at_k, yy, "o", markersize=8,
                color=COLORS["als_plus_popularity"] if row.selected else "#9c9c9c")
        ax.text(row.ndcg_at_k, yy, f"   {row.ndcg_at_k:.6f}{'  (selected)' if row.selected else ''}",
                va="center", fontsize=8, color=TEXT)
    ax.set_yticks(np.arange(len(tuning)))
    ax.set_yticklabels([f"trial {r.trial}: rank={r.rank}, regParam={r.reg_param}"
                        for r in tuning.itertuples()], fontsize=8.5)
    ax.set_ylim(len(tuning) - 0.5, -1.5)
    lo, hi = min(tuning["ndcg_at_k"].min(), low), max(tuning["ndcg_at_k"].max(), high)
    ax.set_xlim(lo - (hi - lo) * 0.3, hi + (hi - lo) * 0.9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.6f"))
    ax.set_xlabel("ALS + Popularity NDCG@10 on Validation (zoomed axis, not from 0)")
    ax.set_title("E5 tuning differences are about the size of seed noise", loc="left")
    ax.grid(axis="y", visible=False)
    _save(fig, out / "fig09_tuning_vs_seed_noise.png",
          f"Tuning spread = {tuning['ndcg_at_k'].max() - tuning['ndcg_at_k'].min():.2e}; seed spread = "
          f"{high - low:.2e}. Source: results/e6/diagnostics/als_tuning.csv and seed_stability.csv.")


def fig_ceiling(results, out):
    protocol = json.loads((results / "protocol.json").read_text())
    head = _headline(results)
    phases = [p for p in ("validation", "test") if p in protocol]
    fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 3.2), squeeze=False)
    fig.subplots_adjust(wspace=0.45)
    methods = ["popularity", "als", "als_plus_popularity"]
    for ax, phase in zip(axes[0], phases):
        means = head[(head["phase"] == phase) & (head["scope"] == "all")
                     & (head["metric"] == "recall")].set_index("method")["mean"]
        values = [protocol[phase]["recall_ceiling"]] + [means[m] for m in methods]
        y = np.arange(len(values))
        ax.barh(y, values, color=["#cfcfcf"] + [COLORS[m] for m in methods], height=0.55)
        for yy, value in zip(y, values):
            ax.text(value, yy, f" {_fmt(value)}", va="center", fontsize=8, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlim(1e-5, 5)
        ax.set_yticks(y)
        ax.set_yticklabels(["Reachable ceiling"] + [LABELS[m] for m in methods], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"Recall@10 vs ceiling ({phase.capitalize()}, log scale)", loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
    _save(fig, out / "fig10_recall_ceiling.png",
          "Ceiling = mean over scored users of min(10, warm targets) / all targets: the best any "
          "history-catalog recommender could reach. Source: results/e6/protocol.json.")


def draw_all(results):
    results = Path(results)
    out = results / "figures"
    _style()
    for draw in (fig_quality, fig_significance, fig_coverage, fig_cohorts, fig_k_sensitivity,
                 fig_beyond):
        draw(results, out)
    diagnostics = results / "diagnostics"
    if (diagnostics / "als_item_support.csv").is_file():
        fig_item_support(results, out)
    if (diagnostics / "als_ablation.csv").is_file():
        fig_ablation(results, out)
    if (diagnostics / "als_tuning.csv").is_file() and (diagnostics / "seed_stability.csv").is_file():
        fig_tuning(results, out)
    fig_ceiling(results, out)
