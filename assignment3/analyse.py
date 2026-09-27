# analyse saved training results
# each fold seed pair is treated as a matched block when comparing networks
#
# usage
#   python analyse.py
#   python analyse.py --results results_M26487_raw

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from preprocess import CONFIGS, load_series

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(HERE, "figures")

MODELS = ["elman", "jordan", "mrn"]
NAMES = {"elman": "Elman", "jordan": "Jordan", "mrn": "MRN"}
COLOURS = {"elman": "#2a78d6", "jordan": "#eb6834", "mrn": "#1baf7a"}
STYLES = {"elman": "-", "jordan": "--", "mrn": ":"}  # so it still works in black and white
GREY = "#555555"


def load_results(folder):
    results = {}
    for path in glob.glob(os.path.join(folder, "*_fold*.json")):
        with open(path) as f:
            r = json.load(f)
        results[(r["dataset"], r["model"], r["fold"])] = r # easy lookup by dataset/model/fold
    return results


# flatten saved runs into one row per seed/fold
def runs_table(results):
    rows = []
    for (dataset, model, fold), r in results.items():
        naive = r["naive_test"]  # same baseline is shared by all seeds in this fold
        for i, run in enumerate(r["runs"]):
            test = run["test"]
            rows.append({
                "dataset": dataset, "model": model, "fold": fold, "seed_idx": i,
                "rmse": test["rmse"], "mae": test["mae"], "smape": test["smape"],
                # < 1 means better than naive on that fold
                "rel_rmse": test["rmse"] / naive["rmse"],
                "rel_mae": test["mae"] / naive["mae"],
                "rel_smape": test["smape"] / naive["smape"],
                "best_epoch": run["best_epoch"],
                "train_mse": run["train_mse_scaled"], "val_mse": run["val_mse_scaled"],
            })
    return pd.DataFrame(rows)


# holm correction for multiple comparisons
def holm(pvals):
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    # smallest p-value gets strongest correction
    for i, idx in enumerate(np.argsort(pvals)):
        # keep adjusted p-values monotonic
        running = max(running, min(1.0, (m - i) * pvals[idx]))
        adjusted[idx] = running
    return adjusted


def compare_models(df, metric, alpha):
    lines = []
    winners = {}
    for dataset in CONFIGS:
        sub = df[df.dataset == dataset]
        if sub.empty:
            continue
        # matched results for each fold/seed
        # align matched runs
        wide = sub.pivot_table(index=["fold", "seed_idx"], columns="model", values=metric).dropna()
        models = [m for m in MODELS if m in wide.columns]
        if len(models) < 2 or len(wide) < 2:
            lines.append(f"{dataset}: not enough finished runs to compare")
            continue

        # lower mean rank is better
        ranks = wide[models].rank(axis=1).mean()
        lines.append(f"\n{dataset} ({metric}, {len(wide)} blocks)")
        for m in models:
            lines.append(f"  {NAMES[m]:7s} mean {wide[m].mean():.4g}  mean rank {ranks[m]:.2f}")
        # candidate winner before pairwise testing
        best = ranks.idxmin()

        if len(models) == 3:
            # omnibus test
            chi2, p = stats.friedmanchisquare(wide[models[0]], wide[models[1]], wide[models[2]])
            lines.append(f"  friedman chi2={chi2:.2f} p={p:.3g}")
            if p >= alpha:
                lines.append("  no significant difference between the models")
                winners[dataset] = "no significant difference"
                continue

        pairs = []
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                pairs.append((models[i], models[j]))
        raw = []
        for a, b in pairs:
            try:
                # paired because blocks are matched
                raw.append(stats.wilcoxon(wide[a], wide[b]).pvalue)
            # happens if every difference is 0
            except ValueError:
                raw.append(1.0)
        # correct the three pairwise comparisons
        adj = holm(np.array(raw))

        best_wins_all = True
        for (a, b), p_raw, p_adj in zip(pairs, raw, adj):
            lower = a if wide[a].mean() < wide[b].mean() else b
            sig = "significant" if p_adj < alpha else "not significant"
            lines.append(f"  {NAMES[a]} vs {NAMES[b]}: p={p_raw:.3g} holm p={p_adj:.3g} -> {NAMES[lower]} lower, {sig}")
            if best in (a, b) and p_adj >= alpha:
                best_wins_all = False

        if best_wins_all:
            winners[dataset] = NAMES[best]
        else:
            winners[dataset] = f"{NAMES[best]} (best mean rank, not significantly better than all)"

    # compare mean performance across datasets
    means = df.groupby(["dataset", "model"])[metric].mean().unstack().dropna()
    if len(means) >= 2 and all(m in means.columns for m in MODELS):
        ranks = means[MODELS].rank(axis=1).mean()
        lines.append(f"\nacross {len(means)} datasets, mean rank: "
                     + ", ".join(f"{NAMES[m]} {ranks[m]:.2f}" for m in MODELS))
        chi2, p = stats.friedmanchisquare(means["elman"], means["jordan"], means["mrn"])
        lines.append(f"  friedman chi2={chi2:.2f} p={p:.3g} (only {len(means)} datasets, so low power)")

    lines.append("\nwinner per dataset:")
    for dataset, w in winners.items():
        lines.append(f"  {dataset}: {w}")
    return "\n".join(lines)


# compare each network with the naive baseline using paired wilcoxon tests
def compare_to_naive(df, results, alpha):
    naive = {(d, m, k): r["naive_test"]["rmse"] for (d, m, k), r in results.items()}
    tests = []
    for dataset in CONFIGS:
        for model in MODELS:
            sub = df[(df.dataset == dataset) & (df.model == model)]
            if sub.empty:
                continue
            # naive rmse for each matching fold
            base = np.array([naive[(dataset, model, k)] for k in sub.fold])
            net = sub.rmse.to_numpy()
            p = stats.wilcoxon(net, base).pvalue  # paired network-vs-naive comparison
            tests.append((dataset, model, (net / base).mean(), int((net < base).sum()), len(net), p))

    adj = holm(np.array([t[5] for t in tests]))
    lines = [f"\nnetwork vs naive (Wilcoxon, Holm correction over {len(tests)} tests)"]
    for (dataset, model, rel, wins, n, p), p_adj in zip(tests, adj):
        if p_adj >= alpha:
            verdict = "no significant difference"
        else:
            verdict = "better than naive" if rel < 1 else "worse than naive"
        lines.append(f"  {dataset:7s} {NAMES[model]:7s} rel rmse {rel:.3f}  better in {wins}/{n}  "
                     f"p={p:.2g} holm p={p_adj:.2g} -> {verdict}")
    return "\n".join(lines)


# measure how error changes after a known break point
# error is normalised by the average pre-break absolute percentage error
def drift_response(t, err, brk, stop, factor, roll, pre_len):

    # normal pre-break error level
    normal = err[(t >= brk - pre_len) & (t < brk)].mean()
    # only analyse this regime
    after = (t >= brk) & (t < stop)
    after_t = t[after]
    after_err = err[after]
    horizon = int(stop - brk)

    ratio = np.empty(len(after_err))
    for i in range(len(after_err)):
        ratio[i] = after_err[max(0, i - roll + 1):i + 1].mean() / normal  # rolling error relative to baseline

    res = {"degrade": horizon, "degrade_censored": True, "recover": np.nan, "recover_censored": False,
           "ratio": ratio}
    # first time error crosses the drift threshold
    above = np.where(ratio > factor)[0]
    if len(above) == 0:
        return res
    
    first = above[0]
    res["degrade"] = int(after_t[first] - brk)
    res["degrade_censored"] = False

    # first recovery after degradation
    below = np.where(ratio[first:] <= factor)[0]
    if len(below) == 0:
        res["recover"] = horizon
        res["recover_censored"] = True
    else:
        res["recover"] = int(after_t[first + below[0]] - brk)
    return res


def drift_table(results, factor, roll, pre_len):
    rows = []
    curves = {}
    for (dataset, model, fold), r in results.items():
        x = load_series(dataset)
        t = np.array(r["after_train_t"])
        breaks = sorted(CONFIGS[dataset]["breaks"])
        for j, brk in enumerate(breaks):
            stop = breaks[j + 1] if j + 1 < len(breaks) else len(x)
            # skip folds where there is not enough pre-break history
            if np.sum((t >= brk - pre_len) & (t < brk)) < 10:
                continue
            # last-value baseline
            naive = drift_response(t, np.abs(x[t - 1] - x[t]) / np.abs(x[t]), brk, stop, factor, roll, pre_len)
            for i, run in enumerate(r["runs"]):
                err = np.abs(np.array(run["pred_after_train"]) - x[t]) / np.abs(x[t])  # absolute percentage error
                res = drift_response(t, err, brk, stop, factor, roll, pre_len)
                rows.append({"dataset": dataset, "break": brk, "model": model, "fold": fold, "seed_idx": i,
                             "degrade": res["degrade"], "degrade_censored": res["degrade_censored"],
                             "recover": res["recover"], "recover_censored": res["recover_censored"],
                             "naive_degrade": naive["degrade"], "naive_recover": naive["recover"],
                             "naive_recover_censored": naive["recover_censored"]})
                curves.setdefault((dataset, brk, model), []).append(res["ratio"])
    return pd.DataFrame(rows), curves


def fmt_steps(steps, censored):
    med = np.nanmedian(steps) if steps.notna().any() else np.nan  # ignore missing recovery times
    return f"median {med:5.0f}  ({censored.sum()}/{len(steps)} censored)"


def drift_report(drift, factor, roll, pre_len):
    lines = [f"rolling abs % error ({roll} steps) compared to {factor}x the error in the "
             f"{pre_len} steps before the break",
             "degrade = first step above threshold, recover = first step back below",
             "'>' = not reached before the next break or end of the series"]
    if drift.empty:
        lines.append("  no folds with a break after the training block (or old result files)")
        return "\n".join(lines)

    for (dataset, brk), sub in drift.groupby(["dataset", "break"], sort=False):
        naive = sub.drop_duplicates("fold")
        lines.append(f"\n{dataset} break at t={brk} ({sub.fold.nunique()} folds)")
        for model in MODELS:
            m = sub[sub.model == model]
            if not m.empty:
                lines.append(f"  {NAMES[model]:7s} degrade {fmt_steps(m.degrade, m.degrade_censored)}   "
                             f"recover {fmt_steps(m.recover, m.recover_censored)}")
        lines.append(f"  {'Naive':7s} degrade median {naive.naive_degrade.median():5.0f}"
                     f"{'':20s}recover {fmt_steps(naive.naive_recover, naive.naive_recover_censored)}")

        groups = [sub[sub.model == m].recover.dropna() for m in MODELS if m in set(sub.model)]
        if len(groups) == 3 and all(len(g) for g in groups) and pd.concat(groups).nunique() > 1:
            h, p = stats.kruskal(*groups)
            lines.append(f"  recovery, kruskal-wallis H={h:.2f} p={p:.3g}")
    return "\n".join(lines)


def summary_table(df, results):
    rows = []
    for dataset in CONFIGS:
        for model in MODELS:
            sub = df[(df.dataset == dataset) & (df.model == model)]
            if sub.empty:
                continue
            folds = [r for k, r in results.items() if k[0] == dataset and k[1] == model]
            windows = sorted(set(r["best_params"]["window"] for r in folds))  # show all selected window sizes
            rows.append({
                "dataset": dataset,
                "model": NAMES[model],
                "folds": len(folds),
                "rmse_mean": sub.rmse.mean(), "rmse_std": sub.rmse.std(),
                "mae_mean": sub.mae.mean(), "mae_std": sub.mae.std(),
                "smape_mean": sub.smape.mean(), "smape_std": sub.smape.std(),
                "naive_rmse": np.mean([r["naive_test"]["rmse"] for r in folds]),
                "hidden_median": float(np.median([r["best_params"]["hidden"] for r in folds])),
                "windows": "/".join(str(w) for w in windows),
                "params_median": float(np.median([r["n_params"] for r in folds])),
                "best_epoch_median": sub.best_epoch.median(),
            })
    return pd.DataFrame(rows)


# quick diagnostic plots

def tidy(ax):
    ax.spines["top"].set_visible(False)  # cleaner report figures
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)


def datasets_in(results):
    return [d for d in CONFIGS if any(k[0] == d for k in results)]


def last_fold(results, dataset):
    return max(k[2] for k in results if k[0] == dataset)


# training/validation loss for the first seed of the final fold
def plot_loss_curves(results, fig_dir):
    datasets = datasets_in(results)
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 2.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        k = last_fold(results, dataset)
        for model in MODELS:
            if (dataset, model, k) not in results:
                continue
            run = results[(dataset, model, k)]["runs"][0]  # first seed is enough for a representative loss curve
            h = run["history"]
            epochs = np.arange(len(h["val"]))
            ax.plot(epochs, h["train"], color=COLOURS[model], linestyle=STYLES[model], linewidth=1, alpha=0.6)
            ax.plot(epochs, h["val"], color=COLOURS[model], linestyle=STYLES[model], linewidth=2,
                    label=NAMES[model])
            ax.plot(run["best_epoch"], h["val"][run["best_epoch"]], "o", color=COLOURS[model],
                    markersize=6, markeredgecolor="white")
        ax.set_yscale("log")
        ax.set_title(f"{dataset} (fold {k})", fontsize=10)
        ax.set_xlabel("epoch")
        tidy(ax)
    axes[0][0].set_ylabel("MSE (scaled)")
    axes[0][0].legend(frameon=False, fontsize=8, title="thick = val, thin = train", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "loss_curves.png"), dpi=200)
    plt.close(fig)


# average prediction over seeds on the final fold
def plot_forecasts(results, fig_dir):
    datasets = datasets_in(results)
    fig, axes = plt.subplots(len(datasets), 1, figsize=(7, 2.0 * len(datasets)), squeeze=False)
    for ax, dataset in zip(axes[:, 0], datasets):
        x = load_series(dataset)
        k = last_fold(results, dataset)
        drew_actual = False
        for model in MODELS:
            if (dataset, model, k) not in results:
                continue
            r = results[(dataset, model, k)]
            t = np.array(r["test_t"])
            if not drew_actual:
                ax.plot(t, x[t], color=GREY, linewidth=2.5, label="actual")
                drew_actual = True
            pred = np.mean([run["pred"] for run in r["runs"]], axis=0)  # average over random seeds
            ax.plot(t, pred, color=COLOURS[model], linestyle=STYLES[model], linewidth=1.5, label=NAMES[model])
        ax.set_title(f"{dataset}, test block of fold {k}", fontsize=10, loc="left")
        tidy(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4, loc="upper center")
    axes[-1, 0].set_xlabel("t")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.3 / (2.0 * len(datasets))))
    fig.savefig(os.path.join(fig_dir, "forecasts.png"), dpi=200)
    plt.close(fig)


# relative test error against the naive baseline
def plot_error_boxes(df, metric, fig_dir):
    datasets = [d for d in CONFIGS if d in set(df.dataset)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.4 * len(datasets), 2.8), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = df[df.dataset == dataset]
        models = [m for m in MODELS if m in set(sub.model)]
        box = ax.boxplot([sub[sub.model == m]["rel_" + metric] for m in models], widths=0.5, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 1.5},
                         flierprops={"markersize": 4, "markeredgecolor": GREY})
        for patch, m in zip(box["boxes"], models):
            patch.set_facecolor(COLOURS[m])
            patch.set_alpha(0.7)
        ax.axhline(1, color=GREY, linewidth=1, linestyle=(0, (4, 3)), zorder=0)  # equal to naive
        ax.text(0.5, 1, "naive", fontsize=7, color=GREY, va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xticks(range(1, len(models) + 1), [NAMES[m] for m in models], fontsize=8)
        ax.set_title(dataset, fontsize=10)
        tidy(ax)
    axes[0][0].set_ylabel(f"test {metric.upper()} / naive {metric.upper()}")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, f"test_{metric}_boxes.png"), dpi=200)
    plt.close(fig)


# average rolling error ratio after each break
def plot_drift(curves, factor, roll, fig_dir):
    keys = sorted({(d, b) for d, b, _ in curves}, key=lambda k: list(CONFIGS).index(k[0]))
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 2.8), squeeze=False)
    for ax, (dataset, brk) in zip(axes[0], keys):
        for model in MODELS:
            runs = curves.get((dataset, brk, model))
            if not runs:
                continue
            # trim curves to common length before averaging
            n = min(len(c) for c in runs)
            # mean drift response over folds/seeds
            mean = np.mean([c[:n] for c in runs], axis=0)
            ax.plot(np.arange(n), mean, color=COLOURS[model], linestyle=STYLES[model], linewidth=2,
                    label=NAMES[model])
        ax.axhline(factor, color=GREY, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(ax.get_xlim()[1], factor, "threshold", fontsize=7, color=GREY, ha="right", va="bottom")
        ax.set_yscale("log")
        ax.set_title(f"{dataset}, break at t={brk}", fontsize=10)
        ax.set_xlabel("steps after break")
        tidy(ax)
    axes[0][0].set_ylabel(f"error / pre-break error\n(rolling {roll})")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "drift_response.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="rmse", choices=["rmse", "mae", "smape"])
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--drift-factor", type=float, default=2.0)
    parser.add_argument("--drift-roll", type=int, default=5)
    parser.add_argument("--drift-pre", type=int, default=50)
    parser.add_argument("--results", default=RESULTS_DIR)
    args = parser.parse_args()

    results_dir = os.path.abspath(args.results)
    # keep figures next to non-default result folders
    fig_dir = FIG_DIR if results_dir == RESULTS_DIR else os.path.join(results_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    results = load_results(results_dir)
    if not results:
        raise SystemExit("no results yet, run train.py first")
    expected = len({k[0] for k in results}) * len(MODELS) * 5  # datasets × models × folds
    print(f"{len(results)} of {expected} runs finished\n")

    df = runs_table(results)
    summary = summary_table(df, results)
    summary.to_csv(os.path.join(results_dir, "summary.csv"), index=False)  # table used in the report
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))

    report = compare_models(df, args.metric, args.alpha) + "\n" + compare_to_naive(df, results, args.alpha)
    with open(os.path.join(results_dir, "stats.txt"), "w") as f:
        f.write(report + "\n")
    print(report)

    plot_loss_curves(results, fig_dir)
    plot_forecasts(results, fig_dir)
    plot_error_boxes(df, args.metric, fig_dir)

    drift, curves = drift_table(results, args.drift_factor, args.drift_roll, args.drift_pre)
    drift.to_csv(os.path.join(results_dir, "drift.csv"), index=False)  # keep raw drift measurements too
    text = drift_report(drift, args.drift_factor, args.drift_roll, args.drift_pre)
    with open(os.path.join(results_dir, "drift.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    plot_drift(curves, args.drift_factor, args.drift_roll, fig_dir)
    print(f"\nfigures saved to {fig_dir}")
