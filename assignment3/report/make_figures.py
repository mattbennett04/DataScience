# figures for the report
# made at the ieee column width so latex doesnt rescale them and the text
# matches the report font size
#
# usage from the assignment3 folder
#   python report/make_figures.py

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)  # so preprocess.py can be imported from the report folder
from preprocess import CONFIGS, load_series, make_folds

OUT = os.path.join(HERE, "figures")
RESULTS = os.path.join(ROOT, "results")

MODELS = ["elman", "jordan", "mrn"]
NAMES = {"elman": "Elman", "jordan": "Jordan", "mrn": "MRN"}
COLOURS = {"elman": "#2a78d6", "jordan": "#eb6834", "mrn": "#1baf7a"}
STYLES = {"elman": "-", "jordan": "--", "mrn": ":"}  # so it still works in black and white
GREY = "#555555"
COL = 3.45  # ieee column width inches

# times-like font at roughly the report text size
plt.rcParams.update({
    "font.family": "Nimbus Roman",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def grid(ax):
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)  # light horizontal lines only
    ax.set_axisbelow(True)


def load():
    res = {}
    for p in glob.glob(os.path.join(RESULTS, "*_fold*.json")):
        r = json.load(open(p))
        res[(r["dataset"], r["model"], r["fold"])] = r  # lookup by dataset/model/fold
    return res


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)


# the five series with the known breaks marked
def series():
    for d in CONFIGS:
        x = load_series(d)
        fig, ax = plt.subplots(figsize=(COL, 0.95))
        ax.plot(x, color=GREY, linewidth=0.6)

        for b in CONFIGS[d]["breaks"]:
            ax.axvline(b, color="#eb6834", linewidth=0.8, linestyle=(0, (3, 2)))  # dashed line at each break

        ax.set_xlim(0, len(x))
        ax.set_xlabel("$t$", labelpad=0)
        grid(ax)
        save(fig, f"series_{d}.pdf")


# expanding window folds drawn for a series of length 100 so the axis reads as %
def folds():
    n = 100
    fig, ax = plt.subplots(figsize=(COL, 1.25))
    parts = [("training", "#2a78d6"), ("validation", "#eda100"), ("test", "#eb6834")]

    for k, fold in enumerate(make_folds(n)):
        for (s, e), (label, c) in zip(fold, parts):
            # one bar per block only label the first fold so the legend isnt repeated
            ax.barh(k, e - s, left=s, height=0.6, color=c, label=label if k == 0 else None,
                    edgecolor="white", linewidth=1)

    ax.set_yticks(range(5), [f"fold {k + 1}" for k in range(5)])
    ax.invert_yaxis()  # fold 1 at the top
    ax.set_xlim(0, n)
    ax.set_xlabel("position in the series (% of its length)")
    ax.legend(ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "folds.pdf")


# optuna trials for one dataset and fold every trial plus the best so far
def optuna_history(res, d="H279", fold=4):
    fig, ax = plt.subplots(figsize=(COL, 1.9))

    for m in MODELS:
        trials = res[(d, m, fold)]["trials"]
        best = np.inf
        best_line = []

        for i, t in enumerate(trials):
            if t["state"] == "COMPLETE":
                ax.plot(i + 1, t["value"], "o", color=COLOURS[m], markersize=3, alpha=0.6)
                best = min(best, t["value"])
            else:
                ax.plot(i + 1, 1e-2, "x", color=COLOURS[m], markersize=3, alpha=0.6)  # pruned trials along the top
            best_line.append(best)

        ax.step(np.arange(1, len(trials) + 1), best_line, where="post", color=COLOURS[m],
                linestyle=STYLES[m], linewidth=1.4, label=NAMES[m])

    ax.set_yscale("log")
    ax.set_xlim(0, 35)  # room for the pruned label
    ax.set_xlabel("trial")
    ax.set_ylabel("validation MSE")
    ax.text(31, 1e-2, "pruned", fontsize=8, color=GREY, va="center")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    grid(ax)
    save(fig, "optuna_history.pdf")


# train and validation loss for the first seed of the last fold
def loss_curves(res):
    for d in ["H279", "W8"]:
        fig, ax = plt.subplots(figsize=(COL, 1.55))

        for m in MODELS:
            run = res[(d, m, 4)]["runs"][0]
            h = run["history"]
            ep = np.arange(1, len(h["val"]) + 1)
            ax.plot(ep, h["train"], color=COLOURS[m], linestyle=STYLES[m], linewidth=0.8, alpha=0.55)  # thin = train
            ax.plot(ep, h["val"], color=COLOURS[m], linestyle=STYLES[m], linewidth=1.5, label=NAMES[m])  # thick = val

            b = run["best_epoch"]
            ax.plot(b + 1, h["val"][b], "o", color=COLOURS[m], markersize=4, markeredgecolor="white")  # epoch that was kept

        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE")
        ax.legend(frameon=False, ncol=3, loc="upper right")
        grid(ax)
        save(fig, f"loss_{d}.pdf")


# test rmse / naive rmse for every fold and seed grouped by dataset
def relative_boxes(res):
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    datasets = list(CONFIGS)
    width = 0.24

    for j, m in enumerate(MODELS):
        data = []
        for d in datasets:
            vals = []
            for k in range(5):
                r = res[(d, m, k)]
                vals += [run["test"]["rmse"] / r["naive_test"]["rmse"] for run in r["runs"]]
            data.append(vals)

        pos = np.arange(len(datasets)) + (j - 1) * (width + 0.03)  # three boxes side by side per dataset
        box = ax.boxplot(data, positions=pos, widths=width, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 1},
                         whiskerprops={"linewidth": 0.6}, capprops={"linewidth": 0.6},
                         flierprops={"markersize": 2, "markeredgecolor": GREY})

        for patch in box["boxes"]:
            patch.set_facecolor(COLOURS[m])
            patch.set_alpha(0.75)
            patch.set_linewidth(0.6)
        box["boxes"][0].set_label(NAMES[m])  # one legend entry per network

    ax.axhline(1, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=0)  # same as naive
    ax.set_yscale("log")
    ax.set_xticks(range(len(datasets)), datasets)
    ax.set_ylabel("RMSE relative to naive")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    grid(ax)
    save(fig, "relative_rmse.pdf")


# end of the last test block mean prediction over the seeds
def forecasts(res):
    for d, n_show in [("H279", 72), ("W8", 120)]:  # 3 days of h279 and 120 weeks of w8
        x = load_series(d)
        fig, ax = plt.subplots(figsize=(COL, 1.45))
        t = np.array(res[(d, "elman", 4)]["test_t"])[-n_show:]
        ax.plot(t, x[t], color=GREY, linewidth=2, label="actual")

        for m in MODELS:
            r = res[(d, m, 4)]
            pred = np.mean([run["pred"] for run in r["runs"]], axis=0)[-n_show:]  # average over the seeds
            ax.plot(t, pred, color=COLOURS[m], linestyle=STYLES[m], linewidth=1.1, label=NAMES[m])

        ax.set_xlabel("$t$", labelpad=0)
        ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        grid(ax)
        save(fig, f"forecast_{d}.pdf")


# rolling percentage error after the break divided by the pre-break error
def drift(res, factor=2.0, roll=5, pre_len=50):
    for d in ["D686", "M6864", "M26487"]:
        x = load_series(d)
        breaks = sorted(CONFIGS[d]["breaks"])
        fig, ax = plt.subplots(figsize=(COL, 1.35))
        drawn = False

        for m in MODELS:
            curves = []
            for k in range(5):
                r = res[(d, m, k)]
                t = np.array(r["after_train_t"])

                for j, brk in enumerate(breaks):
                    stop = breaks[j + 1] if j + 1 < len(breaks) else len(x)
                    if np.sum((t >= brk - pre_len) & (t < brk)) < 10:
                        continue  # break is inside the training block for this fold

                    for run in r["runs"]:
                        err = np.abs(np.array(run["pred_after_train"]) - x[t]) / np.abs(x[t])
                        normal = err[(t >= brk - pre_len) & (t < brk)].mean()  # error before the break
                        after = err[(t >= brk) & (t < stop)]
                        curves.append([after[max(0, i - roll + 1):i + 1].mean() / normal
                                       for i in range(len(after))])

            if not curves:
                continue
            n = min(len(c) for c in curves)  # trim to the same length before averaging
            ax.plot(np.mean([c[:n] for c in curves], axis=0), color=COLOURS[m], linestyle=STYLES[m],
                    linewidth=1.3, label=NAMES[m])
            drawn = True

        if drawn:
            ax.axhline(factor, color=GREY, linewidth=0.8, linestyle=(0, (3, 2)))  # threshold
            ax.set_yscale("log")
            ax.set_xlabel("time steps after the break", labelpad=1)
            ax.set_ylabel("error ratio")
            ax.legend(frameon=False, ncol=3, loc="upper right")
            grid(ax)
        save(fig, f"drift_{d}.pdf")


# m6864 after the break with retraining shared settings run dotted line = first retrain
def retrain_curve(roll=12):
    from retrain import MIN_NEW, ar_reference
    folder = os.path.join(ROOT, "results_retrain_fixed")
    x = load_series("M6864")
    r0 = json.load(open(os.path.join(folder, "M6864_elman_seed0.json")))
    brk, stop, start = r0["break"], r0["stop"], r0["deploy_start"]
    actual = x[brk:stop]

    # rolling absolute percentage error over the last roll steps
    def rolling(pred):
        e = np.abs(pred - actual) / np.abs(actual) * 100
        return np.array([e[max(0, i - roll + 1):i + 1].mean() for i in range(len(e))])

    fig, ax = plt.subplots(figsize=(COL, 1.9))
    steps = np.arange(stop - brk)
    ax.plot(steps, rolling(x[brk - 1:stop - 1]), color=GREY, linewidth=1.8, label="naive")
    ax.plot(steps, rolling(ar_reference("M6864", 12)), color="#999999", linewidth=1, linestyle=(0, (1, 1)),
            label="linear")

    for m in MODELS:
        curves = []
        for seed in range(10):
            r = json.load(open(os.path.join(folder, f"M6864_{m}_seed{seed}.json")))
            pred = np.array([np.nan if v is None else v for v in r["strategies"]["fresh"]["pred"]])[brk - start:]
            curves.append(rolling(pred))
        ax.plot(steps, np.mean(curves, axis=0), color=COLOURS[m], linestyle=STYLES[m], linewidth=1.4,
                label=NAMES[m])  # mean over the 10 seeds

    ax.axvline(MIN_NEW, color=GREY, linewidth=0.8, linestyle=(0, (1, 2)))  # first retrain
    ax.set_xlabel("time steps after the break", labelpad=1)
    ax.set_ylabel("rolling error (%)")
    ax.set_xlim(0, stop - brk)
    ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.5,
              columnspacing=0.8)
    grid(ax)
    save(fig, "retrain_M6864.pdf")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = load()

    series()
    folds()
    optuna_history(res)
    loss_curves(res)
    relative_boxes(res)
    forecasts(res)
    drift(res)
    retrain_curve()

    print("saved to", OUT)
