""" Makes the report figures and final tables from the files in results, pass the step names as arguments """
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from scipy import stats

from pipelines import class_names
from ml_utils import confusion_matrix, per_class_scores, stratified_sample, stratified_kfold, f1_macro, balanced_accuracy

# same colour for each model in every figure
colours = {"rf": "#2a78d6", "tree": "#eb6834", "knn": "#1baf7a"}
labels = {"rf": "Random forest", "tree": "Decision tree", "knn": "kNN"}
weight_colours = {"None": "#2a78d6", "balanced": "#eb6834", "balanced_subsample": "#1baf7a"}
weight_labels = {"None": "no weighting", "balanced": "balanced", "balanced_subsample": "balanced subsample"}

# font sizes to roughly match the report text
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "savefig.bbox": "tight"
})

# one ieee column in inches
col_width = 3.4


def optuna_plots():

    study = optuna.load_study(study_name="rf", storage="sqlite:///results/rf_study.db")
    trials = study.trials_dataframe()
    done = trials[trials["state"] == "COMPLETE"]
    pruned = trials[trials["state"] == "PRUNED"]

    # every completed trial plus the best score so far
    fig, ax = plt.subplots(figsize=(col_width, 2.3))
    ax.scatter(done["number"], done["value"], s=10, color=colours["rf"], label="completed trial")
    ax.plot(done["number"], done["value"].cummax(), color=colours["tree"], label="best so far")
    ax.set_xlabel("trial")
    ax.set_ylabel("objective")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig("figures/optuna_history.pdf")
    plt.close(fig)

    # fanova importance of each hyperparameter
    importance = optuna.importance.get_param_importances(study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42))
    names = {"max_features": "features per split", "min_samples_leaf": "minimum leaf size",
             "class_weight": "class weighting", "limit_depth": "depth limited", "max_depth": "maximum depth",
             "group_proto": "proto grouping"}
    imp = pd.Series(importance).rename(index=names).sort_values()

    fig, ax = plt.subplots(figsize=(col_width, 1.9))
    ax.barh(imp.index, imp.values, color=colours["rf"], height=0.6)

    for i in range(len(imp)):
        ax.text(imp.values[i] + 0.01, i, f"{imp.values[i]:.2f}", va="center", fontsize=7)

    ax.set_xlabel("importance")
    ax.set_xlim(0, imp.max() + 0.12)
    ax.grid(axis="y", visible=False)
    fig.savefig("figures/optuna_importance.pdf")
    plt.close(fig)

    summary = {"n_trials": len(trials), "n_complete": len(done), "n_pruned": len(pruned),
               "best_value": study.best_value, "best_trial": study.best_trial.number,
               "importance": importance}
    json.dump(summary, open("results/optuna_summary.json", "w"), indent=1)
    print(summary)


def sweep_plots():

    sweep = pd.read_json("results/sweep_weight_leaf.json")
    fig, axes = plt.subplots(1, 2, figsize=(col_width * 2, 2.3), sharex=True)

    # left plot is macro-F1 and right plot is balanced accuracy
    for ax, metric, title in zip(axes, ["f1_macro", "balanced_accuracy"], ["Macro-F1", "Balanced accuracy"]):

        for weight in ["None", "balanced", "balanced_subsample"]:

            s = sweep[sweep["class_weight"] == weight]
            scores = np.array(s[metric].tolist())
            ax.errorbar(s["min_samples_leaf"], scores.mean(axis=1), yerr=scores.std(axis=1), marker="o",
                        markersize=4, capsize=2, color=weight_colours[weight], label=weight_labels[weight])

        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30], ["1", "3", "10", "30"])
        ax.set_xlabel("minimum instances per leaf")
        ax.set_title(title)

    axes[0].legend(frameon=False)
    fig.savefig("figures/weight_leaf.pdf")
    plt.close(fig)

    # print the means for writing up
    sweep["f1"] = sweep["f1_macro"].apply(np.mean)
    sweep["ba"] = sweep["balanced_accuracy"].apply(np.mean)
    print(sweep[["class_weight", "min_samples_leaf", "f1", "ba"]].round(4).to_string())

    proto = pd.read_json("results/sweep_proto.json")
    proto["f1"] = proto["f1_macro"].apply(np.mean)
    proto["ba"] = proto["balanced_accuracy"].apply(np.mean)
    print(proto[["max_features", "group_proto", "f1", "ba"]].round(4).to_string())


def size_plot():

    d = json.load(open("results/ensemble_size.json"))
    sizes = np.array(d["sizes"])

    fig, ax = plt.subplots(figsize=(col_width, 2.3))

    for metric, style, label in [("f1_macro", "-", "macro-F1"), ("balanced_accuracy", "--", "balanced accuracy")]:

        # mean and spread over the 5 folds
        scores = np.array(d[metric])
        mean = scores.mean(axis=0)
        std = scores.std(axis=0)

        ax.plot(sizes, mean, style, marker="o", markersize=3.5, color=colours["rf"], label=label)
        ax.fill_between(sizes, mean - std, mean + std, color=colours["rf"], alpha=0.15, lw=0)
        print(metric, dict(zip(sizes.tolist(), mean.round(4))))

    ax.set_xscale("log")
    ax.set_xlabel("number of trees")
    ax.set_ylabel("score")
    ax.legend(frameon=False)
    fig.savefig("figures/ensemble_size.pdf")
    plt.close(fig)


def diversity_plot():

    results = json.load(open("results/diversity.json"))

    m = []
    forest = []
    tree = []
    agreement = []
    for r in results:

        m.append(r["max_features"])
        forest.append(r["ensemble_f1"])
        tree.append(r["tree_f1"])
        agreement.append(r["agreement"])
        print(r["max_features"], np.mean(r["ensemble_f1"]).round(4), np.mean(r["tree_f1"]).round(4), np.mean(r["agreement"]).round(4))

    forest = np.array(forest)
    tree = np.array(tree)
    agreement = np.array(agreement)

    fig, axes = plt.subplots(1, 2, figsize=(col_width * 2, 2.3))

    # left is the score of the forest and of one tree on average
    axes[0].errorbar(m, forest.mean(axis=1), yerr=forest.std(axis=1), marker="o", markersize=3.5, capsize=2,
                     color=colours["rf"], label="forest")
    axes[0].errorbar(m, tree.mean(axis=1), yerr=tree.std(axis=1), marker="o", markersize=3.5, capsize=2,
                     color=colours["tree"], label="mean individual tree")
    axes[0].set_ylabel("macro-F1")
    axes[0].legend(frameon=False)

    # right is how often two trees agree
    axes[1].errorbar(m, agreement.mean(axis=1), yerr=agreement.std(axis=1), marker="o", markersize=3.5, capsize=2,
                     color=colours["rf"])
    axes[1].set_ylabel("mean pairwise agreement")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("features considered per split")
        ax.set_xticks(m, [str(v) for v in m])
        ax.minorticks_off()

    fig.savefig("figures/diversity.pdf")
    plt.close(fig)


def depth_plot():

    d = json.load(open("results/depth_curve.json"))
    fig, ax = plt.subplots(figsize=(col_width, 2.4))

    for name in ["rf", "tree"]:

        # dashed for training and solid for validation
        for split, style, split_label in [("train", "--", "training"), ("test", "-", "validation")]:

            scores = np.array(d[name][split]).mean(axis=1)
            ax.plot(d["depths"], scores, style, marker="o", markersize=3, color=colours[name],
                    label=f"{labels[name]}, {split_label}")
            print(name, split, dict(zip(d["depths"], scores.round(3))))

    ax.set_xlabel("maximum depth")
    ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, loc="lower right")
    fig.savefig("figures/depth.pdf")
    plt.close(fig)


def final_results():

    folds = json.load(open("results/final_folds.json"))
    oof = pd.read_csv("results/final_oof_predictions.csv")
    models = ["rf", "tree", "knn"]

    scores = {}
    for m in models:
        scores[m] = pd.DataFrame(folds[m])

    # mean and std over the folds for each model
    summary = {}
    for m in models:

        summary[m] = {
            "f1_mean": scores[m]["f1_macro"].mean(), "f1_std": scores[m]["f1_macro"].std(),
            "ba_mean": scores[m]["balanced_accuracy"].mean(), "ba_std": scores[m]["balanced_accuracy"].std(),
            "fit_time": scores[m]["fit_time"].mean(), "predict_time": scores[m]["predict_time"].mean()
        }

    summary = pd.DataFrame(summary).T
    summary.to_csv("results/final_summary.csv")

    # each fold trains on 4/5 of the data and validates on 1/5
    n_test = len(oof) / 5
    n_train = len(oof) - n_test

    tests = []
    for other in ["tree", "knn"]:

        for metric in ["f1_macro", "balanced_accuracy"]:

            diff = scores["rf"][metric].to_numpy() - scores[other][metric].to_numpy()

            # normal paired t test like assignment 2
            t, p = stats.ttest_rel(scores["rf"][metric], scores[other][metric])

            # nadeau and bengio correction makes the variance bigger because the training folds overlap
            t_corr = diff.mean() / np.sqrt((1 / 5 + n_test / n_train) * np.var(diff, ddof=1))
            p_corr = 2 * stats.t.sf(abs(t_corr), 4)

            tests.append({"vs": other, "metric": metric, "mean_diff": diff.mean(), "t": t, "p": p,
                          "t_corr": t_corr, "p_corr": p_corr})

    tests = pd.DataFrame(tests)

    # holm correction over the four corrected p values
    order = np.argsort(tests["p_corr"].to_numpy())
    holm = np.empty(len(tests))
    running_max = 0
    for rank in range(len(order)):

        i = order[rank]
        running_max = max(running_max, min(1, (len(tests) - rank) * tests["p_corr"].iloc[i]))
        holm[i] = running_max

    tests["p_holm"] = holm
    tests.to_csv("results/final_tests.csv", index=False)

    # per class scores from the out of fold predictions
    per_class = []
    for m in models:

        precision, recall, f1 = per_class_scores(oof["y"], oof[m])
        per_class.append(pd.DataFrame({
            "model": m, "class": class_names,
            "precision": precision, "recall": recall, "f1": f1,
            "support": np.bincount(oof["y"], minlength=10)
        }))

    per_class = pd.concat(per_class)
    per_class.to_csv("results/final_per_class.csv", index=False)

    width = 0.26

    # bar chart of the two metrics for all three models
    fig, ax = plt.subplots(figsize=(col_width, 2.3))
    x = np.arange(2)

    for i in range(len(models)):

        m = models[i]
        means = [summary.loc[m, "f1_mean"], summary.loc[m, "ba_mean"]]
        stds = [summary.loc[m, "f1_std"], summary.loc[m, "ba_std"]]
        pos = x + (i - 1) * width
        ax.bar(pos, means, width * 0.92, yerr=stds, capsize=2, color=colours[m], label=labels[m])

        # number on top of each bar
        for j in range(2):
            ax.text(pos[j], means[j] + 0.012, f"{means[j]:.3f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x, ["macro-F1", "balanced accuracy"])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("mean score over five folds")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.savefig("figures/comparison.pdf")
    plt.close(fig)

    # f1 for each class
    fig, ax = plt.subplots(figsize=(col_width * 2, 2.3))
    x = np.arange(10)

    for i in range(len(models)):

        m = models[i]
        ax.bar(x + (i - 1) * width, per_class[per_class["model"] == m]["f1"], width * 0.92, color=colours[m], label=labels[m])

    ax.set_xticks(x, class_names, rotation=30, ha="right")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.savefig("figures/per_class.pdf")
    plt.close(fig)

    # rows normalised so each row shows where that class ended up
    cm = confusion_matrix(oof["y"], oof["rf"])
    cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(col_width, col_width))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)

    for i in range(10):

        for j in range(10):

            # skip the tiny cells so the plot isnt cluttered
            if cm[i, j] >= 0.01:

                colour = "white" if cm[i, j] > 0.55 else "#0b0b0b"
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=5.5, color=colour)

    ax.set_xticks(range(10), class_names, rotation=60, ha="right")
    ax.set_yticks(range(10), class_names)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.grid(False)
    fig.savefig("figures/rf_confusion.pdf")
    plt.close(fig)

    # how much of the importance sits in the top features
    imp = pd.read_csv("results/feature_importances.csv", index_col=0)
    for m in imp:

        s = imp[m].sort_values(ascending=False)
        print(m, "top5", s.head(5).sum().round(3), "top15", s.head(15).sum().round(3))
        print(s.head(6).round(3).to_string())

    print(summary.round(4).to_string())
    print(tests.round(4).to_string())
    print(per_class.round(3).to_string())


def corrected_p(a, b, n_runs):
    """ Nadeau and Bengio corrected resampled t test over n_runs scores per model, 4/5 train and 1/5 test """

    diff = np.asarray(a) - np.asarray(b)
    t = diff.mean() / np.sqrt((1 / n_runs + 1 / 4) * np.var(diff, ddof=1))

    return diff.mean(), 2 * stats.t.sf(abs(t), n_runs - 1)


def extra_results():

    folds = json.load(open("results/final_folds.json"))
    oof = pd.read_csv("results/final_oof_predictions.csv")
    y = oof["y"].to_numpy()

    seed42 = {}
    for m in ["rf", "tree", "knn"]:

        seed42[m] = {}
        for metric in ["f1_macro", "balanced_accuracy"]:
            seed42[m][metric] = pd.DataFrame(folds[m])[metric].tolist()

    # forest with smaller leaves against the tree on the seed 42 folds, leaf 20 is the tuned forest
    leaf = json.load(open("results/leaf_full.json"))
    leaf["20"] = seed42["rf"]

    rows = []
    for size in sorted(leaf, key=int):

        row = {"leaf": int(size)}
        for metric in ["f1_macro", "balanced_accuracy"]:

            diff, p = corrected_p(leaf[size][metric], seed42["tree"][metric], 5)
            row[metric] = np.mean(leaf[size][metric])
            row[metric + "_std"] = np.std(leaf[size][metric], ddof=1)
            row[metric + "_diff"] = diff
            row[metric + "_p"] = p

        rows.append(row)

    leaf_table = pd.DataFrame(rows)
    leaf_table.to_csv("results/leaf_full_summary.csv", index=False)
    print(leaf_table.round(4).to_string())

    # trade off between the two metrics, forest leaf sizes as a curve and the other models as points
    fig, ax = plt.subplots(figsize=(col_width, 2.4))
    ax.plot(leaf_table["balanced_accuracy"], leaf_table["f1_macro"], marker="o", markersize=4,
            color=colours["rf"], label=labels["rf"])

    for _, r in leaf_table.iterrows():
        ax.annotate(f"leaf {r['leaf']:.0f}", (r["balanced_accuracy"], r["f1_macro"]), textcoords="offset points",
                    xytext=(4, 4), fontsize=7, color=colours["rf"])

    for m in ["tree", "knn"]:
        ax.scatter(np.mean(seed42[m]["balanced_accuracy"]), np.mean(seed42[m]["f1_macro"]), s=30, marker="s",
                   color=colours[m], label=labels[m], zorder=3)

    ax.set_xlabel("balanced accuracy")
    ax.set_ylabel("macro-F1")
    ax.legend(frameon=False, loc="lower left")
    fig.savefig("figures/tradeoff.pdf")
    plt.close(fig)

    # three repeats of five fold cv, the corrected test uses all 15 paired scores
    rep = json.load(open("results/repeated_cv.json"))
    all_scores = {}
    for m in ["rf", "tree", "knn"]:

        all_scores[m] = {}
        for metric in ["f1_macro", "balanced_accuracy"]:

            all_scores[m][metric] = seed42[m][metric] + rep["43"][m][metric] + rep["44"][m][metric]
            print(m, metric, np.mean(all_scores[m][metric]).round(4), np.std(all_scores[m][metric], ddof=1).round(4))

    tests = []
    for other in ["tree", "knn"]:

        for metric in ["f1_macro", "balanced_accuracy"]:

            diff, p = corrected_p(all_scores["rf"][metric], all_scores[other][metric], 15)
            wins = int((np.array(all_scores["rf"][metric]) > np.array(all_scores[other][metric])).sum())
            tests.append({"vs": other, "metric": metric, "mean_diff": diff, "p_corr": p, "rf_wins": wins})

    tests = pd.DataFrame(tests)

    # holm over the four tests, same as for the single run
    order = np.argsort(tests["p_corr"].to_numpy())
    holm = np.empty(len(tests))
    running_max = 0
    for rank in range(len(order)):

        i = order[rank]
        running_max = max(running_max, min(1, (len(tests) - rank) * tests["p_corr"].iloc[i]))
        holm[i] = running_max

    tests["p_holm"] = holm
    tests.to_csv("results/repeated_tests.csv", index=False)
    print(tests.round(4).to_string())

    # validation scores on only the instances that were never in the tuning sample
    tuned_on = np.zeros(len(y), dtype=bool)
    tuned_on[stratified_sample(y, 50000, seed=42)] = True

    unseen = []
    for m in ["rf", "tree", "knn"]:

        f1 = []
        ba = []
        for _, val_idx in stratified_kfold(y, n_splits=5, seed=42):

            val_idx = val_idx[~tuned_on[val_idx]]
            f1.append(f1_macro(y[val_idx], oof[m].to_numpy()[val_idx]))
            ba.append(balanced_accuracy(y[val_idx], oof[m].to_numpy()[val_idx]))

        unseen.append({"model": m, "f1_macro": np.mean(f1), "balanced_accuracy": np.mean(ba),
                       "n_unseen": int((~tuned_on).sum())})

    unseen = pd.DataFrame(unseen)
    unseen.to_csv("results/unseen_scores.csv", index=False)
    print(unseen.round(4).to_string())


# run whichever steps are given on the command line
for step in sys.argv[1:]:
    globals()[step]()
