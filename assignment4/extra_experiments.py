""" Follow up runs on the full data, leaf sizes on the final folds and the final comparison with two more fold seeds, pass the step names as arguments """
import json
import os
import sys

from pipelines import load_data, ForestModel, TreeModel, KNNModel
from ml_utils import stratified_kfold, f1_macro, balanced_accuracy

X, y, cat, num = load_data()
best = json.load(open("results/best_params.json"))


def make_model(name, **rf_changes):

    if name == "rf":

        params = dict(best["rf"])
        params.update(rf_changes)
        return ForestModel(cat, num, proto_min_frequency=best["proto_min_frequency"],
                           n_estimators=best["n_estimators"], **params)

    if name == "tree":
        return TreeModel(cat, num)

    return KNNModel(X, cat, num)


def cv_scores(name, seed, **rf_changes):
    """ Macro-F1 and balanced accuracy of one model on each of the 5 full data folds for this seed """

    scores = {"f1_macro": [], "balanced_accuracy": []}

    for train_idx, val_idx in stratified_kfold(y, n_splits=5, seed=seed):

        # fresh model every fold
        model = make_model(name, **rf_changes)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[val_idx])

        scores["f1_macro"].append(f1_macro(y[val_idx], pred))
        scores["balanced_accuracy"].append(balanced_accuracy(y[val_idx], pred))
        print(name, seed, rf_changes, scores["f1_macro"][-1], scores["balanced_accuracy"][-1], flush=True)

    return scores


def leaf():

    # leaf 20 is the tuned forest and is already in final_folds.json
    # pick up from an interrupted run, sizes already in the file are skipped
    results = {}
    if os.path.exists("results/leaf_full.json"):
        results = json.load(open("results/leaf_full.json"))

    for size in [3, 6, 10]:

        if str(size) in results:
            continue

        results[str(size)] = cv_scores("rf", 42, min_samples_leaf=size)
        json.dump(results, open("results/leaf_full.json", "w"), indent=1)


def repeat():

    results = {}
    for seed in [43, 44]:

        results[seed] = {}
        for name in ["rf", "tree", "knn"]:

            results[seed][name] = cv_scores(name, seed)
            json.dump(results, open("results/repeated_cv.json", "w"), indent=1)


for step in sys.argv[1:]:
    globals()[step]()
