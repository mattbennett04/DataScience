""" Final comparison of the tuned forest, the assignment 2 tree and the assignment 2 knn on the full data, same 5 folds for all three """
import json
import time

import numpy as np
import pandas as pd

from pipelines import load_data, tuning_sample, ForestModel, TreeModel, KNNModel
from ml_utils import stratified_kfold, f1_macro, balanced_accuracy

X, y, cat, num = load_data()
best = json.load(open("results/best_params.json"))

# score the assignment 2 tree on the tuning sample too so the tuning scores have something to compare to
X_search, y_search = tuning_sample(X, y)
ref = {"f1_macro": [], "balanced_accuracy": []}
for train_idx, val_idx in stratified_kfold(y_search, n_splits=5, seed=42):

    pred = TreeModel(cat, num).fit(X_search.iloc[train_idx], y_search[train_idx]).predict(X_search.iloc[val_idx])
    ref["f1_macro"].append(f1_macro(y_search[val_idx], pred))
    ref["balanced_accuracy"].append(balanced_accuracy(y_search[val_idx], pred))

json.dump(ref, open("results/tree_tuning_ref.json", "w"), indent=1)

fold_results = {}
oof = pd.DataFrame({"y": y})
importances = {}

for name in ["rf", "tree", "knn"]:

    results = []
    oof_pred = np.empty(len(y), dtype=y.dtype)
    fold_imp = []

    # same seed so all three models get the same 5 folds and the scores are paired
    for i, (train_idx, val_idx) in enumerate(stratified_kfold(y, n_splits=5, seed=42)):

        # fresh model every fold
        if name == "rf":
            model = ForestModel(cat, num, proto_min_frequency=best["proto_min_frequency"],
                                n_estimators=best["n_estimators"], **best["rf"])
        elif name == "tree":
            model = TreeModel(cat, num)
        else:
            model = KNNModel(X, cat, num)

        start = time.time()
        model.fit(X.iloc[train_idx], y[train_idx])
        fit_time = time.time() - start

        start = time.time()
        pred = model.predict(X.iloc[val_idx])
        predict_time = time.time() - start

        # keep the out of fold predictions for the per class results
        oof_pred[val_idx] = pred

        results.append({
            "fold": i,
            "f1_macro": f1_macro(y[val_idx], pred),
            "balanced_accuracy": balanced_accuracy(y[val_idx], pred),
            "fit_time": fit_time,
            "predict_time": predict_time
        })
        print(name, results[-1], flush=True)

        # knn has no feature importances
        if name != "knn":
            fold_imp.append(pd.Series(model.forest.feature_importances_, index=model.feature_names))

    fold_results[name] = results
    oof[name] = oof_pred

    # average the importances over the 5 folds
    if name != "knn":
        importances[name] = pd.concat(fold_imp, axis=1).mean(axis=1)

json.dump(fold_results, open("results/final_folds.json", "w"), indent=1)
oof.to_csv("results/final_oof_predictions.csv", index=False)
pd.DataFrame(importances).to_csv("results/feature_importances.csv")
