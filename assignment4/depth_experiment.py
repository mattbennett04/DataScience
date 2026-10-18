""" Training vs validation macro-F1 over max_depth for the tuned forest and the assignment 2 tree """
import json

import numpy as np

from pipelines import load_data, tuning_sample, ForestModel, TreeModel
from ml_utils import stratified_kfold, f1_macro

X, y, cat, num = load_data()
X_search, y_search = tuning_sample(X, y)

best = json.load(open("results/best_params.json"))
depths = [2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40, 60]

results = {"depths": depths}

for name in ["rf", "tree"]:

    train_scores = np.zeros((len(depths), 5))
    test_scores = np.zeros((len(depths), 5))

    for d in range(len(depths)):

        for f, (train_idx, val_idx) in enumerate(stratified_kfold(y_search, n_splits=5, seed=42)):

            if name == "rf":

                params = dict(best["rf"])
                params["max_depth"] = depths[d]
                model = ForestModel(cat, num, proto_min_frequency=best["proto_min_frequency"], n_estimators=100, **params)

            else:

                model = TreeModel(cat, num, max_depth=depths[d])

            model.fit(X_search.iloc[train_idx], y_search[train_idx])

            # score on the data it trained on and on the held out fold
            train_scores[d, f] = f1_macro(y_search[train_idx], model.predict(X_search.iloc[train_idx]))
            test_scores[d, f] = f1_macro(y_search[val_idx], model.predict(X_search.iloc[val_idx]))

        print(name, depths[d], train_scores[d].mean().round(4), test_scores[d].mean().round(4), flush=True)

    results[name] = {"train": train_scores.tolist(), "test": test_scores.tolist()}

json.dump(results, open("results/depth_curve.json", "w"), indent=1)
