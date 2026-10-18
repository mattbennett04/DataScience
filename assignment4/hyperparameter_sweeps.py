""" Sweeps around the Optuna best config, class weighting vs leaf size and proto grouping vs max_features """
import json

import numpy as np
import pandas as pd

from pipelines import load_data, tuning_sample, ForestModel
from ml_utils import stratified_kfold, f1_macro, balanced_accuracy

X, y, cat, num = load_data()
X_search, y_search = tuning_sample(X, y)
folds = stratified_kfold(y_search, n_splits=5, seed=42)

best = json.load(open("results/best_params.json"))


def cv_scores(proto_min_frequency, params):
    """ Macro-F1 and balanced accuracy of a 100 tree forest on each tuning fold """

    f1s, bas = [], []

    for train_idx, val_idx in folds:

        model = ForestModel(cat, num, proto_min_frequency=proto_min_frequency, n_estimators=100, **params)
        model.fit(X_search.iloc[train_idx], y_search[train_idx])
        pred = model.predict(X_search.iloc[val_idx])

        f1s.append(f1_macro(y_search[val_idx], pred))
        bas.append(balanced_accuracy(y_search[val_idx], pred))

    return f1s, bas


# sweep 1 class weighting vs leaf size
# depth is unlimited here so the leaf size is what decides how pure the leaves are
results = []
for class_weight in [None, "balanced", "balanced_subsample"]:

    for leaf in [1, 3, 10, 30]:

        params = dict(best["rf"])
        params["class_weight"] = class_weight
        params["min_samples_leaf"] = leaf
        params["max_depth"] = None

        f1s, bas = cv_scores(best["proto_min_frequency"], params)
        results.append({"class_weight": str(class_weight), "min_samples_leaf": leaf, "f1_macro": f1s, "balanced_accuracy": bas})
        print(class_weight, leaf, round(np.mean(f1s), 4), round(np.mean(bas), 4), flush=True)

pd.DataFrame(results).to_json("results/sweep_weight_leaf.json", orient="records", indent=1)

# sweep 2 grouping the rare proto levels vs max_features
results = []
for max_features in ["sqrt", 0.15, 0.3, 0.5]:

    for proto_min_frequency in [None, 500]:

        params = dict(best["rf"])
        params["max_features"] = max_features

        f1s, bas = cv_scores(proto_min_frequency, params)
        results.append({"max_features": str(max_features), "group_proto": proto_min_frequency is not None,
                        "f1_macro": f1s, "balanced_accuracy": bas})
        print(max_features, proto_min_frequency, round(np.mean(f1s), 4), round(np.mean(bas), 4), flush=True)

pd.DataFrame(results).to_json("results/sweep_proto.json", orient="records", indent=1)
