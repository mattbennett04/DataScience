""" Strength vs correlation of the trees for different max_features values, measured on the validation folds """
import itertools
import json

import numpy as np

from pipelines import load_data, tuning_sample, ForestModel
from ml_utils import stratified_kfold, f1_macro

X, y, cat, num = load_data()
X_search, y_search = tuning_sample(X, y)

best = json.load(open("results/best_params.json"))

# count the encoded features so m can go from 1 all the way up to bagging
p = len(ForestModel(cat, num, proto_min_frequency=best["proto_min_frequency"], n_estimators=1).fit(X_search, y_search).feature_names)
m_values = []
for m in [1, 2, 4, 8, 16, 32, 64, 128]:

    if m < p:
        m_values.append(m)

m_values.append(p)
print("features", p, m_values, flush=True)

# pick 500 random pairs out of the 100 trees for the agreement estimate
rng = np.random.default_rng(42)
all_pairs = list(itertools.combinations(range(100), 2))
pairs = []
for i in rng.choice(len(all_pairs), 500, replace=False):
    pairs.append(all_pairs[i])

results = []
for m in m_values:

    forest_f1 = []
    tree_f1 = []
    agreement = []

    for train_idx, val_idx in stratified_kfold(y_search, n_splits=5, seed=42):

        params = dict(best["rf"])
        params["max_features"] = m

        model = ForestModel(cat, num, proto_min_frequency=best["proto_min_frequency"], n_estimators=100, **params)
        model.fit(X_search.iloc[train_idx], y_search[train_idx])

        y_val = y_search[val_idx]
        classes = model.forest.classes_

        # predictions of every single tree plus the forest average
        preds = []
        prob_sum = 0
        for proba in model.tree_probas(X_search.iloc[val_idx]):

            preds.append(classes[proba.argmax(axis=1)])
            prob_sum = prob_sum + proba

        preds = np.array(preds)
        forest_f1.append(f1_macro(y_val, classes[prob_sum.argmax(axis=1)]))

        # strength is how good one tree is on its own
        scores = []
        for p in preds:
            scores.append(f1_macro(y_val, p))
        tree_f1.append(np.mean(scores))

        # correlation is how often two trees give the same answer
        same = []
        for i, j in pairs:
            same.append((preds[i] == preds[j]).mean())
        agreement.append(np.mean(same))

    results.append({"max_features": m, "ensemble_f1": forest_f1, "tree_f1": tree_f1, "agreement": agreement})
    print(m, np.mean(forest_f1).round(4), np.mean(tree_f1).round(4), np.mean(agreement).round(4), flush=True)

json.dump(results, open("results/diversity.json", "w"), indent=1)
