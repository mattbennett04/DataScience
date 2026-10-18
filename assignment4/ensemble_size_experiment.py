""" Macro-F1 and balanced accuracy vs number of trees, using the first k trees of one 500 tree forest per fold """
import json

import numpy as np

from pipelines import load_data, tuning_sample, ForestModel
from ml_utils import stratified_kfold, f1_macro, balanced_accuracy

X, y, cat, num = load_data()
X_search, y_search = tuning_sample(X, y)

best = json.load(open("results/best_params.json"))
sizes = [1, 2, 5, 10, 20, 50, 100, 150, 200, 300, 400, 500]

all_f1 = []
all_bal_acc = []

for train_idx, val_idx in stratified_kfold(y_search, n_splits=5, seed=42):

    model = ForestModel(cat, num, proto_min_frequency=best["proto_min_frequency"], n_estimators=500, **best["rf"])
    model.fit(X_search.iloc[train_idx], y_search[train_idx])
    y_val = y_search[val_idx]
    classes = model.forest.classes_

    # a forest of k trees predicts the class with the highest average probability
    # so adding the probabilities up one tree at a time gives every size from one forest
    prob_sum = 0
    f1s = []
    bas = []

    for k, proba in enumerate(model.tree_probas(X_search.iloc[val_idx]), start=1):

        prob_sum = prob_sum + proba

        # score it when we hit one of the sizes we want
        if k in sizes:

            pred = classes[prob_sum.argmax(axis=1)]
            f1s.append(f1_macro(y_val, pred))
            bas.append(balanced_accuracy(y_val, pred))

    all_f1.append(f1s)
    all_bal_acc.append(bas)
    print(np.round(f1s, 4), flush=True)

json.dump({"sizes": sizes, "f1_macro": all_f1, "balanced_accuracy": all_bal_acc},
          open("results/ensemble_size.json", "w"), indent=1)
