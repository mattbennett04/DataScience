""" Optuna search for the random forest on the 50k tuning sample, same setup as the knn search in assignment 2 """
import json

import numpy as np
import optuna

from pipelines import load_data, tuning_sample, ForestModel
from ml_utils import stratified_kfold, f1_macro, balanced_accuracy

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y, cat, num = load_data()
X_search, y_search = tuning_sample(X, y)
folds = stratified_kfold(y_search, n_splits=5, seed=42)


def objective(trial):

    max_features = trial.suggest_float("max_features", 0.05, 0.6)
    class_weight = trial.suggest_categorical("class_weight", ["none", "balanced", "balanced_subsample"])
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50, log=True)

    # depth is either unlimited or somewhere between 8 and 40
    limit_depth = trial.suggest_categorical("limit_depth", [True, False])
    max_depth = None
    if limit_depth:
        max_depth = trial.suggest_int("max_depth", 8, 40)

    # put the rare proto levels into one column or not
    group_proto = trial.suggest_categorical("group_proto", [True, False])

    if class_weight == "none":
        class_weight = None

    scores = []
    for i, (train_idx, val_idx) in enumerate(folds):

        model = ForestModel(cat, num, proto_min_frequency=500 if group_proto else None, n_estimators=100,
                            max_features=max_features, class_weight=class_weight,
                            min_samples_leaf=min_samples_leaf, max_depth=max_depth)
        model.fit(X_search.iloc[train_idx], y_search[train_idx])
        pred = model.predict(X_search.iloc[val_idx])

        # average of both metrics so the search cant just sacrifice one of them
        f1 = f1_macro(y_search[val_idx], pred)
        bal_acc = balanced_accuracy(y_search[val_idx], pred)
        scores.append((f1 + bal_acc) / 2)

        # keep the fold scores so the best trial can be reported per metric later
        trial.set_user_attr(f"f1_fold{i}", f1)
        trial.set_user_attr(f"bal_acc_fold{i}", bal_acc)

        # stop early if this trial is already worse than the median
        trial.report(np.mean(scores), i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)


study = optuna.create_study(
    study_name="rf",
    storage="sqlite:///results/rf_study.db",
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=2)
)

# one trial at a time, the forest itself already uses 16 processes
n_left = 120 - len(study.trials)
if n_left > 0:
    study.optimize(objective, n_trials=n_left)

best = study.best_params
print(best, study.best_value)

# fold scores of the best trial
f1s = []
bas = []
for i in range(5):

    f1s.append(study.best_trial.user_attrs[f"f1_fold{i}"])
    bas.append(study.best_trial.user_attrs[f"bal_acc_fold{i}"])

# turn the optuna params back into forest settings
best_params = {
    "proto_min_frequency": 500 if best["group_proto"] else None,
    "rf": {
        "max_features": round(best["max_features"], 3),
        "class_weight": None if best["class_weight"] == "none" else best["class_weight"],
        "min_samples_leaf": best["min_samples_leaf"],
        "max_depth": best["max_depth"] if best["limit_depth"] else None
    },
    "n_estimators": 200,  # picked from ensemble_size_experiment.py
    "tuning_scores": {"f1_macro": np.mean(f1s), "f1_std": np.std(f1s, ddof=1),
                      "balanced_accuracy": np.mean(bas), "bal_acc_std": np.std(bas, ddof=1)}
}

# save the best config for the other scripts
json.dump(best_params, open("results/best_params.json", "w"), indent=1)
print(best_params)
