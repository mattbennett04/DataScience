# tune and train the three networks
# optuna tunes each network on the validation block of every fold then the best
# settings are trained again with 10 seeds and scored on the test block
#
# usage
#   python train.py
#   python train.py --datasets H279 --models elman --trials 5 --seeds 2

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import optuna
import torch

from models import MODELS, build_model
from preprocess import CONFIGS, load_series, make_folds, make_windows, prepare_fold

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MAX_EPOCHS = 300
PATIENCE = 20  # epochs without improvement before stopping


def to_tensors(windows):
    X, y, t = windows
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # shape n x window x 1
    y = torch.tensor(y, dtype=torch.float32)
    return X, y, t


def mse(model, X, y):
    model.eval()
    with torch.no_grad():
        return torch.mean((model(X) - y) ** 2).item()


# train with adam on mse stop early on the validation loss
def train_model(model, train, val, lr, weight_decay, batch_size, trial=None):
    Xtr, ytr, _ = train
    Xva, yva, _ = val
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    history = {"train": [], "val": []}  # for the loss curve plots

    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(ytr))  # shuffle every epoch
        total = 0.0

        for i in range(0, len(ytr), batch_size):
            idx = order[i:i + batch_size]
            opt.zero_grad()
            loss = torch.mean((model(Xtr[idx]) - ytr[idx]) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stop exploding gradients
            opt.step()
            total += loss.item() * len(idx)

        val_loss = mse(model, Xva, yva)
        history["train"].append(total / len(ytr))  # average over the epoch
        history["val"].append(val_loss)

        # keep the best weights stop if no improvement for PATIENCE epochs
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        elif epoch - best_epoch >= PATIENCE:
            break

        # let the median pruner stop bad trials early
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    model.load_state_dict(best_state)  # go back to the best epoch
    return best_loss, best_epoch, history


# predictions back in the original units of the series
def predict(model, windows, pp, x):
    X, _, t = windows
    model.eval()
    with torch.no_grad():
        z_pred = model(X).numpy()
    return pp.inverse_one_step(z_pred, t, x)


def errors(pred, actual):
    err = pred - actual
    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    smape = np.mean(200 * np.abs(err) / (np.abs(actual) + np.abs(pred)))
    return {"rmse": float(rmse), "mae": float(mae), "smape": float(smape)}


def search_space(trial, default_window):
    windows = [default_window // 2, default_window, default_window * 2]
    return {
        "hidden": trial.suggest_int("hidden", 2, 40, log=True),
        "window": trial.suggest_categorical("window", windows),
        "lr": trial.suggest_float("lr", 1e-4, 3e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }


def load_fold(dataset, fold, window):
    d = prepare_fold(dataset, fold, window=window)
    d["train"] = to_tensors(d["train"])
    d["val"] = to_tensors(d["val"])
    d["test"] = to_tensors(d["test"])

    # everything after the training block used for the drift measures
    d["after"] = to_tensors(make_windows(d["z"], window, fold[0][1], len(d["x"])))
    return d


# tune retrain and test one network on one fold of one dataset
def run_one(dataset, model_name, fold_idx, n_trials, n_seeds):
    torch.set_num_threads(1)  # one core per process
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start = time.time()

    x = load_series(dataset)
    fold = make_folds(len(x))[fold_idx]
    default_window = CONFIGS[dataset]["window"]
    data = {}  # windows for each window length only built once

    def objective(trial):
        p = search_space(trial, default_window)
        if p["window"] not in data:
            data[p["window"]] = load_fold(dataset, fold, p["window"])
        d = data[p["window"]]

        torch.manual_seed(trial.number)
        model = build_model(model_name, p["hidden"])
        val_loss, _, _ = train_model(model, d["train"], d["val"], p["lr"], p["weight_decay"],
                                     p["batch_size"], trial=trial)
        return val_loss

    # tpe sampler with a median pruner seeded so it can be repeated
    sampler = optuna.samplers.TPESampler(seed=fold_idx)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)
    best = study.best_params

    # train the best settings again from scratch with different seeds
    if best["window"] not in data:
        data[best["window"]] = load_fold(dataset, fold, best["window"])
    d = data[best["window"]]
    t_test = d["test"][2]

    runs = []
    for seed in range(n_seeds):
        torch.manual_seed(1000 + seed)
        model = build_model(model_name, best["hidden"])
        val_loss, best_epoch, history = train_model(model, d["train"], d["val"], best["lr"],
                                                    best["weight_decay"], best["batch_size"])

        pred = predict(model, d["test"], d["pp"], x)
        runs.append({
            "seed": 1000 + seed,
            "val_mse_scaled": val_loss,
            "train_mse_scaled": mse(model, d["train"][0], d["train"][1]),
            "best_epoch": best_epoch,
            "history": history,
            "test": errors(pred, x[t_test]),
            "pred": pred.tolist(),
            "pred_after_train": predict(model, d["after"], d["pp"], x).tolist(),
        })

    naive = errors(x[t_test - 1], x[t_test])  # last value as the forecast
    trials = [{"params": t.params, "value": t.value, "state": t.state.name} for t in study.trials]

    result = {
        "dataset": dataset,
        "model": model_name,
        "fold": fold_idx,
        "fold_ranges": fold,
        "best_params": best,
        "best_val_mse_scaled": study.best_value,
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_trials": n_trials,
        "n_pruned": sum(t["state"] == "PRUNED" for t in trials),
        "trials": trials,
        "naive_test": naive,
        "test_t": t_test.tolist(),
        "after_train_t": d["after"][2].tolist(),
        "runs": runs,
        "seconds": time.time() - start,
    }
    with open(os.path.join(RESULTS_DIR, f"{dataset}_{model_name}_fold{fold_idx}.json"), "w") as f:
        json.dump(result, f)

    rmse = np.mean([r["test"]["rmse"] for r in runs])
    secs = time.time() - start
    return f"{dataset} {model_name} fold{fold_idx}: test rmse {rmse:.4g} (naive {naive['rmse']:.4g}) best {best} [{secs:.0f}s]"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(CONFIGS))
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() - 2))
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # skip runs that already have a results file
    todo = []
    for dataset in args.datasets:
        for model_name in args.models:
            for k in args.folds:
                if os.path.exists(os.path.join(RESULTS_DIR, f"{dataset}_{model_name}_fold{k}.json")):
                    print(f"skipping {dataset} {model_name} fold{k}, already done")
                else:
                    todo.append((dataset, model_name, k))

    # longest jobs first so the processes finish around the same time
    cost = {"mrn": 3, "elman": 2, "jordan": 1}
    todo.sort(key=lambda job: len(load_series(job[0])) * cost[job[1]], reverse=True)

    print(f"{len(todo)} runs on {args.jobs} processes")
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, *job, args.trials, args.seeds) for job in todo]
        for fut in as_completed(futures):
            print(fut.result(), flush=True)
