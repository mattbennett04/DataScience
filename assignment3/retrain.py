# retraining after drift
#
# train before the break then predict forward
# if the recent error gets too high mark drift
# retrain after MIN_NEW steps then every EVERY steps
#
# none = no retraining
# fresh = new weights
# finetune = keep old weights
#
# python retrain.py
# python retrain.py --fixed --datasets M6864
# python retrain.py --summary

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from models import MODELS, build_model
from preprocess import CONFIGS, Preprocessor, load_series, make_folds, make_windows, prepare_fold
from train import to_tensors, train_model

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results_retrain")

DATASETS = ["D686", "M6864", "M26487"]
KAPPA = 2.0
ROLL = 5
MIN_NEW = 36
EVERY = 12

# fixed settings for comparing the networks
FIXED_PARAMS = {"hidden": 16, "window": 24, "lr": 0.005, "weight_decay": 1e-6, "batch_size": 16}


# pick the fold before the break
def setup(dataset):

    x = load_series(dataset)
    folds = make_folds(len(x))  # make the cv folds
    breaks = sorted(CONFIGS[dataset]["breaks"])  # known break points

    for i, brk in enumerate(breaks):
        before = [k for k, f in enumerate(folds) if f[1][1] <= brk]  # folds that end before the break
        if before:
            k = max(before)
            stop = breaks[i + 1] if i + 1 < len(breaks) else len(x)  # stop at next break or end
            return x, k, folds[k], brk, stop
        
    raise ValueError("no fold ends before a break in " + dataset)


def predict(model, pp, x, window, start, end):

    z = pp.transform(x)  # transform the series
    X, _, t = to_tensors(make_windows(z, window, start, end))
    model.eval()  # prediction mode

    with torch.no_grad():
        z_pred = model(X).numpy()
    return t, pp.inverse_one_step(z_pred, t, x)


def ape(pred, actual):
    return np.abs(pred - actual) / np.abs(actual)


# retrain using recent data keep the last 20% for validation
def retrain(model, x, cfg, params, data_start, now):

    window = params["window"]
    n_val = max(5, (now - data_start) // 5)  # last 20% for validation
    split = now - n_val  # train/validation split

    pp = Preprocessor(log=cfg["log"], seasonal_lag=cfg["seasonal_lag"], diff=cfg["diff"],
                      scaling="robust", clip=3)  # avoid big jumps dominating
    pp.fit(x, split, start=max(1, data_start - window))
    z = pp.transform(x)  # transform the series
    train = to_tensors(make_windows(z, window, data_start, split))
    val = to_tensors(make_windows(z, window, split, now))

    if len(train[1]) < 10:  # not enough training points
        return None, None

    train_model(model, train, val, params["lr"], params["weight_decay"], params["batch_size"])
    return model, pp


def deploy(model, pp, x, cfg, params, start, stop, normal, strategy, seed):
    window = params["window"]
    preds = np.full(stop, np.nan)  # store predictions by time index
    t, p = predict(model, pp, x, window, start, stop)
    preds[t] = p

    detected = None
    next_retrain = None
    retrains = []
    errs = []

    for t in range(start, stop):
        errs.append(ape(preds[t], x[t]))  # current percentage error

        if strategy != "none" and detected is None and np.mean(errs[-ROLL:]) > KAPPA * normal:
            detected = t  # drift found
            next_retrain = t + MIN_NEW  # wait for more new data

        # x[t] is known now so retrain before predicting t+1
        if next_retrain is not None and t + 1 >= next_retrain and t + 1 < stop:

            if strategy == "fresh":
                torch.manual_seed(seed * 1000 + len(retrains))  # different seed each retrain
                new = build_model(params["model"], params["hidden"])  # start from new weights
            else:
                new = model  # keep the current weights
            new, new_pp = retrain(new, x, cfg, params, detected, t + 1)
            if new is not None:
                model, pp = new, new_pp
                retrains.append(t + 1)  # save retrain time
                tt, p = predict(model, pp, x, window, t + 1, stop)
                preds[tt] = p
            next_retrain += EVERY

    return preds, detected, retrains


def run_one(dataset, model_name, seed, fixed=False, out_dir=OUT_DIR):
    torch.set_num_threads(1)
    start_time = time.time()
    x, k, fold, brk, stop = setup(dataset)

    if fixed:
        params = dict(FIXED_PARAMS)  # same settings for every model
    else:
        with open(os.path.join(HERE, "results", f"{dataset}_{model_name}_fold{k}.json")) as f:
            params = dict(json.load(f)["best_params"])  # tuned settings from main run

    params["model"] = model_name
    cfg = CONFIGS[dataset]

    # train the starting model
    d = prepare_fold(dataset, fold, window=params["window"])
    torch.manual_seed(1000 + seed)  # repeatable starting weights
    base = build_model(model_name, params["hidden"])

    train_model(base, to_tensors(d["train"]), to_tensors(d["val"]),
                params["lr"], params["weight_decay"], params["batch_size"])
    base_state = {key: v.clone() for key, v in base.state_dict().items()}  # save starting weights

    # normal error before deployment
    val_start, val_end = fold[1]
    tv, pv = predict(base, d["pp"], x, params["window"], val_start, val_end)
    normal = float(np.mean(ape(pv, x[tv])))

    out = {"dataset": dataset, "model": model_name, "seed": seed, "fold": k, "break": brk,
           "deploy_start": val_end, "stop": stop, "normal_error": normal, "params": params,
           "strategies": {}}
    
    for strategy in ["none", "fresh", "finetune"]:
        base.load_state_dict(base_state)
        model = base

        if strategy == "finetune":
            model = build_model(model_name, params["hidden"])  # make a copy
            model.load_state_dict(base_state)

        preds, detected, retrains = deploy(model, d["pp"], x, cfg, params, val_end, stop, normal, strategy, seed)
        out["strategies"][strategy] = {
            "pred": [None if np.isnan(v) else float(v) for v in preds[val_end:stop]],
            "detected": detected,
            "retrains": retrains,
        }

    with open(os.path.join(out_dir, f"{dataset}_{model_name}_seed{seed}.json"), "w") as f:
        json.dump(out, f)
    return f"{dataset} {model_name} seed{seed} [{time.time() - start_time:.0f}s]"


# simple ar baseline on log differences
# use naive forecast before the first fit
def ar_reference(dataset, p):

    x, k, fold, brk, stop = setup(dataset)
    ly = np.log(x)
    dl = np.diff(ly, prepend=np.nan)  # log changes
    preds = x[np.arange(brk, stop) - 1].astype(float)  # start with naive predictions
    fit_at = brk + MIN_NEW
    w = None

    for t in range(brk, stop):

        if t == fit_at:
            ts = np.arange(brk + p + 1, t)  # skip the jump
            X = np.array([dl[s - p:s] for s in ts])
            X = np.c_[X, np.ones(len(ts))]
            w = np.linalg.lstsq(X, dl[ts], rcond=None)[0]  # fit ar weights
            fit_at += EVERY

        if w is not None:
            preds[t - brk] = np.exp(ly[t - 1] + np.r_[dl[t - p:t], 1.0] @ w)

    return preds


# steps until the error drops back below the threshold
def recovery(err, normal):
    ratio = np.array([err[max(0, i - ROLL + 1):i + 1].mean() for i in range(len(err))]) / normal
    above = np.where(ratio > KAPPA)[0]  # error above threshold

    if len(above) == 0:
        return 0, False
    
    below = np.where(ratio[above[0]:] <= KAPPA)[0]  # error back below threshold

    if len(below) == 0:
        return len(ratio), True
    
    return int(above[0] + below[0]), False


def to_array(preds):
    return np.array([np.nan if v is None else v for v in preds])


def summary(out_dir):
    results = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*_seed*.json"))):
        with open(path) as f:
            r = json.load(f)
        results.setdefault(r["dataset"], {}).setdefault(r["model"], []).append(r)

    lines = [f"retraining under drift: trigger at rolling abs % error > {KAPPA}x validation error, "
             f"first retrain {MIN_NEW} steps after the trigger, then every {EVERY} steps",
             "rmse and recovery are over the steps from the break to the end of the segment",
             "rel = rmse / naive rmse, recovery uses the same definition as analyse.py (50 steps before the break)"]

    for dataset in DATASETS:

        if dataset not in results:
            continue

        first = next(iter(results[dataset].values()))[0]
        brk, stop, start = first["break"], first["stop"], first["deploy_start"]

        x = load_series(dataset)
        actual = x[brk:stop]  # true values after the break
        pre = np.arange(max(start, brk - 50), brk)
        naive_before = ape(x[pre - 1], x[pre]).mean()

        naive = x[brk - 1:stop - 1]
        naive_rmse = np.sqrt(np.mean((naive - actual) ** 2))  # baseline rmse
        rec, cens = recovery(ape(naive, actual), naive_before)

        lines.append(f"\n{dataset}, break at t={brk}, deployed from t={start} (fold {first['fold'] + 1}), "
                     f"{stop - brk} steps after the break")
        lines.append(f"  {'naive':22s} rmse {naive_rmse:9.2f}  rel 1.000  recovery {rec}{'+' if cens else ''}")

        ar = ar_reference(dataset, CONFIGS[dataset]["window"])
        ar_rmse = np.sqrt(np.mean((ar - actual) ** 2))
        rec, cens = recovery(ape(ar, actual), naive_before)
        lines.append(f"  {'linear AR (oracle)':22s} rmse {ar_rmse:9.2f}  rel {ar_rmse / naive_rmse:.3f}  "
                     f"recovery {rec}{'+' if cens else ''}")

        for model in MODELS:
            runs = results[dataset].get(model, [])

            if not runs:
                continue
            for strategy in ["none", "fresh", "finetune"]:

                rmses, recs, delays, n_retrains = [], [], [], []
                n_cens = 0
                false_alarms = 0

                for r in runs:

                    s = r["strategies"][strategy]
                    pred = to_array(s["pred"])
                    after = pred[brk - start:]
                    rmses.append(np.sqrt(np.mean((after - actual) ** 2)))  # rmse for this run

                    rec, cens = recovery(ape(after, actual), ape(pred[pre - start], x[pre]).mean())
                    recs.append(rec)
                    n_cens += cens

                    if s["detected"] is not None:
                        delays.append(s["detected"] - brk)
                        false_alarms += s["detected"] < brk  # drift detected too early
                    n_retrains.append(len(s["retrains"]))

                rel = np.array(rmses) / naive_rmse
                detect = f"detect {np.median(delays):+.0f}" if delays else ""
                lines.append(f"  {model + ' ' + strategy:22s} rmse {np.mean(rmses):9.2f}  rel {rel.mean():.3f} "
                             f"(sd {rel.std():.3f})  recovery median {np.median(recs):.0f} "
                             f"({n_cens}/{len(runs)} censored)  {detect}  false alarms {false_alarms}/{len(runs)}  "
                             f"retrains {np.median(n_retrains):.0f}")

    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args()

    out_dir = os.path.join(HERE, "results_retrain_fixed") if args.fixed else OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not args.summary:

        todo = []
        for d in args.datasets:
            for m in args.models:
                for s in range(args.seeds):
                    if not os.path.exists(os.path.join(out_dir, f"{d}_{m}_seed{s}.json")):
                        todo.append((d, m, s))  # only run missing jobs
                        
        print(f"{len(todo)} runs on {args.jobs} processes", flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_one, d, m, s, args.fixed, out_dir) for d, m, s in todo]  # run jobs in parallel
            for fut in as_completed(futures):
                print(fut.result(), flush=True)

    summary(out_dir)
