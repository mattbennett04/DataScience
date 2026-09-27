# preprocessing for the five m4 series in data/
#
# same pipeline for every dataset only the settings in CONFIGS change
# this was done bc under drift you shouldnt normailse scale min max feature selection ect
# on the whole series so the scaler only ever sees the training part of a fold
#
# order log -> seasonal diff -> first diff -> scale -> sliding windows
# differencing leaves nans at the start those windows just get skipped

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# breaks = known regime changes only used to print which fold they land in
CONFIGS = {
    "H279": {"log": False, "seasonal_lag": 0, "diff": False, "window": 24, "breaks": []},
    "W8": {"log": True, "seasonal_lag": 0, "diff": False, "window": 52, "breaks": []},
    "D686": {"log": True, "seasonal_lag": 0, "diff": True, "window": 14, "breaks": [3626]},
    "M6864": {"log": True, "seasonal_lag": 0, "diff": True, "window": 12, "breaks": [2430]},
    # robust scaling bc the jumps at 303 and 606 are huge single differences see preprocessor.fit
    "M26487": {"log": True, "seasonal_lag": 0, "diff": True, "window": 12, "breaks": [303, 606],
               "scaling": "robust", "clip": 3},
}


def load_series(name):
    df = pd.read_csv(os.path.join(DATA_DIR, name + ".csv"))
    return df["value"].to_numpy(float)


# expanding window folds train always starts at 0 and gets bigger each fold
# val and test come straight after train returns start end pairs end not included
def make_folds(n, n_folds=5, min_train=0.4, val_frac=0.1, test_frac=0.1):
    val_len = int(val_frac * n)
    test_len = int(test_frac * n)

    first = int(min_train * n)
    last = n - val_len - test_len
    train_ends = np.linspace(first, last, n_folds).astype(int)

    folds = []
    for e in train_ends:
        e = int(e)
        train = (0, e)
        val = (e, e + val_len)
        test = (e + val_len, e + val_len + test_len)
        folds.append((train, val, test))
    return folds


# x[t] - x[t-lag] first lag values are nan
def lag_diff(x, lag):
    out = np.full_like(x, np.nan)
    out[lag:] = x[lag:] - x[:-lag]
    return out


class Preprocessor:

    def __init__(self, log=False, seasonal_lag=0, diff=False, scaling="minmax", clip=None,
                 window=None, breaks=None):
        # window and breaks arent used here just means CONFIGS can be passed in directly
        self.log = log
        self.seasonal_lag = seasonal_lag
        self.diff = diff
        self.scaling = scaling
        self.clip = clip

    # returns the series after each step logged seasonal diffed diffed
    # need the earlier ones to undo the differencing later
    def stages(self, x):
        if self.log:
            logged = np.log(x)
        else:
            logged = x.copy()

        seasonal = logged
        if self.seasonal_lag:
            seasonal = lag_diff(logged, self.seasonal_lag)

        diffed = seasonal
        if self.diff:
            diffed = lag_diff(seasonal, 1)

        return logged, seasonal, diffed

    # only fit on the training part [start train_end start is only used when retraining on recent data
    def fit(self, x, train_end, start=0):
        vals = self.stages(x)[2][start:train_end]
        vals = vals[~np.isnan(vals)]  # differencing leaves nans at the start
        if self.scaling == "minmax":
            self.lo = vals.min()
            self.hi = vals.max()
        elif self.scaling == "robust":
            # 1st-99th percentile -> [-1 1] so one or two huge jumps in the training part
            # dont squash all the normal values into a tiny range the jumps just end up outside [-1 1]
            self.lo, self.hi = np.percentile(vals, [1, 99])
        else:
            self.mu = vals.mean()
            self.sd = vals.std()
        return self

    # minmax scales to [-1 1] tanh range
    def scale(self, v):
        if self.scaling in ("minmax", "robust"):
            return 2 * (v - self.lo) / (self.hi - self.lo) - 1
        return (v - self.mu) / self.sd

    def unscale(self, z):
        if self.scaling in ("minmax", "robust"):
            return (z + 1) / 2 * (self.hi - self.lo) + self.lo
        return z * self.sd + self.mu

    # clip = winsorise what the network sees so a single jump cant dominate the mse loss
    # scores are still worked out against the real values see inverse_one_step
    def transform(self, x):
        z = self.scale(self.stages(x)[2])
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z

    # turn scaled predictions for times t back into the original units
    # uses the real past values of x since its one step ahead
    def inverse_one_step(self, z_pred, t, x):
        t = np.asarray(t)
        logged, seasonal, _ = self.stages(x)
        pred = self.unscale(np.asarray(z_pred, float))
        if self.diff:
            pred = pred + seasonal[t - 1]  # add the change back onto the last real value
        if self.seasonal_lag:
            pred = pred + logged[t - self.seasonal_lag]
        if self.log:
            pred = np.exp(pred)  # undo the log
        return pred


# inputs = previous window values target = z[t] for t in [start end
# inputs can go back before start thats fine because they are past values
def make_windows(z, window, start, end):
    X = []
    y = []
    ts = []
    for t in range(max(start, window), end):
        inputs = z[t - window:t]
        if np.isnan(inputs).any() or np.isnan(z[t]):
            continue  # skip windows that touch the nans from differencing
        X.append(inputs)
        y.append(z[t])
        ts.append(t)
    return np.array(X), np.array(y), np.array(ts)


def prepare_fold(name, fold, window=None, scaling="minmax"):
    cfg = CONFIGS[name]
    x = load_series(name)
    if window is None:
        window = cfg["window"]

    train, val, test = fold
    # a dataset can set its own scaling in CONFIGS otherwise use the argument
    settings = dict(cfg)
    settings.setdefault("scaling", scaling)
    pp = Preprocessor(**settings)
    pp.fit(x, train[1])
    z = pp.transform(x)

    return {
        "x": x,
        "z": z,
        "pp": pp,
        "train": make_windows(z, window, 0, train[1]),
        "val": make_windows(z, window, val[0], val[1]),
        "test": make_windows(z, window, test[0], test[1]),
    }


if __name__ == "__main__":
    for name, cfg in CONFIGS.items():
        x = load_series(name)
        folds = make_folds(len(x))

        settings = {k: v for k, v in cfg.items() if k != "breaks"}
        print(f"\n{name}: n={len(x)} config={settings}")

        for i, fold in enumerate(folds):
            d = prepare_fold(name, fold)
            Xtr, ytr, _ = d["train"]
            Xte, yte, tte = d["test"]

            # sanity check - inverting the true targets should give back x
            # skip clipped targets those cant come back exactly
            ok = np.abs(yte) < (d["pp"].clip or np.inf)
            err = np.abs(d["pp"].inverse_one_step(yte[ok], tte[ok], x) - x[tte[ok]]).max()

            # which part of the fold each break falls in
            where = []
            for b in cfg["breaks"]:
                for part, (s, e) in zip(["train", "val", "test"], fold):
                    if s <= b < e:
                        where.append(f"{b}->{part}")

            print(f"  fold {i}: train={fold[0]} val={fold[1]} test={fold[2]} "
                  f"Xtr={Xtr.shape} Xte={Xte.shape} "
                  f"test z-range=[{yte.min():.2f}, {yte.max():.2f}] "
                  f"roundtrip_err={err:.1e} breaks={where}")
