""" Encoding, scaling, sampling, cross validation splits and metrics, all written by hand """
import numpy as np
import pandas as pd


class OneHotEncoder:
    """ One hot encoding of a single column

    With min_frequency set, levels seen fewer times than that in the training data share one infrequent
    column, and levels never seen in training also go to that column if it exists, otherwise all zeros """

    def __init__(self, min_frequency=None):

        self.min_frequency = min_frequency

    def fit(self, col):

        counts = pd.Series(col).astype(str).value_counts()

        if self.min_frequency is None:

            self.levels = sorted(counts.index)
            self.has_infrequent = False

        else:

            self.levels = sorted(counts.index[counts >= self.min_frequency])
            self.has_infrequent = (counts < self.min_frequency).any()

        self.lookup = {level: i for i, level in enumerate(self.levels)}

        return self

    def transform(self, col):

        values = pd.Series(col).astype(str).to_numpy()
        n_cols = len(self.levels) + int(self.has_infrequent)
        out = np.zeros((len(values), n_cols))

        # anything not in the lookup goes to the infrequent column which is the last one
        other = len(self.levels) if self.has_infrequent else -1
        pos = np.array([self.lookup.get(v, other) for v in values])

        known = pos >= 0
        out[np.flatnonzero(known), pos[known]] = 1

        return out

    def names(self, prefix):

        names = [f"{prefix}_{level}" for level in self.levels]
        if self.has_infrequent:
            names.append(f"{prefix}_infrequent")

        return names


class StandardScaler:

    def fit(self, X):

        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

        # constant columns would divide by zero so leave them alone
        self.std[self.std == 0] = 1

        return self

    def transform(self, X):

        return (X - self.mean) / self.std


def stratified_kfold(y, n_splits=5, seed=42):
    """ List of (train, val) index arrays where every fold has close to the same class proportions """

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(y), dtype=int)

    # deal each class out over the folds separately after shuffling it
    for c in np.unique(y):

        rows = rng.permutation(np.flatnonzero(y == c))
        for f, part in enumerate(np.array_split(rows, n_splits)):
            fold_of[part] = f

    folds = []
    for f in range(n_splits):
        folds.append((np.flatnonzero(fold_of != f), np.flatnonzero(fold_of == f)))

    return folds


def stratified_sample(y, n, seed=42):
    """ Indices of a sample of size n with the same class proportions as y """

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)

    # largest remainder rounding so the sizes add up to exactly n
    exact = counts * n / len(y)
    sizes = np.floor(exact).astype(int)
    short = n - sizes.sum()
    sizes[np.argsort(exact - sizes)[::-1][:short]] += 1

    picked = []
    for c, size in zip(classes, sizes):
        picked.append(rng.choice(np.flatnonzero(y == c), size, replace=False))

    return np.sort(np.concatenate(picked))


def undersample_to_median(y, seed=42):
    """ Indices that cut every class bigger than the median class size down to the median """

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    cap = int(np.median(counts))

    keep = []
    for c, count in zip(classes, counts):

        rows = np.flatnonzero(y == c)
        if count > cap:
            rows = rng.choice(rows, cap, replace=False)
        keep.append(rows)

    return np.sort(np.concatenate(keep))


def confusion_matrix(y_true, y_pred, n_classes=10):

    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)

    return cm


def per_class_scores(y_true, y_pred, n_classes=10):

    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)

    # 0 when a class is never predicted or never present instead of dividing by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.nan_to_num(tp / cm.sum(axis=0))
        recall = np.nan_to_num(tp / cm.sum(axis=1))
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))

    return precision, recall, f1


def f1_macro(y_true, y_pred, n_classes=10):

    return per_class_scores(y_true, y_pred, n_classes)[2].mean()


def balanced_accuracy(y_true, y_pred, n_classes=10):

    # only average over classes that are actually in y_true
    present = np.bincount(np.asarray(y_true), minlength=n_classes) > 0
    return per_class_scores(y_true, y_pred, n_classes)[1][present].mean()
