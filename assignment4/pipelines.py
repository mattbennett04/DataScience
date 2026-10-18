""" The three models with their own preprocessing, data loading and the cv folds used everywhere """
import os

import numpy as np

from preprocessing import load_clean_data, get_X_y
from classifiers import DecisionTree, RandomForest, KNN
from ml_utils import OneHotEncoder, StandardScaler, stratified_sample, undersample_to_median

# never use more than 16 cores
N_JOBS = 16

# pin this process to 16 cores, the worker processes inherit it so the whole run can never go over
os.sched_setaffinity(0, range(min(N_JOBS, os.cpu_count())))

class_names = ["Normal", "Reconnaissance", "Backdoor", "DoS", "Exploits",
               "Analysis", "Fuzzers", "Worms", "Shellcode", "Generic"]


def load_data():

    # load the cleaned data the same way as assignment 2
    df_clean = load_clean_data("../assignment2/networkTraffic.csv")
    X, y, cat, num = get_X_y(df_clean)

    return X.reset_index(drop=True), y.to_numpy(), cat, num


def tuning_sample(X, y):

    # 50k stratified sample for all the tuning work
    rows = stratified_sample(y, 50000, seed=42)
    return X.iloc[rows].reset_index(drop=True), y[rows]


class ForestModel:
    """ One hot encoding plus random forest, proto_min_frequency groups the rare proto levels """

    def __init__(self, cat, num, proto_min_frequency=None, **rf_params):

        self.cat = cat
        self.num = num
        self.proto_min_frequency = proto_min_frequency
        self.rf_params = {"n_estimators": 100, "random_state": 42, "n_jobs": N_JOBS}
        self.rf_params.update(rf_params)

    def encode(self, X):

        parts = []
        for col in self.cat:
            parts.append(self.encoders[col].transform(X[col]))

        # numbers go in as they are, trees do not care about scale
        parts.append(X[self.num].to_numpy(dtype=float))

        return np.hstack(parts)

    def fit(self, X, y):

        self.encoders = {}
        for col in self.cat:

            min_freq = self.proto_min_frequency if col == "proto" else None
            self.encoders[col] = OneHotEncoder(min_frequency=min_freq).fit(X[col])

        self.feature_names = []
        for col in self.cat:
            self.feature_names += self.encoders[col].names(col)
        self.feature_names += list(self.num)

        self.forest = RandomForest(**self.rf_params).fit(self.encode(X), y)

        return self

    def predict(self, X):

        return self.forest.predict(self.encode(X))

    def tree_probas(self, X):

        return self.forest.tree_probas(self.encode(X))


class TreeModel(ForestModel):
    """ Final decision tree from assignment 2 with every level of every categorical column one hot encoded """

    def __init__(self, cat, num, **tree_params):

        super().__init__(cat, num)
        self.tree_params = {"criterion": "entropy", "max_depth": 15, "min_samples_leaf": 2,
                            "min_samples_split": 2, "class_weight": "balanced", "random_state": 42}
        self.tree_params.update(tree_params)

    def fit(self, X, y):

        self.encoders = {}
        self.feature_names = []
        for col in self.cat:

            self.encoders[col] = OneHotEncoder().fit(X[col])
            self.feature_names += self.encoders[col].names(col)

        self.feature_names += list(self.num)

        self.tree = DecisionTree(**self.tree_params).fit(self.encode(X), y)
        self.forest = self.tree

        return self


class KNNModel:
    """ Final knn from assignment 2, grouped proto, log of skewed columns, scaling, undersampling to the
    median class size and the 6 features a depth 8 tree finds most important """

    def __init__(self, X_full, cat, num):

        self.cat = cat
        self.num = num

        # skewed columns picked on the full data the same way assignment 2 did it
        skew = X_full[num].skew()
        nunique = X_full[num].nunique()
        self.log_cols = skew[(skew > 2) & (nunique[skew.index] > 2)].index.tolist()

    def encode(self, X):

        parts = []
        for col in self.cat:
            parts.append(self.encoders[col].transform(X[col]))

        numbers = X[self.num].to_numpy(dtype=float).copy()
        for j, col in enumerate(self.num):

            if col in self.log_cols:
                numbers[:, j] = np.log1p(numbers[:, j])

        parts.append(self.scaler.transform(numbers))

        return np.hstack(parts)

    def fit(self, X, y):

        self.encoders = {}
        for col in self.cat:

            min_freq = 500 if col == "proto" else None
            self.encoders[col] = OneHotEncoder(min_frequency=min_freq).fit(X[col])

        # scaler is fitted after the log so it sees the same values as later
        numbers = X[self.num].to_numpy(dtype=float).copy()
        for j, col in enumerate(self.num):

            if col in self.log_cols:
                numbers[:, j] = np.log1p(numbers[:, j])

        self.scaler = StandardScaler().fit(numbers)

        Z = self.encode(X)

        # undersample first then pick features on the undersampled data, same order as the assignment 2 pipeline
        keep = undersample_to_median(y, seed=42)
        Z, y = Z[keep], y[keep]

        selector = DecisionTree(max_depth=8, random_state=42).fit(Z, y)
        self.selected = np.argsort(selector.feature_importances_)[::-1][:6]

        self.knn = KNN(n_neighbors=109, weights="distance", metric="manhattan", n_jobs=N_JOBS).fit(Z[:, self.selected], y)

        return self

    def predict(self, X):

        return self.knn.predict(self.encode(X)[:, self.selected])
