""" Decision tree, random forest and knn written from scratch with numpy

The trees work on binned features, the same idea LightGBM and histogram gradient boosting use,
every feature is cut into at most 256 bins once and the split search then only has to look at bin
boundaries instead of every unique value """
import multiprocessing as mp

import numpy as np


class Binner:
    """ Maps each feature to at most max_bins bins using quantiles of the training data """

    def __init__(self, max_bins=256):

        self.max_bins = max_bins

    def fit(self, X):

        self.edges = []

        for j in range(X.shape[1]):

            col = X[:, j]
            uniq = np.unique(col)

            if len(uniq) <= self.max_bins:

                # few distinct values so every value gets its own bin
                # the edge sits halfway between two neighbouring values like a normal tree threshold
                edges = (uniq[:-1] + uniq[1:]) / 2

            else:

                # too many values so use quantiles as the bin edges
                qs = np.quantile(col, np.linspace(0, 1, self.max_bins + 1)[1:-1])
                edges = np.unique(qs)

            self.edges.append(edges)

        return self

    def transform(self, X):

        Xb = np.empty(X.shape, dtype=np.uint8)

        for j in range(X.shape[1]):

            # bin b holds the values with edges[b-1] < x <= edges[b]
            Xb[:, j] = np.searchsorted(self.edges[j], X[:, j], side="left")

        return Xb


def balanced_weights(y, n_classes, counts=None):
    """ Class weights n / (C * n_c) like the balanced option in the report, counts lets a bootstrap sample be used """

    per_class = np.bincount(y, weights=counts, minlength=n_classes)
    present = per_class > 0

    weights = np.zeros(n_classes)
    weights[present] = per_class.sum() / (present.sum() * per_class[present])

    return weights


class DecisionTree:
    """ CART classification tree on binned features

    Supports gini or entropy, sample and class weights, max_depth, min_samples_split, min_samples_leaf
    and a random subset of max_features candidate features at every split for use inside a forest """

    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, class_weight=None, max_bins=256, random_state=None):

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.max_bins = max_bins
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):

        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        self.binner = Binner(self.max_bins).fit(X)
        Xb = self.binner.transform(X)

        w = np.ones(len(y_idx)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if self.class_weight == "balanced":
            w = w * balanced_weights(y_idx, n_classes)[y_idx]

        self.grow(Xb, y_idx, w, n_classes, np.random.default_rng(self.random_state))

        return self

    def n_candidates(self, n_features):

        # how many features to look at in each split
        mf = self.max_features

        if mf is None:
            return n_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if isinstance(mf, float):
            return max(1, int(mf * n_features))

        return min(int(mf), n_features)

    def impurity(self, counts, totals):

        # counts has the class axis last and totals is the matching sum of weights
        # an empty side has total 0 so divide by 1 there and the impurity just comes out as 0
        safe = np.where(totals > 1e-9, totals, 1.0)
        p = np.clip(counts / safe[..., None], 0, 1)

        if self.criterion == "gini":
            return 1 - (p * p).sum(axis=-1)

        # entropy with 0 log 0 taken as 0
        logp = np.log2(np.where(p > 0, p, 1))
        return -(p * logp).sum(axis=-1)

    def grow(self, Xb, y, w, n_classes, rng):
        """ Builds the tree depth first with an explicit stack, Xb is already binned and y is 0 to C-1 """

        self.n_classes = n_classes
        n_features = Xb.shape[1]
        m = self.n_candidates(n_features)
        max_depth = np.inf if self.max_depth is None else self.max_depth

        feature, threshold, left, right, value = [], [], [], [], []
        importances = np.zeros(n_features)
        self.depth = 0

        # rows with zero weight are not in this tree at all, this is how bootstrap samples come in
        root = np.flatnonzero(w > 0)
        stack = [(root, 0, -1, False)]

        while stack:

            idx, depth, parent, is_left = stack.pop()
            node = len(feature)
            self.depth = max(self.depth, depth)

            # hook this node onto its parent
            if parent >= 0:
                if is_left:
                    left[parent] = node
                else:
                    right[parent] = node

            counts = np.bincount(y[idx], weights=w[idx], minlength=n_classes)
            total = counts.sum()
            node_imp = self.impurity(counts, np.asarray(total))

            feature.append(-1)
            threshold.append(0)
            left.append(-1)
            right.append(-1)
            value.append(counts / total)

            # stopping rules
            n = len(idx)
            if depth >= max_depth or n < self.min_samples_split or n < 2 * self.min_samples_leaf or node_imp <= 1e-12:
                continue

            split = self.find_split(Xb, y, w, idx, counts, total, node_imp, m, rng)
            if split is None:
                continue

            f, b, gain = split
            feature[node] = f
            threshold[node] = b
            importances[f] += gain

            go_left = Xb[idx, f] <= b

            # right first so the left child gets popped next
            stack.append((idx[~go_left], depth + 1, node, False))
            stack.append((idx[go_left], depth + 1, node, True))

        self.feature = np.array(feature, dtype=np.int32)
        self.threshold = np.array(threshold, dtype=np.int16)
        self.left = np.array(left, dtype=np.int32)
        self.right = np.array(right, dtype=np.int32)
        self.value = np.array(value, dtype=np.float32)

        # normalise so the importances of one tree add up to 1 like the mean decrease in impurity
        total_gain = importances.sum()
        self.feature_importances_ = importances / total_gain if total_gain > 0 else importances

        return self

    def find_split(self, Xb, y, w, idx, counts, total, node_imp, m, rng):
        """ Best split over a random subset of m features that are not constant in this node

        Like the usual CART implementations, constant features do not count towards the m candidates,
        picking m at random from the non constant ones gives the same result as drawing until m useful ones turn up """

        rows = Xb[idx]
        useful = np.flatnonzero(rows.min(axis=0) != rows.max(axis=0))
        if len(useful) == 0:
            return None

        feats = rng.choice(useful, min(m, len(useful)), replace=False)

        # small nodes are cheaper to sort than to histogram over all 256 bins
        if len(idx) < 256:
            gain, bins = self.split_sorted(Xb, y, w, idx, feats, counts, total, node_imp)
        else:
            gain, bins = self.split_hist(Xb, y, w, idx, feats, counts, total, node_imp)

        k = np.argmax(gain)
        if not gain.flat[k] > 1e-12:
            return None

        i, j = np.unravel_index(k, gain.shape)
        return int(feats[j]), int(bins[i, j]), float(gain[i, j])

    def split_hist(self, Xb, y, w, idx, feats, counts, total, node_imp):

        n = len(idx)
        k = len(feats)
        C = self.n_classes
        B = 256
        leaf = self.min_samples_leaf

        codes = Xb[idx[:, None], feats[None, :]].astype(np.int64)
        offset = codes + np.arange(k) * B

        # one bincount gives the weighted class histogram of every candidate feature at once
        key = offset * C + y[idx][:, None]
        hist = np.bincount(key.ravel(), weights=np.repeat(w[idx], k), minlength=k * B * C).reshape(k, B, C)
        cnt = np.bincount(offset.ravel(), minlength=k * B).reshape(k, B)

        # everything up to and including bin b goes left
        left_counts = np.cumsum(hist, axis=1)
        right_counts = np.clip(counts - left_counts, 0, None)
        n_left = np.cumsum(cnt, axis=1)
        n_right = n - n_left

        w_left = left_counts.sum(axis=2)
        w_right = total - w_left

        child = w_left * self.impurity(left_counts, w_left) + w_right * self.impurity(right_counts, w_right)
        gain = total * node_imp - child

        # only real boundaries between filled bins that leave enough rows on both sides
        valid = (cnt > 0) & (n_left >= leaf) & (n_right >= leaf)
        gain = np.where(valid, gain, -np.inf)

        bins = np.broadcast_to(np.arange(B)[None, :], (k, B))

        # transpose so rows are thresholds and columns are features like split_sorted
        return gain.T, bins.T

    def split_sorted(self, Xb, y, w, idx, feats, counts, total, node_imp):

        n = len(idx)
        k = len(feats)
        C = self.n_classes
        leaf = self.min_samples_leaf

        codes = Xb[idx[:, None], feats[None, :]]
        order = np.argsort(codes, axis=0, kind="stable")
        sorted_codes = np.take_along_axis(codes, order, axis=0)

        ys = y[idx][order]
        ws = w[idx][order]

        # spread the weights out by class then cumsum down the sorted rows
        onehot = np.zeros((n, k, C))
        onehot[np.arange(n)[:, None], np.arange(k)[None, :], ys] = ws
        left_counts = np.cumsum(onehot, axis=0)[:-1]
        right_counts = np.clip(counts - left_counts, 0, None)

        w_left = left_counts.sum(axis=2)
        w_right = total - w_left

        child = w_left * self.impurity(left_counts, w_left) + w_right * self.impurity(right_counts, w_right)
        gain = total * node_imp - child

        # a split after row i is only real if the next row has a different bin
        n_left = np.arange(1, n)[:, None]
        valid = (sorted_codes[:-1] != sorted_codes[1:]) & (n_left >= leaf) & (n - n_left >= leaf)
        gain = np.where(valid, gain, -np.inf)

        return gain, sorted_codes[:-1]

    def predict_binned(self, Xb):

        # move every row down one level per loop until all rows sit in a leaf
        node = np.zeros(len(Xb), dtype=np.int64)

        while True:

            f = self.feature[node]
            rows = np.flatnonzero(f >= 0)
            if len(rows) == 0:
                break

            current = node[rows]
            go_left = Xb[rows, f[rows]] <= self.threshold[current]
            node[rows] = np.where(go_left, self.left[current], self.right[current])

        return self.value[node]

    def predict_proba(self, X):

        return self.predict_binned(self.binner.transform(np.asarray(X, dtype=float)))

    def predict(self, X):

        return self.classes_[self.predict_proba(X).argmax(axis=1)]


# data the forest workers need, set right before the pool forks so the children share it for free
_forest_data = {}


def _grow_one_tree(seed):

    d = _forest_data
    rng = np.random.default_rng(seed)
    y = d["y"]
    n = len(y)

    if d["bootstrap"]:

        # a bootstrap sample is just how many times each row got drawn, used as a weight
        draws = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)

    else:

        draws = np.ones(n)

    if d["class_weight"] == "balanced_subsample":

        # weights worked out from the bootstrap sample instead of the full training set
        w = draws * balanced_weights(y, d["n_classes"], draws)[y]

    else:

        w = draws * d["base_weight"]

    tree = DecisionTree(**d["tree_params"])
    tree.grow(d["Xb"], y, w, d["n_classes"], rng)

    oob = None
    if d["oob_score"]:

        # rows this tree never saw
        oob_rows = np.flatnonzero(draws == 0)
        oob = (oob_rows, tree.predict_binned(d["Xb"][oob_rows]))

    return tree, oob


class RandomForest:
    """ Random forest of DecisionTree objects

    Bootstrap samples are passed to the trees as integer sample weights so no data gets copied,
    trees are grown in parallel processes and every tree gets its own seed from one SeedSequence,
    so the forest is the same no matter how many processes are used """

    def __init__(self, n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt", class_weight=None, bootstrap=True,
                 oob_score=False, max_bins=256, n_jobs=1, random_state=None):

        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.max_bins = max_bins
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):

        global _forest_data

        X = np.asarray(X, dtype=float)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        # bin once and share the binned data with every tree
        self.binner = Binner(self.max_bins).fit(X)
        Xb = self.binner.transform(X)

        base_weight = np.ones(len(y_idx))
        if self.class_weight == "balanced":
            base_weight = balanced_weights(y_idx, n_classes)[y_idx]

        tree_params = {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "max_bins": self.max_bins
        }

        _forest_data = {
            "Xb": Xb, "y": y_idx, "n_classes": n_classes, "base_weight": base_weight,
            "bootstrap": self.bootstrap, "class_weight": self.class_weight,
            "oob_score": self.oob_score, "tree_params": tree_params
        }

        seeds = np.random.SeedSequence(self.random_state).generate_state(self.n_estimators)

        if self.n_jobs > 1:

            with mp.get_context("fork").Pool(self.n_jobs) as pool:
                results = pool.map(_grow_one_tree, seeds, chunksize=max(1, self.n_estimators // (4 * self.n_jobs)))

        else:

            results = [_grow_one_tree(s) for s in seeds]

        _forest_data = {}

        self.estimators_ = []
        for tree, _ in results:

            # trees share the forest binner and class order
            tree.binner = self.binner
            tree.classes_ = self.classes_
            self.estimators_.append(tree)

        self.feature_importances_ = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)

        if self.oob_score:

            # add up the votes each row got from the trees that did not train on it
            votes = np.zeros((len(y_idx), n_classes))
            for _, (rows, proba) in results:
                votes[rows] += proba

            has_vote = votes.sum(axis=1) > 0
            self.oob_decision_ = votes
            self.oob_score_ = (votes[has_vote].argmax(axis=1) == y_idx[has_vote]).mean()

        return self

    def tree_probas(self, X):
        """ Class probabilities of every tree one at a time, used for the ensemble size and diversity experiments """

        Xb = self.binner.transform(np.asarray(X, dtype=float))
        for tree in self.estimators_:
            yield tree.predict_binned(Xb)

    def predict_proba(self, X):

        # average of the leaf class distributions of all trees
        total = None
        for proba in self.tree_probas(X):
            total = proba.astype(float) if total is None else total + proba

        return total / len(self.estimators_)

    def predict(self, X):

        return self.classes_[self.predict_proba(X).argmax(axis=1)]


_knn_data = {}


def _knn_chunk(bounds):

    d = _knn_data
    start, stop = bounds
    return d["model"].votes(d["X"][start:stop])


class KNN:
    """ Brute force k nearest neighbours with euclidean, manhattan, chebyshev or minkowski distance

    Queries are handled in chunks so the distance matrix never gets too big, and the chunks can be
    spread over several processes """

    def __init__(self, n_neighbors=5, weights="uniform", metric="euclidean", p=2, chunk_size=256, n_jobs=1):

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs

    def fit(self, X, y):

        self.X = np.asarray(X, dtype=float)
        self.classes_, self.y = np.unique(y, return_inverse=True)

        return self

    def distances(self, Q):

        X = self.X

        if self.metric == "euclidean":

            # |q - x|^2 = |q|^2 + |x|^2 - 2 q.x, clipped because rounding can go slightly negative
            sq = (Q * Q).sum(axis=1)[:, None] + (X * X).sum(axis=1)[None, :] - 2 * Q @ X.T
            return np.sqrt(np.clip(sq, 0, None))

        # the rest are built up one feature at a time so only a chunk x train matrix is ever in memory
        D = np.zeros((len(Q), len(X)))

        for j in range(X.shape[1]):

            diff = np.abs(Q[:, j][:, None] - X[:, j][None, :])

            if self.metric == "manhattan":
                D += diff
            elif self.metric == "chebyshev":
                np.maximum(D, diff, out=D)
            else:
                D += diff ** self.p

        if self.metric == "minkowski":
            D = D ** (1 / self.p)

        return D

    def votes(self, Q):

        k = self.n_neighbors
        D = self.distances(Q)

        # the k smallest distances in each row, order inside the k does not matter for voting
        nn = np.argpartition(D, k - 1, axis=1)[:, :k]
        dist = np.take_along_axis(D, nn, axis=1)
        labels = self.y[nn]

        if self.weights == "distance":

            with np.errstate(divide="ignore"):
                w = 1 / dist

            # an exact match gets all the say, same as the usual convention
            exact = dist == 0
            has_exact = exact.any(axis=1)
            w[has_exact] = exact[has_exact].astype(float)

        else:

            w = np.ones_like(dist)

        votes = np.zeros((len(Q), len(self.classes_)))
        for c in range(len(self.classes_)):
            votes[:, c] = (w * (labels == c)).sum(axis=1)

        return votes

    def predict_proba(self, X):

        global _knn_data

        Q = np.asarray(X, dtype=float)
        bounds = [(s, min(s + self.chunk_size, len(Q))) for s in range(0, len(Q), self.chunk_size)]

        if self.n_jobs > 1:

            _knn_data = {"model": self, "X": Q}
            with mp.get_context("fork").Pool(self.n_jobs) as pool:
                parts = pool.map(_knn_chunk, bounds)
            _knn_data = {}

        else:

            parts = [self.votes(Q[s:e]) for s, e in bounds]

        votes = np.vstack(parts)
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X):

        return self.classes_[self.predict_proba(X).argmax(axis=1)]
