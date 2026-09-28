"""Per-item Dirichlet backoff on the GIVEN block.

For every PREDICT item the answer distribution of a held-out respondent is
estimated from the visible rows that share some of its GIVEN answers. The
strata are nested: level d conditions on the first d features of an item's
feature list, and each level is smoothed toward the level above it,

    p_d(y | s) = (n_d(s, y) + a_d * p_{d-1}(y | s)) / (n_d(s) + a_d)

with level 0 the item's marginal plus half a count per option. The feature
list and the a_d are chosen greedily per item by the exact leave-one-out log
score on the visible rows, which is cheap for count models. NA_GATED is an
ordinary class here, so gate routing and its violations are learned from the
data rather than assumed from the gate rules.
"""

import numpy as np
import pandas as pd

EPS = 1e-3
ALPHAS = (1.0, 4.0, 16.0, 64.0)
MAX_DEPTH = 4
MIN_GAIN = 1e-4


def _codes(column):
    codes, _ = pd.factorize(column.astype(object), use_na_sentinel=False)
    return codes.astype(np.int64)


def _nest(parent, codes):
    """Stratum index of (parent stratum, feature code), compacted."""
    key = parent * (codes.max() + 1) + codes
    _, inverse = np.unique(key, return_inverse=True)
    return inverse.astype(np.int64)


def _loo_level(strata, y, k, alpha, prev):
    """Leave-one-out probability of each training row's own answer."""
    n_strata = strata.max() + 1
    cell = np.bincount(strata * k + y, minlength=n_strata * k)
    size = np.bincount(strata, minlength=n_strata)
    return (cell[strata * k + y] - 1 + alpha * prev) / (size[strata] - 1 + alpha)


def _log_score(p, k):
    return float(np.mean(np.log(EPS + (1 - k * EPS) * p)))


def _fit_item(features, train, y, k):
    """Greedy feature list and smoothing for one item, by LOO log score."""
    n = len(y)
    marginal = np.bincount(y, minlength=k)
    prev = (marginal[y] - 1 + 0.5) / (n - 1 + 0.5 * k)
    best = _log_score(prev, k)
    strata = np.zeros(len(features[0]), np.int64)
    chosen = []
    for _ in range(MAX_DEPTH):
        step = None
        for f, codes in enumerate(features):
            if f in [c for c, _ in chosen]:
                continue
            nested = _nest(strata, codes)
            for alpha in ALPHAS:
                p = _loo_level(nested[train], y, k, alpha, prev)
                s = _log_score(p, k)
                if step is None or s > step[0]:
                    step = (s, f, alpha, nested, p)
        if step is None or step[0] < best + MIN_GAIN:
            break
        best, f, alpha, strata, prev = step
        chosen.append((f, alpha))
    return chosen


def _predict_item(features, train, held, y, k, chosen):
    n = len(y)
    marginal = np.bincount(y, minlength=k).astype(float)
    p = np.tile((marginal + 0.5) / (n + 0.5 * k), (len(held), 1))
    strata = np.zeros(len(features[0]), np.int64)
    for f, alpha in chosen:
        strata = _nest(strata, features[f])
        n_strata = strata.max() + 1
        cell = np.bincount(strata[train] * k + y, minlength=n_strata * k)
        cell = cell.reshape(n_strata, k).astype(float)
        size = cell.sum(1)
        s = strata[held]
        p = (cell[s] + alpha * p) / (size[s] + alpha)[:, None]
    return p


def predict(frame, schema):
    records = schema["items"]
    items = [name for name, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [name for name in items if records[name]["class"] == "GIVEN"]
    targets = [name for name in items if records[name]["class"] == "PREDICT"]

    blank = frame[targets].isna().to_numpy()
    held = np.flatnonzero(blank.any(axis=1))
    train = np.flatnonzero(~blank.any(axis=1))
    features = [_codes(frame[g]) for g in given]

    out = {}
    for name in targets:
        r = records[name]
        options = list(r["values"]) + ([schema["gated_value"]] if r.get("gate") else [])
        k = len(options)
        index = {v: i for i, v in enumerate(options)}
        y = frame[name].iloc[train].map(index)
        known = y.notna().to_numpy()
        rows, y = train[known], y[known].to_numpy(np.int64)
        if len(y) < 2 or not features:
            counts = np.bincount(y, minlength=k) + 0.5
            out[name] = np.tile(counts / counts.sum(), (len(held), 1))
            continue
        chosen = _fit_item(features, rows, y, k)
        out[name] = _predict_item(features, rows, held, y, k, chosen)

    position = {row: i for i, row in enumerate(held)}
    values = frame[items].to_numpy(dtype=object)
    vectors = []
    for row in range(values.shape[0]):
        for column, name in enumerate(items):
            if pd.isna(values[row, column]):
                if name in out:
                    vectors.append(out[name][position[row]].tolist())
                else:
                    r = records[name]
                    k = len(r["values"]) + (1 if r.get("gate") else 0)
                    vectors.append([1.0] * k)
    return vectors
