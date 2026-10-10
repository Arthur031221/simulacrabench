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

A second member, an L2 multinomial logistic regression on the one-hot GIVEN
block and country/wave interactions, is mixed in per item. Both members are
refitted on the same three training complements to choose the mixture by
held-out log score. Complete backoff predictions are computed first. Each
optional item runs in a separate process with a time limit and keeps its
backoff prediction if the process fails or runs late.
"""

import multiprocessing
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from threadpoolctl import threadpool_limits

EPS = 1e-3
ALPHAS = (1.0, 4.0, 16.0, 64.0)
MAX_DEPTH = 4
MIN_GAIN = 1e-4
FOLDS = 3
LR_C = 1.0
WEIGHTS = np.linspace(0.0, 1.0, 11)
BUDGET_SECONDS = 180.0
ITEM_SECONDS = 20.0
PAIR_COLUMNS = 256
INTERACTION_COLUMNS = 1024
COUNTRY_COLUMNS = ("Country", "user.country", "s1_2_country_asylum_lbl")


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
    return chosen, prev


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


def _one_hot(features):
    blocks = []
    for codes in features:
        width = codes.max() + 1
        if width > 1:
            rows = np.flatnonzero(codes)
            blocks.append(sparse.csr_matrix(
                (np.ones(len(rows)), (rows, codes[rows] - 1)),
                shape=(len(codes), width - 1),
            ))
    return sparse.hstack(blocks, format="csr") if blocks else None


def _lr_proba(x_fit, y_fit, x_out, k):
    out = np.zeros((x_out.shape[0], k))
    classes = np.unique(y_fit)
    if len(classes) == 1:
        out[:, classes[0]] = 1.0
        return out
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(
            C=LR_C, solver="lbfgs", max_iter=200, tol=1e-4,
        ).fit(x_fit, y_fit)
        out[:, model.classes_] = model.predict_proba(x_out)
    return out


def _interaction_pairs(given, records):
    """Choose bounded pairs from declared GIVEN supports, without labels."""
    anchors = [i for i, name in enumerate(given)
               if name in COUNTRY_COLUMNS or name == "wave"]
    pairs, seen, columns = [], set(), 0
    for anchor in anchors:
        for other in range(len(given)):
            pair = tuple(sorted((anchor, other)))
            if anchor == other or pair in seen:
                continue
            seen.add(pair)
            widths = [len(records[given[i]]["values"])
                      + bool(records[given[i]].get("gate")) for i in pair]
            width = widths[0] * widths[1]
            if width > PAIR_COLUMNS or columns + width > INTERACTION_COLUMNS:
                continue
            pairs.append((pair[0], pair[1], width))
            columns += width
    return pairs


def _design(features, given, records):
    blocks = []
    additive = _one_hot(features)
    if additive is not None:
        blocks.append(additive)
    for first, second, bound in _interaction_pairs(given, records):
        joint = _nest(features[first], features[second])
        width = int(joint.max()) + 1
        if width > bound:
            continue
        blocks.append(sparse.csr_matrix(
            (np.ones(len(joint)), (np.arange(len(joint)), joint)),
            shape=(len(joint), width),
        ))
    return sparse.hstack(blocks, format="csr") if blocks else None


def _oof_members(features, rows, y, k, x, fold, deadline):
    backoff, regression = np.empty(len(y)), np.empty(len(y))
    for f in range(FOLDS):
        excluded = fold == f
        if not excluded.any():
            continue
        if (~excluded).sum() < 2 or time.monotonic() >= deadline:
            return None
        fit_rows, fit_y = rows[~excluded], y[~excluded]
        chosen, _ = _fit_item(features, fit_rows, fit_y, k)
        p = _predict_item(features, fit_rows, rows[excluded], fit_y, k, chosen)
        backoff[excluded] = p[np.arange(excluded.sum()), y[excluded]]
        p = _lr_proba(x[fit_rows], fit_y, x[rows[excluded]], k)
        regression[excluded] = p[np.arange(excluded.sum()), y[excluded]]
    return (backoff, regression) if time.monotonic() < deadline else None


def _stack_worker(sender, features, rows, held, y, k, x, fold, deadline):
    try:
        result = None
        own = _oof_members(features, rows, y, k, x, fold, deadline)
        if own is not None:
            weight = _best_weight(*own, k)
            if weight < 1.0 and time.monotonic() < deadline:
                p = _lr_proba(x[rows], y, x[held], k)
                if time.monotonic() < deadline:
                    result = (weight, p)
        sender.send(result)
    except Exception:
        pass
    finally:
        sender.close()


def _bounded_stack(features, rows, held, y, k, x, fold, deadline):
    """Return a complete optional member, cleaning up its process on exit."""
    if time.monotonic() >= deadline:
        return None
    receiver = sender = process = None
    started = False
    item_start = time.monotonic()
    item_deadline = min(deadline, item_start + ITEM_SECONDS)
    try:
        context = multiprocessing.get_context("fork")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_stack_worker,
            args=(sender, features, rows, held, y, k, x, fold, item_deadline),
        )
        process.start()
        started = True
        sender.close()
        remaining = item_deadline - time.monotonic()
        if remaining > 0 and receiver.poll(remaining):
            result = receiver.recv()
            if time.monotonic() < item_deadline:
                return result
    except (OSError, EOFError, RuntimeError, ValueError):
        pass
    finally:
        if receiver is not None:
            receiver.close()
        if sender is not None:
            sender.close()
        if started:
            process.join(timeout=0.05)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
    return None


def _best_weight(backoff, lr, k):
    """Weight on the backoff member that maximises the mixture's log score."""
    scores = [_log_score(w * backoff + (1 - w) * lr, k) for w in WEIGHTS]
    return float(WEIGHTS[int(np.argmax(scores))])


def predict(frame, schema):
    deadline = time.monotonic() + BUDGET_SECONDS
    records = schema["items"]
    items = [name for name, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [name for name in items if records[name]["class"] == "GIVEN"]
    targets = [name for name in items if records[name]["class"] == "PREDICT"]

    blank = frame[targets].isna().to_numpy()
    held = np.flatnonzero(blank.any(axis=1))
    train = np.flatnonzero(~blank.any(axis=1))
    features = [_codes(frame[g]) for g in given]
    fold = np.random.default_rng(0).permutation(len(train)) % FOLDS

    out, training = {}, {}
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
        chosen, _ = _fit_item(features, rows, y, k)
        p = _predict_item(features, rows, held, y, k, chosen)
        out[name] = p
        training[name] = (rows, y, k, fold[known])

    if features and time.monotonic() < deadline:
        x = _design(features, given, records)
        if x is not None:
            for name, (rows, y, k, item_fold) in training.items():
                if time.monotonic() >= deadline:
                    break
                if len(y) < 2 * FOLDS:
                    continue
                member = _bounded_stack(features, rows, held, y, k, x, item_fold, deadline)
                if member is not None and time.monotonic() < deadline:
                    weight, regression = member
                    out[name] = weight * out[name] + (1 - weight) * regression

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
