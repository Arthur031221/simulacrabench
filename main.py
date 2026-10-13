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

An L2 multinomial logistic regression on the one-hot GIVEN block and
country/wave interactions is selected against backoff on matched folds. A
schema-categorical shallow HistGradientBoosting member can then join that
blend, selected on the same folds with fold-local marginal smoothing and a
temperature grid. Complete backoff predictions are computed first. Optional
item workers run serially with time limits. Incomplete or late work preserves
the last complete blend.
"""

import multiprocessing
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
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


def _instrument_budget(records):
    """Reserve more optional fitting time for the larger UNHCR item set."""
    for name, seconds in (("wave", 450.0), ("group", 180.0), ("Country", 90.0)):
        if records.get(name, {}).get("class") == "GIVEN":
            return seconds
    return 240.0


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
    """Complete matched-fold true-class predictions for backoff and LR."""
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


def _tree_categories(frame, given, records, gated_value):
    """Use schema-ordered categories. Unknown and missing values become NaN."""
    columns = {}
    for name in given:
        values = list(records[name]["values"])
        if records[name].get("gate"):
            values.append(gated_value)
        # Category dtype retains the full declared ordering even when a fold
        # has not observed a category. Values outside it are converted to NA.
        columns[name] = pd.Categorical(frame[name], categories=values, ordered=True)
    return pd.DataFrame(columns, index=frame.index)


def _tree_proba(x_fit, y_fit, x_out, k):
    """Fit one shallow categorical HGB and align its classes to schema order."""
    counts = np.bincount(y_fit, minlength=k).astype(float)
    marginal = (counts + 0.5) / (len(y_fit) + 0.5 * k)
    if len(np.unique(y_fit)) < 2:
        raw = np.tile(marginal, (len(x_out), 1))
    else:
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = HistGradientBoostingClassifier(
                loss="log_loss", max_iter=60, learning_rate=0.08,
                max_depth=3, max_leaf_nodes=7, min_samples_leaf=40,
                l2_regularization=5, max_bins=255, early_stopping=False,
                random_state=0, categorical_features="from_dtype",
            ).fit(x_fit, y_fit)
            raw = np.zeros((len(x_out), k), dtype=float)
            raw[:, model.classes_] = model.predict_proba(x_out)
    mixed = 0.99 * raw + 0.01 * marginal[None, :]
    return mixed / mixed.sum(axis=1, keepdims=True)


def _temperature(p, temperature):
    q = np.maximum(p, 1e-15) ** (1.0 / temperature)
    return q / q.sum(axis=1, keepdims=True)


def _epsilon_true_score(p, y, k):
    return float(np.mean(np.log(EPS + (1 - k * EPS) * p[np.arange(len(y)), y])))


def _select_tree(backoff_true, lr_true, tree_oof, y, k):
    """Grid search simplex weights and temperature with deterministic ties."""
    best = None
    for tree_weight in np.arange(0.0, 0.50001, 0.1):
        for temperature in (0.75, 1.0, 1.5, 2.0):
            calibrated = _temperature(tree_oof, temperature)
            for backoff_weight in np.arange(0.0, 1.00001 - tree_weight, 0.1):
                lr_weight = 1.0 - tree_weight - backoff_weight
                truth = (backoff_weight * backoff_true + lr_weight * lr_true
                         + tree_weight * calibrated[np.arange(len(y)), y])
                score = float(np.mean(np.log(EPS + (1 - k * EPS) * truth)))
                # Prefer zero tree weight, then more backoff. Stable temperature
                # order resolves any remaining exact ties.
                key = (score, -tree_weight, backoff_weight, -temperature)
                if best is None or key > best[0]:
                    best = (key, backoff_weight, lr_weight, tree_weight,
                            temperature)
    return best[1:]


def _tree_oof(x, rows, y, k, fold, deadline):
    pred = np.empty((len(y), k), dtype=float)
    for f in range(FOLDS):
        excluded = fold == f
        if not excluded.any():
            continue
        if (~excluded).sum() < 2 or time.monotonic() >= deadline:
            return None
        pred[excluded] = _tree_proba(x.iloc[rows[~excluded]], y[~excluded],
                                     x.iloc[rows[excluded]], k)
    return pred if time.monotonic() < deadline else None


def _lr_worker(sender, features, rows, held, y, k, x, fold, deadline):
    try:
        result = None
        oof = _oof_members(features, rows, y, k, x, fold, deadline)
        if oof is not None:
            backoff_true, lr_true = oof
            weight = _best_weight(backoff_true, lr_true, k)
            lr_full = None
            if weight < 1.0 and time.monotonic() < deadline:
                lr_full = _lr_proba(x[rows], y, x[held], k)
                if time.monotonic() >= deadline:
                    lr_full = None
            result = (backoff_true, lr_true, weight, lr_full)
        sender.send(result)
    except Exception:
        pass
    finally:
        sender.close()


def _tree_worker(sender, xcat, rows, held, y, k, fold, backoff_true,
                 lr_true, lr_full, current_weight, xlr, deadline):
    try:
        result = None
        tree_oof = _tree_oof(xcat, rows, y, k, fold, deadline)
        if tree_oof is not None:
            bw, lw, tw, temperature = _select_tree(
                backoff_true, lr_true, tree_oof, y, k)
            if tw == 0:
                result = (bw, lw, tw, None, None)
            elif time.monotonic() < deadline:
                tree_full = _tree_proba(xcat.iloc[rows], y, xcat.iloc[held], k)
                if lw > 0 and lr_full is None and time.monotonic() < deadline:
                    lr_full = _lr_proba(xlr[rows], y, xlr[held], k)
                if lw == 0 or lr_full is not None:
                    tree_full = _temperature(tree_full, temperature)
                    result = (bw, lw, tw, lr_full, tree_full)
        sender.send(result)
    except Exception:
        pass
    finally:
        sender.close()


def _bounded_worker(target, args, deadline):
    """Run one item worker, receiving before joining and always reaping it."""
    if time.monotonic() >= deadline:
        return None
    receiver = sender = process = None
    started = False
    item_deadline = min(deadline, time.monotonic() + ITEM_SECONDS)
    try:
        context = multiprocessing.get_context("fork")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(target=target, args=(sender, *args, item_deadline))
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


def _bounded_stack(features, rows, held, y, k, x, fold, deadline):
    return _bounded_worker(_lr_worker, (features, rows, held, y, k, x, fold), deadline)


def _bounded_tree(xcat, rows, held, y, k, fold, backoff_true, lr_true,
                  lr_full, current_weight, xlr, deadline):
    return _bounded_worker(_tree_worker,
        (xcat, rows, held, y, k, fold, backoff_true, lr_true, lr_full,
         current_weight, xlr), deadline)


def _best_weight(backoff, lr, k):
    """Weight on the backoff member that maximises the mixture's log score."""
    scores = [_log_score(w * backoff + (1 - w) * lr, k) for w in WEIGHTS]
    return float(WEIGHTS[int(np.argmax(scores))])


def predict(frame, schema):
    entry = time.monotonic()
    records = schema["items"]
    allocation = _instrument_budget(records)
    lr_deadline = entry + min(BUDGET_SECONDS, allocation)
    tree_deadline = entry + allocation
    items = [name for name, r in records.items() if r["class"] in ("GIVEN", "PREDICT")]
    given = [name for name in items if records[name]["class"] == "GIVEN"]
    targets = [name for name in items if records[name]["class"] == "PREDICT"]

    blank = frame[targets].isna().to_numpy()
    held = np.flatnonzero(blank.any(axis=1))
    train = np.flatnonzero(~blank.any(axis=1))
    features = [_codes(frame[g]) for g in given]
    fold = np.random.default_rng(0).permutation(len(train)) % FOLDS

    out, backoff_predictions, training = {}, {}, {}
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
        backoff_predictions[name] = p
        training[name] = (rows, y, k, fold[known])

    lr_cache = {}
    if features and time.monotonic() < lr_deadline:
        x = _design(features, given, records)
        if x is not None:
            for name, (rows, y, k, item_fold) in training.items():
                if time.monotonic() >= lr_deadline:
                    break
                if len(y) < 2 * FOLDS or len(np.unique(y)) < 2:
                    continue
                member = _bounded_stack(features, rows, held, y, k, x, item_fold, lr_deadline)
                if member is not None and time.monotonic() < lr_deadline:
                    backoff_true, lr_true, weight, regression = member
                    lr_cache[name] = (backoff_true, lr_true, regression, weight)
                    if regression is not None:
                        out[name] = weight * out[name] + (1 - weight) * regression

    # Trees are strictly optional: each item's previous committed prediction
    # remains intact unless its complete OOF selection and full refit succeed.
    if features and lr_cache and time.monotonic() < tree_deadline:
        xcat = _tree_categories(frame, given, records, schema["gated_value"])
        xlr = _design(features, given, records)
        eligible = [name for name, (rows, y, k, item_fold) in training.items()
                    if name in lr_cache and len(y) >= 600
                    and len(np.unique(y)) >= 2
                    and all(len(records[g]["values"]) + bool(records[g].get("gate")) <= 255
                            for g in given)]
        gated = sorted((n for n in eligible if records[n].get("gate")),
                       key=lambda n: (len(records[n]["values"]) + 1, n))
        ungated = sorted((n for n in eligible if not records[n].get("gate")),
                         key=lambda n: (len(records[n]["values"]), n))
        queue = []
        while gated or ungated:
            if gated:
                queue.append(gated.pop(0))
            if ungated:
                queue.append(ungated.pop(0))
        for name in queue:
            if time.monotonic() >= tree_deadline:
                break
            rows, y, k, item_fold = training[name]
            backoff_true, lr_true, lr_full, old_weight = lr_cache[name]
            member = _bounded_tree(xcat, rows, held, y, k, item_fold,
                                   backoff_true, lr_true, lr_full, old_weight,
                                   xlr, tree_deadline)
            if member is not None and time.monotonic() < tree_deadline:
                bw, lw, tw, regression, tree = member
                if tw > 0:
                    lr_part = 0.0 if lw == 0 else lw * regression
                    out[name] = (bw * backoff_predictions[name]
                                 + lr_part + tw * tree)

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
