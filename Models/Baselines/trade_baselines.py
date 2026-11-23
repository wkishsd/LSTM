"""Non-neural baselines trained and tested on exactly the same entries as the trade MLP.

The trainer (`python -m mlp_trades train --baselines atr,logistic,gbt`) forks this
module before the MLP starts, so both use the same prepared data and chronological
split. Everything here runs on the CPU with NumPy only, leaving the GPU to the MLP.

Models
    atr       Rank entries by M5 ATR / close at the decision minute. No fitting.
    logistic  L2-regularized logistic regression (one-vs-rest for profit buckets),
              fitted by Newton's method on standardized decision-time features.
    gbt       Histogram gradient-boosted trees with early stopping on validation.
    hour      Hour-of-day rule: each test entry scores its UTC hour's outcome rates on the
              train split. Fitted on train only, so no hour picked from the test period.

The decision-time features use only rows at or before each entry's feature end row.
"""

import json
import math
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

MODELS = ("atr", "logistic", "gbt", "hour")
MODEL_LABELS = {"atr": "ATR rule", "logistic": "Logistic regression", "gbt": "Boosted trees", "hour": "Hour-of-day rule"}
TIMEFRAMES = ("M1", "M5", "M15", "M30")
RETURN_LAGS = (1, 5, 15, 60, 240, 720, 1409)
ANCHOR_WINDOW = 120
VOLUME_WINDOW = 60
BOUNDED_LIMIT = 1 - 2.0 ** -24
HOUR_SMOOTHING = 20.0  # pseudo-entries pulling a thin hour towards the overall train rates


def log(message):
    print(f"[baselines] {message}", flush=True)


# ----------------------------------------------------------------------------- features

def window_sum(cumulative, ends, length):
    """Sum of rows ends-length+1 .. ends using a cumulative sum with a leading zero row."""
    return cumulative[ends + 1] - cumulative[ends + 1 - length]


def decision_features(timeline, positions, times, bounded_macd):
    """Compact features at each entry's last closed minute. Returns (matrix, names)."""
    timeline = np.asarray(timeline, dtype=np.float32)
    current = timeline[positions].astype(np.float64)
    columns, names = [], []
    for k, tf in enumerate(TIMEFRAMES):
        base = 9 * k
        close = np.exp(current[:, base])
        for offset, name in ((2, "rsi"), (3, "atr_over_close"), (6, "adx"), (7, "di_plus"), (8, "di_minus")):
            columns.append(current[:, base + offset])
            names.append(f"{tf}.{name}")
        macd = current[:, base + 4:base + 6]
        if bounded_macd:
            macd = np.clip(macd, -BOUNDED_LIMIT, BOUNDED_LIMIT)
            macd = macd / np.sqrt((1 - macd) * (1 + macd))
        relative = macd / close[:, None]
        columns += [relative[:, 0], relative[:, 1], relative[:, 0] - relative[:, 1]]
        names += [f"{tf}.macd_over_close", f"{tf}.signal_over_close", f"{tf}.hist_over_close"]
        volume = np.concatenate(([0.0], np.cumsum(timeline[:, base + 1], dtype=np.float64)))
        columns.append(current[:, base + 1] - window_sum(volume, positions, VOLUME_WINDOW) / VOLUME_WINDOW)
        names.append(f"{tf}.log_volume_vs_{VOLUME_WINDOW}m_mean")
    price = timeline[:, 0].astype(np.float64)
    for lag in RETURN_LAGS:
        columns.append(100 * (price[positions] - price[positions - lag]))
        names.append(f"M1.return_{lag}m_pct")
    squared = np.concatenate(([0.0, 0.0], np.cumsum(np.diff(price) ** 2)))
    for length in (60, 240):
        columns.append(100 * np.sqrt(window_sum(squared, positions, length) / length))
        names.append(f"M1.realized_vol_{length}m_pct")
    anchors = timeline[:, 36:]
    recent = np.concatenate((np.zeros((1, anchors.shape[1])), np.cumsum(anchors, axis=0, dtype=np.float64)))
    recent = window_sum(recent, positions, ANCHOR_WINDOW) > 0
    for j in range(anchors.shape[1]):
        columns += [current[:, 36 + j], recent[:, j].astype(np.float64)]
        names += [f"anchor{j:02d}.now", f"anchor{j:02d}.last_{ANCHOR_WINDOW}m"]
    hour = (times % 86400) / 3600
    weekday = ((times // 86400) + 3) % 7  # 1970-01-01 was a Thursday; Monday = 0
    for value, period, name in ((hour, 24, "hour"), (weekday, 7, "weekday")):
        columns += [np.sin(2 * np.pi * value / period), np.cos(2 * np.pi * value / period)]
        names += [f"{name}_sin", f"{name}_cos"]
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise ValueError("Baseline features contain nonfinite values")
    return matrix, names


# ----------------------------------------------------------------------------- metrics

def evaluate(targets, scores, num_classes, probabilities=True, returns=None):
    """Metrics with the same definitions as the MLP trainer."""
    from mlp_trades.metrics import FIXED_QUOTA, Metrics, average_precision, quota_metrics

    result = {}
    if probabilities:
        probs = np.clip(scores, 1e-12, 1.0)
        loss = float(-np.mean(np.log(probs[np.arange(len(targets)), targets])))
        metrics = Metrics(num_classes)
        metrics.update(targets, probs.argmax(1), loss)
        result.update(metrics.result())
    if num_classes == 2:
        ranking = scores[:, 1] if scores.ndim == 2 else scores
        result["average_precision"] = average_precision(targets, ranking)
        result["fixed_quota"] = quota_metrics(targets, ranking, FIXED_QUOTA, returns)
        result["positive_rate"] = float(targets.mean())
    else:
        observed = np.unique(targets)
        result["average_precision"] = float(np.mean([
            average_precision(targets == label, scores[:, label]) for label in observed
        ]))
        result["average_precision_baseline"] = 1 / len(observed)
    result.setdefault("count", int(len(targets)))
    return result


def selection_score(result):
    return result["average_precision"]


# ----------------------------------------------------------------------------- logistic regression

def fit_binary_logistic(x, y, l2, iterations=50):
    """Newton's method on mean log loss + l2/2 * |w|^2 (the intercept is not penalized)."""
    n, f = x.shape
    design = np.hstack((x, np.ones((n, 1))))
    weights = np.zeros(f + 1)
    penalty = np.full(f + 1, l2)
    penalty[-1] = 0.0
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-np.clip(design @ weights, -30, 30)))
        gradient = design.T @ (p - y) / n + penalty * weights
        hessian = (design.T * (p * (1 - p))) @ design / n + np.diag(penalty + 1e-9)
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return weights


def logistic_probabilities(models, x):
    design = np.hstack((x, np.ones((len(x), 1))))
    raw = np.column_stack([1 / (1 + np.exp(-np.clip(design @ w, -30, 30))) for w in models])
    if len(models) == 1:
        return np.column_stack((1 - raw[:, 0], raw[:, 0]))
    return raw / raw.sum(axis=1, keepdims=True)


def fit_logistic(x, y, num_classes, l2):
    labels = [1] if num_classes == 2 else range(num_classes)
    return [fit_binary_logistic(x, (y == label).astype(np.float64), l2) for label in labels]


# ----------------------------------------------------------------------------- boosted trees

class HistogramGBT:
    """Gradient-boosted regression trees on quantile bins, logistic or softmax loss."""

    def __init__(self, num_classes, depth=4, learning_rate=0.05, l2=1.0, min_hessian=20.0,
                 subsample=0.7, max_rounds=600, patience=50, bins=64, seed=42):
        self.num_classes = num_classes
        self.outputs = 1 if num_classes == 2 else num_classes
        self.depth, self.learning_rate, self.l2 = depth, learning_rate, l2
        self.min_hessian, self.subsample = min_hessian, subsample
        self.max_rounds, self.patience, self.bins = max_rounds, patience, bins
        self.rng = np.random.default_rng(seed)
        self.trees = []  # one list of `outputs` trees per round

    def fit_bins(self, x):
        quantiles = np.linspace(0, 1, self.bins + 1)[1:-1]
        self.edges = [np.unique(np.quantile(column, quantiles)) for column in x.T]

    def encode(self, x):
        return np.column_stack([np.searchsorted(edges, column, side="right")
                                for edges, column in zip(self.edges, x.T)]).astype(np.int64)

    def build_tree(self, codes, gradient, hessian, rows):
        features = codes.shape[1]
        offsets = np.arange(features) * self.bins
        nodes = [{"rows": rows}]
        frontier = [0]
        for _ in range(self.depth):
            next_frontier = []
            for index in frontier:
                node = nodes[index]
                rows_here = node["rows"]
                flat = (codes[rows_here] + offsets).ravel()
                size = features * self.bins
                g_hist = np.bincount(flat, np.repeat(gradient[rows_here], features), size).reshape(features, -1)
                h_hist = np.bincount(flat, np.repeat(hessian[rows_here], features), size).reshape(features, -1)
                g_left, h_left = np.cumsum(g_hist, axis=1), np.cumsum(h_hist, axis=1)
                g_total, h_total = g_left[0, -1], h_left[0, -1]
                g_right, h_right = g_total - g_left, h_total - h_left
                gain = (g_left ** 2 / (h_left + self.l2) + g_right ** 2 / (h_right + self.l2)
                        - g_total ** 2 / (h_total + self.l2))
                gain[(h_left < self.min_hessian) | (h_right < self.min_hessian)] = -np.inf
                feature, threshold = np.unravel_index(np.argmax(gain), gain.shape)
                if not gain[feature, threshold] > 1e-9:
                    continue
                goes_left = codes[rows_here, feature] <= threshold
                node.update(feature=int(feature), threshold=int(threshold),
                            left=len(nodes), right=len(nodes) + 1)
                nodes += [{"rows": rows_here[goes_left]}, {"rows": rows_here[~goes_left]}]
                next_frontier += [node["left"], node["right"]]
            frontier = next_frontier
        feature = np.full(len(nodes), -1)
        threshold = np.zeros(len(nodes), dtype=np.int64)
        children = np.zeros((len(nodes), 2), dtype=np.int64)
        value = np.zeros(len(nodes))
        for index, node in enumerate(nodes):
            if "feature" in node:
                feature[index], threshold[index] = node["feature"], node["threshold"]
                children[index] = node["left"], node["right"]
            else:
                rows_here = node["rows"]
                value[index] = -self.learning_rate * gradient[rows_here].sum() / (hessian[rows_here].sum() + self.l2)
        return feature, threshold, children, value

    def tree_output(self, tree, codes):
        feature, threshold, children, value = tree
        node = np.zeros(len(codes), dtype=np.int64)
        for _ in range(self.depth):
            split = feature[node] >= 0
            if not split.any():
                break
            idx = np.flatnonzero(split)
            left = codes[idx, feature[node[idx]]] <= threshold[node[idx]]
            node[idx] = np.where(left, children[node[idx], 0], children[node[idx], 1])
        return value[node]

    def probabilities_from_raw(self, raw):
        if self.outputs == 1:
            p = 1 / (1 + np.exp(-np.clip(raw[:, 0], -30, 30)))
            return np.column_stack((1 - p, p))
        shifted = np.exp(raw - raw.max(axis=1, keepdims=True))
        return shifted / shifted.sum(axis=1, keepdims=True)

    def initial_raw(self, y):
        if self.outputs == 1:
            rate = np.clip(y.mean(), 1e-6, 1 - 1e-6)
            return np.array([math.log(rate / (1 - rate))])
        return np.log(np.clip(np.bincount(y, minlength=self.outputs) / len(y), 1e-6, None))

    def fit(self, x, y, x_val=None, y_val=None):
        self.fit_bins(x)
        codes = self.encode(x)
        self.base = self.initial_raw(y)
        raw = np.tile(self.base, (len(y), 1))
        targets = np.eye(self.num_classes)[y][:, -self.outputs:] if self.outputs == 1 else np.eye(self.outputs)[y]
        validation = None
        if x_val is not None:
            val_codes = self.encode(x_val)
            val_raw = np.tile(self.base, (len(y_val), 1))
            validation = []
        best_round, best_loss, since_best = 0, np.inf, 0
        for round_number in range(1, self.max_rounds + 1):
            probs = self.probabilities_from_raw(raw)
            p = probs[:, 1:] if self.outputs == 1 else probs
            gradient_all, hessian_all = p - targets, np.maximum(p * (1 - p), 1e-6)
            rows = np.flatnonzero(self.rng.random(len(y)) < self.subsample)
            trees = []
            for output in range(self.outputs):
                tree = self.build_tree(codes, gradient_all[:, output], hessian_all[:, output], rows)
                trees.append(tree)
                raw[:, output] += self.tree_output(tree, codes)
                if validation is not None:
                    val_raw[:, output] += self.tree_output(tree, val_codes)
            self.trees.append(trees)
            if validation is not None:
                val_probs = np.clip(self.probabilities_from_raw(val_raw), 1e-12, 1)
                loss = float(-np.mean(np.log(val_probs[np.arange(len(y_val)), y_val])))
                validation.append(loss)
                if loss < best_loss - 1e-6:
                    best_round, best_loss, since_best = round_number, loss, 0
                else:
                    since_best += 1
                    if since_best >= self.patience:
                        break
        self.rounds = best_round if validation is not None and best_round else len(self.trees)
        self.trees = self.trees[:self.rounds]
        self.validation_losses = validation
        return self

    def predict_proba(self, x):
        codes = self.encode(x)
        raw = np.tile(self.base, (len(x), 1))
        for trees in self.trees:
            for output, tree in enumerate(trees):
                raw[:, output] += self.tree_output(tree, codes)
        return self.probabilities_from_raw(raw)

    def split_counts(self, names):
        counts = {}
        for trees in self.trees:
            for feature, *_ in trees:
                for f in feature[feature >= 0]:
                    counts[names[f]] = counts.get(names[f], 0) + 1
        return dict(sorted(counts.items(), key=lambda item: -item[1])[:15])


# ----------------------------------------------------------------------------- hour-of-day rule

def utc_hours(times):
    return ((np.asarray(times, dtype=np.int64) % 86400) // 3600).astype(np.int64)


def fit_hour_rates(hours, y, num_classes, smoothing=HOUR_SMOOTHING):
    """Per-hour class probabilities on the train split, smoothed towards the overall rates."""
    overall = np.bincount(y, minlength=num_classes) / len(y)
    counts = np.zeros((24, num_classes))
    np.add.at(counts, (hours, y), 1.0)
    return (counts + smoothing * overall) / (counts.sum(axis=1, keepdims=True) + smoothing)


# ----------------------------------------------------------------------------- runner

def run(data, args, out):
    """Train each requested baseline on the MLP's train split and test it once."""
    from mlp_trades.data import mmap_array
    from mlp_trades.schema import WIDTH
    from mlp_trades.split import training_split

    started = perf_counter()
    requested = [name.strip() for name in args.baselines.split(",") if name.strip()]
    splits, _ = training_split(data, args.train_fraction, args.validation_fraction)
    target = data.schema.get("profit_target")
    num_classes = len(target["class_names"]) if target else 2
    timeline = mmap_array(data.paths.timeline, "timeline", (WIDTH,))
    bounded = data.schema.get("macd_normalization") == "x / sqrt(1 + x^2)"
    features, labels, returns, hours = {}, {}, {}, {}
    for name, rows in splits.items():
        hours[name] = utc_hours(data.times[rows])
        features[name], names = decision_features(timeline, data.history_rows[rows], data.times[rows], bounded)
        labels[name] = data.labels[rows]
        returns[name] = None if getattr(data, "returns", None) is None else data.returns[rows]

    def score(split, values, probabilities=True):
        return evaluate(labels[split], values, num_classes, probabilities, returns[split])
    log(f"{len(names)} decision-time features for " + ", ".join(f"{k} {len(v):,}" for k, v in labels.items()))
    has_validation = "validation" in splits
    report = {"feature_names": names, "split_counts": {k: int(len(v)) for k, v in labels.items()},
              "num_classes": num_classes, "models": {}}
    for name in requested:
        model_started = perf_counter()
        entry = {"label": MODEL_LABELS[name]}
        if name == "atr":
            if num_classes != 2:
                entry["skipped"] = "The ATR rule ranks a single outcome; it does not apply to profit buckets."
                report["models"][name] = entry
                log(f"{MODEL_LABELS[name]} skipped: {entry['skipped']}")
                continue
            column = names.index("M5.atr_over_close")
            entry["settings"] = {"score": "M5 ATR / close at the decision minute (higher = enter)"}
            if has_validation:
                entry["validation"] = score("validation", features["validation"][:, column], False)
            entry["test"] = score("test", features["test"][:, column], False)
        elif name == "logistic":
            mean = features["train"].mean(axis=0)
            scale = features["train"].std(axis=0)
            scale[scale < 1e-12] = 1.0
            standardized = {k: (v - mean) / scale for k, v in features.items()}
            grid = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0) if has_validation else (1e-1,)
            best = None
            for l2 in grid:
                models = fit_logistic(standardized["train"], labels["train"], num_classes, l2)
                if has_validation:
                    result = score("validation", logistic_probabilities(models, standardized["validation"]))
                    log(f"logistic l2={l2:g} validation average precision {result['average_precision']:.4f}")
                    if best is None or selection_score(result) > selection_score(best[2]):
                        best = (l2, models, result)
                else:
                    best = (l2, models, None)
            l2, models, validation = best
            entry["settings"] = {"l2": l2, "selected_by": "validation average precision" if has_validation else None}
            if validation is not None:
                entry["validation"] = validation
            entry["test"] = score("test", logistic_probabilities(models, standardized["test"]))
            weights = models[0][:-1] if num_classes == 2 else np.abs(np.array(models)[:, :-1]).mean(axis=0)
            order = np.argsort(-np.abs(weights))[:15]
            entry["largest_coefficients"] = {names[i]: float(weights[i]) for i in order}
        elif name == "gbt":
            model = HistogramGBT(num_classes, seed=args.seed,
                                 max_rounds=600 if num_classes == 2 else 300)
            model.fit(features["train"], labels["train"],
                      features.get("validation"), labels.get("validation"))
            entry["settings"] = {"depth": model.depth, "learning_rate": model.learning_rate, "l2": model.l2,
                                 "min_hessian": model.min_hessian, "subsample": model.subsample,
                                 "bins": model.bins, "rounds": model.rounds,
                                 "early_stopping": "validation log loss" if has_validation else None}
            if has_validation:
                entry["validation"] = score("validation", model.predict_proba(features["validation"]))
            entry["test"] = score("test", model.predict_proba(features["test"]))
            entry["most_used_features"] = model.split_counts(names)
        elif name == "hour":
            rates = fit_hour_rates(hours["train"], labels["train"], num_classes)
            entry["settings"] = {"score": "train-split outcome rates of the entry's UTC hour",
                                 "smoothing_entries": HOUR_SMOOTHING}
            if has_validation:
                entry["validation"] = score("validation", rates[hours["validation"]])
            entry["test"] = score("test", rates[hours["test"]])
            column = 1 if num_classes == 2 else num_classes - 1
            entry["train_rate_by_utc_hour"] = {f"{h:02d}": float(rates[h, column]) for h in range(24)}
            entry["train_entries_by_utc_hour"] = np.bincount(hours["train"], minlength=24).tolist()
        else:
            raise ValueError(f"Unknown baseline model: {name}")
        entry["seconds"] = perf_counter() - model_started
        report["models"][name] = entry
        test = entry["test"]
        summary = f"test average precision {test['average_precision']:.4f}"
        if "fixed_quota" in test:
            quota = test["fixed_quota"]
            summary += (f", top {quota['calls']:,}: {quota['positives']:.0f} winners "
                        f"vs {quota['baseline_positives']:.0f} at random")
            if "net_return_pct_per_trade" in quota:
                summary += (f", net {quota['net_return_pct_per_trade']:+.3f}%/trade "
                            f"vs {quota['all_net_return_pct_per_trade']:+.3f}% taking all")
        log(f"{MODEL_LABELS[name]} {summary} ({entry['seconds']:.0f} s)")
        write(out, report)  # keep partial results if a later model fails
    report["seconds"] = perf_counter() - started
    write(out, report)
    log(f"finished in {report['seconds']:.0f} s")
    return report


def write(out, report):
    path = Path(out) / "baselines.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def child_main(data, args, out):
    """Entry point inside the forked process; never touches CUDA."""
    try:
        os.nice(5)
    except (AttributeError, OSError):
        pass
    try:
        run(data, args, out)
    except Exception as error:  # report instead of killing the MLP run
        log(f"ERROR: {error}")
        write(out, {"error": str(error)})
        sys.exit(1)
