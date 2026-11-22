"""Compare entry snapshots and chronological summaries on expanding time splits."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np


EXPORT = Path("Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd")
FRACTION = 1000 / 15728
WEEK = 7 * 86400
PERIODS = ((0.40, 0.55), (0.55, 0.70), (0.70, 0.85), (0.85, 1.0))


def window_summaries(timeline, rows, names):
    by_name = {name: column for column, name in enumerate(names)}
    columns = []
    for lag in (15, 60, 240, 719):
        columns.append(100 * (timeline[rows, 0] - timeline[rows - lag, 0]))
    for tf in ("5m", "15m", "30m"):
        market = {
            "RSI": timeline[:, by_name[f"norm_rsi_14_{tf}"]],
            "ATR": timeline[:, by_name[f"norm_atr_14_{tf}"]],
            "ADX": timeline[:, by_name[f"norm_adx_14_4_{tf}"]],
            "DI spread": (timeline[:, by_name[f"norm_di_plus_14_{tf}"]]
                          - timeline[:, by_name[f"norm_di_minus_14_{tf}"]]),
        }
        for name, period in (("macd", "12_26"), ("macd_signal", "12_26_9")):
            bounded = timeline[:, by_name[f"norm_{name}_{period}_{tf}"]]
            close = np.exp(timeline[:, by_name[f"norm_price_{tf}"]])
            market[name] = bounded / (np.sqrt((1 - bounded) * (1 + bounded)) * close)
        for values in market.values():
            for lag in (15, 60, 240):
                columns.append(values[rows] - values[rows - lag])

    log_price = timeline[:, 0].astype(np.float64)
    returns = np.diff(log_price, prepend=log_price[0])
    squared_returns = np.r_[0, np.cumsum(returns * returns)]
    volume = timeline[:, by_name["norm_volume_5m"]]
    volume_sum = np.r_[0, np.cumsum(volume, dtype=np.float64)]
    for start in (540, 360, 180, 0):
        last = rows - start
        # At the oldest end of the sample the block has 179 preceding intervals.
        first = last - 179
        columns.append(100 * (log_price[last] - log_price[first]))
        columns.append(100 * np.sqrt(squared_returns[last + 1] - squared_returns[first + 1]))
        columns.append(volume[rows] - (volume_sum[last + 1] - volume_sum[first]) / 180)
    return np.column_stack(columns).astype(np.float32)


def weekly_calls(times, target, scores):
    weeks = (times + 3 * 86400) // WEEK
    wins = {}
    for week in np.unique(weeks):
        rows = np.flatnonzero(weeks == week)
        calls = round(len(rows) * FRACTION)
        top = rows[np.argsort(-scores[rows], kind="stable")[:calls]]
        wins[int(week)] = (int(target[top].sum()), calls)
    return wins


def train_tree(train_x, y, test_x):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=100, max_leaf_nodes=7, min_samples_leaf=180,
        l2_regularization=3.0, early_stopping=False, random_state=42,
    )
    model.fit(train_x, y)
    return model.predict_proba(test_x)[:, 1]


def train_ranker(train_x, y, train_times, test_x):
    from lightgbm import LGBMRanker

    _, groups = np.unique((train_times + 3 * 86400) // WEEK, return_counts=True)
    model = LGBMRanker(
        objective="lambdarank", lambdarank_truncation_level=45,
        learning_rate=0.06, n_estimators=100, num_leaves=7, min_child_samples=180,
        reg_lambda=3.0, random_state=42, n_jobs=4, verbosity=-1,
    )
    model.fit(train_x, y, group=groups)
    return model.predict(test_x)


def main():
    from sklearn.metrics import average_precision_score

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("results.json"))
    parser.add_argument("--ranker", action="store_true", help="Include LightGBM's within-week ranking model")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / "Training/rocm-9070"))
    from mlp_trades.bitfinex import export_paths, prepare_bitfinex

    path = root / EXPORT
    data = prepare_bitfinex(export_paths(path), training=True)
    timeline = np.load(path / "feature_timeline.npy", mmap_mode="r")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["feature_names_per_minute"]
    order = data.valid[np.argsort(data.times[data.valid], kind="stable")]
    rows = data.history_rows[order]
    times = data.times[order]
    target = data.labels[order]
    snapshot = timeline[rows]
    combined = np.column_stack((snapshot, window_summaries(timeline, rows, names)))
    atr = snapshot[:, names.index("norm_atr_14_5m")]
    print(f"{len(order):,} entries, {snapshot.shape[1]} snapshot and {combined.shape[1]} combined features", flush=True)

    model_names = ["atr", "snapshot", "window"]
    if args.ranker:
        model_names.append("weekly_ranker")
    all_weeks = {key: {} for key in model_names}
    result = {"quota_fraction_per_week": FRACTION, "folds": []}
    for start, stop in PERIODS:
        start_at = times[int(len(order) * start)]
        stop_at = times[int(len(order) * stop)] if stop < 1 else times[-1] + 1
        observe = (times >= start_at) & (times < stop_at)
        # Use the trainer's original 1,410-minute input and outcome embargo.
        fit_until = data.starts[order[observe]].min()
        train = np.maximum(times, data.ends[order]) < fit_until
        outcome = target[observe]
        fold = {
            "start": datetime.fromtimestamp(int(times[observe][0]), timezone.utc).date().isoformat(),
            "end": datetime.fromtimestamp(int(times[observe][-1]), timezone.utc).date().isoformat(),
            "training_entries": int(train.sum()),
            "evaluation_entries": int(observe.sum()),
            "positive_rate": float(outcome.mean()),
            "models": {},
        }
        predictions = {
            "atr": atr[observe],
            "snapshot": train_tree(snapshot[train], target[train], snapshot[observe]),
            "window": train_tree(combined[train], target[train], combined[observe]),
        }
        if args.ranker:
            predictions["weekly_ranker"] = train_ranker(combined[train], target[train], times[train], combined[observe])
        for key, scores in predictions.items():
            counts = weekly_calls(times[observe], outcome, scores)
            winners = sum(count[0] for count in counts.values())
            calls = sum(count[1] for count in counts.values())
            for week, count in counts.items():
                old = all_weeks[key].get(week, (0, 0))
                all_weeks[key][week] = (old[0] + count[0], old[1] + count[1])
            fold["models"][key] = {
                "average_precision": float(average_precision_score(outcome, scores)),
                "winners": winners, "calls": calls,
            }
        result["folds"].append(fold)
        print(json.dumps(fold), flush=True)

    rng = np.random.default_rng(42)
    weeks = sorted(all_weeks["atr"])
    block_count = (len(weeks) + 3) // 4
    starts = rng.integers(len(weeks), size=(4000, block_count, 1))
    draws = ((starts + np.arange(4)) % len(weeks)).reshape(4000, -1)[:, :len(weeks)]
    total = {}
    for key, weeks_counts in all_weeks.items():
        counts = np.array([weeks_counts[week] for week in weeks])
        difference = counts[:, 0] - np.array([all_weeks["atr"][week][0] for week in weeks])
        low, high = np.quantile(difference[draws].sum(axis=1), [0.025, 0.975])
        total[key] = {
            "winners": int(counts[:, 0].sum()),
            "calls": int(counts[:, 1].sum()),
            "extra_winners_vs_atr": int(difference.sum()),
            "four_week_block_bootstrap_interval_extra_winners": [float(low), float(high)],
        }
    result["total"] = total
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(total), flush=True)


if __name__ == "__main__":
    main()
