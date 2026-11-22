"""Compare the strategy target with an ATR-scaled first-touch target using frozen candles."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Training/rocm-9070"))
sys.path.insert(0, str(ROOT / "Experiments/bitfinex-summary-pilot"))

from compare_models import EXPORT, PERIODS, train_tree, weekly_calls, window_summaries
from mlp_trades.bitfinex import export_paths, prepare_bitfinex


HORIZON = 60
MISSING, AMBIGUOUS, DOWN, UP, NEITHER = -1, -2, 1, 2, 0


def frozen_high_low(first, n, cache=ROOT / "Exports/candle_cache/ETHUSD/1m"):
    high, low, close = (np.full(n, np.nan) for _ in range(3))
    for path in sorted(cache.glob("*.json")):
        rows = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
        if not len(rows):
            continue
        pos = (rows[:, 0].astype(np.int64) // 1000 - first) // 60
        use = (pos >= 0) & (pos < n)
        pos, rows = pos[use], rows[use]
        for arr, column in ((high, 3), (low, 4), (close, 2)):
            overlap = np.isfinite(arr[pos])
            if not np.array_equal(arr[pos[overlap]], rows[overlap, column]):
                raise ValueError(f"Conflicting snapshot candle values in {path.name}")
            arr[pos] = rows[:, column]
    return high, low, close


def make_barrier_labels(data, timeline, names, manifest, cache=ROOT / "Exports/candle_cache/ETHUSD/1m"):
    clock_first = int(datetime.fromisoformat(manifest["feature_timeline_first_close"]).timestamp())
    positions = data.history_rows
    high, low, close = frozen_high_low(clock_first, len(timeline) + HORIZON, cache)
    assert np.array_equal(data.times, clock_first + positions * 60)
    origin = np.exp(timeline[positions, names.index("norm_price")].astype(float))
    cached = close[positions - 1]
    observed = np.isfinite(cached)
    np.testing.assert_allclose(origin[observed], cached[observed], rtol=2e-6, atol=0.0)
    scale = timeline[positions, names.index("norm_atr_14_5m")].astype(float)
    upper, lower = origin * (1 + scale), origin * (1 - scale)
    label = np.full(len(positions), NEITHER, dtype=np.int8)
    for step in range(HORIZON):
        active = label == NEITHER
        if not active.any():
            break
        # At step zero the candle opens at the entry timestamp and closes one minute later.
        hi, lo = high[positions + step], low[positions + step]
        both = (hi >= upper) & (lo <= lower)
        unseen = ~np.isfinite(hi) | ~np.isfinite(lo)
        label[active & (lo <= lower)] = DOWN
        label[active & (hi >= upper)] = UP
        label[active & both] = AMBIGUOUS
        label[active & unseen] = MISSING
    missing_cumulative = np.r_[0, (~np.isfinite(close)).cumsum()]
    missing_window = missing_cumulative[positions] - missing_cumulative[positions - 720]
    return label, missing_window


def evaluate(y, scores, times):
    weekly = weekly_calls(times, y, scores)
    return {"average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "winners": sum(row[0] for row in weekly.values()),
            "calls": sum(row[1] for row in weekly.values())}


def main():
    path = ROOT / EXPORT
    data = prepare_bitfinex(export_paths(path), training=True)
    timeline = np.load(path / "feature_timeline.npy", mmap_mode="r")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["feature_names_per_minute"]
    barrier, missing_window = make_barrier_labels(data, timeline, names, manifest)
    order = data.valid[np.argsort(data.times[data.valid], kind="stable")]
    positions, times = data.history_rows[order], data.times[order]
    features = np.column_stack((timeline[positions], window_summaries(timeline, positions, names)))
    atr = features[:, names.index("norm_atr_14_5m")]
    classes = barrier[order]
    targets = {"strategy": data.labels[order], "downside_first": (classes == DOWN).astype(np.int64)}
    eligible = classes >= 0
    folded = []
    for start, stop in PERIODS:
        left = times[int(len(times) * start)]
        right = times[int(len(times) * stop)] if stop < 1 else times[-1] + 1
        block = (times >= left) & (times < right)
        test = block & eligible
        embargo = data.starts[order[test]].min()
        train = eligible & (np.maximum(data.ends[order], times + 60 * HORIZON) < embargo)
        evaluations = {
            "start": datetime.fromtimestamp(int(left), timezone.utc).date().isoformat(),
            "count": int(test.sum()), "train": int(train.sum()),
            "missing_before_touch": int((block & (classes == MISSING)).sum()),
            "ambiguous_first_touch": int((block & (classes == AMBIGUOUS)).sum()),
            "median_missing_minutes_in_input": float(np.median(missing_window[order[block]])),
            "target_base_rates": {key: float(y[test].mean()) for key, y in targets.items()},
            "strategy_rates_by_barrier": {},
            "scores": {},
        }
        for name, tag in (("down", DOWN), ("up", UP), ("neither", NEITHER)):
            selected = test & (classes == tag)
            rate = float(targets["strategy"][selected].mean()) if selected.any() else None
            evaluations["strategy_rates_by_barrier"][name] = {"count": int(selected.sum()), "strategy_win_rate": rate}
        predictions = {"atr": atr[test]}
        for name, y in targets.items():
            predictions[name] = train_tree(features[train], y[train], features[test])
        for name, score in predictions.items():
            evaluations["scores"][name] = {key: evaluate(y[test], score, times[test]) for key, y in targets.items()}
        folded.append(evaluations)
        print(json.dumps(evaluations), flush=True)
    out = {"horizon_minutes": HORIZON, "barrier_relative_to_entry": "one M5 ATR each side",
           "unknown": "missing candle before first touch or both barriers touched in first hit candle",
           "folds": folded}
    Path(__file__).with_name("barrier_results.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
