"""Test whether learning continuous trade outcomes helps on the same weekly call quota."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Training/rocm-9070"))
sys.path.insert(0, str(ROOT / "Experiments/bitfinex-summary-pilot"))
from compare_models import EXPORT, FRACTION, PERIODS, WEEK, train_tree, window_summaries
from mlp_trades.bitfinex import export_paths, prepare_bitfinex


def rank_return(features, observed, train, test):
    y = observed[train]
    # Bound a few extreme outliers in the fitted target using training data alone.
    cap = np.quantile(y, [0.005, 0.995])
    model = HistGradientBoostingRegressor(
        learning_rate=0.06, max_iter=100, max_leaf_nodes=7, min_samples_leaf=180,
        l2_regularization=3.0, early_stopping=False, random_state=42,
    )
    model.fit(features[train], np.clip(y, *cap))
    return model.predict(features[test])


def score(scores, times, labels, net):
    weeks = (times + 3 * 86400) // WEEK
    chosen = []
    for week in np.unique(weeks):
        ix = np.flatnonzero(weeks == week)
        n = round(len(ix) * FRACTION)
        chosen.extend(ix[np.argsort(-scores[ix], kind="stable")[:n]])
    chosen = np.asarray(chosen, dtype=np.int64)
    return {"winners": int(labels[chosen].sum()), "calls": len(chosen),
            "ap": float(average_precision_score(labels, scores)),
            "mean_realized_net_return_pct": float(net[chosen].mean())}


def main():
    path = ROOT / EXPORT
    data = prepare_bitfinex(export_paths(path), training=True)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    names = manifest["feature_names_per_minute"]
    timeline = np.load(path / "feature_timeline.npy", mmap_mode="r")
    with open(data.paths.entries, encoding="utf-8-sig", newline="") as file:
        outcomes = {row["entry_id"]: row for row in csv.DictReader(file)}
    order = data.valid[np.argsort(data.times[data.valid], kind="stable")]
    pos, times, labels = data.history_rows[order], data.times[order], data.labels[order]
    ids = data.ids[order]
    gross = np.array([float(outcomes[i]["target_gross_return_pct"]) for i in ids])
    fees = np.array([float(outcomes[i]["commission_pct"]) for i in ids])
    net = gross - fees
    assert np.array_equal(net > 0, labels)
    features = np.column_stack((timeline[pos], window_summaries(timeline, pos, names)))
    atr = features[:, names.index("norm_atr_14_5m")]
    folds = []
    for start, stop in PERIODS:
        left = times[int(len(times) * start)]
        right = times[int(len(times) * stop)] if stop < 1 else times[-1] + 1
        test = (times >= left) & (times < right)
        embargo = data.starts[order[test]].min()
        train = np.maximum(times, data.ends[order]) < embargo
        predictions = {
            "atr": atr[test],
            "binary": train_tree(features[train], labels[train], features[test]),
            "direct_net": rank_return(features, net, train, test),
            "gross_minus_fees": (rank_return(features, gross, train, test)
                                 - rank_return(features, fees, train, test)),
        }
        result = {"start": datetime.fromtimestamp(int(left), timezone.utc).date().isoformat(),
                  "predictions": {name: score(s, times[test], labels[test], net[test])
                                  for name, s in predictions.items()}}
        folds.append(result)
        print(json.dumps(result), flush=True)
    output = {"unit": "mean realized net return per selected entry in percent", "folds": folds}
    Path(__file__).with_name("return_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
