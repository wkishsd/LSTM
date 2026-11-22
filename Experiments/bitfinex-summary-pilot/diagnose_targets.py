"""Test whether forecasting a fixed price move is easier than the strategy label."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from compare_models import EXPORT, PERIODS, train_tree, weekly_calls, window_summaries


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "Training/rocm-9070"))
    from mlp_trades.bitfinex import export_paths, prepare_bitfinex
    from mlp_trades.split import chronological_split

    path = root / EXPORT
    data = prepare_bitfinex(export_paths(path), training=True)
    timeline = np.load(path / "feature_timeline.npy", mmap_mode="r")
    names = json.loads((path / "manifest.json").read_text(encoding="utf-8"))["feature_names_per_minute"]
    with (path / "entry_targets.csv").open(encoding="utf-8-sig", newline="") as source:
        records = {record["entry_id"]: record for record in csv.DictReader(source)}
    labels = data.labels
    gross = np.array([float(records[i]["target_gross_return_pct"]) for i in data.ids])
    filled = np.array([int(records[i]["filled_entry_orders"]) > 0 for i in data.ids])
    fixed_fee = (gross > 0.10) & filled
    split, _ = chronological_split(data, 0.7, 0.15)
    rates = {}
    for part, ix in split.items():
        rates[part] = {
            "entries": len(ix),
            "fill_rate": float(filled[ix].mean()),
            "gross_positive_rate": float(np.mean((gross[ix] > 0) & filled[ix])),
            "positive_at_fixed_0_10_percent_fee": float(fixed_fee[ix].mean()),
            "observed_net_positive_rate": float(labels[ix].mean()),
        }

    order = data.valid[np.argsort(data.times[data.valid], kind="stable")]
    rows = data.history_rows[order]
    times = data.times[order]
    horizon_rows = rows + 30
    available = horizon_rows < len(timeline)
    short_upside = np.zeros(len(order), dtype=bool)
    short_upside[available] = timeline[horizon_rows[available], 0] < timeline[rows[available], 0]
    features = np.column_stack((timeline[rows], window_summaries(timeline, rows, names)))
    atr = timeline[rows, names.index("norm_atr_14_5m")]
    folds = []
    for start, stop in PERIODS:
        left = times[int(len(times) * start)]
        right = times[int(len(times) * stop)] if stop < 1 else times[-1] + 1
        evaluate = (times >= left) & (times < right) & available
        embargo = data.starts[order[evaluate]].min()
        train = available & (np.maximum.reduce((times, data.ends[order], times + 1800)) < embargo)
        fold = {"start": datetime.fromtimestamp(int(left), timezone.utc).date().isoformat(),
                "n": int(evaluate.sum()), "targets": {}}
        for name, target in (("30_minute_price_fall", short_upside), ("fixed_fee_win", fixed_fee[order])):
            scores = train_tree(features[train], target[train], features[evaluate])
            ix = np.flatnonzero(evaluate)
            holdout = target[ix]
            original = labels[order[ix]]
            scores_metrics = {
                "target_rate": float(holdout.mean()),
                "target_auc": float(roc_auc_score(holdout, scores)),
                "target_average_precision": float(average_precision_score(holdout, scores)),
                "strategy_auc_using_these_scores": float(roc_auc_score(original, scores)),
            }
            for key, ranking in (("model", scores), ("atr", atr[ix])):
                counts = weekly_calls(times[ix], original, ranking).values()
                scores_metrics[f"strategy_{key}_winners"] = sum(pair[0] for pair in counts)
                scores_metrics["strategy_calls"] = sum(pair[1] for pair in counts)
            fold["targets"][name] = scores_metrics
        folds.append(fold)
        print(json.dumps(fold), flush=True)
    output = {"rates": rates, "folds": folds}
    Path(__file__).with_name("target_diagnosis.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(rates), flush=True)


if __name__ == "__main__":
    main()
