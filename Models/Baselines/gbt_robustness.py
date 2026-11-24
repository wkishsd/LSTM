"""Robustness of boosted trees on the risk-unit target: seeds, thresholds, top-K, exports, walk-forward.

    python Models/Baselines/gbt_robustness.py --exports current_rules_15m current_rules_5m

Uses the trainer's own data loading, chronological split and purging, the 85 decision-time features
and the HistogramGBT of trade_baselines.py (early stopping on validation log loss).

Targets: net >= 0 (the export label, as a control) and net / ATR 14 on M15 >= k for k in --thresholds
(risk_targets.csv). Money is always the export's own series net return in % after fees.

Two schemes:
- standard: the trainer's 70 / 15 / 15 split, repeated over --seeds;
- walk-forward: for each 3-month test block from --wf-start, train on everything before the 3 months
  that precede it, validate on those 3 months, test on the block. Entries whose outcome reaches into
  the next block are purged from the earlier part.

For every fit it records, on validation and test, the mean net of the top 500 / 1,000 / 2,000 entries by
score and of all entries, and the number of class-1 entries among the top 1,000. Writes results.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "Training/rocm-9070"), str(HERE)]

from mlp_trades.bitfinex import apply_risk_target, export_paths, prepare_bitfinex  # noqa: E402
from mlp_trades.data import mmap_array  # noqa: E402
from mlp_trades.schema import WIDTH  # noqa: E402
from mlp_trades.split import training_split  # noqa: E402
from trade_baselines import HistogramGBT, decision_features  # noqa: E402

TOPS = (500, 1000, 2000)


def score_split(model, x, net, labels):
    score = model.predict_proba(x)[:, 1]
    order = np.argsort(-score, kind="stable")
    out = {"n": len(net), "all_net": float(net.mean())}
    for k in TOPS:
        top = order[:min(k, len(order))]
        out[f"top{k}_net"] = float(net[top].mean())
    top = order[:min(1000, len(order))]
    out["top1000_pos"] = int(labels[top].sum())
    out["top1000_pos_random"] = float(labels.mean() * len(top))
    return out


def walk_forward(data, start, months=3):
    """Expanding-window folds: (train, validation, test) index arrays per 3-month test block."""
    times = pd.to_datetime(data.times, unit="s", utc=True)
    valid = data.valid
    edge = pd.Timestamp(start, tz="UTC")
    last = times[valid].max()
    folds = []
    while edge < last:
        val_start, test_end = edge - pd.DateOffset(months=months), edge + pd.DateOffset(months=months)
        t = times[valid]
        test = valid[(t >= edge) & (t < test_end)]
        val = valid[(t >= val_start) & (t < edge)]
        train = valid[t < val_start]
        # Purge: an earlier part may not hold entries whose outcome is known only after the next part starts.
        train = train[data.ends[train] < int(val_start.timestamp())]
        val = val[data.ends[val] < int(edge.timestamp())]
        if len(test) >= 2000 and len(val) and len(train):
            folds.append((f"{edge:%Y-%m}", train, val, test))
        edge = test_end
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exports", nargs="+", default=["current_rules_15m", "current_rules_5m"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 1, 2, 3, 4])
    parser.add_argument("--wf-start", default="2025-05-01")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/training/gbt_robustness")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.exports:
        directory = ROOT / "Exports/training" / f"mlp_2023-11-13_to_2026-11-13_{name}"
        base = prepare_bitfinex(export_paths(directory), training=True)
        manifest = (directory / "manifest.json").read_text(encoding="utf-8")
        timeline = mmap_array(str(directory / "feature_timeline.npy"), "timeline", (WIDTH,))
        bounded = '"x / sqrt(1 + x^2)"' in manifest
        features, _ = decision_features(timeline, base.history_rows[base.valid], base.times[base.valid], bounded)
        x = np.zeros((len(base.ids), features.shape[1]))
        x[base.valid] = features
        net = base.returns
        targets = [("net >= 0", base)] + [(f"r >= {k:g}", apply_risk_target(base, directory / "risk_targets.csv", k))
                                          for k in args.thresholds]
        splits, _ = training_split(base, 0.7, 0.15)
        schemes = [("standard", seed, "2026 split", splits["train"], splits["validation"], splits["test"])
                   for seed in args.seeds]
        schemes += [("walk-forward", 42, fold, tr, va, te) for fold, tr, va, te in walk_forward(base, args.wf_start)]
        for target, data in targets:
            y = data.labels
            for scheme, seed, fold, tr, va, te in schemes:
                model = HistogramGBT(2, seed=seed).fit(x[tr], y[tr], x[va], y[va])
                rec = {"export": name, "target": target, "scheme": scheme, "seed": seed, "fold": fold,
                       "rounds": model.rounds, "train": len(tr)}
                for part, rows in (("validation", va), ("test", te)):
                    rec.update({f"{part}.{k}": v for k, v in score_split(model, x[rows], net[rows], y[rows]).items()})
                records.append(rec)
                print(name, target, scheme, seed, fold, model.rounds,
                      f"val top1000 {rec['validation.top1000_net']:+.4f} (all {rec['validation.all_net']:+.4f})",
                      f"test top1000 {rec['test.top1000_net']:+.4f} (all {rec['test.all_net']:+.4f})", flush=True)
                pd.DataFrame(records).to_csv(args.output / "results.csv", index=False)


if __name__ == "__main__":
    main()
