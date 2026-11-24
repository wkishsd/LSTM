"""Readable entry rules from a shallow decision tree on the risk-unit target.

    python Models/Baselines/tree_rules.py --export current_rules_15m --threshold 1.0 --depth 3

Target: class 1 when the series net / ATR 14 on M15 >= --threshold (risk_targets.csv). Features: the 85
decision-time features of trade_baselines.py, with hour (UTC) and weekday (Monday = 0) as plain numbers
instead of sine/cosine pairs so every rule reads directly.

The tree splits on quantile thresholds by Gini gain, with at least --min-leaf of the train entries per leaf.
It is fitted on the trainer's train split (2023-2025); every leaf becomes a rule, scored on validation and
test by entries, class-1 rate and mean series net in % after fees. It is then refitted in each 3-month
walk-forward block to see which splits come back. Writes rules.csv, walk_forward.csv and rules.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "Training/rocm-9070"), str(HERE)]

from gbt_robustness import walk_forward  # noqa: E402
from mlp_trades.bitfinex import apply_risk_target, export_paths, prepare_bitfinex  # noqa: E402
from mlp_trades.data import mmap_array  # noqa: E402
from mlp_trades.schema import WIDTH  # noqa: E402
from mlp_trades.split import training_split  # noqa: E402
from trade_baselines import decision_features  # noqa: E402

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def gini(y):
    p = y.mean() if len(y) else 0.0
    return 2 * p * (1 - p)


def grow(x, y, rows, depth, min_leaf, path=()):
    """Nodes as dicts; leaves carry their path of (feature index, '<=' or '>', threshold)."""
    if depth == 0 or len(rows) < 2 * min_leaf:
        return {"leaf": True, "path": path, "rows": rows}
    best = None
    parent = gini(y[rows]) * len(rows)
    for j in range(x.shape[1]):
        values = x[rows, j]
        for t in np.unique(np.quantile(values, np.linspace(0.05, 0.95, 19))):
            left = values <= t
            n_left = int(left.sum())
            if n_left < min_leaf or len(rows) - n_left < min_leaf:
                continue
            gain = parent - gini(y[rows[left]]) * n_left - gini(y[rows[~left]]) * (len(rows) - n_left)
            if best is None or gain > best[0]:
                best = (gain, j, float(t), left)
    if best is None or best[0] <= 0:
        return {"leaf": True, "path": path, "rows": rows}
    _, j, t, left = best
    return {"leaf": False, "feature": j, "threshold": t,
            "left": grow(x, y, rows[left], depth - 1, min_leaf, path + ((j, "<=", t),)),
            "right": grow(x, y, rows[~left], depth - 1, min_leaf, path + ((j, ">", t),))}


def leaves(node):
    return [node] if node["leaf"] else leaves(node["left"]) + leaves(node["right"])


def describe(path, names):
    parts = []
    for j, op, t in path:
        name = names[j]
        if name == "weekday":
            days = [WEEKDAYS[d] for d in range(7) if (d <= t if op == "<=" else d > t)]
            parts.append("weekday in " + "/".join(days))
        elif name == "hour_utc":
            parts.append(f"hour UTC {'<=' if op == '<=' else '>'} {t:g}")
        else:
            parts.append(f"{name} {op} {t:.4g}")
    return " AND ".join(parts) or "all entries"


def matches(x, path):
    mask = np.ones(len(x), dtype=bool)
    for j, op, t in path:
        mask &= x[:, j] <= t if op == "<=" else x[:, j] > t
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", default="current_rules_15m")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--min-leaf", type=float, default=0.03, help="Minimum leaf size as a share of train")
    parser.add_argument("--wf-start", default="2025-05-01")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/training/tree_rules")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    directory = ROOT / "Exports/training" / f"mlp_2023-11-13_to_2026-11-13_{args.export}"
    base = prepare_bitfinex(export_paths(directory), training=True)
    data = apply_risk_target(base, directory / "risk_targets.csv", args.threshold)
    timeline = mmap_array(str(directory / "feature_timeline.npy"), "timeline", (WIDTH,))
    bounded = '"x / sqrt(1 + x^2)"' in (directory / "manifest.json").read_text(encoding="utf-8")
    feats, names = decision_features(timeline, data.history_rows[data.valid], data.times[data.valid], bounded)
    keep = [i for i, n in enumerate(names) if not n.startswith(("hour_", "weekday_"))]
    times = data.times[data.valid]
    x_valid = np.column_stack([feats[:, keep], (times % 86400) // 3600, ((times // 86400) + 3) % 7])
    names = [names[i] for i in keep] + ["hour_utc", "weekday"]
    x = np.zeros((len(data.ids), x_valid.shape[1]))
    x[data.valid] = x_valid
    y, net = data.labels, data.returns

    splits, _ = training_split(data, 0.7, 0.15)
    min_leaf = int(args.min_leaf * len(splits["train"]))
    tree = grow(x, y, splits["train"], args.depth, min_leaf)
    records, text = [], []
    for leaf in leaves(tree):
        rule = describe(leaf["path"], names)
        rec = {"rule": rule}
        for part, rows in splits.items():
            m = matches(x[rows], leaf["path"])
            rec.update({f"{part}.entries": int(m.sum()), f"{part}.share": float(m.mean()),
                        f"{part}.class1": float(y[rows][m].mean()) if m.any() else np.nan,
                        f"{part}.net": float(net[rows][m].mean()) if m.any() else np.nan,
                        f"{part}.all_net": float(net[rows].mean())})
        records.append(rec)
    table = pd.DataFrame(records).sort_values("train.class1", ascending=False)
    table.to_csv(args.output / "rules.csv", index=False)

    folds = []
    for fold, tr, va, te in walk_forward(data, args.wf_start):
        t = grow(x, y, tr, args.depth, int(args.min_leaf * len(tr)))
        best = max(leaves(t), key=lambda leaf: y[leaf["rows"]].mean())
        m = matches(x[te], best["path"])
        folds.append({"fold": fold, "root_split": describe(((t["feature"], "<=", t["threshold"]),), names)
                      if not t["leaf"] else "none", "best_leaf": describe(best["path"], names),
                      "train_class1": float(y[best["rows"]].mean()), "test_entries": int(m.sum()),
                      "test_net": float(net[te][m].mean()) if m.any() else np.nan, "test_all_net": float(net[te].mean())})
    folds = pd.DataFrame(folds)
    folds.to_csv(args.output / "walk_forward.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    show = ["rule", "train.entries", "train.class1", "train.net", "validation.entries", "validation.class1",
            "validation.net", "test.entries", "test.class1", "test.net"]
    print(table[show].round(4).to_string(index=False))
    print("all entries net: train %.4f  validation %.4f  test %.4f" % tuple(table.iloc[0][f"{p}.all_net"] for p in ("train", "validation", "test")))
    print(folds.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
