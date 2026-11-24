"""Which single features separate winning from losing entries, and does it hold across periods?

    python Pipelines/Bitfinex/feature_separation.py --export Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m

Features per filled entry, all known at the decision minute:
- the 85 decision-time features the CPU baselines use (indicators on M1/M5/M15/M30, returns,
  realized volatility, hour and weekday, the 18 main anchors now and in the last 120 minutes);
- all 35 anchors of the registry (main and additional) on the entry's closed M5 candle, and whether
  each fired on any of the previous 12 closed M5 candles (one hour), recomputed from native candles
  with the collector's own detect_anchor_events.

Targets:
- robust: entries every one of the 15 exit configurations wins (label_agreement.csv wins == 15)
  against entries every configuration loses (wins == 0); the entries in between are left out;
- export: the export's own binary label on every filled entry.

For each feature, target and split (the trainer's chronological 70/15/15 split with purging) it reports
the ROC AUC of the feature as a score (0.5 = no separation) with its standard error, and for flags
the class-1 rate with and without the flag. Writes separation.csv and summary.json to --output.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "Training/rocm-9070"), str(ROOT / "Models/Baselines")]
_spec = importlib.util.spec_from_file_location("bitfinex_indicators_v2_1", HERE / "bitfinex_indicators_v2.1.py")
pipeline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pipeline
_spec.loader.exec_module(pipeline)

from mlp_trades.bitfinex import export_paths, prepare_bitfinex  # noqa: E402
from mlp_trades.data import mmap_array  # noqa: E402
from mlp_trades.schema import WIDTH  # noqa: E402
from mlp_trades.split import training_split  # noqa: E402
from trade_baselines import decision_features  # noqa: E402

LOOKBACK_CANDLES = 12


def offline(*_args, **_kwargs):
    raise RuntimeError("Use the existing candle cache; a missing cache file would need the API")


def auc(score, label):
    """ROC AUC with average ranks for ties, and the Hanley-McNeil standard error."""
    pos, neg = int(label.sum()), int((~label).sum())
    if not pos or not neg or np.all(score == score[0]):
        return 0.5, np.nan
    ranks = pd.Series(score).rank().to_numpy()
    a = (ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (pos - 1) * (q1 - a * a) + (neg - 1) * (q2 - a * a)) / (pos * neg))
    return float(a), float(se)


def anchor_groups():
    groups = {n: "main" for n in pipeline.MAIN_ANCHOR_IDS}
    for key, (_, ids) in pipeline.ADDITIONAL_ANCHOR_GROUPS.items():
        groups.update({n: key for n in ids})
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/training/feature_separation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.export / "manifest.json").read_text(encoding="utf-8"))
    anchor_tf = manifest.get("anchor_timeframe", "5m")

    # Entries, labels and splits exactly as the trainer sees them.
    data = prepare_bitfinex(export_paths(args.export), training=True)
    splits, _ = training_split(data, 0.7, 0.15)
    split_of = np.full(len(data.ids), "", dtype=object)
    for name, rows in splits.items():
        split_of[rows] = name
    rows = np.concatenate(list(splits.values()))
    timeline = mmap_array(str(args.export / "feature_timeline.npy"), "timeline", (WIDTH,))
    bounded = manifest.get("macd_normalization") == "x / sqrt(1 + x^2)"
    base, base_names = decision_features(timeline, data.history_rows[rows], data.times[rows], bounded)
    frame = pd.DataFrame(base, columns=base_names)
    frame.insert(0, "entry_id", data.ids[rows])
    frame.insert(1, "split", split_of[rows])
    frame.insert(2, "entry_time", pd.to_datetime(data.times[rows], unit="s", utc=True))
    frame["export_label"] = data.labels[rows]

    agreement = pd.read_csv(args.export / "label_agreement.csv", dtype={"entry_id": str}).set_index("entry_id")
    frame["wins"] = frame.entry_id.map(agreement.wins)
    total = int(agreement.configurations.iloc[0])

    # All 35 anchors from native candles with the collector's rules.
    start, cutoff = pd.Timestamp(manifest["start_inclusive"]), pd.Timestamp(manifest["snapshot_cutoff"])
    pipeline.request_candles = offline
    native, _ = pipeline.collect_training_history(start, cutoff, ROOT / "Exports/candle_cache", manifest["window_minutes"])
    flags = pipeline.detect_anchor_events(pipeline.calculate_native(native[anchor_tf]), anchor_tf)
    anchor_columns = [c for c in flags.columns if c.startswith("anchor_")]
    recent = flags[anchor_columns].shift(1).rolling(LOOKBACK_CANDLES, min_periods=1).max()
    now = flags[anchor_columns].reindex(pd.DatetimeIndex(frame.entry_time))
    if now.isna().any().any():
        raise SystemExit("Some entries have no anchor row at their decision time")
    main_now = now[[f"anchor_{n:02d}" for n in pipeline.MAIN_ANCHOR_IDS]].to_numpy()
    if not main_now.any(axis=1).all():
        raise SystemExit("Recomputed anchors disagree with the export: an entry has no main anchor")
    groups = anchor_groups()
    for column in anchor_columns:
        n = int(column.split("_")[1])
        frame[f"{column}.now"] = now[column].to_numpy(dtype=float)
        frame[f"{column}.last_{LOOKBACK_CANDLES * 5}m"] = recent[column].reindex(now.index).fillna(0).to_numpy(dtype=float)
    frame["anchors_now_count"] = now.sum(axis=1).to_numpy(dtype=float)
    frame.to_parquet(args.output / "entry_features.parquet")

    features = [c for c in frame.columns if c not in ("entry_id", "split", "entry_time", "export_label", "wins")]
    targets = {"robust": (frame.wins.isin([0, total]), frame.wins.eq(total)),
               "export": (pd.Series(True, index=frame.index), frame.export_label.eq(1))}
    records = []
    for feature in features:
        values = frame[feature].to_numpy(dtype=float)
        binary = set(np.unique(values)) <= {0.0, 1.0}
        origin = "anchor" if feature.startswith("anchor_") and feature.count(".") == 1 and feature[7:9].isdigit() \
            and feature.split(".")[1] in ("now", f"last_{LOOKBACK_CANDLES * 5}m") and "anchor_" in feature else "decision"
        number = int(feature[7:9]) if origin == "anchor" else None
        record = {"feature": feature, "binary": binary,
                  "anchor_group": groups.get(number) if number is not None else None}
        for target, (keep, positive) in targets.items():
            for split in ("train", "validation", "test"):
                mask = (frame.split.eq(split) & keep).to_numpy()
                y, x = positive.to_numpy()[mask], values[mask]
                a, se = auc(x, y)
                record[f"{target}.{split}.auc"], record[f"{target}.{split}.se"] = a, se
                if binary:
                    on = x == 1
                    record[f"{target}.{split}.share_on"] = float(on.mean())
                    record[f"{target}.{split}.rate_on"] = float(y[on].mean()) if on.any() else np.nan
                    record[f"{target}.{split}.rate_off"] = float(y[~on].mean()) if (~on).any() else np.nan
        records.append(record)
    table = pd.DataFrame(records)
    for target in targets:
        # Consistent separation: the same side of 0.5 in all three periods, sized by the weakest period.
        sides = np.sign(table[[f"{target}.{s}.auc" for s in ("train", "validation", "test")]] - 0.5)
        weakest = (table[[f"{target}.{s}.auc" for s in ("train", "validation", "test")]] - 0.5).abs().min(axis=1)
        table[f"{target}.consistent"] = sides.nunique(axis=1).eq(1) & sides.iloc[:, 0].ne(0)
        table[f"{target}.weakest_edge"] = np.where(table[f"{target}.consistent"], weakest, 0.0)
        z = (table[f"{target}.test.auc"] - 0.5) / table[f"{target}.test.se"]
        table[f"{target}.test.z"] = z
    table = table.sort_values("robust.weakest_edge", ascending=False)
    table.to_csv(args.output / "separation.csv", index=False)
    counts = {split: {"robust_entries": int((frame.split.eq(split) & targets["robust"][0]).sum()),
                      "robust_win_rate": float(frame.loc[frame.split.eq(split) & targets["robust"][0], "wins"].eq(total).mean()),
                      "export_entries": int(frame.split.eq(split).sum()),
                      "export_win_rate": float(frame.loc[frame.split.eq(split), "export_label"].mean())}
              for split in ("train", "validation", "test")}
    summary = {"export": str(args.export), "anchor_timeframe": anchor_tf, "features": len(features),
               "configurations": total, "counts": counts,
               "periods": {s: [str(frame.loc[frame.split.eq(s), "entry_time"].min()),
                               str(frame.loc[frame.split.eq(s), "entry_time"].max())] for s in ("train", "validation", "test")}}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    show = ["feature", "robust.train.auc", "robust.validation.auc", "robust.test.auc", "robust.test.z", "export.test.auc"]
    print(table[show].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
