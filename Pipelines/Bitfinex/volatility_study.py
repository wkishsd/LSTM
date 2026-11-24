"""How well do the volatility measures sort entries, and does skipping calm markets pay?

    python Pipelines/Bitfinex/volatility_study.py --export Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m

For each measure in volatility_features.py and each period of the trainer's chronological split:
- ROC AUC for unanimous winners vs unanimous losers (label_agreement.csv) and for the export label;
- quintiles with edges fixed on the train period: entries, export win rate, mean series net, unanimous win share;
- a skip filter: drop entries below the train 20th / 40th / 60th percentile; mean net of the kept entries.
Writes auc.csv, quintiles.csv, filters.csv and summary.json to --output.
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
sys.path[:0] = [str(ROOT / "Training/rocm-9070"), str(HERE)]
_spec = importlib.util.spec_from_file_location("bitfinex_indicators_v2_1", HERE / "bitfinex_indicators_v2.1.py")
pipeline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pipeline
_spec.loader.exec_module(pipeline)

from feature_separation import auc  # noqa: E402
from mlp_trades.bitfinex import export_paths, prepare_bitfinex  # noqa: E402
from mlp_trades.split import training_split  # noqa: E402
from volatility_features import volatility_features  # noqa: E402

PERIODS = ("train", "validation", "test")


def offline(*_args, **_kwargs):
    raise RuntimeError("Use the existing candle cache; a missing cache file would need the API")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/training/volatility_study")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.export / "manifest.json").read_text(encoding="utf-8"))
    data = prepare_bitfinex(export_paths(args.export), training=True)
    splits, _ = training_split(data, 0.7, 0.15)
    rows = np.concatenate([splits[s] for s in PERIODS])
    frame = pd.DataFrame({"entry_id": data.ids[rows],
                          "split": np.concatenate([[s] * len(splits[s]) for s in PERIODS]),
                          "time": pd.to_datetime(data.times[rows], unit="s", utc=True),
                          "win": data.labels[rows], "net": data.returns[rows]})
    agreement = pd.read_csv(args.export / "label_agreement.csv", dtype={"entry_id": str}).set_index("entry_id")
    frame["wins"] = frame.entry_id.map(agreement.wins).to_numpy()
    total = int(agreement.configurations.iloc[0])
    pipeline.request_candles = offline
    native, _ = pipeline.collect_training_history(pd.Timestamp(manifest["start_inclusive"]),
                                                  pd.Timestamp(manifest["snapshot_cutoff"]),
                                                  ROOT / "Exports/candle_cache", manifest["window_minutes"])
    features = volatility_features(pipeline, native, frame.time)
    frame = pd.concat([frame, features], axis=1)
    frame.to_parquet(args.output / "entry_volatility.parquet")
    names = list(features.columns)
    robust = frame.wins.isin([0, total])

    auc_rows, quintile_rows, filter_rows = [], [], []
    for name in names:
        values = frame[name]
        record = {"measure": name, "missing": int(values.isna().sum())}
        for period in PERIODS:
            m = frame.split.eq(period) & values.notna()
            a, se = auc(values[m & robust].to_numpy(), frame.wins[m & robust].eq(total).to_numpy())
            b, _ = auc(values[m].to_numpy(), frame.win[m].eq(1).to_numpy())
            record.update({f"robust.{period}": a, f"robust.{period}.se": se, f"export.{period}": b})
        auc_rows.append(record)
        train = values[frame.split.eq("train")].dropna()
        edges = np.quantile(train, [0.2, 0.4, 0.6, 0.8])
        bucket = np.searchsorted(edges, values.to_numpy(), side="right")
        for period in PERIODS:
            m = frame.split.eq(period) & values.notna()
            for q in range(5):
                part = frame[m & (bucket == q)]
                rob = part[part.wins.isin([0, total])]
                quintile_rows.append({"measure": name, "period": period, "quintile": q + 1, "entries": len(part),
                                      "share": len(part) / max(int(m.sum()), 1), "win_rate": part.win.mean(),
                                      "mean_net": part.net.mean(),
                                      "unanimous_win_share": rob.wins.eq(total).mean() if len(rob) else np.nan})
            for cut in (0.2, 0.4, 0.6):
                threshold = np.quantile(train, cut)
                kept = frame[m & (values >= threshold)]
                filter_rows.append({"measure": name, "period": period, "skip_below_train_pct": cut,
                                    "threshold": threshold, "kept_share": len(kept) / max(int(m.sum()), 1),
                                    "kept_mean_net": kept.net.mean(), "all_mean_net": frame[m].net.mean(),
                                    "kept_win_rate": kept.win.mean(), "all_win_rate": frame[m].win.mean()})
    auc_table = pd.DataFrame(auc_rows)
    sides = np.sign(auc_table[[f"robust.{p}" for p in PERIODS]] - 0.5)
    auc_table["consistent"] = sides.nunique(axis=1).eq(1)
    auc_table["weakest_edge"] = np.where(auc_table.consistent,
                                         (auc_table[[f"robust.{p}" for p in PERIODS]] - 0.5).abs().min(axis=1), 0)
    auc_table = auc_table.sort_values("weakest_edge", ascending=False)
    auc_table.to_csv(args.output / "auc.csv", index=False)
    pd.DataFrame(quintile_rows).to_csv(args.output / "quintiles.csv", index=False)
    filters = pd.DataFrame(filter_rows)
    filters["gain"] = filters.kept_mean_net - filters.all_mean_net
    filters.to_csv(args.output / "filters.csv", index=False)
    pd.set_option("display.width", 250)
    print(auc_table.round(4).to_string(index=False))
    best = auc_table.measure.iloc[:4].tolist()
    print(filters[filters.measure.isin(best)].round(4).to_string(index=False))
    q = pd.DataFrame(quintile_rows)
    print(q[q.measure.isin(best[:2])].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
