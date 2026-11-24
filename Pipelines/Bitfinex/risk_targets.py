"""Risk-unit targets: the series net result measured in units of volatility at the decision time.

    python Pipelines/Bitfinex/risk_targets.py --export Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m

    r_atr_m15 = series net return (%) / ATR 14 on M15 at the decision minute (% of close)

The risk unit is ATR on M15 because the stop trails on 15m candles. The sign matches the net label, so
the unit matters through thresholds (class 1 when r_atr_m15 >= k) or as a regression target.

Writes risk_targets.csv (entry_id, entry_at, atr_m15_pct, net_pct, r_atr_m15) into the export, which the
trainer reads with --target risk_units. With --evaluate it also checks what the targets give, per period
of the trainer's split: base rates, the best single features by AUC, and a logistic regression on all
decision-time features (needs the parquet files from feature_separation.py and volatility_study.py),
scored by AUC and by the mean net of its top 20% of test entries.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "Training/rocm-9070"), str(ROOT / "Models/Baselines"), str(HERE)]

PERIODS = ("train", "validation", "test")
THRESHOLDS = (0.0, 0.5, 1.0, 2.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m")
    parser.add_argument("--volatility", type=Path, default=ROOT / "Exports/training/volatility_study/entry_volatility.parquet")
    parser.add_argument("--features", type=Path, default=ROOT / "Exports/training/feature_separation/entry_features.parquet")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/training/risk_targets")
    args = parser.parse_args()
    # The net result comes from this export (its own stop and exit rules); ATR 14 on M15 is read from the
    # native candles closed at each entry's decision time.
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("bitfinex_indicators_v2_1", HERE / "bitfinex_indicators_v2.1.py")
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    from volatility_features import _at

    def offline(*_a, **_k):
        raise RuntimeError("Use the existing candle cache; a missing cache file would need the API")
    manifest = json.loads((args.export / "manifest.json").read_text(encoding="utf-8"))
    pipeline.request_candles = offline
    native, _ = pipeline.collect_training_history(pd.Timestamp(manifest["start_inclusive"]),
                                                  pd.Timestamp(manifest["snapshot_cutoff"]),
                                                  ROOT / "Exports/candle_cache", manifest["window_minutes"])
    m15 = pipeline.calculate_native(native["15m"])
    entries = pd.read_csv(args.export / "entry_targets.csv", dtype={"entry_id": str},
                          usecols=["entry_id", "entry_at", "net_return_pct", "initial_order_status"])
    out = entries[entries.initial_order_status.eq("filled")].drop(columns="initial_order_status").copy()
    atr = 100 * m15["atr_14"].astype(float) / m15.close.astype(float)
    out["atr_m15_pct"] = _at(m15, atr, pd.to_datetime(out.entry_at, utc=True))
    vol = pd.read_parquet(args.volatility) if args.evaluate else None
    out = out.rename(columns={"net_return_pct": "net_pct"})
    out["r_atr_m15"] = out.net_pct / out.atr_m15_pct
    if out.atr_m15_pct.le(0).any() or out.r_atr_m15.isna().any():
        raise SystemExit("Nonpositive ATR or missing risk target")
    out.to_csv(args.export / "risk_targets.csv", index=False)
    print(f"{args.export / 'risk_targets.csv'}: {len(out):,} entries")
    if not args.evaluate:
        return

    from feature_separation import auc
    from trade_baselines import fit_logistic, logistic_probabilities

    args.output.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(args.features)
    features["entry_id"] = features.entry_id.astype(str)
    frame = features.merge(vol.drop(columns=["split", "time", "win", "net", "wins"]), on="entry_id")
    frame["r"] = frame.entry_id.map(out.set_index("entry_id").r_atr_m15)
    frame["net"] = frame.entry_id.map(out.set_index("entry_id").net_pct)
    names = [c for c in frame.columns if c not in ("entry_id", "split", "entry_time", "export_label", "wins", "r", "net")]
    x_all = frame[names].to_numpy(dtype=float)
    x_all = np.where(np.isfinite(x_all), x_all, np.nan)
    medians = np.nanmedian(x_all[frame.split.eq("train").to_numpy()], axis=0)
    x_all = np.where(np.isnan(x_all), medians, x_all)
    rows = {p: frame.split.eq(p).to_numpy() for p in PERIODS}
    base = []
    for k in THRESHOLDS:
        base.append({"target": f"r >= {k:g}", **{p: float((frame.r[rows[p]] >= k).mean()) for p in PERIODS}})
    pd.DataFrame(base).to_csv(args.output / "base_rates.csv", index=False)

    single = []
    for j, name in enumerate(names):
        rec = {"feature": name}
        for k in (0.0, 1.0):
            for p in PERIODS:
                rec[f"r>={k:g}.{p}"] = auc(x_all[rows[p], j], (frame.r[rows[p]] >= k).to_numpy())[0]
        for p in PERIODS:
            rec[f"spearman.{p}"] = float(pd.Series(x_all[rows[p], j]).rank().corr(frame.r[rows[p]].rank().reset_index(drop=True)))
        single.append(rec)
    single = pd.DataFrame(single)
    for k in ("0", "1"):
        cols = [f"r>={k}.{p}" for p in PERIODS]
        same = np.sign(single[cols] - 0.5).nunique(axis=1).eq(1)
        single[f"r>={k}.weakest_edge"] = np.where(same, (single[cols] - 0.5).abs().min(axis=1), 0)
    single = single.sort_values("r>=1.weakest_edge", ascending=False)
    single.to_csv(args.output / "single_features.csv", index=False)

    mean, scale = x_all[rows["train"]].mean(0), x_all[rows["train"]].std(0)
    scale[scale < 1e-12] = 1
    z = (x_all - mean) / scale
    models = []
    for k in THRESHOLDS:
        y = (frame.r >= k).to_numpy().astype(np.int64)
        best = None
        for l2 in (1e-2, 1e-1, 1.0, 10.0, 100.0):
            w = fit_logistic(z[rows["train"]], y[rows["train"]], 2, l2)
            a = auc(logistic_probabilities(w, z[rows["validation"]])[:, 1], y[rows["validation"]] == 1)[0]
            if best is None or a > best[0]:
                best = (a, l2, w)
        _, l2, w = best
        rec = {"target": f"r >= {k:g}", "l2": l2}
        for p in PERIODS:
            score = logistic_probabilities(w, z[rows[p]])[:, 1]
            rec[f"auc.{p}"] = auc(score, y[rows[p]] == 1)[0]
            top = score >= np.quantile(score, 0.8)
            net = frame.net.to_numpy()[rows[p]]
            rec[f"top20_net.{p}"] = float(net[top].mean())
            rec[f"all_net.{p}"] = float(net.mean())
            rec[f"top20_win.{p}"] = float((net[top] > 0).mean())
        models.append(rec)
    models = pd.DataFrame(models)
    models.to_csv(args.output / "logistic.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(base).round(3).to_string(index=False))
    print(single.head(15).round(4).to_string(index=False))
    print(models.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
