"""Do the old no-trade windows (rules E, F, H) hold over three years of entries?

    python Pipelines/Bitfinex/window_rules.py --export Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m

Rules, on closed candles (an entry sees the state of the last candle closed at its decision time):
    E  1h ADX 14/4 crosses 25 upwards while DI+ > 25. The window stays closed until ADX or DI+ crosses 25 down.
    F  M30 RSI 14 crosses 60 upwards. The window stays closed until RSI crosses 50 down.
    H  E or F active.
Also E on M30 ("30M ADX" in the old table) and E(M30) or E(1h). 1h candles are built from native M30 candles.

For each rule and period (the trainer's chronological train / validation / test split) it compares entries
inside the window with entries outside: series win rate and mean series net (export labels, fees included),
and the share that win in all 15 exit configurations. Writes windows.csv and windows.json to --output.
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
sys.path[:0] = [str(ROOT / "Training/rocm-9070")]
_spec = importlib.util.spec_from_file_location("bitfinex_indicators_v2_1", HERE / "bitfinex_indicators_v2.1.py")
pipeline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pipeline
_spec.loader.exec_module(pipeline)

from mlp_trades.bitfinex import export_paths, prepare_bitfinex  # noqa: E402
from mlp_trades.split import training_split  # noqa: E402


def offline(*_args, **_kwargs):
    raise RuntimeError("Use the existing candle cache; a missing cache file would need the API")


def hourly(m30):
    """1h candles from native M30 candles, aligned to hour openings."""
    frame = m30.set_index(pd.to_datetime(m30.timestamp, utc=True)).sort_index()
    out = frame.resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = frame.close.resample("1h", label="left", closed="left").count()
    out = out[counts == 2].reset_index().rename(columns={"index": "timestamp"})
    out["timestamp"] = out.pop(out.columns[0]) if "timestamp" not in out else out.timestamp
    out["available_at"] = out.timestamp + pd.Timedelta(hours=1)
    return out[[c for c in m30.columns if c in out.columns]]


def windows(cross_on, cross_off_list):
    """Closed from a candle where cross_on is true until a candle where any cross_off is true."""
    state, active = False, np.zeros(len(cross_on), dtype=bool)
    off = np.logical_or.reduce(cross_off_list)
    for i, (start, stop) in enumerate(zip(cross_on, off)):
        if state and stop:
            state = False
        elif not state and start:
            state = True
        active[i] = state
    return active


def cross_up(series, level):
    return (series.shift(1) <= level) & (series > level)


def cross_down(series, level):
    return (series.shift(1) >= level) & (series < level)


def rule_e(native):
    adx, plus = native["adx_14_4"], native["di_plus_14"]
    on = (cross_up(adx, 25) & (plus > 25)).to_numpy()
    return windows(on, [cross_down(adx, 25).to_numpy(), cross_down(plus, 25).to_numpy()])


def rule_f(native):
    rsi = native["rsi_14"]
    return windows(cross_up(rsi, 60).to_numpy(), [cross_down(rsi, 50).to_numpy()])


def state_at(native, active, times):
    """Window state of the last candle closed at or before each decision time."""
    closes = pd.to_datetime(native.available_at, utc=True).to_numpy()
    idx = np.searchsorted(closes, times.to_numpy(), side="right") - 1
    return np.where(idx >= 0, active[np.clip(idx, 0, None)], False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/training/window_rules")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.export / "manifest.json").read_text(encoding="utf-8"))
    data = prepare_bitfinex(export_paths(args.export), training=True)
    splits, _ = training_split(data, 0.7, 0.15)
    rows = np.concatenate(list(splits.values()))
    split = np.concatenate([[name] * len(r) for name, r in splits.items()])
    frame = pd.DataFrame({"entry_id": data.ids[rows], "split": split,
                          "time": pd.to_datetime(data.times[rows], unit="s", utc=True),
                          "win": data.labels[rows], "net": data.returns[rows]})
    agreement = pd.read_csv(args.export / "label_agreement.csv", dtype={"entry_id": str}).set_index("entry_id")
    frame["wins"] = frame.entry_id.map(agreement.wins)
    total = int(agreement.configurations.iloc[0])

    start, cutoff = pd.Timestamp(manifest["start_inclusive"]), pd.Timestamp(manifest["snapshot_cutoff"])
    pipeline.request_candles = offline
    native, _ = pipeline.collect_training_history(start, cutoff, ROOT / "Exports/candle_cache", manifest["window_minutes"])
    m30 = pipeline.calculate_native(native["30m"])
    h1 = pipeline.calculate_native(hourly(native["30m"]))
    rules = {"E (1h ADX)": state_at(h1, rule_e(h1), frame.time),
             "F (M30 RSI 60+)": state_at(m30, rule_f(m30), frame.time),
             "E on M30 (M30 ADX)": state_at(m30, rule_e(m30), frame.time)}
    rules["H (E or F)"] = rules["E (1h ADX)"] | rules["F (M30 RSI 60+)"]
    rules["E on M30 or E on 1h"] = rules["E on M30 (M30 ADX)"] | rules["E (1h ADX)"]
    old = (frame.time >= "2026-08-01") & (frame.time < "2026-09-01")

    def stats(mask, inside):
        a, b = frame[mask & inside], frame[mask & ~inside]
        robust_a, robust_b = a[a.wins.isin([0, total])], b[b.wins.isin([0, total])]
        return {"entries": int(mask.sum()), "inside": int(len(a)), "share_inside": float(len(a) / max(mask.sum(), 1)),
                "win_inside": float(a.win.mean()) if len(a) else np.nan,
                "win_outside": float(b.win.mean()) if len(b) else np.nan,
                "net_inside": float(a.net.mean()) if len(a) else np.nan,
                "net_outside": float(b.net.mean()) if len(b) else np.nan,
                "net_all": float(frame[mask].net.mean()),
                "net_inside_se": float(a.net.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else np.nan,
                "unanimous_win_inside": float(robust_a.wins.eq(total).mean()) if len(robust_a) else np.nan,
                "unanimous_win_outside": float(robust_b.wins.eq(total).mean()) if len(robust_b) else np.nan}
    records = []
    for name, active in rules.items():
        inside = pd.Series(active, index=frame.index)
        for period, mask in [("all", pd.Series(True, index=frame.index))] + \
                [(s, frame.split.eq(s)) for s in ("train", "validation", "test")] + [("august_2026", old)]:
            records.append({"rule": name, "period": period, **stats(mask, inside)})
    table = pd.DataFrame(records)
    table.to_csv(args.output / "windows.csv", index=False)
    (args.output / "windows.json").write_text(json.dumps(records, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    show = table[["rule", "period", "entries", "share_inside", "win_inside", "win_outside", "net_inside",
                  "net_outside", "net_all", "net_inside_se", "unanimous_win_inside", "unanimous_win_outside"]]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
