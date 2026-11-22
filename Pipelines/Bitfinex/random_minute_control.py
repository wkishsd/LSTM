"""Random-minute control: do main-anchor entries beat the same trade rules at random times?

Samples M5 decision times on which no main anchor fires, places the same initial short
stop order (previous closed M1 Low minus the offset), and labels each with the unchanged
exit simulator and fees. It then compares fill rate, winner rate and net return with the
main-anchor export over the same period, overall, by time period and by M5 ATR / close
band. If random times win as often as anchor times, the anchors add no edge for these exits.

Runs offline from the candle cache; it refuses to call the Bitfinex API.

    python Pipelines/Bitfinex/random_minute_control.py --samples 12000 --workers 6
"""

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PIPELINE = Path(__file__).resolve().with_name("bitfinex_indicators_v2.1.py")
if importlib.util.find_spec("openpyxl") is None:
    # The collector imports openpyxl only for its Excel export, which the control never writes.
    import types

    def _excel_unavailable(*_args, **_kwargs):
        raise RuntimeError("openpyxl is not installed; Excel export is unavailable in the control")

    _styles = types.ModuleType("openpyxl.styles")
    _styles.Alignment = _styles.Font = _excel_unavailable
    _utils = types.ModuleType("openpyxl.utils")
    _utils.get_column_letter = _excel_unavailable
    _openpyxl = types.ModuleType("openpyxl")
    _openpyxl.styles, _openpyxl.utils = _styles, _utils
    sys.modules.update({"openpyxl": _openpyxl, "openpyxl.styles": _styles, "openpyxl.utils": _utils})
# Registered at import time so spawned label workers can unpickle the pipeline's functions.
_spec = importlib.util.spec_from_file_location("bitfinex_indicators_v2_1", PIPELINE)
pipeline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pipeline
_spec.loader.exec_module(pipeline)


def offline(*_args, **_kwargs):
    raise RuntimeError("The control must use the existing candle cache; a missing cache file would need the API")


def settings_from(manifest, stop_timeframe=None):
    exit_settings = dict(manifest["exit_settings"])
    if stop_timeframe:
        exit_settings["stop_timeframe"] = stop_timeframe
    return pipeline.ExitSettings(**exit_settings)


def label(native, entries, settings, output, workers):
    audit = pipeline.label_mlp_entries(entries, native, settings, output, workers=workers).reindex(entries.index)
    audit["entry_at"] = entries.entry_at
    return audit


def atr_over_close(native, decision_times):
    five = pipeline.calculate_native(native["5m"]).set_index("available_at")
    rows = five.reindex(pd.DatetimeIndex(decision_times))
    return (rows[f"atr_{pipeline.ATR_PERIOD}"] / rows["close"]).to_numpy(dtype=float)


def summarize(frame):
    filled = frame.loc[frame.initial_order_status.eq("filled") & frame.binary_target.notna()]
    n = len(filled)
    rate = float(filled.binary_target.astype(int).mean()) if n else math.nan
    net = filled.net_return_pct.astype(float)
    return {
        "decisions": int(len(frame)),
        "fill_rate": float(frame.initial_order_status.eq("filled").mean()) if len(frame) else math.nan,
        "filled_labeled": n,
        "winner_rate": rate,
        "winner_rate_se": math.sqrt(rate * (1 - rate) / n) if n else math.nan,
        "net_return_pct_mean": float(net.mean()) if n else math.nan,
        "net_return_pct_se": float(net.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan,
        "gross_return_pct_mean": float(filled.target_gross_return_pct.astype(float).mean()) if n else math.nan,
        "trades_per_series": float(filled.trades_count.astype(float).mean()) if n else math.nan,
    }


def compare(anchor, control, edges, atr_edges):
    result = {"overall": {"anchor": summarize(anchor), "random": summarize(control)}, "periods": [], "atr_bands": []}
    for left, right in zip(edges, edges[1:]):
        part = lambda f: f.loc[(f.entry_at >= left) & (f.entry_at < right)]
        result["periods"].append({"from": left.isoformat(), "to": right.isoformat(),
                                  "anchor": summarize(part(anchor)), "random": summarize(part(control))})
    bounds = [-np.inf, *atr_edges, np.inf]
    for band, (low, high) in enumerate(zip(bounds, bounds[1:]), 1):
        part = lambda f: f.loc[(f.atr_over_close >= low) & (f.atr_over_close < high)]
        result["atr_bands"].append({"band": band, "atr_over_close_from": None if band == 1 else float(low),
                                    "anchor": summarize(part(anchor)), "random": summarize(part(control))})
    return result


def line(label, a, r):
    def fmt(s):
        return (f"win {s['winner_rate']:.3f}±{s['winner_rate_se']:.3f}  net {s['net_return_pct_mean']:+.3f}%"
                f"±{s['net_return_pct_se']:.3f}  fill {s['fill_rate']:.2f}  n {s['filled_labeled']:,}")
    return f"{label:<26} anchor: {fmt(a)}\n{'':<26} random: {fmt(r)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anchor-export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd")
    parser.add_argument("--history-cache", type=Path, default=ROOT / "Exports/candle_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/training/random_minute_control")
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--stop-timeframe", choices=pipeline.STOP_TIMEFRAMES, default=None,
                        help="Stop timeframe. When given, anchor entries are relabeled with the current exit "
                             "code too, so both sides use identical rules (the export may predate rule changes).")
    parser.add_argument("--anchor-samples", type=int, default=0,
                        help="With --stop-timeframe: relabel this many sampled anchor entries (0 = all)")
    args = parser.parse_args()

    manifest = json.loads((args.anchor_export / "manifest.json").read_text(encoding="utf-8"))
    settings = settings_from(manifest, args.stop_timeframe)
    settings.namespace()
    relabel = args.stop_timeframe is not None
    start, end, cutoff = (pd.Timestamp(manifest[k]) for k in ("start_inclusive", "end_exclusive", "snapshot_cutoff"))
    pipeline.request_candles = offline
    native, cutoff = pipeline.collect_training_history(start, cutoff, args.history_cache, manifest["window_minutes"])

    # Same decision clock and range as the anchor collector: M5 closes in [first, end).
    flags = pipeline.detect_anchor_events(pipeline.calculate_native(native["5m"]))
    first = max(native["1m"].timestamp.iloc[0] + pd.Timedelta(days=3), start)
    last = min(native["1m"].available_at.iloc[-1], end)
    flags = flags.loc[(flags.index >= first) & (flags.index < last)]
    main_columns = [f"anchor_{n:02d}" for n in pipeline.MAIN_ANCHOR_IDS]
    candidates = flags.loc[~flags[main_columns].any(axis=1)]
    rng = np.random.default_rng(args.seed)
    chosen = np.sort(rng.choice(len(candidates), size=min(args.samples, len(candidates)), replace=False))
    selected = candidates.iloc[chosen]
    print(f"{len(flags):,} M5 decision times, {len(candidates):,} without a main anchor; sampled {len(selected):,}",
          flush=True)

    entries = pipeline.entries_from_decisions(native, selected, settings.initial_entry_offset_down)
    args.output.mkdir(parents=True, exist_ok=True)
    audit = label(native, entries, settings, args.output, args.workers)
    audit["atr_over_close"] = atr_over_close(native, entries.entry_at)
    audit.to_csv(args.output / "random_entry_targets.csv")

    if relabel:
        # The export's labels used other exit rules; relabel the anchor entries the same way.
        anchor_entries, _ = pipeline.filter_main_anchor_entries(native, 0, start, end,
                                                                settings.initial_entry_offset_down)
        if 0 < args.anchor_samples < len(anchor_entries):
            keep = np.sort(rng.choice(len(anchor_entries), size=args.anchor_samples, replace=False))
            anchor_entries = anchor_entries.iloc[keep]
        print(f"Relabeling {len(anchor_entries):,} anchor entries with {settings.stop_timeframe} stops", flush=True)
        anchor = label(native, anchor_entries, settings, args.output, args.workers)
        anchor.to_csv(args.output / "anchor_entry_targets.csv")
    else:
        anchor = pd.read_csv(args.anchor_export / "entry_targets.csv", parse_dates=["entry_at"])
    anchor["atr_over_close"] = atr_over_close(native, anchor.entry_at)
    edges = list(pd.date_range(first, last, periods=5))
    atr_edges = list(np.quantile(anchor.atr_over_close.dropna(), [0.25, 0.5, 0.75]))
    result = compare(anchor, audit, edges, atr_edges)
    result.update(samples=int(len(selected)), candidates=int(len(candidates)), seed=args.seed,
                  exit_settings=vars(settings), period={"start": first.isoformat(), "end": last.isoformat()},
                  random_with_additional_anchor=int(flags.loc[selected.index].drop(columns=main_columns)
                                                    .any(axis=1).sum()))
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2, default=float) + "\n", encoding="utf-8")

    print("\nFilled series only; ± is one standard error (neighbouring trades overlap, so treat it as a floor).")
    print(line("All", result["overall"]["anchor"], result["overall"]["random"]))
    for period in result["periods"]:
        print(line(f"{period['from'][:10]} → {period['to'][:10]}", period["anchor"], period["random"]))
    for band in result["atr_bands"]:
        print(line(f"ATR/close quartile {band['band']}", band["anchor"], band["random"]))
    print(f"\nSaved {args.output / 'comparison.json'}")


if __name__ == "__main__":
    main()
