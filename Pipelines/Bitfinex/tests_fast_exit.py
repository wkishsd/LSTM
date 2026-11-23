"""Equivalence test: fast_exit (array loop) against the v9 exit engine.

Runs the same sampled main-anchor entries through calculate_exit_targets twice,
once with the v9 loop and once with fast_exit installed, and requires exact
equality of targets, trades, stop events, support searches and re-entry
orders (on the fields fast_exit produces). Runs offline from the candle cache.

    python Pipelines/Bitfinex/tests_fast_exit.py --samples 300 --stop-timeframes 1m 5m 15m

Exit code 0 only if every compared table matches for every stop timeframe.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import random_minute_control as rmc  # noqa: E402  (stubs openpyxl, loads the pipeline offline-capable)
import fast_exit  # noqa: E402

p = rmc.pipeline

TRADE_COLUMNS = ["entry_id", "trade_no", "trade_source", "trade_entry_ms_utc", "trade_entry_price", "entry_order_no",
                 "first_exit_stop_level", "last_exit_stop_level", "stop_moves_count", "stop_candidates_count",
                 "stop_ignored_count", "exit_ms_utc", "exit_price", "exit_reason", "stop_trigger_event_no",
                 "short_result_pct", "trade_status"]
STOP_COLUMNS = ["entry_id", "stop_event_no", "trade_no", "event_type", "event_ms_utc", "minute_mts_utc",
                "old_stop_level", "new_stop_level", "candidate_stop_level"]
SUPPORT_COLUMNS = ["entry_id", "support_search_no", "search_ms_utc", "source_macd_1m_wave_no", "support_found",
                   "support_level", "support_is_usable"]
ORDER_COLUMNS = ["entry_id", "entry_order_no", "order_status", "order_set_ms_utc", "entry_order_level",
                 "trigger_ms_utc"]


def normalized(frame, columns):
    """Columns in a comparable form: numbers as float64 (NaN-aware), everything else as text."""
    out = pd.DataFrame(index=range(len(frame)))
    for column in columns:
        values = frame[column] if column in frame.columns else pd.Series(np.nan, index=frame.index)
        values = values.reset_index(drop=True)
        if pd.api.types.is_datetime64_any_dtype(values):
            stamps = pd.to_datetime(values, utc=True)
            out[column] = stamps.dt.tz_convert(None).astype("datetime64[ns]").astype("int64").where(stamps.notna()).astype("float64")
            continue
        numeric = pd.to_numeric(values, errors="coerce")
        if values.isna().equals(numeric.isna()) and not values.map(lambda v: isinstance(v, (bool, np.bool_))).any():
            out[column] = numeric.astype("float64")
        else:
            out[column] = values.astype(str)
    return out


def compare(name, reference, candidate, columns):
    a, b = normalized(reference, columns), normalized(candidate, columns)
    if len(a) != len(b):
        return f"{name}: {len(a)} v9 rows vs {len(b)} fast rows"
    for column in columns:
        x, y = a[column], b[column]
        if x.dtype == "float64" and y.dtype == "float64":
            bad = ~((x == y) | (x.isna() & y.isna()))
        else:
            bad = x.astype(str) != y.astype(str)
        if bad.any():
            i = int(np.flatnonzero(bad.to_numpy())[0])
            return (f"{name}.{column}: {int(bad.sum())} mismatches, first at row {i}: "
                    f"v9={x.iloc[i]!r} fast={y.iloc[i]!r} (entry {a['entry_id'].iloc[i]})")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules",
                        help="Export whose manifest supplies the period, cutoff and exit settings")
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--stop-timeframes", nargs="+", default=["1m", "5m", "15m"], choices=p.STOP_TIMEFRAMES)
    parser.add_argument("--anchor-timeframe", default="5m", choices=p.ANCHOR_TIMEFRAMES)
    args = parser.parse_args()

    import json
    manifest = json.loads((args.export / "manifest.json").read_text(encoding="utf-8"))
    start, end, cutoff = (pd.Timestamp(manifest[k]) for k in ("start_inclusive", "end_exclusive", "snapshot_cutoff"))
    p.request_candles = rmc.offline
    native, cutoff = p.collect_training_history(start, cutoff, ROOT / "Exports/candle_cache", manifest["window_minutes"])
    entries, _ = p.filter_main_anchor_entries(native, 0, start, end, 0.1, args.anchor_timeframe)
    rng = np.random.default_rng(args.seed)
    sample = entries.iloc[np.sort(rng.choice(len(entries), min(args.samples, len(entries)), replace=False))]
    engine = p.filled_initial_order_engine()
    wave = p.original_wave_engine()
    print(f"{len(sample)} sampled {args.anchor_timeframe}-anchor entries, seed {args.seed}", flush=True)

    # Bit-for-bit check of the replaced v5 helpers on the whole 1m close series.
    close = pd.Series(native["1m"]["close"].to_numpy(dtype=float))
    fast_exit.uninstall(engine, wave)
    ref_macd = wave.calculate_macd(pd.DataFrame({"close": close}))
    ref_waves = wave.build_waves(ref_macd["hist"])
    fast_exit.install(engine, wave)
    new_macd = wave.calculate_macd(pd.DataFrame({"close": close}))
    new_waves = wave.build_waves(new_macd["hist"])
    fast_exit.uninstall(engine, wave)
    helpers_ok = all(np.array_equal(ref_macd[c].to_numpy(), new_macd[c].to_numpy(), equal_nan=True)
                     for c in ("ema_fast", "ema_slow", "macd", "signal", "hist")) and ref_waves == new_waves
    print(f"MACD/EMA and wave builder on {len(close):,} 1m closes: {'IDENTICAL' if helpers_ok else 'MISMATCH'}",
          flush=True)

    failures = 0 if helpers_ok else 1
    for tf in args.stop_timeframes:
        settings = rmc.settings_from(manifest, tf)
        builder = p.LegacyExitInputBuilder(entries.iloc[:1], native, settings)
        runs = {}
        for label in ("v9", "fast"):
            (fast_exit.install if label == "fast" else fast_exit.uninstall)(engine, wave)
            t0 = time.perf_counter()
            runs[label] = p.calculate_exit_targets(sample, native, settings, progress=False,
                                                   return_initial_orders=True, builder=builder)
            runs[label] = runs[label] + (time.perf_counter() - t0,)
        fast_exit.uninstall(engine, wave)
        (t_ref, f_ref, _, s_ref), (t_new, f_new, _, s_new) = runs["v9"], runs["fast"]
        problems = [compare("targets", t_ref.reset_index(), t_new.reset_index(), list(t_ref.reset_index().columns)),
                    compare("trades", f_ref[0], f_new[0], TRADE_COLUMNS),
                    compare("stop_events", f_ref[1], f_new[1], STOP_COLUMNS),
                    compare("support_searches", f_ref[2], f_new[2], SUPPORT_COLUMNS),
                    compare("entry_orders", f_ref[3], f_new[3], ORDER_COLUMNS)]
        problems = [x for x in problems if x]
        filled = int((t_ref.trades_count > 0).sum())
        status = "IDENTICAL" if not problems else "MISMATCH"
        print(f"{tf} stop: {status} | {filled} filled series, {len(f_ref[0])} trades, {len(f_ref[1])} stop events | "
              f"v9 {s_ref:.1f}s, fast {s_new:.1f}s ({s_ref / s_new:.1f}x)", flush=True)
        for problem in problems:
            print("   ", problem, flush=True)
        failures += bool(problems)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
