"""Trade-level performance of the trailing-stop exit strategy, per anchor and stop timeframe.

    python Pipelines/Bitfinex/exit_strategy_study.py --combos 5m:1m 5m:5m 5m:15m 15m:1m 15m:5m 15m:15m

Goal: judge the exit strategy itself, not the series boundaries. So each run
- switches the ADX forced close off (adx_threshold = infinity), and
- pushes the series cutoff far out (default: start of M5 half-wave 30),
and statistics are computed per TRADE (initial entry or support re-entry),
using only trades closed by the trailing stop. Trades still open at the
cutoff are counted and reported separately, never mixed into the stop-exit
statistics. Overlapping series often hold the very same market trade
(mostly re-entries at a shared support level), so every statistic counts
each distinct trade (same entry/exit time and price) once. Per trade it also measures the favourable / adverse excursion
(lowest Low / highest High while open, as v9's path statistics define them)
and duration. Fees: entry + exit commission per trade (0.05 + 0.05 points).

Outputs in --output: trades_<anchor>_<stop>.parquet per combination (all
trades), series_<anchor>_<stop>.csv, and tables.json with every summary table.
Labeling uses the fast, result-identical exit path; batches are checkpointed.
"""
import argparse
import importlib.util
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("bitfinex_indicators_v2_1", HERE / "bitfinex_indicators_v2.1.py")
pipeline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pipeline
_spec.loader.exec_module(pipeline)
sys.path.insert(0, str(HERE))
import fast_exit  # noqa: E402

fast_exit.install(pipeline.filled_initial_order_engine(), pipeline.original_wave_engine())

PERIOD_EDGES = ["2023-11-12T21:00:00+00:00", "2024-08-12T21:00:00+00:00", "2025-05-13T21:00:00+00:00",
                "2026-02-11T21:00:00+00:00", "2026-11-12T21:00:00+00:00"]
TRADE_COLUMNS = ["entry_id", "trade_no", "trade_source", "trade_entry_ms_utc", "trade_entry_price", "exit_ms_utc",
                 "exit_price", "exit_reason", "short_result_pct", "trade_status", "first_exit_stop_level",
                 "stop_moves_count", "stop_candidates_count"]
_worker = {}


def offline(*_args, **_kwargs):
    raise RuntimeError("The study must use the existing candle cache; a missing cache file would need the API")


def _init(native, settings, first_entry):
    _worker.update(native=native, settings=settings,
                   builder=pipeline.LegacyExitInputBuilder(first_entry, native, settings))


def _batch(chunk, checkpoint):
    if checkpoint.exists():
        return pd.read_pickle(checkpoint)
    targets, frames, orders = pipeline.calculate_exit_targets(
        chunk, _worker["native"], _worker["settings"], progress=False, return_initial_orders=True,
        collect_audit=False, builder=_worker["builder"])
    trades = frames[0]
    trades = trades[[c for c in TRADE_COLUMNS if c in trades.columns]] if not trades.empty else pd.DataFrame(columns=TRADE_COLUMNS)
    series = targets[["trades_count", "series_end_reason", "series_cutoff_at"]].join(
        orders[["initial_order_status"]])
    result = (trades, series)
    temp = checkpoint.with_suffix(".tmp")
    pd.to_pickle(result, temp)
    temp.replace(checkpoint)
    return result


def simulate(entries, native, settings, folder, workers):
    folder.mkdir(parents=True, exist_ok=True)
    chunks = [entries.iloc[i:i + 300] for i in range(0, len(entries), 300)]
    paths = [folder / f"{int(c.index[0]):06d}.pkl" for c in chunks]
    results = [None] * len(chunks)
    natives = {tf: native[tf] for tf in ("1m", "5m")}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(natives, settings, entries.iloc[:1])) as pool:
        futures = {pool.submit(_batch, chunks[i], paths[i]): i for i in range(len(chunks))}
        for done, future in enumerate(as_completed(futures), 1):
            results[futures[future]] = future.result()
            if done % 25 == 0 or done == len(chunks):
                print(f"   batches {done}/{len(chunks)}", flush=True)
    trades = pd.concat([r[0] for r in results], ignore_index=True)
    series = pd.concat([r[1] for r in results])
    return trades, series


def excursions(trades, minute):
    """v9 path statistics per trade: candles with end > entry and start < exit (actual M1 candles)."""
    mts = minute.mts.to_numpy(dtype=np.int64)
    high, low = minute.high.to_numpy(dtype=float), minute.low.to_numpy(dtype=float)
    entry = trades.trade_entry_ms_utc.to_numpy(dtype=np.int64)
    exit_ms = trades.exit_ms_utc.to_numpy(dtype=float)
    ok = np.isfinite(exit_ms)
    lo = np.searchsorted(mts, entry - pipeline.TF_MS_1M, side="right")  # first candle with end > entry
    hi = np.searchsorted(mts, np.where(ok, exit_ms, 0).astype(np.int64), side="left")  # start < exit
    mfe = np.full(len(trades), np.nan)
    mae = np.full(len(trades), np.nan)
    price = trades.trade_entry_price.to_numpy(dtype=float)
    valid = ok & (hi > lo)
    idx = np.flatnonzero(valid)
    if len(idx):
        starts, ends = lo[idx], hi[idx]
        min_low = np.array([np.nanmin(low[s:e]) for s, e in zip(starts, ends)])
        max_high = np.array([np.nanmax(high[s:e]) for s, e in zip(starts, ends)])
        mfe[idx] = np.maximum(0.0, (price[idx] - min_low) / price[idx] * 100.0)
        mae[idx] = np.maximum(0.0, (max_high - price[idx]) / price[idx] * 100.0)
    return mfe, mae


def summarize(trades, fee):
    """Counts, sums and quantiles for one group of stop-exited trades (returns in percentage points)."""
    net = trades.net_pct.to_numpy(dtype=float)
    n = len(net)
    if n == 0:
        return {"trades": 0}
    win = net >= 0
    mfe, mae = trades.mfe_pct.to_numpy(dtype=float), trades.mae_pct.to_numpy(dtype=float)
    gross = trades.short_result_pct.to_numpy(dtype=float)
    capture = gross[mfe > 0] / mfe[mfe > 0]
    q = np.percentile(net, [5, 10, 25, 50, 75, 90, 95])
    return {"trades": n, "winning_trades": int(win.sum()), "sum_net": float(net.sum()),
            "sum_net_sq": float((net ** 2).sum()), "sum_net_wins": float(net[win].sum()),
            "sum_net_losses": float(net[~win].sum()), "sum_gross": float(gross.sum()),
            "net_p05": q[0], "net_p10": q[1], "net_p25": q[2], "net_median": q[3], "net_p75": q[4],
            "net_p90": q[5], "net_p95": q[6],
            "mfe_mean": float(np.nanmean(mfe)), "mfe_median": float(np.nanmedian(mfe)),
            "mae_mean": float(np.nanmean(mae)), "mae_median": float(np.nanmedian(mae)),
            "capture_median": float(np.median(capture)) if len(capture) else np.nan,
            "mfe_over_fee_share": float(np.nanmean(mfe > fee)),
            "duration_median_min": float(trades.duration_min.median()),
            "duration_mean_min": float(trades.duration_min.mean()),
            "stop_moves_mean": float(trades.stop_moves_count.mean())}


TRADE_KEY = ["trade_entry_ms_utc", "trade_entry_price", "exit_ms_utc", "exit_price"]


def unique_trades(trades):
    """Overlapping series often contain the very same market trade (same entry and exit time and
    price), mostly re-entries at a shared support level. Keep one row per market trade, from the
    earliest series, and record how many series contain it."""
    ordered = trades.sort_values(["entry_id", "trade_no"], kind="mergesort")
    key = ordered[TRADE_KEY].astype(str).agg("|".join, axis=1)
    unique = ordered.loc[~key.duplicated()].copy()
    unique["series_sharing"] = key.map(key.value_counts()).loc[unique.index].to_numpy()
    return unique


def tables_for(all_trades, series, combo, fee):
    trades = unique_trades(all_trades)
    stop = trades.loc[trades.exit_reason.eq("exit_stop_triggered")]
    other = trades.loc[~trades.exit_reason.eq("exit_stop_triggered")]
    filled = series.loc[series.initial_order_status.eq("filled")]
    out = {}
    out["overview"] = [dict(combo, group="All stop-exited trades", **summarize(stop, fee),
                            trade_rows_in_series=len(all_trades), all_trades=len(trades),
                            series_sharing_mean=float(trades.series_sharing.mean()), stop_exited=len(stop),
                            closed_at_cutoff=int(other.exit_reason.eq("market_close_wave6").sum()),
                            other_unclosed=int((~other.exit_reason.eq("market_close_wave6")).sum()),
                            cutoff_net_mean=float(other.net_pct.mean()) if len(other) else np.nan,
                            filled_series=len(filled),
                            trades_per_filled_series=len(all_trades) / max(len(filled), 1))]
    out["by_source"] = [dict(combo, group=s, **summarize(g, fee)) for s, g in stop.groupby("trade_source")]
    number = stop.trade_no.clip(upper=5).map(lambda k: f"{int(k)}" if k < 5 else "5+")
    out["by_trade_number"] = [dict(combo, group=k, **summarize(g, fee)) for k, g in stop.groupby(number)]
    period = pd.cut(stop.entry_time, [pd.Timestamp(x) for x in PERIOD_EDGES], right=False,
                    labels=[f"{a[:10]} to {b[:10]}" for a, b in zip(PERIOD_EDGES, PERIOD_EDGES[1:])])
    out["by_period"] = [dict(combo, group=str(k), **summarize(g, fee)) for k, g in stop.groupby(period, observed=True)]
    buckets = pd.cut(stop.duration_min, [0, 15, 60, 240, 720, np.inf], right=False,
                     labels=["under 15 min", "15–60 min", "1–4 h", "4–12 h", "12 h or more"])
    out["by_duration"] = [dict(combo, group=str(k), **summarize(g, fee)) for k, g in stop.groupby(buckets, observed=True)]
    hour = stop.entry_time.dt.tz_convert(pipeline.DISPLAY_TIMEZONE).dt.hour // 4 * 4
    out["by_hour"] = [dict(combo, group=f"{h:02d}:00–{h + 3:02d}:59", **summarize(g, fee)) for h, g in stop.groupby(hour)]
    bins = np.r_[-np.inf, np.arange(-1.5, 2.01, 0.25), np.inf]
    counts, _ = np.histogram(stop.net_pct, bins=bins)
    out["histogram"] = [dict(combo, bin_from=float(a), bin_to=float(b), trades=int(c))
                        for a, b, c in zip(bins[:-1], bins[1:], counts)]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules",
                        help="Export whose manifest supplies the period, cutoff and base exit settings")
    parser.add_argument("--combos", nargs="+", default=["5m:1m", "5m:5m", "5m:15m", "15m:1m", "15m:5m", "15m:15m"],
                        help="anchor_timeframe:stop_timeframe pairs")
    parser.add_argument("--wave-horizon", type=int, default=30)
    parser.add_argument("--keep-adx", action="store_true", help="Keep the ADX forced close (default: off)")
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--sample", type=int, default=0, help="Optional: only the first N entries (for a quick check)")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/training/exit_strategy_study")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.export / "manifest.json").read_text(encoding="utf-8"))
    start, end, cutoff = (pd.Timestamp(manifest[k]) for k in ("start_inclusive", "end_exclusive", "snapshot_cutoff"))
    pipeline.request_candles = offline
    native, cutoff = pipeline.collect_training_history(start, cutoff, ROOT / "Exports/candle_cache",
                                                       manifest["window_minutes"])
    minute = pipeline._collector_frame(native["1m"])
    base = dict(manifest["exit_settings"])
    fee = float(base["entry_commission_pct"]) + float(base["exit_commission_pct"])
    tables = {}
    entry_sets = {}
    for combo_text in args.combos:
        anchor_tf, stop_tf = combo_text.split(":")
        t0 = time.perf_counter()
        if anchor_tf not in entry_sets:
            entry_sets[anchor_tf], _ = pipeline.filter_main_anchor_entries(native, 0, start, end, 0.1, anchor_tf)
        entries = entry_sets[anchor_tf]
        if args.sample:
            entries = entries.iloc[:args.sample]
        settings = pipeline.ExitSettings(**dict(base, stop_timeframe=stop_tf, wave_horizon=args.wave_horizon,
                                                adx_threshold=base["adx_threshold"] if args.keep_adx else float("inf")))
        settings.namespace()
        name = f"{anchor_tf}anchor_{stop_tf}stop"
        print(f"{name}: {len(entries):,} entries, horizon {args.wave_horizon}, "
              f"ADX close {'on' if args.keep_adx else 'off'}", flush=True)
        trades, series = simulate(entries, native, settings, args.output / "checkpoints" / name, args.workers)
        trades = trades.copy()
        trades["entry_time"] = pd.to_datetime(entries.entry_at.reindex(trades.entry_id).to_numpy(), utc=True)
        trades["net_pct"] = trades.short_result_pct.astype(float) - fee
        trades["duration_min"] = (trades.exit_ms_utc.astype(float) - trades.trade_entry_ms_utc.astype(float)) / 60000
        trades["mfe_pct"], trades["mae_pct"] = excursions(trades, minute)
        trades.to_parquet(args.output / f"trades_{anchor_tf}_{stop_tf}.parquet", index=False)
        series.to_csv(args.output / f"series_{anchor_tf}_{stop_tf}.csv")
        combo = {"anchor_timeframe": anchor_tf, "stop_timeframe": stop_tf}
        for key, rows in tables_for(trades, series, combo, fee).items():
            tables.setdefault(key, []).extend(rows)
        ov = tables["overview"][-1]
        print(f"   {ov['stop_exited']:,} stop-exited trades: win {ov['winning_trades'] / ov['trades']:.3f}, "
              f"net {ov['sum_net'] / ov['trades']:+.4f} points | closed at cutoff {ov['closed_at_cutoff']:,} | "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        meta = {"wave_horizon": args.wave_horizon, "adx_close": bool(args.keep_adx), "fee_per_trade_pct": fee,
                "period": [str(start), str(end)], "exit_settings": base, "tables": tables}
        (args.output / "tables.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
