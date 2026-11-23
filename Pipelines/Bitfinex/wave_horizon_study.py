"""Label the same entries with series cutoffs at M5 half-waves 6..N and summarize.

    python Pipelines/Bitfinex/wave_horizon_study.py --anchor-timeframe 15m --stop-timeframe 15m --horizons 6-20

Each horizon reuses the collector's labeling (fast, result-identical exit path,
checkpointed, parallel). Only the cutoff moves: a series ends at the start of
M5 half-wave H, where an open position closes at the next M1 Open. Outputs go
to --output: targets_wH.csv per horizon, summary.csv (one row per horizon and
period) and summary.json. Runs offline from the candle cache.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location("bitfinex_indicators_v2_1", HERE / "bitfinex_indicators_v2.1.py")
pipeline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pipeline  # label workers unpickle functions by this module name
_spec.loader.exec_module(pipeline)
sys.path.insert(0, str(HERE))
import fast_exit  # noqa: E402

fast_exit.install(pipeline.filled_initial_order_engine(), pipeline.original_wave_engine())

PERIOD_EDGES = ["2023-11-12T21:00:00+00:00", "2024-08-12T21:00:00+00:00", "2025-05-13T21:00:00+00:00",
                "2026-02-11T21:00:00+00:00", "2026-11-12T21:00:00+00:00"]


def offline(*_args, **_kwargs):
    raise RuntimeError("The study must use the existing candle cache; a missing cache file would need the API")


def horizons(text):
    if "-" in text:
        a, b = (int(x) for x in text.split("-"))
        return list(range(a, b + 1))
    return [int(x) for x in text.split(",")]


def summarize(audit, horizon, settings):
    filled = audit.loc[audit.initial_order_status.eq("filled") & audit.binary_target.notna()].copy()
    filled["net"] = filled.net_return_pct.astype(float)
    spans = [("All", None, None)] + [(f"{a[:10]} to {b[:10]}", pd.Timestamp(a), pd.Timestamp(b))
                                     for a, b in zip(PERIOD_EDGES, PERIOD_EDGES[1:])]
    rows = []
    for label, a, b in spans:
        part = filled if a is None else filled.loc[(filled.entry_at >= a) & (filled.entry_at < b)]
        net = part.net
        wins = part.binary_target.astype(int).eq(1)
        winner, loser = net[wins], net[~wins]
        held = (part.series_cutoff_at - part.entry_at).dt.total_seconds() / 3600
        rows.append({
            "wave_horizon": horizon, "period": label, "filled_series": len(part), "winning_series": int(wins.sum()),
            "win_rate": wins.mean(), "net_return_mean_pct": net.mean(),
            "net_return_se_pct": net.std(ddof=1) / np.sqrt(len(net)) if len(net) > 1 else np.nan,
            "net_return_median_pct": net.median(), "sum_net_pct": net.sum(), "sum_net_sq": float((net ** 2).sum()),
            "gross_return_mean_pct": part.target_gross_return_pct.astype(float).mean(),
            "trades_per_series": part.trades_count.mean(),
            "avg_winner_pct": winner.mean(), "avg_loser_pct": loser.mean(),
            "breakeven_win_rate": -loser.mean() / (winner.mean() - loser.mean()) if len(winner) and len(loser) else np.nan,
            "closed_at_cutoff_share": part.wave6_market_exit_price.notna().mean() if "wave6_market_exit_price" in part else np.nan,
            "adx_terminated_share": part.series_end_reason.eq("adx_gt_25").mean(),
            "median_horizon_hours": held.median(),
            "stop_timeframe": settings.stop_timeframe})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules",
                        help="Export whose manifest supplies the period, cutoff and exit settings")
    parser.add_argument("--anchor-timeframe", default="15m", choices=pipeline.ANCHOR_TIMEFRAMES)
    parser.add_argument("--stop-timeframe", default="15m", choices=pipeline.STOP_TIMEFRAMES)
    parser.add_argument("--horizons", type=horizons, default=horizons("6-20"), help="e.g. 6-20 or 6,8,12")
    parser.add_argument("--workers", type=int, default=11)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    out = args.output or ROOT / f"Exports/training/wave_horizon_{args.anchor_timeframe}anchor_{args.stop_timeframe}stop"
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.export / "manifest.json").read_text(encoding="utf-8"))
    start, end, cutoff = (pd.Timestamp(manifest[k]) for k in ("start_inclusive", "end_exclusive", "snapshot_cutoff"))
    pipeline.request_candles = offline
    native, cutoff = pipeline.collect_training_history(start, cutoff, ROOT / "Exports/candle_cache",
                                                       manifest["window_minutes"])
    entries, _ = pipeline.filter_main_anchor_entries(native, 0, start, end, 0.1, args.anchor_timeframe)
    exit_settings = dict(manifest["exit_settings"], stop_timeframe=args.stop_timeframe)
    print(f"{len(entries):,} {args.anchor_timeframe}-anchor entries, {args.stop_timeframe} stop, "
          f"horizons {args.horizons}", flush=True)

    rows = []
    for horizon in args.horizons:
        t0 = time.perf_counter()
        settings = pipeline.ExitSettings(**dict(exit_settings, wave_horizon=horizon))
        settings.namespace()
        audit = pipeline.label_mlp_entries(entries, native, settings, out / "label_checkpoints", workers=args.workers)
        audit = audit.reindex(entries.index)
        audit["entry_at"] = pd.to_datetime(audit.entry_at, utc=True)
        audit.to_csv(out / f"targets_w{horizon}.csv")
        rows.extend(summarize(audit, horizon, settings))
        best = rows[-5]
        print(f"wave {horizon}: win {best['win_rate']:.3f}, net {best['net_return_mean_pct']:+.4f}% "
              f"(SE {best['net_return_se_pct']:.4f}), trades {best['trades_per_series']:.2f}, "
              f"closed at cutoff {best['closed_at_cutoff_share']:.2f} | {time.perf_counter() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    meta = {"anchor_timeframe": args.anchor_timeframe, "stop_timeframe": args.stop_timeframe,
            "horizons": args.horizons, "entries": len(entries), "period": [str(start), str(end)],
            "exit_settings": exit_settings, "rows": rows}
    (out / "summary.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
