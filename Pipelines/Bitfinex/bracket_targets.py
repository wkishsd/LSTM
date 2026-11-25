"""Fixed-bracket targets on every closed 5m candle: does the trade reach +X% net before -Y% net?

    python Pipelines/Bitfinex/bracket_targets.py
    python Pipelines/Bitfinex/bracket_targets.py --tp 2 --sl 1          # 2:1 bracket, for long moves

No moving stop and no re-entries. On every M5 close t the trade enters at the open of the first M1 at or
after t (a market order) and is held until one of two fixed exits is touched on M1 candles, checked
minute by minute:

- short (default): take profit when the price falls far enough for +tp% net, stop when it rises far
  enough for -sl% net. "Net" includes both fills' commissions, so with 0.05 + 0.05 the take profit sits
  1.1% below the entry and the stop 0.9% above it (--barriers price puts both at exactly tp% / sl%);
- a minute that touches both levels is counted as a stop (--ambiguous loss) or dropped (--ambiguous drop);
- if the price opens beyond a level, the exit is that open, not the level;
- a trade still open after --max-hours closes at the last M1 close (outcome "timeout"); an entry whose
  window runs past the candle data without touching a level is dropped as unresolved.

The output is a training export in the trainer's own format, so mlp_trades, the training panel and
trade_baselines.py read it unchanged: the source export's feature timeline (hard-linked, not copied),
its manifest with the bracket settings, and entry_targets.csv with one filled entry per M5 close.
entry_rule = every_m5_close tells the loader that entries need no anchor.

A twin export with the 18 anchor columns of the timeline set to 0 (--twin, on by default) has identical
entries and labels, so "MLP with anchors" and "MLP without anchors" differ only in the anchor inputs.

summary.json in each export holds the random-entry baseline: win rate and mean net overall, per period
of the trainer's 70/15/15 split and per month; outcome shares and time to exit; random samples of
--sample-size entries (spread of what a random pick gets); and one-position-at-a-time trading of every
candle in order (enter whenever flat), the "always in" random trader the models must beat.
"""
import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "Training/rocm-9070")]

from mlp_trades.schema import HISTORY, WIDTH  # noqa: E402

DEFAULT_SOURCE = ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m"
CHUNK = 4000


def load_pipeline():
    spec = importlib.util.spec_from_file_location("bitfinex_indicators_v2_1", HERE / "bitfinex_indicators_v2.1.py")
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)

    def offline(*_args, **_kwargs):
        raise RuntimeError("Use the existing candle cache; a missing cache file would need the API")
    pipeline.request_candles = offline
    return pipeline


def native_candles(manifest):
    pipeline = load_pipeline()
    native, _ = pipeline.collect_training_history(pd.Timestamp(manifest["start_inclusive"]),
                                                  pd.Timestamp(manifest["snapshot_cutoff"]),
                                                  ROOT / "Exports/candle_cache", manifest["window_minutes"])
    return pipeline, native


def dense_minutes(m1):
    """open/high/low/close on a gapless minute grid (NaN where Bitfinex has no candle); grid[0] is the first open."""
    opens = (m1.timestamp_ms.to_numpy() // 60_000).astype(np.int64)
    first = int(opens[0])
    n = int(opens[-1]) - first + 1
    grid = {c: np.full(n, np.nan) for c in ("open", "high", "low", "close")}
    for c in grid:
        grid[c][opens - first] = m1[c].to_numpy(dtype=float)
    # next_valid[k]: the first minute at or after k with a candle.
    valid = ~np.isnan(grid["open"])
    idx = np.where(valid, np.arange(n), n)
    next_valid = np.minimum.accumulate(idx[::-1])[::-1]
    return first * 60, grid, next_valid


def first_true(mask):
    """Index of the first True per row, or -1."""
    hit = mask.any(axis=1)
    return np.where(hit, mask.argmax(axis=1), -1)


def bracket_outcomes(t0, grid, next_valid, times, args):
    """One row per entry time (unix seconds at an M5 close)."""
    fee = args.entry_fee + args.exit_fee
    up = (args.sl - fee) if args.barriers == "net" else args.sl          # adverse move in % of entry, short
    down = (args.tp + fee) if args.barriers == "net" else args.tp        # favourable move
    if args.side == "long":
        up, down = down, up   # long: take profit above, stop below
    if up <= 0 or down <= 0:
        raise SystemExit("Both levels must be beyond the entry; raise --tp / --sl or lower the fees")
    horizon = int(args.max_hours * 60)
    n = len(grid["open"])
    o, h, lo, c = (np.concatenate([grid[k], np.full(horizon, np.nan)]) for k in ("open", "high", "low", "close"))
    start = (times - t0) // 60
    ok = (start >= 0) & (start < n)
    start = np.where(ok, start, 0)
    entry_idx = next_valid[start]
    ok &= (entry_idx < n) & (entry_idx - start <= args.max_entry_delay)
    entry_idx = np.where(ok, entry_idx, 0)
    entry_price = o[entry_idx]
    out = {k: np.full(len(times), np.nan) for k in ("exit_idx", "exit_price")}
    outcome = np.full(len(times), "", dtype=object)
    view_h = np.lib.stride_tricks.sliding_window_view(h, horizon)
    view_l = np.lib.stride_tricks.sliding_window_view(lo, horizon)
    for a in range(0, len(times), CHUNK):
        rows = np.arange(a, min(a + CHUNK, len(times)))
        rows = rows[ok[rows]]
        if not len(rows):
            continue
        e = entry_idx[rows]
        p = entry_price[rows]
        hi_w, lo_w = view_h[e], view_l[e]
        if args.side == "short":
            stop, take = p * (1 + up / 100), p * (1 - down / 100)
            hit_stop, hit_take = hi_w >= stop[:, None], lo_w <= take[:, None]
        else:
            stop, take = p * (1 - up / 100), p * (1 + down / 100)
            hit_stop, hit_take = lo_w <= stop[:, None], hi_w >= take[:, None]
        s_at, t_at = first_true(hit_stop), first_true(hit_take)
        s_key = np.where(s_at < 0, horizon, s_at)
        t_key = np.where(t_at < 0, horizon, t_at)
        both = (s_key == t_key) & (s_key < horizon)
        stop_first = (s_key < t_key) | both
        take_first = t_key < s_key
        k = np.where(stop_first, s_key, np.where(take_first, t_key, horizon - 1))
        exit_idx = e + k
        bar_open = o[exit_idx]
        # A level crossed by the open fills at the open (worse for a stop, better for a take profit).
        if args.side == "short":
            stop_fill = np.where(bar_open > stop, bar_open, stop)
            take_fill = np.where(bar_open < take, bar_open, take)
        else:
            stop_fill = np.where(bar_open < stop, bar_open, stop)
            take_fill = np.where(bar_open > take, bar_open, take)
        stop_fill = np.where(k == 0, stop, stop_fill)
        take_fill = np.where(k == 0, take, take_fill)
        # Timeout: the last M1 close inside the window.
        closes = np.lib.stride_tricks.sliding_window_view(c, horizon)[e]
        last = horizon - 1 - np.argmax(~np.isnan(closes[:, ::-1]), axis=1)
        timeout_fill = closes[np.arange(len(e)), last]
        price = np.where(stop_first, stop_fill, np.where(take_first, take_fill, timeout_fill))
        res = np.where(both, "ambiguous", np.where(stop_first, "stop", np.where(take_first, "take_profit", "timeout")))
        timeout_idx = e + last
        exit_idx = np.where(res == "timeout", timeout_idx, exit_idx)
        # A window cut off by the end of the candle data cannot time out.
        unresolved = (res == "timeout") & (e + horizon > n)
        res = np.where(unresolved, "unresolved", res)
        out["exit_idx"][rows], out["exit_price"][rows], outcome[rows] = exit_idx, price, res
    outcome[~ok] = "no_entry_candle"
    frame = pd.DataFrame({"t": times, "entry_idx": entry_idx, "entry_price": entry_price,
                          "exit_idx": out["exit_idx"], "exit_price": out["exit_price"], "outcome": outcome})
    sign = 1 if args.side == "long" else -1
    frame["gross_pct"] = sign * (frame.exit_price - frame.entry_price) / frame.entry_price * 100
    frame["net_pct"] = frame.gross_pct - fee
    return frame


def write_export(target, source, manifest, entries, args, zero_anchors):
    if target.exists():
        if not args.overwrite:
            raise SystemExit(f"{target} exists; pass --overwrite to rebuild it")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    timeline = source / "feature_timeline.npy"
    if zero_anchors:
        src = np.load(timeline, mmap_mode="r")
        dst = np.lib.format.open_memmap(target / "feature_timeline.npy", mode="w+", dtype=src.dtype, shape=src.shape)
        for a in range(0, len(src), 200_000):
            block = np.array(src[a:a + 200_000])
            block[:, 36:] = 0
            dst[a:a + 200_000] = block
        dst.flush()
        del dst
    else:
        try:
            os.link(timeline, target / "feature_timeline.npy")
        except OSError:
            shutil.copyfile(timeline, target / "feature_timeline.npy")
    meta = dict(manifest)
    for key in ("unique_entry_ids", "binary_labels", "individual_anchor_activations", "train_rows", "test_rows",
                "unresolved_rows", "purged_training_rows", "first_test_decision_at", "wave6_market_closes",
                "train_class_counts", "test_class_counts", "exit_settings", "target_rule", "wave_count_rule"):
        meta.pop(key, None)
    meta.update({
        "dataset": "bracket_m5",
        "entry_rule": "every_m5_close",
        "source_export": source.name,
        "entry_commission_pct_per_fill": args.entry_fee,
        "exit_commission_pct_per_fill": args.exit_fee,
        "anchor_inputs": "zeroed" if zero_anchors else "main_18",
        "bracket": {"side": args.side, "take_profit_pct": args.tp, "stop_pct": args.sl, "barriers": args.barriers,
                    "max_hours": args.max_hours, "ambiguous": args.ambiguous, "entry": "market at the first M1 open at or after the M5 close",
                    "start": args.start, "end": args.end},
        "target_rule": f"{args.side}: +{args.tp}% before -{args.sl}% ({args.barriers}), label = gross > fees",
    })
    (target / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    entries.to_csv(target / "entry_targets.csv", index=False)


def one_position(times, ends, net):
    """Take every entry in time order whenever flat."""
    order = np.argsort(times, kind="stable")
    free, taken = -1, []
    for i in order:
        if times[i] >= free:
            taken.append(i)
            free = ends[i]
    taken = np.array(taken)
    return {"trades": len(taken), "total_net": float(net[taken].sum()), "net_per_trade": float(net[taken].mean()),
            "win_rate": float((net[taken] > 0).mean())}


def summary(export, entries, args):
    from mlp_trades.bitfinex import export_paths, prepare_bitfinex
    from mlp_trades.split import training_split

    data = prepare_bitfinex(export_paths(export), training=True)
    splits, _ = training_split(data, 0.7, 0.15)
    by_id = entries.assign(entry_id=entries.entry_id.astype(str)).set_index("entry_id")
    fee = args.entry_fee + args.exit_fee
    win_net = args.tp if args.barriers == "net" else args.tp - fee
    loss_net = args.sl if args.barriers == "net" else args.sl + fee
    result = {"export": export.name, "entries": len(entries), "accepted_by_trainer": int(len(data.valid)),
              "trainer_excluded": data.report["reason_counts"],
              "break_even_win_rate": loss_net / (win_net + loss_net),
              "outcomes": entries.outcome.value_counts(normalize=True).round(5).to_dict(),
              "minutes_to_exit": entries.minutes_to_exit.describe(percentiles=[.25, .5, .75, .9]).round(1).to_dict()}
    rng = np.random.default_rng(args.seed)
    periods = {}
    for name, idx in splits.items():
        ids = data.ids[idx]
        sub = by_id.loc[ids]
        y, net = sub.binary_target.to_numpy(), sub.net_return_pct.to_numpy()
        draws = rng.integers(0, len(sub), size=(args.samples, min(args.sample_size, len(sub))))
        wr = y[draws].mean(axis=1)
        mn = net[draws].mean(axis=1)
        t = pd.to_datetime(sub.entry_at, utc=True)
        periods[name] = {
            "from": str(t.min()), "to": str(t.max()), "entries": len(sub), "win_rate": float(y.mean()),
            "mean_net_pct": float(net.mean()),
            f"random_{args.sample_size}_win_rate_p5_p50_p95": np.percentile(wr, [5, 50, 95]).round(4).tolist(),
            f"random_{args.sample_size}_mean_net_p5_p50_p95": np.percentile(mn, [5, 50, 95]).round(4).tolist(),
            "one_position_every_candle": one_position(data.times[idx], data.ends[idx], net),
        }
    result["periods"] = periods
    month = pd.to_datetime(entries.entry_at, utc=True).dt.strftime("%Y-%m")
    result["by_month"] = entries.groupby(month).agg(entries=("binary_target", "size"), win_rate=("binary_target", "mean"),
                                                    mean_net=("net_return_pct", "mean")).round(4).reset_index().to_dict("records")
    (export / "summary.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="export whose feature timeline is reused")
    parser.add_argument("--side", choices=("short", "long"), default="short")
    parser.add_argument("--tp", type=float, default=1.0, help="take profit, %% net (or price with --barriers price)")
    parser.add_argument("--sl", type=float, default=1.0, help="stop, %% net (or price with --barriers price)")
    parser.add_argument("--barriers", choices=("net", "price"), default="net")
    parser.add_argument("--entry-fee", type=float, default=0.05, help="%% per entry fill")
    parser.add_argument("--exit-fee", type=float, default=0.05, help="%% per exit fill")
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--max-entry-delay", type=int, default=5, help="minutes to wait for the first M1 after the close")
    parser.add_argument("--ambiguous", choices=("loss", "drop"), default="loss")
    parser.add_argument("--start", help="first M5 close to use (UTC), default the source's start")
    parser.add_argument("--end", help="M5 closes before this time (UTC), default the source's end")
    parser.add_argument("--name", help="export folder suffix; default from the settings")
    parser.add_argument("--no-twin", dest="twin", action="store_false", help="skip the zeroed-anchor twin export")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source.resolve()
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    timeline_rows = len(np.load(source / "feature_timeline.npy", mmap_mode="r"))
    first_close = int(pd.Timestamp(manifest["feature_timeline_first_close"]).timestamp())
    start = pd.Timestamp(args.start or manifest["start_inclusive"])
    end = pd.Timestamp(args.end or manifest["end_exclusive"])
    start, end = (x.tz_localize("UTC") if x.tzinfo is None else x for x in (start, end))
    lo = max(int(start.ceil("5min").timestamp()), first_close + HISTORY * 60)
    hi = min(int(end.timestamp()), first_close + (timeline_rows - 1) * 60)
    times = np.arange(lo + (-(lo % 300)) % 300, hi, 300, dtype=np.int64)
    print(f"{len(times):,} M5 closes from {pd.Timestamp(times[0], unit='s', tz='UTC')} "
          f"to {pd.Timestamp(times[-1], unit='s', tz='UTC')}", flush=True)

    _, native = native_candles(manifest)
    t0, grid, next_valid = dense_minutes(native["1m"])
    frame = bracket_outcomes(t0, grid, next_valid, times, args)
    counts = frame.outcome.value_counts().to_dict()
    keep = ~frame.outcome.isin(["no_entry_candle", "unresolved"] + (["ambiguous"] if args.ambiguous == "drop" else []))
    frame = frame[keep].reset_index(drop=True)
    entry_at = pd.to_datetime(frame.t, unit="s", utc=True)
    exit_open = pd.to_datetime(t0 + frame.exit_idx.astype(np.int64) * 60, unit="s", utc=True)
    entries = pd.DataFrame({
        "entry_id": np.arange(1, len(frame) + 1),
        "entry_at": entry_at.map(lambda x: x.isoformat()),
        "feature_end_row": ((frame.t - first_close) // 60).astype(np.int64),
        "entry_fill_at": pd.to_datetime(t0 + frame.entry_idx.astype(np.int64) * 60, unit="s", utc=True).map(lambda x: x.isoformat()),
        "entry_price": frame.entry_price.round(6),
        "outcome": frame.outcome,
        "exit_at": exit_open.map(lambda x: x.isoformat()),
        "exit_price": frame.exit_price.round(6),
        "minutes_to_exit": (frame.exit_idx - frame.entry_idx + 1).astype(np.int64),
        "label_available_at": (exit_open + pd.Timedelta(minutes=1)).map(lambda x: x.isoformat()),
        "target_gross_return_pct": frame.gross_pct.round(6),
        "commission_pct": args.entry_fee + args.exit_fee,
        "filled_entry_orders": 1, "filled_exit_orders": 1, "initial_order_status": "filled",
        "net_return_pct": frame.net_pct.round(6),
    })
    entries["binary_target"] = (entries.target_gross_return_pct > entries.commission_pct).astype(int)
    tag = args.name or (f"bracket_{args.side}_tp{args.tp:g}_sl{args.sl:g}" + ("" if args.barriers == "net" else "_price")
                        + ("" if args.entry_fee + args.exit_fee == 0.1 else f"_fee{args.entry_fee + args.exit_fee:g}"))
    stem = source.name.split("_current_rules")[0].split("_bounded")[0]
    stem = stem if stem.startswith("mlp_") else "mlp_2023-11-13_to_2026-11-13"
    targets = [(ROOT / "Exports/training" / f"{stem}_{tag}", False)]
    if args.twin:
        targets.append((ROOT / "Exports/training" / f"{stem}_{tag}_noanchors", True))
    print(f"outcomes before filtering: {counts}", flush=True)
    for target, zero in targets:
        write_export(target, source, manifest, entries, args, zero)
        result = summary(target, entries, args)
        print(f"{target.name}: {len(entries):,} entries, trainer accepts {result['accepted_by_trainer']:,}", flush=True)
    print(json.dumps({k: result[k] for k in ("break_even_win_rate", "outcomes", "minutes_to_exit", "periods")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
