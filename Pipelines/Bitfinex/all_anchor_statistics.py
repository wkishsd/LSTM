"""Anchor statistics for all 35 anchors under one set of trailing-stop rules.

The published main-anchor table came from an export built with an older version of the exit code, and
additional anchors never created entries there. This rebuilds the table from scratch with the current code:
at every M5 close where any of the 35 anchors fired it places the collector's initial short stop order
(previous closed M1 Low minus the offset) and labels the series with the exit simulator and fees from the
export's manifest (1m trailing stop, wave-6 horizon, ADX 25, 0.05% per fill). A random-time reference uses the
same rules at M5 closes where no anchor fired. Offline from the candle cache, like random_minute_control.py.

Per anchor: activations (closes where it fired, alone or with others), fired alone (no other of the 35
anchors fired), filled, winners, win rate and its binomial SE, win rate relative to all anchor entries, the
same for alone entries and for both halves (split at the published table's date), mean net / gross return and
trades per series, total and best net return. Writes all_anchor_statistics.csv and the labelled entries.

    python Pipelines/Bitfinex/all_anchor_statistics.py --workers 4
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import random_minute_control as control
from random_minute_control import pipeline

ROOT = Path(__file__).resolve().parents[2]
# Halves split of the published main-anchor table (its middle entry), kept so the halves stay comparable.
HALFWAY = pd.Timestamp("2025-05-17 16:35:00+00:00")
KEEP = ["entry_at", "initial_order_status", "binary_target", "net_return_pct", "target_gross_return_pct",
        "trades_count"]


def stats(frame):
    filled = frame.loc[frame.initial_order_status.eq("filled") & frame.binary_target.notna()]
    n = len(filled)
    win = filled.binary_target.astype(int)
    net = filled.net_return_pct.astype(float)
    out = {"activations": int(len(frame)), "filled": n, "fill_rate": n / len(frame) if len(frame) else math.nan,
           "winners": int(win.sum()), "win_rate": float(win.mean()) if n else math.nan}
    out["win_rate_se"] = math.sqrt(out["win_rate"] * (1 - out["win_rate"]) / n) if n else math.nan
    for name, part in (("first_half", filled.entry_at < HALFWAY), ("second_half", filled.entry_at >= HALFWAY)):
        out[f"filled_{name}"] = int(part.sum())
        out[f"winners_{name}"] = int(win[part].sum())
        out[f"win_rate_{name}"] = float(win[part].mean()) if part.any() else math.nan
    out.update(net_return_pct=float(net.mean()) if n else math.nan,
               gross_return_pct=float(filled.target_gross_return_pct.astype(float).mean()) if n else math.nan,
               trades_per_series=float(filled.trades_count.astype(float).mean()) if n else math.nan,
               total_net_pct=float(net.sum()), best_series_net_pct=float(net.max()) if n else math.nan)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anchor-export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd",
                        help="export whose manifest gives the range and exit settings")
    parser.add_argument("--history-cache", type=Path, default=ROOT / "Exports/candle_cache")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/anchor_statistics/all_35")
    parser.add_argument("--random-samples", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--count-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()

    manifest = json.loads((args.anchor_export / "manifest.json").read_text(encoding="utf-8"))
    settings = control.settings_from(manifest)
    settings.namespace()
    start, end, cutoff = (pd.Timestamp(manifest[k]) for k in ("start_inclusive", "end_exclusive", "snapshot_cutoff"))
    pipeline.request_candles = control.offline
    native, cutoff = pipeline.collect_training_history(start, cutoff, args.history_cache, manifest["window_minutes"])

    # Same decision clock and range as the anchor collector.
    flags = pipeline.detect_anchor_events(pipeline.calculate_native(native["5m"]))
    first = max(native["1m"].timestamp.iloc[0] + pd.Timedelta(days=3), start)
    last = min(native["1m"].available_at.iloc[-1], end)
    flags = flags.loc[(flags.index >= first) & (flags.index < last)]
    columns = sorted(c for c in flags.columns if c.startswith("anchor_"))
    main_columns = [f"anchor_{n:02d}" for n in pipeline.MAIN_ANCHOR_IDS]
    groups = {f"anchor_{n:02d}": "main" for n in pipeline.MAIN_ANCHOR_IDS}
    groups.update({f"anchor_{n:02d}": key.replace("additional_", "").replace("_", " ")
                   for key, (_, ids) in pipeline.ADDITIONAL_ANCHOR_GROUPS.items() for n in ids})

    anchored = flags.loc[flags[columns].any(axis=1)]
    quiet = flags.loc[~flags[columns].any(axis=1)]
    rng = np.random.default_rng(args.seed)
    random = quiet.iloc[np.sort(rng.choice(len(quiet), size=min(args.random_samples, len(quiet)), replace=False))]
    print(f"{len(flags):,} M5 closes; {len(anchored):,} with any anchor "
          f"({int(anchored[main_columns].any(axis=1).sum()):,} with a main anchor); {len(quiet):,} with none; "
          f"random sample {len(random):,}", flush=True)
    if args.count_only:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    decisions = pd.concat([anchored, random]).sort_index()
    entries = pipeline.entries_from_decisions(native, decisions, settings.initial_entry_offset_down)
    labelled = control.label(native, entries, settings, args.output, args.workers)[KEEP]
    labelled["entry_at"] = pd.to_datetime(labelled.entry_at, utc=True)
    rows = labelled.set_index("entry_at").join(decisions[columns]).reset_index()
    rows["random_reference"] = rows.entry_at.isin(random.index).astype(np.uint8)
    rows.to_csv(args.output / "entry_targets.csv", index=False)

    anchor_rows = rows.loc[rows.random_reference.eq(0)]
    count = anchor_rows[columns].sum(axis=1)
    overall = stats(anchor_rows)
    table = []
    for column in columns:
        on = anchor_rows[column].eq(1)
        alone = stats(anchor_rows.loc[on & count.eq(1)])
        row = {"anchor": column, "group": groups[column], **stats(anchor_rows.loc[on])}
        row.update(fired_alone=alone["activations"], filled_alone=alone["filled"],
                   winners_alone=alone["winners"], win_rate_alone=alone["win_rate"])
        table.append(row)
    table.sort(key=lambda r: -r["win_rate"])
    totals = [
        {"anchor": "All anchor entries (35 anchors)", "group": "", **overall},
        {"anchor": "Main-anchor entries", "group": "main", **stats(anchor_rows.loc[anchor_rows[main_columns].any(axis=1)])},
        {"anchor": "Additional anchors only (no main)", "group": "additional",
         **stats(anchor_rows.loc[~anchor_rows[main_columns].any(axis=1)])},
        {"anchor": "Random times, no anchor", "group": "", **stats(rows.loc[rows.random_reference.eq(1)])},
    ]
    out = pd.DataFrame(table + totals)
    out["win_rate_vs_all"] = out.win_rate / overall["win_rate"]
    out.to_csv(args.output / "all_anchor_statistics.csv", index=False)
    (args.output / "run.json").write_text(json.dumps(
        {"exit_settings": vars(settings), "halfway": HALFWAY.isoformat(), "period": [first.isoformat(), last.isoformat()],
         "random_seed": args.seed, "anchor_closes": int(len(anchored))}, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(out[["anchor", "group", "activations", "filled", "winners", "win_rate", "net_return_pct",
               "total_net_pct"]].to_string(index=False))
    print(f"Wrote {args.output / 'all_anchor_statistics.csv'}")


if __name__ == "__main__":
    main()
