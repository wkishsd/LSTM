"""Write label_agreement.csv into training exports from the exit strategy study runs.

For each filled entry: in how many exit configurations (stop timeframe x wave cutoff, ADX close off)
does its series end with a net return above zero. The trainer's --label-agreement option reads it.

    python Pipelines/Bitfinex/label_agreement.py --anchor-timeframe 5m \
        --export Exports/training/mlp_2023-11-13_to_2026-11-13_current_rules_15m

The study folders are found under --study-root by their tables.json wave_horizon, so the wave-30 run
(exit_strategy_study) and the shorter runs (exit_strategy_study_wH) are all picked up.
"""

import argparse
import csv
import json
from pathlib import Path

import pandas as pd


def study_dirs(root, waves):
    found = {}
    for path in sorted(Path(root).glob("exit_strategy_study*")):
        tables = path / "tables.json"
        if path.is_dir() and tables.exists():
            meta = json.loads(tables.read_text(encoding="utf-8"))
            if meta.get("adx_close") is False:
                found.setdefault(int(meta["wave_horizon"]), path)
    missing = [w for w in waves if w not in found]
    if missing:
        raise SystemExit(f"No exit strategy study run for wave horizon(s) {missing} under {root}")
    return {w: found[w] for w in waves}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study-root", default="Exports/training")
    parser.add_argument("--anchor-timeframe", choices=("5m", "15m"), required=True)
    parser.add_argument("--stops", nargs="+", default=["1m", "5m", "15m"])
    parser.add_argument("--waves", nargs="+", type=int, default=[6, 10, 15, 20, 30])
    parser.add_argument("--export", action="append", required=True,
                        help="Training export folder to write label_agreement.csv into; repeatable")
    args = parser.parse_args()
    dirs = study_dirs(args.study_root, args.waves)
    wins, filled = None, None
    for stop in args.stops:
        for wave, folder in dirs.items():
            name = f"{args.anchor_timeframe}_{stop}"
            series = pd.read_csv(folder / f"series_{name}.csv", usecols=["entry_id", "initial_order_status"])
            ok = set(series.entry_id[series.initial_order_status.eq("filled")].astype(str))
            filled = ok if filled is None else filled & ok
            trades = pd.read_parquet(folder / f"trades_{name}.parquet", columns=["entry_id", "net_pct"])
            won = (trades.groupby("entry_id").net_pct.sum() > 0).astype(int)
            won.index = won.index.astype(str)
            wins = won if wins is None else wins.add(won, fill_value=0)
    configurations = len(args.stops) * len(args.waves)
    for export in args.export:
        export = Path(export)
        entries = pd.read_csv(export / "entry_targets.csv", usecols=["entry_id", "entry_at", "initial_order_status"],
                              dtype={"entry_id": str})
        entries = entries[entries.initial_order_status.eq("filled")]
        unknown = sorted(set(entries.entry_id) - filled)
        if unknown:
            raise SystemExit(f"{len(unknown):,} filled entries in {export} are not filled in every study run, "
                             f"e.g. {unknown[:3]}; was the study run on these anchors?")
        out = export / "label_agreement.csv"
        with out.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["entry_id", "entry_at", "wins", "configurations"])
            for entry_id, entry_at in zip(entries.entry_id, entries.entry_at):
                writer.writerow([entry_id, entry_at, int(wins.get(entry_id, 0)), configurations])
        k = entries.entry_id.map(wins).fillna(0)
        print(f"{out}: {len(entries):,} entries, all {configurations} win {(k == configurations).mean():.1%}, "
              f"all lose {(k == 0).mean():.1%}, mixed 5-10 {((k >= 5) & (k <= 10)).mean():.1%}")


if __name__ == "__main__":
    main()
