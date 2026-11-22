"""Collect a fixed-horizon net-return label for each confirmed Bitfinex short fill."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np

from .schema import DEFAULT_MAX_EXIT_DELAY, DEFAULT_PROFIT_HOLD, FILL_DECISION_CLOCK, PROFIT_CLASSES, PROFIT_DATASET


def short_outcome(entry, exit_price, entry_fee, exit_fee):
    """Return percent of entry notional, with each fee charged on its own fill."""
    if not all(value.is_finite() and value > 0 for value in (entry, exit_price)):
        raise ValueError("Trade prices must be finite and positive")
    gross = 100 * (entry - exit_price) / entry
    fees = (entry_fee * entry + exit_fee * exit_price) / entry
    net_amount = (100 - entry_fee) * entry - (100 + exit_fee) * exit_price
    bucket = sum(net_amount >= entry * boundary for boundary in (0, 1, 2, 3))
    return gross, fees, net_amount / entry, bucket


def actual_opens(directory, first, cutoff):
    prices = {}
    for path in sorted(directory.glob("*.json")):
        left, sep, right = path.stem.partition("_")
        if sep and left.isdecimal() and right.isdecimal():
            if int(right) < 1000 * first or int(left) >= 1000 * cutoff:
                continue
        for stamp, opening, *_ in json.loads(path.read_text(encoding="utf-8"), parse_float=Decimal):
            second, remainder = divmod(stamp, 1000)
            if remainder or second % 60:
                raise ValueError(f"M1 open is off the minute clock in {path.name}")
            if not (first <= second and second + 60 <= cutoff):
                continue
            price = Decimal(opening)
            if not price.is_finite() or price <= 0 or prices.setdefault(second, price) != price:
                raise ValueError(f"Invalid or conflicting M1 open at {second} in {path.name}")
    if not prices:
        raise ValueError(f"No actual M1 candles in {directory} for the requested exit times")
    return np.array(sorted(prices), dtype=np.int64), prices


def iso(seconds):
    return datetime.fromtimestamp(int(seconds), timezone.utc).isoformat()


def feature_end(fill, placed):
    """Close of the last M1 candle that ended before the fill candle opened.

    The source export stamps gap/open fills at the fill candle's open but level-touch
    fills at its close (a time proxy), and it does not record which applies. Ending the
    window one minute before the stamp is correct for touch fills and one extra minute
    conservative for open fills, so no fill-candle data reaches the features. The order
    is never filled before its placement candle, so the window never ends before it.
    """
    return max(fill - 60, placed)


def collect(source, candles, output, hold_minutes, max_exit_delay_minutes=DEFAULT_MAX_EXIT_DELAY):
    from .bitfinex import export_schema, nonnegative_decimal, timestamp

    if hold_minutes < 1:
        raise ValueError("Profit hold must be at least one minute")
    if max_exit_delay_minutes < 0:
        raise ValueError("Maximum exit delay must be nonnegative")
    source, candles, output = map(Path, (source, candles, output))
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    export_schema(manifest)
    if manifest.get("dataset", "bitfinex") != "bitfinex":
        raise ValueError("Use the original Bitfinex anchor export as the fill source")
    first = timestamp(manifest["feature_timeline_first_close"])
    cutoff = timestamp(manifest["snapshot_cutoff"])
    timeline_path = source / "feature_timeline.npy"
    timeline_rows = len(np.load(timeline_path, mmap_mode="r", allow_pickle=False))
    entry_fee = nonnegative_decimal(manifest["entry_commission_pct_per_fill"])
    exit_fee = nonnegative_decimal(manifest["exit_commission_pct_per_fill"])
    counts = Counter()
    fills = []
    with (source / "entry_targets.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["initial_order_status"] != "filled":
                counts["without_confirmed_fill"] += 1
                continue
            decision = timestamp(row["initial_fill_at"])
            original = timestamp(row["entry_at"])
            if decision < original or decision > cutoff or decision % 60 or original % 60:
                raise ValueError(f"Invalid confirmed fill time for entry {row['entry_id']}")
            window_end = feature_end(decision, original)
            position = (window_end - first) // 60
            if not manifest["window_minutes"] - 1 <= position < timeline_rows:
                counts["fill_outside_feature_timeline"] += 1
                continue
            fills.append((decision, window_end, position, row))
    if not fills:
        raise ValueError("No confirmed fills with a complete feature window")
    fills.sort(key=lambda item: item[0])
    hold = hold_minutes * 60
    if fills[0][0] + hold >= cutoff:
        raise ValueError("Holding time exceeds the available outcome history")
    minutes, prices = actual_opens(candles, fills[0][0] + hold, cutoff)
    times = np.array([decision + hold for decision, _, _, _ in fills], dtype=np.int64)
    exits = minutes.searchsorted(times, side="left")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".profit-") as folder:
        staged = Path(folder) / "export"
        staged.mkdir()
        with (staged / "entry_targets.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("entry_id", "entry_at", "signal_at", "initial_fill_at", "main_anchor_numbers",
                             "feature_end_row", "initial_fill_price", "scheduled_exit_at", "exit_at", "exit_price",
                             "target_gross_return_pct", "commission_pct", "net_return_pct", "profit_class",
                             "label_available_at", "target_status"))
            for (decision, window_end, position, row), scheduled, slot in zip(fills, times, exits):
                fill_price = Decimal(row["initial_fill_price"])
                common = [row["entry_id"], iso(window_end), row["entry_at"], iso(decision),
                          row["main_anchor_numbers"], position, fill_price, iso(scheduled)]
                if slot == len(minutes):
                    counts["without_exit"] += 1
                    writer.writerow([*common, "", "", "", "", "", "", "", "exit_unavailable"])
                    continue
                end = int(minutes[slot])
                if end - scheduled > 60 * max_exit_delay_minutes:
                    # A candle-cache gap would otherwise silently stretch the holding period.
                    counts["exit_gap_too_long"] += 1
                    writer.writerow([*common, "", "", "", "", "", "", "", "exit_gap_too_long"])
                    continue
                close = prices[end]
                gross, fees, net, label = short_outcome(fill_price, close, entry_fee, exit_fee)
                counts[str(label)] += 1
                writer.writerow([*common, iso(end), close, gross, fees, net, label, iso(end + 60), "complete"])
        try:
            os.link(timeline_path, staged / timeline_path.name)
        except OSError:
            shutil.copyfile(timeline_path, staged / timeline_path.name)
        keys = ("ticker", "start_inclusive", "end_exclusive", "snapshot_cutoff", "window_minutes",
                "feature_names_per_minute", "flatten_order", "macd_normalization", "feature_timeline_first_close",
                "features_per_example", "entry_commission_pct_per_fill", "exit_commission_pct_per_fill", "main_anchor_ids")
        target = {"hold_minutes": hold_minutes, "side": "short", "decision_clock": FILL_DECISION_CLOCK,
                  "hold_clock": "initial_fill_at", "max_exit_delay_minutes": max_exit_delay_minutes,
                  "exit_clock": "first_actual_m1_open_at_or_after_hold", "label_clock": "exit_m1_close",
                  "class_names": list(PROFIT_CLASSES)}
        run_manifest = {key: manifest[key] for key in keys if key in manifest}
        run_manifest.update(dataset=PROFIT_DATASET, profit_target=target, source_export=str(source.resolve()),
                            candle_cache=str(candles.resolve()), collection_counts=dict(counts))
        (staged / "manifest.json").write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")
        staged.rename(output)
    print("Profit bucket collection", json.dumps(dict(counts)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-minutes", type=int, default=DEFAULT_PROFIT_HOLD)
    parser.add_argument("--max-exit-delay-minutes", type=int, default=DEFAULT_MAX_EXIT_DELAY,
                        help="Leave a trade unlabeled if no M1 open exists within this many minutes of the timed exit")
    args = parser.parse_args()
    collect(args.source, args.candles, args.output, args.hold_minutes, args.max_exit_delay_minutes)


if __name__ == "__main__":
    main()
