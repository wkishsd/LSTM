"""Read Bitfinex windows from a shared memory-mapped minute timeline."""

import csv
import json
from calendar import monthrange
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from .data import Prepared, TimelinePaths, mmap_array
from .schema import (
    FILL_DECISION_CLOCK, HISTORY, INPUT_SIZE, PROFIT_CLASSES, PROFIT_DATASET, STATES, WIDTH, default_schema,
)


def export_paths(directory, entries=None):
    root = Path(directory)
    return TimelinePaths(str(root / "feature_timeline.npy"),
                         entries or str(root / "entry_targets.csv"),
                         str(root / "manifest.json"))


def timestamp(value):
    dt = datetime.fromisoformat(value)
    if dt.utcoffset() is None or dt.microsecond:
        raise ValueError("Expected timezone-aware timestamp at whole-second precision")
    seconds = int(dt.timestamp())
    if not 0 <= seconds <= 32_503_680_000:
        raise ValueError("Timestamp out of supported range")
    return seconds


def select_bitfinex_period(data, period):
    if period == "all":
        return data
    if period != "last_6_months":
        raise ValueError(f"Unknown Bitfinex data period: {period}")
    manifest = json.loads(Path(data.paths.manifest).read_text(encoding="utf-8"))
    end = datetime.fromtimestamp(timestamp(manifest["end_exclusive"]), timezone.utc)
    year, month_zero = divmod(end.year * 12 + end.month - 1 - 6, 12)
    month = month_zero + 1
    start = end.replace(year=year, month=month, day=min(end.day, monthrange(year, month)[1]))
    ix = data.valid
    selected = ix[(data.times[ix] >= int(start.timestamp())) & (data.times[ix] < int(end.timestamp()))]
    if not len(selected):
        raise ValueError("No eligible Bitfinex entries in the last six calendar months of the export")
    report = {**data.report, "accepted": len(selected), "out_of_period_count": len(ix) - len(selected),
              "data_period": {"start_inclusive": start.isoformat(), "end_exclusive": end.isoformat()}}
    return replace(data, valid=selected, report=report)


def export_schema(manifest):
    if (manifest["window_minutes"] != STATES or manifest["features_per_example"] != INPUT_SIZE
            or manifest["flatten_order"] != "oldest minute first, features in listed order"):
        raise ValueError("Bitfinex export must use 1410 minutes and 76140 features in chronological order")
    expected_features = []
    for tf in ("1m", "5m", "15m", "30m"):
        expected_features.append("norm_price" if tf == "1m" else f"norm_price_{tf}")
        expected_features.extend(
            f"norm_{feature}_{tf}" for feature in (
                "volume", "rsi_14", "atr_14", "macd_12_26", "macd_signal_12_26_9",
                "adx_14_4", "di_plus_14", "di_minus_14",
            )
        )
    anchor_ids = manifest["main_anchor_ids"]
    if (len(anchor_ids) != 18 or any(type(n) is not int or n < 1 for n in anchor_ids)
            or len(set(anchor_ids)) != 18):
        raise ValueError("Bitfinex manifest must contain 18 unique positive anchor IDs")
    anchors = [f"anchor_{n:02d}" for n in anchor_ids]
    if manifest["feature_names_per_minute"] != expected_features + anchors:
        raise ValueError("Unexpected Bitfinex feature or anchor order")
    schema = default_schema()
    if "macd_normalization" in manifest:
        transform = manifest["macd_normalization"]
        if transform != "x / sqrt(1 + x^2)":
            raise ValueError(f"Unsupported Bitfinex MACD normalization: {transform!r}")
        schema["macd_normalization"] = transform
    columns = schema["current_columns"][:36] + anchors
    schema["current_columns"] = schema["history_columns"] = columns
    if manifest.get("dataset") == PROFIT_DATASET:
        target = manifest["profit_target"]
        if (target.get("class_names") != list(PROFIT_CLASSES) or target.get("side") != "short"
                or target.get("decision_clock") != FILL_DECISION_CLOCK
                or target.get("hold_clock") != "initial_fill_at"
                or type(target.get("max_exit_delay_minutes")) is not int or target["max_exit_delay_minutes"] < 0
                or target.get("exit_clock") != "first_actual_m1_open_at_or_after_hold"
                or target.get("label_clock") != "exit_m1_close"
                or type(target.get("hold_minutes")) is not int or target["hold_minutes"] < 1):
            raise ValueError("Unsupported Bitfinex profit target contract; re-collect the profit export "
                             "(older exports let the fill candle into the feature window)")
        schema["profit_target"] = {
            **target, "entry_commission_pct_per_fill": str(nonnegative_decimal(manifest["entry_commission_pct_per_fill"])),
            "exit_commission_pct_per_fill": str(nonnegative_decimal(manifest["exit_commission_pct_per_fill"])),
        }
    return schema


def bad_row_prefix(timeline):
    # Scan each minute once, then check a full window with two prefix lookups.
    prefix = np.zeros(len(timeline) + 1, dtype=np.int64)
    for start in range(0, len(timeline), 16384):
        block = timeline[start:start + 16384]
        invalid = (~np.isfinite(block) | (block > np.finfo(np.float32).max)
                   | (block < -np.finfo(np.float32).max)).any(axis=1)
        anchors = block[:, 36:]
        invalid |= ((anchors != 0) & (anchors != 1)).any(axis=1)
        prefix[start + 1:start + len(block) + 1] = prefix[start] + np.cumsum(invalid)
    return prefix


def nonnegative_decimal(value):
    number = Decimal(str(value))
    if not number.is_finite() or number < 0:
        raise ValueError("Expected a finite nonnegative amount")
    return number


def prepare_bitfinex(paths, *, training):
    manifest = json.loads(Path(paths.manifest).read_text(encoding="utf-8"))
    schema = export_schema(manifest)
    profit = schema.get("profit_target")
    first = timestamp(manifest["feature_timeline_first_close"])
    timeline = mmap_array(paths.timeline, "feature_timeline", (WIDTH,))
    if first % 60 or len(timeline) < STATES:
        raise ValueError("Bitfinex timeline must start at a minute boundary and contain at least 1410 rows")
    required = {"entry_id", "entry_at", "feature_end_row"}
    if profit:
        required.add("signal_at")
    if training:
        required.add("label_available_at")
        if profit:
            from .profit_collector import short_outcome
            required |= {"initial_fill_at", "initial_fill_price", "scheduled_exit_at", "exit_price", "exit_at",
                         "profit_class"}
        else:
            required |= {"target_gross_return_pct", "commission_pct", "filled_entry_orders", "filled_exit_orders",
                         "initial_order_status"}
        entry_fee = nonnegative_decimal(manifest["entry_commission_pct_per_fill"])
        exit_fee = nonnegative_decimal(manifest["exit_commission_pct_per_fill"])
    with Path(paths.entries).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"Missing Bitfinex CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        rows = list(reader)
    ids = np.array([str(row["entry_id"] or "") for row in rows], dtype=str)
    if any(not value.strip() for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("Bitfinex CSV must have nonempty and unique entry_id values")
    times, starts, ends, positions = (np.zeros(len(rows), dtype=np.int64) for _ in range(4))
    labels = np.zeros(len(rows), dtype=np.int64) if training else None
    returns = np.full(len(rows), np.nan) if training else None
    prefix = bad_row_prefix(timeline)
    valid, excluded, counts = [], [], Counter()
    for i, row in enumerate(rows):
        reasons = []
        try:
            pos = int(row["feature_end_row"])
            t = timestamp(row["entry_at"])
            if not HISTORY <= pos < len(timeline) or t != first + pos * 60:
                raise ValueError("Minute index and entry timestamp disagree")
        except (TypeError, ValueError):
            reasons.append("invalid_window_position_or_time")
        else:
            times[i], starts[i], ends[i], positions[i] = t, t - HISTORY * 60, t, pos
            if prefix[pos + 1] != prefix[pos - HISTORY]:
                reasons.append("invalid_features_or_anchors")
            if profit:
                try:
                    signal = timestamp(row["signal_at"])
                    signal_pos, remainder = divmod(signal - first, 60)
                    if remainder or not 0 <= signal_pos <= pos or not np.any(timeline[signal_pos, 36:] == 1):
                        raise ValueError("Missing source anchor before the confirmed fill")
                except (TypeError, ValueError):
                    reasons.append("invalid_fill_anchor")
            elif not np.any(timeline[pos, 36:] == 1):
                reasons.append("no_current_anchor")
        if training and not profit and row["initial_order_status"] != "filled":
            # The export labels canceled orders 0, but "never filled" is not "lost money".
            reasons.append("unfilled_initial_order")
        elif training:
            if not row["label_available_at"] or not row["profit_class" if profit else "target_gross_return_pct"]:
                reasons.append("unresolved_outcome")
            else:
                try:
                    end = timestamp(row["label_available_at"])
                    if end < times[i]:
                        raise ValueError("Outcome precedes entry")
                    ends[i] = end
                    if profit:
                        exit_at = timestamp(row["exit_at"])
                        fill_at = timestamp(row["initial_fill_at"])
                        scheduled = timestamp(row["scheduled_exit_at"])
                        if (fill_at < times[i] or scheduled != fill_at + 60 * profit["hold_minutes"]
                                or exit_at % 60 or end != exit_at + 60 or exit_at < scheduled
                                or exit_at - scheduled > 60 * profit["max_exit_delay_minutes"]):
                            raise ValueError("Timed exit disagrees with the fill, hold or label availability")
                        _, _, net, label = short_outcome(Decimal(row["initial_fill_price"]),
                                                         Decimal(row["exit_price"]), entry_fee, exit_fee)
                        if label != int(row["profit_class"]):
                            raise ValueError("Reported profit class disagrees with prices and commissions")
                        labels[i] = label
                        returns[i] = float(net)
                    else:
                        gross = Decimal(row["target_gross_return_pct"])
                        entered, exited = int(row["filled_entry_orders"]), int(row["filled_exit_orders"])
                        if not gross.is_finite() or min(entered, exited) < 0:
                            raise ValueError("Invalid outcome")
                        fee = entry_fee * entered + exit_fee * exited
                        reported = nonnegative_decimal(row["commission_pct"])
                        # CSV amounts are serialized from floats. Check against the exact rate/count product.
                        if abs(fee - reported) > max(Decimal("1e-12"), fee * Decimal("1e-12")):
                            raise ValueError("Reported commission disagrees with executed orders")
                        labels[i] = int(gross > fee)
                        # Series return in percentage points: summed per-trade results minus all fills' fees.
                        returns[i] = float(gross - fee)
                except (TypeError, ValueError, InvalidOperation):
                    reasons.append("invalid_outcome_or_commission")
        if reasons:
            counts.update(reasons)
            excluded.append({"entry_id": str(ids[i]), "reasons": reasons})
        else:
            valid.append(i)
    report = {"current_rows": len(rows), "union_ids": len(rows), "accepted": len(valid),
              "excluded_count": len(excluded), "reason_counts": dict(counts), "excluded": excluded,
              "outcome_end_source": "label_available_at" if training else None}
    if profit:
        report["profit_target"] = profit
        report["collection_counts"] = manifest.get("collection_counts", {})
    return Prepared(paths, schema, ids, times, starts, ends, positions, labels,
                    np.array(valid, dtype=np.int64), report, returns)
