"""Verify sampled first fills and estimate the wave-6 exit timing effect."""

from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
import json

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13"
RUN = ROOT / "Training/rocm-9070/runs/rx9070-20261120-023615-8a20"


def utc_ms(value):
    return int(datetime.fromisoformat(value).timestamp()) * 1000


def load_raw():
    files = (ROOT / "Exports/candle_cache/ETHUSD/1m").glob("*.json")
    rows = np.concatenate([np.asarray(json.loads(f.read_text(encoding="utf-8")))[:, :5] for f in files])
    rows = rows[rows[:, 0].argsort(kind="stable")]
    repeated = np.flatnonzero(np.diff(rows[:, 0]) == 0)
    np.testing.assert_array_equal(rows[repeated], rows[repeated + 1])
    return rows[np.r_[True, np.diff(rows[:, 0]) != 0]]


def first_fills(export, splits, raw):
    rng = np.random.default_rng(11)
    times = raw[:, 0]
    report = Counter()
    for part, indices in splits.items():
        selected = export.loc[indices]
        filled = selected.loc[selected.initial_order_status.eq("filled")]
        for label in (0, 1):
            candidates = filled.index[filled.binary_target.eq(label)]
            for eid in rng.choice(candidates, 20, replace=False):
                row = export.loc[eid]
                decision = utc_ms(row.entry_at)
                fill = utc_ms(row.initial_fill_at)
                j = np.searchsorted(times, decision) - 1
                gap = bool(times[j] != decision - 60_000)
                level = (raw[j, 2] if gap else raw[j, 4]) - .10
                end = np.searchsorted(times, fill, side="right")
                touched = np.flatnonzero(raw[j + 1:end, 4] <= level)
                assert len(touched), eid
                candle = raw[j + 1 + touched[0]]
                opened = bool(candle[1] <= level)
                expected_time = int(candle[0]) + (0 if opened else 60_000)
                expected_price = float(candle[1] if opened else level)
                assert fill == expected_time and abs(row.initial_fill_price - expected_price) < 1e-9, eid
                report[part] += 1
                report["fills_just_after_gap"] += gap
                report["open_fills" if opened else "intraminute_fills"] += 1
    return dict(report)


def wave6_timing(export, splits, raw):
    market = export.loc[export.wave6_market_exit_price.notna()].copy()
    wave_time = market.series_cutoff_at.map(utc_ms).to_numpy(dtype=np.int64)
    confirmed = wave_time + 300_000
    execution = market.wave6_market_exit_ms_utc.to_numpy(dtype=np.int64)
    raw_rows = np.searchsorted(raw[:, 0], execution)
    np.testing.assert_array_equal(raw[raw_rows, 0], execution)
    np.testing.assert_allclose(raw[raw_rows, 1], market.wave6_market_exit_price, atol=1e-9, rtol=0)
    before = execution < confirmed
    # For a single trade the initial fill also gives the final leg's entry price.
    row_ix = np.flatnonzero(before & market.trades_count.eq(1).to_numpy())
    insertion = np.searchsorted(raw[:, 0], confirmed[row_ix])
    observable = insertion < len(raw)
    studied = market.iloc[row_ix[observable]]
    exit_prices = raw[insertion[observable], 1]
    fill_prices = studied.initial_fill_price.to_numpy()
    initial = 100 * (fill_prices - studied.wave6_market_exit_price.to_numpy()) / fill_prices
    np.testing.assert_allclose(initial, studied.target_gross_return_pct, atol=.0000011, rtol=0)
    updated_gross = np.round(100 * (fill_prices - exit_prices) / fill_prices, 6)
    # Compare the same six-decimal gross-return rule to the same flat fees.
    fee = studied.commission_pct.to_numpy()
    now = np.array([Decimal(str(gross)) > Decimal(str(comm)) for gross, comm in zip(updated_gross, fee)])
    original = studied.binary_target.to_numpy(dtype=bool)
    count = {"recorded_wave6_market_exits": len(market), "market_exits_before_wave6_confirmation": int(before.sum()),
             "single_trade_exits_repriced": len(studied), "positive_to_negative": int((original & ~now).sum()),
             "negative_to_positive": int((~original & now).sum()),
             "confirmation_delay_minutes_median": float(np.median((confirmed[before] - execution[before]) / 60_000)),
             "median_gross_difference_percentage_points": float(np.median(updated_gross - studied.target_gross_return_pct)),
             "example": {}}
    flips = np.flatnonzero(original != now)
    for k in flips[:2]:
        row = studied.iloc[k]
        count["example"][str(row.name)] = {"entry_at": row.entry_at, "fill_price": float(row.initial_fill_price),
                                            "historical_exit_price": float(row.wave6_market_exit_price),
                                            "confirmation_exit_price": float(exit_prices[k]),
                                            "original_gross_percent": float(row.target_gross_return_pct),
                                            "causal_gross_percent": float(updated_gross[k]), "commission_percent": float(fee[k]),
                                            "historical_exit_ms": int(row.wave6_market_exit_ms_utc),
                                            "confirmation_exit_ms": int(raw[insertion[observable][k], 0])}
    count["flip_counts_by_split"] = {part: int(studied.index[flips].isin(entries).sum()) for part, entries in splits.items()}
    count["early_market_exits_by_split"] = {part: int(market.index[before].isin(entries).sum())
                                           for part, entries in splits.items()}
    return count


def main():
    export = pd.read_csv(Path(str(SOURCE) + "_bounded_macd") / "entry_targets.csv", index_col="entry_id")
    split_report = json.loads((RUN / "split_report.json").read_text(encoding="utf-8"))["splits"]
    splits = {key: np.array(part["entry_ids"], dtype=np.int64) for key, part in split_report.items()}
    raw = load_raw()
    result = {"sampled_initial_fills": first_fills(export, splits, raw), "wave6_counterfactual": wave6_timing(export, splits, raw)}
    (HERE / "raw_execution_checks.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
