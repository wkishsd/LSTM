"""Native Bitfinex 30m MACD and red-start cycle-length research.

Run: python macd_30m_cycle_analysis.py --candles 100000
Only the trade:30m endpoint is called. No 1m collection or resampling exists.
CSV/JSON/NumPy outputs are self-contained. --xlsx invokes the companion
macd_30m_report.mjs using the bundled Node/artifact-tool runtime or --node.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
try:
    import requests
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".python-deps"))
    import requests


BASE_URL = "https://api-pub.bitfinex.com/v2"
TIMEFRAME = "30m"
MINUTES_PER_CANDLE = 30
STEP_MS = MINUTES_PER_CANDLE * 60_000
API_PAGE_LIMIT = 10_000
DEFAULT_TIMEOUT = 30.0
DEFAULT_REQUEST_INTERVAL = 2.1
DISPLAY_OFFSET = "Etc/GMT-3"  # Fixed UTC+3, without daylight saving time.
SOURCE_URL = "https://docs.bitfinex.com/reference/rest-public-candles"
CANDLE_COLUMNS = ["timestamp_ms", "open", "close", "high", "low", "volume"]


class CandleCollector:
    """Paginate native 30m candles with timeout, spacing and bounded retries."""

    def __init__(self, timeout=DEFAULT_TIMEOUT, interval=DEFAULT_REQUEST_INTERVAL):
        self.timeout, self.interval = timeout, interval
        self.session = requests.Session()
        self.last_request_at = None
        self.requests_made = 0
        self.endpoint = None

    def request_page(self, symbol, limit, end_ms):
        api_symbol = "t" + symbol.removeprefix("t").upper()
        self.endpoint = f"{BASE_URL}/candles/trade:{TIMEFRAME}:{api_symbol}/hist"
        for attempt in range(6):
            if self.last_request_at is not None:
                delay = self.interval - (time.monotonic() - self.last_request_at)
                if delay > 0:
                    time.sleep(delay)
            self.last_request_at = time.monotonic()
            self.requests_made += 1
            try:
                response = self.session.get(
                    self.endpoint,
                    params={"limit": limit, "sort": -1, "end": int(end_ms)},
                    timeout=self.timeout,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt == 5:
                        response.raise_for_status()
                    try:
                        retry_after = float(response.headers.get("Retry-After", 0))
                    except ValueError:
                        retry_after = 0
                    time.sleep(max(retry_after, self.interval * 2**attempt))
                    continue
                response.raise_for_status()
                page = response.json()
                if (not isinstance(page, list) or
                        any(not isinstance(r, list) or len(r) != 6 for r in page)):
                    raise RuntimeError(f"Unexpected Bitfinex candle response: {str(page)[:160]}")
                return page
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 5:
                    raise
                time.sleep(self.interval * 2**attempt)
        raise RuntimeError("Candle retries exhausted")

    def fetch(self, symbol, count, cutoff):
        last_open_ms = int(cutoff.floor("30min").timestamp() * 1000) - STEP_MS
        end_ms, records = last_open_ms, {}
        while len(records) < count:
            page = self.request_page(symbol, min(API_PAGE_LIMIT, count - len(records)), end_ms)
            if not page:
                break
            oldest = min(int(r[0]) for r in page)
            if oldest > end_ms:
                raise RuntimeError("Pagination did not move backwards")
            before = len(records)
            for row in page:
                stamp = int(row[0])
                if stamp > end_ms or stamp > last_open_ms:
                    raise RuntimeError("The API returned a candle outside the requested closed range")
                if stamp in records and records[stamp] != row:
                    raise RuntimeError("Conflicting duplicate candle timestamp")
                records[stamp] = row
            if len(records) == before:
                raise RuntimeError("No new native candles were collected")
            print(f"Fetched {len(records):,}/{count:,} native 30m candles", flush=True)
            end_ms = oldest - 1
        if len(records) < count:
            raise RuntimeError(f"Requested {count:,} native candles; API returned {len(records):,}")
        return [records[key] for key in sorted(records)][-count:]


def prepare_timeline(records):
    """Keep native candles intact and flag causal flat slots for internal gaps."""
    raw = pd.DataFrame(records, columns=CANDLE_COLUMNS).sort_values("timestamp_ms")
    raw["timestamp_ms"] = raw.timestamp_ms.astype("int64")
    if raw.timestamp_ms.duplicated().any() or (raw.timestamp_ms % STEP_MS != 0).any():
        raise ValueError("Native candle timestamps must be unique and aligned to 30m")
    values = raw[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if (not np.isfinite(values).all() or (raw[["open", "high", "low", "close"]] <= 0).any().any()
            or (raw.volume < 0).any() or (raw.high < raw[["open", "close", "low"]].max(axis=1)).any()
            or (raw.low > raw[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("Invalid native OHLCV values")
    grid = np.arange(int(raw.timestamp_ms.iloc[0]), int(raw.timestamp_ms.iloc[-1]) + STEP_MS,
                     STEP_MS, dtype=np.int64)
    full = raw.set_index("timestamp_ms").reindex(grid)
    missing = full.close.isna()
    previous_close = full.close.ffill()
    for column in ("open", "high", "low", "close"):
        full.loc[missing, column] = previous_close.loc[missing]
    full.loc[missing, "volume"] = 0.0
    full["is_synthetic"] = missing.astype(int)
    full.index.name = "timestamp_ms"
    return full.reset_index(), raw.reset_index(drop=True)


def ema(values, period):
    """Same EMA as the main indicator script: recursive, first-value seed."""
    return values.ewm(span=period, adjust=False, min_periods=period).mean()


def calculate_macd(candles, fast=12, slow=26, signal=9):
    """Extracted MACD/signal calculation; no other indicators are calculated."""
    result = candles.copy()
    result["macd"] = ema(result.close, fast) - ema(result.close, slow)
    result["signal"] = ema(result.macd.dropna(), signal).reindex(result.index)
    result["hist"] = result.macd - result.signal
    return result


def find_red_start_cycles(calculated, count, percentile=90.0, start_ms=None):
    """Return a separate histogram array, red-start indices and interval array.

    Positive hist is green, negative is red; exact zeros retain the preceding
    nonzero sign, matching the original half-wave helper. A new red wave is
    the first negative value after a positive half-wave, not a darker shade.
    """
    full = calculated.copy()
    full["hist_sign"] = np.sign(full["hist"]).replace(0, np.nan).ffill()
    full["green_to_red"] = ((full.hist_sign.shift(1) > 0) & (full.hist_sign < 0)).astype(int)
    full["red_to_green"] = ((full.hist_sign.shift(1) < 0) & (full.hist_sign > 0)).astype(int)
    result = (full.tail(count) if start_ms is None else full.loc[full.timestamp_ms >= start_ms]).reset_index(drop=True)
    if ((start_ms is None and len(result) != count)
            or (start_ms is not None and int(result.is_synthetic.eq(0).sum()) != count)
            or not np.isfinite(result[["macd", "signal", "hist"]]).all().all()):
        raise ValueError("Insufficient valid, warmed-up MACD output")
    result.insert(0, "candle_no", np.arange(1, len(result) + 1, dtype=np.int64))
    for name, delta in (("candle_open", 0), ("available_at", STEP_MS)):
        result[name] = pd.to_datetime(result.timestamp_ms + delta, unit="ms", utc=True)
    result["hist_color"] = np.where(result.hist_sign < 0, "red", "green")
    result["exact_zero_hist"] = result["hist"].eq(0).astype(int)
    hist_values = result["hist"].to_numpy(dtype=np.float64, copy=True)
    red_starts = np.flatnonzero(result.green_to_red.to_numpy(dtype=bool))
    interval_lengths = np.diff(red_starts).astype(np.int64)
    if not len(interval_lengths):
        raise ValueError("Fewer than two green-to-red crossings in the analysis range")
    next_green = np.flatnonzero(result.red_to_green.to_numpy(dtype=bool))
    interval_rows = []
    for i, (start, stop) in enumerate(zip(red_starts[:-1], red_starts[1:]), 1):
        green_candidates = next_green[(next_green > start) & (next_green < stop)]
        if len(green_candidates) != 1:
            raise ValueError("A completed red-start cycle must contain exactly one red-to-green crossing")
        green = int(green_candidates[0])
        length = int(stop - start)
        a, b = result.iloc[start], result.iloc[stop]
        interval_rows.append({
            "cycle_no": i, "start_candle_no": int(a.candle_no), "end_candle_no": int(b.candle_no),
            "start_open_ms": int(a.timestamp_ms), "end_open_ms": int(b.timestamp_ms),
            "start_available_ms": int(a.timestamp_ms + STEP_MS),
            "end_available_ms": int(b.timestamp_ms + STEP_MS),
            "cycle_candles": length, "strictly_between_candles": length - 1,
            "window_including_both_starts": length + 1,
            "red_halfwave_candles": green - int(start),
            "green_halfwave_candles": int(stop) - green,
            "elapsed_hours": length / 2.0,
            "start_hist": float(a["hist"]), "end_hist": float(b["hist"]),
        })
    cycles = pd.DataFrame(interval_rows)
    ordered = np.sort(interval_lengths)
    rank = math.ceil(percentile / 100 * len(ordered))
    threshold = int(ordered[rank - 1])
    covered = int(np.count_nonzero(interval_lengths <= threshold))
    covered_previous = int(np.count_nonzero(interval_lengths < threshold))
    summary = {
        "analysis_native_candles_requested": count,
        "analysis_candles": len(result),
        "native_candles_in_analysis": int(result.is_synthetic.eq(0).sum()),
        "synthetic_slots_in_analysis": int(result.is_synthetic.sum()),
        "green_to_red_crossings": int(len(red_starts)),
        "completed_cycles": int(len(interval_lengths)),
        "percentile": percentile, "nearest_rank": rank,
        "p90_cycle_candles": threshold,
        "p90_hours": threshold / 2,
        "p90_days": threshold / 48,
        "p90_strictly_between_candles": threshold - 1,
        "p90_window_including_both_starts": threshold + 1,
        "cycles_at_or_below_p90": covered,
        "coverage_at_p90": covered / len(ordered),
        "coverage_one_candle_shorter": covered_previous / len(ordered),
        "p90_linear_interpolated": float(np.percentile(ordered, percentile)),
        "minimum_cycle_candles": int(ordered[0]),
        "median_cycle_candles": float(np.median(ordered)),
        "mean_cycle_candles": float(np.mean(ordered)),
        "maximum_cycle_candles": int(ordered[-1]),
        "leading_candles_before_first_crossing": int(red_starts[0]),
        "incomplete_tail_candles": int(len(result) - red_starts[-1]),
        "first_open_utc": result.candle_open.iloc[0].isoformat(),
        "last_open_utc": result.candle_open.iloc[-1].isoformat(),
        "last_available_utc": result.available_at.iloc[-1].isoformat(),
    }
    result["last_red_start_candle"] = result.candle_no.where(result.green_to_red.eq(1)).ffill().astype("Int64")
    result["candles_since_red_start"] = result.candle_no - result.last_red_start_candle
    result["completed_cycle_candles"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result.loc[red_starts[1:], "completed_cycle_candles"] = interval_lengths
    cycles["at_or_below_p90"] = (cycles.cycle_candles <= threshold).astype(int)
    return result, cycles, summary, hist_values, red_starts, interval_lengths


def utc3_strings(values):
    return pd.to_datetime(values, unit="ms", utc=True).dt.tz_convert(DISPLAY_OFFSET).dt.strftime("%Y-%m-%d %H:%M")


def summarize_monthly_cycles(cycles, timeline, p90_threshold):
    """Group completed cycles by the UTC+3 month of their ending candle close."""
    first = timeline.candle_open.iloc[0].tz_convert(DISPLAY_OFFSET).tz_localize(None)
    last = timeline.available_at.iloc[-1].tz_convert(DISPLAY_OFFSET).tz_localize(None)
    end_months = (pd.to_datetime(cycles["end_available_ms"], unit="ms", utc=True)
                  .dt.tz_convert(DISPLAY_OFFSET).dt.tz_localize(None).dt.to_period("M"))
    grouped = {month: group["cycle_candles"].to_numpy(dtype=np.int64)
               for month, group in cycles.groupby(end_months)}
    rows = []
    for month in pd.period_range(first, last, freq="M"):
        lengths = grouped.get(month, np.empty(0, dtype=np.int64))
        n = len(lengths)
        total = int(lengths.sum())
        covered = int(np.count_nonzero(lengths <= p90_threshold))
        rows.append({
            "month_utc3": month.start_time.strftime("%Y-%m-%d %H:%M"),
            "sample": ("Full month" if first <= month.start_time
                       and last >= (month + 1).start_time else "Partial month"),
            "completed_cycles": n,
            "total_cycle_periods": total,
            "mean_cycle_periods": total / n if n else None,
            "mean_cycle_hours": total * MINUTES_PER_CANDLE / (60 * n) if n else None,
            "p90_cycle_periods": int(np.sort(lengths)[math.ceil(0.90 * n) - 1]) if n else None,
            "cycles_at_or_below_global_p90": covered,
            "share_at_or_below_global_p90": covered / n if n else None,
        })
    return pd.DataFrame(rows)


def export_analysis(output_dir, timeline, cycles, summary, arrays):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = timeline.copy()
    rows.insert(1, "open_utc3", rows.candle_open.dt.tz_convert(DISPLAY_OFFSET).dt.strftime("%Y-%m-%d %H:%M"))
    rows.insert(2, "available_utc3", rows.available_at.dt.tz_convert(DISPLAY_OFFSET).dt.strftime("%Y-%m-%d %H:%M"))
    rows = rows.drop(columns=["candle_open", "available_at", "hist_sign"])
    rows.to_csv(output_dir / "candles_30m.csv", index=False, float_format="%.15g")
    exported_cycles = cycles.copy()
    for col in ("start_open", "end_open", "start_available", "end_available"):
        exported_cycles[col + "_utc3"] = utc3_strings(exported_cycles[col + "_ms"])
    exported_cycles.to_csv(output_dir / "red_start_cycles.csv", index=False, float_format="%.15g")
    monthly = summarize_monthly_cycles(cycles, timeline, summary["p90_cycle_candles"])
    monthly.assign(month_utc3=monthly["month_utc3"].str[:7]).to_csv(
        output_dir / "monthly_cycle_lengths.csv", index=False, float_format="%.15g")
    values, frequencies = np.unique(arrays[2], return_counts=True)
    distribution = pd.DataFrame({"cycle_candles": values, "cycle_count": frequencies})
    distribution["cumulative_count"] = distribution.cycle_count.cumsum()
    distribution["cumulative_share"] = distribution.cumulative_count / len(arrays[2])
    distribution["hours"] = distribution.cycle_candles / 2
    distribution.to_csv(output_dir / "length_distribution.csv", index=False, float_format="%.15g")
    np.savez_compressed(output_dir / "analysis_arrays.npz", hist_values=arrays[0],
                        red_start_indices=arrays[1], interval_lengths=arrays[2],
                        timestamp_ms=timeline.timestamp_ms.to_numpy(dtype=np.int64))
    pd.DataFrame(list(summary.items()), columns=["metric", "value"]).to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Typed, Excel-ready input for the optional artifact-tool workbook exporter.
    date_cols = {"open_utc3", "available_utc3", "start_open_utc3", "end_open_utc3",
                 "start_available_utc3", "end_available_utc3", "month_utc3"}
    candle_cols = ["candle_no", "open_utc3", "available_utc3", "open", "high", "low", "close", "volume",
                   "macd", "signal", "hist", "hist_color", "green_to_red", "red_to_green",
                   "completed_cycle_candles", "candles_since_red_start", "is_synthetic"]
    cycle_cols = ["cycle_no", "start_open_utc3", "end_open_utc3", "start_available_utc3", "end_available_utc3",
                  "start_candle_no", "end_candle_no", "cycle_candles", "strictly_between_candles",
                  "window_including_both_starts", "red_halfwave_candles", "green_halfwave_candles",
                  "elapsed_hours", "start_hist", "end_hist", "at_or_below_p90"]
    def package(frame, columns):
        clean = frame[columns].astype(object).where(pd.notna(frame[columns]), None)
        return {"columns": columns, "rows": clean.values.tolist(),
                "date_columns": [x for x in columns if x in date_cols]}
    payload = {"summary": summary, "candles": package(rows, candle_cols),
               "cycles": package(exported_cycles, cycle_cols),
               "monthly": package(monthly, list(monthly.columns)),
               "distribution": package(distribution, list(distribution.columns))}
    (output_dir / "workbook_data.json").write_text(json.dumps(payload, allow_nan=False, separators=(",", ":")), encoding="utf-8")


def export_xlsx(output_dir, node_path=None):
    exporter = Path(__file__).with_name("macd_30m_report.mjs")
    if not exporter.is_file():
        raise FileNotFoundError(f"Companion workbook exporter is required: {exporter}")
    if node_path is None:
        bundle = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe"
        if bundle.is_file():
            node_path = str(bundle)
        else:
            import shutil
            node_path = shutil.which("node")
    if node_path is None:
        raise RuntimeError("CSV/JSON/NumPy files are complete; pass --node to produce the optional XLSX")
    subprocess.run([str(node_path), "--max-old-space-size=10000", str(exporter), str(output_dir.resolve())], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="ETHUSD")
    parser.add_argument("--candles", type=int, default=100_000)
    parser.add_argument("--warmup", type=int, default=260)
    parser.add_argument("--fast", type=int, default=12)
    parser.add_argument("--slow", type=int, default=26)
    parser.add_argument("--signal", type=int, default=9)
    parser.add_argument("--end", help="UTC cutoff, e.g. 2026-11-07T12:00:00Z; default now")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "Exports/macd_30m")
    parser.add_argument("--cache", type=Path, help="Replay a saved native_30m_snapshot.json without network calls")
    parser.add_argument("--request-interval", type=float, default=DEFAULT_REQUEST_INTERVAL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--xlsx", action="store_true", help="Also run the companion artifact-tool XLSX exporter")
    parser.add_argument("--node", help="Optional Node executable for --xlsx")
    args = parser.parse_args()
    args.percentile = 90.0
    if (args.candles < 2 or not 0 < args.fast < args.slow or args.signal < 1
            or args.warmup < max(args.slow * 10, args.slow + args.signal)
            or not 0 < args.percentile <= 100 or args.request_interval < 2.1 or args.timeout <= 0):
        parser.error("Require positive periods, fast < slow, warmup >= 10*slow, percentile in (0,100], and interval >= 2.1 seconds")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.cache:
        snapshot = json.loads(args.cache.read_text(encoding="utf-8"))
        if snapshot["timeframe"] != TIMEFRAME or snapshot["symbol"] != args.symbol.removeprefix("t").upper():
            raise ValueError("Cache symbol/timeframe does not match this run")
        cutoff = pd.Timestamp(snapshot["cutoff_utc"])
        records = snapshot["records"]
        request_count = 0
    else:
        cutoff = pd.Timestamp(args.end) if args.end else pd.Timestamp.now(tz="UTC")
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        collector = CandleCollector(args.timeout, args.request_interval)
        records = collector.fetch(args.symbol, args.candles + args.warmup, cutoff)
        request_count = collector.requests_made
        snapshot = {"symbol": args.symbol.removeprefix("t").upper(), "timeframe": TIMEFRAME,
                    "cutoff_utc": cutoff.isoformat(), "endpoint": collector.endpoint,
                    "api_requests": request_count, "records": records}
    snapshot_path = args.output_dir / "native_30m_snapshot.json"
    snapshot_path.write_text(json.dumps(snapshot, separators=(",", ":")), encoding="utf-8")
    full, native = prepare_timeline(records)
    if len(full) < args.candles + args.warmup:
        raise ValueError("Snapshot has insufficient warm-up history")
    if int(full.timestamp_ms.iloc[-1]) + STEP_MS > int(cutoff.timestamp() * 1000):
        raise ValueError("Snapshot contains an unclosed candle")
    calculated = calculate_macd(full, args.fast, args.slow, args.signal)
    analysis_start_ms = int(native.tail(args.candles).timestamp_ms.iloc[0])
    rows, cycles, summary, hist_values, red_starts, interval_lengths = find_red_start_cycles(
        calculated, args.candles, args.percentile, start_ms=analysis_start_ms)
    summary.update({
        "symbol": snapshot["symbol"], "timeframe": TIMEFRAME, "macd_fast": args.fast,
        "macd_slow": args.slow, "macd_signal": args.signal,
        "warmup_requested": args.warmup, "warmup_slots_used": len(full) - len(rows),
        "native_candles_fetched": len(native), "native_1m_candles_fetched": 0,
        "api_requests_this_run": request_count,
        "api_requests_original_snapshot": snapshot.get("api_requests"), "cutoff_utc": cutoff.isoformat(),
        "source_endpoint": snapshot["endpoint"], "source_documentation": SOURCE_URL,
        "snapshot_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        "candle_gap_policy": "Internal missing 30m slots use prior close OHLC and zero volume, with is_synthetic=1; no 1m data",
        "ema_method": "pandas ewm(span=period, adjust=False, min_periods=period), first-value initialization; signal uses finite MACD only",
        "crossing_rule": "First negative hist after positive half-wave; exact zeros retain previous nonzero sign",
        "interval_definition": "Consecutive green-to-red start-index difference; includes one red and one green half-wave",
        "percentile_method": "Nearest rank: sorted_lengths[ceil(p*N)-1], the minimum integer interval covering at least p of completed cycles",
        "boundary_policy": "Both red starts must lie inside the analysis range; leading and incomplete final cycles excluded",
        "display_timezone": "UTC+3 fixed",
    })
    export_analysis(args.output_dir, rows, cycles, summary, (hist_values, red_starts, interval_lengths))
    print(json.dumps(summary, indent=2), flush=True)
    if args.xlsx:
        export_xlsx(args.output_dir, args.node)


if __name__ == "__main__":
    main()
