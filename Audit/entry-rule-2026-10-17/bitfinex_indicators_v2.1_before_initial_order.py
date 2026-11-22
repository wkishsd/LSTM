#!/usr/bin/env python3
"""Bitfinex indicators v2.1: native timeframes aligned to closed 1m candles.

Install dependencies: pip install numpy pandas requests openpyxl
Run: python bitfinex_indicators_v2.1.py  # main-anchor training table, 2000 cases
     python bitfinex_indicators_v2.1.py --indicators-only  # original all-candle workbook
Training API: collect_training_datasets() returns separate features, model_features and targets.

For use from other Python code:
    datasets = collect_datasets()
    raw_data = datasets.raw
    normalized_data = datasets.normalized
    anchor_data = datasets.anchors
    overlap_report = datasets.overlap
These are separate DataFrames with matching row_id indexes. The Excel export
keeps raw_data, normalized_data and anchor_data in one workbook. Anchor rules
use closed 5m candles; their IDs appear on the M1 row at the next 5m opening.

Each row describes a 1m candle whose indicator features become available at
its close, recorded in available_at. Higher-timeframe features use only
candles that have already closed by available_at. Source candle columns show
the opening time identifying each completed candle. Excel times use UTC+3.
"""

from datetime import datetime, timedelta, timezone
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import time
import textwrap

import numpy as np
import pandas as pd
try:
    import requests
except ModuleNotFoundError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent / ".python-deps"))
    import requests
from openpyxl.styles import Alignment, Font
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter

VERSION = "2.1"
BASE_URL = "https://api-pub.bitfinex.com/v2"
API_REQUEST_TIMEOUT_SECONDS = 30
API_MIN_INTERVAL_SECONDS = 2.1  # The candle endpoint permits 30 requests/min.
API_MAX_PAGE_SIZE = 10000
API_MAX_ATTEMPTS = 4
_last_api_request_at = None

# =========================
# EDIT THESE PROJECT VALUES
# =========================
TICKER = "ETHUSD"
LOOKBACK_CANDLES = 20000  # Number of closed 1m candles in the output.
BASE_TIMEFRAME = "1m"
TIMEFRAMES = ("1m", "5m", "15m", "30m")
OUTPUT_FILE = Path(__file__).with_name("bitfinex_indicators_v2.1.xlsx")
DISPLAY_TIMEZONE = timezone(timedelta(hours=3))  # Fixed UTC+3 throughout the year.
DISPLAY_TIMEZONE_LABEL = "UTC+3"

# Apply these SAME periods separately to every native timeframe.
RSI_PERIOD = 14
ATR_PERIOD = 14
MACD_FAST_PERIOD = 12
MACD_SLOW_PERIOD = 26
MACD_SIGNAL_PERIOD = 9
DI_PERIOD = 14
ADX_PERIOD = 4
WARMUP_MULTIPLIER = 10

# Former anchor 23 was removed on 2026-10-14. Former 24..36 are now 23..35.
# No group assignments. Keep this mapping when reading older exported data.
ANCHOR_SCHEMA_VERSION = "35-renumbered"
LEGACY_ANCHOR_ID_MAP = {old: old if old < 23 else old - 1
                        for old in range(1, 37) if old != 23}
ANCHOR_TIMEFRAME = "5m"
ANCHOR_OVERLAP_TREND_INTERVAL = "1h"
ANCHOR_DOCUMENT_URL = "https://docs.google.com/document/d/13-9HR0mEcAhQ7WrKXFY0Cw4QPH3R1xc8wR0N-xpn8Pg/edit"
ANCHOR_RULES = {
    1: "RSI пересекает 50 вниз.",
    2: "RSI пересекает 30 вниз.",
    3: "RSI пересекает 70 вниз.",
    4: "RSI пересекает 60 вниз.",
    5: "RSI меняет рост на снижение; предыдущее значение >70.",
    6: "RSI пересекает вниз среднее своих предыдущих 5 значений.",
    7: "MACD пересекает ноль вниз.",
    8: "MACD пересекает Signal вниз.",
    9: "Hist меняет рост на снижение, оставаясь >0.",
    10: "Hist меняет рост на снижение, оставаясь <0.",
    11: "MACD меняет рост на снижение, оставаясь >0.",
    12: "Signal пересекает ноль вниз.",
    13: "DI− пересекает DI+ вверх.",
    14: "ADX пересекает 25 вверх; DI− > DI+.",
    15: "ADX меняет снижение на рост; предыдущее значение ≤20, DI− > DI+.",
    16: "Разница DI− − DI+ пересекает 10 пунктов вверх.",
    17: "DI− пересекает 25 вверх; DI− > DI+.",
    18: "DI+ пересекает 20 вниз; DI− > DI+.",
    19: "ADX меняет рост на снижение; предыдущее значение >35, DI+ > DI−.",
    20: "Впервые DI− растет, а DI+ уменьшается 3 последовательных периода.",
    21: "ADX пересекает 25 вниз; DI+ > DI−.",
    22: "MACD пересекает Signal вниз при ADX >25.",
    23: "Впервые DI− > DI+ на протяжении 3 последовательных периодов.",
    24: "Hist опускается ниже минимума своих предыдущих 10 значений; MACD <0.",
    25: "ATR пересекает вверх 75-й процентиль своих предыдущих 80 значений; DI− > DI+.",
    26: "Объем впервые >2 × средний объем предыдущих 20 периодов; DI− > DI+.",
    27: "ATR пересекает вверх среднее своих предыдущих 20 значений; DI− > DI+.",
    28: "RSI опускается ниже минимума своих предыдущих 10 значений, оставаясь <50.",
    29: "RSI пересекает 70 вверх.",
    30: "Signal меняет рост на снижение, оставаясь >0.",
    31: "Впервые Hist уменьшается 3 последовательных периода; все 3 значения >0.",
    32: "DI+ меняет рост на снижение, оставаясь выше DI−.",
    33: "Прирост DI− за период впервые >5 пунктов; DI− > DI+.",
    34: "Объем впервые >3 × средний объем предыдущих 20 периодов; RSI >70.",
    35: "ATR уменьшается после 3 последовательных увеличений; DI+ > DI−.",
}

TIMEFRAME_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}
CANDLE_COLUMNS = ["timestamp", "timestamp_ms", "open", "high", "low", "close", "volume"]
INDICATOR_COLUMNS = (
    f"rsi_{RSI_PERIOD}", f"atr_{ATR_PERIOD}",
    f"macd_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}",
    f"macd_signal_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}_{MACD_SIGNAL_PERIOD}",
    f"adx_{DI_PERIOD}_{ADX_PERIOD}", f"di_plus_{DI_PERIOD}", f"di_minus_{DI_PERIOD}",
)


@dataclass
class AnchorOverlapReport:
    """Global 5m concurrence metrics, without assigning anchor groups."""

    summary: pd.DataFrame
    anchor_summary: pd.DataFrame
    counts: pd.DataFrame
    jaccard: pd.DataFrame
    correlation: pd.DataFrame
    combinations: pd.DataFrame
    trend: pd.DataFrame


@dataclass
class IndicatorDatasets:
    """Independent indicator and anchor DataFrames from one candle snapshot.

    All indexes are named row_id and identify the same chronological rows.
    raw contains timestamps, OHLCV and unnormalized indicators; normalized
    contains only numeric model features. anchors copies raw and adds
    anchor_numbers (0, one ID or '8;13;22'), anchor_ids (integer tuples), and
    anchor_01..anchor_35 (numeric 0/1 flags). Only anchor_numbers is added to
    the anchor_data worksheet; the tuples and flags are for Python use.
    These flags mark an entry opening, while that row's M1 OHLC/indicators
    become available one minute later. Do not use them at the entry opening.
    overlap contains counts, Jaccard and binary correlation matrices,
    per-anchor concurrence rates, exact combinations, and hourly trends.
    """

    raw: pd.DataFrame
    normalized: pd.DataFrame
    cutoff: pd.Timestamp
    counts: dict
    anchors: pd.DataFrame
    overlap: AnchorOverlapReport


def warmup_candles():
    indicator_warmup = WARMUP_MULTIPLIER * max(
        RSI_PERIOD, ATR_PERIOD, MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD,
        DI_PERIOD, ADX_PERIOD,
    )
    # Rule 26 needs 80 prior valid ATR observations plus a prior crossing
    # reference; retain that history even if the multiplier is reduced.
    anchor_warmup = max(ATR_PERIOD, RSI_PERIOD, DI_PERIOD + ADX_PERIOD,
                        MACD_SLOW_PERIOD + MACD_SIGNAL_PERIOD) + 82
    return max(indicator_warmup, anchor_warmup)


def validate_settings():
    periods = [LOOKBACK_CANDLES, RSI_PERIOD, ATR_PERIOD, MACD_FAST_PERIOD,
               MACD_SLOW_PERIOD, MACD_SIGNAL_PERIOD, DI_PERIOD, ADX_PERIOD,
               WARMUP_MULTIPLIER]
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in periods):
        raise ValueError("Lookback, periods and warm-up multiplier must be positive integers")
    if MACD_FAST_PERIOD >= MACD_SLOW_PERIOD:
        raise ValueError("MACD fast period must be smaller than its slow period")
    if BASE_TIMEFRAME != "1m" or not TIMEFRAMES or TIMEFRAMES[0] != "1m":
        raise ValueError("The base timeframe and first feature timeframe must be 1m")
    if len(set(TIMEFRAMES)) != len(TIMEFRAMES):
        raise ValueError("TIMEFRAMES must not contain duplicates")
    if any(tf not in TIMEFRAME_MINUTES for tf in TIMEFRAMES):
        raise ValueError("Supported feature timeframes are 1m, 5m, 15m and 30m")
    if ANCHOR_TIMEFRAME != "5m" or ANCHOR_TIMEFRAME not in TIMEFRAMES:
        raise ValueError("Anchor detection requires native 5m features")


def request_candles(session, symbol, timeframe, limit, end_ms):
    global _last_api_request_at
    symbol = symbol.strip()
    api_symbol = "t" + (symbol[1:] if symbol.startswith("t") else symbol).upper()
    url = f"{BASE_URL}/candles/trade:{timeframe}:{api_symbol}/hist"
    for attempt in range(API_MAX_ATTEMPTS):
        if _last_api_request_at is not None:
            wait = API_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_api_request_at)
            if wait > 0:
                time.sleep(wait)
        _last_api_request_at = time.monotonic()
        try:
            response = session.get(
                url, params={"limit": limit, "sort": -1, "end": int(end_ms)},
                timeout=API_REQUEST_TIMEOUT_SECONDS,
            )
        except (requests.Timeout, requests.ConnectionError):
            if attempt + 1 == API_MAX_ATTEMPTS:
                raise
            time.sleep(API_MIN_INTERVAL_SECONDS * 2 ** attempt)
            continue
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt + 1 == API_MAX_ATTEMPTS:
                response.raise_for_status()
            try:
                retry_after = float(response.headers.get("Retry-After", "0"))
            except ValueError:
                retry_after = 0
            time.sleep(max(retry_after, API_MIN_INTERVAL_SECONDS * 2 ** attempt))
            continue
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list) or any(not isinstance(row, list) or len(row) != 6 for row in data):
            raise RuntimeError(f"Unexpected candle response for {api_symbol} / {timeframe}")
        return data
    raise RuntimeError("Candle request exhausted its retry attempts")


def fetch_candles(session, symbol, timeframe, count, cutoff):
    """Fetch prior native candles; cutoff is a fixed UTC availability time."""
    duration = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    last_open = cutoff.floor(f"{TIMEFRAME_MINUTES[timeframe]}min") - duration
    end_ms = last_open.value // 1_000_000
    records = {}
    while len(records) < count:
        page = request_candles(
            session, symbol, timeframe,
            min(count - len(records), API_MAX_PAGE_SIZE), end_ms,
        )
        if not page:
            break
        oldest = min(int(row[0]) for row in page)
        if oldest > end_ms:
            raise RuntimeError("The API returned candles outside the requested time range")
        for row in page:
            if int(row[0]) <= end_ms:
                records[int(row[0])] = row
        end_ms = oldest - 1
    if not records:
        raise RuntimeError(f"No closed {timeframe} candles returned for {symbol}")
    df = pd.DataFrame(records.values(), columns=["timestamp_ms", "open", "close", "high", "low", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df["available_at"] = df["timestamp"] + duration
    df = df.loc[df["available_at"] <= cutoff].sort_values("timestamp").tail(count).reset_index(drop=True)
    if len(df) < count:
        raise RuntimeError(f"Need {count} closed {timeframe} candles including warm-up; received {len(df)}")
    prices = df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=float)
    if not np.isfinite(prices).all() or (df["close"] <= 0).any():
        raise RuntimeError(f"Invalid candle prices returned for {timeframe}")
    return df[CANDLE_COLUMNS + ["available_at"]]


def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder_rma(series, period):
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) < period:
        return pd.Series(result, index=series.index)
    seed_end = int(valid[period - 1])
    result[seed_end] = values[valid[:period]].mean()
    for i in range(seed_end + 1, len(values)):
        result[i] = result[i - 1] if np.isnan(values[i]) else result[i - 1] + (values[i] - result[i - 1]) / period
    return pd.Series(result, index=series.index)


def true_range(df):
    previous_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"], (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)


def rsi(close, period):
    delta = close.diff()
    gain = wilder_rma(delta.clip(lower=0), period)
    loss = wilder_rma(-delta.clip(upper=0), period)
    result = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    result = result.mask((loss == 0) & (gain > 0), 100.0)
    result = result.mask((gain == 0) & (loss > 0), 0.0)
    return result.mask((gain == 0) & (loss == 0), 50.0)


def directional_movement(df):
    # Decimal comparisons preserve true ties in prices such as 0.6 versus 0.6.
    highs = [Decimal(str(value)) for value in df["high"]]
    lows = [Decimal(str(value)) for value in df["low"]]
    plus = np.zeros(len(df))
    minus = np.zeros(len(df))
    for i in range(1, len(df)):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        if up > down and up > 0:
            plus[i] = float(up)
        elif down > up and down > 0:
            minus[i] = float(down)
    return pd.Series(plus, index=df.index), pd.Series(minus, index=df.index)


def fill_candle_gaps(candles):
    """Include empty time slots in indicator history, as the chart does.

    The public API omits some candles with no trades. Between returned
    candles, use the previous close for OHLC and zero volume. This matters
    even when no price changes: EMA/RMA must still advance one period.
    Only internal gaps are filled; no prices before or after the fetched
    history are inferred. Exported base rows remain actual API candles.
    """
    if candles.empty:
        return candles.copy()
    ordered = candles.sort_values("timestamp").reset_index(drop=True)
    durations = ordered["available_at"] - ordered["timestamp"]
    duration = durations.iloc[0]
    if duration <= pd.Timedelta(0) or not (durations == duration).all():
        raise ValueError("Candle durations must be positive and consistent")
    index = pd.date_range(ordered["timestamp"].iloc[0],
                          ordered["timestamp"].iloc[-1], freq=duration)
    if not ordered["timestamp"].isin(index).all():
        raise ValueError("Candle timestamps must align to their timeframe")
    result = ordered.set_index("timestamp").reindex(index)
    missing = result["close"].isna()
    previous_close = result["close"].ffill()
    for column in ("open", "high", "low", "close"):
        result.loc[missing, column] = previous_close.loc[missing]
    result.loc[missing, "volume"] = 0.0
    result.index.name = "timestamp"
    result = result.reset_index()
    result["timestamp_ms"] = result["timestamp"].astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
    result["available_at"] = result["timestamp"] + duration
    return result


def calculate_native(candles):
    """Calculate the same indicator settings on one timeframe's OHLC data."""
    result = fill_candle_gaps(candles)
    result[f"rsi_{RSI_PERIOD}"] = rsi(result["close"], RSI_PERIOD)
    result[f"atr_{ATR_PERIOD}"] = wilder_rma(true_range(result), ATR_PERIOD)
    line = ema(result["close"], MACD_FAST_PERIOD) - ema(result["close"], MACD_SLOW_PERIOD)
    result[f"macd_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}"] = line
    result[f"macd_signal_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}_{MACD_SIGNAL_PERIOD}"] = ema(
        line.dropna(), MACD_SIGNAL_PERIOD,
    ).reindex(result.index)
    plus_dm, minus_dm = directional_movement(result)
    denominator = wilder_rma(true_range(result), DI_PERIOD)
    plus_di = 100 * wilder_rma(plus_dm, DI_PERIOD) / denominator.replace(0, np.nan)
    minus_di = 100 * wilder_rma(minus_dm, DI_PERIOD) / denominator.replace(0, np.nan)
    plus_di = plus_di.mask(denominator == 0, 0.0)
    minus_di = minus_di.mask(denominator == 0, 0.0)
    total = plus_di + minus_di
    dx = (100 * (plus_di - minus_di).abs() / total.replace(0, np.nan)).mask(total == 0, 0.0)
    result[f"adx_{DI_PERIOD}_{ADX_PERIOD}"] = wilder_rma(dx, ADX_PERIOD)
    result[f"di_plus_{DI_PERIOD}"] = plus_di
    result[f"di_minus_{DI_PERIOD}"] = minus_di
    return result


def align_timeframe(base_rows, native_features, timeframe):
    """Match by closing time; display the opening time of the completed candle."""
    # Full OHLCV is already present in the native candle response. Carry the
    # higher-timeframe candle's prices and volume with its indicator values.
    candle_columns = ["close"] if timeframe == BASE_TIMEFRAME else ["open", "high", "low", "close", "volume"]
    feature_columns = ["timestamp", "available_at", *candle_columns, *INDICATOR_COLUMNS]
    renamed = {column: f"{column}_{timeframe}" for column in feature_columns}
    renamed["timestamp"] = f"source_candle_{timeframe}"
    close_key = f"_source_available_at_{timeframe}"
    renamed["available_at"] = close_key
    right = native_features[feature_columns].rename(columns=renamed)
    aligned = pd.merge_asof(
        base_rows.sort_values("available_at"), right.sort_values(close_key),
        left_on="available_at", right_on=close_key,
        direction="backward", allow_exact_matches=True,
    )
    source = aligned[close_key]
    if source.isna().any() or (source > aligned["available_at"]).any():
        raise RuntimeError(f"Could not align {timeframe} features using closed candles")
    return aligned.drop(columns=close_key)


def normalize_log(values, label):
    """Natural-log normalization with an explicit zero-input exception.

    Positive input: ln(value). Zero input: 0, without evaluating ln(0).
    Negative, missing or infinite inputs are rejected rather than encoded
    as valid features. This same rule applies to every log-normalized field.
    """
    values = pd.to_numeric(values, errors="raise").astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Cannot log-normalize missing or infinite {label}")
    if (values < 0).any():
        raise ValueError(f"Cannot log-normalize negative {label}")
    result = pd.Series(0.0, index=values.index)
    positive = values > 0
    # Zero values retain the explicit 0 encoding and never reach np.log.
    result.loc[positive] = np.log(values.loc[positive])
    return result


def normalize(raw):
    """Normalize native-timeframe features using their own ATR and close.

    Price and per-candle volume use normalize_log: ln(value) for positive
    inputs and an explicit 0 encoding for zero inputs. RSI is centered on
    50 and divided by 50. MACD and its signal use unnormalized ATR as their
    denominator. ATR is divided by the source candle's close. ADX and both
    DI values are divided by 100. Only numeric features are returned;
    source timestamps remain in raw_data. row_id is the Excel row header.
    """
    normalized = pd.DataFrame(index=raw.index)
    normalized.index = pd.RangeIndex(1, len(raw) + 1, name="row_id")
    # Use positional input indexing so every feature shares the row headers.
    raw = raw.reset_index(drop=True).copy()
    raw.index = normalized.index
    normalized["price"] = normalize_log(raw["close"], "1m close")
    for tf in TIMEFRAMES:
        if tf != BASE_TIMEFRAME:
            normalized[f"price_{tf}"] = normalize_log(raw[f"close_{tf}"], f"{tf} close")
        volume = raw["volume"] if tf == BASE_TIMEFRAME else raw[f"volume_{tf}"]
        normalized[f"volume_{tf}"] = normalize_log(volume, f"{tf} candle volume")
        normalized[f"rsi_{RSI_PERIOD}_{tf}"] = (raw[f"rsi_{RSI_PERIOD}_{tf}"] - 50) / 50
        atr_value = raw[f"atr_{ATR_PERIOD}_{tf}"]
        normalized[f"atr_{ATR_PERIOD}_{tf}"] = atr_value / raw[f"close_{tf}"]
        for key in [f"macd_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}",
                    f"macd_signal_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}_{MACD_SIGNAL_PERIOD}"]:
            value = raw[f"{key}_{tf}"]
            normalized[f"{key}_{tf}"] = (value / atr_value.replace(0, np.nan)).mask(
                (atr_value == 0) & (value == 0), 0.0,
            )
        for key in [f"adx_{DI_PERIOD}_{ADX_PERIOD}", f"di_plus_{DI_PERIOD}", f"di_minus_{DI_PERIOD}"]:
            normalized[f"{key}_{tf}"] = raw[f"{key}_{tf}"] / 100
    finite = np.isfinite(normalized.to_numpy(dtype=float))
    if not finite.all():
        invalid = normalized.columns[~finite.all(axis=0)].tolist()
        raise ValueError(f"Feature table contains undefined values in: {', '.join(invalid)}")
    return normalized.astype(float)


def detect_anchor_events(native_5m):
    """Return 0/1 events for all 35 rules on unnormalized native 5m data.

    Cross down: previous >= boundary and current < boundary; cross up:
    previous <= boundary and current > boundary. Moving boundaries use
    their value at each observation. A reversal requires a strictly
    positive prior change and negative current change (or vice versa).
    Rolling reference windows exclude the current candle. 'First' means
    false -> true, so a persistent condition does not repeat every candle.
    Hist is MACD minus its signal, including rule 32's 'H' in the document.
    """
    native = native_5m.sort_values("timestamp").reset_index(drop=True)
    duration = pd.Timedelta(minutes=TIMEFRAME_MINUTES[ANCHOR_TIMEFRAME])
    timestamps = pd.to_datetime(native["timestamp"], utc=True)
    closes = pd.to_datetime(native["available_at"], utc=True)
    if timestamps.duplicated().any() or not (closes - timestamps == duration).all():
        raise ValueError("Anchor history must contain unique native 5m candles")
    if not (timestamps == timestamps.dt.floor("5min")).all():
        raise ValueError("Anchor candles must align to 5m openings")
    if len(native) > 1 and not (timestamps.diff().iloc[1:] == duration).all():
        raise ValueError("Fill missing 5m slots before detecting anchors")

    r = native[f"rsi_{RSI_PERIOD}"]
    atr = native[f"atr_{ATR_PERIOD}"]
    macd = native[f"macd_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}"]
    signal = native[f"macd_signal_{MACD_FAST_PERIOD}_{MACD_SLOW_PERIOD}_{MACD_SIGNAL_PERIOD}"]
    hist = macd - signal
    adx = native[f"adx_{DI_PERIOD}_{ADX_PERIOD}"]
    plus = native[f"di_plus_{DI_PERIOD}"]
    minus = native[f"di_minus_{DI_PERIOD}"]
    volume = native["volume"]
    bearish = minus > plus
    bullish = plus > minus

    def cross_down(value, boundary):
        previous_boundary = boundary.shift(1) if isinstance(boundary, pd.Series) else boundary
        return (value.shift(1) >= previous_boundary) & (value < boundary)

    def cross_up(value, boundary):
        previous_boundary = boundary.shift(1) if isinstance(boundary, pd.Series) else boundary
        return (value.shift(1) <= previous_boundary) & (value > boundary)

    def turn_down(value):
        change = value.diff()
        return (change.shift(1) > 0) & (change < 0)

    def turn_up(value):
        change = value.diff()
        return (change.shift(1) < 0) & (change > 0)

    def first(condition):
        condition = condition.fillna(False).astype(bool)
        return condition & ~condition.shift(1, fill_value=False)

    def three(condition):
        return condition.astype(int).rolling(3, min_periods=3).sum() == 3

    prior_rsi_mean = r.shift(1).rolling(5, min_periods=5).mean()
    prior_hist_min = hist.shift(1).rolling(10, min_periods=10).min()
    prior_rsi_min = r.shift(1).rolling(10, min_periods=10).min()
    prior_atr_quantile = atr.shift(1).rolling(80, min_periods=80).quantile(0.75)
    prior_atr_mean = atr.shift(1).rolling(20, min_periods=20).mean()
    prior_volume_mean = volume.shift(1).rolling(20, min_periods=20).mean()
    hist_declining_positive = three(hist.diff() < 0) & three(hist > 0)
    atr_change = atr.diff()
    macd_signal_cross = cross_down(macd, signal)
    rsi_70_down = cross_down(r, 70)
    events = {
        1: cross_down(r, 50),
        2: cross_down(r, 30),
        3: rsi_70_down,
        4: cross_down(r, 60),
        5: turn_down(r) & (r.shift(1) > 70),
        6: cross_down(r, prior_rsi_mean),
        7: cross_down(macd, 0),
        8: macd_signal_cross,
        9: turn_down(hist) & (hist > 0),
        10: turn_down(hist) & (hist < 0),
        11: turn_down(macd) & (macd > 0),
        12: cross_down(signal, 0),
        13: cross_up(minus, plus),
        14: cross_up(adx, 25) & bearish,
        15: turn_up(adx) & (adx.shift(1) <= 20) & bearish,
        16: cross_up(minus - plus, 10),
        17: cross_up(minus, 25) & bearish,
        18: cross_down(plus, 20) & bearish,
        19: turn_down(adx) & (adx.shift(1) > 35) & bullish,
        20: first(three((minus.diff() > 0) & (plus.diff() < 0))),
        21: cross_down(adx, 25) & bullish,
        22: macd_signal_cross & (adx > 25),
        23: first(three(bearish)),
        24: cross_down(hist, prior_hist_min) & (macd < 0),
        25: cross_up(atr, prior_atr_quantile) & bearish,
        26: first((volume > 2 * prior_volume_mean) & bearish),
        27: cross_up(atr, prior_atr_mean) & bearish,
        28: cross_down(r, prior_rsi_min) & (r < 50),
        29: cross_up(r, 70),
        30: turn_down(signal) & (signal > 0),
        31: first(hist_declining_positive),
        32: turn_down(plus) & bullish,
        33: first((minus.diff() > 5) & bearish),
        34: first((volume > 3 * prior_volume_mean) & (r > 70)),
        35: (atr_change < 0) & three(atr_change > 0).shift(1, fill_value=False) & bullish,
    }
    flags = pd.DataFrame({
        f"anchor_{number:02d}": events[number].fillna(False).to_numpy(dtype=np.uint8)
        for number in ANCHOR_RULES
    }, index=pd.DatetimeIndex(closes, name="entry_at"))
    return flags


def filter_anchors(raw, native_5m):
    """Annotate the exact M1 opening following each closed 5m anchor candle.

    Example: the anchor on 12:55's 5m candle is known at 13:00 and maps to
    the M1 timestamp 13:00, never to 12:59 or to a later available API row.
    Detection uses the full warm-up history before trimming output rows.
    Every output row remains present. No anchor -> 0, one -> integer ID,
    overlaps -> sorted IDs separated by ';'. anchor_ids stores tuples and
    anchor_01..anchor_35 store numeric flags for exact programmatic filtering.
    A row can be selected independently by any of its anchor flags.

    raw's features still describe that M1 candle at its close. This table
    marks entry candidates; it is not a feature snapshot at the opening.
    Internal missing API rows are reported in attrs['unmapped_entries'];
    they are never silently shifted onto a different minute. Entries after
    the last output opening are recorded separately in 'pending_entries'.
    """
    result = raw.copy(deep=True)
    timestamps = pd.to_datetime(result["timestamp"], utc=True)
    if timestamps.duplicated().any() or not timestamps.is_monotonic_increasing:
        raise ValueError("Base rows must have unique chronological timestamps")
    flags = detect_anchor_events(native_5m)
    aligned = flags.reindex(pd.DatetimeIndex(timestamps)).fillna(0).astype(np.uint8)
    numbers = tuple(ANCHOR_RULES)
    ids = [tuple(number for number, active in zip(numbers, row) if active)
           for row in aligned.to_numpy()]
    result.insert(min(3, len(result.columns)), "anchor_numbers", [
        0 if not row else row[0] if len(row) == 1 else ";".join(map(str, row))
        for row in ids
    ])
    result["anchor_ids"] = pd.Series(ids, index=result.index, dtype=object)
    for column in aligned.columns:
        result[column] = aligned[column].to_numpy()
    events = flags.loc[flags.any(axis=1)]
    missing = events.loc[~events.index.isin(timestamps)]
    if len(result):
        in_range = (missing.index >= timestamps.iloc[0]) & (missing.index <= timestamps.iloc[-1])
        pending = missing.index > timestamps.iloc[-1]
    else:
        in_range = np.zeros(len(missing), dtype=bool)
        pending = np.zeros(len(missing), dtype=bool)

    def entry_ids(rows):
        return {str(timestamp): tuple(number for number, active in zip(numbers, row) if active)
                for timestamp, row in zip(rows.index, rows.to_numpy())}

    result.attrs["unmapped_entries"] = entry_ids(missing.loc[in_range])
    result.attrs["pending_entries"] = entry_ids(missing.loc[pending])
    result.attrs["anchor_timeframe"] = ANCHOR_TIMEFRAME
    result.attrs["anchor_schema_version"] = ANCHOR_SCHEMA_VERSION
    result.attrs["anchor_rule_source"] = ANCHOR_DOCUMENT_URL
    return result


def calculate_anchor_overlap(anchors, trend_interval=ANCHOR_OVERLAP_TREND_INTERVAL):
    """Build global concurrence metrics on eligible 5m signal closes.

    Include every 5m closing time within the exported M1 opening range,
    including zero-signal observations and entry minutes without API M1
    candles. Recover their signals from filter_anchors' unmapped_entries.
    Exclude warm-up and pending entries outside this range. This measures
    simultaneous signals, not trade outcomes or a position execution rule.

    Count matrix: number of observations with both anchors; diagonal is
    each anchor's total. Jaccard: intersection / union. Correlation: Pearson
    correlation of binary event flags (phi), using zero-signal observations
    too. Undefined ratios/correlations remain NaN, not a misleading zero.
    Exact combinations include overlaps of any size. Hourly trends count
    each anchor, active observations, and observations with >=2 anchors.
    """
    numbers = tuple(ANCHOR_RULES)
    columns = [f"anchor_{number:02d}" for number in numbers]
    timestamps = pd.DatetimeIndex(pd.to_datetime(anchors["timestamp"], utc=True))
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Overlap input must have unique chronological timestamps")
    if not anchors[columns].isin([0, 1]).all().all():
        raise ValueError("Overlap metrics require numeric 0/1 anchor flags")
    if len(timestamps):
        grid = pd.date_range(timestamps[0].ceil("5min"), timestamps[-1].floor("5min"),
                             freq="5min", name="entry_at")
    else:
        grid = pd.DatetimeIndex([], tz="UTC", name="entry_at")
    observed = anchors[columns].copy()
    observed.index = timestamps
    flags = observed.reindex(grid).fillna(0).astype(np.int64)
    recovered = 0
    for entry, ids in anchors.attrs.get("unmapped_entries", {}).items():
        timestamp = pd.Timestamp(entry).tz_convert("UTC")
        if timestamp in flags.index:
            for number in ids:
                if number not in ANCHOR_RULES:
                    raise ValueError(f"Unknown anchor in unmapped entry: {number}")
                flags.loc[timestamp, f"anchor_{number:02d}"] = 1
            recovered += 1

    values = flags.to_numpy(dtype=np.int64)
    totals = values.sum(axis=0)
    intersection = values.T @ values
    union = totals[:, None] + totals[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.full(union.shape, np.nan), where=union > 0)
    n = len(flags)
    variance = totals.astype(float) * (n - totals)
    denominator = np.sqrt(variance[:, None] * variance[None, :])
    numerator = n * intersection.astype(float) - totals[:, None].astype(float) * totals[None, :]
    correlation = np.divide(numerator, denominator, out=np.full(union.shape, np.nan), where=denominator > 0)
    simultaneous = values.sum(axis=1)
    active = int((simultaneous > 0).sum())
    overlapping = int((simultaneous >= 2).sum())

    def matrix(data):
        frame = pd.DataFrame(data, index=pd.Index(numbers, name="anchor_id"), columns=numbers)
        frame.columns.name = "anchor_id"
        return frame

    overlap_counts = values[simultaneous >= 2].sum(axis=0)
    solo_counts = totals - overlap_counts
    rates = np.divide(overlap_counts, totals, out=np.full(len(numbers), np.nan), where=totals > 0)
    anchor_summary = pd.DataFrame({
        "anchor_id": numbers, "signal_candles": totals, "solo_candles": solo_counts,
        "overlap_candles": overlap_counts, "overlap_rate": rates,
        "rule": [ANCHOR_RULES[number] for number in numbers],
    }).sort_values(["overlap_candles", "overlap_rate", "anchor_id"],
                  ascending=[False, False, True], na_position="last").reset_index(drop=True)
    patterns = Counter(tuple(number for number, signal in zip(numbers, row) if signal)
                       for row in values[simultaneous >= 2])
    combinations = pd.DataFrame([
        {"anchor_numbers": ";".join(map(str, ids)), "anchor_count": len(ids),
         "candles": count, "share_of_overlap_candles": count / overlapping}
        for ids, count in sorted(patterns.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    ], columns=["anchor_numbers", "anchor_count", "candles", "share_of_overlap_candles"])

    time_data = flags.copy()
    time_data["eligible_5m_candles"] = 1
    time_data["active_candles"] = (simultaneous > 0).astype(int)
    time_data["overlap_candles"] = (simultaneous >= 2).astype(int)
    time_data["anchor_signals"] = simultaneous
    trend = time_data.resample(trend_interval).sum()
    trend["max_simultaneous"] = time_data["anchor_signals"].resample(trend_interval).max()
    trend["overlap_rate"] = trend["overlap_candles"] / trend["active_candles"].replace(0, np.nan)
    trend = trend[["eligible_5m_candles", "active_candles", "overlap_candles", "anchor_signals",
                   "max_simultaneous", "overlap_rate", *columns]]
    trend.index.name = "interval_start"
    trend = trend.reset_index()
    window = (timestamps[0].tz_convert(DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M") + " to " +
              timestamps[-1].tz_convert(DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M")) if len(timestamps) else "Empty"
    summary = pd.DataFrame([
        ["Window (M1 openings, UTC+3)", window],
        ["Eligible closed 5m observations", n],
        ["Observations with any anchor", active],
        ["Observations with 2+ anchors", overlapping],
        ["Total individual anchor signals", int(totals.sum())],
        ["Maximum simultaneous anchors", int(simultaneous.max()) if n else 0],
        ["Exact overlap combinations", len(combinations)],
        ["Empty M1 entry minutes recovered", recovered],
        ["Trend interval", trend_interval],
    ], columns=["Metric", "Value"])
    return AnchorOverlapReport(summary=summary, anchor_summary=anchor_summary,
                               counts=matrix(intersection), jaccard=matrix(similarity),
                               correlation=matrix(correlation), combinations=combinations, trend=trend)


def _collect_snapshot(cutoff=None):
    """Collect using UTC internally; an optional naive cutoff is in UTC+3."""
    validate_settings()
    cutoff = pd.Timestamp.now(tz="UTC") if cutoff is None else pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize(DISPLAY_TIMEZONE) if cutoff.tzinfo is None else cutoff
    cutoff = cutoff.tz_convert("UTC")
    cutoff = cutoff.floor("min")
    warmup = warmup_candles()
    counts = {}
    anchor_history = None
    with requests.Session() as session:
        count = LOOKBACK_CANDLES + warmup
        print(f"Fetching {count} closed 1m candles for {TICKER}...")
        base = fetch_candles(session, TICKER, "1m", count, cutoff)
        raw = base.tail(LOOKBACK_CANDLES).reset_index(drop=True)
        raw = raw[["timestamp", "timestamp_ms", "available_at", "open", "high", "low", "close", "volume"]]
        first_decision = raw["available_at"].iloc[0]
        for tf in TIMEFRAMES:
            if tf == "1m":
                native = base
            else:
                minutes = TIMEFRAME_MINUTES[tf]
                duration = pd.Timedelta(minutes=minutes)
                first_open = first_decision.floor(f"{minutes}min") - duration
                last_open = cutoff.floor(f"{minutes}min") - duration
                count = int((last_open - first_open) // duration) + 1 + warmup
                print(f"Fetching {count} closed {tf} candles for {TICKER}...")
                native = fetch_candles(session, TICKER, tf, count, cutoff)
            counts[tf] = len(native)
            native_features = calculate_native(native)
            if tf == ANCHOR_TIMEFRAME:
                anchor_history = native_features
            raw = align_timeframe(raw, native_features, tf)
    expected = [f"{column}_{tf}" for tf in TIMEFRAMES for column in INDICATOR_COLUMNS]
    if raw[expected].isna().any().any():
        raise RuntimeError("Some indicators are unavailable after warm-up; increase the warm-up multiplier")
    normalized = normalize(raw)
    raw.index = normalized.index.copy()
    anchors = filter_anchors(raw, anchor_history)
    return raw, normalized, cutoff, counts, anchors


def collect_dataset(cutoff=None):
    """Compatibility API: return raw, normalized, cutoff and candle counts.

    Use collect_datasets() to access the separate anchor dataset as well.
    """
    raw, normalized, snapshot_cutoff, counts, _anchors = _collect_snapshot(cutoff)
    return raw.reset_index(drop=True), normalized, snapshot_cutoff, counts


def collect_datasets(cutoff=None):
    """Return datasets.raw, datasets.normalized and datasets.anchors.

    Fetch and calculate once. Each DataFrame has its own data, so editing
    one does not edit the other. The normalized values describe this
    snapshot; call normalize again if raw values are subsequently changed.
    """
    raw, normalized, snapshot_cutoff, counts, anchors = _collect_snapshot(cutoff)
    raw_data = raw.reset_index(drop=True).copy()
    raw_data.index = normalized.index.copy()
    return IndicatorDatasets(
        raw=raw_data,
        normalized=normalized.copy(),
        cutoff=snapshot_cutoff,
        counts=counts.copy(),
        anchors=anchors.copy(deep=True),
        overlap=calculate_anchor_overlap(anchors),
    )


def parameters(cutoff, counts):
    return pd.DataFrame([
        ["Version", VERSION], ["Exchange", "Bitfinex"], ["Ticker", TICKER],
        ["Base timeframe", BASE_TIMEFRAME], ["Feature timeframes", ", ".join(TIMEFRAMES)],
        ["Output candles", LOOKBACK_CANDLES],
        ["Display timezone", DISPLAY_TIMEZONE_LABEL + " (fixed offset)"],
        [f"Snapshot cutoff {DISPLAY_TIMEZONE_LABEL}", cutoff.tz_convert(DISPLAY_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S")],
        ["Candle source", "Native exchange OHLCV on each timeframe"],
        ["Timestamp", f"Candle opening time in {DISPLAY_TIMEZONE_LABEL}; timestamp_ms remains the Unix epoch in milliseconds"],
        ["Available at", "1m candle closing time; the row's features may be used from this time onward"],
        ["Higher-timeframe alignment", "Last candle whose closing time is <= available_at; carry forward until the next close"],
        ["Source candle columns", f"Opening time identifying the completed candle ({DISPLAY_TIMEZONE_LABEL}). Example: row 13:01 uses the 12:55 5m candle, closed at 13:00."],
        ["Warm-up candles", warmup_candles()], ["Fetched candles", "; ".join(f"{tf}: {count}" for tf, count in counts.items())],
        ["Empty candle handling", "Fill internal API gaps for indicator calculation: OHLC = previous close, volume = 0. Advance EMA/RMA each period. Base output rows remain actual API candles."],
        ["RSI", f"Period {RSI_PERIOD}; close; Wilder RMA of gains and losses on each timeframe"],
        ["ATR", f"Period {ATR_PERIOD}; Wilder RMA of each timeframe's true range"],
        ["MACD", f"EMA({MACD_FAST_PERIOD}) - EMA({MACD_SLOW_PERIOD}); signal EMA({MACD_SIGNAL_PERIOD}) on each timeframe"],
        ["ADX/DMI", f"DI period {DI_PERIOD}; ADX smoothing {ADX_PERIOD}; Wilder RMA; decimal movements with ties set to zero"],
        ["RSI normalization", "(RSI - 50) / 50 on every timeframe"], ["MACD normalization", f"MACD and signal divided by raw ATR({ATR_PERIOD}); if both are zero, export 0; other undefined ratios stop export"],
        ["ADX/DI normalization", "Divide by 100"], ["ATR normalization", "ATR divided by the source candle's close on the same timeframe"],
        ["Price normalization", "Natural logarithm of close; price is 1m, price_5m/15m/30m use their completed source candles"],
        ["Volume normalization", "ln(volume) for positive per-candle volume on each timeframe; zero volume is explicitly encoded as 0"],
        ["Log zero-input exception", "All natural-log features use ln(x) when x > 0 and encode x = 0 as 0 without evaluating ln(0)"],
        ["Model feature table", "normalized_data contains only numeric features. row_id is the row header, excluded from model inputs."],
        ["Model import", "pandas.read_excel(path, sheet_name='normalized_data', index_col='row_id').to_numpy(dtype=float)"],
        ["Python datasets", "collect_datasets() returns independent datasets.raw, datasets.normalized and datasets.anchors, with matching row_id indexes"],
        ["Anchor rules source", ANCHOR_DOCUMENT_URL + "; original rules read on 2026-10-11. Former anchor 23 removed on 2026-10-14; former 24-36 renumbered to 23-35. User selected 5m detection instead of the document's M1; no group assignments."],
        ["Anchor schema", ANCHOR_SCHEMA_VERSION],
        ["Anchor detection", "detect_anchor_events() evaluates all 35 rules on closed native 5m candles with full warm-up, using raw indicators. Hist (H) = MACD - Signal."],
        ["Anchor reference windows", "Previous 5/10/20/80 native 5m periods, excluding the current candle; percentile = pandas quantile(0.75), linear interpolation"],
        ["Anchor event boundaries", "Cross down: previous >= threshold and current < threshold. Cross up: previous <= threshold and current > threshold. Strict reversals require opposite consecutive changes; flat changes do not count. 'First' conditions fire on false-to-true transitions."],
        ["Anchor row assignment", "filter_anchors(): a 12:55-13:00 5m anchor maps to the M1 opening 13:00, at the exact 5m close. All other rows have 0."],
        ["Overlapping anchors", "anchor_numbers contains a single integer or sorted IDs joined by ';', e.g. 8;13;22. Each ID remains independent. No groups have been assigned."],
        ["Anchor Python filtering", "datasets.anchors includes integer tuples in anchor_ids and numeric 0/1 columns anchor_01..anchor_35. Example: datasets.anchors.loc[datasets.anchors['anchor_08'].eq(1)]"],
        ["Anchor sheet", "anchor_data copies raw rows and adds anchor_numbers; tuple and 0/1 representations are available in Python only"],
        ["Anchor entry feature timing", "An anchor at 13:00 is available at 13:00. That marked M1 row's OHLC/indicators are only available at 13:01; for entry at 13:00, use features whose available_at <= 13:00."],
        ["Missing anchor entry rows", "Never shift an anchor to another minute. Missing internal API entry rows are retained in datasets.anchors.attrs['unmapped_entries'] and reported; future entry openings beyond the last closed output row are in 'pending_entries'."],
        ["Global overlap universe", "All eligible 5m signal closes within the exported M1 opening range, including zero-signal observations and signals on missing API M1 entry minutes. Excludes warm-up and pending entries outside this range."],
        ["Global overlap matrices", "anchor_overlap: joint counts (diagonal = anchor totals), Jaccard intersection/union, and binary Pearson/phi correlation. Undefined Jaccard/correlation cells are blank."],
        ["Global overlap views", "Only anchor_data and anchor_overlap are exported for anchor research. Per-anchor summaries, exact combinations and hourly trends remain available in datasets.overlap for Python use."],
        ["Overlap rate", "Observations containing this anchor and at least one other / all observations containing this anchor. Combination share = exact-set count / observations with 2+ anchors."],
        ["Row mapping", "row_id 1 corresponds to raw_data Excel row 2; subsequent rows follow the same chronological order"],
        ["Higher-timeframe OHLCV", "Full OHLCV of the last closed 5m, 15m and 30m source candles; carried forward with their indicators"],
        ["Volume", "1m per-candle volume; higher-timeframe volume belongs to its source candle and repeats between source closes"],
        ["API pacing", f"At least {API_MIN_INTERVAL_SECONDS} seconds between request starts; timeout {API_REQUEST_TIMEOUT_SECONDS} seconds"],
        ["API documentation", "https://docs.bitfinex.com/reference/rest-public-candles"],
    ] + [[f"Anchor {number:02d}", description] for number, description in ANCHOR_RULES.items()],
        columns=["Parameter", "Value"])


def write_overlap_view(writer, report):
    """Export the global matrices as one readable Excel heatmap view."""
    ws = writer.book.create_sheet("anchor_overlap")
    last_column = len(ANCHOR_RULES) + 1
    last_letter = get_column_letter(last_column)
    body_font = Font(name="Arial", size=10)
    heading_font = Font(name="Arial", size=11, bold=True)
    ws.freeze_panes = "B4"
    ws.column_dimensions["A"].width = 14
    for column in range(2, last_column + 1):
        ws.column_dimensions[get_column_letter(column)].width = 5

    def heading(row, value, height=26):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=last_column)
        cell = ws.cell(row, 1, value)
        cell.font = heading_font
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        ws.row_dimensions[row].height = height

    heading(1, "Global anchor concurrence on closed 5m candles")
    heading(2, "All 35 anchors (former 23 removed; former 24-36 renumbered). Same-candle signals, without group assignments. Missing M1 entry minutes included.", 30)
    for row, (metric, value) in enumerate(report.summary.itertuples(index=False, name=None), start=4):
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=8)
        ws.merge_cells(start_row=row, start_column=9, end_row=row, end_column=last_column)
        ws.cell(row, 1, metric).font = body_font
        ws.cell(row, 1).alignment = Alignment(vertical="center", wrap_text=True)
        ws.cell(row, 9, value).font = body_font
        ws.cell(row, 9).alignment = Alignment(vertical="center")
        ws.row_dimensions[row].height = 24

    row = 15
    sections = [
        ("Joint occurrence counts", "Rows and columns are anchor IDs. Diagonal = anchor frequency; off-diagonal = shared 5m observations.", report.counts, "0"),
        ("Jaccard similarity", "Shared observations / observations with either anchor. 100% = identical event sets. Blank = neither anchor occurred.", report.jaccard, "0%"),
        ("Binary event correlation (phi)", "Uses all eligible 5m observations, including no-signal candles. -1 to +1. Blank = an anchor never fired or fired on every observation.", report.correlation, "0.00"),
    ]
    for title, description, frame, number_format in sections:
        heading(row, title)
        heading(row + 1, description, 30)
        ws.cell(row + 2, 1, "anchor_id")
        for column, number in enumerate(frame.columns, start=2):
            ws.cell(row + 2, column, int(number))
        for offset, (number, values) in enumerate(frame.iterrows(), start=3):
            ws.cell(row + offset, 1, int(number))
            for column, value in enumerate(values, start=2):
                ws.cell(row + offset, column, None if pd.isna(value) else float(value))
                ws.cell(row + offset, column).number_format = number_format
        for cells in ws.iter_rows(min_row=row + 2, max_row=row + 2 + len(frame), max_col=last_column):
            for cell in cells:
                cell.font = heading_font if cell.row == row + 2 or cell.column == 1 else body_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
        area = f"B{row + 3}:{last_letter}{row + 2 + len(frame)}"
        if title.startswith("Binary"):
            rule = ColorScaleRule(start_type="num", start_value=-1, start_color="B4C7E7",
                                  mid_type="num", mid_value=0, mid_color="FFFFFF",
                                  end_type="num", end_value=1, end_color="F4B183")
        else:
            maximum = max(1, int(report.counts.to_numpy().max())) if number_format == "0" else 1
            rule = ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF",
                                  end_type="num", end_value=maximum, end_color="F4B183")
        ws.conditional_formatting.add(area, rule)
        row += len(frame) + 6


def export_excel(raw, normalized, cutoff, counts, output=OUTPUT_FILE, anchors=None, overlap=None):
    """Export indicator sheets and, when supplied, the separate anchor table."""
    frames = {"raw_data": raw.copy(), "normalized_data": normalized.copy()}
    if anchors is not None:
        if not anchors.index.equals(raw.index) or not anchors["timestamp"].equals(raw["timestamp"]):
            raise ValueError("Anchor rows must match the raw dataset's row IDs and timestamps")
        columns = list(raw.columns)
        columns.insert(min(3, len(columns)), "anchor_numbers")
        frames["anchor_data"] = anchors[columns].copy()
        overlap = calculate_anchor_overlap(anchors) if overlap is None else overlap
    frames["parameters"] = parameters(cutoff, counts)
    for frame in frames.values():
        for column in frame.columns:
            if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
                frame[column] = frame[column].dt.tz_convert(DISPLAY_TIMEZONE).dt.tz_localize(None)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl", datetime_format="yyyy-mm-dd hh:mm:ss") as writer:
        normal_font = Font(name="Arial", size=10)
        bold_font = Font(name="Arial", size=10, bold=True)
        body_alignment = Alignment(vertical="center")
        for name, frame in frames.items():
            is_feature_table = name == "normalized_data"
            has_row_header = is_feature_table or name == "anchor_data"
            frame.to_excel(writer, sheet_name=name, index=has_row_header)
            column_names = ([frame.index.name] if has_row_header else []) + list(frame.columns)
            ws = writer.book[name]
            ws.freeze_panes = "F2" if name == "anchor_data" else "B2" if has_row_header else "A2"
            ws.auto_filter.ref = ws.dimensions
            ws.sheet_view.showGridLines = True
            ws.row_dimensions[1].height = 30
            for cell in ws[1]:
                cell.font = Font(name="Arial", size=10, bold=True, color="000000")
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for row in ws.iter_rows(min_row=2):
                for cell in row:
                    is_row_header = has_row_header and cell.column == 1
                    cell.font = bold_font if is_row_header else normal_font
                    cell.alignment = body_alignment
                    if isinstance(cell.value, datetime):
                        cell.number_format = "yyyy-mm-dd hh:mm:ss"
                    elif isinstance(cell.value, (int, float)):
                        cell.number_format = "0" if is_row_header or column_names[cell.column - 1] == "timestamp_ms" else "General"
                        if column_names[cell.column - 1] in ("overlap_rate", "share_of_overlap_candles"):
                            cell.number_format = "0.0%"
            for i, column in enumerate(column_names, start=1):
                width = 10 if column == "row_id" else max(27, len(column) + 2)
                if column == "anchor_numbers":
                    width = max(width, max((len(str(value)) for value in frame[column]), default=1) + 2)
                if column in frame.columns and pd.api.types.is_datetime64_any_dtype(frame[column]):
                    width = max(width, 22)
                ws.column_dimensions[get_column_letter(i)].width = width if column == "anchor_numbers" else min(width, 34)
            if name == "parameters":
                ws.column_dimensions["A"].width = 30
                ws.column_dimensions["B"].width = 92
                for row in range(2, ws.max_row + 1):
                    ws.cell(row, 1).font = Font(name="Arial", size=10, bold=True)
                    ws.cell(row, 2).alignment = Alignment(vertical="top", wrap_text=True)
                    lines = len(textwrap.wrap(str(ws.cell(row, 2).value or ""), width=88)) or 1
                    ws.row_dimensions[row].height = max(20, lines * 15)
        if overlap is not None:
            write_overlap_view(writer, overlap)
            # Place the main global view beside the anchor data it describes.
            writer.book.move_sheet("anchor_overlap", offset=2 - writer.book.sheetnames.index("anchor_overlap"))
    return output


def _indicator_only_main():
    datasets = collect_datasets()
    raw = datasets.raw
    output = export_excel(raw, datasets.normalized, datasets.cutoff, datasets.counts,
                          anchors=datasets.anchors, overlap=datasets.overlap)
    print(f"Wrote {len(raw)} rows to {output}")
    print(f"First 1m candle: {raw['timestamp'].iloc[0].tz_convert(DISPLAY_TIMEZONE)}")
    print(f"Last 1m candle:  {raw['timestamp'].iloc[-1].tz_convert(DISPLAY_TIMEZONE)}")
    print(f"Workbook timezone: {DISPLAY_TIMEZONE_LABEL}; source columns identify completed candles by opening time.")
    print("All higher-timeframe features are aligned by candle closing time.")
    marked = datasets.anchors["anchor_ids"].map(bool)
    overlaps = datasets.anchors["anchor_ids"].map(len).gt(1)
    print(f"5m anchors: {int(marked.sum())} marked M1 rows, {int(overlaps.sum())} overlapping rows.")
    missing = datasets.anchors.attrs["unmapped_entries"]
    if missing:
        print(f"WARNING: {len(missing)} anchor entry openings have no API M1 output row: {missing}")
    pending = datasets.anchors.attrs["pending_entries"]
    if pending:
        print(f"Anchor entry openings beyond the last closed M1 output row: {pending}")


# Required original functions from macd_stop_trades_from_dataset_v9.py; isolated module namespace.
ORIGINAL_EXIT_SHA256 = "e9d28fd0199668b4e61bf64d8fca112f30a0b30afc65cc1f3654c47e7cbb3eb1"
EMBEDDED_EXIT_SHA256 = "7e827bca1f1f8a6424656616a5a02ef0ece3ca2f74213b3d51d482caae7f3627"
ORIGINAL_EXIT_SOURCE = r'''from __future__ import annotations


import argparse


import datetime as dt


import math


from dataclasses import dataclass, field


from pathlib import Path


from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


import numpy as np


import pandas as pd


try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None


TF_MS_1M = 60 * 1000


TF_MS_5M = 5 * 60 * 1000


PRE_ENTRY_RED_HISTORY_LIMIT = 20


def align_ms_down(ms: int, tf_ms: int) -> int:
    return int(ms) - (int(ms) % int(tf_ms))


def _tz(tz_name: Optional[str], tz_offset: int) -> dt.tzinfo:
    if tz_name:
        if ZoneInfo is None:
            raise RuntimeError("zoneinfo is not available. Use Python 3.9+ or omit --tz-name.")
        return ZoneInfo(tz_name)
    return dt.timezone(dt.timedelta(hours=int(tz_offset)))


def ms_to_local(ms: Optional[Union[int, float]], tz_name: Optional[str], tz_offset: int) -> str:
    if ms is None:
        return ""
    try:
        if pd.isna(ms):
            return ""
    except Exception:
        pass
    utc_dt = dt.datetime.fromtimestamp(int(ms) / 1000, tz=dt.timezone.utc)
    return utc_dt.astimezone(_tz(tz_name, tz_offset)).strftime("%d.%m.%Y %H:%M")


def parse_local_time_to_ms(value: Any, tz_name: Optional[str], tz_offset: int) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        naive = value.to_pydatetime().replace(tzinfo=None)
    elif isinstance(value, dt.datetime):
        naive = value.replace(tzinfo=None)
    else:
        s = str(value).strip()
        if not s or s.lower() in {"nan", "nat", "none"}:
            return None
        naive = None
        for fmt in (
            "%d.%m.%Y %H:%M:%S",
            "%d.%m.%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
        ):
            try:
                naive = dt.datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if naive is None:
            parsed = pd.to_datetime(s, errors="coerce")
            if pd.isna(parsed):
                return None
            naive = parsed.to_pydatetime().replace(tzinfo=None)

    local = naive.replace(tzinfo=_tz(tz_name, tz_offset))
    utc = local.astimezone(dt.timezone.utc)
    return int(utc.timestamp() * 1000)


def minute_end_ms(minute_start_ms: int) -> int:
    return int(minute_start_ms) + TF_MS_1M


def safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def safe_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if pd.isna(v):
            return default
        return int(v)
    except Exception:
        return default


def candle_color(row: pd.Series) -> str:
    op = safe_float(row.get("open"))
    cl = safe_float(row.get("close"))
    if pd.isna(op) or pd.isna(cl):
        return "unknown"
    if op > cl:
        return "red"
    if op < cl:
        return "green"
    return "neutral"


@dataclass
class SupportResult:
    found: bool
    support_level: Optional[float] = None
    support_type: str = ""
    reason: str = ""
    threshold: float = 0.0
    lows_count: int = 0
    lower_cluster_count: int = 0
    base_cluster_count: int = 0
    lower_cluster_lows: str = ""
    base_cluster_lows: str = ""
    lower_cluster_times: str = ""
    base_cluster_times: str = ""
    sorted_lows: str = ""
    source_min_low_time: str = ""
    source_min_low_mts_utc: Any = np.nan


def is_close(a: float, base: float, threshold: float) -> bool:
    if base == 0 or pd.isna(a) or pd.isna(base):
        return False
    return abs(float(a) - float(base)) / abs(float(base)) <= float(threshold)


def _fmt_join(values: Iterable[Any], max_items: int = 40, ndigits: int = 6) -> str:
    out: List[str] = []
    for i, v in enumerate(values):
        if i >= max_items:
            out.append("...")
            break
        try:
            if isinstance(v, (float, int, np.floating, np.integer)) and not pd.isna(v):
                out.append(str(round(float(v), ndigits)))
            else:
                out.append(str(v))
        except Exception:
            out.append(str(v))
    return ";".join(out)


def find_support_level_from_candles(
    candles: pd.DataFrame,
    threshold: float,
    tz_name: Optional[str],
    tz_offset: int,
) -> SupportResult:
    """
    Rich version of find_support_level(low_values, threshold).

    Uses real candle low values only. Candles are sorted by low ascending.
    The returned support_level, when found, is always the global minimum low.
    """
    if candles is None or candles.empty:
        return SupportResult(found=False, reason="no_candles", threshold=float(threshold), lows_count=0)

    need_cols = {"low", "minute_mts_utc"}
    missing = need_cols - set(candles.columns)
    if missing:
        return SupportResult(found=False, reason=f"missing_columns:{','.join(sorted(missing))}", threshold=float(threshold))

    work = candles.copy()
    work["low"] = pd.to_numeric(work["low"], errors="coerce")
    work["minute_mts_utc"] = pd.to_numeric(work["minute_mts_utc"], errors="coerce")
    work = work.dropna(subset=["low", "minute_mts_utc"]).sort_values(
        ["low", "minute_mts_utc"], ascending=[True, True]
    ).reset_index(drop=True)

    n = len(work)
    if n < 3:
        return SupportResult(
            found=False,
            reason="less_than_3_lows",
            threshold=float(threshold),
            lows_count=n,
            sorted_lows=_fmt_join(work["low"].tolist()),
        )

    lows = work["low"].astype(float).tolist()
    times = [ms_to_local(int(x), tz_name, tz_offset) for x in work["minute_mts_utc"].tolist()]

    def collect_cluster(start_idx: int) -> Tuple[List[int], int]:
        base = lows[start_idx]
        idxs: List[int] = []
        i = start_idx
        while i < n and is_close(lows[i], base, float(threshold)):
            idxs.append(i)
            i += 1
        return idxs, i

    def search_base_cluster(start_idx: int) -> Optional[Tuple[int, List[int]]]:
        i = start_idx
        while i < n:
            cluster, next_i = collect_cluster(i)
            if len(cluster) >= 3:
                return i, cluster
            i = max(next_i, i + 1)
        return None

    def result(
        support_type: str,
        lower_cluster: List[int],
        base_cluster: List[int],
        reason: str = "ok",
    ) -> SupportResult:
        min_row = work.iloc[0]
        return SupportResult(
            found=True,
            support_level=float(lows[0]),
            support_type=support_type,
            reason=reason,
            threshold=float(threshold),
            lows_count=n,
            lower_cluster_count=len(lower_cluster),
            base_cluster_count=len(base_cluster),
            lower_cluster_lows=_fmt_join([lows[i] for i in lower_cluster]),
            base_cluster_lows=_fmt_join([lows[i] for i in base_cluster]),
            lower_cluster_times=_fmt_join([times[i] for i in lower_cluster], ndigits=0),
            base_cluster_times=_fmt_join([times[i] for i in base_cluster], ndigits=0),
            sorted_lows=_fmt_join(lows),
            source_min_low_time=ms_to_local(int(min_row["minute_mts_utc"]), tz_name, tz_offset),
            source_min_low_mts_utc=int(min_row["minute_mts_utc"]),
        )

    # Branch A: arr[1] close to arr[0]
    if is_close(lows[1], lows[0], float(threshold)):
        if is_close(lows[2], lows[0], float(threshold)):
            lower_cluster, _ = collect_cluster(0)
            return result("type1", lower_cluster, [], "lower_cluster_3_plus")

        lower_cluster, _ = collect_cluster(0)  # normally exactly 2 because arr[2] is not close
        base = search_base_cluster(2)
        if base is not None:
            _, base_cluster = base
            return result("type3", lower_cluster, base_cluster, "lower_pair_plus_upper_base_cluster")
        return SupportResult(
            found=False,
            reason="type3_no_upper_base_cluster",
            threshold=float(threshold),
            lows_count=n,
            lower_cluster_count=len(lower_cluster),
            lower_cluster_lows=_fmt_join([lows[i] for i in lower_cluster]),
            lower_cluster_times=_fmt_join([times[i] for i in lower_cluster], ndigits=0),
            sorted_lows=_fmt_join(lows),
            source_min_low_time=times[0],
            source_min_low_mts_utc=int(work.iloc[0]["minute_mts_utc"]),
        )

    # Branch B: arr[1] is not close to arr[0]
    base = search_base_cluster(1)
    if base is not None:
        _, base_cluster = base
        return result("type2", [0], base_cluster, "single_low_below_upper_base_cluster")

    return SupportResult(
        found=False,
        reason="type2_no_upper_base_cluster",
        threshold=float(threshold),
        lows_count=n,
        lower_cluster_count=1,
        lower_cluster_lows=_fmt_join([lows[0]]),
        lower_cluster_times=_fmt_join([times[0]], ndigits=0),
        sorted_lows=_fmt_join(lows),
        source_min_low_time=times[0],
        source_min_low_mts_utc=int(work.iloc[0]["minute_mts_utc"]),
    )


@dataclass
class ActiveTrade:
    trade_no: int
    trade_source: str
    entry_time_ms: int
    entry_price: float
    entry_order_no: Optional[int] = None
    entry_order_set_time_ms: Optional[int] = None
    entry_order_level: Optional[float] = None
    support_search_no: Optional[int] = None
    support_type: str = ""
    support_source_macd_1m_wave_no: Any = np.nan
    raw_support_level: Any = np.nan
    reentry_offset_down: float = 0.0
    first_stop_set_time_ms: Optional[int] = None
    first_stop_level: Optional[float] = None
    current_stop_level: Optional[float] = None
    current_stop_set_time_ms: Optional[int] = None
    stop_moves_count: int = 0
    stop_candidates_count: int = 0
    stop_ignored_count: int = 0
    initial_red_history_count: int = 0
    initial_red_history_start_time_ms: Optional[int] = None
    initial_red_history_end_time_ms: Optional[int] = None
    red_history: List[pd.Series] = field(default_factory=list)


@dataclass
class PendingEntryOrder:
    order_no: int
    set_time_ms: int
    level: float
    support_search_no: int
    support_type: str
    support_source_macd_1m_wave_no: Any
    support_source_wave_start_ms: Any
    support_source_wave_end_ms: Any
    raw_support_level: Any = np.nan
    reentry_offset_down: float = 0.0
    status: str = "pending"


def require_columns(df: pd.DataFrame, cols: List[str], sheet_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Sheet '{sheet_name}' is missing required columns: {missing}")


def normalize_dataset(
    summary: pd.DataFrame,
    adx_hits: pd.DataFrame,
    minute_candles: pd.DataFrame,
    macd_1m_waves: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_columns(summary, ["entry_id", "entry_datetime", "entry_price", "entry_ms_utc", "wave6_start_time"], "summary")
    require_columns(minute_candles, ["entry_id", "minute_mts_utc", "open", "high", "low", "close"], "minute_candles")
    require_columns(macd_1m_waves, ["entry_id", "macd_1m_wave_no", "macd_1m_wave_sign", "wave_end_ms_utc"], "macd_1m_waves")

    summary = summary.copy()
    adx_hits = adx_hits.copy()
    minute_candles = minute_candles.copy()
    macd_1m_waves = macd_1m_waves.copy()

    for df in (summary, adx_hits, minute_candles, macd_1m_waves):
        if "entry_id" in df.columns:
            df["entry_id"] = pd.to_numeric(df["entry_id"], errors="coerce").astype("Int64")

    summary["entry_price"] = pd.to_numeric(summary["entry_price"], errors="coerce")
    summary["entry_ms_utc"] = pd.to_numeric(summary["entry_ms_utc"], errors="coerce")
    summary["wave6_start_ms_utc"] = summary["wave6_start_time"].apply(
        lambda x: parse_local_time_to_ms(x, args.tz_name, args.tz_offset)
    )

    for col in ["minute_mts_utc", "open", "high", "low", "close", "macd_1m_wave_no", "macd_1m_wave_sign"]:
        if col in minute_candles.columns:
            minute_candles[col] = pd.to_numeric(minute_candles[col], errors="coerce")

    for col in [
        "macd_1m_wave_no",
        "macd_1m_wave_sign",
        "real_wave_start_ms_utc",
        "range_wave_start_ms_utc",
        "wave_end_ms_utc",
    ]:
        if col in macd_1m_waves.columns:
            macd_1m_waves[col] = pd.to_numeric(macd_1m_waves[col], errors="coerce")

    if "candle_mts_utc" in adx_hits.columns:
        adx_hits["candle_mts_utc"] = pd.to_numeric(adx_hits["candle_mts_utc"], errors="coerce")

    return summary, adx_hits, minute_candles, macd_1m_waves


def base_entry_fields(entry: pd.Series, args: argparse.Namespace) -> Dict[str, Any]:
    entry_ms = safe_int(entry.get("entry_ms_utc"))
    wave6_ms = safe_int(entry.get("wave6_start_ms_utc"))
    return {
        "entry_id": int(entry["entry_id"]),
        "source_row": safe_int(entry.get("source_row"), np.nan),
        "original_entry_datetime": entry.get("entry_datetime", ""),
        "original_entry_price": safe_float(entry.get("entry_price")),
        "entry_ms_utc": entry_ms if entry_ms is not None else np.nan,
        "wave6_start_time": ms_to_local(wave6_ms, args.tz_name, args.tz_offset),
        "wave6_start_ms_utc": wave6_ms if wave6_ms is not None else np.nan,
    }


def support_result_to_row(sr: SupportResult) -> Dict[str, Any]:
    return {
        "support_found": bool(sr.found),
        "support_level": sr.support_level if sr.support_level is not None else np.nan,
        "support_type": sr.support_type,
        "support_reason": sr.reason,
        "support_threshold": sr.threshold,
        "support_low_count": sr.lows_count,
        "support_lower_cluster_count": sr.lower_cluster_count,
        "support_base_cluster_count": sr.base_cluster_count,
        "support_lower_cluster_lows": sr.lower_cluster_lows,
        "support_base_cluster_lows": sr.base_cluster_lows,
        "support_lower_cluster_times": sr.lower_cluster_times,
        "support_base_cluster_times": sr.base_cluster_times,
        "support_sorted_lows": sr.sorted_lows,
        "support_min_low_time": sr.source_min_low_time,
        "support_min_low_mts_utc": sr.source_min_low_mts_utc,
    }


def make_red_history_row(row: pd.Series, origin: str) -> pd.Series:
    """Return a copy of a red candle row tagged with its origin in red_history."""
    out = row.copy()
    out["red_history_origin"] = origin
    return out


def append_red_history(trade: ActiveTrade, row: pd.Series, origin: str) -> bool:
    """Append a red candle to a trade history without duplicating the same minute.

    The initial history may already contain the entry candle. When the main loop
    later visits that same candle, we do not append it twice.
    """
    if candle_color(row) != "red":
        return False
    mts = safe_int(row.get("minute_mts_utc"))
    if mts is not None:
        for existing in trade.red_history:
            if safe_int(existing.get("minute_mts_utc")) == int(mts):
                return False
    trade.red_history.append(make_red_history_row(row, origin))
    return True


def build_initial_red_history(
    minutes: pd.DataFrame,
    entry_candle_start_ms: int,
    limit: int = PRE_ENTRY_RED_HISTORY_LIMIT,
) -> Tuple[List[pd.Series], Optional[int], Optional[int]]:
    """Build pre-entry red context for a newly opened trade.

    Fixed rule for this branch: take up to the last 20 red candles whose
    minute_start is less than or equal to the entry candle start. The entry
    candle is therefore included when it is red. This applies equally to the
    initial trade and to every re-entry.

    Only candles already present in the input dataset can be used. If the
    dataset starts exactly at the initial entry, the initial trade can only get
    context from the entry candle itself and later candles inside the dataset.
    """
    if minutes is None or minutes.empty:
        return [], None, None
    df = minutes.dropna(subset=["minute_mts_utc"]).copy()
    df["minute_mts_utc"] = pd.to_numeric(df["minute_mts_utc"], errors="coerce")
    df = df.dropna(subset=["minute_mts_utc"])
    df = df[df["minute_mts_utc"] <= int(entry_candle_start_ms)].sort_values("minute_mts_utc")
    reds: List[pd.Series] = []
    for _, r in df.iterrows():
        if candle_color(r) == "red":
            reds.append(make_red_history_row(r, "pre_entry_context"))
    if limit > 0:
        reds = reds[-int(limit):]
    if not reds:
        return [], None, None
    first_ms = safe_int(reds[0].get("minute_mts_utc"))
    last_ms = safe_int(reds[-1].get("minute_mts_utc"))
    return reds, first_ms, last_ms


def find_basis_red_candle(red_history: List[pd.Series], green_high: float) -> Optional[pd.Series]:
    """Search left: nearest previous red candle whose high is >= current green high."""
    for red in reversed(red_history):
        if safe_float(red.get("high")) >= float(green_high):
            return red
    return None


def get_earliest_adx_stop_info(adx_hits: pd.DataFrame, entry_id: int, range_start_ms: int, wave6_ms: int) -> Optional[Dict[str, Any]]:
    """Return the earliest ADX M5 stop event inside the research range.

    `adx_hits.candle_mts_utc` is the START of the 5m candle whose closed ADX > threshold.
    In a historical simulation this fact is only known after that 5m candle closes,
    so the actionable detection time is candle_mts_utc + 5 minutes.
    """
    if adx_hits is None or adx_hits.empty or "candle_mts_utc" not in adx_hits.columns:
        return None
    df = adx_hits[adx_hits["entry_id"].astype("Int64") == int(entry_id)].copy()
    if df.empty:
        return None
    df["candle_mts_utc"] = pd.to_numeric(df["candle_mts_utc"], errors="coerce")
    df = df.dropna(subset=["candle_mts_utc"])
    if df.empty:
        return None
    df["adx_detection_ms_utc"] = df["candle_mts_utc"].astype("int64") + TF_MS_5M
    df = df[(df["adx_detection_ms_utc"] >= int(range_start_ms)) & (df["adx_detection_ms_utc"] < int(wave6_ms))]
    if df.empty:
        return None
    row = df.sort_values(["adx_detection_ms_utc", "candle_mts_utc"], ascending=[True, True]).iloc[0]
    return {
        "adx_detection_ms_utc": int(row["adx_detection_ms_utc"]),
        "adx_5m_candle_mts_utc": int(row["candle_mts_utc"]),
        "adx_14_4": safe_float(row.get("adx_14_4")),
        "plus_di_14": safe_float(row.get("plus_di_14")),
        "minus_di_14": safe_float(row.get("minus_di_14")),
        "wave_no": safe_int(row.get("wave_no"), np.nan),
    }


def previous_negative_wave_ready_from_current_minute(
    waves_df: pd.DataFrame,
    current_minute_row: pd.Series,
    entry_id: int,
    at_ms: int,
    already_checked: set,
) -> Optional[pd.Series]:
    """Return ONLY the previous negative 1m MACD wave when it is actionable now.

    The level for re-entry may be searched only on the previous negative 1m MACD
    half-wave, after the first candle of the following positive half-wave has closed.

    Therefore the current minute must belong to a positive 1m MACD half-wave, and
    the previous wave number must be negative. The negative wave becomes actionable
    at wave_end_ms_utc + 60_000. Older unchecked waves are intentionally NOT used.
    """
    if waves_df is None or waves_df.empty:
        return None

    cur_wave_no = safe_int(current_minute_row.get("macd_1m_wave_no"))
    cur_wave_sign = safe_int(current_minute_row.get("macd_1m_wave_sign"))
    if cur_wave_no is None or cur_wave_sign != 1:
        return None

    prev_wave_no = int(cur_wave_no) - 1
    if prev_wave_no in already_checked:
        return None

    df = waves_df[waves_df["entry_id"].astype("Int64") == int(entry_id)].copy()
    if df.empty:
        return None
    df = df[
        (df["macd_1m_wave_no"] == int(prev_wave_no))
        & (df["macd_1m_wave_sign"] == -1)
    ].copy()
    if df.empty:
        return None

    row = df.iloc[0]
    wave_end_ms = safe_int(row.get("wave_end_ms_utc"))
    if wave_end_ms is None:
        return None

    search_ready_ms = int(wave_end_ms) + TF_MS_1M
    if int(at_ms) < search_ready_ms:
        return None

    return row


def candles_for_macd_wave(minute_df: pd.DataFrame, entry_id: int, wave_no: int) -> pd.DataFrame:
    if minute_df is None or minute_df.empty:
        return pd.DataFrame()
    df = minute_df[minute_df["entry_id"].astype("Int64") == int(entry_id)].copy()
    if "macd_1m_wave_no" not in df.columns:
        return pd.DataFrame()
    df["macd_1m_wave_no"] = pd.to_numeric(df["macd_1m_wave_no"], errors="coerce")
    df = df[df["macd_1m_wave_no"] == int(wave_no)].copy()
    df = df.sort_values("minute_mts_utc").reset_index(drop=True)
    return df


def process_entry(
    entry: pd.Series,
    adx_hits: pd.DataFrame,
    all_minutes: pd.DataFrame,
    all_macd_waves: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    entry_id = int(entry["entry_id"])
    entry_price = safe_float(entry.get("entry_price"))
    entry_ms = safe_int(entry.get("entry_ms_utc"))
    wave6_ms = safe_int(entry.get("wave6_start_ms_utc"))

    trades_rows: List[Dict[str, Any]] = []
    stop_events_rows: List[Dict[str, Any]] = []
    support_rows: List[Dict[str, Any]] = []
    entry_order_rows: List[Dict[str, Any]] = []

    summary_row = base_entry_fields(entry, args)
    summary_row.update(
        {
            "final_reason": "",
            "adx_stop_time": "",
            "adx_stop_ms_utc": np.nan,
            "adx_5m_candle_start_time": "",
            "adx_5m_candle_mts_utc": np.nan,
            "adx_14_4": np.nan,
            "adx_plus_di_14": np.nan,
            "adx_minus_di_14": np.nan,
            "trades_count": 0,
            "closed_trades_count": 0,
            "stops_triggered_count": 0,
            "support_searches_count": 0,
            "entry_orders_count": 0,
            "reentries_executed_count": 0,
            "last_state": "",
            "notes": "",
        }
    )

    if entry_ms is None or pd.isna(entry_price):
        summary_row["final_reason"] = "bad_entry_data"
        return trades_rows, stop_events_rows, support_rows, entry_order_rows, summary_row
    if wave6_ms is None:
        summary_row["final_reason"] = "no_wave6"
        return trades_rows, stop_events_rows, support_rows, entry_order_rows, summary_row

    range_start_ms = align_ms_down(int(entry_ms), TF_MS_1M)
    range_end_ms = int(wave6_ms)  # A: до wave6_start_time не включительно

    # v7 keeps two minute DataFrames:
    #   - minutes_all_for_entry: all exported 1m rows for this entry_id, including
    #     the pre-entry context generated by macd_halfwaves_adx_minutes_v5.py;
    #   - minutes: the actual trading/research range [entry_minute, wave6_start).
    # The pre-entry rows must NOT be processed as live candles before the original
    # entry, but they must be available when seeding initial red_history.
    minutes_all_for_entry = all_minutes[all_minutes["entry_id"].astype("Int64") == int(entry_id)].copy()
    minutes_all_for_entry = minutes_all_for_entry.dropna(subset=["minute_mts_utc"]).copy()
    minutes_all_for_entry["minute_mts_utc"] = pd.to_numeric(minutes_all_for_entry["minute_mts_utc"], errors="coerce")
    minutes_all_for_entry = minutes_all_for_entry.dropna(subset=["minute_mts_utc"]).copy()
    minutes_all_for_entry = minutes_all_for_entry[minutes_all_for_entry["minute_mts_utc"] < range_end_ms]
    minutes_all_for_entry = minutes_all_for_entry.sort_values("minute_mts_utc").reset_index(drop=True)

    minutes = minutes_all_for_entry[
        (minutes_all_for_entry["minute_mts_utc"] >= range_start_ms)
        & (minutes_all_for_entry["minute_mts_utc"] < range_end_ms)
    ].copy()
    minutes = minutes.sort_values("minute_mts_utc").reset_index(drop=True)

    if minutes.empty:
        summary_row["final_reason"] = "no_minute_data"
        return trades_rows, stop_events_rows, support_rows, entry_order_rows, summary_row

    adx_stop_info = get_earliest_adx_stop_info(adx_hits, entry_id, range_start_ms, range_end_ms)
    adx_stop_ms = None
    if adx_stop_info is not None:
        adx_stop_ms = safe_int(adx_stop_info.get("adx_detection_ms_utc"))
        adx_5m_candle_ms = safe_int(adx_stop_info.get("adx_5m_candle_mts_utc"))
        summary_row["adx_stop_time"] = ms_to_local(adx_stop_ms, args.tz_name, args.tz_offset)
        summary_row["adx_stop_ms_utc"] = int(adx_stop_ms) if adx_stop_ms is not None else np.nan
        summary_row["adx_5m_candle_start_time"] = ms_to_local(adx_5m_candle_ms, args.tz_name, args.tz_offset)
        summary_row["adx_5m_candle_mts_utc"] = int(adx_5m_candle_ms) if adx_5m_candle_ms is not None else np.nan
        summary_row["adx_14_4"] = adx_stop_info.get("adx_14_4", np.nan)
        summary_row["adx_plus_di_14"] = adx_stop_info.get("plus_di_14", np.nan)
        summary_row["adx_minus_di_14"] = adx_stop_info.get("minus_di_14", np.nan)

    checked_support_waves: set = set()
    support_search_no = 0
    order_no = 0
    trade_no = 1
    stops_triggered_count = 0
    reentries_executed_count = 0

    initial_entry_candle_start_ms = align_ms_down(int(entry_ms), TF_MS_1M)
    initial_red_history, initial_red_start_ms, initial_red_end_ms = build_initial_red_history(
        minutes_all_for_entry,
        initial_entry_candle_start_ms,
    )
    active_trade: Optional[ActiveTrade] = ActiveTrade(
        trade_no=trade_no,
        trade_source="initial",
        entry_time_ms=int(entry_ms),
        entry_price=float(entry_price),
        initial_red_history_count=len(initial_red_history),
        initial_red_history_start_time_ms=initial_red_start_ms,
        initial_red_history_end_time_ms=initial_red_end_ms,
        red_history=initial_red_history,
    )
    pending_order: Optional[PendingEntryOrder] = None
    active_trade_open_idx = 0

    def compute_trade_path_stats(trade: ActiveTrade, exit_time_ms: Optional[int]) -> Dict[str, Any]:
        """Calculate intratrade movement statistics from 1m candles.

        For SHORT research, two movement depths are especially useful:
          - max_favorable_move_pct_short: how far price moved DOWN from entry
            before the trade was closed;
          - max_adverse_move_pct_short: how far price moved UP against entry
            before the trade was closed.

        Candle inclusion is interval-based and avoids pre-entry candles:
          candle_end > trade_entry_time and candle_start < effective_exit_time.
        For re-entries whose execution is only known at the close of the trigger
        candle, this intentionally does not count that trigger candle as part of
        the post-entry path.
        """
        if minutes_all_for_entry is None or minutes_all_for_entry.empty:
            return {}
        if not trade.entry_price or pd.isna(trade.entry_price):
            return {}

        effective_exit_ms = int(exit_time_ms) if exit_time_ms is not None else int(range_end_ms)
        entry_time_ms = int(trade.entry_time_ms)
        if effective_exit_ms <= entry_time_ms:
            return {}

        work = minutes_all_for_entry.copy()
        work["minute_mts_utc"] = pd.to_numeric(work["minute_mts_utc"], errors="coerce")
        work = work.dropna(subset=["minute_mts_utc"]).copy()
        work["minute_mts_utc"] = work["minute_mts_utc"].astype("int64")
        for col in ["open", "high", "low", "close"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")

        candle_end = work["minute_mts_utc"] + TF_MS_1M
        work = work[(candle_end > entry_time_ms) & (work["minute_mts_utc"] < effective_exit_ms)].copy()
        work = work.dropna(subset=["high", "low"]).sort_values("minute_mts_utc").reset_index(drop=True)
        if work.empty:
            return {
                "trade_candles_count": 0,
                "max_high_during_trade": np.nan,
                "max_high_time": "",
                "max_high_ms_utc": np.nan,
                "min_low_during_trade": np.nan,
                "min_low_time": "",
                "min_low_ms_utc": np.nan,
                "max_favorable_move_pct_short": np.nan,
                "max_adverse_move_pct_short": np.nan,
                "trade_price_range_pct": np.nan,
                "capture_of_max_favorable_pct": np.nan,
            }

        max_high = float(work["high"].max())
        min_low = float(work["low"].min())
        max_high_row = work[work["high"] == max_high].iloc[0]
        min_low_row = work[work["low"] == min_low].iloc[0]
        entry_price = float(trade.entry_price)

        # For a SHORT position, favorable depth is movement DOWN from entry,
        # adverse depth is movement UP against entry. If price never moved in
        # that direction, report 0 instead of a negative depth.
        max_favorable_pct = max(0.0, ((entry_price - min_low) / entry_price) * 100.0)
        max_adverse_pct = max(0.0, ((max_high - entry_price) / entry_price) * 100.0)
        trade_range_pct = ((max_high - min_low) / entry_price) * 100.0

        return {
            "trade_candles_count": int(len(work)),
            "max_high_during_trade": round(max_high, 8),
            "max_high_time": ms_to_local(int(max_high_row["minute_mts_utc"]), args.tz_name, args.tz_offset),
            "max_high_ms_utc": int(max_high_row["minute_mts_utc"]),
            "min_low_during_trade": round(min_low, 8),
            "min_low_time": ms_to_local(int(min_low_row["minute_mts_utc"]), args.tz_name, args.tz_offset),
            "min_low_ms_utc": int(min_low_row["minute_mts_utc"]),
            "max_favorable_move_pct_short": round(max_favorable_pct, 6),
            "max_adverse_move_pct_short": round(max_adverse_pct, 6),
            "trade_price_range_pct": round(trade_range_pct, 6),
            # This field is filled below after short_result_pct is known.
            "capture_of_max_favorable_pct": np.nan,
        }

    def close_or_snapshot_trade(
        trade: ActiveTrade,
        exit_reason: str,
        exit_time_ms: Optional[int] = None,
        exit_price: Optional[float] = None,
        stop_trigger_event_no: Optional[int] = None,
    ) -> None:
        nonlocal trades_rows
        duration_min = np.nan
        if exit_time_ms is not None:
            duration_min = round((int(exit_time_ms) - int(trade.entry_time_ms)) / 60000.0, 2)
        stop_dev_pct = np.nan
        short_result_pct = np.nan
        if exit_price is not None and not pd.isna(exit_price) and trade.entry_price:
            stop_dev_pct = round(((float(exit_price) - float(trade.entry_price)) / float(trade.entry_price)) * 100.0, 6)
            short_result_pct = round(((float(trade.entry_price) - float(exit_price)) / float(trade.entry_price)) * 100.0, 6)

        first_stop_delay_min = np.nan
        last_stop_delay_min = np.nan
        first_stop_distance_pct = np.nan
        last_stop_distance_pct = np.nan
        if trade.first_stop_set_time_ms is not None:
            first_stop_delay_min = round((int(trade.first_stop_set_time_ms) - int(trade.entry_time_ms)) / 60000.0, 2)
        if trade.current_stop_set_time_ms is not None:
            last_stop_delay_min = round((int(trade.current_stop_set_time_ms) - int(trade.entry_time_ms)) / 60000.0, 2)
        if trade.first_stop_level is not None and trade.entry_price:
            first_stop_distance_pct = round(((float(trade.first_stop_level) - float(trade.entry_price)) / float(trade.entry_price)) * 100.0, 6)
        if trade.current_stop_level is not None and trade.entry_price:
            last_stop_distance_pct = round(((float(trade.current_stop_level) - float(trade.entry_price)) / float(trade.entry_price)) * 100.0, 6)

        path_stats = compute_trade_path_stats(trade, exit_time_ms)
        max_favorable = safe_float(path_stats.get("max_favorable_move_pct_short"), np.nan)
        if not pd.isna(short_result_pct) and not pd.isna(max_favorable) and float(max_favorable) > 0:
            path_stats["capture_of_max_favorable_pct"] = round((float(short_result_pct) / float(max_favorable)) * 100.0, 6)

        row = base_entry_fields(entry, args)
        row.update(
            {
                "trade_no": trade.trade_no,
                "trade_source": trade.trade_source,
                "trade_entry_time": ms_to_local(trade.entry_time_ms, args.tz_name, args.tz_offset),
                "trade_entry_ms_utc": int(trade.entry_time_ms),
                "trade_entry_price": float(trade.entry_price),
                "entry_order_no": trade.entry_order_no if trade.entry_order_no is not None else np.nan,
                "entry_order_set_time": ms_to_local(trade.entry_order_set_time_ms, args.tz_name, args.tz_offset),
                "entry_order_level": trade.entry_order_level if trade.entry_order_level is not None else np.nan,
                "raw_support_level": trade.raw_support_level if trade.raw_support_level is not None else np.nan,
                "reentry_offset_down": trade.reentry_offset_down,
                "support_search_no": trade.support_search_no if trade.support_search_no is not None else np.nan,
                "support_type": trade.support_type,
                "support_source_macd_1m_wave_no": trade.support_source_macd_1m_wave_no,
                "first_exit_stop_set_time": ms_to_local(trade.first_stop_set_time_ms, args.tz_name, args.tz_offset),
                "first_exit_stop_level": trade.first_stop_level if trade.first_stop_level is not None else np.nan,
                "last_exit_stop_set_time": ms_to_local(trade.current_stop_set_time_ms, args.tz_name, args.tz_offset),
                "last_exit_stop_level": trade.current_stop_level if trade.current_stop_level is not None else np.nan,
                "first_stop_delay_minutes": first_stop_delay_min,
                "last_stop_delay_minutes": last_stop_delay_min,
                "first_stop_distance_pct_from_entry": first_stop_distance_pct,
                "last_stop_distance_pct_from_entry": last_stop_distance_pct,
                "stop_moves_count": trade.stop_moves_count,
                "stop_candidates_count": trade.stop_candidates_count,
                "stop_ignored_count": trade.stop_ignored_count,
                "initial_red_history_count": trade.initial_red_history_count,
                "initial_red_history_start_time": ms_to_local(trade.initial_red_history_start_time_ms, args.tz_name, args.tz_offset),
                "initial_red_history_start_ms_utc": trade.initial_red_history_start_time_ms if trade.initial_red_history_start_time_ms is not None else np.nan,
                "initial_red_history_end_time": ms_to_local(trade.initial_red_history_end_time_ms, args.tz_name, args.tz_offset),
                "initial_red_history_end_ms_utc": trade.initial_red_history_end_time_ms if trade.initial_red_history_end_time_ms is not None else np.nan,
                "exit_time": ms_to_local(exit_time_ms, args.tz_name, args.tz_offset),
                "exit_ms_utc": int(exit_time_ms) if exit_time_ms is not None else np.nan,
                "exit_price": float(exit_price) if exit_price is not None and not pd.isna(exit_price) else np.nan,
                "exit_reason": exit_reason,
                "stop_trigger_event_no": stop_trigger_event_no if stop_trigger_event_no is not None else np.nan,
                "stop_deviation_pct_from_trade_entry": stop_dev_pct,
                "short_result_pct": short_result_pct,
                "duration_minutes": duration_min,
                "trade_status": "closed" if exit_price is not None else "open_or_unclosed",
            }
        )
        row.update(path_stats)
        trades_rows.append(row)

    def search_and_maybe_place_entry_order(
        at_ms: int,
        current_price: float,
        current_minute_row: pd.Series,
        trigger_trade_no: Optional[int],
        trigger_stop_no: Optional[int],
    ) -> None:
        nonlocal support_search_no, order_no, pending_order, entry_order_rows, support_rows
        wave = previous_negative_wave_ready_from_current_minute(
            all_macd_waves, current_minute_row, entry_id, at_ms, checked_support_waves
        )
        if wave is None:
            return
        wave_no = safe_int(wave.get("macd_1m_wave_no"))
        if wave_no is None:
            return
        checked_support_waves.add(int(wave_no))

        wave_candles = candles_for_macd_wave(all_minutes, entry_id, int(wave_no))
        support_search_no += 1
        sr = find_support_level_from_candles(wave_candles, float(args.support_threshold), args.tz_name, args.tz_offset)

        support_level = sr.support_level if sr.found else None
        reentry_offset_down = float(args.reentry_offset_down)
        reentry_order_level = None
        usable = False
        skip_reason = ""
        if sr.found and support_level is not None:
            # The support level is the raw level found by find_support_level().
            # The actual re-entry STOP order is shifted DOWN by an absolute
            # price offset from that support level.
            reentry_order_level = float(support_level) - reentry_offset_down
            if float(reentry_order_level) < float(current_price):
                usable = True
            else:
                skip_reason = "reentry_order_level_not_below_current_price"
        else:
            skip_reason = sr.reason or "support_not_found"

        source_wave_start_ms = safe_int(wave.get("range_wave_start_ms_utc"), safe_int(wave.get("real_wave_start_ms_utc"), None))
        source_wave_end_ms = safe_int(wave.get("wave_end_ms_utc"), None)
        support_search_ready_ms = (int(source_wave_end_ms) + TF_MS_1M) if source_wave_end_ms is not None else None
        current_macd_1m_wave_no = safe_int(current_minute_row.get("macd_1m_wave_no"), np.nan)
        current_macd_1m_wave_sign = safe_int(current_minute_row.get("macd_1m_wave_sign"), np.nan)

        row = base_entry_fields(entry, args)
        row.update(
            {
                "support_search_no": support_search_no,
                "search_time": ms_to_local(at_ms, args.tz_name, args.tz_offset),
                "search_ms_utc": int(at_ms),
                "trigger_trade_no": trigger_trade_no if trigger_trade_no is not None else np.nan,
                "trigger_stop_event_no": trigger_stop_no if trigger_stop_no is not None else np.nan,
                "current_price_for_check": float(current_price),
                "raw_support_level": float(support_level) if support_level is not None else np.nan,
                "reentry_offset_down": reentry_offset_down,
                "reentry_order_level_candidate": float(reentry_order_level) if reentry_order_level is not None else np.nan,
                "current_macd_1m_wave_no": current_macd_1m_wave_no,
                "current_macd_1m_wave_sign": current_macd_1m_wave_sign,
                "source_macd_1m_wave_no": int(wave_no),
                "source_macd_1m_wave_sign": safe_int(wave.get("macd_1m_wave_sign"), np.nan),
                "source_wave_start_time": ms_to_local(source_wave_start_ms, args.tz_name, args.tz_offset),
                "source_wave_start_ms_utc": source_wave_start_ms if source_wave_start_ms is not None else np.nan,
                "source_wave_end_time": ms_to_local(source_wave_end_ms, args.tz_name, args.tz_offset),
                "source_wave_end_ms_utc": source_wave_end_ms if source_wave_end_ms is not None else np.nan,
                "support_search_ready_time": ms_to_local(support_search_ready_ms, args.tz_name, args.tz_offset),
                "support_search_ready_ms_utc": support_search_ready_ms if support_search_ready_ms is not None else np.nan,
                "support_is_usable": usable,
                "support_skip_reason": skip_reason,
            }
        )
        row.update(support_result_to_row(sr))
        support_rows.append(row)

        if not usable:
            return

        order_no += 1
        pending_order = PendingEntryOrder(
            order_no=order_no,
            set_time_ms=int(at_ms),
            level=float(reentry_order_level),
            support_search_no=support_search_no,
            support_type=sr.support_type,
            support_source_macd_1m_wave_no=int(wave_no),
            support_source_wave_start_ms=source_wave_start_ms if source_wave_start_ms is not None else np.nan,
            support_source_wave_end_ms=source_wave_end_ms if source_wave_end_ms is not None else np.nan,
            raw_support_level=float(support_level),
            reentry_offset_down=reentry_offset_down,
        )

        order_row = base_entry_fields(entry, args)
        order_row.update(
            {
                "entry_order_no": order_no,
                "order_status": "pending",
                "order_set_time": ms_to_local(at_ms, args.tz_name, args.tz_offset),
                "order_set_ms_utc": int(at_ms),
                "entry_order_level": float(reentry_order_level),
                "raw_support_level": float(support_level),
                "reentry_offset_down": reentry_offset_down,
                "support_search_no": support_search_no,
                "support_type": sr.support_type,
                "source_macd_1m_wave_no": int(wave_no),
                "source_wave_start_time": ms_to_local(source_wave_start_ms, args.tz_name, args.tz_offset),
                "source_wave_end_time": ms_to_local(source_wave_end_ms, args.tz_name, args.tz_offset),
                "support_search_ready_time": ms_to_local(support_search_ready_ms, args.tz_name, args.tz_offset),
                "support_search_ready_ms_utc": support_search_ready_ms if support_search_ready_ms is not None else np.nan,
                "trigger_time": "",
                "trigger_ms_utc": np.nan,
                "trigger_candle_low": np.nan,
                "cancel_reason": "",
            }
        )
        entry_order_rows.append(order_row)

    def update_latest_order_status(order: PendingEntryOrder, status: str, mrow: Optional[pd.Series] = None, cancel_reason: str = "") -> None:
        if not entry_order_rows:
            return
        for row in reversed(entry_order_rows):
            if int(row.get("entry_order_no", -1)) == int(order.order_no):
                row["order_status"] = status
                row["cancel_reason"] = cancel_reason
                if mrow is not None:
                    mts = int(mrow["minute_mts_utc"])
                    row["trigger_time"] = ms_to_local(minute_end_ms(mts), args.tz_name, args.tz_offset)
                    row["trigger_ms_utc"] = minute_end_ms(mts)
                    row["trigger_candle_low"] = safe_float(mrow.get("low"))
                return

    final_reason = "wave6_reached"
    last_state = "position_active"

    for idx, mrow in minutes.iterrows():
        m_start = int(mrow["minute_mts_utc"])
        m_end = minute_end_ms(m_start)

        # ADX stop: ADX is M5 data. A 5m ADX hit is actionable only after the
        # 5m candle closes, i.e. candle_mts_utc + 5 minutes. adx_stop_ms stores
        # this detection time, not the 5m candle start. Active trade is closed
        # by the first market level available at detection: OPEN of current 1m candle.
        if adx_stop_ms is not None and m_start >= int(adx_stop_ms):
            final_reason = "adx_gt_25"
            if active_trade is not None:
                forced_exit_price = safe_float(mrow.get("open"))
                forced_exit_time_ms = int(m_start)

                event_no = len(stop_events_rows) + 1
                event_row = base_entry_fields(entry, args)
                event_row.update(
                    {
                        "stop_event_no": event_no,
                        "trade_no": active_trade.trade_no,
                        "event_type": "forced_close_adx_gt_25",
                        "event_time": ms_to_local(forced_exit_time_ms, args.tz_name, args.tz_offset),
                        "event_ms_utc": forced_exit_time_ms,
                        "minute_start_time": ms_to_local(m_start, args.tz_name, args.tz_offset),
                        "minute_mts_utc": m_start,
                        "candle_color": candle_color(mrow),
                        "candle_open": safe_float(mrow.get("open")),
                        "candle_high": safe_float(mrow.get("high")),
                        "candle_low": safe_float(mrow.get("low")),
                        "candle_close": safe_float(mrow.get("close")),
                        "old_stop_level": active_trade.current_stop_level if active_trade.current_stop_level is not None else np.nan,
                        "new_stop_level": active_trade.current_stop_level if active_trade.current_stop_level is not None else np.nan,
                        "candidate_stop_level": np.nan,
                        "basis_red_time": "",
                        "basis_red_mts_utc": np.nan,
                        "basis_red_high": np.nan,
                        "basis_green_time": "",
                        "basis_green_mts_utc": np.nan,
                        "basis_green_high": np.nan,
                        "forced_exit_price": forced_exit_price,
                        "reason": "adx_gt_25_close_at_current_1m_open",
                    }
                )
                stop_events_rows.append(event_row)

                close_or_snapshot_trade(
                    active_trade,
                    "study_ended_adx_gt_25",
                    forced_exit_time_ms,
                    forced_exit_price,
                    event_no,
                )
                active_trade = None
            if pending_order is not None:
                update_latest_order_status(pending_order, "canceled", None, "adx_gt_25")
                pending_order = None
            last_state = "terminated_by_adx"
            break

        # If flat with a pending entry order, monitor order execution.
        if active_trade is None and pending_order is not None:
            if safe_float(mrow.get("low")) <= float(pending_order.level):
                update_latest_order_status(pending_order, "triggered", mrow)
                trade_no += 1
                reentries_executed_count += 1
                # The re-entry order is detected on this 1m candle. Its exact
                # intraminute path is unknown, but by strategy rule we still include
                # this candle in the initial red context if it is red.
                reentry_red_history, reentry_red_start_ms, reentry_red_end_ms = build_initial_red_history(
                    minutes_all_for_entry,
                    m_start,
                )
                active_trade = ActiveTrade(
                    trade_no=trade_no,
                    trade_source="reentry",
                    entry_time_ms=m_end,
                    entry_price=float(pending_order.level),
                    entry_order_no=pending_order.order_no,
                    entry_order_set_time_ms=pending_order.set_time_ms,
                    entry_order_level=pending_order.level,
                    support_search_no=pending_order.support_search_no,
                    support_type=pending_order.support_type,
                    support_source_macd_1m_wave_no=pending_order.support_source_macd_1m_wave_no,
                    raw_support_level=pending_order.raw_support_level,
                    reentry_offset_down=pending_order.reentry_offset_down,
                    initial_red_history_count=len(reentry_red_history),
                    initial_red_history_start_time_ms=reentry_red_start_ms,
                    initial_red_history_end_time_ms=reentry_red_end_ms,
                    red_history=reentry_red_history,
                )
                pending_order = None
                last_state = "position_active_after_reentry"
                # Entry is assumed known at candle close; do not build/trigger exit stop inside same 1m candle.
                continue
            last_state = "pending_reentry_order"
            continue

        # If flat without a pending order, try one new completed negative 1m MACD wave at a time.
        if active_trade is None and pending_order is None:
            search_and_maybe_place_entry_order(
                at_ms=m_end,
                current_price=safe_float(mrow.get("close")),
                current_minute_row=mrow,
                trigger_trade_no=None,
                trigger_stop_no=None,
            )
            last_state = "flat_waiting_for_support_or_entry"
            continue

        assert active_trade is not None

        # Active exit stop must be checked on every candle, regardless of candle color.
        if active_trade.current_stop_level is not None and safe_float(mrow.get("high")) >= float(active_trade.current_stop_level):
            stops_triggered_count += 1
            event_no = len(stop_events_rows) + 1
            trigger_level = float(active_trade.current_stop_level)
            trigger_time_ms = m_end
            event_row = base_entry_fields(entry, args)
            event_row.update(
                {
                    "stop_event_no": event_no,
                    "trade_no": active_trade.trade_no,
                    "event_type": "exit_stop_triggered",
                    "event_time": ms_to_local(trigger_time_ms, args.tz_name, args.tz_offset),
                    "event_ms_utc": trigger_time_ms,
                    "minute_start_time": ms_to_local(m_start, args.tz_name, args.tz_offset),
                    "minute_mts_utc": m_start,
                    "candle_color": candle_color(mrow),
                    "candle_open": safe_float(mrow.get("open")),
                    "candle_high": safe_float(mrow.get("high")),
                    "candle_low": safe_float(mrow.get("low")),
                    "candle_close": safe_float(mrow.get("close")),
                    "old_stop_level": active_trade.current_stop_level,
                    "new_stop_level": active_trade.current_stop_level,
                    "candidate_stop_level": np.nan,
                    "basis_red_time": "",
                    "basis_red_mts_utc": np.nan,
                    "basis_red_high": np.nan,
                    "basis_green_time": "",
                    "basis_green_mts_utc": np.nan,
                    "basis_green_high": np.nan,
                    "reason": "minute_high_gte_stop_level",
                }
            )
            stop_events_rows.append(event_row)

            close_or_snapshot_trade(
                active_trade,
                exit_reason="exit_stop_triggered",
                exit_time_ms=trigger_time_ms,
                exit_price=trigger_level,
                stop_trigger_event_no=event_no,
            )
            finished_trade_no = active_trade.trade_no
            active_trade = None
            last_state = "flat_after_exit_stop"

            # Search for a new entry level immediately after the stop candle closes.
            search_and_maybe_place_entry_order(
                at_ms=trigger_time_ms,
                current_price=safe_float(mrow.get("close")),
                current_minute_row=mrow,
                trigger_trade_no=finished_trade_no,
                trigger_stop_no=event_no,
            )
            continue

        color = candle_color(mrow)
        if color == "red":
            append_red_history(active_trade, mrow, "after_entry")
            continue
        if color != "green":
            continue

        # Green candle: try to build/update exit stop from a previous red candle.
        green_high = safe_float(mrow.get("high"))
        basis_red = find_basis_red_candle(active_trade.red_history, green_high)
        if basis_red is None:
            event_no = len(stop_events_rows) + 1
            active_trade.stop_candidates_count += 1
            active_trade.stop_ignored_count += 1
            event_row = base_entry_fields(entry, args)
            event_row.update(
                {
                    "stop_event_no": event_no,
                    "trade_no": active_trade.trade_no,
                    "event_type": "candidate_not_found",
                    "event_time": ms_to_local(m_end, args.tz_name, args.tz_offset),
                    "event_ms_utc": m_end,
                    "minute_start_time": ms_to_local(m_start, args.tz_name, args.tz_offset),
                    "minute_mts_utc": m_start,
                    "candle_color": color,
                    "candle_open": safe_float(mrow.get("open")),
                    "candle_high": safe_float(mrow.get("high")),
                    "candle_low": safe_float(mrow.get("low")),
                    "candle_close": safe_float(mrow.get("close")),
                    "old_stop_level": active_trade.current_stop_level if active_trade.current_stop_level is not None else np.nan,
                    "new_stop_level": active_trade.current_stop_level if active_trade.current_stop_level is not None else np.nan,
                    "candidate_stop_level": np.nan,
                    "basis_red_time": "",
                    "basis_red_mts_utc": np.nan,
                    "basis_red_high": np.nan,
                    "basis_red_origin": "",
                    "basis_green_time": ms_to_local(m_start, args.tz_name, args.tz_offset),
                    "basis_green_mts_utc": m_start,
                    "basis_green_high": green_high,
                    "reason": "no_previous_red_high_gte_green_high",
                }
            )
            stop_events_rows.append(event_row)
            continue

        red_high = safe_float(basis_red.get("high"))
        candidate_stop = float(red_high) + float(args.stop_x)
        old_stop = active_trade.current_stop_level
        active_trade.stop_candidates_count += 1

        event_type = ""
        reason = ""
        new_stop = old_stop
        if old_stop is None:
            event_type = "exit_stop_set"
            reason = "first_stop_for_trade"
            new_stop = candidate_stop
            active_trade.current_stop_level = candidate_stop
            active_trade.current_stop_set_time_ms = m_end
            active_trade.first_stop_set_time_ms = m_end
            active_trade.first_stop_level = candidate_stop
        elif candidate_stop < float(old_stop):
            event_type = "exit_stop_moved_down"
            reason = "candidate_below_current_stop"
            new_stop = candidate_stop
            active_trade.current_stop_level = candidate_stop
            active_trade.current_stop_set_time_ms = m_end
            active_trade.stop_moves_count += 1
        else:
            event_type = "candidate_ignored_not_lower"
            reason = "candidate_not_below_current_stop"
            active_trade.stop_ignored_count += 1

        event_no = len(stop_events_rows) + 1
        event_row = base_entry_fields(entry, args)
        event_row.update(
            {
                "stop_event_no": event_no,
                "trade_no": active_trade.trade_no,
                "event_type": event_type,
                "event_time": ms_to_local(m_end, args.tz_name, args.tz_offset),
                "event_ms_utc": m_end,
                "minute_start_time": ms_to_local(m_start, args.tz_name, args.tz_offset),
                "minute_mts_utc": m_start,
                "candle_color": color,
                "candle_open": safe_float(mrow.get("open")),
                "candle_high": safe_float(mrow.get("high")),
                "candle_low": safe_float(mrow.get("low")),
                "candle_close": safe_float(mrow.get("close")),
                "old_stop_level": old_stop if old_stop is not None else np.nan,
                "new_stop_level": new_stop if new_stop is not None else np.nan,
                "candidate_stop_level": candidate_stop,
                "basis_red_time": ms_to_local(safe_int(basis_red.get("minute_mts_utc")), args.tz_name, args.tz_offset),
                "basis_red_mts_utc": safe_int(basis_red.get("minute_mts_utc"), np.nan),
                "basis_red_high": red_high,
                "basis_red_origin": str(basis_red.get("red_history_origin", "")),
                "basis_green_time": ms_to_local(m_start, args.tz_name, args.tz_offset),
                "basis_green_mts_utc": m_start,
                "basis_green_high": green_high,
                "reason": reason,
            }
        )
        stop_events_rows.append(event_row)

    else:
        final_reason = "wave6_reached"

    # End-of-range snapshots / pending order cancellation.
    if active_trade is not None:
        close_or_snapshot_trade(active_trade, f"study_ended_{final_reason}", int(range_end_ms), None)
        last_state = "position_open_at_study_end"
    if pending_order is not None:
        update_latest_order_status(pending_order, "unfilled", None, final_reason)
        last_state = "pending_order_unfilled_at_study_end"

    summary_row.update(
        {
            "final_reason": final_reason,
            "trades_count": len(trades_rows),
            "closed_trades_count": sum(1 for r in trades_rows if r.get("trade_status") == "closed"),
            "stops_triggered_count": stops_triggered_count,
            "support_searches_count": len(support_rows),
            "entry_orders_count": len(entry_order_rows),
            "reentries_executed_count": reentries_executed_count,
            "last_state": last_state,
            "notes": "",
        }
    )
    return trades_rows, stop_events_rows, support_rows, entry_order_rows, summary_row

'''


# Required original functions from macd_halfwaves_adx_minutes_v5.py; isolated module namespace.
ORIGINAL_WAVE_SHA256 = "c8ae9bb6ffec2aae49750c6d51059c42051be9fb0d822c32c9fbbf5c67765b1b"
EMBEDDED_WAVE_SHA256 = "c663230c6bed303fc28998979fe3dd9ba62dabb741080ae361c1947ec9e2f64f"
ORIGINAL_WAVE_SOURCE = r'''from __future__ import annotations


import argparse


import datetime as dt


import math


import time


from dataclasses import dataclass


from pathlib import Path


from typing import Any, Dict, List, Optional, Tuple


import numpy as np


import pandas as pd


try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None


TF_MS_1M = 60 * 1000


TF_MS_5M = 5 * 60 * 1000


PRE_ENTRY_RED_CONTEXT_LIMIT = 20


def align_ms_down(ms: int, tf_ms: int) -> int:
    ms = int(ms)
    tf_ms = int(tf_ms)
    return ms - (ms % tf_ms)


def find_pre_entry_red_context_start_ms(
    df_1m_with_warmup: pd.DataFrame,
    entry_minute_start_ms: int,
    limit: int = PRE_ENTRY_RED_CONTEXT_LIMIT,
) -> Tuple[int, int]:
    """Return the left export boundary that contains up to 20 red candles before entry.

    The stop script seeds red_history from the last 20 red candles whose
    minute_start <= entry_candle_start. To make those candles available in the
    dataset, the collector looks inside the already fetched 1m warmup data and
    starts minute_candles from the timestamp of the 20th red candle from the
    right. Non-red candles between that candle and entry are exported too, so
    the chronological dataset remains continuous.

    If fewer than 20 red candles exist in the fetched warmup window, export from
    the first available 1m candle. The function returns (context_start_ms,
    red_count_found_before_or_on_entry).
    """
    entry_minute_start_ms = int(entry_minute_start_ms)
    if df_1m_with_warmup is None or df_1m_with_warmup.empty:
        return entry_minute_start_ms, 0

    df = df_1m_with_warmup.dropna(subset=["mts"]).copy()
    for col in ["mts", "open", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mts", "open", "close"]).copy()
    df["mts"] = df["mts"].astype("int64")
    df = df[df["mts"] <= entry_minute_start_ms].sort_values("mts").reset_index(drop=True)
    if df.empty:
        return entry_minute_start_ms, 0

    red_df = df[df["open"] > df["close"]].copy()
    red_count = int(len(red_df))
    if red_count >= int(limit):
        context_start_ms = int(red_df.iloc[-int(limit)]["mts"])
    else:
        context_start_ms = int(df.iloc[0]["mts"])
    return context_start_ms, red_count


def fmt_local(ms: Optional[int], tz_offset_hours: int, tz_name: Optional[str]) -> str:
    if ms is None or (isinstance(ms, float) and math.isnan(ms)):
        return ""
    utc_dt = dt.datetime.fromtimestamp(int(ms) / 1000, tz=dt.timezone.utc)
    if tz_name and ZoneInfo is not None:
        local = utc_dt.astimezone(ZoneInfo(tz_name))
    else:
        local = utc_dt.astimezone(dt.timezone(dt.timedelta(hours=tz_offset_hours)))
    return local.strftime("%d.%m.%Y %H:%M")


def ema_sma_seed(values: pd.Series, length: int) -> pd.Series:
    """EMA seeded with SMA of first `length` consecutive non-NaN values."""
    alpha = 2.0 / (length + 1.0)
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)

    start: Optional[int] = None
    for i in range(0, n - length + 1):
        w = arr[i : i + length]
        if not np.isnan(w).any():
            start = i
            break
    if start is None:
        return pd.Series(out, index=values.index)

    seed_i = start + length - 1
    seed = float(np.mean(arr[start : seed_i + 1]))
    out[seed_i] = seed
    prev = seed

    for i in range(seed_i + 1, n):
        v = arr[i]
        if np.isnan(v):
            out[i] = prev
            continue
        prev = (v - prev) * alpha + prev
        out[i] = prev

    return pd.Series(out, index=values.index)


def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)
    ema_fast = ema_sma_seed(close, fast)
    ema_slow = ema_sma_seed(close, slow)
    macd = ema_fast - ema_slow
    sig = ema_sma_seed(macd, signal)
    hist = macd - sig

    out = df.copy()
    out["ema_fast"] = ema_fast
    out["ema_slow"] = ema_slow
    out["macd"] = macd
    out["signal"] = sig
    out["hist"] = hist
    return out


def wilder_rma(values: pd.Series, length: int) -> pd.Series:
    """Wilder RMA seeded by SMA over first `length` non-NaN values."""
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)

    start: Optional[int] = None
    for i in range(0, n - length + 1):
        w = arr[i : i + length]
        if not np.isnan(w).any():
            start = i
            break
    if start is None:
        return pd.Series(out, index=values.index)

    seed_i = start + length - 1
    seed = float(np.mean(arr[start : seed_i + 1]))
    out[seed_i] = seed
    prev = seed

    for i in range(seed_i + 1, n):
        v = arr[i]
        if np.isnan(v):
            out[i] = prev
            continue
        prev = ((prev * (length - 1)) + v) / length
        out[i] = prev

    return pd.Series(out, index=values.index)


def calculate_adx(df: pd.DataFrame, di_length: int = 14, adx_smoothing: int = 4) -> pd.DataFrame:
    """Classic DMI/ADX: DI length and ADX smoothing are both Wilder RMA."""
    high = pd.to_numeric(df["high"], errors="coerce").astype(float)
    low = pd.to_numeric(df["low"], errors="coerce").astype(float)
    close = pd.to_numeric(df["close"], errors="coerce").astype(float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1, skipna=True)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
        dtype="float64",
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
        dtype="float64",
    )

    atr = wilder_rma(tr, di_length)
    plus_dm_smoothed = wilder_rma(plus_dm, di_length)
    minus_dm_smoothed = wilder_rma(minus_dm, di_length)

    plus_di = 100.0 * plus_dm_smoothed / atr
    minus_di = 100.0 * minus_dm_smoothed / atr
    di_sum = plus_di + minus_di
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum.replace(0, np.nan)
    adx = wilder_rma(dx, adx_smoothing)

    out = df.copy()
    out[f"plus_di_{di_length}"] = plus_di
    out[f"minus_di_{di_length}"] = minus_di
    out[f"adx_{di_length}_{adx_smoothing}"] = adx
    return out


@dataclass
class Wave:
    idx: int
    sign: int
    start_i: int
    end_i: int


def build_waves(hist: pd.Series) -> Tuple[List[Wave], List[int]]:
    """Build MACD halfwaves using strict histogram sign: >0 positive, <0 negative."""
    arr = pd.to_numeric(hist, errors="coerce").to_numpy(dtype=float)
    n = len(arr)
    wave_id = [-1] * n
    waves: List[Wave] = []

    def sgn(x: float) -> int:
        if x > 0.0:
            return 1
        if x < 0.0:
            return -1
        return 0

    first: Optional[int] = None
    cur_sign = 0
    for i in range(n):
        if np.isnan(arr[i]):
            continue
        si = sgn(float(arr[i]))
        if si != 0:
            first = i
            cur_sign = si
            break

    if first is None:
        return [], wave_id

    cur_start = first
    cur_wave_idx = 0
    prev_sign = cur_sign

    for i in range(first, n):
        if np.isnan(arr[i]):
            wave_id[i] = cur_wave_idx
            continue
        si = sgn(float(arr[i]))
        if si == 0:
            wave_id[i] = cur_wave_idx
            continue
        if si != prev_sign:
            waves.append(Wave(idx=cur_wave_idx, sign=prev_sign, start_i=cur_start, end_i=i - 1))
            cur_wave_idx += 1
            cur_start = i
            prev_sign = si
        wave_id[i] = cur_wave_idx

    waves.append(Wave(idx=cur_wave_idx, sign=prev_sign, start_i=cur_start, end_i=n - 1))
    return waves, wave_id


def find_entry_candle_index(mts: np.ndarray, entry_ms: int) -> int:
    """Return 5m candle index whose start <= entry_ms < next start."""
    return int(np.searchsorted(mts, int(entry_ms), side="right") - 1)


def rounded_or_nan(x: Any, ndigits: int = 6) -> Any:
    if x is None or pd.isna(x):
        return np.nan
    return round(float(x), ndigits)


def sign_to_type(sign: int) -> str:
    if int(sign) > 0:
        return "positive"
    if int(sign) < 0:
        return "negative"
    return "neutral"


def build_macd_1m_wave_markers_for_entry(
    entry: pd.Series,
    df_1m_with_warmup: pd.DataFrame,
    range_start_ms: int,
    range_end_exclusive_ms: int,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """
    Build 1m MACD halfwave markers that intersect [range_start_ms, range_end_exclusive_ms).

    MACD is calculated on df_1m_with_warmup, i.e. including candles before range_start_ms.
    Returned wave markers keep both real wave start and clipped start inside the research range.
    The minute map assigns macd_1m_wave_no/sign/type to each 1m candle start inside the clipped range.
    """
    wave_rows: List[Dict[str, Any]] = []
    minute_marker_map: Dict[int, Dict[str, Any]] = {}

    range_start_ms = int(range_start_ms)
    range_end_exclusive_ms = int(range_end_exclusive_ms)
    if range_end_exclusive_ms <= range_start_ms:
        return wave_rows, minute_marker_map
    if df_1m_with_warmup is None or df_1m_with_warmup.empty:
        return wave_rows, minute_marker_map

    df_src = df_1m_with_warmup.drop_duplicates(subset=["mts"]).sort_values("mts").reset_index(drop=True).copy()
    df_src = df_src[df_src["mts"] < range_end_exclusive_ms].reset_index(drop=True)
    if df_src.empty:
        return wave_rows, minute_marker_map

    ind_1m = calculate_macd(
        df_src,
        fast=int(args.macd_fast),
        slow=int(args.macd_slow),
        signal=int(args.macd_signal),
    )
    waves_1m, _ = build_waves(ind_1m["hist"])
    if not waves_1m:
        return wave_rows, minute_marker_map

    mts_1m = ind_1m["mts"].to_numpy(dtype=np.int64)
    entry_id = int(entry["entry_id"])
    source_row = int(entry["source_row"])
    entry_datetime = entry["entry_datetime"]
    entry_price = float(entry["entry_price"])
    entry_ms = int(entry["entry_ms_utc"])

    wave_no = 0
    for w in waves_1m:
        real_start_ms = int(mts_1m[w.start_i])
        real_end_exclusive_ms = int(mts_1m[w.end_i]) + TF_MS_1M

        # Keep only waves crossing the research interval.
        if real_end_exclusive_ms <= range_start_ms:
            continue
        if real_start_ms >= range_end_exclusive_ms:
            break

        range_wave_start_ms = max(real_start_ms, range_start_ms)
        wave_end_ms = min(real_end_exclusive_ms, range_end_exclusive_ms)
        if wave_end_ms <= range_wave_start_ms:
            continue

        wave_no += 1
        wave_type = sign_to_type(int(w.sign))
        first_hist = ind_1m.at[w.start_i, "hist"]
        last_hist = ind_1m.at[w.end_i, "hist"]

        row = {
            "entry_id": entry_id,
            "source_row": source_row,
            "entry_datetime": entry_datetime,
            "entry_price": entry_price,
            "entry_ms_utc": entry_ms,
            "macd_1m_wave_no": wave_no,
            "macd_1m_wave_sign": int(w.sign),
            "macd_1m_wave_type": wave_type,
            "real_wave_start_time": fmt_local(real_start_ms, args.tz_offset, args.tz_name),
            "range_wave_start_time": fmt_local(range_wave_start_ms, args.tz_offset, args.tz_name),
            "wave_end_time": fmt_local(wave_end_ms, args.tz_offset, args.tz_name),
            "real_wave_start_ms_utc": real_start_ms,
            "range_wave_start_ms_utc": range_wave_start_ms,
            "wave_end_ms_utc": wave_end_ms,
            "first_candle_mts_utc": int(mts_1m[w.start_i]),
            "last_candle_mts_utc": int(mts_1m[w.end_i]),
            "first_hist": rounded_or_nan(first_hist, 12),
            "last_hist": rounded_or_nan(last_hist, 12),
        }
        wave_rows.append(row)

        cur = align_ms_down(range_wave_start_ms, TF_MS_1M)
        while cur < wave_end_ms:
            minute_marker_map[int(cur)] = {
                "macd_1m_wave_no": wave_no,
                "macd_1m_wave_sign": int(w.sign),
                "macd_1m_wave_type": wave_type,
            }
            cur += TF_MS_1M

    return wave_rows, minute_marker_map


ADX_COLUMNS = [
    "entry_id",
    "source_row",
    "entry_datetime",
    "entry_price",
    "entry_ms_utc",
    "wave_no",
    "wave_start_time",
    "next_wave_start_time",
    "candle_start_time",
    "candle_end_time",
    "candle_mts_utc",
    "adx_14_4",
    "plus_di_14",
    "minus_di_14",
    "open",
    "high",
    "low",
    "close",
]


MINUTE_COLUMNS = [
    "entry_id",
    "source_row",
    "entry_datetime",
    "entry_price",
    "entry_ms_utc",
    "minute_start_time",
    "minute_mts_utc",
    "macd_1m_wave_no",
    "macd_1m_wave_sign",
    "macd_1m_wave_type",
    "open",
    "high",
    "low",
    "close",
]


MACD_1M_WAVE_COLUMNS = [
    "entry_id",
    "source_row",
    "entry_datetime",
    "entry_price",
    "entry_ms_utc",
    "macd_1m_wave_no",
    "macd_1m_wave_sign",
    "macd_1m_wave_type",
    "real_wave_start_time",
    "range_wave_start_time",
    "wave_end_time",
    "real_wave_start_ms_utc",
    "range_wave_start_ms_utc",
    "wave_end_ms_utc",
    "first_candle_mts_utc",
    "last_candle_mts_utc",
    "first_hist",
    "last_hist",
]


def make_summary_row(
    entry: pd.Series,
    wave_start_ms: List[Optional[int]],
    entry_candle_start_ms: Optional[int],
    minute_start_ms: Optional[int],
    minute_end_exclusive_ms: Optional[int],
    status: str,
    args: argparse.Namespace,
    pre_entry_red_context_target: Optional[int] = None,
    pre_entry_red_context_found: Optional[int] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "entry_id": int(entry["entry_id"]),
        "source_row": int(entry["source_row"]),
        "entry_datetime": entry["entry_datetime"],
        "entry_price": float(entry["entry_price"]),
        "entry_ms_utc": int(entry["entry_ms_utc"]),
        "entry_candle_start_time": fmt_local(entry_candle_start_ms, args.tz_offset, args.tz_name),
        "wave1_start_time": fmt_local(wave_start_ms[0] if len(wave_start_ms) > 0 else None, args.tz_offset, args.tz_name),
        "wave2_start_time": fmt_local(wave_start_ms[1] if len(wave_start_ms) > 1 else None, args.tz_offset, args.tz_name),
        "wave3_start_time": fmt_local(wave_start_ms[2] if len(wave_start_ms) > 2 else None, args.tz_offset, args.tz_name),
        "wave4_start_time": fmt_local(wave_start_ms[3] if len(wave_start_ms) > 3 else None, args.tz_offset, args.tz_name),
        "wave5_start_time": fmt_local(wave_start_ms[4] if len(wave_start_ms) > 4 else None, args.tz_offset, args.tz_name),
        "wave6_start_time": fmt_local(wave_start_ms[5] if len(wave_start_ms) > 5 else None, args.tz_offset, args.tz_name),
        "wave6_first_candle_end_time": fmt_local(
            (wave_start_ms[5] + TF_MS_5M) if len(wave_start_ms) > 5 and wave_start_ms[5] is not None else None,
            args.tz_offset,
            args.tz_name,
        ),
        "minute_range_start_time": fmt_local(minute_start_ms, args.tz_offset, args.tz_name),
        "minute_range_end_exclusive_time": fmt_local(minute_end_exclusive_ms, args.tz_offset, args.tz_name),
        "pre_entry_red_context_target": pre_entry_red_context_target if pre_entry_red_context_target is not None else np.nan,
        "pre_entry_red_context_found": pre_entry_red_context_found if pre_entry_red_context_found is not None else np.nan,
        "status": status,
    }
    return row

'''


# ================= Main-anchor entry samples and original exit engine =================
# Appended integration: existing indicator formulas above are unchanged.

import argparse as _argparse
import hashlib as _hashlib
import json as _json
import sys as _sys
import types as _types
from dataclasses import field as _field

TRAINING_CASES = 2000
TRAINING_OUTPUT_FILE = Path(__file__).with_name("bitfinex_training_dataset_v2.1.xlsx")
ANCHOR_GROUP_SOURCE = "Основные и допополнительные якоря 15.10.2026.xlsx"
MAIN_ANCHOR_IDS = (1, 3, 5, 7, 8, 9, 13, 14, 15, 16, 19, 22, 23, 24, 25, 26, 28, 33)
ADDITIONAL_ANCHOR_GROUPS = {
    "additional_movement_start": ("Начало движения", (11, 30, 32)),
    "additional_movement_strength": ("Усиление движения", (12, 17, 18, 20)),
    "additional_trend_change": ("Возможное изменение тренда", (10, 31)),
    "additional_reversals": ("Потенциальные развороты", (2, 4, 6, 29)),
    "additional_trend_end": ("Конец тренда", (21,)),
    "additional_volume": ("Обьем", (27, 34, 35)),
}


@dataclass
class ExitSettings:
    stop_x: float = 0.3
    support_threshold: float = 0.0003
    reentry_offset_down: float = 0.0
    adx_threshold: float = 25.0
    tz_name: str = ""
    tz_offset: int = 3

    def namespace(self):
        if not np.isfinite(self.stop_x) or self.stop_x < 0:
            raise ValueError("stop_x must be finite and nonnegative")
        if not np.isfinite(self.support_threshold) or self.support_threshold < 0:
            raise ValueError("support_threshold must be finite and nonnegative")
        if not np.isfinite(self.reentry_offset_down) or self.reentry_offset_down < 0:
            raise ValueError("reentry_offset_down must be finite and nonnegative")
        return _argparse.Namespace(**vars(self), macd_fast=12, macd_slow=26,
                                  macd_signal=9, di_length=14, adx_smoothing=4,
                                  max_entries=0, entry_id="", verbose=False)


@dataclass
class TrainingDatasets:
    """Entry features, model inputs, future outcomes, and full trade audit separately.

    features: closed-candle OHLCV + normalized indicators + main-anchor flags.
    model_features: numeric inputs only; never includes outcomes or future times.
    targets: numeric gross-return target and availability/status information.
    table: features joined to targets, intended for the human-readable export.
    group_events: seven separate event sets (main plus six additional categories).
    """
    entries: pd.DataFrame
    features: pd.DataFrame
    model_features: pd.DataFrame
    targets: pd.DataFrame
    table: pd.DataFrame
    groups: pd.DataFrame
    group_events: dict
    trades: pd.DataFrame
    stop_events: pd.DataFrame
    support_searches: pd.DataFrame
    entry_orders: pd.DataFrame
    entry_summary: pd.DataFrame
    settings: dict
    comparison: pd.DataFrame = _field(default_factory=pd.DataFrame)


def _verify_embedded_sources():
    """Reject accidental source edits before execution, also under python -O."""
    for label, source, expected in (
        ("exit", ORIGINAL_EXIT_SOURCE, EMBEDDED_EXIT_SHA256),
        ("wave", ORIGINAL_WAVE_SOURCE, EMBEDDED_WAVE_SHA256),
    ):
        actual = _hashlib.sha256(source.encode("utf8")).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Embedded {label} source SHA-256 mismatch: {actual}")


_verify_embedded_sources()


def _embedded_module(name, source):
    _verify_embedded_sources()
    if name not in _sys.modules:
        module = _types.ModuleType(name)
        module.__file__ = str(Path(__file__).resolve()) + "::" + name
        _sys.modules[name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    return _sys.modules[name]


def original_exit_engine():
    """Original v9 exit functions, isolated from the market-feature formulas."""
    return _embedded_module("_bitfinex_original_exit_v9", ORIGINAL_EXIT_SOURCE)


def original_wave_engine():
    """Required v5 MACD wave and ADX helpers, retaining original calculations."""
    return _embedded_module("_bitfinex_original_waves_v5", ORIGINAL_WAVE_SOURCE)


def anchor_group_definitions():
    definitions = {"main": ("Все основные якоря", MAIN_ANCHOR_IDS), **ADDITIONAL_ANCHOR_GROUPS}
    flat = [n for _, ids in definitions.values() for n in ids]
    if sorted(flat) != list(ANCHOR_RULES) or len(flat) != len(set(flat)):
        raise ValueError("Every current anchor must belong to exactly one main/additional group")
    return definitions


def validate_anchor_group_workbook(path):
    """Read membership only. Spreadsheet text is data, never executable instructions."""
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        actual_main = []
        additional = []
        for row in list(book.worksheets[0].values)[1:]:
            role, values = row[0], row[3]
            if role is None or values is None:
                continue
            ids = parse_anchor_membership_cell(values)
            if str(role).strip().startswith("Главные"):
                actual_main.extend(ids)
            elif str(role).strip().startswith("Дополнительные"):
                additional.append(ids)
        if tuple(sorted(actual_main)) != MAIN_ANCHOR_IDS:
            raise ValueError("Main-anchor membership differs from the confirmed configuration")
        expected = [ids for _, ids in ADDITIONAL_ANCHOR_GROUPS.values()]
        if additional != expected:
            raise ValueError("Additional-anchor categories differ from the confirmed configuration")
    finally:
        book.close()


def parse_anchor_membership_cell(value):
    """Parse text ID lists; reject ambiguous numeric decimal lists.

    Excel has already discarded a numeric cell's trailing zeros by the time
    openpyxl reads it. Only the two explicitly confirmed numeric lists are
    accepted. Other lists must be stored as text, e.g. '10,30' or '10.30'.
    """
    if isinstance(value, bool):
        raise ValueError("Anchor IDs cannot be boolean")
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Anchor membership must be finite")
        confirmed = {10.31: (10, 31), 25.26: (25, 26)}
        if number.is_integer():
            ids = (int(number),)
        elif number in confirmed:
            ids = confirmed[number]
        else:
            raise ValueError(f"Ambiguous numeric anchor list {value!r}; store IDs as text separated by commas")
    else:
        tokens = str(value).replace(".", ",").replace(";", ",").split(",")
        ids = tuple(int(token.strip()) for token in tokens if token.strip())
    if not ids or len(ids) != len(set(ids)) or any(n not in ANCHOR_RULES for n in ids):
        raise ValueError(f"Invalid or duplicate anchor IDs: {ids}")
    return ids


def _anchor_workbook_metadata(path):
    if path is None:
        return {"anchor_membership_validated_against_workbook": False,
                "anchor_membership_mode": "embedded_confirmed_configuration"}
    path = Path(path).resolve()
    return {"anchor_membership_validated_against_workbook": True,
            "anchor_membership_mode": "explicit_workbook_validated",
            "anchor_membership_workbook_path": str(path),
            "anchor_membership_workbook_sha256": _hashlib.sha256(path.read_bytes()).hexdigest()}


def closed_indicator_history(candles, cutoff):
    """Advance closed indicator slots to a known decision clock, without future data.

    An API snapshot may end with empty minutes. Those slots must advance the
    indicators just as internal no-trade gaps do. Only slots already closed at
    the explicitly supplied cutoff are padded, never a forming candle.
    """
    cutoff = pd.Timestamp(cutoff)
    cutoff = cutoff.tz_localize(DISPLAY_TIMEZONE) if cutoff.tzinfo is None else cutoff
    cutoff = cutoff.tz_convert("UTC")
    history = candles.loc[candles.available_at.le(cutoff)].copy()
    if history.empty:
        raise ValueError(f"No closed indicator history before {cutoff}")
    result = fill_candle_gaps(history)
    duration = result.available_at.iloc[0] - result.timestamp.iloc[0]
    last_closed_open = cutoff.floor(duration) - duration
    last_open = result.timestamp.iloc[-1]
    if last_open < last_closed_open:
        times = pd.date_range(last_open + duration, last_closed_open, freq=duration)
        price = float(result.close.iloc[-1])
        tail = pd.DataFrame({"timestamp": times,
                             "timestamp_ms": times.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000,
                             "open": price, "high": price, "low": price, "close": price,
                             "volume": 0.0, "available_at": times + duration})
        result = pd.concat([result, tail], ignore_index=True)
    return result


def validate_training_history(native, entries):
    """Require configured closed-candle warm-up at the first entry on every TF."""
    first = pd.Timestamp(entries.entry_at.min())
    counts = {}
    for tf in TIMEFRAMES:
        frame = closed_indicator_history(native[tf], first)
        count = int(frame.available_at.le(first).sum())
        counts[tf] = count
        if count < warmup_candles():
            raise ValueError(f"Insufficient {tf} indicator warm-up before {first}: "
                             f"{count} closed slots; need at least {warmup_candles()}. Fetch more history.")
    return counts


def _api_records_frame(records, timeframe):
    frame = pd.DataFrame(records, columns=["timestamp_ms", "open", "close", "high", "low", "volume"])
    frame = frame.drop_duplicates("timestamp_ms", keep="last").sort_values("timestamp_ms").reset_index(drop=True)
    frame["timestamp"] = pd.to_datetime(frame.timestamp_ms, unit="ms", utc=True)
    frame["available_at"] = frame.timestamp + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    return frame[CANDLE_COLUMNS + ["available_at"]]


def load_training_candles(cache_dir=None, cutoff=None, history_m1_candles=45000):
    """Use a reproducible native-candle cache, or collect a new public API snapshot.

    Cache files: 1m.json, 5m.json, 15m.json, 30m.json in Bitfinex array order.
    No synthetic candles are passed to the legacy exit engine. Existing feature
    calculations still fill gaps internally, as before.
    """
    if cache_dir is not None:
        root = Path(cache_dir)
        native = {tf: _api_records_frame(_json.loads((root / f"{tf}.json").read_text(encoding="utf8")), tf)
                  for tf in TIMEFRAMES}
        limit = pd.Timestamp(cutoff) if cutoff is not None else native["1m"].available_at.iloc[-1]
    else:
        limit = pd.Timestamp.now(tz="UTC").floor("min") if cutoff is None else pd.Timestamp(cutoff)
        limit = limit.tz_localize(DISPLAY_TIMEZONE) if limit.tzinfo is None else limit
        limit = limit.tz_convert("UTC")
        native = {}
        with requests.Session() as session:
            native["1m"] = fetch_candles(session, TICKER, "1m", history_m1_candles, limit)
            first_entry = native["1m"].timestamp.iloc[0] + pd.Timedelta(days=3)
            for tf in TIMEFRAMES[1:]:
                duration = pd.Timedelta(minutes=TIMEFRAME_MINUTES[tf])
                # Three days of original-exit warm-up before the first entry.
                start = min(first_entry - pd.Timedelta(days=3), first_entry - duration * warmup_candles())
                count = int((limit - start) // duration) + warmup_candles()
                native[tf] = fetch_candles(session, TICKER, tf, count, limit)
    limit = limit.tz_localize(DISPLAY_TIMEZONE) if limit.tzinfo is None else limit
    limit = limit.tz_convert("UTC")
    for tf in TIMEFRAMES:
        native[tf] = native[tf].loc[native[tf].available_at <= limit].reset_index(drop=True)
        if native[tf].empty:
            raise ValueError(f"No closed native {tf} candles before cutoff")
    return native, limit


def filter_main_anchor_entries(native, case_limit=TRAINING_CASES, start=None, end=None):
    """Only main anchors create rows; simultaneous main triggers share one row.

    All 35 flags remain available on the complete 5m event timeline. Additional
    anchors never create entries by themselves. Missing M1 opening candles use
    the last observed close as the entry reference, explicitly flagged synthetic.
    """
    features_5m = calculate_native(native["5m"])
    all_flags = detect_anchor_events(features_5m)
    first = native["1m"].timestamp.iloc[0] + pd.Timedelta(days=3)
    last = native["1m"].available_at.iloc[-1]
    if start is not None:
        value = pd.Timestamp(start)
        first = max(first, value.tz_localize(DISPLAY_TIMEZONE).tz_convert("UTC") if value.tzinfo is None else value.tz_convert("UTC"))
    if end is not None:
        value = pd.Timestamp(end)
        last = min(last, value.tz_localize(DISPLAY_TIMEZONE).tz_convert("UTC") if value.tzinfo is None else value.tz_convert("UTC"))
    all_flags = all_flags.loc[(all_flags.index >= first) & (all_flags.index < last)].copy()
    main_columns = [f"anchor_{n:02d}" for n in MAIN_ANCHOR_IDS]
    selected = all_flags.loc[all_flags[main_columns].any(axis=1)].copy()
    if case_limit:
        if len(selected) < case_limit:
            raise ValueError(f"Need {case_limit} main-anchor cases; only {len(selected)} in candle snapshot")
        selected = selected.iloc[:case_limit].copy()
    if selected.empty:
        raise ValueError("No main-anchor entries in the selected range")
    entries = pd.DataFrame({"entry_at": selected.index})
    entries.insert(0, "entry_id", np.arange(1, len(entries) + 1))
    entries["source_row"] = entries.entry_id
    entries["entry_ms_utc"] = entries.entry_at.astype("datetime64[ns, UTC]").astype("int64") // 1_000_000
    entries["entry_datetime"] = entries.entry_at.dt.tz_convert(DISPLAY_TIMEZONE).dt.strftime("%d.%m.%Y %H:%M")
    opens = native["1m"].set_index("timestamp")["open"]
    entries["entry_price"] = entries.entry_at.map(opens)
    entries["entry_price_is_synthetic"] = entries.entry_price.isna().astype(np.uint8)
    previous = pd.merge_asof(entries[["entry_at"]].sort_values("entry_at"),
                             native["1m"][["available_at", "close"]].sort_values("available_at"),
                             left_on="entry_at", right_on="available_at", direction="backward")
    entries["entry_price"] = entries.entry_price.fillna(pd.Series(previous.close.to_numpy(), index=entries.index))
    if entries.entry_price.isna().any():
        raise ValueError("Cannot establish an entry price from available market data")
    for column in selected.columns:
        entries[column] = selected[column].to_numpy(dtype=np.uint8)
    entries["main_anchor_numbers"] = [";".join(str(n) for n in MAIN_ANCHOR_IDS if row[f"anchor_{n:02d}"])
                                      for _, row in entries.iterrows()]
    entries = entries.set_index("entry_id", drop=False)
    group_events = {}
    final_time = selected.index[-1]
    full_range = all_flags.loc[all_flags.index <= final_time]
    for key, (_, ids) in anchor_group_definitions().items():
        columns = [f"anchor_{n:02d}" for n in ids]
        group_events[key] = full_range.loc[full_range[columns].any(axis=1), columns].copy()
    return entries, group_events


def features_at_entry(entries, native):
    """All market features are known at entry: never use the entry candle close."""
    validate_training_history(native, entries)
    histories = {tf: closed_indicator_history(native[tf], entries.entry_at.max()) for tf in TIMEFRAMES}
    minute = histories["1m"]
    indices = np.searchsorted(minute.available_at.astype("datetime64[ns, UTC]").astype("int64").to_numpy(), entries.entry_at.astype("datetime64[ns, UTC]").astype("int64").to_numpy(), side="right") - 1
    if (indices < 0).any():
        raise ValueError("Entry precedes available feature history")
    raw = minute.iloc[indices][["timestamp", "timestamp_ms", "available_at", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
    raw["entry_at"] = entries.entry_at.to_numpy()
    for tf in TIMEFRAMES:
        raw = align_timeframe(raw, calculate_native(histories[tf]), tf)
    if (raw.available_at.to_numpy() > entries.entry_at.to_numpy()).any():
        raise AssertionError("Feature availability exceeds the entry decision time")
    normalized = normalize(raw)
    normalized.index = entries.index.copy()
    features = pd.DataFrame(index=entries.index)
    features["entry_at"] = entries.entry_at
    features["feature_candle_1m"] = raw.timestamp.to_numpy()
    features["feature_available_at"] = raw.available_at.to_numpy()
    features["entry_price"] = entries.entry_price
    features["entry_price_is_synthetic"] = entries.entry_price_is_synthetic
    ohlcv = ["open", "high", "low", "close", "volume"]
    for column in [*ohlcv, *[f"{c}_{tf}" for tf in TIMEFRAMES[1:] for c in ohlcv],
                   *[f"source_candle_{tf}" for tf in TIMEFRAMES]]:
        features[column] = raw[column].to_numpy()
    # Explicit norm_ prefix distinguishes normalized values from raw OHLCV.
    for column in normalized.columns:
        features[f"norm_{column}"] = normalized[column]
    features["main_anchor_numbers"] = entries.main_anchor_numbers
    for n in MAIN_ANCHOR_IDS:
        column = f"anchor_{n:02d}"
        features[column] = entries[column].astype(np.uint8)
    model_columns = [c for c in features if c.startswith("norm_") or c.startswith("anchor_")]
    model = features[model_columns].copy()
    if not np.isfinite(model.to_numpy(dtype=float)).all():
        raise ValueError("Model features contain missing or infinite values")
    return features, model


def _collector_frame(frame):
    return frame[["timestamp_ms", "open", "close", "high", "low", "volume"]].rename(columns={"timestamp_ms": "mts"}).reset_index(drop=True)


class LegacyExitInputBuilder:
    """Prepare the four original input tables without changing the v9 simulator.

    Original exit rules and histogram-sign wave helpers are retained. Indicator
    history includes empty time slots, and M5 DMI uses the shared decimal-safe
    implementation. Stops, support clusters and fills use actual API candles.
    The M5 warm-up begins three days before the first selected entry. M1 warm-up
    begins three days before each entry, matching the original collector.
    """
    def __init__(self, entries, native, settings=None):
        self.entries = entries
        self.settings = settings or ExitSettings()
        self.args = self.settings.namespace()
        self.wave_module = original_wave_engine()
        self.minute = _collector_frame(native["1m"])
        self.indicator_minute = _collector_frame(fill_candle_gaps(native["1m"]))
        left = int(entries.entry_ms_utc.iloc[0]) - 3 * 86400000
        five_native = native["5m"].loc[native["5m"].timestamp_ms >= left].reset_index(drop=True)
        five_closed = int(fill_candle_gaps(five_native).available_at.le(entries.entry_at.min()).sum())
        if five_closed < warmup_candles():
            raise ValueError(f"Exit M5 warm-up has {five_closed} closed slots; need {warmup_candles()}")
        five_features = calculate_native(five_native)
        self.five = _collector_frame(five_features)
        self.five["macd"] = five_features["macd_12_26"].to_numpy()
        self.five["signal"] = five_features["macd_signal_12_26_9"].to_numpy()
        self.five["hist"] = self.five["macd"] - self.five["signal"]
        self.five["adx_14_4"] = five_features["adx_14_4"].to_numpy()
        self.five["plus_di_14"] = five_features["di_plus_14"].to_numpy()
        self.five["minus_di_14"] = five_features["di_minus_14"].to_numpy()
        self.waves, self.wave_ids = self.wave_module.build_waves(self.five["hist"])
        self.mts = self.five.mts.to_numpy(dtype=np.int64)

    def for_entry(self, entry):
        wave = self.wave_module
        args = self.args
        entry_ms = int(entry.entry_ms_utc)
        index = wave.find_entry_candle_index(self.mts, entry_ms)
        wid = self.wave_ids[index] if 0 <= index < len(self.wave_ids) else -1
        if wid < 0 or wid + 5 >= len(self.waves):
            summary = {"entry_id": int(entry.entry_id), "entry_datetime": entry.entry_datetime,
                       "entry_ms_utc": entry_ms, "entry_price": float(entry.entry_price), "wave6_start_time": ""}
            return (pd.DataFrame([summary]), pd.DataFrame(columns=wave.ADX_COLUMNS),
                    pd.DataFrame(columns=wave.MINUTE_COLUMNS), pd.DataFrame(columns=wave.MACD_1M_WAVE_COLUMNS))
        starts = [int(self.mts[self.waves[wid + k].start_i]) for k in range(6)]
        end = starts[5]
        actual_work = self.minute.loc[(self.minute.mts >= entry_ms - 3 * 86400000) & (self.minute.mts < end)].reset_index(drop=True)
        context_start, reds = wave.find_pre_entry_red_context_start_ms(actual_work, entry_ms)
        work = self.indicator_minute.loc[(self.indicator_minute.mts >= entry_ms - 3 * 86400000)
                                         & (self.indicator_minute.mts < end)].reset_index(drop=True)
        if int(work.mts.lt(entry_ms).sum()) < warmup_candles():
            raise ValueError(f"Insufficient exit M1 MACD warm-up for entry {entry.entry_id}")
        summary = wave.make_summary_row(entry, starts, int(self.mts[index]), context_start,
                                        end + 300000, "ok", args, 20, reds)
        adx_rows = []
        for wave_no in (2, 4):
            current = self.waves[wid + wave_no - 1]
            for ci in range(current.start_i, current.end_i + 1):
                value = self.five.at[ci, "adx_14_4"]
                if pd.isna(value) or float(value) <= self.settings.adx_threshold:
                    continue
                row = self.five.iloc[ci]
                adx_rows.append({"entry_id": int(entry.entry_id), "candle_mts_utc": int(row.mts),
                                 "wave_no": wave_no, "adx_14_4": round(float(value), 6),
                                 "plus_di_14": round(float(row.plus_di_14), 6), "minus_di_14": round(float(row.minus_di_14), 6)})
        wave_rows, markers = wave.build_macd_1m_wave_markers_for_entry(entry, work, context_start, end, args)
        exported = self.minute.loc[(self.minute.mts >= context_start) & (self.minute.mts < end + 300000)].copy()
        exported.insert(0, "entry_id", int(entry.entry_id))
        exported = exported.rename(columns={"mts": "minute_mts_utc"})
        exported["macd_1m_wave_no"] = exported.minute_mts_utc.map(lambda t: markers.get(int(t), {}).get("macd_1m_wave_no", np.nan))
        exported["macd_1m_wave_sign"] = exported.minute_mts_utc.map(lambda t: markers.get(int(t), {}).get("macd_1m_wave_sign", np.nan))
        return (pd.DataFrame([summary]),
                pd.DataFrame(adx_rows, columns=["entry_id", "candle_mts_utc", "wave_no", "adx_14_4", "plus_di_14", "minus_di_14"]),
                exported, pd.DataFrame(wave_rows, columns=wave.MACD_1M_WAVE_COLUMNS))


def calculate_exit_targets(entries, native, settings=None, progress=True):
    settings = settings or ExitSettings()
    builder = LegacyExitInputBuilder(entries, native, settings)
    engine = original_exit_engine()
    output = [[], [], [], [], []]
    for count, (_, entry) in enumerate(entries.iterrows(), 1):
        tables = engine.normalize_dataset(*builder.for_entry(entry), builder.args)
        summary, adx, minutes, waves = tables
        rows = engine.process_entry(summary.iloc[0], adx, minutes, waves, builder.args)
        for container, values in zip(output[:4], rows[:4]):
            container.extend(values)
        output[4].append(rows[4])
        if progress and (count % 100 == 0 or count == len(entries)):
            print(f"Exit targets: {count}/{len(entries)} entries", flush=True)
    frames = tuple(pd.DataFrame(rows) for rows in output)
    trades, _, _, _, summaries = frames
    target_rows = []
    for _, summary in summaries.iterrows():
        entry_id = int(summary.entry_id)
        rows = trades.loc[trades.entry_id.eq(entry_id)] if not trades.empty else pd.DataFrame()
        unfinished = not rows.empty and rows.trade_status.ne("closed").any()
        valid_end = summary.final_reason in ("adx_gt_25", "wave6_reached")
        available = bool(valid_end and not unfinished and not rows.empty)
        gross = float(pd.to_numeric(rows.short_result_pct, errors="coerce").sum()) if not rows.empty else np.nan
        status = "complete" if available else "open_position_at_wave_cutoff" if unfinished else str(summary.final_reason)
        if str(summary.last_state) == "pending_order_unfilled_at_study_end" and available:
            status = "complete_pending_order_unfilled"
        target_rows.append({"entry_id": entry_id, "target_gross_return_pct": round(gross, 6) if available else np.nan,
                            "target_available": int(available), "target_status": status,
                            "realized_gross_return_pct": round(gross, 6) if np.isfinite(gross) else np.nan,
                            "series_end_reason": summary.final_reason, "series_last_state": summary.last_state,
                            "trades_count": int(summary.trades_count), "closed_trades_count": int(summary.closed_trades_count),
                            "reentries_count": int(summary.reentries_executed_count),
                            "series_cutoff_at": pd.to_datetime(summary.wave6_start_ms_utc, unit="ms", utc=True) if pd.notna(summary.wave6_start_ms_utc) else pd.NaT,
                            "adx_termination_at": pd.to_datetime(summary.adx_stop_ms_utc, unit="ms", utc=True) if pd.notna(summary.adx_stop_ms_utc) else pd.NaT})
    targets = pd.DataFrame(target_rows).set_index("entry_id")
    return targets, frames


def build_training_dataset(native, case_limit=TRAINING_CASES, start=None, end=None, settings=None):
    settings = settings or ExitSettings()
    entries, group_events = filter_main_anchor_entries(native, case_limit, start, end)
    features, model = features_at_entry(entries, native)
    targets, output = calculate_exit_targets(entries, native, settings)
    trades, stops, support, orders, summaries = output
    groups = pd.DataFrame([{"group": key, "category": category, "anchor_numbers": ";".join(map(str, ids)),
                            "creates_entries": int(key == "main"), "anchor_count": len(ids)}
                           for key, (category, ids) in anchor_group_definitions().items()])
    metadata = {**vars(settings), "anchor_source": ANCHOR_GROUP_SOURCE, "anchor_schema": ANCHOR_SCHEMA_VERSION,
                "cases": len(entries), "main_anchor_count": len(MAIN_ANCHOR_IDS), "groups": len(groups),
                "entry_first_utc3": entries.entry_at.iloc[0].tz_convert(DISPLAY_TIMEZONE).isoformat(),
                "entry_last_utc3": entries.entry_at.iloc[-1].tz_convert(DISPLAY_TIMEZONE).isoformat(),
                "exit_source_sha256": ORIGINAL_EXIT_SHA256, "wave_source_sha256": ORIGINAL_WAVE_SHA256,
                "embedded_exit_sha256": EMBEDDED_EXIT_SHA256, "embedded_wave_sha256": EMBEDDED_WAVE_SHA256,
                "embedded_source_hashes_verified": True,
                **_anchor_workbook_metadata(None),
                "minimum_closed_warmup_slots": warmup_candles(),
                "closed_warmup_slots_before_first_entry": validate_training_history(native, entries),
                "feature_decision_clock": "Closed slots advance through entry_at, including trailing no-trade gaps; future candles excluded",
                "features": "OHLCV and normalized indicators from candles closed by entry_at; main flags from the closed M5 anchor candle",
                "target": "Sum of original rounded short_result_pct across all closed trades of this entry; unlevered, no fees or slippage. Unfinished series: target unavailable.",
                "wave_cutoff": "Original wave6_start boundary; open positions are snapshots, never silently force-closed",
                "model_feature_columns": list(model.columns),
                "original_logic": "M5 ADX hits use numbered waves 2/4 exactly as in v5, regardless of histogram sign at entry",
                "exit_indicator_inputs": "chart_aligned: empty slots advance MACD; M5 ADX shares decimal-safe DMI with model features",
                "gap_policy": "Model and exit indicators fill empty native slots; stop/support/fill processing uses actual API candles only"}
    return TrainingDatasets(entries, features, model, targets, features.join(targets), groups,
                            group_events, trades, stops, support, orders, summaries, metadata)


def collect_training_datasets(case_limit=TRAINING_CASES, cache_dir=None, cutoff=None,
                              start=None, end=None, settings=None, anchor_workbook=None):
    """Public entry point: collect once and return independent input/target tables."""
    if anchor_workbook is not None:
        validate_anchor_group_workbook(anchor_workbook)
    native, snapshot_cutoff = load_training_candles(cache_dir, cutoff)
    datasets = build_training_dataset(native, case_limit, start, end, settings)
    datasets.settings.update(_anchor_workbook_metadata(anchor_workbook))
    datasets.settings["snapshot_cutoff_utc3"] = snapshot_cutoff.tz_convert(DISPLAY_TIMEZONE).isoformat()
    return datasets


def export_training_dataset(datasets, output=TRAINING_OUTPUT_FILE):
    """Spreadsheet pages plus separate reusable feature/target datasets.

    Re-run the Python script to refresh this calculated snapshot. No future
    target field is part of model_features. CSV preserves full numeric values.
    """
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = {"training_data": datasets.table, "model_features": datasets.model_features,
              "targets": datasets.targets, "anchor_groups": datasets.groups,
              "trades": datasets.trades, "entry_summary": datasets.entry_summary}
    if not datasets.comparison.empty:
        frames["comparison"] = datasets.comparison
    params = pd.DataFrame([(key, _json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value)
                           for key, value in datasets.settings.items()], columns=["parameter", "value"])
    frames["parameters"] = params
    book = Workbook()
    book.remove(book.active)
    for name, frame in frames.items():
        data = frame.reset_index() if frame.index.name is not None else frame.copy()
        sheet = book.create_sheet(name)
        sheet.append(list(data.columns))
        for values in data.itertuples(index=False, name=None):
            converted = []
            for value in values:
                if pd.isna(value):
                    converted.append(None)
                elif isinstance(value, pd.Timestamp):
                    converted.append(value.tz_convert(DISPLAY_TIMEZONE).tz_localize(None).to_pydatetime() if value.tzinfo else value.to_pydatetime())
                elif isinstance(value, np.generic):
                    converted.append(value.item())
                else:
                    converted.append(value)
            sheet.append(converted)
        sheet.freeze_panes = "B2"
        sheet.sheet_view.showGridLines = False
        sheet.auto_filter.ref = sheet.dimensions
        sheet.row_dimensions[1].height = 58 if name in ("training_data", "model_features", "targets") else 42
        for i, column in enumerate(data.columns, 1):
            is_date = pd.api.types.is_datetime64_any_dtype(data[column])
            width = 24 if is_date else 18 if len(str(column)) > 12 else 14
            if not is_date and not pd.api.types.is_numeric_dtype(data[column]):
                longest = max((len(str(value)) for value in data[column].dropna()), default=0)
                width = max(width, min(longest + 2, 55))
            if column in ("target_status", "series_last_state", "series_end_reason", "category", "parameter", "final_reason"):
                width = 36
            if column == "value":
                width = 105
            sheet.column_dimensions[get_column_letter(i)].width = width
            header = sheet.cell(1, i)
            header.font = Font(name="Arial", size=10, bold=True)
            header.fill = PatternFill("solid", fgColor="E3EAF0")
            header.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            integer = (pd.api.types.is_integer_dtype(data[column]) or str(column).startswith("anchor_") or str(column).endswith(("_count", "_ms_utc", "_no"))
                       or column in ("entry_id", "target_available", "entry_price_is_synthetic", "source_row"))
            for row in range(2, len(data) + 2):
                cell = sheet.cell(row, i)
                cell.font = Font(name="Arial", size=10)
                cell.alignment = Alignment(vertical="center", horizontal="right" if isinstance(cell.value, (int, float)) else "left", wrap_text=column == "value")
                if is_date:
                    cell.number_format = "yyyy-mm-dd hh:mm:ss"
                elif isinstance(cell.value, (int, float)):
                    cell.number_format = "0" if integer else "0.00000000"
                if str(column).startswith("target_"):
                    cell.fill = PatternFill("solid", fgColor="F0F0F0")
        if name == "parameters":
            for row in range(2, len(data) + 2):
                value = str(sheet.cell(row, 2).value or "")
                sheet.row_dimensions[row].height = max(32, 15 * (len(textwrap.wrap(value, width=98)) + 1))
        sheet.sheet_properties.tabColor = "315F80" if name == "training_data" else None
    book.save(output)
    prefix = output.with_suffix("")
    for label, frame in (("features", datasets.features), ("model_features", datasets.model_features),
                         ("targets", datasets.targets), ("training_data", datasets.table),
                         ("stop_events", datasets.stop_events), ("support_searches", datasets.support_searches),
                         ("entry_orders", datasets.entry_orders)):
        exported = frame.copy()
        for column in exported.columns:
            if isinstance(exported[column].dtype, pd.DatetimeTZDtype):
                exported[column] = exported[column].dt.tz_convert(DISPLAY_TIMEZONE)
        exported.to_csv(prefix.with_name(prefix.name + "_" + label + ".csv"), index=exported.index.name is not None, encoding="utf-8-sig")
    prefix.with_name(prefix.name + "_settings.json").write_text(_json.dumps(datasets.settings, indent=2, ensure_ascii=False), encoding="utf8")
    return output


def main():
    parser = _argparse.ArgumentParser(description="Bitfinex main-anchor features and original v9 exit targets")
    parser.add_argument("--indicators-only", action="store_true", help="Run the original all-candle indicator export")
    parser.add_argument("--cases", type=int, default=TRAINING_CASES)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--cutoff", default=None)
    parser.add_argument("--start", default=None, help="First entry time, UTC+3 if naive")
    parser.add_argument("--end", default=None, help="Last entry time, UTC+3 if naive")
    parser.add_argument("--history-m1-candles", type=int, default=45000)
    parser.add_argument("--stop-x", type=float, default=0.3)
    parser.add_argument("--support-threshold", type=float, default=0.0003)
    parser.add_argument("--reentry-offset-down", type=float, default=0.0)
    parser.add_argument("--anchor-workbook", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=TRAINING_OUTPUT_FILE)
    args = parser.parse_args()
    if args.indicators_only:
        return _indicator_only_main()
    if args.cases < 0:
        parser.error("--cases must be nonnegative; 0 means all matching entries")
    if args.anchor_workbook:
        validate_anchor_group_workbook(args.anchor_workbook)
    settings = ExitSettings(args.stop_x, args.support_threshold, args.reentry_offset_down)
    native, cutoff = load_training_candles(args.cache_dir, args.cutoff, args.history_m1_candles)
    datasets = build_training_dataset(native, args.cases, args.start, args.end, settings)
    datasets.settings.update(_anchor_workbook_metadata(args.anchor_workbook))
    datasets.settings["snapshot_cutoff_utc3"] = cutoff.tz_convert(DISPLAY_TIMEZONE).isoformat()
    path = export_training_dataset(datasets, args.output)
    print(f"Exported {len(datasets.table)} main-anchor entries to {path}")
    print(f"Available targets: {int(datasets.targets.target_available.sum())}/{len(datasets.targets)}")


if __name__ == "__main__":
    main()
