"""Volatility measures known at an entry's decision time, from gap-filled native candles.

Every measure uses only candles closed at or before the decision time t (the M5 close the entry sits on).
Returns are log returns in percent. "Relative" measures divide by the measure's own median over the
previous 7 days, so they describe how active the market is now compared with its recent normal, which
keeps their meaning when the overall level of volatility drifts (as it did in 2026).

    rv_1h, rv_4h, rv_24h      realized volatility: std of M5 log returns over 12 / 48 / 288 candles (% per M5)
    parkinson_1h              Parkinson high-low volatility over 12 M5 candles (% per M5)
    range_1h_pct              (highest high - lowest low) / close over the last 12 M5 candles, in %
    fee_cover_1h              range_1h_pct / round-trip fee (0.10): how many round-trip fees the last hour moved
    atr_m5, atr_m15, atr_m30, atr_1h   ATR 14 / close on each timeframe, in %
    rv_1h_rel, atr_m5_rel, range_1h_rel   the measure / its median over the previous 7 days
    vol_1h_rel                M5 volume of the last hour / its median hourly volume over the previous 7 days, log
    abs_return_1h             |close now / close 12 candles ago - 1| in % (direction-free move, for contrast)
"""
import numpy as np
import pandas as pd

M5_PER_HOUR = 12
M5_PER_WEEK = 12 * 24 * 7
ROUND_TRIP_FEE_PCT = 0.10


def hourly_from_m30(m30):
    frame = m30.set_index(pd.to_datetime(m30.timestamp, utc=True)).sort_index()
    out = frame[["open", "high", "low", "close", "volume"]].resample("1h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    counts = frame.close.resample("1h", label="left", closed="left").count()
    out = out[counts == 2]
    out.index.name = "timestamp"
    out = out.reset_index()
    out["available_at"] = out.timestamp + pd.Timedelta(hours=1)
    out["timestamp_ms"] = out.timestamp.astype("int64") // 1_000_000
    return out


def _at(frame, values, times):
    """Value of the last candle closed at or before each time (NaN before the first close)."""
    closes = pd.to_datetime(frame.available_at, utc=True).to_numpy()
    idx = np.searchsorted(closes, pd.DatetimeIndex(times).to_numpy(), side="right") - 1
    out = np.asarray(values, dtype=float)[np.clip(idx, 0, None)]
    out[idx < 0] = np.nan
    return out


def _relative(series, window=M5_PER_WEEK):
    """Value / median of the previous `window` values (current excluded)."""
    reference = series.shift(1).rolling(window, min_periods=window // 2).median()
    return series / reference.replace(0, np.nan)


def volatility_features(pipeline, native, times):
    """DataFrame of volatility measures for decision `times`, using the collector's indicator code."""
    m5 = pipeline.calculate_native(native["5m"])
    close, high, low = m5.close.astype(float), m5.high.astype(float), m5.low.astype(float)
    ret = 100 * np.log(close).diff()
    rv = {h: ret.rolling(n, min_periods=n).std() for h, n in (("1h", 12), ("4h", 48), ("24h", 288))}
    parkinson = 100 * np.sqrt((np.log(high / low) ** 2).rolling(12, min_periods=12).mean() / (4 * np.log(2)))
    hour_range = 100 * (high.rolling(12).max() - low.rolling(12).min()) / close
    abs_move = 100 * (close / close.shift(12) - 1).abs()
    hour_volume = m5.volume.astype(float).rolling(12, min_periods=12).sum()
    atr = {}
    for name, frame in (("m5", m5), ("m15", pipeline.calculate_native(native["15m"])),
                        ("m30", pipeline.calculate_native(native["30m"])),
                        ("1h", pipeline.calculate_native(hourly_from_m30(native["30m"])))):
        atr[name] = (frame, 100 * frame["atr_14"].astype(float) / frame.close.astype(float))
    out = pd.DataFrame(index=pd.RangeIndex(len(times)))
    out["rv_1h"], out["rv_4h"], out["rv_24h"] = (_at(m5, rv[h], times) for h in ("1h", "4h", "24h"))
    out["parkinson_1h"] = _at(m5, parkinson, times)
    out["range_1h_pct"] = _at(m5, hour_range, times)
    out["fee_cover_1h"] = out.range_1h_pct / ROUND_TRIP_FEE_PCT
    for name, (frame, values) in atr.items():
        out[f"atr_{name}"] = _at(frame, values, times)
    out["rv_1h_rel"] = _at(m5, _relative(rv["1h"]), times)
    out["atr_m5_rel"] = _at(m5, _relative(atr["m5"][1]), times)
    out["range_1h_rel"] = _at(m5, _relative(hour_range), times)
    out["vol_1h_rel"] = np.log(_at(m5, _relative(hour_volume), times))
    out["abs_return_1h"] = _at(m5, abs_move, times)
    return out
