"""An explicit on-disk feature contract shared by training and inference."""

import json
from pathlib import Path

HISTORY = 1409
STATES = 1410
WIDTH = 54
INPUT_SIZE = STATES * WIDTH
LOSS_LABELS = {"cross_entropy": "Cross-entropy", "mse": "MSE (probabilities)"}
DATASET_LABELS = {
    "bitfinex": "Bitfinex ETHUSD", "bitfinex_profit": "Bitfinex · Profit buckets",
    "uci_har": "UCI HAR · Stairs", "uci_air": "Beijing PM2.5",
}
PROFIT_DATASET = "bitfinex_profit"
PROFIT_CLASSES = ("Negative", "0–<1%", "1–<2%", "2–<3%", "3%+")
DEFAULT_PROFIT_HOLD = 60
DEFAULT_MAX_EXIT_DELAY = 5
# Features end at the last M1 close before the fill candle opened (see profit_collector.feature_end).
FILL_DECISION_CLOCK = "m1_close_before_fill_candle"
DATA_PERIOD_LABELS = {"all": "Full export", "last_6_months": "Last 6 months"}
# Train-split label options from label_agreement.csv; see mlp_trades/agreement.py.
LABEL_AGREEMENT_LABELS = {"off": "Export labels", "weight": "Weight by agreement",
                          "unanimous": "All 15 win vs rest", "unanimous_vs_lose": "All 15 win vs all lose"}
HAR_STEPS = 128
AIR_STEPS = 168
BENCHMARK_LOOKBACKS = {"uci_har": (HAR_STEPS, "samples"), "uci_air": (AIR_STEPS, "hours")}
MODEL_LABELS = {
    "dense": "Dense MLP",
    "shared": "Shared encoder",
    "per_minute": "Per-minute encoder",
}
ENCODER_WIDTHS = (16, 8)
ENCODER_PRESET = {"lookback_minutes": 720, "hidden1": 128, "hidden2": 64}
# 5m keeps one row per M5 close (the entry row and every fifth row before it) and drops the M1 columns;
# anchor flags become "fired during that 5-minute block". The lookback stays in minutes.
INPUT_TIMEFRAME_LABELS = {"1m": "1m rows (all columns)", "5m": "5m rows (M5, M15, M30)"}
INPUT_STEP = {"1m": 1, "5m": 5}
FEATURE_INPUT_LABELS = {
    "exported": "Exported features",
    "relative_price": "Relative price",
    "relative_macd": "Relative MACD",
    "stationary": "Stationary",
}
INDICATORS = ("price", "volume", "rsi", "atr", "macd", "macd_signal", "adx", "di_plus", "di_minus")
FEATURES = [f"{tf}.{name}" for tf in ("M1", "M5", "M15", "M30") for name in INDICATORS]


def window_input_size(lookback_minutes):
    if type(lookback_minutes) is not int or not 1 <= lookback_minutes <= STATES:
        raise ValueError(f"Lookback must be a whole number of minutes from 1 to {STATES}")
    return lookback_minutes * WIDTH


def default_schema():
    columns = FEATURES + [f"anchor_{i:02d}" for i in range(1, 19)]
    return {
        "version": 1,
        "current_columns": columns,
        "history_columns": columns,
        "normalization": "pre_normalized_v1",
        "timestamp_unit": "unix_seconds_utc",
        "history_order": "oldest_to_newest",
        "higher_timeframes_own_ohlcv": True,
        "closed_candles_only": True,
    }


def read_schema(path):
    schema = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = default_schema()
    for key in expected.keys() - {"current_columns", "history_columns"}:
        if schema.get(key) != expected[key]:
            raise ValueError(f"Schema: expected {key}={expected[key]!r}")
    columns = schema.get("current_columns")
    if not isinstance(columns, list) or len(columns) != WIDTH:
        raise ValueError("Schema must declare exactly 54 current_columns")
    if any(not isinstance(c, str) or not c.strip() for c in columns):
        raise ValueError("Column names must be nonempty strings")
    if columns[:36] != FEATURES or len(set(columns)) != WIDTH:
        raise ValueError("Incorrect feature order or duplicate column names")
    if schema.get("history_columns") != columns:
        raise ValueError("Current and history column order differs")
    return {key: schema[key] for key in expected}
