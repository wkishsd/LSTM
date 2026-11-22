"""Build minute-aligned, causal Bitfinex and cross-venue features."""

import argparse
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "Training/rocm-9070"))
sys.path.insert(0, str(ROOT / "Experiments/bitfinex-summary-pilot"))

from compare_models import EXPORT, window_summaries
from mlp_trades.bitfinex import export_paths, prepare_bitfinex


def load_exchange(symbol, directory=HERE / "source", *, earlier=False):
    chunks = []
    first_month = "2023-02" if earlier else "2023-11"
    prefix = f"{symbol}-1m-"
    for archive in sorted(directory.glob(f"{prefix}*.zip")):
        if archive.stem.removeprefix(prefix) < first_month:
            continue
        with ZipFile(archive) as source, source.open(source.namelist()[0]) as file:
            header = int(file.readline().startswith(b"open_time"))
            file.seek(0)
            rows = np.loadtxt(file, delimiter=",", usecols=(0, 4, 5, 6, 8, 9), ndmin=2, skiprows=header)
        # Spot archives changed from millisecond to microsecond timestamps in 2025.
        micros = rows[0, 0] > 1e14
        unit = 1e6 if micros else 1e3
        opened, closed = rows[:, 0] / unit, rows[:, 3] / unit
        if not np.all((opened % 60 == 0) & (closed >= opened) & (closed < opened + 60)):
            raise ValueError(f"Unexpected publication boundary in {archive.name}")
        rows[:, 0] = opened
        # An archived candle can close early. Wait for the scheduled minute boundary.
        rows[:, 3] = opened + 60
        chunks.append(rows)
    if not chunks:
        raise ValueError(f"No exchange archives found for {symbol}")
    raw = np.concatenate(chunks)
    raw = raw[np.argsort(raw[:, 0], kind="stable")]
    overlap = np.flatnonzero(np.diff(raw[:, 0]) == 0)
    if not np.array_equal(raw[overlap], raw[overlap + 1]):
        raise ValueError(f"Conflicting exchange candles for {symbol}")
    raw = raw[np.r_[True, np.diff(raw[:, 0]) != 0]]
    jumps = np.flatnonzero(np.diff(raw[:, 0]) != 60)
    if len(jumps):
        left = raw[jumps, 0]
        right = raw[jumps + 1, 0]
        if not earlier or len(jumps) != 1 or left[0] != 1679661540 or right[0] != left[0] + 81 * 60:
            raise ValueError(f"Unrecognized gap in {symbol} closed minutes")
        start = int(jumps[0])
        minutes = np.arange(left[0] + 60, right[0], 60)
        flat = np.zeros((len(minutes), raw.shape[1]))
        flat[:, 0], flat[:, 1], flat[:, 3] = minutes, raw[start, 1], minutes + 60
        raw = np.concatenate((raw[:start + 1], flat, raw[start + 1:]))
        print(symbol, "advanced 80 no-data minutes with the last known price", flush=True)
    if not np.all((raw[:, 5] >= 0) & (raw[:, 5] <= raw[:, 2] + 1e-8)):
        raise ValueError(f"Invalid taker volumes for {symbol}")
    return raw


def exchange_features(raw, times):
    row = np.searchsorted(raw[:, 3], times, side="right") - 1
    if row.min() < 720 or not np.array_equal(raw[row, 3], times):
        raise ValueError("Missing exchange observations before an entry")
    price = np.log(raw[:, 1])
    volume, count, taker = raw[:, 2], raw[:, 4], raw[:, 5]
    v, c, signed = (np.r_[0, a.cumsum()] for a in (volume, count, 2 * taker - volume))
    per_minute = np.diff(price, prepend=price[0])
    vol = np.r_[0, np.cumsum(per_minute ** 2)]
    features, returns = [], []
    for width in (5, 15, 60, 180, 720):
        end, start = row + 1, row + 1 - width
        base = v[end] - v[start]
        change = 100 * (price[row] - price[row - width])
        imbalance = (signed[end] - signed[start]) / np.maximum(base, 1e-8)
        rate = np.log1p((c[end] - c[start]) / width)
        realized = 100 * np.sqrt(vol[end] - vol[start])
        features.extend((change, imbalance, rate, realized))
        returns.append(change)
    return np.column_stack(features), np.column_stack(returns)


def feature_rows(timeline, names, rows, times, eth_raw, btc_raw):
    snapshot = timeline[rows].copy()
    stable_columns = [j for j, name in enumerate(names)
                      if not name.startswith(("norm_price", "norm_volume", "norm_macd"))]
    summaries = window_summaries(timeline, rows, names)
    stable = np.column_stack((snapshot[:, stable_columns], summaries))
    eth, eth_returns = exchange_features(eth_raw, times)
    btc, btc_returns = exchange_features(btc_raw, times)
    bf_returns = np.column_stack([100 * (timeline[rows, 0] - timeline[rows - k, 0])
                                 for k in (5, 15, 60, 180, 720)])
    market = np.column_stack((eth, btc, eth_returns - btc_returns, bf_returns - eth_returns))
    atr = snapshot[:, names.index("norm_atr_14_5m")]
    if not all(np.isfinite(x).all() for x in (stable, market, atr)):
        raise ValueError("Nonfinite feature matrix")
    return stable.astype(np.float32), market.astype(np.float32), atr


def build(export, output, *, earlier=False):
    data = prepare_bitfinex(export_paths(export), training=True)
    manifest = json.loads(Path(data.paths.manifest).read_text(encoding="utf-8"))
    names = manifest["feature_names_per_minute"]
    timeline = np.load(data.paths.timeline, mmap_mode="r")
    ix = data.valid[np.argsort(data.times[data.valid], kind="stable")]
    rows, times = data.history_rows[ix], data.times[ix]
    stable, market, atr = feature_rows(timeline, names, rows, times,
                                       load_exchange("ETHUSDT", earlier=earlier),
                                       load_exchange("BTCUSDT", earlier=earlier))
    np.savez(output, base=stable, flow=market, atr=atr,
             ids=data.ids[ix], time=times, start=data.starts[ix], available=data.ends[ix], y=data.labels[ix])
    print("Built", len(ix), "entries", stable.shape[1], "Bitfinex features", market.shape[1], "cross-venue features",
          "output", output, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--export", type=Path, default=ROOT / EXPORT)
    p.add_argument("--output", type=Path, default=HERE / "features.npz")
    p.add_argument("--earlier", action="store_true", help="Include the separate earlier spot history")
    args = p.parse_args()
    build(args.export, args.output, earlier=args.earlier)
