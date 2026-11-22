"""Add observable price geometry for the collector's actual initial order rule."""

from datetime import datetime
import json
from pathlib import Path

import numpy as np

from features import HERE, ROOT, EXPORT


def unpack_raw(directory, first, n):
    channels = np.full((n, 4), np.nan)
    for path in sorted(directory.glob("*.json")):
        block = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
        if not len(block):
            continue
        available = block[:, 0].astype(np.int64) // 1000 + 60
        pos = (available - first) // 60
        keep = (pos >= 0) & (pos < n)
        p, x = pos[keep], block[keep, 1:5]
        overlap = np.isfinite(channels[p, 1])
        if not np.array_equal(channels[p[overlap]], x[overlap]):
            raise ValueError("Cached candle values disagree")
        channels[p] = x
    return channels


def sums(values, positions, width):
    accum = np.r_[0, np.cumsum(values)]
    return accum[positions + 1] - accum[positions + 1 - width]


def ranges(values, positions, width, op):
    sliding = np.lib.stride_tricks.sliding_window_view(values, width)
    return np.concatenate([op(sliding[group - width + 1], axis=1)
                           for group in np.array_split(positions, max(1, len(positions) // 2000))])


def build(export, cache, prefix):
    manifest = json.loads((export / "manifest.json").read_text())
    first = int(datetime.fromisoformat(manifest["feature_timeline_first_close"]).timestamp())
    timeline = np.load(export / "feature_timeline.npy", mmap_mode="r")
    raw = unpack_raw(cache, first, len(timeline))
    observed = np.isfinite(raw[:, 1])
    closes = np.exp(timeline[:, 0].astype(float))
    np.testing.assert_allclose(closes[observed], raw[observed, 1], rtol=2e-6)
    raw[~observed] = closes[~observed, None]
    o, c, hi, lo = raw.T
    with np.load(HERE / f"{prefix}features.npz", allow_pickle=False) as inputs:
        frame = {key: inputs[key] for key in inputs.files}
    times = frame["time"]
    pos = (times - first) // 60
    assert np.array_equal(first + pos * 60, times) and pos.min() >= 720
    current = c[pos]
    atr = np.maximum(frame["atr"] * current, 0.05)
    offset = manifest["exit_settings"]["initial_entry_offset_down"]
    order_price = lo[pos] - offset
    result = [(current - order_price) / atr, (o[pos] - c[pos]) / atr,
              (hi[pos] - current) / atr, (current - lo[pos]) / atr]
    for width in (5, 15, 60, 240, 720):
        minimum = ranges(lo, pos, width, np.min)
        maximum = ranges(hi, pos, width, np.max)
        result.extend(((order_price - minimum) / atr, (maximum - current) / atr,
                       (current - minimum) / np.maximum(maximum - minimum, .01),
                       sums(~observed, pos, width) / width))
    frame["geometry"] = np.column_stack(result).astype(np.float32)
    assert np.isfinite(frame["geometry"]).all()
    np.savez(HERE / f"{prefix}geometry_features.npz", **frame)
    print(prefix or "historical", len(times), frame["geometry"].shape[1], "order geometry inputs", flush=True)


if __name__ == "__main__":
    build(ROOT / EXPORT, ROOT / "Exports/candle_cache/ETHUSD/1m", "")
    build(HERE / "fresh_export", HERE / "fresh_candle_cache/ETHUSD/1m", "fresh_")
