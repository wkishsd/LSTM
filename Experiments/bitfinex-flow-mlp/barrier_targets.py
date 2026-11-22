"""Add first-touch labels for supervised use alongside the actual strategy outcomes."""

import json
import sys

import numpy as np

from features import EXPORT, HERE, ROOT
from mlp_trades.bitfinex import export_paths, prepare_bitfinex

sys.path.insert(0, str(ROOT / "Experiments/bitfinex-controlled-audit"))
from barrier_study import DOWN, make_barrier_labels


def build(export, cache, prefix):
    data = prepare_bitfinex(export_paths(export), training=True)
    manifest = json.loads((export / "manifest.json").read_text())
    timeline = np.load(export / "feature_timeline.npy", mmap_mode="r")
    label, _ = make_barrier_labels(data, timeline, manifest["feature_names_per_minute"], manifest, cache)
    with np.load(HERE / f"{prefix}volatility_features.npz", allow_pickle=False) as f:
        pack = {name: f[name] for name in f.files}
    order = data.valid[np.argsort(data.times[data.valid], kind="stable")]
    assert np.array_equal(data.ids[order], pack["ids"])
    pack["barrier_target"] = (label[order] == DOWN).astype(np.float32)
    pack["barrier_valid"] = label[order] >= 0
    np.savez(HERE / f"{prefix}barrier_features.npz", **pack)
    print(prefix or "historical", int(pack["barrier_valid"].sum()), "ordered first-touch labels", flush=True)


if __name__ == "__main__":
    build(ROOT / EXPORT, ROOT / "Exports/candle_cache/ETHUSD/1m", "")
    build(HERE / "fresh_export", HERE / "fresh_candle_cache/ETHUSD/1m", "fresh_")
