"""Create pretraining examples at all closed M5 timestamps, including ordinary minutes."""

from datetime import datetime
import json

import numpy as np

from features import EXPORT, HERE, ROOT, feature_rows, load_exchange


def main():
    folder = ROOT / EXPORT
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    first = int(datetime.fromisoformat(manifest["feature_timeline_first_close"]).timestamp())
    end = int(datetime.fromisoformat(manifest["end_exclusive"]).timestamp())
    timeline = np.load(folder / "feature_timeline.npy", mmap_mode="r")
    a = 720 + (-(first // 60 + 720)) % 5
    rows = np.arange(a, min(len(timeline), (end - first) // 60), 5)
    time = first + rows * 60
    eth_raw, btc_raw = load_exchange("ETHUSDT"), load_exchange("BTCUSDT")
    base, flow, atr = feature_rows(timeline, manifest["feature_names_per_minute"], rows, time, eth_raw, btc_raw)
    price = np.log(eth_raw[:, 1])
    source = np.searchsorted(eth_raw[:, 3], time)
    assert np.array_equal(eth_raw[source, 3], time)
    returns = np.diff(price, prepend=price[0])
    square = np.r_[0, np.cumsum(returns ** 2)]
    features = []
    for horizon in (15, 60, 120):
        end_row = source + horizon
        assert end_row.max() < len(price)
        sigma = np.sqrt(square[end_row + 1] - square[source + 1])
        future = np.lib.stride_tricks.sliding_window_view(price, horizon)
        minimum = np.concatenate([future[p + 1].min(1) for p in np.array_split(source, 80)])
        features.extend((100 * (price[end_row] - price[source]), np.log(np.maximum(100 * sigma, .005)),
                         100 * (price[source] - minimum)))
    np.savez(HERE / "dense_scenarios.npz", x=np.column_stack((base, flow)), atr=atr,
             time=time, available=time + 7200, target=np.column_stack(features).astype(np.float32))
    print("Prepared", len(time), "decision-time snapshots with subsequent price scenarios", flush=True)


if __name__ == "__main__":
    main()
