"""Store an auxiliary, future-realized-volatility label for each original entry."""

import numpy as np

from features import HERE, load_exchange


def label_volatility(data, raw):
    log_price = np.log(raw[:, 1])
    returns = np.diff(log_price, prepend=log_price[0])
    variance = np.r_[0, np.cumsum(returns ** 2)]
    times = data["time"]
    close_row = np.searchsorted(raw[:, 3], times, side="right") - 1
    stop = close_row + 60
    assert np.array_equal(raw[close_row, 3], times) and stop.max() < len(raw)
    sigma = 100 * np.sqrt(variance[stop + 1] - variance[close_row + 1])
    data["volatility"] = np.log(np.maximum(sigma, .005)).astype(np.float32)
    data["volatility_available"] = times + 3600


def build():
    raw = load_exchange("ETHUSDT")
    for prefix in ("", "fresh_"):
        with np.load(HERE / f"{prefix}features.npz", allow_pickle=False) as source:
            data = {name: source[name] for name in source.files}
        label_volatility(data, raw)
        np.savez(HERE / f"{prefix}volatility_features.npz", **data)
        print(prefix or "historical", len(data["time"]), "expected 60-minute volatility targets", flush=True)


if __name__ == "__main__":
    build()
