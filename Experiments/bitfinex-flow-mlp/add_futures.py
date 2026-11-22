"""Add timestamp-aligned perpetual-futures activity and published funding rates."""

import json
from zipfile import ZipFile

import numpy as np

from features import HERE, exchange_features, load_exchange


def funding_in_effect(directory, times):
    parts = []
    for filename in sorted(directory.glob("ETHUSDT-fundingRate-*.zip")):
        with ZipFile(filename) as archive, archive.open(archive.namelist()[0]) as data:
            parts.append(np.loadtxt(data, delimiter=",", skiprows=1, ndmin=2))
    tail = json.loads((directory / "funding_november.json").read_text(encoding="utf-8"))
    parts.append(np.array([(row["fundingTime"], 0, float(row["fundingRate"])) for row in tail]))
    rates = np.concatenate(parts)
    assert np.all(np.diff(rates[:, 0]) > 0)
    available = np.searchsorted(rates[:, 0] / 1000, times, side="left") - 1
    assert available.min() >= 1
    assert np.all(times - rates[available, 0] / 1000 <= 24 * 3600)
    return np.column_stack((1e4 * rates[available, 2], 1e4 * (rates[available, 2] - rates[available - 1, 2])))


def build():
    directory = HERE / "futures_source"
    future = load_exchange("ETHUSDT", directory)
    spot = load_exchange("ETHUSDT")
    for prefix in ("", "fresh_"):
        with np.load(HERE / f"{prefix}features.npz", allow_pickle=False) as data:
            frame = {key: data[key] for key in data.files}
        times = frame["time"]
        futures_features, futures_returns = exchange_features(future, times)
        _, spot_returns = exchange_features(spot, times)
        forward_pos = np.searchsorted(future[:, 3], times, side="right") - 1
        spot_pos = np.searchsorted(spot[:, 3], times, side="right") - 1
        current_premium = 1e4 * (np.log(future[forward_pos, 1]) - np.log(spot[spot_pos, 1]))
        funding = funding_in_effect(directory, times)
        frame["futures"] = np.column_stack((futures_features, futures_returns - spot_returns,
                                            current_premium, funding)).astype(np.float32)
        assert np.isfinite(frame["futures"]).all()
        np.savez(HERE / f"{prefix}futures_features.npz", **frame)
        print(prefix or "historical", len(times), frame["futures"].shape[1], "futures inputs", flush=True)


if __name__ == "__main__":
    build()
