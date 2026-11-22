"""Add recent outcome statistics that were available at each decision time."""

import numpy as np

from train_flow import HERE, view


DAY = 86400


def rolling_group_rates(times, available, outcomes, members, windows):
    ix = np.flatnonzero(members)
    order = ix[np.argsort(available[ix], kind="stable")]
    end = available[order]
    cum = np.r_[0, outcomes[order].cumsum()]
    hi = np.searchsorted(end, times, side="left")
    result = []
    for window in windows:
        lo = np.searchsorted(end, times - DAY * window, side="left")
        result.append((cum[hi] - cum[lo], hi - lo))
    return result


def derive(history, fresh):
    times = np.concatenate((history["time"], fresh["time"]))
    end = np.concatenate((history["available"], fresh["available"]))
    y = np.concatenate((history["y"], fresh["y"]))
    assert np.all(np.diff(times) >= 0) and np.all(end >= times)
    # The stable entry snapshot keeps 5 indicators per timeframe, then the 18 anchor bits.
    flags = np.vstack((history["base"][:, 20:38], fresh["base"][:, 20:38])).astype(bool)
    assert np.all(flags.any(axis=1))
    atr = np.concatenate((history["atr"], fresh["atr"]))
    buckets = np.digitize(atr, [.00075, .0015, .0025])
    hours = (times % DAY) // (6 * 3600)
    days = times // DAY
    weekend = (days + 3) % 7 >= 5
    features = [np.sin(2 * np.pi * (times % DAY) / DAY),
                np.cos(2 * np.pi * (times % DAY) / DAY), weekend]
    windows = (14, 45, 120)
    global_counts = rolling_group_rates(times, end, y, np.ones(len(times), dtype=bool), windows)
    rates = [(win + 25) / (count + 100) for win, count in global_counts]
    features.extend(rates)
    groups = [flags[:, i] for i in range(flags.shape[1])]
    groups.extend(buckets == i for i in range(4))
    groups.extend(hours == i for i in range(4))
    groups.extend([weekend, ~weekend])
    for mask in groups:
        subgroup = rolling_group_rates(times, end, y, mask, windows)
        for (wins, sample), prior in zip(subgroup, rates):
            adjusted = (wins + 40 * prior) / (sample + 40)
            features.append(np.where(mask, adjusted - prior, 0))
    recent = np.column_stack(features).astype(np.float32)
    assert np.isfinite(recent).all()
    return recent


def main():
    history = view(HERE / "features.npz")
    fresh = view(HERE / "fresh_features.npz")
    recent = derive(history, fresh)
    for prefix, data in (("", history), ("fresh_", fresh)):
        data["feedback"] = recent[:len(history["y"])] if not prefix else recent[len(history["y"]):]
        np.savez(HERE / f"{prefix}adaptive_features.npz", **data)
        print(prefix or "historical", len(data["y"]), data["feedback"].shape[1], "feedback and calendar inputs", flush=True)


if __name__ == "__main__":
    main()
