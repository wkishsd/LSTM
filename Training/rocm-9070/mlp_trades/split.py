"""Conservative temporal split: disjoint full input and outcome intervals."""

import numpy as np

from .har import HARPrepared, subject_split


def training_split(data, train_fraction, validation_fraction):
    if isinstance(data, HARPrepared):
        return subject_split(data, train_fraction, validation_fraction)
    splits, report = chronological_split(data, train_fraction, validation_fraction)
    if "data_period" in data.report:
        report["data_period"] = data.report["data_period"]
    if target := data.schema.get("profit_target"):
        for name, indices in splits.items():
            report["splits"][name]["class_counts"] = np.bincount(
                data.labels[indices], minlength=len(target["class_names"]),
            ).tolist()
    return splits, report


def chronological_split(data, train_fraction=0.7, validation_fraction=0.15):
    if not (0 < train_fraction < 1 and 0 <= validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("Train fraction must be positive, validation fraction nonnegative, and their sum below one")
    order = data.valid[np.argsort(data.times[data.valid], kind="stable")]
    n = len(order)
    a, b = int(n * train_fraction), int(n * (train_fraction + validation_fraction))
    if not (0 < a <= b < n) or (validation_fraction > 0 and a == b):
        raise ValueError("Too few valid rows for the requested temporal partitions")
    t1, t2 = data.times[order[a]], data.times[order[b]]
    train = order[data.times[order] < t1]
    val = order[(data.times[order] >= t1) & (data.times[order] < t2)]
    test = order[data.times[order] >= t2]
    if not len(train) or not len(test) or (validation_fraction and not len(val)):
        raise ValueError("Empty partition: equal timestamps must stay in the same partition")
    test_start = int(data.starts[test].min())
    report = {
        "policy": "earlier_outcome_end < later_first_input_time",
        "test_boundary": int(t2), "test_input_start": test_start,
    }
    candidates = {"train": train, "test": test}
    limits = {"train": test_start}
    if validation_fraction:
        # Use the original validation input boundary even if some validation rows are purged.
        val_start = int(data.starts[val].min())
        report.update(validation_boundary=int(t1), validation_input_start=val_start)
        candidates = {"train": train, "validation": val, "test": test}
        limits = {"train": val_start, "validation": test_start}
    purged = []
    splits = {}
    for name, rows in candidates.items():
        if name in limits:
            keep = np.maximum(data.times[rows], data.ends[rows]) < limits[name]
            purged.extend(data.ids[rows[~keep]].tolist())
            rows = rows[keep]
            if not len(rows):
                raise ValueError("Purging emptied a partition; provide a longer time span or change split fractions")
        splits[name] = rows
    report.update(purged_count=len(purged), purged_ids=purged)
    report["splits"] = {name: {"count": len(rows), "entry_ids": data.ids[rows].tolist(),
                              "first_signal": int(data.times[rows[0]]),
                              "last_signal": int(data.times[rows[-1]])}
                        for name, rows in splits.items()}
    return splits, report
