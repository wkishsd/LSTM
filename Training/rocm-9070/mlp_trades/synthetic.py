"""Small reproducible data with independently shuffled file groups."""

from pathlib import Path

import numpy as np

from .io_utils import new_output_dir, write_json
from .schema import HISTORY, WIDTH, default_schema


def generate(output, samples=80, seed=42):
    if samples < 24:
        raise ValueError("Use at least 24 synthetic samples so purging leaves three partitions")
    out = new_output_dir(output)
    rng = np.random.default_rng(seed)
    step = 720
    offsets = HISTORY + np.arange(samples) * step
    timeline = rng.normal(0, 0.2, (int(offsets[-1]) + 1, WIDTH)).astype(np.float32)
    timeline[:, 36:] = rng.integers(0, 2, timeline[:, 36:].shape)
    timeline[offsets, 36] = 1
    timeline[offsets, 2] = np.where(np.arange(samples) % 2, 0.8, -0.8)
    ids = np.arange(1000, 1000 + samples, dtype=np.int64)
    times = np.int64(1_704_067_200) + offsets * 60
    permutations = [rng.permutation(samples) for _ in range(3)]
    c, h, o = permutations
    np.save(out / "current.npy", timeline[offsets[c]])
    np.save(out / "current_ids.npy", ids[c])
    np.save(out / "current_times.npy", times[c])
    np.save(out / "history_ids.npy", ids[h])
    history = np.lib.format.open_memmap(out / "history.npy", mode="w+", dtype="float32", shape=(samples, HISTORY, WIDTH))
    history_times = np.lib.format.open_memmap(out / "history_times.npy", mode="w+", dtype="int64", shape=(samples, HISTORY))
    for row, original in enumerate(h):
        end = offsets[original]
        history[row] = timeline[end - HISTORY:end]
        history_times[row] = times[original] - np.arange(HISTORY, 0, -1, dtype=np.int64) * 60
    history.flush()
    history_times.flush()
    del history, history_times
    np.save(out / "outcome_ids.npy", ids[o])
    np.save(out / "gross.npy", np.where(o % 2, 0.011, -0.009))
    np.save(out / "commission.npy", np.full(samples, 0.001))
    np.save(out / "close_times.npy", times[o] + 3600)
    write_json(out / "schema.json", default_schema())
    config = {name: f"{name}.npy" for name in (
        "current", "current_ids", "current_times", "history", "history_ids", "history_times",
        "gross", "commission", "outcome_ids", "close_times")}
    config.update({"schema": "schema.json", "gross_unit": "fraction", "commission_unit": "fraction",
                   "batch_size": 32, "hidden1": 128, "hidden2": 64, "epochs": 3,
                   "learning_rate": 0.001, "optimizer": "adam", "device": "auto", "seed": seed})
    write_json(out / "train.json", config)
    inference = {name: config[name] for name in (
        "current", "current_ids", "current_times", "history", "history_ids", "history_times", "schema")}
    write_json(out / "predict.json", inference)
    return Path(out)
