"""UCI HAR upstairs/downstairs benchmark using the published sensor sequences."""

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .data import Prepared, TimelinePaths, mmap_array
from .schema import HAR_STEPS, WIDTH


HAR_DIR = Path(__file__).resolve().parent.parent / "datasets" / "uci_har"
CHANNELS = tuple(f"{sensor}_{axis}" for sensor in ("total_acc", "body_acc", "body_gyro") for axis in "xyz")
SCHEMA = {
    "dataset": "uci_har_stairs", "version": 1, "sampling_hz": 50, "window_steps": HAR_STEPS,
    "columns": [*CHANNELS, *(f"padding_{i}" for i in range(WIDTH - len(CHANNELS)))],
    "labels": {"0": "walking downstairs", "1": "walking upstairs"},
    "source": "https://doi.org/10.24432/C54S4K", "license": "CC BY 4.0",
}


@dataclass
class HARPrepared(Prepared):
    subjects: np.ndarray
    official_test: np.ndarray


def convert_har(archive_path, output=HAR_DIR):
    """Convert the official download locally. No model or split fitting is involved."""
    groups = []
    with ZipFile(archive_path) as outer, ZipFile(BytesIO(outer.read("UCI HAR Dataset.zip"))) as archive:
        root = "UCI HAR Dataset"
        classes = dict(line.split() for line in archive.read(f"{root}/activity_labels.txt").decode().splitlines() if line.strip())
        if (classes.get("2"), classes.get("3")) != ("WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"):
            raise ValueError("Unexpected UCI HAR activity classes")
        for part in ("train", "test"):
            labels = np.loadtxt(BytesIO(archive.read(f"{root}/{part}/y_{part}.txt")), dtype=np.int64)
            subjects = np.loadtxt(BytesIO(archive.read(f"{root}/{part}/subject_{part}.txt")), dtype=np.int64)
            stairs = np.isin(labels, (2, 3))
            features = np.zeros((int(stairs.sum()), HAR_STEPS, WIDTH), dtype=np.float32)
            for column, channel in enumerate(CHANNELS):
                file = f"{root}/{part}/Inertial Signals/{channel}_{part}.txt"
                features[..., column] = np.loadtxt(BytesIO(archive.read(file)), dtype=np.float32)[stairs]
            groups.append((features, (labels[stairs] == 2).astype(np.int64), subjects[stairs]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, test = groups
    np.save(output / "timeline.npy", np.concatenate((train[0], test[0])).reshape(-1, WIDTH))
    np.savez(output / "entries.npz", labels=np.r_[train[1], test[1]], subjects=np.r_[train[2], test[2]],
             official_test=np.r_[np.zeros(len(train[1]), dtype=bool), np.ones(len(test[1]), dtype=bool)])
    (output / "manifest.json").write_text(json.dumps(SCHEMA, indent=2) + "\n", encoding="utf-8")
    return prepare_har(output)


def prepare_har(directory=HAR_DIR):
    root = Path(directory)
    paths = TimelinePaths(str(root / "timeline.npy"), str(root / "entries.npz"), str(root / "manifest.json"))
    schema = json.loads(Path(paths.manifest).read_text(encoding="utf-8"))
    if schema != SCHEMA:
        raise ValueError("Unexpected UCI HAR dataset schema")
    timeline = mmap_array(paths.timeline, "HAR sensor timeline", (WIDTH,))
    with np.load(paths.entries, allow_pickle=False) as entries:
        labels, subjects, test = (entries[key] for key in ("labels", "subjects", "official_test"))
    n = len(labels)
    if (not n or labels.shape != (n,) or subjects.shape != (n,) or test.shape != (n,)
            or test.dtype != np.dtype(bool) or len(timeline) != n * HAR_STEPS
            or not np.isin(labels, (0, 1)).all() or not np.isfinite(timeline).all()
            or not test.any() or test.all() or np.intersect1d(subjects[test], subjects[~test]).size):
        raise ValueError("Invalid HAR sensor windows, labels, or official subject split")
    indices = np.arange(n, dtype=np.int64)
    ends = (indices + 1) * HAR_STEPS - 1
    report = {"accepted": n, "excluded_count": 0, "window_steps": HAR_STEPS, "sampling_hz": 50,
              "subjects": int(len(np.unique(subjects))), "upstairs": int(labels.sum())}
    # Independent windows are packed consecutively for the shared CPU and GPU loaders.
    return HARPrepared(paths, schema, indices + 1, ends, ends - HAR_STEPS + 1, ends, ends,
                       labels, indices, report, None, subjects, test)


def subject_split(data, train_fraction, validation_fraction):
    if not (train_fraction > 0 and validation_fraction >= 0 and train_fraction + validation_fraction < 1):
        raise ValueError("Use a positive train fraction, validation of 0 or more, and a sum below 1")
    train_subjects = np.unique(data.subjects[~data.official_test])
    splits = {"train": np.flatnonzero(~data.official_test), "test": np.flatnonzero(data.official_test)}
    if validation_fraction:
        share = validation_fraction / (train_fraction + validation_fraction)
        count = min(len(train_subjects) - 1, max(1, round(len(train_subjects) * share)))
        # Keep the subjects fixed across model seeds.
        held_out = np.random.default_rng(42).permutation(train_subjects)[:count]
        validation = np.isin(data.subjects, held_out)
        splits["train"] = np.flatnonzero(~data.official_test & ~validation)
        splits["validation"] = np.flatnonzero(validation)
    report = {"policy": "disjoint_subjects_official_test", "purged_count": 0, "splits": {}}
    for name, ix in splits.items():
        if not len(ix) or len(np.unique(data.labels[ix])) != 2:
            raise ValueError(f"The HAR {name} split requires both stair activities")
        report["splits"][name] = {"count": len(ix), "entry_ids": data.ids[ix].tolist(),
                                  "subjects": np.unique(data.subjects[ix]).tolist()}
    return splits, report
