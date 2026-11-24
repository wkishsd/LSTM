"""Train-split label options built from label agreement across exit configurations.

`label_agreement.csv` (written by Pipelines/Bitfinex/label_agreement.py) records, for each filled
entry, how many of the exit-strategy study's configurations (3 stop timeframes x 5 wave cutoffs)
end its series with a net return above zero. Agreement is known only after the fact, like the label
itself, so it never becomes a feature. It only changes what the model is trained on:

    off                 the export's own labels (default)
    weight              same labels; each entry is drawn with weight |2 * wins / configurations - 1|,
                        so unanimous entries count fully and a 7-8 split barely counts
    unanimous           class 1 only when every configuration wins; everything else is class 0
    unanimous_vs_lose   class 1 when every configuration wins, class 0 when every configuration
                        loses; entries in between are left out of training

Validation and test always keep every entry and the export's own labels, so every mode is measured
on the trades the model would really face.
"""

import csv
from pathlib import Path

import numpy as np

from .schema import LABEL_AGREEMENT_LABELS

MODES = tuple(LABEL_AGREEMENT_LABELS)
FILE_NAME = "label_agreement.csv"


def default_path(bitfinex_dir):
    return Path(bitfinex_dir) / FILE_NAME


def load(path):
    """entry_id -> (wins, configurations, entry_at)."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        needed = {"entry_id", "entry_at", "wins", "configurations"}
        if not needed <= set(reader.fieldnames or []):
            raise ValueError(f"{path} needs columns {sorted(needed)}")
        table = {}
        for row in reader:
            wins, total = int(row["wins"]), int(row["configurations"])
            if not 0 <= wins <= total or total <= 0:
                raise ValueError(f"Invalid agreement counts for entry {row['entry_id']}")
            table[str(row["entry_id"])] = (wins, total, row["entry_at"])
    return table


def train_labels(data, train, mode, path):
    """Return (rows, labels, weights, report) for the train split under `mode`.

    `rows` may be a subset of `train` (unanimous_vs_lose); `weights` is None unless mode is weight.
    Entry times are checked against the export, so a file built for other entries is refused.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown label agreement mode {mode!r}")
    report = {"mode": mode}
    if mode == "off":
        return train, data.labels[train], None, report
    table = load(path)
    ids = data.ids[train]
    missing = [entry_id for entry_id in ids if entry_id not in table]
    if missing:
        raise ValueError(f"{len(missing):,} training entries are missing from {path}, e.g. {missing[:3]}")
    from .bitfinex import timestamp

    mismatched = [entry_id for entry_id, t in zip(ids, data.times[train]) if timestamp(table[entry_id][2]) != t]
    if mismatched:
        raise ValueError(f"{len(mismatched):,} entries have a different entry time in {path}, "
                         f"e.g. {mismatched[:3]}; rebuild it for this export")
    wins = np.array([table[entry_id][0] for entry_id in ids], dtype=np.int64)
    total = np.array([table[entry_id][1] for entry_id in ids], dtype=np.int64)
    unanimous_win, unanimous_loss = wins == total, wins == 0
    original = data.labels[train]
    weights = None
    if mode == "weight":
        rows, labels = train, original
        weights = np.abs(2 * wins / total - 1).astype(np.float64)
        report["mean_weight"] = float(weights.mean())
        report["effective_entries"] = float(weights.sum() ** 2 / max((weights ** 2).sum(), 1e-12))
    elif mode == "unanimous":
        rows, labels = train, unanimous_win.astype(np.int64)
    else:
        keep = unanimous_win | unanimous_loss
        rows, labels = train[keep], unanimous_win[keep].astype(np.int64)
    if mode != "weight" and len(np.unique(labels)) < 2:
        raise ValueError("The label agreement mode leaves only one class in the training split")
    report.update(
        file=str(path), configurations=sorted(set(total.tolist())),
        train_entries_before=int(len(train)), train_entries_after=int(len(rows)),
        positive_rate_before=float(original.mean()), positive_rate_after=float(labels.mean()),
        changed_labels=int((labels != original[np.isin(train, rows)]).sum()) if mode != "weight" else 0,
        unanimous_win_share=float(unanimous_win.mean()), unanimous_loss_share=float(unanimous_loss.mean()),
    )
    return rows, labels, weights, report
