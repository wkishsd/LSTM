"""Compare scores on the exact same older outcomes before and after five extra minutes."""

from argparse import ArgumentParser, Namespace
import csv
import json
from pathlib import Path

import numpy as np
import torch

from audit import HERE, average_precision, export_paths, load_model, prepare_bitfinex, score_checkpoint, stamp, training_split


def outcome_baselines(y, early, late, move):
    return {"matched_pairs": len(y), "winners": int(y.sum()), "base_rate": float(y.mean()) if len(y) else None,
            "early_ap": average_precision(y, early), "five_minutes_later_ap": average_precision(y, late),
            "visible_five_minute_short_move_ap": average_precision(y, move)}


def weekly_match(score, labels, times):
    groups = (times + 3 * 86400) // (7 * 86400)
    chosen = score > 0
    expected = 0.0
    for group in np.unique(groups):
        ix = groups == group
        expected += int(chosen[ix].sum()) * float(labels[ix].mean())
    return {"calls": int(chosen.sum()), "winners": int(labels[chosen].sum()),
            "expected_from_each_week_win_rate": expected}


def check(run):
    cfg = Namespace(**json.loads((run / "run.json").read_text(encoding="utf-8"))["config"])
    source = Path(cfg.bitfinex_dir)
    dataset = prepare_bitfinex(export_paths(source), training=True)
    ix, _ = training_split(dataset, cfg.train_fraction, cfg.validation_fraction)
    model, checkpoint = load_model(run / "best.pt", torch.device("cpu"), dataset.schema)
    closes = np.load(dataset.paths.timeline, mmap_mode="r", allow_pickle=False)[:, 0]
    with (source / "entry_targets.csv").open(newline="", encoding="utf-8-sig") as stream:
        fills = np.array([stamp(r["initial_fill_at"]) if r["initial_fill_at"] else -1
                          for r in csv.DictReader(stream)], dtype=np.int64)
    report = {"epoch": checkpoint["epoch"]}
    for name in ("validation", "test"):
        indices = ix[name]
        score, label = score_checkpoint(model, dataset, cfg, indices)
        times = dataset.times[indices]
        wanted = times + 300
        nxt = times.searchsorted(wanted).clip(0, len(times) - 1)
        left = np.flatnonzero(times[nxt] == wanted)
        right = nxt[left]
        older, later = indices[left], indices[right]
        source_move = closes[dataset.history_rows[older]] - closes[dataset.history_rows[later]]
        known = dataset.ends[older] <= times[right]
        has_fill = (fills[older] >= 0) & (fills[older] <= times[right])
        groups = {"all_pairs": np.ones(len(left), dtype=bool), "already_resolved": known,
                  "still_unresolved": ~known, "filled_and_unresolved": has_fill & ~known,
                  "no_fill_yet_and_unresolved": ~has_fill & ~known}
        report[name] = {key: outcome_baselines(label[left][mask], score[left][mask], score[right][mask], source_move[mask])
                        for key, mask in groups.items()}
        report[name]["within_week_call_baseline"] = weekly_match(score, label, times)
    (HERE / "five_minute_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    torch.set_num_threads(3)
    check(parser.parse_args().run)
