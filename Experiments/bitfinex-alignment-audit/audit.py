"""Check the live binary export path and test small score/label misalignments."""

from argparse import ArgumentParser, Namespace
import csv
from datetime import datetime
from decimal import Decimal
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "Training/rocm-9070"))

from mlp_trades.bitfinex import export_paths, prepare_bitfinex
from mlp_trades.engine import DeviceWindows, load_model, make_loader
from mlp_trades.metrics import average_precision
from mlp_trades.split import training_split


def stamp(value):
    return int(datetime.fromisoformat(value).timestamp())


def audit_csv(source, manifest, data, timeline):
    first = stamp(manifest["feature_timeline_first_close"])
    fees = [Decimal(str(manifest[f"{kind}_commission_pct_per_fill"])) for kind in ("entry", "exit")]
    anchors = manifest["main_anchor_ids"]
    file = source / "entry_targets.csv"
    counts = {"rows": 0, "csv_binary_disagreements": 0, "exact_break_even": 0}
    with file.open(newline="", encoding="utf-8-sig") as stream:
        for i, row in enumerate(csv.DictReader(stream)):
            t = stamp(row["entry_at"])
            feature_row, remainder = divmod(t - first, 60)
            assert remainder == 0 and data.times[i] == t and data.ids[i] == row["entry_id"], row["entry_id"]
            assert feature_row == int(row["feature_end_row"]) == data.history_rows[i], row["entry_id"]
            requested = {int(item) for item in row["main_anchor_numbers"].split(";")}
            actual = {n for n, flag in zip(anchors, timeline[feature_row, 36:]) if flag == 1}
            assert actual == requested, row["entry_id"]
            cash = Decimal(row["target_gross_return_pct"])
            fee = sum(rate * int(row[f"filled_{kind}_orders"]) for rate, kind in zip(fees, ("entry", "exit")))
            assert abs(fee - Decimal(row["commission_pct"])) < Decimal("1e-10"), row["entry_id"]
            assert abs(cash - fee - Decimal(row["net_return_pct"])) <= Decimal("0.000001"), row["entry_id"]
            pos = int(cash > fee)
            assert int(data.labels[i]) == pos, row["entry_id"]
            assert data.ends[i] == stamp(row["label_available_at"]) >= t, row["entry_id"]
            counts["csv_binary_disagreements"] += pos != int(Decimal(row["binary_target"]))
            counts["exact_break_even"] += cash == fee
            counts["rows"] += 1
    assert counts["rows"] == len(data.valid) == len(data.ids)
    return counts


def check_raw_closes(times, rows, timeline):
    cache = ROOT / "Exports/candle_cache/ETHUSD"
    matches = {}
    for j, mins in enumerate((1, 5, 15, 30)):
        tf = f"{mins}m"
        duration = mins * 60
        fields = []
        for path in (cache / tf).glob("*.json"):
            candles = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
            if candles.size:
                fields.append(candles[:, [0, 2, 5]])
        raw = np.concatenate(fields)
        raw = raw[np.argsort(raw[:, 0], kind="stable")]
        repeated = np.flatnonzero(np.diff(raw[:, 0]) == 0)
        np.testing.assert_array_equal(raw[repeated], raw[repeated + 1])
        availability = raw[:, 0].astype(np.int64) // 1000 + duration
        which = availability.searchsorted(times, side="right") - 1
        assert np.all(which >= 0)
        expected_close = np.log(raw[which, 1])
        gap = availability[which] != times // duration * duration
        activity = np.where(gap, 0, raw[which, 2])
        expected_volume = np.log(np.where(activity > 0, activity, 1))
        np.testing.assert_allclose(timeline[rows, j * 9], expected_close, atol=1e-6, rtol=0)
        np.testing.assert_allclose(timeline[rows, j * 9 + 1], expected_volume, atol=2e-6, rtol=0)
        matches[tf] = {"checked": len(rows), "missing_candle_slots": int(gap.sum())}
    return matches


def check_batches(data, cfg, ii, timeline):
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(cfg.seed)
    expected = DataLoader(torch.from_numpy(ii), batch_size=32, shuffle=True, generator=gen)
    cpu_loader = make_loader(data, ii, cfg, cpu, shuffle=True, batch_size=32)
    cache = DeviceWindows(torch.tensor(np.asarray(timeline), device=gpu), cfg.lookback_minutes)
    gpu_loader = make_loader(data, ii, cfg, gpu, shuffle=True, batch_size=32, windows=cache)
    batches, count = 0, 0
    for order, (x, y), (x_gpu, y_gpu) in zip(expected, cpu_loader, gpu_loader, strict=True):
        pick = order.numpy()
        idx = data.history_rows[pick, None] + np.arange(1 - cfg.lookback_minutes, 1)
        raw = timeline[idx].reshape(len(pick), -1)
        np.testing.assert_array_equal(raw, x.numpy())
        np.testing.assert_array_equal(raw, x_gpu.cpu().numpy())
        np.testing.assert_array_equal(y.numpy(), data.labels[pick])
        np.testing.assert_array_equal(y_gpu.numpy(), data.labels[pick])
        count += len(order)
        batches += 1
    return {"shuffled_examples": count, "shuffled_batches": batches, "lookback": cfg.lookback_minutes}


def score_checkpoint(model, data, cfg, ix):
    cpu = torch.device("cpu")
    loader = make_loader(data, ix, cfg, cpu, batch_size=256)
    results = []
    with torch.inference_mode():
        for features, labels in loader:
            logits = model(features)
            results.append(((logits[:, 1] - logits[:, 0]).numpy(), labels.numpy()))
    score, y = (np.concatenate(values) for values in zip(*results))
    np.testing.assert_array_equal(y, data.labels[ix])
    return score, y


def evaluate(score, labels):
    calls = score > 0
    use = int(calls.sum())
    tp = int(labels[calls].sum())
    rate = float(labels.mean())
    return {"sample": len(labels), "calls": use, "true_positives": tp, "positives": int(labels.sum()),
            "expected_at_base_rate": rate * use, "average_precision": average_precision(labels, score)}


def shifts(score, y, time):
    lags = (-24, -12, -6, -3, -2, -1, 0, 1, 2, 3, 6, 12, 24)
    fixed = np.arange(24, len(y) - 24)
    signal = {str(i): evaluate(score[fixed], y[fixed + i]) for i in lags}
    minutes = {}
    for delta in (-60, -30, -15, -10, -5, 5, 10, 15, 30, 60):
        target = time + 60 * delta
        k = time.searchsorted(target).clip(0, len(time) - 1)
        match = time[k] == target
        minutes[str(delta)] = {"shifted": evaluate(score[match], y[k[match]]),
                               "unshifted_same_scores": evaluate(score[match], y[match])}
    return {"unshifted": evaluate(score, y), "signal_offsets": signal, "minute_offsets": minutes}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(3)
    run = args.run
    record = json.loads((run / "run.json").read_text(encoding="utf-8"))
    cfg = Namespace(**record["config"])
    assert cfg.dataset == "bitfinex"
    source = Path(cfg.bitfinex_dir)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    data = prepare_bitfinex(export_paths(source), training=True)
    timeline = np.load(data.paths.timeline, mmap_mode="r", allow_pickle=False)
    result = {"run": run.name, "csv_audit": audit_csv(source, manifest, data, timeline)}
    parts, boundaries = training_split(data, cfg.train_fraction, cfg.validation_fraction)
    assert max(data.ends[parts["train"]]) < boundaries["validation_input_start"]
    assert max(data.ends[parts["validation"]]) < boundaries["test_input_start"]
    rng = np.random.default_rng(69)
    sample = np.concatenate([rng.choice(v, 128, replace=False) for v in parts.values()])
    result["batch_audit"] = check_batches(data, cfg, sample, timeline)
    offsets = np.array([0, 1, cfg.lookback_minutes - 1])
    # Take 32 uniform examples from each time split for the candle-source check.
    raw_sample = sample.reshape(3, -1)[:, :32].flatten()
    r = (data.history_rows[raw_sample, None] - offsets).ravel()
    t = (data.times[raw_sample, None] - 60 * offsets).ravel()
    result["raw_candle_audit"] = check_raw_closes(t, r, timeline)
    print("Verified CSV, source candle times, and shuffled windows", flush=True)
    for checkpoint, names in (("best.pt", ("validation", "test")), ("last.pt", ("validation",))):
        model, details = load_model(run / checkpoint, torch.device("cpu"), data.schema)
        result[checkpoint] = {"epoch": details["epoch"]}
        for name in names:
            scores, labels = score_checkpoint(model, data, cfg, parts[name])
            result[checkpoint][name] = shifts(scores, labels, data.times[parts[name]])
            print(checkpoint, details["epoch"], name, result[checkpoint][name]["unshifted"], flush=True)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
