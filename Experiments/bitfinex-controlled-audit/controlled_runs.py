"""Run matched Bitfinex checks using the existing MLP, loaders and optimizer loop."""

import argparse
from dataclasses import replace
from itertools import islice
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "Training/rocm-9070"
sys.path.insert(0, str(TRAINER))

from mlp_trades.bitfinex import export_paths, prepare_bitfinex, select_bitfinex_period, timestamp
from mlp_trades.engine import (
    cache_timeline, fit_feature_scaling, load_model, make_loader, resolve_device,
    run_epoch, save_checkpoint, seed_everything,
)
from mlp_trades.metrics import average_precision
from mlp_trades.model import TradeMLP
from mlp_trades.split import training_split


RECENT_RUN = TRAINER / "runs/rx9070-20261120-004746-af2d"
WEEK = 7 * 86400
QUOTA = 1000 / 15728


def batches_forever(loader):
    while True:
        yield from loader


def score(model, loader, device):
    model.eval()
    with torch.inference_mode():
        scores = []
        for x, _ in loader:
            logits = model(x.to(device))
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    return np.concatenate(scores).astype(np.float64)


def summarize(scores, data, indices):
    y = data.labels[indices]
    weeks = (data.times[indices] + 3 * 86400) // WEEK
    calls = winners = 0
    weekly = {}
    for week in np.unique(weeks):
        local = np.flatnonzero(weeks == week)
        k = round(len(local) * QUOTA)
        selected = local[np.argsort(-scores[local], kind="stable")[:k]]
        hits = int(y[selected].sum())
        calls += k
        winners += hits
        weekly[str(int(week))] = [hits, k]
    return {
        "count": len(y), "winner_rate": float(y.mean()),
        "average_precision": average_precision(y, scores),
        "accuracy": float(np.mean((scores > 0) == y)),
        "loss": float(np.mean(np.logaddexp(0, scores) - scores * y)),
        "weekly_quota_winners": winners, "weekly_quota_calls": calls,
        "weekly": weekly,
    }


def fit(name, data, splits, device, out, *, steps, seed, lookback=720, model_type="dense"):
    opts = SimpleNamespace(seed=seed, num_workers=0, batch_size=512, lookback_minutes=lookback,
                           timeline_cache="auto")
    architecture = {"model": model_type, "hidden1": 128, "hidden2": 64,
                    "hidden3": 32 if model_type == "dense" else 0, "lookback_minutes": lookback,
                    "activation": "relu", "dropout": 0.2, "standardize": True}
    seed_everything(seed)
    model = TradeMLP(**architecture)
    fit_feature_scaling(model, data, splits["train"])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    windows = cache_timeline(data, opts, device, opts.batch_size)
    train = batches_forever(make_loader(data, splits["train"], opts, device, shuffle=True, windows=windows))
    valid = make_loader(data, splits["validation"], opts, device, windows=windows)
    history = []
    best = None
    ckpt = out / f"{name}.pt"
    for start in range(0, steps, 20):
        count = min(20, steps - start)
        train_metrics = run_epoch(model, islice(train, count), device, optimizer)
        valid_metrics = run_epoch(model, valid, device)
        evaluated = {"updates": start + count, "train": train_metrics, "validation": valid_metrics}
        history.append(evaluated)
        criteria = valid_metrics["average_precision"], -valid_metrics["loss"]
        if best is None or criteria > best:
            best = criteria
            save_checkpoint(ckpt, {
                "format_version": 1, "input_size": model.input_size, "schema": data.schema,
                "architecture": architecture, "model_state": model.state_dict(),
                "optimizer_updates": start + count,
            })
        if start == 0 or (start + count) % 200 == 0:
            print(name, start + count, "train loss", round(train_metrics["loss"], 4),
                  "validation AP", round(valid_metrics["average_precision"], 4), flush=True)
    model, saved = load_model(ckpt, device, data.schema)
    result = {"name": name, "architecture": architecture, "seed": seed,
              "max_updates": steps, "best_updates": saved["optimizer_updates"],
              "training_rows": len(splits["train"]), "history": history}
    for part in ("validation", "test"):
        ix = splits[part]
        loader = make_loader(data, ix, opts, device, windows=windows)
        result[part] = summarize(score(model, loader, device), data, ix)
    (out / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(name, "selected", result["best_updates"], "val AP", result["validation"]["average_precision"],
          "test AP", result["test"]["average_precision"], "test weekly hits",
          result["test"]["weekly_quota_winners"], "/", result["test"]["weekly_quota_calls"], flush=True)


def known_signal(data, splits, timeline, names):
    rows = data.history_rows
    rsi = timeline[:, names.index("norm_rsi_14_5m")]
    hist = timeline[:, names.index("norm_macd_12_26_5m")] - timeline[:, names.index("norm_macd_signal_12_26_9_5m")]
    early = rsi[rows - 60] - rsi[rows - 90]
    late = hist[rows] - hist[rows - 30]
    e, h = np.median(early[splits["train"]]), np.median(late[splits["train"]])
    first, second = early > e, late > h
    labels = (first ^ second).astype(np.int64)
    for part, indices in splits.items():
        print("control", part, "positive rate", round(labels[indices].mean(), 4),
              "early bit alone AP", round(average_precision(labels[indices], first[indices].astype(float)), 4),
              "late bit alone AP", round(average_precision(labels[indices], second[indices].astype(float)), 4),
              flush=True)
    return replace(data, labels=labels, schema={**data.schema, "diagnostic_target": "xor_of_two_past_moves",
                                               "training_medians": [float(e), float(h)]})


def replace_volume(data, timeline, names, manifest, out):
    """Change only four volume columns after verifying the original values from raw candles."""
    first = timestamp(manifest["feature_timeline_first_close"])
    count = len(timeline)
    dest = out / "volume_log1p.npy"
    corrected = np.lib.format.open_memmap(dest, mode="w+", dtype=np.float32, shape=timeline.shape)
    corrected[:] = timeline
    times = first + np.arange(count, dtype=np.int64) * 60
    for timeframe, minutes in (("1m", 1), ("5m", 5), ("15m", 15), ("30m", 30)):
        seconds = minutes * 60
        close_slots = times // seconds
        start = close_slots[0]
        volumes = np.zeros(close_slots[-1] - start + 1)
        seen = np.zeros(len(volumes), dtype=bool)
        directory = ROOT / "Exports/candle_cache/ETHUSD" / timeframe
        for path in sorted(directory.glob("*.json")):
            rows = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=float)
            if not rows.size:
                continue
            pos = (rows[:, 0].astype(np.int64) // 1000 + seconds) // seconds - start
            match = (pos >= 0) & (pos < len(volumes))
            pos, current = pos[match], rows[match, 5]
            duplicate = seen[pos]
            if not np.array_equal(volumes[pos[duplicate]], current[duplicate]):
                raise ValueError(f"Conflicting cached volumes in {path.name}")
            volumes[pos], seen[pos] = current, True
        raw = volumes[close_slots - start]
        assert np.all(np.isfinite(raw)) and np.all(raw >= 0)
        expected = np.zeros(len(raw))
        np.log(raw, out=expected, where=raw > 0)
        col = names.index(f"norm_volume_{timeframe}")
        np.testing.assert_allclose(expected.astype(np.float32), timeline[:, col], rtol=2e-6, atol=2e-6,
                                   err_msg=f"Could not reconstruct {timeframe} exported volume")
        corrected[:, col] = np.log1p(raw)
        print("verified", timeframe, "zero-volume minutes", int((raw == 0).sum()),
              "exactly-one-volume minutes", int((raw == 1).sum()), flush=True)
    corrected.flush()
    del corrected
    paths = replace(data.paths, timeline=str(dest))
    return replace(data, paths=paths, schema={**data.schema, "diagnostic_features": "log1p_volume"})


def study(mode, seed, steps, out):
    torch.set_num_threads(4)
    device = resolve_device("cuda:0")
    config = json.loads((RECENT_RUN / "run.json").read_text(encoding="utf-8"))["config"]
    export = ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd"
    paths = export_paths(export)
    full = prepare_bitfinex(paths, training=True)
    recent = select_bitfinex_period(full, "last_6_months")
    splits, report = training_split(recent, config["train_fraction"], config["validation_fraction"])
    timeline = np.load(paths.timeline, mmap_mode="r")
    manifest = json.loads(Path(paths.manifest).read_text())
    names = manifest["feature_names_per_minute"]
    if mode == "signal":
        data = known_signal(full, splits, timeline, names)
        fit("control_dense_720", data, splits, device, out, steps=steps, seed=seed)
        fit("control_dense_1", data, splits, device, out, steps=steps, seed=seed, lookback=1)
        fit("control_per_minute_720", data, splits, device, out, steps=steps, seed=seed, model_type="per_minute")
    else:
        ix = full.valid
        earlier = ix[np.maximum(full.times[ix], full.ends[ix]) < report["validation_input_start"]]
        assert np.all(np.isin(splits["train"], earlier))
        assert len(earlier) > len(splits["train"])
        atr = timeline[full.history_rows, names.index("norm_atr_14_5m")]
        reference = {part: summarize(atr[splits[part]], full, splits[part]) for part in ("validation", "test")}
        reference["split"] = report["data_period"]
        (out / "atr.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
        if mode == "history":
            fit(f"trade_recent_seed{seed}", full, splits, device, out, steps=steps, seed=seed)
            fit(f"trade_all_seed{seed}", full, {**splits, "train": earlier}, device, out, steps=steps, seed=seed)
        else:
            amended = replace_volume(full, timeline, names, manifest, out)
            fit(f"trade_log1p_all_seed{seed}", amended, {**splits, "train": earlier}, device, out,
                steps=steps, seed=seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("signal", "history", "volume"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--updates", type=int, default=600)
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("runs"))
    args = parser.parse_args()
    if args.updates < 1:
        parser.error("--updates must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    study(args.mode, args.seed, args.updates, args.out)


if __name__ == "__main__":
    main()
