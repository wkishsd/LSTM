"""Test a two-stage model of initial fill and a win conditional on filling."""

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_flow import HERE, SEEDS, WEEK, evaluate, fixed_fold, matrix, recency, view


STEPS = (40, 160, 400)
MODES = ("fill_times_win", "conditional_win", "direct_win")


def add_fill(data, source):
    with source.open(newline="", encoding="utf-8-sig") as stream:
        outcomes = {row["entry_id"]: row for row in csv.DictReader(stream)}
    rows = [outcomes[i] for i in data["ids"]]
    times = np.array([int(datetime.fromisoformat(row["entry_at"]).timestamp()) for row in rows])
    if not np.array_equal(times, data["time"]):
        raise ValueError(f"Entry times do not align in {source}")
    data["filled"] = np.array([row["initial_order_status"] == "filled" for row in rows])
    if np.any(data["y"].astype(bool) & ~data["filled"]):
        raise ValueError(f"A positive target has no fill in {source}")


def make_model(width):
    return nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Dropout(.1),
                         nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 3))


def fit(x, xx, y, filled, probability, seed, device, steps, checkpoint):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 1e-8, sd, 1)
    x, xx = (torch.as_tensor(np.clip((v - mu) / sd, -5, 5), device=device) for v in (x, xx))
    target = torch.as_tensor(np.column_stack((filled, y, y)), dtype=torch.float32, device=device)
    model = make_model(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    scores = {}
    for step in range(1, max(steps) + 1):
        model.train()
        ix = rng.choice(len(y), 512, p=probability)
        logits = model(x[ix])
        batch = target[ix]
        error = F.binary_cross_entropy_with_logits(logits, batch, reduction="none")
        mask = batch[:, 0]
        loss = error[:, 0].mean() + (error[:, 1] * mask).sum() / mask.sum().clamp_min(1) + error[:, 2].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step in steps:
            model.eval()
            with torch.inference_mode():
                fill, win, direct = model(xx).sigmoid().cpu().numpy().T
            scores[step] = np.column_stack((fill * win, win, direct, fill))
            if checkpoint and step == checkpoint[0]:
                torch.save({"model_state": model.state_dict(), "mean": torch.from_numpy(mu),
                            "std": torch.from_numpy(sd), "steps": step, "seed": seed}, checkpoint[1])
    return scores


def walk(history, dest, indices, half, device, steps=STEPS, save_step=None):
    inputs = matrix(history, "geometry")
    other = matrix(dest, "geometry")
    result = {step: np.empty((len(indices), 4)) for step in steps}
    block = (dest["time"][indices] + 3 * 86400) // (4 * WEEK)
    for b in np.unique(block):
        mask = block == b
        test = indices[mask]
        tr = np.flatnonzero(history["available"] < dest["start"][test].min())
        probability = recency(history["time"][tr], half)
        scores = [fit(inputs[tr], other[test], history["y"][tr], history["filled"][tr], probability, seed,
                      device, steps, (save_step, HERE / f"fill_mlp_fresh_block{b}_seed{seed}.pt")
                      if save_step else None)
                  for seed in SEEDS]
        for step in steps:
            result[step][mask] = np.mean([forecast[step] for forecast in scores], axis=0)
    return result


def replay():
    frame = view(HERE / "fresh_geometry_features.npz")
    add_fill(frame, HERE / "fresh_export/entry_targets.csv")
    x = matrix(frame, "geometry")
    blocks = (frame["time"] + 3 * 86400) // (4 * WEEK)
    estimated = np.empty(len(x))
    for block in np.unique(blocks):
        rows = blocks == block
        predictions = []
        for seed in SEEDS:
            path = HERE / f"fill_mlp_fresh_block{block}_seed{seed}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            model = make_model(x.shape[1])
            model.load_state_dict(checkpoint["model_state"])
            model.eval()
            inputs = (torch.from_numpy(x[rows]) - checkpoint["mean"]) / checkpoint["std"]
            with torch.inference_mode():
                predictions.append(model(inputs.clamp(-5, 5))[:, 0].sigmoid().numpy())
        estimated[rows] = np.mean(predictions, axis=0)
    actual = evaluate(estimated, frame["filled"], frame["time"])["ap"]
    report = json.loads((HERE / "fill_mlp_results.json").read_text(encoding="utf-8"))
    expected = report["fresh"]["per_step"][str(report["recipe"][1])]["fill_ap"]
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    print(f"Replayed fill commitments for {len(x)} examples, AP {actual:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Verify the saved model on the new six-day slice")
    args = parser.parse_args()
    torch.set_num_threads(3)
    if args.replay:
        replay()
        return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = view(HERE / "geometry_features.npz")
    export = "Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd/entry_targets.csv"
    add_fill(data, HERE.parents[1] / export)
    dev = defaultdict(list)
    for bounds in ((.40, .55), (.55, .70), (.70, .85)):
        _, ix = fixed_fold(data, *bounds)
        for half in (45, 180):
            for step, scores in walk(data, data, ix, half, device).items():
                for col, mode in enumerate(MODES):
                    dev[half, step, mode].append(evaluate(scores[:, col], data["y"][ix], data["time"][ix]))
        print("Fill MLP development fold", bounds, flush=True)
    selected = max(dev, key=lambda k: sum(m["wins"] for m in dev[k]))
    half, step, mode = selected
    _, test = fixed_fold(data, .85, 1)
    fresh = view(HERE / "fresh_geometry_features.npz")
    add_fill(fresh, HERE / "fresh_export/entry_targets.csv")
    results = {"recipe": list(selected), "development": dev[selected]}
    print("Recipe selected on development", selected, flush=True)
    for label, frame, ix in (("historical", data, test), ("fresh", fresh, np.arange(len(fresh["time"])))):
        scores = walk(data, frame, ix, half, device, save_step=step if label == "fresh" else None)
        time, y, filled = (frame[key][ix] for key in ("time", "y", "filled"))
        atr = evaluate(frame["atr"][ix], y, time)
        result = {"atr": atr, "per_step": {}}
        for i, prediction in scores.items():
            metrics = {key: evaluate(prediction[:, c], y, time) for c, key in enumerate(MODES)}
            metrics["fill_ap"] = evaluate(prediction[:, 3], filled, time)["ap"]
            metrics["fill_rate"] = float(filled.mean())
            result["per_step"][str(i)] = metrics
            print(label, i, "winner counts", [metrics[key]["wins"] for key in MODES],
                  "ATR", atr["wins"], "fill AP", metrics["fill_ap"], flush=True)
        results[label] = result
    (HERE / "fill_mlp_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
