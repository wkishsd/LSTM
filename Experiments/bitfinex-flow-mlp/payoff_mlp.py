"""Walk-forward MLP with trade, gross-return, and fee prediction heads."""

from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_flow import HERE, QUOTA, SEEDS, WEEK, evaluate, fixed_fold, matrix, recency, view


ROOT = HERE.parents[1]
STEPS = (40, 160, 400)
NAMES = ("probability", "payoff", "mixed")


def add_outcomes(data, path):
    with path.open(newline="", encoding="utf-8-sig") as source:
        outcomes = {r["entry_id"]: r for r in csv.DictReader(source)}
    data["gross"] = np.array([float(outcomes[i]["target_gross_return_pct"]) for i in data["ids"]], dtype=np.float32)
    data["fees"] = np.array([float(outcomes[i]["commission_pct"]) for i in data["ids"]], dtype=np.float32)
    data["net"] = data["gross"] - data["fees"]
    assert np.array_equal(data["net"] > 0, data["y"])


def train_model(data, dest, training, test, half_life, seed, device, steps, save):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    raw, raw_test = matrix(data, "combined")[training], matrix(dest, "combined")[test]
    mu, sd = raw.mean(0), raw.std(0)
    sd = np.where(sd > 1e-8, sd, 1)
    x, xx = (torch.as_tensor(np.clip((v - mu) / sd, -5, 5), device=device) for v in (raw, raw_test))
    log_atr = np.log(np.maximum(data["atr"][training], 1e-8))
    a0, a1 = float(log_atr.mean()), float(max(log_atr.std(), 1e-8))
    base = .45 * (log_atr - a0) / a1
    base_test = .45 * (np.log(np.maximum(dest["atr"][test], 1e-8)) - a0) / a1
    weights = recency(data["time"][training], half_life)
    y, gross, fees = (data[k][training] for k in ("y", "gross", "fees"))
    prior = float(weights @ y)
    gross_cap = np.quantile(gross, [.005, .995]).astype(np.float32)
    target = torch.as_tensor(np.column_stack((y, 2 * np.clip(gross, *gross_cap), 2 * fees)),
                             dtype=torch.float32, device=device)
    offset = torch.as_tensor(base + np.log(prior / (1 - prior)), device=device)
    model = nn.Sequential(nn.Linear(x.shape[1], 48), nn.GELU(), nn.Dropout(.1),
                          nn.Linear(48, 24), nn.GELU(), nn.Linear(24, 3)).to(device)
    nn.init.zeros_(model[-1].weight)
    nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    scores = {}
    for i in range(1, max(steps) + 1):
        model.train()
        ix = rng.choice(len(y), 512, p=weights)
        output = model(x[ix])
        loss = F.binary_cross_entropy_with_logits(offset[ix] + output[:, 0], target[ix, 0])
        loss = loss + .6 * F.smooth_l1_loss(output[:, 1:], target[ix, 1:])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i in steps:
            model.eval()
            with torch.inference_mode():
                result = model(xx).cpu().numpy()
            p = base_test + result[:, 0]
            net = (result[:, 1] - result[:, 2]) / 2
            scores[i] = np.column_stack((p, net, p + 2 * net))
    if save:
        torch.save({"state": model.state_dict(), "feature_mean": torch.from_numpy(mu),
                    "feature_scale": torch.from_numpy(sd), "atr_mean": a0, "atr_std": a1,
                    "steps": max(steps), "seed": seed}, HERE / f"payoff_mlp_seed{seed}.pt")
    return scores


def walk(history, dest, indices, half, device, steps=STEPS, save=False):
    dates = dest["time"][indices]
    result = {step: np.empty((len(indices), len(NAMES))) for step in steps}
    block = (dates + 3 * 86400) // (4 * WEEK)
    for b in np.unique(block):
        mask = block == b
        test = indices[mask]
        tr = np.flatnonzero(history["available"] < dest["start"][test].min())
        outputs = [train_model(history, dest, tr, test, half, seed, device, steps, save) for seed in SEEDS]
        for step in steps:
            result[step][mask] = np.mean([r[step] for r in outputs], axis=0)
    return result


def result_for(s, data, ix):
    time, target = data["time"][ix], data["y"][ix]
    result = evaluate(s, target, time)
    weeks = (time + 3 * 86400) // WEEK
    chosen = []
    for week in np.unique(weeks):
        group = np.flatnonzero(weeks == week)
        positions = np.argsort(-s[group], kind="stable")[:round(len(group) * QUOTA)]
        chosen.extend(group[positions])
    result["mean_net_percent"] = float(data["net"][ix][chosen].mean())
    return result


def main():
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    history = view(HERE / "features.npz")
    add_outcomes(history, ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd/entry_targets.csv")
    dev = defaultdict(list)
    for bounds in ((.40, .55), (.55, .70), (.70, .85)):
        _, ix = fixed_fold(history, *bounds)
        for half in (45, 180):
            for step, outputs in walk(history, history, ix, half, device).items():
                for col, name in enumerate(NAMES):
                    dev[half, step, name].append(result_for(outputs[:, col], history, ix))
        print("Payoff MLP development fold", bounds, flush=True)
    key = max(dev, key=lambda k: sum(x["mean_net_percent"] * x["calls"] for x in dev[k]))
    half, step, mode = key
    print("Payoff recipe locked on development", key, flush=True)
    _, holdout = fixed_fold(history, .85, 1)
    fresh = view(HERE / "fresh_features.npz")
    add_outcomes(fresh, HERE / "fresh_export/entry_targets.csv")
    result = {"recipe": list(key), "development": dev[key]}
    for prefix, data, ix in (("historical", history, holdout), ("fresh", fresh, np.arange(len(fresh["y"])))):
        outputs = walk(history, data, ix, half, device, (step,), save=prefix == "fresh")[step]
        outcomes = {"atr": result_for(data["atr"][ix], data, ix)}
        for i, name in enumerate(NAMES):
            outcomes[name] = result_for(outputs[:, i], data, ix)
        result[prefix] = outcomes
        print(prefix, {k: (r["wins"], r["calls"], r["mean_net_percent"]) for k, r in outcomes.items()}, flush=True)
    (HERE / "payoff_mlp_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
