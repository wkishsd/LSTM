"""Check whether forecasting future market volatility helps rank the original trade target."""

from collections import defaultdict
import argparse
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_flow import HERE, SEEDS, WEEK, evaluate, fixed_fold, matrix, recency, view


CHECK_STEPS = (20, 120, 240, 480)
MODES = ("volatility", "trade", "both")
TAIL_MODES = ("mean_vol", "upper_vol", "mixed_vol")
BARRIER_MODES = ("volatility", "trade", "first_touch", "strategy_plus_touch")


class VolatilityMLP(nn.Module):
    def __init__(self, width, targets=2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width, 48), nn.GELU(), nn.Dropout(.10),
                                 nn.Linear(48, 24), nn.GELU(), nn.Linear(24, targets))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


def run_fold(data, target_frame, ix, half_life, device, steps=CHECK_STEPS, checkpoints=None, tails=False):
    output = {}
    period = (target_frame["time"][ix] + 3 * 86400) // (4 * WEEK)
    for seen in np.unique(period):
        block_ix = ix[period == seen]
        cutoff = target_frame["start"][block_ix].min()
        tr = np.flatnonzero(np.maximum(data["available"], data["volatility_available"]) < cutoff)
        scores = [train_once(data, target_frame, tr, block_ix, seed, half_life, device, steps,
                             f"{checkpoints}_block{seen}_seed{seed}.pt" if checkpoints else None, tails)
                  for seed in SEEDS]
        for step in steps:
            blend = np.mean([pred[step] for pred, _ in scores], axis=0)
            target = output.setdefault(step, {"score": np.empty((len(ix), blend.shape[1])),
                                             "baseline_vol": np.empty(len(ix))})
            target["score"][period == seen] = blend
            target["baseline_vol"][period == seen] = scores[0][1]
    return output


def train_once(data, eval_data, tr, va, seed, half_life, device, steps, checkpoint, tails):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = matrix(data, "combined")[tr]
    xx = matrix(eval_data, "combined")[va]
    mean, std = x.mean(0), x.std(0)
    std = np.where(std > 1e-8, std, 1)
    train_x, test_x = (torch.as_tensor(np.clip((values - mean) / std, -5, 5), device=device)
                       for values in (x, xx))
    y = data["y"][tr]
    sigma = data["volatility"][tr]
    base_atr = np.log(np.maximum(data["atr"][tr], 1e-8))
    test_atr = np.log(np.maximum(eval_data["atr"][va], 1e-8))
    mu, sd = float(base_atr.mean()), float(max(base_atr.std(), 1e-8))
    atr, atr_test = (base_atr - mu) / sd, (test_atr - mu) / sd
    recent = recency(data["time"][tr], half_life)
    # Fit the volatility reference and the classification prior on earlier entries.
    a_mean, vol_mean = float(recent @ atr), float(recent @ sigma)
    a_centered, vol_centered = atr - a_mean, sigma - vol_mean
    slope = float(recent @ (a_centered * vol_centered) / (recent @ (a_centered ** 2)))
    offset = vol_mean - slope * a_mean
    train_vol_base, eval_vol_base = offset + slope * atr, offset + slope * atr_test
    sigma_target = torch.as_tensor(sigma - train_vol_base, device=device)
    win_target = torch.as_tensor(y, dtype=torch.float32, device=device)
    prior = float(recent @ y)
    base_scores = torch.as_tensor(0.45 * atr + np.log(prior / (1 - prior)), device=device)

    barrier = "barrier_target" in data
    if barrier:
        first_touch = torch.as_tensor(data["barrier_target"][tr], device=device)
        valid_touch = torch.as_tensor(data["barrier_valid"][tr], device=device)
    model = VolatilityMLP(x.shape[1], 3 if barrier else 2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.03)
    result = {}
    for i in range(1, max(steps) + 1):
        model.train()
        rows = rng.choice(len(y), size=512, p=recent)
        forecast = model(train_x[rows])
        if tails:
            # Pinball loss estimates the upper conditional 85th percentile.
            error = sigma_target[rows] - forecast[:, 1]
            pinball = torch.maximum(.85 * error, -.15 * error).mean()
            loss = F.smooth_l1_loss(forecast[:, 0], sigma_target[rows]) + pinball
        else:
            loss = (F.binary_cross_entropy_with_logits(base_scores[rows] + forecast[:, 0], win_target[rows])
                    + F.smooth_l1_loss(forecast[:, 1], sigma_target[rows]))
            if barrier:
                group = valid_touch[rows]
                per_row = F.binary_cross_entropy_with_logits(forecast[:, 2], first_touch[rows], reduction="none")
                loss = loss + .5 * (per_row * group).sum() / group.sum().clamp_min(1)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if i in steps:
            model.eval()
            with torch.inference_mode():
                pred = model(test_x).cpu().numpy()
            if tails:
                result[i] = np.column_stack((eval_vol_base + pred[:, 0], eval_vol_base + pred[:, 1],
                                             eval_vol_base + pred.mean(1)))
            elif barrier:
                vol = eval_vol_base + pred[:, 1]
                trade = .45 * atr_test + pred[:, 0]
                both = .45 * atr_test + .5 * pred[:, 0] + .5 * pred[:, 2]
                result[i] = np.column_stack((vol, trade, pred[:, 2], both))
            else:
                vol = eval_vol_base + pred[:, 1]
                trade = 0.45 * atr_test + pred[:, 0]
                both = 0.45 * atr_test + 0.5 * pred[:, 0] + 0.5 * pred[:, 1]
                result[i] = np.column_stack((vol, trade, both))
    if checkpoint:
        torch.save({"model_state": model.state_dict(), "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
                    "atr_mean": mu, "atr_std": sd, "vol_offset": offset, "vol_slope": slope,
                    "steps": max(steps), "seed": seed, "half_life": half_life, "tails": tails,
                    "targets": 3 if barrier else 2}, HERE / checkpoint)
    return result, eval_vol_base


def metrics(predictions, frame, ix, modes):
    scores = predictions["score"]
    target = frame["volatility"][ix]
    result = {name: evaluate(scores[:, col], frame["y"][ix], frame["time"][ix])
              for col, name in enumerate(modes)}
    result["volatility_mae"] = float(np.mean(abs(scores[:, 0] - target)))
    result["atr_volatility_mae"] = float(np.mean(abs(predictions["baseline_vol"] - target)))
    if "first_touch" in modes:
        valid = frame["barrier_valid"][ix]
        labels = frame["barrier_target"][ix][valid]
        s = scores[:, modes.index("first_touch")][valid]
        _, bins = np.unique(s, return_inverse=True)
        positives = np.bincount(bins, weights=labels)
        negatives = np.bincount(bins, weights=1 - labels)
        result["first_touch_auc"] = float(positives @ (negatives.cumsum() - negatives / 2)
                                          / (positives.sum() * negatives.sum()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes_parser = parser.add_mutually_exclusive_group()
    modes_parser.add_argument("--tails", action="store_true", help="Predict mean and upper-quantile future volatility")
    modes_parser.add_argument("--barriers", action="store_true", help="Learn first touch as an auxiliary target")
    args = parser.parse_args()
    modes = TAIL_MODES if args.tails else BARRIER_MODES if args.barriers else MODES
    name = "volatility_tail_mlp" if args.tails else "barrier_mlp" if args.barriers else "volatility_mlp"
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = "barrier_features.npz" if args.barriers else "volatility_features.npz"
    data = view(HERE / source)
    development = defaultdict(list)
    for bounds in ((0.40, 0.55), (0.55, 0.70), (0.70, 0.85)):
        _, ix = fixed_fold(data, *bounds)
        for half_life in (45, 180):
            scores = run_fold(data, data, ix, half_life, device, tails=args.tails)
            for step, p in scores.items():
                development[half_life, step].append(metrics(p, data, ix, modes))
        print("Volatility MLP development fold", bounds, flush=True)
    selected = max(((half, step, mode) for half, step in development for mode in modes),
                   key=lambda k: sum(r[k[2]]["wins"] for r in development[k[:2]]))
    half, step, mode = selected
    print("Volatility MLP chosen from development", selected, flush=True)
    _, ix = fixed_fold(data, 0.85, 1.0)
    historic = run_fold(data, data, ix, half, device, tails=args.tails)
    fresh = view(HERE / f"fresh_{source}")
    later = np.arange(len(fresh["y"]))
    fresh_result = run_fold(data, fresh, later, half, device, (step,), name, tails=args.tails)
    results = {
        "recipe": list(selected), "development": development[half, step],
        "historical_atr": evaluate(data["atr"][ix], data["y"][ix], data["time"][ix]),
        "history": {str(i): metrics(pred, data, ix, modes) for i, pred in historic.items()},
        "fresh_atr": evaluate(fresh["atr"], fresh["y"], fresh["time"]),
        "fresh": metrics(fresh_result[step], fresh, later, modes),
    }
    (HERE / f"{name}_results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for key in ("historical_atr", "fresh_atr"):
        print(key, results[key]["wins"], results[key]["calls"])
    for i, r in results["history"].items():
        print(i, "test wins", [(key, r[key]["wins"]) for key in modes],
              "forecast MAE", r["volatility_mae"], "ATR MAE", r["atr_volatility_mae"])
    print("fresh", [(key, results["fresh"][key]["wins"]) for key in modes], flush=True)


if __name__ == "__main__":
    main()
