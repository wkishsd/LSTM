"""Train compact neural rankers and choose a recipe on expanding development folds."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
WEEK = 604800
QUOTA = 1000 / 15728
STEPS = (40, 120, 240, 480)
SEEDS = (42, 12)
MODES = ("base", "flow", "combined")
SHRINK = (0.25, 0.5, 1.0)


class ResidualMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, x):
        return self.layers(x).flatten()


def ap(scores, target):
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], target[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    tp = y.cumsum()[ends]
    return float(np.diff(tp, prepend=0) @ (tp / (ends + 1)) / target.sum())


def evaluate(scores, y, t):
    groups = (t + 3 * 86400) // WEEK
    weeks = {}
    for group in np.unique(groups):
        ix = np.flatnonzero(groups == group)
        k = round(QUOTA * len(ix))
        take = ix[np.argsort(-scores[ix], kind="stable")[:k]]
        weeks[str(int(group))] = [int(y[take].sum()), k]
    return {"wins": sum(w[0] for w in weeks.values()), "calls": sum(w[1] for w in weeks.values()),
            "ap": ap(scores, y), "rate": float(y.mean()), "weeks": weeks}


def matrix(data, mode):
    if mode == "combined":
        return np.column_stack((data["base"], data["flow"]))
    if mode == "combined_futures":
        return np.column_stack((data["base"], data["flow"], data["futures"]))
    if mode == "feedback":
        return np.column_stack((data["base"], data["flow"], data["feedback"]))
    if mode == "geometry":
        return np.column_stack((data["base"], data["flow"], data["geometry"]))
    return data[mode]


def recency(times, half_life):
    age = times.max() - times
    weight = np.exp2(-age / (half_life * 86400))
    return weight / weight.sum()


def pair_sampler(times, target, half_life):
    weeks = (times + 3 * 86400) // WEEK
    pos, neg, stamp = [], [], []
    for week in np.unique(weeks):
        ix = np.flatnonzero(weeks == week)
        a, b = ix[target[ix] == 1], ix[target[ix] == 0]
        if len(a) and len(b):
            pos.append(a)
            neg.append(b)
            stamp.append(times[ix[-1]])
    positive, negative = np.concatenate(pos), np.concatenate(neg)
    n_pos, n_neg = np.array(list(map(len, pos))), np.array(list(map(len, neg)))
    start_pos, start_neg = np.cumsum(n_pos) - n_pos, np.cumsum(n_neg) - n_neg
    weights = recency(np.array(stamp), half_life)

    def sample(rng, size):
        group = rng.choice(len(pos), size=size, p=weights)
        a = positive[start_pos[group] + (rng.random(size) * n_pos[group]).astype(np.int64)]
        b = negative[start_neg[group] + (rng.random(size) * n_neg[group]).astype(np.int64)]
        return a, b
    return sample


def train(data, train_ix, evaluate_data, evaluate_ix, mode, objective, seed, device, steps=STEPS,
          checkpoint=None, half_life=180, sample_weights=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    x = matrix(data, mode)[train_ix]
    test = matrix(evaluate_data, mode)[evaluate_ix]
    center, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-8, scale, 1)
    x = torch.as_tensor(np.clip((x - center) / scale, -5, 5), device=device)
    xx = torch.as_tensor(np.clip((test - center) / scale, -5, 5), device=device)
    t, y = data["time"][train_ix], data["y"][train_ix]
    train_weights = recency(t, half_life)
    if sample_weights is not None:
        if objective != "bce":
            raise ValueError("Sample weights require the BCE objective")
        train_weights *= sample_weights
        train_weights /= train_weights.sum()
    if objective == "pair":
        pairs = pair_sampler(t, y, half_life)
    target = torch.as_tensor(y, dtype=torch.float32, device=device)
    atr = np.log(np.maximum(data["atr"][train_ix], 1e-8))
    test_atr = np.log(np.maximum(evaluate_data["atr"][evaluate_ix], 1e-8))
    atr_mean, atr_scale = float(atr.mean()), float(max(atr.std(), 1e-8))
    atr_weight = 0.45
    anchor = torch.as_tensor(atr_weight * (atr - atr_mean) / atr_scale, device=device)
    anchor_test = atr_weight * (test_atr - atr_mean) / atr_scale
    prior = float(train_weights @ y)
    bias = np.log(prior / (1 - prior))
    model = ResidualMLP(len(center)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.03)
    predictions = {}
    for step in range(1, max(steps) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        if objective == "pair":
            a, b = pairs(rng, 256)
            ix = np.concatenate((a, b))
            delta = model(x[ix])
            margin = delta + anchor[ix]
            loss = F.softplus(margin[256:] - margin[:256]).mean()
        else:
            ix = rng.choice(len(y), size=512, p=train_weights)
            delta = model(x[ix])
            loss = F.binary_cross_entropy_with_logits(delta + anchor[ix] + bias, target[ix])
        loss = loss + 0.05 * delta.square().mean()
        loss.backward()
        opt.step()
        if step in steps:
            model.eval()
            with torch.inference_mode():
                predictions[step] = model(xx).cpu().numpy()
    if checkpoint:
        torch.save({
            "model_state": model.state_dict(), "feature_mode": mode, "objective": objective,
            "mean": torch.from_numpy(center), "scale": torch.from_numpy(scale),
            "atr_mean": atr_mean, "atr_scale": atr_scale, "atr_weight": atr_weight,
            "steps": max(steps), "seed": seed, "half_life_days": half_life,
        }, checkpoint)
    return anchor_test, predictions


def fixed_fold(data, first, last):
    times = data["time"]
    start = times[int(len(times) * first)]
    end = times[int(len(times) * last)] if last < 1 else times[-1] + 1
    test = np.flatnonzero((times >= start) & (times < end))
    train = np.flatnonzero(np.maximum(times, data["available"]) < data["start"][test].min())
    return train, test


def view(path):
    with np.load(path, allow_pickle=False) as source:
        return {k: source[k] for k in source.files}


def main():
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = view(HERE / "features.npz")
    dev_periods = ((0.40, 0.55), (0.55, 0.70), (0.70, 0.85))
    recipes = defaultdict(list)
    baselines = []
    for bounds in dev_periods:
        tr, va = fixed_fold(data, *bounds)
        reference = evaluate(data["atr"][va], data["y"][va], data["time"][va])
        baselines.append(reference)
        for mode in MODES:
            for objective in ("pair", "binary"):
                models = [train(data, tr, data, va, mode, objective, seed, device) for seed in SEEDS]
                anchor = models[0][0]
                for step in STEPS:
                    residual = np.mean([scores[step] for _, scores in models], axis=0)
                    for weight in SHRINK:
                        s = anchor + weight * residual
                        r = evaluate(s, data["y"][va], data["time"][va])
                        recipes[mode, objective, step, weight].append(r)
        print("development fold", bounds, "ATR wins", reference["wins"], "/", reference["calls"], flush=True)
    for recipe, folds in sorted(recipes.items(), key=lambda item: -sum(r["wins"] for r in item[1]))[:10]:
        print(recipe, [r["wins"] for r in folds], flush=True)
    winner = max(recipes, key=lambda recipe: (sum(r["wins"] for r in recipes[recipe]),
                                             sum(r["ap"] for r in recipes[recipe])))
    mode, objective, step, weight = winner
    print("frozen recipe", winner, flush=True)
    tr, test = fixed_fold(data, 0.85, 1.0)
    models = [train(data, tr, data, test, mode, objective, seed, device, (step,),
                    HERE / f"selected_seed{seed}.pt") for seed in SEEDS]
    pred = models[0][0] + weight * np.mean([scores[step] for _, scores in models], axis=0)
    y, times, atr = data["y"][test], data["time"][test], data["atr"][test]
    result = {"recipe": list(winner), "fit_rows": len(tr),
              "seeds": list(SEEDS), "development_atr": baselines, "development_mlp": recipes[winner],
              "development_grid": [{"recipe": list(key), "folds": folds} for key, folds in recipes.items()],
              "test_atr": evaluate(atr, y, times), "test_mlp": evaluate(pred, y, times)}
    (HERE / "locked_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    np.savez(HERE / "locked_predictions.npz", score=pred, atr=atr, time=times, labels=y, ids=data["ids"][test])
    for key in ("test_atr", "test_mlp"):
        stats = result[key]
        print(key, {k: v for k, v in stats.items() if k != "weeks"}, flush=True)


if __name__ == "__main__":
    main()
