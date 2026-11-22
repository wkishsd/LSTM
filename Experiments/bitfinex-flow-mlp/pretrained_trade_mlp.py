"""Test supervised transfer from dense future-price scenarios to the sparse trade labels."""

from collections import defaultdict
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from train_flow import HERE, SEEDS, WEEK, evaluate, fixed_fold, matrix, recency, view


STEPS = (40, 160, 400)


def fit(source, x, y, available, times, xa, atr, eval_atr, cutoff, device, seed, steps, save):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    prior_scenarios = np.searchsorted(source["available"], cutoff, side="left")
    dense = source["x"][:prior_scenarios]
    mu, sd = dense.mean(0), dense.std(0)
    sd = np.where(sd > 1e-8, sd, 1)

    def tensor(v):
        return torch.as_tensor(np.clip((v - mu) / sd, -5, 5), device=device)

    observations, test_x = tensor(x), tensor(xa)
    pretrain_x = tensor(dense)
    scenarios = source["target"][:prior_scenarios]
    avg, std = scenarios.mean(0), scenarios.std(0)
    pretrain_y = torch.as_tensor(np.clip((scenarios - avg) / np.maximum(std, 1e-5), -5, 5), device=device)
    p = recency(times, 180)
    how_recent = recency(source["time"][:prior_scenarios], 180)
    assert available.max() < cutoff
    train_y = torch.as_tensor(y, dtype=torch.float32, device=device)
    a = np.log(np.maximum(atr, 1e-8))
    a_mu, a_std = float(a.mean()), float(max(a.std(), 1e-8))
    anchor = .45 * (a - a_mu) / a_std
    test_anchor = .45 * (np.log(np.maximum(eval_atr, 1e-8)) - a_mu) / a_std
    prior = float(p @ y)
    offset = torch.as_tensor(anchor + np.log(prior / (1 - prior)), device=device)
    body = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, 32), nn.GELU()).to(device)
    scene = nn.Linear(32, pretrain_y.shape[1]).to(device)
    head = nn.Linear(32, 1).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    random_start = {key: value.clone() for key, value in body.state_dict().items()}
    trainer = torch.optim.AdamW([*body.parameters(), *scene.parameters()], lr=.001, weight_decay=.03)
    for _ in range(1000):
        i = rng.choice(prior_scenarios, 512, p=how_recent)
        loss = F.smooth_l1_loss(scene(body(pretrain_x[i])), pretrain_y[i])
        trainer.zero_grad(set_to_none=True)
        loss.backward()
        trainer.step()

    predictions = {}
    for method in ("pretrained", "from_scratch"):
        if method == "from_scratch":
            body.load_state_dict(random_start)
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        rng = np.random.default_rng(seed)
        trainer = torch.optim.AdamW([{"params": body.parameters(), "lr": .0003}, {"params": head.parameters()}],
                                    lr=.001, weight_decay=.03)
        for step in range(1, max(steps) + 1):
            i = rng.choice(len(y), 512, p=p)
            residual = head(F.dropout(body(observations[i]), p=.10)).flatten()
            loss = F.binary_cross_entropy_with_logits(offset[i] + residual, train_y[i])
            trainer.zero_grad(set_to_none=True)
            loss.backward()
            trainer.step()
            if step in steps:
                with torch.inference_mode():
                    delta = head(body(test_x)).flatten().cpu().numpy()
                predictions[method, step] = test_anchor + .35 * delta
            if save and method == "pretrained" and step == save[0]:
                torch.save({"body": body.state_dict(), "head": head.state_dict(), "mean": torch.from_numpy(mu),
                            "std": torch.from_numpy(sd), "atr_mu": a_mu, "atr_std": a_std,
                            "atr_weight": .45, "residual_weight": .35, "steps": step}, save[1])
    return predictions


def run(dense, data, dest, ix, device, steps=STEPS, save_step=None):
    x, xx = matrix(data, "combined"), matrix(dest, "combined")
    weeks = (dest["time"][ix] + 3 * 86400) // (4 * WEEK)
    result = {}
    for b in np.unique(weeks):
        mask = weeks == b
        test = ix[mask]
        cutoff = dest["start"][test].min()
        tr = np.flatnonzero(data["available"] < cutoff)
        fits = [fit(dense, x[tr], data["y"][tr], data["available"][tr], data["time"][tr], xx[test],
                    data["atr"][tr], dest["atr"][test], cutoff, device, seed, steps,
                    (save_step, HERE / f"pretrained_trade_fresh_block{b}_seed{seed}.pt")
                    if save_step else None) for seed in SEEDS]
        for key in fits[0]:
            result.setdefault(key, np.empty(len(ix)))[mask] = np.mean([r[key] for r in fits], axis=0)
    return result


def main():
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dense = view(HERE / "dense_scenarios.npz")
    data = view(HERE / "features.npz")
    dev = defaultdict(list)
    for bounds in ((.40, .55), (.55, .70), (.70, .85)):
        _, ix = fixed_fold(data, *bounds)
        for key, score in run(dense, data, data, ix, device).items():
            dev[key].append(evaluate(score, data["y"][ix], data["time"][ix]))
        print("Transfer MLP development fold", bounds, flush=True)
    selected = {name: max(STEPS, key=lambda step: sum(r["wins"] for r in dev[name, step]))
                for name in ("pretrained", "from_scratch")}
    print("Transfer recipes chosen on development", selected, flush=True)
    _, test = fixed_fold(data, .85, 1)
    fresh = view(HERE / "fresh_features.npz")
    result = {"recipe": selected, "development": {name: dev[name, step] for name, step in selected.items()}}
    for name, frame, ix in (("historical", data, test), ("fresh", fresh, np.arange(len(fresh["time"])))):
        y, t = frame["y"][ix], frame["time"][ix]
        compare = run(dense, data, frame, ix, device, tuple(sorted(set(selected.values()))),
                      save_step=selected["pretrained"] if name == "fresh" else None)
        result[name] = {method: evaluate(compare[method, step], y, t) for method, step in selected.items()}
        result[name]["atr"] = evaluate(frame["atr"][ix], y, t)
        print(name, {k: (r["wins"], r["calls"], r["ap"]) for k, r in result[name].items()}, flush=True)
    (HERE / "pretrained_trade_mlp_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
