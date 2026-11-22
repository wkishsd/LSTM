"""Replay the Bitfinex trade-target MLP on its separate 2023 evaluation period."""

import json

import numpy as np
import torch

from train_flow import HERE, SEEDS, WEEK, ResidualMLP, evaluate, view


def main():
    torch.set_num_threads(3)
    data = view(HERE / "earlier_trade_replay.npz")
    results = json.loads((HERE / "earlier_trade_curve.json").read_text(encoding="utf-8"))
    _, steps, weight = results["recipe"]
    t, x, a, gate = (data[key] for key in ("time", "features", "atr", "gate"))
    blocks = (t + 3 * 86400) // (4 * WEEK)
    scores = np.empty(len(t))
    for b in np.unique(blocks):
        mask = blocks == b
        forecast = []
        for seed in SEEDS:
            path = HERE / f"tail_pair_2023_block{b}_seed{seed}.pt"
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            model = ResidualMLP(x.shape[1])
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            inp = (torch.from_numpy(x[mask]) - ckpt["mean"]) / ckpt["scale"]
            with torch.inference_mode():
                forecast.append(model(inp.clamp(-5, 5)).numpy())
        reference = ckpt["atr_weight"] * ((np.log(np.maximum(a[mask], 1e-8)) - ckpt["atr_mean"]) / ckpt["atr_scale"])
        scores[mask] = reference + weight * gate[mask] * np.clip(np.mean(forecast, axis=0), -1, 1)
    np.testing.assert_allclose(scores, data["reference"], rtol=0, atol=1e-5)
    metrics = evaluate(scores, data["label"], t)
    assert metrics["weeks"] == results["steps"][str(steps)]["weeks"]
    print(f"Reproduced {len(t):,} original target scores on CPU.")
    print(f"Selected {metrics['wins']} positive targets at {metrics['calls']} matched calls.")


if __name__ == "__main__":
    main()
