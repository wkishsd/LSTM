"""Reproduce the forecast curve and evaluate errors on paired time blocks."""

import json

import numpy as np
import torch

from train_flow import HERE, WEEK, fixed_fold, view
from volatility_mlp import run_fold


def compare(prediction, target, time):
    model = abs(prediction["score"][:, 0] - target)
    reference = abs(prediction["baseline_vol"] - target)
    weeks = (time + 3 * 86400) // WEEK
    group = np.unique(weeks)
    difference = np.array([np.sum((model - reference)[weeks == w]) for w in group])
    count = np.array([np.sum(weeks == w) for w in group])
    interval = None
    if len(group) >= 8:
        rng = np.random.default_rng(42)
        starts = rng.integers(len(group), size=(4000, (len(group) + 3) // 4, 1))
        ix = ((starts + np.arange(4)) % len(group)).reshape(4000, -1)[:, :len(group)]
        interval = np.quantile(difference[ix].sum(1) / count[ix].sum(1), [.025, .975]).tolist()
    return {"mae": float(model.mean()), "atr_mae": float(reference.mean()),
            "better_weeks": int((difference < 0).sum()), "weeks": len(group),
            "paired_four_week_mae_difference_interval": interval}


def main():
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = view(HERE / "volatility_features.npz")
    fresh = view(HERE / "fresh_volatility_features.npz")
    recipe = json.loads((HERE / "volatility_mlp_results.json").read_text(encoding="utf-8"))
    half_life, selected_step, _ = recipe["recipe"]
    _, test = fixed_fold(source, .85, 1.)
    result = {}
    for part, data, ix in (("historical", source, test), ("fresh", fresh, np.arange(len(fresh["time"])))):
        checkpoints = f"volatility_verified_{part}" if part == "fresh" else None
        forecasts = run_fold(source, data, ix, half_life, device, (20, 120, selected_step), checkpoints)
        result[part] = {str(step): compare(scores, data["volatility"][ix], data["time"][ix])
                        for step, scores in forecasts.items()}
        result[part]["count"] = len(ix)
        expected = recipe["history"][str(selected_step)] if part == "historical" else recipe["fresh"]
        np.testing.assert_allclose(result[part][str(selected_step)]["mae"], expected["volatility_mae"], atol=1e-5)
        np.savez(HERE / f"verified_volatility_{part}.npz", entry=data["ids"][ix], time=data["time"][ix],
                 log_vol=data["volatility"][ix], model_log_vol=forecasts[selected_step]["score"][:, 0],
                 atr_log_vol=forecasts[selected_step]["baseline_vol"])
        print(part, result[part], flush=True)
    (HERE / "volatility_verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
