"""Measure the fixed trade-ranker recipe throughout training on the separate 2023 test."""

import json

import numpy as np
import torch

from tail_pair_mlp import make_gate, walk
from train_flow import HERE, evaluate, fixed_fold, view


def paired_weeks(scores, reference):
    rows = list(reference["weeks"])
    differences = np.array([scores["weeks"][key][0] - reference["weeks"][key][0] for key in rows])
    count = np.array([reference["weeks"][key][1] for key in rows])
    rng = np.random.default_rng(42)
    chunks = rng.integers(len(rows), size=(4000, (len(rows) + 3) // 4, 1))
    ix = ((chunks + np.arange(4)) % len(rows)).reshape(4000, -1)[:, :len(rows)]
    differences_pct = 100 * differences[ix].sum(1) / count[ix].sum(1)
    return {"better_weeks": int((differences > 0).sum()), "worse_weeks": int((differences < 0).sum()),
            "tied_weeks": int((differences == 0).sum()),
            "paired_four_week_precision_diff_percent_points": np.quantile(differences_pct, [.025, .975]).tolist()}


def main():
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = view(HERE / "earlier_features.npz")
    make_gate(data, {k: v[:0] for k, v in data.items()})
    recipe = json.loads((HERE / "tail_pair_mlp_results.json").read_text(encoding="utf-8"))["recipe"]
    half_life, chosen, weight = recipe
    steps = (20, 40, 120, 240, chosen)
    _, ix = fixed_fold(data, .45, 1)
    time, y, atr = (data[key][ix] for key in ("time", "y", "atr"))
    predictions = walk(data, data, ix, half_life, device, steps, (weight,), "tail_pair_2023")
    baseline = evaluate(atr, y, time)
    results = {"recipe": recipe, "atr": baseline, "steps": {}}
    for (step, _), scores in predictions.items():
        result = evaluate(scores, y, time)
        result.update(paired_weeks(result, baseline))
        results["steps"][str(step)] = result
        print(step, result["wins"], result["calls"], result["ap"],
              result["paired_four_week_precision_diff_percent_points"], flush=True)
    pinned = json.loads((HERE / "earlier_history_results.json").read_text(encoding="utf-8"))["trades"]["tail_pair_mlp"]
    assert results["steps"][str(chosen)]["weeks"] == pinned["weeks"]
    (HERE / "earlier_trade_curve.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    np.savez(HERE / "earlier_trade_predictions.npz", time=time, label=y, atr=atr, entry=data["ids"][ix],
             **{f"score_{step}": pred for (step, _), pred in predictions.items()})


if __name__ == "__main__":
    main()
