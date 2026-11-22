"""Run already-selected recipes on a separate 2023 price regime."""

from datetime import datetime, timezone
import json

import numpy as np
import torch

from features import load_exchange
from tail_pair_mlp import make_gate, walk
from train_flow import HERE, evaluate, fixed_fold, view
from verify_volatility import compare
from volatility_mlp import run_fold
from volatility_targets import label_volatility
from walk_forward_flow import only_selected


def main():
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = view(HERE / "earlier_features.npz")
    reference = view(HERE / "features.npz")
    assert data["time"].max() < reference["start"].min()
    del reference
    label_volatility(data, load_exchange("ETHUSDT", earlier=True))
    _, ix = fixed_fold(data, .45, 1.)
    times, y = data["time"][ix], data["y"][ix]
    outputs = {"atr": evaluate(data["atr"][ix], y, times)}
    frozen = json.loads((HERE / "walk_forward_results.json").read_text(encoding="utf-8"))["recipe"]
    rank_recipe = (frozen["half_life_days"], frozen["updates"], frozen["residual_weight"])
    ranks = only_selected(data, data, ix, frozen["feature_mode"], frozen["objective"], frozen["seeds"],
                          rank_recipe, device)
    outputs["cross_venue_pair_mlp"] = evaluate(ranks, y, times)
    earlier = {key: arr[:0] for key, arr in data.items()}
    make_gate(data, earlier)
    candidate = json.loads((HERE / "tail_pair_mlp_results.json").read_text(encoding="utf-8"))["recipe"]
    half, updates, weight = candidate
    ranks = walk(data, data, ix, half, device, (updates,), (weight,))[updates, weight]
    outputs["tail_pair_mlp"] = evaluate(ranks, y, times)
    drift = json.loads((HERE / "volatility_mlp_results.json").read_text(encoding="utf-8"))["recipe"]
    half, step, _ = drift
    forecast = run_fold(data, data, ix, half, device, (20, 120, step))
    outputs["volatility_mlp_trade"] = evaluate(forecast[step]["score"][:, 1], y, times)
    result = {"test_count": len(ix), "trades": outputs,
              "volatility_mae": {str(i): compare(p, data["volatility"][ix], times) for i, p in forecast.items()},
              "test_from_utc": datetime.fromtimestamp(int(times[0]), timezone.utc).isoformat(),
              "test_to_utc": datetime.fromtimestamp(int(times[-1]), timezone.utc).isoformat()}
    (HERE / "earlier_history_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(result["test_from_utc"], result["test_to_utc"], result["test_count"])
    print({key: (r["wins"], r["calls"], r["ap"]) for key, r in outputs.items()})
    print(result["volatility_mae"], flush=True)


if __name__ == "__main__":
    main()
