"""Verify the saved MLP and check its frozen recipe on the later Bitfinex slice."""

import json
from pathlib import Path

import numpy as np
import torch

from train_flow import HERE, ResidualMLP, evaluate, fixed_fold, matrix, train, view


def load_and_score(paths, data, ix, mode, weight, device):
    raw = matrix(data, mode)[ix]
    residuals = []
    for path in paths:
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        model = ResidualMLP(raw.shape[1]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        center, scale = checkpoint["mean"].numpy(), checkpoint["scale"].numpy()
        x = np.clip((raw - center) / scale, -5, 5)
        with torch.inference_mode():
            residuals.append(model(torch.as_tensor(x, device=device)).cpu().numpy())
    atr = np.log(np.maximum(data["atr"][ix], 1e-8))
    anchor = checkpoint["atr_weight"] * (atr - checkpoint["atr_mean"]) / checkpoint["atr_scale"]
    return anchor + weight * np.mean(residuals, axis=0)


def main():
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    recipe = json.loads((HERE / "locked_results.json").read_text(encoding="utf-8"))
    mode, objective, step, weight = recipe["recipe"]
    seeds = recipe["seeds"]
    files = [HERE / f"selected_seed{seed}.pt" for seed in seeds]
    history = view(HERE / "features.npz")
    fresh = view(HERE / "fresh_features.npz")
    old_train, old_test = fixed_fold(history, 0.85, 1.0)
    old = load_and_score(files, history, old_test, mode, weight, device)
    expected = view(HERE / "locked_predictions.npz")
    assert np.array_equal(history["ids"][old_test], expected["ids"])
    np.testing.assert_allclose(old, expected["score"], rtol=1e-5, atol=1e-6)
    fresh_ix = np.arange(len(fresh["time"]))
    new_train = np.flatnonzero(
        np.maximum(history["time"], history["available"]) < fresh["start"].min()
    )
    assert history["time"].max() < fresh["time"].min()
    assert len(new_train) > len(old_train)
    earlier = load_and_score(files, fresh, fresh_ix, mode, weight, device)
    fit = [train(history, new_train, fresh, fresh_ix, mode, objective, seed, device,
                 (step,), HERE / f"fresh_recipe_seed{seed}.pt") for seed in seeds]
    updated = fit[0][0] + weight * np.mean([pred[step] for _, pred in fit], axis=0)
    result = {"recipe": recipe["recipe"], "count": len(fresh_ix), "updated_training_count": len(new_train)}
    for k, s in (("atr", fresh["atr"]), ("locked_mlp", earlier), ("refit_mlp", updated)):
        result[k] = evaluate(s, fresh["y"], fresh["time"])
    (HERE / "fresh_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
