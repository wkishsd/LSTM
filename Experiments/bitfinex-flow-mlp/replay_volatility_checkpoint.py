"""Reproduce the new-period next-hour ETH volatility predictions without training."""

import argparse
from pathlib import Path

import numpy as np
import torch

from train_flow import HERE, SEEDS, WEEK, matrix, view
from volatility_mlp import VolatilityMLP


def predict(frame, checkpoint_folder):
    features = matrix(frame, "combined")
    log_atr = np.log(np.maximum(frame["atr"], 1e-8))
    blocks = (frame["time"] + 3 * 86400) // (4 * WEEK)
    output = np.empty(len(features), dtype=np.float32)
    for block in np.unique(blocks):
        use = blocks == block
        forecasts = []
        for seed in SEEDS:
            path = checkpoint_folder / f"volatility_verified_fresh_block{block}_seed{seed}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            state = checkpoint["model_state"]
            model = VolatilityMLP(features.shape[1], targets=len(state["net.5.bias"]))
            model.load_state_dict(state)
            model.eval()
            x = (torch.from_numpy(features[use]) - checkpoint["mean"]) / checkpoint["std"]
            a = (log_atr[use] - checkpoint["atr_mean"]) / checkpoint["atr_std"]
            baseline = checkpoint["vol_offset"] + checkpoint["vol_slope"] * a
            with torch.inference_mode():
                forecasts.append(baseline + model(x.clamp(-5, 5))[:, 1].numpy())
        output[use] = np.mean(forecasts, axis=0)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=HERE / "fresh_volatility_features.npz")
    parser.add_argument("--checkpoints", type=Path, default=HERE)
    parser.add_argument("--verify-against", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    data = view(args.features)
    forecasts = predict(data, args.checkpoints)
    if args.verify_against:
        reference = view(args.verify_against)
        np.testing.assert_array_equal(data["time"], reference["time"])
        np.testing.assert_allclose(forecasts, reference["model_log_vol"], atol=1e-5, rtol=0)
        print(f"Reproduced {len(forecasts)} saved forecasts on CPU.")
    if "volatility" in data:
        print(f"Log-volatility MAE {abs(forecasts - data['volatility']).mean():.6f}")
    if args.output:
        np.savez(args.output, time=data["time"], model_log_vol=forecasts)


if __name__ == "__main__":
    main()
