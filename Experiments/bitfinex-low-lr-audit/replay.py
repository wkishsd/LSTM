"""Reproduce the low-learning-rate run's entry counts and test a different starting bias."""

from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Training" / "rocm-9070"))

from mlp_trades.bitfinex import export_paths, prepare_bitfinex
from mlp_trades.engine import cache_timeline, fit_feature_scaling, load_model, make_loader, run_epoch, seed_everything
from mlp_trades.metrics import average_precision
from mlp_trades.model import TradeMLP
from mlp_trades.split import training_split


def summary(model, loader, labels):
    scores = []
    model.eval()
    with torch.inference_mode():
        for x, _ in loader:
            logits = model(x)
            scores.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    margin = np.concatenate(scores)
    probs = 1 / (1 + np.exp(-margin))
    calls = margin > 0
    return {"calls": int(calls.sum()), "count": len(labels), "winners": int(labels[calls].sum()),
            "average_precision": average_precision(labels, margin),
            "margin_quantiles": np.quantile(margin, [0, .1, .5, .9, 1]).tolist(),
            "probability_quantiles": np.quantile(probs, [0, .1, .5, .9, 1]).tolist(),
            "loss": float(np.logaddexp(0, -np.where(labels, margin, -margin)).mean()),
            "head_bias_difference": float((model.layers[-1].bias[1] - model.layers[-1].bias[0]).detach().cpu())}


def replay(data, splits, cfg, architecture, device, windows, prior, target_epochs):
    seed_everything(cfg.seed)
    model = TradeMLP(**architecture)
    fit_feature_scaling(model, data, splits["train"])
    if prior:
        rate = float(data.labels[splits["train"]].mean())
        # Leave every weight unchanged. Replace only the two random output biases.
        with torch.no_grad():
            model.layers[-1].bias.copy_(torch.tensor([np.log1p(-rate), np.log(rate)]))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    train = make_loader(data, splits["train"], cfg, device, shuffle=True, windows=windows)
    validation = make_loader(data, splits["validation"], cfg, device, batch_size=512, windows=windows)
    labels = data.labels[splits["validation"]]
    snapshots = {"0": summary(model, validation, labels)}
    initial_weights = [p.detach().clone() for p in model.parameters()]
    for epoch in range(1, max(target_epochs) + 1):
        run_epoch(model, train, device, optimizer)
        if epoch in target_epochs:
            snapshots[str(epoch)] = summary(model, validation, labels)
    snapshots["max_parameter_change"] = max(float((p.detach() - base).abs().max().cpu())
                                              for p, base in zip(model.parameters(), initial_weights))
    return snapshots


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(3)
    metadata = json.loads((args.run / "run.json").read_text(encoding="utf-8"))
    cfg = Namespace(**metadata["config"])
    data = prepare_bitfinex(export_paths(cfg.bitfinex_dir), training=True)
    splits, _ = training_split(data, cfg.train_fraction, cfg.validation_fraction)
    device = torch.device("cuda:0")
    last, checkpoint = load_model(args.run / "last.pt", device, data.schema)
    architecture = checkpoint["architecture"]
    windows = cache_timeline(data, cfg, device, cfg.batch_size)
    assert windows is not None
    val = make_loader(data, splits["validation"], cfg, device, batch_size=512, windows=windows)
    y = data.labels[splits["validation"]]
    steps = sorted({int(state["step"]) for state in checkpoint["optimizer_state"]["state"].values()})
    history = [json.loads(line) for line in (args.run / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    result = {"updates_in_last_checkpoint": steps, "training_prior": float(data.labels[splits["train"]].mean()),
              "validation_prior": float(y.mean()), "original_last": summary(last, val, y)}
    assert result["original_last"]["calls"] == history[-1]["validation"]["allowed"]
    epochs = (1, 4, 12)
    result["original_replay"] = replay(data, splits, cfg, architecture, device, windows, False, epochs)
    for ep in epochs:
        assert result["original_replay"][str(ep)]["calls"] == history[ep - 1]["validation"]["allowed"], ep
    print("Reproduced the original calls at epochs 1, 4, 12 and 320", flush=True)
    result["same_setup_prior_bias"] = replay(data, splits, cfg, architecture, device, windows, True, epochs)
    (HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for condition in ("original_replay", "same_setup_prior_bias"):
        for epoch, scores in result[condition].items():
            if isinstance(scores, dict):
                print(condition, epoch, scores["calls"], scores["probability_quantiles"][2],
                      scores["loss"], scores["average_precision"], flush=True)
    print("saved final", result["original_last"], flush=True)


if __name__ == "__main__":
    main()
