"""Repeat the profit-bucket run with equal aggregate weight per training class."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "Training" / "rocm-9070"
sys.path.insert(0, str(PROJECT))

from mlp_trades import engine
from mlp_trades.bitfinex import export_paths, prepare_bitfinex
from mlp_trades.split import training_split


def snapshots(records, epochs):
    rows = []
    for record in records:
        if record["epoch"] not in epochs:
            continue
        val = record["validation"]
        rows.append({"epoch": record["epoch"], "train_acc": record["train"]["accuracy"],
                     "val_loss": val["loss"], "val_acc": val["accuracy"],
                     "loss_corrected_for_class_weight": val.get("loss_corrected_for_class_weight"),
                     "macro_recall": val["balanced_accuracy"], "macro_ap": val["average_precision"],
                     "validation_calls": [row["calls"] for row in val["per_class"]]})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    config = json.loads((args.run / "run.json").read_text(encoding="utf-8"))["config"]
    data = prepare_bitfinex(export_paths(config["bitfinex_dir"]), training=True)
    train_args = argparse.Namespace(**config)
    train_args.epochs = args.epochs
    splits, _ = training_split(data, train_args.train_fraction, train_args.validation_fraction)
    train = np.bincount(data.labels[splits["train"]], minlength=5)
    validation = np.bincount(data.labels[splits["validation"]], minlength=5)
    proportions = train / train.sum()
    inverse = 1 / (len(train) * proportions)
    baseline_loss = float(-np.log(proportions) @ (validation / validation.sum()))
    out = HERE / f"weighted_{args.epochs}"
    out.mkdir(exist_ok=False)
    train_args.output_dir = str(out)
    previous_run_epoch = engine.run_epoch
    previous_criterion = engine.nn.CrossEntropyLoss

    def run_epoch(model, loader, device, optimizer=None):
        weight = torch.as_tensor(inverse, dtype=torch.float32, device=device)
        adjusted_loss, count = 0.0, 0
        if optimizer is None:
            original_loss = previous_criterion()
            log_weight = weight.log()

            def score(logits, labels):
                nonlocal adjusted_loss, count
                # Undo the odds shift induced by weighting to compare loss on natural rates.
                adjusted_loss += torch.nn.functional.cross_entropy(logits - log_weight, labels, reduction="sum").item()
                count += labels.numel()
                return original_loss(logits, labels)

            engine.nn.CrossEntropyLoss = lambda: score
        else:
            engine.nn.CrossEntropyLoss = lambda: previous_criterion(weight=weight)
        try:
            metrics = previous_run_epoch(model, loader, device, optimizer)
            if optimizer is None:
                metrics["loss_corrected_for_class_weight"] = adjusted_loss / count
            return metrics
        finally:
            engine.nn.CrossEntropyLoss = previous_criterion

    engine.run_epoch = run_epoch
    print(f"Running {args.epochs} otherwise matched epochs with weights {inverse.round(3).tolist()}", flush=True)
    with (out / "console.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
        best = engine.train(data, train_args, out)

    def read(folder):
        return [json.loads(line) for line in (folder / "history.jsonl").read_text(encoding="utf-8").splitlines()]

    weighted, original = read(out), read(args.run)
    r = [row for row in original if row["epoch"] <= args.epochs]
    originals = {min(r, key=lambda q: q["validation"]["loss"])["epoch"],
                 max(r, key=lambda q: q["validation"]["average_precision"])["epoch"], args.epochs}
    picked = {min(weighted, key=lambda q: q["validation"]["loss"])["epoch"], best["best_epoch"], args.epochs}
    picked.add(min(weighted, key=lambda q: q["validation"]["loss_corrected_for_class_weight"])["epoch"])
    output = {"source_run": args.run.name, "weighted_epochs": args.epochs, "class_weights": inverse.tolist(),
              "validation_loss_pooled_training_prior": baseline_loss, "validation_support": validation.tolist(),
              "original_snapshots": snapshots(original, originals), "weighted_snapshots": snapshots(weighted, picked),
              "weighted_test": best["test"]}
    (HERE / f"comparison_{args.epochs}.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
