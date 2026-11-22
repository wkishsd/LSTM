"""Compare two 12-epoch continuations from the exact epoch-320 model and Adam state."""

from argparse import Namespace
import json

import torch

from checkpoints import EXPORT, HERE, ORIGINAL, read_json
from mlp_trades.bitfinex import export_paths, prepare_bitfinex
from mlp_trades.engine import cache_timeline, load_model, make_loader, run_epoch, seed_everything
from mlp_trades.split import training_split


def main():
    torch.set_num_threads(3)
    cfg = Namespace(**read_json(ORIGINAL / "run.json")["config"])
    device = torch.device("cuda:0")
    data = prepare_bitfinex(export_paths(EXPORT), training=True)
    splits, report = training_split(data, cfg.train_fraction, cfg.validation_fraction)
    assert report == read_json(ORIGINAL / "split_report.json")
    windows = cache_timeline(data, cfg, device, cfg.batch_size)
    assert windows is not None
    reference, checkpoint = load_model(ORIGINAL / "last.pt", device, data.schema)
    validation = make_loader(data, splits["validation"], cfg, device, windows=windows)
    baseline = run_epoch(reference, validation, device)
    assert abs(baseline["average_precision"] - checkpoint["validation"]["average_precision"]) < 1e-6
    result = {"start_epoch": checkpoint["epoch"], "start_validation": baseline, "branches": {}}
    del reference, checkpoint
    # The checkpoint stores no RNG state. Both arms get the same fresh seed
    # and the same data order/dropout sequence, starting from the same weights.
    for learning_rate in (cfg.learning_rate, 1e-5):
        seed_everything(cfg.seed)
        model, checkpoint = load_model(ORIGINAL / "last.pt", device, data.schema)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        optimizer.param_groups[0]["lr"] = learning_rate
        train = make_loader(data, splits["train"], cfg, device, shuffle=True, windows=windows)
        records = []
        for epoch in range(1, 13):
            trained = run_epoch(model, train, device, optimizer)
            if epoch in (1, 4, 12):
                validation_metrics = run_epoch(model, validation, device)
                records.append({"continuation_epoch": epoch, "train_loss": trained["loss"],
                                "validation": validation_metrics})
                print(learning_rate, epoch, validation_metrics["average_precision"], flush=True)
        result["branches"][str(learning_rate)] = records
    (HERE / "continuation_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
