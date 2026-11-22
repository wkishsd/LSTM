"""Minibatch training, checkpoint evaluation and standalone inference."""

import csv
import json
import os
import platform
import random
from time import perf_counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import TimelinePaths, TradeDataset, mmap_array
from .io_utils import write_json
from .metrics import FIXED_QUOTA, Metrics, average_precision, quota_metrics
from .model import MACD_RELATIVE_INPUTS, TradeMLP, relative_macd_features
from .schema import FEATURES, STATES, WIDTH, window_input_size
from .split import training_split


def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def resolve_device(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N, auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; install a compatible CUDA PyTorch build or use --device cpu")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    return device


class DeviceWindows:
    def __init__(self, timeline, lookback_minutes=STATES):
        self.timeline = timeline
        self.input_size = window_input_size(lookback_minutes)
        self.offsets = torch.arange(1 - lookback_minutes, 1, device=timeline.device)

    def batches(self, loader):
        for ends, labels in loader:
            rows = ends.to(self.timeline.device, non_blocking=True)[:, None] + self.offsets
            windows = self.timeline.index_select(0, rows.flatten()).reshape(len(ends), self.input_size)
            yield windows, labels


def make_loader(data, indices, args, device, shuffle=False, batch_size=None, windows=None,
                lookback_minutes=None):
    generator = torch.Generator().manual_seed(args.seed)
    if windows is None:
        if lookback_minutes is None:
            lookback_minutes = getattr(args, "lookback_minutes", STATES)
        dataset = TradeDataset(data, indices, lookback_minutes)
    else:
        dataset = TensorDataset(torch.from_numpy(data.history_rows[indices]),
                                torch.from_numpy(data.labels[indices]))
    loader = DataLoader(
        dataset, batch_size=batch_size or args.batch_size,
        shuffle=shuffle, num_workers=args.num_workers, pin_memory=device.type == "cuda",
        drop_last=False, worker_init_fn=seed_worker, generator=generator,
        persistent_workers=args.num_workers > 0,
    )
    return loader if windows is None else DeviceWindowLoader(loader, windows)


class DeviceWindowLoader:
    def __init__(self, loader, windows):
        self.loader = loader
        self.windows = windows

    def __iter__(self):
        return self.windows.batches(self.loader)

    def __len__(self):
        return len(self.loader)


def cache_timeline(data, args, device, batch_size):
    if (getattr(args, "timeline_cache", "off") != "auto" or device.type != "cuda"
            or not isinstance(data.paths, TimelinePaths)):
        return None
    timeline = mmap_array(data.paths.timeline, "timeline", (WIDTH,))
    itemsize = np.dtype(np.float32).itemsize
    cache_bytes = timeline.size * itemsize
    # Leave VRAM for the model, optimizer and window/standardization buffers.
    lookback_minutes = getattr(args, "lookback_minutes", STATES)
    workspace_bytes = max(2 * cache_bytes, 4 * batch_size * window_input_size(lookback_minutes) * itemsize)
    free, _ = torch.cuda.mem_get_info(device)
    if cache_bytes + workspace_bytes > free:
        return None
    try:
        host_copy = torch.from_numpy(np.array(timeline, dtype=np.float32, copy=True))
        return DeviceWindows(host_copy.to(device), lookback_minutes)
    except torch.cuda.OutOfMemoryError:
        return None


def memory_stats(device):
    if device.type != "cuda":
        return {"device": "cpu"}
    free, total = torch.cuda.mem_get_info(device)
    return {"device": str(device), "name": torch.cuda.get_device_name(device),
            "allocated_mib": torch.cuda.memory_allocated(device) / 2**20,
            "reserved_mib": torch.cuda.memory_reserved(device) / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "free_mib": free / 2**20, "total_mib": total / 2**20}


def probability_mse(logits, labels):
    targets = nn.functional.one_hot(labels, num_classes=logits.size(1)).to(logits.dtype)
    return nn.functional.mse_loss(logits.softmax(dim=1), targets)


def monitored_optimizer_step(model, optimizer):
    weights = [(name, param) for name, param in model.named_parameters()
               if name.rsplit(".", 1)[-1] == "weight" and param.grad is not None]
    with torch.no_grad():
        snapshots = [param.detach().clone() for _, param in weights]
        gradients = [param.grad.norm() for _, param in weights]
        optimizer.step()
        norms = [torch.stack((grad, before.norm(), (param - before).norm()))
                 for (_, param), before, grad in zip(weights, snapshots, gradients, strict=True)]
        values = torch.stack(norms).cpu().tolist() if norms else []
    return {
        name: {"gradient_norm": grad, "update_norm": update, "relative_update": update / weight if weight else None}
        for (name, _), (grad, weight, update) in zip(weights, values, strict=True)
    }


def split_returns(data, indices):
    returns = getattr(data, "returns", None)
    return None if returns is None else returns[indices]


def run_epoch(model, loader, device, optimizer=None, *, loss_name="cross_entropy",
              diagnostic_steps=0, step_offset=0, quota=0, returns=None):
    training = optimizer is not None
    model.train(training)
    num_classes = getattr(model, "num_classes", 2)
    metrics = Metrics(num_classes)
    evaluation_targets, evaluation_scores = [], []
    step_samples = []
    criterion = {"cross_entropy": nn.functional.cross_entropy, "mse": probability_mse}[loss_name]
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for step, (features, labels) in enumerate(loader, step_offset + 1):
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise ValueError("Nonfinite logits/loss; inspect feature scale and learning rate")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"), error_if_nonfinite=True)
                if diagnostic_steps and step % diagnostic_steps == 0:
                    step_samples.append({"step": step, "weights": monitored_optimizer_step(model, optimizer)})
                else:
                    optimizer.step()
            targets = labels.detach().cpu().numpy()
            metrics.update(targets, logits.detach().argmax(1).cpu().numpy(), loss.item())
            if not training:
                evaluation_targets.append(targets)
                scores = logits[:, 1] - logits[:, 0] if num_classes == 2 else logits.softmax(dim=1)
                evaluation_scores.append(scores.cpu().numpy())
    result = metrics.result()
    if step_samples:
        result["optimizer_samples"] = step_samples
    if not training:
        targets, scores = np.concatenate(evaluation_targets), np.concatenate(evaluation_scores)
        if num_classes == 2:
            result["average_precision"] = average_precision(targets, scores)
            if quota:
                # Evaluation loaders are unshuffled, so `returns` follows the loader's row order.
                result["fixed_quota"] = quota_metrics(targets, scores, quota, returns)
        else:
            observed = np.unique(targets)
            result["average_precision"] = float(np.mean([
                average_precision(targets == label, scores[:, label]) for label in observed
            ]))
            result["average_precision_baseline"] = 1 / len(observed)
    return result


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def load_model(path, device, schema=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Incompatible checkpoint format or input size")
    architecture = checkpoint["architecture"]
    expected_inputs = window_input_size(architecture.get("lookback_minutes", STATES))
    if checkpoint.get("input_size") != expected_inputs:
        raise ValueError("Incompatible checkpoint format or input size")
    if schema is not None and checkpoint["schema"] != schema:
        raise ValueError("Inference schema differs from training schema (including anchor order and normalization)")
    model = TradeMLP(**architecture)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model, checkpoint


def fit_feature_scaling(model, data, train_indices):
    if not isinstance(data.paths, TimelinePaths):
        raise ValueError("Feature standardization requires a Bitfinex timeline export")
    timeline = mmap_array(data.paths.timeline, "timeline", (WIDTH,))
    ends = data.history_rows[train_indices]
    # Count each minute once across the overlapping training windows.
    coverage = np.bincount(ends + 1 - model.lookback_minutes, minlength=len(timeline) + 1)
    coverage -= np.bincount(ends + 1, minlength=len(timeline) + 1)
    market_columns = len(FEATURES)
    minutes = timeline[np.cumsum(coverage)[:-1] > 0, :market_columns]
    if model.feature_input in MACD_RELATIVE_INPUTS:
        minutes = relative_macd_features(torch.from_numpy(minutes)).numpy()
    mean = minutes.mean(axis=0, dtype=np.float64)
    scale = minutes.std(axis=0, dtype=np.float64)
    model.feature_mean[:market_columns].copy_(torch.from_numpy(mean))
    model.feature_scale[:market_columns].copy_(torch.from_numpy(np.where(scale > 1e-8, scale, 1.0)))


def train(data, args, out):
    splits, split_report = training_split(data, args.train_fraction, args.validation_fraction)
    target = data.schema.get("profit_target")
    write_json(out / "split_report.json", split_report)
    if len(np.unique(data.labels[splits["train"]])) < 2:
        raise ValueError("Training partition contains only one class")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    has_validation = "validation" in splits
    selection = args.selection if has_validation else "final_epoch"
    min_entry_calls = getattr(args, "min_entry_calls", 30)
    use_call_minimum = selection == "profitable_above_baseline"
    if use_call_minimum and not 1 <= min_entry_calls <= len(splits["validation"]):
        raise ValueError("Minimum entry calls must be between 1 and the number of validation entries")
    feature_input = getattr(args, "feature_input", "exported")
    if feature_input != "exported" and not isinstance(data.paths, TimelinePaths):
        raise ValueError("Feature input transformations require a Bitfinex timeline")
    if feature_input in MACD_RELATIVE_INPUTS and data.schema.get("macd_normalization") != "x / sqrt(1 + x^2)":
        raise ValueError("Relative MACD and stationary inputs require the bounded-MACD Bitfinex export")
    architecture = {"model": getattr(args, "model", "dense"), "feature_input": feature_input,
                    "hidden1": args.hidden1, "hidden2": args.hidden2,
                    "hidden3": getattr(args, "hidden3", 0),
                    "activation": getattr(args, "activation", "relu"),
                    "dropout": getattr(args, "dropout", 0.0),
                    "lookback_minutes": getattr(args, "lookback_minutes", STATES),
                    "standardize": getattr(args, "feature_scaling", "none") == "standardize"}
    if target:
        architecture["num_classes"] = len(target["class_names"])
    model = TradeMLP(**architecture)
    if architecture["standardize"]:
        fit_feature_scaling(model, data, splits["train"])
    # Constant initializations (e.g. all weights 1) make every unit in a layer identical, so
    # they receive identical gradients and the layer collapses to one effective neuron.
    if getattr(args, "initialization", "default") != "default":
        raise ValueError("Only the default (random) initialization is supported")
    model.to(device)
    optimizers = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    optimizer = optimizers[args.optimizer](model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    loss_name = getattr(args, "loss", "cross_entropy")
    eval_split = "validation" if has_validation else "train"
    eval_batch = args.batch_size if has_validation else 256
    windows = cache_timeline(data, args, device, max(args.batch_size, eval_batch))
    train_loader = make_loader(data, splits["train"], args, device, shuffle=True, windows=windows)
    eval_loader = make_loader(data, splits[eval_split], args, device, batch_size=eval_batch, windows=windows)
    diagnostic_steps = getattr(args, "diagnostic_steps", 0)
    fixed_train_loader = None
    if diagnostic_steps:
        indices = splits["train"]
        sample = np.random.default_rng(args.seed).choice(indices, min(len(indices), 512), replace=False)
        fixed_train_loader = make_loader(data, np.sort(sample), args, device, batch_size=min(eval_batch, 512),
                                         windows=windows)
    # Fixed-quota precision makes the MLP comparable with the baseline models on the same test entries.
    quota = FIXED_QUOTA if getattr(args, "dataset", "bitfinex") == "bitfinex" and model.num_classes == 2 else 0
    print(f"Bitfinex window assembly: {'GPU timeline cache' if windows else 'CPU loader'}", flush=True)
    metadata = {
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda, "parameter_count": sum(p.numel() for p in model.parameters()),
        "selection": selection,
        "selection_tie_break": "lower_validation_loss_then_earlier_epoch" if has_validation else None,
        "macd_normalization": data.schema.get("macd_normalization"),
        "timeline_cached_on_gpu": windows is not None,
        "config": vars(args), "paths": asdict(data.paths), "memory_at_start": memory_stats(device),
        "files": {name: {"bytes": Path(path).stat().st_size, "mtime_ns": Path(path).stat().st_mtime_ns}
                  for name, path in asdict(data.paths).items() if path},
    }
    if target:
        metadata["profit_target"] = target
    write_json(out / "run.json", metadata)
    best_key, best_epoch = None, None
    total_epoch_seconds = 0.0
    with (out / "history.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            epoch_started = perf_counter()
            training = run_epoch(model, train_loader, device, optimizer, loss_name=loss_name,
                                 diagnostic_steps=diagnostic_steps, step_offset=(epoch - 1) * len(train_loader))
            evaluation = run_epoch(model, eval_loader, device, loss_name=loss_name, quota=quota,
                                   returns=split_returns(data, splits[eval_split]))
            evaluation_key = "validation" if has_validation else "train_evaluation"
            eligible = has_validation and (not use_call_minimum or evaluation["allowed"] >= min_entry_calls)
            is_best = False
            if eligible:
                score = -evaluation["loss"] if selection == "loss" else evaluation[selection]
                key = (score, -evaluation["loss"])
                is_best = best_key is None or key > best_key
            payload = {
                "format_version": 1, "input_size": model.input_size,
                "architecture": architecture,
                "schema": data.schema, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "epoch": epoch,
                evaluation_key: evaluation, "selection": selection, "seed": args.seed,
                "config": vars(args), "torch_version": str(torch.__version__),
            }
            save_checkpoint(out / "last.pt", payload)
            if is_best:
                best_key, best_epoch = key, epoch
                save_checkpoint(out / "best.pt", payload)
            fixed = None
            if fixed_train_loader is not None and (epoch in (1, args.epochs) or "optimizer_samples" in training):
                fixed = run_epoch(model, fixed_train_loader, device, loss_name=loss_name)
            epoch_seconds = perf_counter() - epoch_started
            total_epoch_seconds += epoch_seconds
            record = {"epoch": epoch, "train": training, evaluation_key: evaluation,
                      "selection": selection, "selection_eligible": eligible, "best": is_best,
                      "epoch_seconds": epoch_seconds,
                      "average_epoch_seconds": total_epoch_seconds / epoch,
                      "memory": memory_stats(device)}
            if fixed is not None:
                record["fixed_train"] = {"loss": fixed["loss"], "count": fixed["count"]}
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
    timing = {
        "epochs_completed": args.epochs,
        "total_epoch_seconds": total_epoch_seconds,
        "average_epoch_seconds": total_epoch_seconds / args.epochs,
    }
    write_json(out / "run.json", {**metadata, "timing": timing})
    # Test is evaluated once, after all hyperparameters and checkpoint selection are fixed.
    summary = {"best_epoch": best_epoch, "timing": timing}
    if has_validation and best_epoch is None:
        summary.update(checkpoint_epoch=None, checkpoint_file=None, test=None,
                       test_reason=f"No epoch reached the minimum of {min_entry_calls} validation entry calls.")
    else:
        checkpoint_file = "best.pt" if has_validation else "last.pt"
        model, checkpoint = load_model(out / checkpoint_file, device, data.schema)
        test_loader = make_loader(data, splits["test"], args, device, batch_size=eval_batch, windows=windows)
        test = run_epoch(model, test_loader, device, loss_name=loss_name, quota=quota,
                         returns=split_returns(data, splits["test"]))
        summary.update(checkpoint_epoch=checkpoint["epoch"], checkpoint_file=checkpoint_file, test=test)
    summary["memory"] = memory_stats(device)
    write_json(out / "test_metrics.json", summary)
    print(json.dumps(summary, allow_nan=False), flush=True)
    return summary


def predict(data, args, out):
    if not len(data.valid):
        raise ValueError("No valid rows to predict")
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device, data.schema)
    loader = make_loader(data, data.valid, args, device, lookback_minutes=model.lookback_minutes)
    allowed, total = 0, 0
    target = checkpoint["schema"].get("profit_target")
    counts = np.zeros(model.num_classes, dtype=np.int64)
    temp = out / "predictions.csv.tmp"
    with temp.open("w", newline="", encoding="utf-8") as stream, torch.inference_mode():
        writer = csv.writer(stream)
        if target:
            probability_columns = [f"p_class_{i}" for i in range(model.num_classes)]
            writer.writerow(["entry_id", "fill_time_utc_seconds", *probability_columns, "predicted_class"])
        else:
            writer.writerow(["entry_id", "signal_time_utc_seconds", "p_skip", "p_enter", "predicted_class"])
        for features, indices in loader:
            probs = model.probabilities(features.to(device, non_blocking=True)).cpu().numpy()
            if not np.isfinite(probs).all():
                raise ValueError("Nonfinite prediction probabilities")
            classes = probs.argmax(axis=1)  # ties -> 0, the same rule as evaluation
            for i, p, prediction in zip(indices.tolist(), probs, classes.tolist()):
                writer.writerow([data.ids[i].item(), int(data.times[i]), *map(float, p), prediction])
            if target:
                counts += np.bincount(classes, minlength=model.num_classes)
            else:
                allowed += int(classes.sum())
            total += len(classes)
    temp.replace(out / "predictions.csv")
    result = {"count": total, "checkpoint_epoch": checkpoint["epoch"], "memory": memory_stats(device)}
    if target:
        result.update(class_names=target["class_names"], predicted_class_counts=counts.tolist())
    else:
        result.update(allowed=allowed, rejected=total - allowed)
    write_json(out / "prediction_summary.json", result)
    print(json.dumps(result), flush=True)
    return result
