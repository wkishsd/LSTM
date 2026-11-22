"""Recheck the original binary checkpoints, Adam, and held-out scores on the frozen export."""

from argparse import ArgumentParser, Namespace
import csv
from datetime import datetime, timezone
from hashlib import file_digest
from itertools import groupby
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINER = ROOT / "Training/rocm-9070"
ORIGINAL = TRAINER / "runs/rx9070-20261120-024355-b25e"
EXPORT = ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd"
sys.path.insert(0, str(TRAINER))

from mlp_trades.bitfinex import export_paths, prepare_bitfinex
from mlp_trades.engine import DeviceWindows, cache_timeline, load_model, make_loader, run_epoch, seed_everything
from mlp_trades.metrics import average_precision
from mlp_trades.split import training_split


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_history(path):
    return [json.loads(line) for line in (path / "history.jsonl").read_text().splitlines()]


def hash_file(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def reference_ap(labels, scores):
    records = sorted(zip(scores.tolist(), labels.tolist()), reverse=True)
    hits = seen = 0
    area = 0.0
    for _, group in groupby(records, key=lambda pair: pair[0]):
        group = list(group)
        new_hits = sum(label for _, label in group)
        hits += new_hits
        seen += len(group)
        area += new_hits * hits / seen
    return area / hits if hits else 0.0


def scores_for(model, loader, labels):
    parts, targets = [], []
    model.eval()
    with torch.inference_mode():
        for features, y in loader:
            logits = model(features)
            parts.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
            targets.append(y.numpy())
    np.testing.assert_array_equal(np.concatenate(targets), labels)
    return np.concatenate(parts).astype(np.float64)


def rank_summary(y, score, times, margin=False):
    ap = reference_ap(y, score)
    assert abs(ap - average_precision(y, score)) < 1e-12
    weeks = (times + 3 * 86400) // (7 * 86400)
    weekly_hits, weekly_calls, weekly_expected, excess_wins = 0, 0, 0.0, []
    for week in np.unique(weeks):
        idx = np.flatnonzero(weeks == week)
        quota = round(len(idx) * 1000 / 15728)
        top = idx[np.argsort(-score[idx], kind="stable")[:quota]]
        wins = int(y[top].sum())
        expected = quota * y[idx].mean()
        weekly_hits += wins
        weekly_calls += quota
        weekly_expected += expected
        excess_wins.append(wins - expected)
    result = {"ap": ap, "week_quota_calls": weekly_calls, "week_quota_winners": weekly_hits,
              "week_quota_expected_winners": float(weekly_expected),
              "weeks_above_quota_baseline": sum(bool(x > 0) for x in excess_wins)}
    if margin:
        called = score > 0
        result.update(calls_at_50_percent=int(called.sum()), winners_at_50_percent=int(y[called].sum()),
                      log_loss=float(np.mean(np.logaddexp(0, score) - y * score)))
    return result


def verify_adam(model, checkpoint, batch, device):
    """Compare the production step against the Adam equations using saved moments."""
    group = checkpoint["optimizer_state"]["param_groups"][0]
    optimizer = torch.optim.Adam(model.parameters(), lr=group["lr"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    weights = list(model.parameters())
    before = [param.detach().clone() for param in weights]
    state_before = [{key: value.clone() for key, value in optimizer.state[param].items()}
                    for param in weights]
    head_inputs, outputs = [], []
    head = model.layers[-1]

    def capture_head(_module, inputs, out):
        head_inputs.append(inputs[0].detach())
        outputs.append(out.detach())

    hook = head.register_forward_hook(capture_head)
    run_epoch(model, [batch], device, optimizer)
    hook.remove()
    probabilities = outputs[0].double().softmax(1)
    probabilities[torch.arange(len(probabilities), device=device), batch[1].to(device)] -= 1
    reference_gradient = probabilities.T @ head_inputs[0].double() / len(probabilities)
    head_gradient_error = float((head.weight.grad - reference_gradient).abs().max().cpu())
    torch.testing.assert_close(head.weight.grad.double(), reference_gradient, rtol=2e-5, atol=2e-6)
    beta1, beta2 = group["betas"]
    errors = []
    for param, initial, prior in zip(weights, before, state_before, strict=True):
        observed = optimizer.state[param]
        step = int(prior["step"]) + 1
        assert int(observed["step"]) == step
        gradient = param.grad.detach().double()
        first = beta1 * prior["exp_avg"].double() + (1 - beta1) * gradient
        second = beta2 * prior["exp_avg_sq"].double() + (1 - beta2) * gradient.square()
        expected = initial.double() - group["lr"] * (first / (1 - beta1 ** step)) / (
            (second / (1 - beta2 ** step)).sqrt() + group["eps"])
        torch.testing.assert_close(observed["exp_avg"].double(), first, rtol=2e-6, atol=1e-9)
        torch.testing.assert_close(observed["exp_avg_sq"].double(), second, rtol=2e-6, atol=1e-9)
        torch.testing.assert_close(param.detach().double(), expected, rtol=1e-7, atol=1e-9)
        errors.append(float((param.detach() - expected).abs().max().cpu()))
    return {"checked_parameters": sum(p.numel() for p in weights),
            "learning_rate": group["lr"], "checked_batch_rows": len(batch[1]),
            "step": step, "max_expected_update_rounding_error": max(errors),
            "max_head_gradient_error": head_gradient_error,
            "max_observed_parameter_update": max(float((p.detach() - q).abs().max().cpu())
                                                 for p, q in zip(weights, before, strict=True))}


def best_epoch(history, metric, minimum=0):
    eligible = [row for row in history if row["validation"]["allowed"] >= minimum]
    return max(eligible, key=lambda row: (row["validation"][metric], -row["validation"]["loss"],
                                          -row["epoch"]))["epoch"]


def history_summary(history, epochs):
    return [{"epoch": row["epoch"], "train_loss": row["train"]["loss"],
             "ap": row["validation"]["average_precision"], "loss": row["validation"]["loss"],
             "calls": row["validation"]["allowed"],
             "excess_winners": row["validation"]["profitable_above_baseline"]}
            for row in history if row["epoch"] in epochs]


def wave6_envelope(labels, scores, changeable):
    """Bound evaluation-only AP with any labels on recorded pre-confirmation market exits."""
    order = np.argsort(-scores, kind="stable")
    ranked_y, ranked_scores, editable = labels[order], scores[order], changeable[order]
    starts = np.r_[0, np.flatnonzero(ranked_scores[:-1] != ranked_scores[1:]) + 1]
    ends = np.r_[starts[1:], len(order)]
    fixed = np.add.reduceat(ranked_y * ~editable, starts)
    movable = np.add.reduceat(editable.astype(int), starts)
    # At each distinct score keep the best accumulated precision sum for each
    # possible count of newly assigned positives. Divide by total positives at the end.
    counts = np.arange(int(editable.sum()) + 1)
    accumulated = np.full(len(counts), -np.inf)
    accumulated[0] = 0.0
    fixed_total = 0
    for end, hits, editable_count in zip(ends, fixed, movable, strict=True):
        fixed_total += hits
        precision = (fixed_total + counts) / end
        if editable_count == 0:
            accumulated += hits * precision
        else:
            candidates = accumulated + hits * precision
            for new_hits in range(1, editable_count + 1):
                adjusted = accumulated[:-new_hits] + (hits + new_hits) * precision[new_hits:]
                candidates[new_hits:] = np.maximum(candidates[new_hits:], adjusted)
            accumulated = candidates
    return {"changeable_rows": int(editable.sum()),
            "best_possible_ap": float(np.max(accumulated / (fixed_total + counts)))}


def paired_ap_intervals(y, times, scores, pairs, repetitions=2000):
    """Resample adjacent two-week groups jointly across methods."""
    blocks = (times + 3 * 86400) // (14 * 86400)
    unique, block_index = np.unique(blocks, return_inverse=True)
    rng = np.random.default_rng(42)
    resamples = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)), size=repetitions)
    replicates = {}
    for name, score in scores.items():
        order = np.argsort(-score, kind="stable")
        positions = np.r_[0, np.flatnonzero(score[order][1:] != score[order][:-1]) + 1]
        groups = block_index[order]
        labels = y[order]
        estimates = []
        for sample in resamples:
            weights = sample[groups]
            new_hits = np.add.reduceat(weights * labels, positions)
            totals = np.add.reduceat(weights, positions).cumsum()
            precision = np.divide(new_hits.cumsum(), totals, out=np.zeros(len(totals)), where=totals > 0)
            estimates.append(float((new_hits * precision).sum() / new_hits.sum()))
        replicates[name] = np.array(estimates)
    return {"blocks": len(unique), "resamples": repetitions, "difference_intervals": {
        f"{left} minus {right}": np.quantile(replicates[left] - replicates[right], [.025, .5, .975]).tolist()
        for left, right in pairs
    }}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, action="append", default=[])
    args = parser.parse_args()
    torch.set_num_threads(3)
    device = torch.device("cuda:0")
    cfg = Namespace(**read_json(ORIGINAL / "run.json")["config"])
    seed_everything(cfg.seed)
    data = prepare_bitfinex(export_paths(EXPORT), training=True)
    splits, report = training_split(data, cfg.train_fraction, cfg.validation_fraction)
    original_split = read_json(ORIGINAL / "split_report.json")
    assert report == original_split
    manifest = read_json(EXPORT / "manifest.json")
    timeline = np.load(data.paths.timeline, mmap_mode="r", allow_pickle=False)
    windows = cache_timeline(data, cfg, device, cfg.batch_size)
    assert windows is not None
    atr_column = manifest["feature_names_per_minute"].index("norm_atr_14_5m")
    with (EXPORT / "entry_targets.csv").open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    np.testing.assert_array_equal([r["entry_id"] for r in rows], data.ids)
    np.testing.assert_array_equal([int(r["binary_target"]) for r in rows], data.labels)
    early_exit = np.array([
        bool(r["wave6_market_exit_ms_utc"]) and bool(r["series_cutoff_at"])
        and float(r["wave6_market_exit_ms_utc"]) < 1000 * (
            datetime.fromisoformat(r["series_cutoff_at"]).timestamp() + 300)
        for r in rows
    ])
    del rows
    res = {
        "source": {"dataset": str(EXPORT.relative_to(ROOT)), "signals": len(data.valid),
                   "sha256": {name: hash_file(path) for name, path in vars(data.paths).items()},
                   "purged": report["purged_count"]},
        "splits": {}, "original": {}, "controls": {},
    }
    scored = {}
    for split, ix in splits.items():
        y, times = data.labels[ix], data.times[ix]
        prices = timeline[data.history_rows[ix], 0]
        atr = timeline[data.history_rows[ix], atr_column]
        scored[split] = {"M5 ATR": atr}
        res["splits"][split] = {
            "count": len(y), "winner_rate": float(y.mean()),
            "first_signal_utc": datetime.fromtimestamp(int(times[0]), timezone.utc).isoformat(),
            "last_signal_utc": datetime.fromtimestamp(int(times[-1]), timezone.utc).isoformat(),
            "median_price_usd": float(np.exp(np.median(prices))),
            "m5_atr": rank_summary(y, atr, times),
        }
    for run in [ORIGINAL, *args.control]:
        original = run == ORIGINAL
        run_cfg = Namespace(**read_json(run / "run.json")["config"])
        report_for_run = read_json(run / "split_report.json")
        assert report_for_run == original_split
        run_windows = windows if run_cfg.lookback_minutes == cfg.lookback_minutes else (
            DeviceWindows(windows.timeline, run_cfg.lookback_minutes))
        changes = {key: [value, vars(run_cfg)[key]] for key, value in vars(cfg).items()
                   if key not in {"config", "output_dir", "bitfinex_dir"} and value != vars(run_cfg)[key]}
        history = read_history(run)
        ranked = best_epoch(history, "average_precision")
        selected = best_epoch(history, "profitable_above_baseline", cfg.min_entry_calls) if original else ranked
        summary = {"epochs": len(history), "selected_epoch": selected, "highest_ap_epoch": ranked,
                   "learning_rate": run_cfg.learning_rate, "config_changes": changes,
                   "history": history_summary(history, {1, 12, 47, 48, 100, 200, 320,
                                                        selected, ranked, len(history)}),
                   "checkpoints": {}}
        for filename, ep in [("best.pt", selected), ("last.pt", len(history))]:
            model, checkpoint = load_model(run / filename, device, data.schema)
            assert checkpoint["epoch"] == ep
            assert checkpoint["validation"] == history[ep - 1]["validation"]
            saved = {"epoch": ep, "sha256": hash_file(run / filename)}
            batches = math.ceil(len(splits["train"]) / cfg.batch_size)
            for state in checkpoint["optimizer_state"]["state"].values():
                assert int(state["step"]) == ep * batches
            saved["optimizer_steps"] = ep * batches
            for split in ("validation", "test", "train") if original else ("validation", "test"):
                ix = splits[split]
                y, times = data.labels[ix], data.times[ix]
                loader = make_loader(data, ix, run_cfg, device, windows=run_windows)
                score = scores_for(model, loader, y)
                if split in ("validation", "test") and (original or filename == "best.pt"):
                    scored[split][f"{'original' if original else run.name} {filename}"] = score
                metrics = rank_summary(y, score, times, margin=True)
                recorded = checkpoint["validation"] if split == "validation" else (
                    read_json(run / "test_metrics.json")["test"] if split == "test" and filename == "best.pt"
                    else None)
                if recorded:
                    assert abs(recorded["average_precision"] - metrics["ap"]) < 1e-6, (run, filename, split)
                    assert abs(recorded["loss"] - metrics["log_loss"]) < 1e-6
                    assert recorded["allowed"] == metrics["calls_at_50_percent"]
                    assert recorded["confusion_matrix"]["TP"] == metrics["winners_at_50_percent"]
                if original and split in ("validation", "test"):
                    metrics["known_wave6_exit_envelope"] = wave6_envelope(y, score, early_exit[ix])
                saved[split] = metrics
            if original and filename == "last.pt":
                batch = next(iter(make_loader(data, splits["train"], cfg, device, shuffle=True, windows=windows)))
                res["adam"] = verify_adam(model, checkpoint, batch, device)
            summary["checkpoints"][filename] = saved
        if original:
            res["original"] = summary
            res["original_curve"] = history_summary(history, set(range(1, len(history) + 1)))
        else:
            res["controls"][run.name] = summary
        print(run.name, "selects", selected, "max AP epoch", ranked, flush=True)
    for split in ("validation", "test"):
        ix = splits[split]
        scores = scored[split]
        pairs = [("original last.pt", "original best.pt"), ("M5 ATR", "original last.pt")]
        pairs += [(f"{run.name} best.pt", "original last.pt") for run in args.control]
        pairs += [("M5 ATR", f"{run.name} best.pt") for run in args.control]
        res["splits"][split]["paired_ap_bootstrap"] = paired_ap_intervals(
            data.labels[ix], data.times[ix], scores, pairs)
    (HERE / "checkpoint_results.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
