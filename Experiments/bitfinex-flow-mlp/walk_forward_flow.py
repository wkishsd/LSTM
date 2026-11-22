"""Evaluate scheduled refits of the neural ranker using strictly earlier outcomes."""

from collections import defaultdict
import argparse
import json

import numpy as np
import torch

from train_flow import HERE, WEEK, evaluate, fixed_fold, train, view


HALF_LIVES = (45, 180)
REFIT_WEEKS = 4
WEIGHTS = (0.25, 0.5, 1.0)
STEPS = (120, 240, 480)


def refit_scores(history, destination, evaluation, mode, objective, seeds, half_life, device):
    times = destination["time"][evaluation]
    period = (times + 3 * 86400) // (WEEK * REFIT_WEEKS)
    keys = [(step, weight) for step in STEPS for weight in WEIGHTS]
    scores = {key: np.empty(len(evaluation)) for key in keys}
    for window in np.unique(period):
        local = np.flatnonzero(period == window)
        batch = evaluation[local]
        # Keep the strict same input/label separation as the original export trainer.
        cutoff = destination["start"][batch].min()
        ix = np.flatnonzero(np.maximum(history["time"], history["available"]) < cutoff)
        models = [train(history, ix, destination, batch, mode, objective, seed, device,
                        STEPS, half_life=half_life) for seed in seeds]
        anchor = models[0][0]
        for step, weight in keys:
            residual = np.mean([pred[step] for _, pred in models], axis=0)
            scores[step, weight][local] = anchor + weight * residual
    return scores


def only_selected(history, destination, evaluation, mode, objective, seeds, selected, device, save_as=None):
    half, step, weight = selected
    times = destination["time"][evaluation]
    period = (times + 3 * 86400) // (WEEK * REFIT_WEEKS)
    scores = np.empty(len(evaluation))
    for window in np.unique(period):
        local = np.flatnonzero(period == window)
        batch = evaluation[local]
        cutoff = destination["start"][batch].min()
        ix = np.flatnonzero(np.maximum(history["time"], history["available"]) < cutoff)
        preds = []
        for seed in seeds:
            ckpt = HERE / f"{save_as}_seed{seed}.pt" if save_as else None
            preds.append(train(history, ix, destination, batch, mode, objective, seed, device,
                               (step,), checkpoint=ckpt, half_life=half))
        scores[local] = preds[0][0] + weight * np.mean([pred[step] for _, pred in preds], axis=0)
    return scores


def metric_summary(metrics):
    return {k: v for k, v in metrics.items() if k != "weeks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument("--futures", action="store_true")
    choices.add_argument("--adaptive", action="store_true")
    choices.add_argument("--geometry", action="store_true")
    args = parser.parse_args()
    feature_sets = {
        "futures": "combined_futures", "adaptive": "feedback", "geometry": "geometry",
    }
    feature_set = next((key for key in feature_sets if getattr(args, key)), None)
    tag = f"{feature_set}_walk_forward" if feature_set else "walk_forward"
    source = f"{feature_set}_features.npz" if feature_set else "features.npz"
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = view(HERE / source)
    frozen = json.loads((HERE / "locked_results.json").read_text(encoding="utf-8"))
    mode, objective = frozen["recipe"][:2]
    if feature_set:
        mode = feature_sets[feature_set]
    seeds = frozen["seeds"]
    candidates = defaultdict(list)
    development_atr = []
    for bounds in ((0.40, 0.55), (0.55, 0.70), (0.70, 0.85)):
        _, validation = fixed_fold(data, *bounds)
        t, y = data["time"][validation], data["y"][validation]
        baseline = evaluate(data["atr"][validation], y, t)
        development_atr.append(baseline)
        for half_life in HALF_LIVES:
            scores = refit_scores(data, data, validation, mode, objective, seeds, half_life, device)
            for (step, weight), pred in scores.items():
                candidates[half_life, step, weight].append(evaluate(pred, y, t))
        print("walk-forward development", bounds, "ATR", baseline["wins"], flush=True)
    selected = max(candidates, key=lambda key: (sum(r["wins"] for r in candidates[key]),
                                                 sum(r["ap"] for r in candidates[key])))
    print("walk-forward chosen from development", selected,
          [r["wins"] for r in candidates[selected]], flush=True)
    _, test = fixed_fold(data, 0.85, 1.0)
    pred = only_selected(data, data, test, mode, objective, seeds, selected, device)
    t, y, a = data["time"][test], data["y"][test], data["atr"][test]
    output = {"recipe": {"feature_mode": mode, "objective": objective, "half_life_days": selected[0],
                          "updates": selected[1], "residual_weight": selected[2],
                          "refit_weeks": REFIT_WEEKS, "seeds": seeds},
              "development_atr": development_atr, "development_mlp": candidates[selected],
              "test_atr": evaluate(a, y, t), "test_mlp": evaluate(pred, y, t)}
    (HERE / f"{tag}_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    np.savez(HERE / f"{tag}_predictions.npz", score=pred, atr=a, time=t, labels=y, ids=data["ids"][test])
    print("walk-forward test", metric_summary(output["test_mlp"]), "ATR", metric_summary(output["test_atr"]), flush=True)
    fresh = view(HERE / f"fresh_{source}")
    assert data["time"].max() < fresh["time"].min()
    ix = np.arange(len(fresh["time"]))
    scores = only_selected(data, fresh, ix, mode, objective, seeds, selected, device, save_as=tag)
    y, t, a = fresh["y"], fresh["time"], fresh["atr"]
    output["fresh_atr"] = evaluate(a, y, t)
    output["fresh_mlp"] = evaluate(scores, y, t)
    (HERE / f"{tag}_results.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    np.savez(HERE / f"{tag}_fresh_predictions.npz", score=scores, atr=a, time=t, labels=y, ids=fresh["ids"])
    print("walk-forward fresh", metric_summary(output["fresh_mlp"]), "ATR",
          metric_summary(output["fresh_atr"]), flush=True)


if __name__ == "__main__":
    main()
