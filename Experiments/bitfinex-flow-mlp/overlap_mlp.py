"""Give less training weight to trade labels sharing the same outcome interval."""

from collections import defaultdict
import json

import numpy as np
import torch

from train_flow import HERE, SEEDS, WEEK, evaluate, fixed_fold, train, view


STEPS = (40, 160, 400)


def average_uniqueness(start, available):
    end = np.maximum(start + 60, available)
    time = np.unique(np.r_[start, end])
    lo, hi = np.searchsorted(time, start), np.searchsorted(time, end)
    delta = np.bincount(lo, minlength=len(time)) - np.bincount(hi, minlength=len(time))
    concurrent = np.cumsum(delta)[:-1]
    balance = np.r_[0, np.cumsum(np.diff(time) / np.maximum(concurrent, 1))]
    return (balance[hi] - balance[lo]) / (end - start)


def walk(history, target, ix, half_life, power, device, steps=STEPS, save=False):
    blocks = (target["time"][ix] + 3 * 86400) // (4 * WEEK)
    result = {i: np.empty(len(ix)) for i in steps}
    for block in np.unique(blocks):
        mask = blocks == block
        test = ix[mask]
        tr = np.flatnonzero(history["available"] < target["start"][test].min())
        weights = average_uniqueness(history["time"][tr], history["available"][tr]) ** power
        models = [train(history, tr, target, test, "combined", "bce", seed, device, steps,
                        checkpoint=(HERE / f"overlap_mlp_fresh_block{block}_seed{seed}.pt" if save else None),
                        half_life=half_life, sample_weights=weights) for seed in SEEDS]
        for step in steps:
            result[step][mask] = models[0][0] + np.mean([p[step] for _, p in models], axis=0)
    return result


def main():
    np.testing.assert_allclose(average_uniqueness(np.array([0, 60]), np.array([120, 180])), [.75, .75])
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = view(HERE / "features.npz")
    weights = average_uniqueness(data["time"], data["available"])
    counts = np.quantile(weights, [.1, .5, .9])
    print("Average uniqueness quantiles", counts.tolist(), flush=True)
    dev = defaultdict(list)
    for bounds in ((.40, .55), (.55, .70), (.70, .85)):
        _, ix = fixed_fold(data, *bounds)
        for power in (0, .5, 1):
            for step, score in walk(data, data, ix, 180, power, device).items():
                dev[power, step].append(evaluate(score, data["y"][ix], data["time"][ix]))
        print("Overlap MLP development fold", bounds, flush=True)
    candidates = {str(power): max((key for key in dev if key[0] == power),
                                  key=lambda k: sum(x["wins"] for x in dev[k])) for power in (0, .5, 1)}
    selected = max((v for k, v in candidates.items() if k != '0'), key=lambda k: sum(x["wins"] for x in dev[k]))
    _, test = fixed_fold(data, .85, 1)
    fresh = view(HERE / "fresh_features.npz")
    result = {"uniqueness_quantiles": counts.tolist(), "recipe": list(selected), "recipes": candidates,
              "development": {k: dev[v] for k, v in candidates.items()}}
    for name, frame, ix in (("historical", data, test), ("fresh", fresh, np.arange(len(fresh["time"])))):
        t, y = frame["time"][ix], frame["y"][ix]
        results = {"atr": evaluate(frame["atr"][ix], y, t)}
        for kind, (power, step) in (("ordinary", candidates["0"]), ("overlap_weighted", selected)):
            score = walk(data, frame, ix, 180, power, device, (step,), save=name == "fresh" and bool(power))[step]
            results[kind] = evaluate(score, y, t)
        result[name] = results
        print(name, {key: (v["wins"], v["calls"], v["ap"]) for key, v in results.items()}, flush=True)
    (HERE / "overlap_mlp_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
