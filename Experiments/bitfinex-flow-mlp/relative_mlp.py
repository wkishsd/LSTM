"""Test market features expressed relative to conditions known before each decision."""

from collections import defaultdict
import json

import numpy as np
import torch

from train_flow import HERE, SEEDS, WEEK, evaluate, fixed_fold, train, view


STEPS = (40, 160, 400)


def causal_features(history, fresh):
    times = np.concatenate((history["time"], fresh["time"]))
    assert np.all(np.diff(times) > 0)
    values = np.column_stack((np.concatenate((history["flow"], fresh["flow"])),
                              np.log(np.concatenate((history["atr"], fresh["atr"])).clip(1e-8))))
    end = np.arange(len(times))
    total = np.vstack((np.zeros(values.shape[1]), np.cumsum(values, axis=0, dtype=np.float64)))
    squared = np.vstack((np.zeros(values.shape[1]), np.cumsum(values ** 2, axis=0, dtype=np.float64)))
    frames = []
    for days in (1, 7, 30):
        start = np.searchsorted(times, times - days * 86400)
        n = np.maximum(end - start, 1)[:, None]
        mu = (total[end] - total[start]) / n
        variance = (squared[end] - squared[start]) / n - mu ** 2
        x = (values - mu) / np.sqrt(variance.clip(1e-12))
        x[end - start < 20] = 0
        frames.append(x.clip(-5, 5).astype(np.float32))
    relative = np.column_stack(frames)
    combined = np.column_stack((history["base"], relative[:len(history["time"])]))
    other = np.column_stack((fresh["base"], relative[len(history["time"]):]))
    history["relative"], fresh["relative"] = combined, other


def walk(history, dest, ix, objective, half, device, steps=STEPS, save=False):
    periods = (dest["time"][ix] + 3 * 86400) // (4 * WEEK)
    output = {step: np.empty(len(ix)) for step in steps}
    for block in np.unique(periods):
        mask = periods == block
        test = ix[mask]
        tr = np.flatnonzero(history["available"] < dest["start"][test].min())
        fits = [train(history, tr, dest, test, "relative", objective, seed, device, steps,
                      checkpoint=HERE / f"relative_mlp_fresh_block{block}_seed{seed}.pt" if save else None,
                      half_life=half) for seed in SEEDS]
        for step in steps:
            output[step][mask] = fits[0][0] + np.mean([scores[step] for _, scores in fits], axis=0)
    return output


def main():
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = view(HERE / "features.npz")
    fresh = view(HERE / "fresh_features.npz")
    causal_features(data, fresh)
    assert np.isfinite(data["relative"]).all() and np.isfinite(fresh["relative"]).all()
    dev = defaultdict(list)
    for bounds in ((.40, .55), (.55, .70), (.70, .85)):
        _, ix = fixed_fold(data, *bounds)
        for objective in ("pair", "bce"):
            for half in (45, 180):
                scores = walk(data, data, ix, objective, half, device)
                for n, score in scores.items():
                    dev[objective, half, n].append(evaluate(score, data["y"][ix], data["time"][ix]))
        print("Relative MLP development fold", bounds, flush=True)
    key = max(dev, key=lambda k: sum(row["wins"] for row in dev[k]))
    obj, half, steps = key
    print("Relative model chosen on development", key, flush=True)
    _, test = fixed_fold(data, .85, 1)
    result = {"recipe": list(key), "development": dev[key]}
    for name, frame, ix in (("historical", data, test), ("fresh", fresh, np.arange(len(fresh["time"])))):
        y, t = frame["y"][ix], frame["time"][ix]
        scores = walk(data, frame, ix, obj, half, device, (steps,), save=name == "fresh")[steps]
        result[name] = {"atr": evaluate(frame["atr"][ix], y, t), "mlp": evaluate(scores, y, t)}
        print(name, {k: (r["wins"], r["calls"], r["ap"]) for k, r in result[name].items()}, flush=True)
    (HERE / "relative_mlp_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
