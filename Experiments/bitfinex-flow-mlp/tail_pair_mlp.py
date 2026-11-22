"""Learn to reorder high-ATR candidates with a bounded neural adjustment."""

from collections import defaultdict
import json

import numpy as np
import torch

from train_flow import HERE, SEEDS, WEEK, evaluate, fixed_fold, train, view


STEPS = (40, 160, 400)
SHRINK = (.07, .2, .5)


def make_gate(history, fresh):
    times = np.concatenate((history["time"], fresh["time"]))
    atr = np.concatenate((history["atr"], fresh["atr"]))
    days = times // 86400
    gate = np.zeros(len(times), dtype=np.float32)
    high = np.zeros(len(times), dtype=bool)
    for day in np.unique(days):
        lo, start, stop = np.searchsorted(days, [day - 28, day, day + 1])
        if start - lo < 100:
            continue
        q60, q85 = np.quantile(atr[lo:start], [.60, .85])
        gate[start:stop] = np.clip((atr[start:stop] - q60) / max(q85 - q60, 1e-8), 0, 1)
        high[start:stop] = atr[start:stop] >= q60
    n = len(history["time"])
    history["gate"], fresh["gate"] = gate[:n], gate[n:]
    history["atr_candidate"] = high[:n]


def walk(history, dest, ix, half, device, steps=STEPS, weights=SHRINK, checkpoints=None):
    periods = (dest["time"][ix] + 3 * 86400) // (4 * WEEK)
    result = {(step, weight): np.empty(len(ix)) for step in steps for weight in weights}
    for block in np.unique(periods):
        mask = periods == block
        test = ix[mask]
        tr = np.flatnonzero((history["available"] < dest["start"][test].min()) & history["atr_candidate"])
        fits = [train(history, tr, dest, test, "combined", "pair", seed, device, steps,
                      checkpoint=HERE / f"{checkpoints}_block{block}_seed{seed}.pt" if checkpoints else None,
                      half_life=half) for seed in SEEDS]
        for step in steps:
            residual = np.mean([score[step] for _, score in fits], axis=0)
            correction = dest["gate"][test] * residual.clip(-1, 1)
            for weight in weights:
                result[step, weight][mask] = fits[0][0] + weight * correction
    return result


def main():
    torch.set_num_threads(3)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = view(HERE / "features.npz")
    fresh = view(HERE / "fresh_features.npz")
    make_gate(data, fresh)
    dev = defaultdict(list)
    references = []
    for bounds in ((.40, .55), (.55, .70), (.70, .85)):
        _, ix = fixed_fold(data, *bounds)
        y, t = data["y"][ix], data["time"][ix]
        references.append(evaluate(data["atr"][ix], y, t)["wins"])
        for half in (45, 180):
            for (step, weight), score in walk(data, data, ix, half, device).items():
                dev[half, step, weight].append(evaluate(score, y, t))
        print("Tail-pair development fold", bounds, flush=True)

    def rank(key):
        lift = [r["wins"] - a for a, r in zip(references, dev[key])]
        return min(lift), sum(lift)

    selected = max(dev, key=rank)
    half, step, weight = selected
    print("Tail-pair recipe selected on development", selected, rank(selected), flush=True)
    _, test = fixed_fold(data, .85, 1)
    result = {"recipe": list(selected), "development_atr_wins": references, "development": dev[selected]}
    for label, frame, ix in (("historical", data, test), ("fresh", fresh, np.arange(len(fresh["time"])))):
        t, y = frame["time"][ix], frame["y"][ix]
        checkpoints = "tail_pair_fresh" if label == "fresh" else None
        score = walk(data, frame, ix, half, device, (step,), (weight,), checkpoints)[step, weight]
        result[label] = {"atr": evaluate(frame["atr"][ix], y, t), "mlp": evaluate(score, y, t)}
        print(label, {k: (r["wins"], r["calls"], r["ap"]) for k, r in result[label].items()}, flush=True)
    (HERE / "tail_pair_mlp_results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
