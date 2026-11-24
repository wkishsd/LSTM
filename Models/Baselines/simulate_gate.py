"""Chronological one-position-at-a-time trading with a boosted-tree entry gate, walk-forward.

    python Models/Baselines/simulate_gate.py --export current_rules_15m --threshold 1.0

For every 3-month block from --wf-start: boosted trees (trade_baselines.HistogramGBT, 85 decision-time
features) are trained on all earlier filled entries with the target net / ATR 14 on M15 >= --threshold,
early-stopped on the 3 months before the block (purged as in gbt_robustness.py). The score cut-off is the
--band quantile of the validation scores, so the gate passes roughly the top (1 - band) of entries.

Then the block is replayed in time order over every anchor entry, including orders that never fill
(the document's is_flat rule): when flat and the entry's score passes the cut-off, the order is placed and
the position stays busy until the entry's label_available_at (series end, or cancellation for an unfilled
order); anchors firing meanwhile are skipped. Each taken entry adds its series net return in % after fees
(0 for an unfilled order), one unit of notional per series, no compounding.

Controls on the same block: 'all' takes every anchor when flat; 'random' passes a random entry with the
same pass rate as the gate (mean of 20 draws). Writes trades.csv and summary.csv.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "Training/rocm-9070"), str(HERE)]

from gbt_robustness import walk_forward  # noqa: E402
from mlp_trades.bitfinex import apply_risk_target, export_paths, prepare_bitfinex, timestamp  # noqa: E402
from mlp_trades.data import mmap_array  # noqa: E402
from mlp_trades.schema import WIDTH  # noqa: E402
from trade_baselines import HistogramGBT, decision_features  # noqa: E402


def replay(order, times, busy_until, net, passes):
    """Indices taken when flat and passing, in time order."""
    taken, free_at = [], -1
    for i in order:
        if times[i] >= free_at and passes[i]:
            taken.append(i)
            free_at = busy_until[i]
    return np.array(taken, dtype=np.int64)


def stats(taken, net, filled):
    pnl = net[taken]
    curve = np.cumsum(pnl)
    drawdown = float((np.maximum.accumulate(np.concatenate(([0.0], curve)))[1:] - curve).max()) if len(curve) else 0.0
    f = filled[taken]
    return {"orders": len(taken), "filled": int(f.sum()), "total_net": float(pnl.sum()),
            "net_per_filled": float(pnl[f].mean()) if f.any() else np.nan,
            "win_rate": float((pnl[f] > 0).mean()) if f.any() else np.nan, "max_drawdown": drawdown}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", default="current_rules_15m")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--bands", nargs="+", type=float, default=[0.7, 0.8, 0.85, 0.9, 0.95])
    parser.add_argument("--wf-start", default="2025-05-01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/training/gate_simulation")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    directory = ROOT / "Exports/training" / f"mlp_2023-11-13_to_2026-11-13_{args.export}"
    paths = export_paths(directory)
    labelled = apply_risk_target(prepare_bitfinex(paths, training=True), directory / "risk_targets.csv", args.threshold)
    scoring = prepare_bitfinex(paths, training=False)  # every anchor entry, filled or not
    if not np.array_equal(labelled.ids, scoring.ids):
        raise SystemExit("Entry order differs between the training and scoring loads")
    timeline = mmap_array(str(directory / "feature_timeline.npy"), "timeline", (WIDTH,))
    bounded = '"x / sqrt(1 + x^2)"' in (directory / "manifest.json").read_text(encoding="utf-8")
    feats, _ = decision_features(timeline, scoring.history_rows[scoring.valid], scoring.times[scoring.valid], bounded)
    x = np.zeros((len(scoring.ids), feats.shape[1]))
    x[scoring.valid] = feats

    entries = pd.read_csv(paths.entries, dtype={"entry_id": str},
                          usecols=["entry_id", "initial_order_status", "label_available_at", "net_return_pct"])
    if not np.array_equal(entries.entry_id.to_numpy(), labelled.ids):
        raise SystemExit("entry_targets.csv order differs from the loaded entries")
    filled = entries.initial_order_status.eq("filled").to_numpy()
    net = np.where(filled, entries.net_return_pct.fillna(0).to_numpy(dtype=float), 0.0)
    busy_until = np.array([timestamp(v) if isinstance(v, str) else 0 for v in entries.label_available_at])
    times = scoring.times
    y = labelled.labels
    rng = np.random.default_rng(args.seed)

    summary, trades = [], []
    for fold, tr, va, te in walk_forward(labelled, args.wf_start):
        model = HistogramGBT(2, seed=args.seed).fit(x[tr], y[tr], x[va], y[va])
        val_scores = model.predict_proba(x[va])[:, 1]
        start, end = times[te].min(), times[te].max() + 1
        block = scoring.valid[(times[scoring.valid] >= start) & (times[scoring.valid] < end)]
        order = block[np.argsort(times[block], kind="stable")]
        score = np.zeros(len(times))
        score[block] = model.predict_proba(x[block])[:, 1]
        everything = np.ones(len(times), dtype=bool)
        base = stats(replay(order, times, busy_until, net, everything), net, filled)
        summary.append({"fold": fold, "gate": "all", "band": np.nan, "pass_rate": 1.0, "rounds": model.rounds, **base})
        for band in args.bands:
            cut = np.quantile(val_scores, band)
            passes = score >= cut
            taken = replay(order, times, busy_until, net, passes)
            rec = stats(taken, net, filled)
            pass_rate = float(passes[block].mean())
            summary.append({"fold": fold, "gate": "trees", "band": band, "pass_rate": pass_rate,
                            "rounds": model.rounds, **rec})
            draws = [stats(replay(order, times, busy_until, net, rng.random(len(times)) < pass_rate), net, filled)
                     for _ in range(20)]
            summary.append({"fold": fold, "gate": "random", "band": band, "pass_rate": pass_rate, "rounds": model.rounds,
                            **{k: float(np.nanmean([d[k] for d in draws])) for k in draws[0]}})
            trades += [{"fold": fold, "band": band, "entry_id": labelled.ids[i], "time": int(times[i]),
                        "filled": bool(filled[i]), "net": float(net[i]), "score": float(score[i])} for i in taken]
        print(fold, "done", flush=True)
    summary = pd.DataFrame(summary)
    summary.to_csv(args.output / "summary.csv", index=False)
    pd.DataFrame(trades).to_csv(args.output / "trades.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    totals = summary.groupby(["gate", "band"], dropna=False)[["orders", "filled", "total_net", "max_drawdown"]].sum()
    totals["net_per_filled"] = totals.total_net / totals.filled
    print(totals.round(4).to_string())


if __name__ == "__main__":
    main()
