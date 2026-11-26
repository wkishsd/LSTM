"""Random entries, anchor entries and the MLP on one bracket export, with the same few metrics.

    python Models/Baselines/bracket_comparison.py \
        --export Exports/training/mlp_2023-11-13_to_2026-11-13_bracket_short_tp1_sl1 \
        --predictions Training/rocm-9070/runs/<run>/predictions/predictions.csv

Every 5m candle of the export is a possible short entry labelled by the bracket (take profit before stop).
A "run" is a rule that picks some candles of the period; for each run:

- entries: candles picked; winners: picked candles that hit the take profit first;
- win rate: winners / entries; winners caught: winners / all winners of the period;
- accuracy: the run read as an enter / skip call on every candle, share of calls that were right
  (picked winners plus skipped losers, over all candles). Skipping everything scores 1 - base win rate;
- net per trade: mean result of the picked candles after fees, in %;
- AUC: for MLP rows only, how well the score ranks winners above losers over the whole period.

Runs: every candle; random picks (expected values at the anchors' pass rate and at 10%); every candle where
any of the 35 anchors fired, and where any of the 18 main ones fired; the MLP's own calls (p_enter >= 0.5);
the MLP's top 5 / 10 / 20% (thresholds from the validation period) and the MLP at the anchors' pass rate.
Per-anchor rows: candles where that anchor fired. Default period: test of the trainer's 70/15/15 split.
Writes comparison.csv and per_anchor.csv to --output.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "Training/rocm-9070"), str(ROOT / "Pipelines/Bitfinex"), str(HERE)]

from bracket_ranking import auc, read_scores  # noqa: E402
from bracket_targets import native_candles  # noqa: E402
from mlp_trades.bitfinex import export_paths, prepare_bitfinex  # noqa: E402
from mlp_trades.split import training_split  # noqa: E402


def metrics(name, picked, y, r, kind, score=None):
    n, entries = len(y), int(picked.sum())
    winners = int(y[picked].sum())
    return {"run": name, "kind": kind, "entries": entries, "entry_share": entries / n, "winners": winners,
            "win_rate": winners / entries if entries else np.nan,
            "winners_caught": winners / max(int(y.sum()), 1),
            "accuracy": (winners + int((~y[~picked]).sum())) / n,
            "net_per_trade": float(r[picked].mean()) if entries else np.nan,
            "auc": float(auc(score, y)) if score is not None else np.nan}


def expected_random(name, rate, y, r):
    base = float(y.mean())
    n = len(y)
    return {"run": name, "kind": "random", "entries": int(round(rate * n)), "entry_share": rate,
            "winners": int(round(rate * y.sum())), "win_rate": base, "winners_caught": rate,
            "accuracy": rate * base + (1 - rate) * (1 - base), "net_per_trade": float(r.mean()), "auc": 0.5}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--period", choices=("validation", "test"), default="test")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export = args.export.resolve()
    output = args.output or export / "comparison"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    data = prepare_bitfinex(export_paths(export), training=True)
    splits, _ = training_split(data, 0.7, 0.15)
    idx = splits[args.period]
    y, r = data.labels[idx] == 1, data.returns[idx]
    entries = pd.read_csv(export / "entry_targets.csv", dtype={"entry_id": str}).set_index("entry_id")
    when = pd.DatetimeIndex(pd.to_datetime(entries.loc[data.ids[idx], "entry_at"], utc=True))

    pipeline, native = native_candles(manifest)
    flags = pipeline.detect_anchor_events(pipeline.calculate_native(native["5m"]), "5m")
    columns = sorted(c for c in flags.columns if c.startswith("anchor_"))
    now = flags[columns].reindex(when)
    if now.isna().any().any():
        raise SystemExit("Some candles have no anchor row")
    now = now.to_numpy(dtype=bool)
    main_ids = [columns.index(f"anchor_{n:02d}") for n in pipeline.MAIN_ANCHOR_IDS]
    groups = {f"anchor_{n:02d}": "Main" for n in pipeline.MAIN_ANCHOR_IDS}
    for key, (_, ids) in pipeline.ADDITIONAL_ANCHOR_GROUPS.items():
        groups.update({f"anchor_{n:02d}": "Additional · " + key.replace("additional_", "").replace("_", " ")
                       for n in ids})

    score_all = read_scores(args.predictions, None, data)
    score = score_all[idx]
    validation = score_all[splits["validation"]]
    if np.isnan(score).any() or np.isnan(validation).any():
        raise SystemExit("The predictions do not cover every candle of the period")
    any35, any18 = now.any(axis=1), now[:, main_ids].any(axis=1)

    rows = [metrics("Every candle", np.ones(len(y), dtype=bool), y, r, "all"),
            expected_random(f"Random · same share as anchors ({any35.mean():.0%})", any35.mean(), y, r),
            expected_random("Random · 10% of candles", 0.10, y, r),
            metrics("Anchors · any of the 35 fires", any35, y, r, "anchors"),
            metrics("Anchors · any of the 18 main fires", any18, y, r, "anchors"),
            metrics("MLP · its own calls (p ≥ 0.5)", score >= 0.5, y, r, "mlp", score)]
    for rate in (0.05, 0.10, 0.20):
        rows.append(metrics(f"MLP · top {rate:.0%} (validation threshold)", score >= np.quantile(validation, 1 - rate),
                            y, r, "mlp", score))
    share = float(any35.mean())
    rows.append(metrics(f"MLP · same share as anchors ({share:.0%})", score >= np.quantile(validation, 1 - share),
                        y, r, "mlp", score))
    comparison = pd.DataFrame(rows)
    per_anchor = pd.DataFrame([{**metrics(c, now[:, j], y, r, "anchor"), "group": groups[c]}
                               for j, c in enumerate(columns)]).sort_values("win_rate", ascending=False)
    comparison.to_csv(output / "comparison.csv", index=False)
    per_anchor.to_csv(output / "per_anchor.csv", index=False)
    info = {"export": export.name, "predictions": str(args.predictions), "period": args.period,
            "from": str(when.min()), "to": str(when.max()), "candles": int(len(idx)), "winners": int(y.sum()),
            "base_win_rate": float(y.mean()), "break_even": manifest.get("bracket", {})}
    (output / "comparison.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    pd.set_option("display.width", 250)
    print(json.dumps(info, default=str))
    print(comparison.round(4).to_string(index=False))
    print(per_anchor.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
