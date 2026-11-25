"""Steps 3 and 5 of the bracket plan: do model scores rank candles better than a random ranking?

    python Models/Baselines/bracket_ranking.py --export Exports/training/mlp_2023-11-13_to_2026-11-13_bracket_short_tp1_sl1 \
        --scores mlp=runs/<run>/predictions/predictions.csv --baselines logistic,gbt

    # step 5: the same entries with and without anchor inputs, compared pair by pair
    python Models/Baselines/bracket_ranking.py --export ..._bracket_short_tp1_sl1 \
        --scores mlp_anchors=<with>/predictions.csv --scores mlp_no_anchors=<without>/predictions.csv \
        --compare mlp_anchors,mlp_no_anchors

Scores come from `python -m mlp_trades predict` (column p_enter) or any CSV with entry_id and a score column.
--baselines adds scorers fitted here on the 85 decision-time features (trained on the trainer's train
split, tuned or early-stopped on its validation split): logistic, gbt; a uniform random scorer is
always included. On the zeroed-anchor twin export the anchor features are 0, so the baselines run without
anchors there too.

Everything is measured on --period (default test) of the trainer's own 70/15/15 split:

- AUC, with a 95% interval and P(AUC <= 0.5) from a bootstrap over whole UTC days. Neighbouring candles
  share most of their future price path, so day blocks are the honest unit of resampling;
- top 1/5/10/20% of the period by score: win rate, mean net and the lift over the period's base rate, with
  day-bootstrap intervals;
- trading one position at a time: the score threshold is the validation quantile that passes 5/10/20%
  of candles; in time order a passing candle is entered when flat and the position is held until its exit.
  Compared with "enter every candle when flat" and with 500 random gates of the same pass rate: the
  percentile of the model's total net among the random gates;
- --compare A,B: the paired day-bootstrap difference A - B in AUC and in top-10% win rate.

Verdict per scorer (fixed in advance so step 6 has a rule): beats random when the AUC interval is above
0.5, the top-10% lift interval is above 0, and the replay beats at least 95% of the random gates at two of
the three pass rates. Writes ranking.json and ranking.csv to --output.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT / "Training/rocm-9070"), str(HERE)]

from mlp_trades.bitfinex import export_paths, prepare_bitfinex  # noqa: E402
from mlp_trades.data import mmap_array  # noqa: E402
from mlp_trades.schema import WIDTH  # noqa: E402
from mlp_trades.split import training_split  # noqa: E402

TOPS = (0.01, 0.05, 0.10, 0.20)
PASS_RATES = (0.05, 0.10, 0.20)


def auc(score, label):
    pos = int(label.sum())
    neg = len(label) - pos
    if not pos or not neg:
        return np.nan
    ranks = np.empty(len(score))
    ranks[np.argsort(score, kind="stable")] = np.arange(1, len(score) + 1)
    return (ranks[label].sum() - pos * (pos + 1) / 2) / (pos * neg)


class DayBootstrap:
    def __init__(self, days, draws, rng):
        uniq, inverse = np.unique(days, return_inverse=True)
        self.members = [np.flatnonzero(inverse == d) for d in range(len(uniq))]
        self.picks = rng.integers(0, len(uniq), size=(draws, len(uniq)))

    def samples(self):
        for pick in self.picks:
            yield np.concatenate([self.members[d] for d in pick])


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def replay(order, times, ends, passes):
    free, taken = -1, []
    for i in order:
        if passes[i] and times[i] >= free:
            taken.append(i)
            free = ends[i]
    return np.array(taken, dtype=np.int64)


def fit_baselines(names, data, splits, seed):
    from trade_baselines import HistogramGBT, decision_features, fit_logistic, logistic_probabilities

    timeline = mmap_array(data.paths.timeline, "timeline", (WIDTH,))
    bounded = data.schema.get("macd_normalization") == "x / sqrt(1 + x^2)"
    rows = np.concatenate([splits[s] for s in ("train", "validation", "test")])
    feats, _ = decision_features(timeline, data.history_rows[rows], data.times[rows], bounded)
    x = np.zeros((len(data.ids), feats.shape[1]))
    x[rows] = feats
    y = data.labels
    tr, va = splits["train"], splits["validation"]
    scores = {}
    if "logistic" in names:
        mean, scale = x[tr].mean(0), x[tr].std(0)
        scale[scale < 1e-12] = 1
        z = (x - mean) / scale
        best = None
        for l2 in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
            models = fit_logistic(z[tr], y[tr], 2, l2)
            a = auc(logistic_probabilities(models, z[va])[:, 1], y[va] == 1)
            print(f"logistic l2={l2:g} validation AUC {a:.4f}", flush=True)
            if best is None or a > best[0]:
                best = (a, models)
        s = np.full(len(y), np.nan)
        s[rows] = logistic_probabilities(best[1], z[rows])[:, 1]
        scores["logistic"] = s
    if "gbt" in names:
        model = HistogramGBT(2, seed=seed).fit(x[tr], y[tr], x[va], y[va])
        s = np.full(len(y), np.nan)
        s[rows] = model.predict_proba(x[rows])[:, 1]
        scores["gbt"] = s
        print(f"gbt: {model.rounds} rounds", flush=True)
    return scores


def read_scores(path, column, data):
    frame = pd.read_csv(path, dtype={"entry_id": str})
    column = column or ("p_enter" if "p_enter" in frame else "score")
    lookup = dict(zip(frame.entry_id, frame[column].astype(float)))
    s = np.array([lookup.get(i, np.nan) for i in data.ids])
    return s


def evaluate(name, score, data, splits, period, net, boot, rng, gates):
    idx = splits[period]
    s, y, r = score[idx], data.labels[idx] == 1, net[idx]
    if np.isnan(s).any():
        raise SystemExit(f"{name}: {int(np.isnan(s).sum())} {period} candles have no score")
    out = {"auc": float(auc(s, y))}
    draws = [auc(s[b], y[b]) for b in boot.samples()]
    out["auc_95"] = interval(draws)
    out["p_auc_le_0.5"] = float(np.mean(np.asarray(draws) <= 0.5))
    base = float(y.mean())
    out["base_win_rate"], out["base_mean_net"] = base, float(r.mean())
    order = np.argsort(-s, kind="stable")
    tops = {}
    for top in TOPS:
        k = max(1, int(round(top * len(s))))
        chosen = np.zeros(len(s), dtype=bool)
        chosen[order[:k]] = True
        lifts = []
        for b in boot.samples():
            c = chosen[b]
            if c.any():
                lifts.append(y[b][c].mean() - y[b].mean())
        tops[f"top_{top:.0%}"] = {"candles": k, "winners": int(y[chosen].sum()),
                                  "share_of_all_winners": float(y[chosen].sum() / max(y.sum(), 1)),
                                  "win_rate": float(y[chosen].mean()), "mean_net": float(r[chosen].mean()),
                                  "lift": float(y[chosen].mean() - base), "lift_95": interval(lifts)}
    out["tops"] = tops
    out["candles"], out["winners"] = int(len(s)), int(y.sum())
    # One position at a time; thresholds from validation scores.
    val = score[splits["validation"]]
    times, ends = data.times[idx], data.ends[idx]
    chron = np.argsort(times, kind="stable")
    every = replay(chron, times, ends, np.ones(len(s), dtype=bool))
    out["every_candle_when_flat"] = {"trades": len(every), "trade_wins": int(y[every].sum()),
                                     "total_net": float(r[every].sum()), "win_rate": float(y[every].mean())}
    replays = {}
    for rate in PASS_RATES:
        cut = np.quantile(val, 1 - rate)
        passes = s >= cut
        taken = replay(chron, times, ends, passes)
        realized = float(passes.mean())
        random_runs = [replay(chron, times, ends, rng.random(len(s)) < realized) for _ in range(gates)]
        random_totals = np.array([r[run].sum() for run in random_runs])
        random_trades = np.array([len(run) for run in random_runs])
        total = float(r[taken].sum()) if len(taken) else 0.0
        replays[f"pass_{rate:.0%}"] = {
            "threshold": float(cut), "realized_pass_rate": realized,
            # Every candle the gate passes, whether or not a position was already open.
            "passed": int(passes.sum()), "passed_winners": int(y[passes].sum()),
            "share_of_all_winners": float(y[passes].sum() / max(y.sum(), 1)),
            "passed_win_rate": float(y[passes].mean()) if passes.any() else np.nan,
            "passed_mean_net": float(r[passes].mean()) if passes.any() else np.nan,
            # One position at a time: what was actually traded.
            "trades": len(taken), "trade_wins": int(y[taken].sum()), "total_net": total,
            "net_per_trade": total / len(taken) if len(taken) else np.nan,
            "win_rate": float(y[taken].mean()) if len(taken) else np.nan,
            "random_gate_total_net_p5_p50_p95": np.percentile(random_totals, [5, 50, 95]).round(3).tolist(),
            "random_gate_trades_median": float(np.median(random_trades)),
            "random_gate_net_per_trade_median": float(np.median(random_totals / np.maximum(random_trades, 1))),
            "beats_random_share": float(np.mean(total > random_totals)),
        }
    out["one_position"] = replays
    beats = sum(v["beats_random_share"] >= 0.95 for v in replays.values())
    out["verdict_beats_random"] = bool(out["auc_95"][0] > 0.5 and tops["top_10%"]["lift_95"][0] > 0 and beats >= 2)
    return out


def compare(a, b, scores, data, splits, period, boot):
    idx = splits[period]
    y = data.labels[idx] == 1
    sa, sb = scores[a][idx], scores[b][idx]
    k = max(1, int(round(0.10 * len(idx))))
    ca = np.zeros(len(idx), dtype=bool)
    cb = np.zeros(len(idx), dtype=bool)
    ca[np.argsort(-sa, kind="stable")[:k]] = True
    cb[np.argsort(-sb, kind="stable")[:k]] = True
    d_auc, d_top = [], []
    for s in boot.samples():
        d_auc.append(auc(sa[s], y[s]) - auc(sb[s], y[s]))
        if ca[s].any() and cb[s].any():
            d_top.append(y[s][ca[s]].mean() - y[s][cb[s]].mean())
    d_auc, d_top = np.asarray(d_auc), np.asarray(d_top)
    return {"pair": f"{a} - {b}", "auc_difference": float(auc(sa, y) - auc(sb, y)), "auc_difference_95": interval(d_auc),
            "p_a_not_better_auc": float(np.mean(d_auc <= 0)),
            "top10_win_rate_difference": float(y[ca].mean() - y[cb].mean()), "top10_difference_95": interval(d_top),
            "p_a_not_better_top10": float(np.mean(d_top <= 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--scores", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--score-column")
    parser.add_argument("--baselines", default="", help="comma list: logistic,gbt")
    parser.add_argument("--compare", action="append", default=[], metavar="A,B")
    parser.add_argument("--period", choices=("validation", "test"), default="test")
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--gates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export = args.export.resolve()
    output = args.output or export / "ranking"
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_bitfinex(export_paths(export), training=True)
    if json.loads((export / "manifest.json").read_text(encoding="utf-8")).get("entry_rule") != "every_m5_close":
        raise SystemExit("Expected an export from bracket_targets.py")
    splits, _ = training_split(data, 0.7, 0.15)
    net = data.returns
    rng = np.random.default_rng(args.seed)

    scores = {"random": rng.random(len(data.ids))}
    for item in args.scores:
        name, path = item.split("=", 1)
        scores[name] = read_scores(path, args.score_column, data)
    names = [n.strip() for n in args.baselines.split(",") if n.strip()]
    if names:
        scores.update(fit_baselines(names, data, splits, args.seed))

    days = pd.to_datetime(data.times[splits[args.period]], unit="s", utc=True).strftime("%Y-%m-%d").to_numpy()
    boot = DayBootstrap(days, args.draws, rng)
    report = {"export": export.name, "period": args.period, "candles": int(len(splits[args.period])),
              "days": int(len(np.unique(days))), "scorers": {}, "comparisons": []}
    rows = []
    for name, score in scores.items():
        result = evaluate(name, score, data, splits, args.period, net, boot, rng, args.gates)
        report["scorers"][name] = result
        rows.append({"scorer": name, "auc": result["auc"], "auc_lo": result["auc_95"][0], "auc_hi": result["auc_95"][1],
                     **{f"{k}_win": v["win_rate"] for k, v in result["tops"].items()},
                     **{f"{k}_lift_lo": v["lift_95"][0] for k, v in result["tops"].items()},
                     **{f"{k}_net": v["total_net"] for k, v in result["one_position"].items()},
                     **{f"{k}_beats": v["beats_random_share"] for k, v in result["one_position"].items()},
                     "verdict": result["verdict_beats_random"]})
        print(f"{name}: AUC {result['auc']:.4f} {result['auc_95']}, top10% win "
              f"{result['tops']['top_10%']['win_rate']:.3f} vs base {result['base_win_rate']:.3f}, "
              f"beats random: {result['verdict_beats_random']}", flush=True)
    for pair in args.compare:
        a, b = pair.split(",")
        report["comparisons"].append(compare(a, b, scores, data, splits, args.period, boot))
        print(json.dumps(report["comparisons"][-1]), flush=True)
    (output / "ranking.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(rows).to_csv(output / "ranking.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
