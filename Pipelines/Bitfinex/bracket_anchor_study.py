"""Step 4 of the bracket plan: how each of the 35 anchors relates to the bracket win rate.

    python Pipelines/Bitfinex/bracket_anchor_study.py --export Exports/training/mlp_2023-11-13_to_2026-11-13_bracket_short_tp1_sl1

Reads an export from bracket_targets.py (every M5 close labelled "take profit before stop") and recomputes
all 35 anchors of the registry on native 5m candles with the collector's own rules. For every anchor, two
flags: fired on the decision candle ("now") and fired on any of the 12 candles before it ("last_60m").
Also: any main anchor now (anchor candles against all other candles, the question behind the old
random-minute control) and the number of anchors firing now.

Per flag and per period of the trainer's 70/15/15 split:
- share of candles with the flag, win rate with / without it, the difference (lift, in points), the mean net
  with / without it, and the difference in mean net;
- a 95% interval and a two-sided p-value for both differences from a bootstrap over whole UTC days.
  Neighbouring 5m candles share most of their future price path, so their labels are strongly correlated;
  resampling days keeps that correlation inside each draw and gives honest intervals, where treating
  every candle as independent would make tiny effects look significant;
- Benjamini-Hochberg q-values over all flags, per period, for the multiple testing of 70+ flags.

A flag counts as consistent when its win-rate lift has the same sign in all three periods and is
significant (q < 0.05) on the train period; the same for the net difference. Writes anchor_study.csv
and summary.json to --output (default: the export's folder / anchor_study).
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

from bracket_targets import native_candles  # noqa: E402
from mlp_trades.bitfinex import export_paths, prepare_bitfinex  # noqa: E402
from mlp_trades.split import training_split  # noqa: E402

PERIODS = ("train", "validation", "test")
LOOKBACK = 12


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def day_bootstrap(day, flags, win, net, draws, rng):
    """Bootstrap over days of (win rate on - off) and (mean net on - off) for every flag column."""
    days, d = np.unique(day, return_inverse=True)
    D = len(days)
    f = flags.astype(np.float64)

    def per_day(weights):
        return np.bincount(d, weights=weights, minlength=D)
    n_on = np.column_stack([per_day(f[:, j]) for j in range(f.shape[1])])
    w_on = np.column_stack([per_day(f[:, j] * win) for j in range(f.shape[1])])
    s_on = np.column_stack([per_day(f[:, j] * net) for j in range(f.shape[1])])
    n_all, w_all, s_all = per_day(np.ones(len(day))), per_day(win), per_day(net)
    counts = rng.multinomial(D, np.full(D, 1 / D), size=draws).astype(np.float32)

    def stat(c):
        N_on, W_on, S_on = c @ n_on, c @ w_on, c @ s_on
        N_all, W_all, S_all = c @ n_all, c @ w_all, c @ s_all
        N_off = N_all[:, None] - N_on
        with np.errstate(invalid="ignore", divide="ignore"):
            lift = W_on / N_on - (W_all[:, None] - W_on) / N_off
            dnet = S_on / N_on - (S_all[:, None] - S_on) / N_off
        return lift, dnet

    point = stat(np.ones((1, D), dtype=np.float32))
    boot = stat(counts)
    out = {}
    for name, p, b in (("lift", point[0][0], boot[0]), ("net_diff", point[1][0], boot[1])):
        lo, hi = np.nanpercentile(b, [2.5, 97.5], axis=0)
        below = np.nanmean(b <= 0, axis=0)
        above = np.nanmean(b >= 0, axis=0)
        out[name] = p
        out[f"{name}_lo"], out[f"{name}_hi"] = lo, hi
        out[f"{name}_p"] = np.clip(2 * np.minimum(below, above), 1 / draws, 1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--min-share", type=float, default=0.002, help="skip flags rarer than this on the train period")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    export = args.export.resolve()
    output = args.output or export / "anchor_study"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("entry_rule") != "every_m5_close":
        raise SystemExit("Expected an export from bracket_targets.py")

    data = prepare_bitfinex(export_paths(export), training=True)
    splits, _ = training_split(data, 0.7, 0.15)
    entries = pd.read_csv(export / "entry_targets.csv", dtype={"entry_id": str}).set_index("entry_id")

    pipeline, native = native_candles(manifest)
    flags = pipeline.detect_anchor_events(pipeline.calculate_native(native["5m"]), "5m")
    columns = [c for c in flags.columns if c.startswith("anchor_")]
    recent = flags[columns].shift(1).rolling(LOOKBACK, min_periods=1).max()
    main_ids = [f"anchor_{n:02d}" for n in pipeline.MAIN_ANCHOR_IDS]
    groups = {n: "main" for n in pipeline.MAIN_ANCHOR_IDS}
    for key, (_, ids) in pipeline.ADDITIONAL_ANCHOR_GROUPS.items():
        groups.update({n: key for n in ids})

    rng = np.random.default_rng(args.seed)
    rows, summary = [], {"export": export.name, "bracket": manifest.get("bracket"), "periods": {}}
    for period in PERIODS:
        idx = splits[period]
        ids = data.ids[idx]
        sub = entries.loc[ids]
        when = pd.DatetimeIndex(pd.to_datetime(sub.entry_at, utc=True))
        now = flags[columns].reindex(when)
        if now.isna().any().any():
            raise SystemExit(f"{int(now.isna().any(axis=1).sum())} {period} candles have no anchor row")
        last = recent.reindex(when).fillna(0)
        table = {f"{c}.now": now[c].to_numpy() for c in columns}
        table.update({f"{c}.last_{LOOKBACK * 5}m": last[c].to_numpy() for c in columns})
        table["any_main.now"] = now[main_ids].max(axis=1).to_numpy()
        table["any_anchor.now"] = now[columns].max(axis=1).to_numpy()
        count = now[columns].sum(axis=1).to_numpy()
        table["two_or_more_anchors.now"] = (count >= 2).astype(float)
        names = list(table)
        matrix = np.column_stack([table[n] for n in names]).astype(np.float32)
        win = data.labels[idx].astype(np.float32)
        net = sub.net_return_pct.to_numpy(dtype=np.float32)
        day = when.strftime("%Y-%m-%d").to_numpy()
        share = matrix.mean(axis=0)
        stats = day_bootstrap(day, matrix, win, net, args.draws, rng)
        for name in ("lift", "net_diff"):
            stats[f"{name}_q"] = bh(np.nan_to_num(stats[f"{name}_p"], nan=1.0))
        on = matrix.astype(bool)
        for j, name in enumerate(names):
            m = on[:, j]
            number = int(name[7:9]) if name.startswith("anchor_") else None
            rows.append({"flag": name, "anchor_group": groups.get(number, "combined"), "period": period,
                         "share": float(share[j]), "candles_on": int(m.sum()),
                         "win_on": float(win[m].mean()) if m.any() else np.nan,
                         "win_off": float(win[~m].mean()) if (~m).any() else np.nan,
                         "net_on": float(net[m].mean()) if m.any() else np.nan,
                         "net_off": float(net[~m].mean()) if (~m).any() else np.nan,
                         **{k: float(v[j]) for k, v in stats.items()}})
        summary["periods"][period] = {"candles": len(idx), "days": int(len(np.unique(day))),
                                      "win_rate": float(win.mean()), "mean_net": float(net.mean())}
        print(f"{period}: {len(idx):,} candles over {len(np.unique(day))} days", flush=True)

    table = pd.DataFrame(rows)
    wide = table.pivot(index="flag", columns="period")
    train_share = wide[("share", "train")]
    keep = train_share >= args.min_share
    result = []
    for kind in ("lift", "net_diff"):
        values = wide[kind][list(PERIODS)]
        same = (np.sign(values).nunique(axis=1) == 1) & values.ne(0).all(axis=1)
        sig = wide[(f"{kind}_q", "train")] < 0.05
        good = same & sig & keep
        wide[("consistent", kind)] = good
        result.append((kind, sorted(wide.index[good])))
    summary["consistent_win_rate_lift"] = result[0][1]
    summary["consistent_net_difference"] = result[1][1]
    summary["flags_tested"] = int(keep.sum())
    table.to_csv(output / "anchor_study.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    show = table[table.flag.isin(wide.index[keep])].pivot(index="flag", columns="period", values=["lift", "net_diff"])
    order = (show["lift"].abs().min(axis=1)).sort_values(ascending=False).index
    pd.set_option("display.width", 220)
    show = show.loc[order].head(25)
    show["lift"] = show["lift"] * 100   # win-rate points; net_diff is already in % per trade
    print("lift in win-rate points, net_diff in % per trade")
    print(show.round(3).to_string())
    print(json.dumps({k: summary[k] for k in ("consistent_win_rate_lift", "consistent_net_difference", "flags_tested")}, indent=2))


if __name__ == "__main__":
    main()
