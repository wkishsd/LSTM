"""Per-anchor activation counts, fill rates, win rates and net returns from an anchor export.

Writes a CSV with one row per main anchor, sorted by win rate; the readable copy lives in
Google Sheets. If the random-minute control has run, its results are added as a reference row: the
same entry and exit rules at random M5 times with no main anchor.

    python Pipelines/Bitfinex/anchor_statistics.py
"""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = [
    ("anchor", "Anchor"),
    ("activations", "Activations"),
    ("solo_activations", "Alone"),
    ("filled", "Filled"),
    ("fill_rate", "Fill rate"),
    ("winners", "Winners"),
    ("win_rate", "Win rate"),
    ("win_rate_se", "± (1 SE)"),
    ("win_rate_vs_all", "vs all anchors"),
    ("solo_win_rate", "Win rate alone"),
    ("win_rate_first_half", "Win rate 1st half"),
    ("win_rate_second_half", "Win rate 2nd half"),
    ("net_return_pct", "Net %/series"),
    ("gross_return_pct", "Gross %/series"),
    ("trades_per_series", "Trades/series"),
]


def stats(rows, halfway):
    filled = [r for r in rows if r["initial_order_status"] == "filled" and r["binary_target"] != ""]
    wins = sum(int(r["binary_target"]) for r in filled)
    n = len(filled)
    rate = wins / n if n else math.nan
    halves = [[r for r in filled if (r["entry_at"] < halfway) == first] for first in (True, False)]
    return {
        "activations": len(rows), "filled": n, "fill_rate": n / len(rows) if rows else math.nan,
        "winners": wins, "win_rate": rate,
        "win_rate_se": math.sqrt(rate * (1 - rate) / n) if n else math.nan,
        "win_rate_first_half": mean([int(r["binary_target"]) for r in halves[0]]),
        "win_rate_second_half": mean([int(r["binary_target"]) for r in halves[1]]),
        "net_return_pct": mean([float(r["net_return_pct"]) for r in filled]),
        "gross_return_pct": mean([float(r["target_gross_return_pct"]) for r in filled]),
        "trades_per_series": mean([float(r["trades_count"]) for r in filled]),
    }


def mean(values):
    return sum(values) / len(values) if values else math.nan


def fmt(key, value):
    if isinstance(value, str):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if key in ("fill_rate", "win_rate", "win_rate_se", "solo_win_rate", "win_rate_first_half", "win_rate_second_half"):
        return f"{100 * value:.1f}%"
    if key == "win_rate_vs_all":
        return f"{value:.2f}×"
    if key in ("net_return_pct", "gross_return_pct"):
        return f"{value:+.3f}%"
    if key == "trades_per_series":
        return f"{value:.2f}"
    return f"{value:,}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anchor-export", type=Path,
                        default=ROOT / "Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd")
    parser.add_argument("--control", type=Path, default=ROOT / "Exports/training/random_minute_control")
    parser.add_argument("--output", type=Path, default=ROOT / "Exports/anchor_statistics")
    args = parser.parse_args()

    with (args.anchor_export / "entry_targets.csv").open(encoding="utf-8-sig", newline="") as stream:
        entries = list(csv.DictReader(stream))
    entries.sort(key=lambda r: r["entry_at"])
    halfway = entries[len(entries) // 2]["entry_at"]
    by_anchor, solo = defaultdict(list), defaultdict(list)
    for row in entries:
        anchors = [int(a) for a in row["main_anchor_numbers"].split(";") if a]
        for anchor in anchors:
            by_anchor[anchor].append(row)
        if len(anchors) == 1:
            solo[anchors[0]].append(row)

    overall = stats(entries, halfway)
    table = []
    for anchor in sorted(by_anchor):
        row = {"anchor": f"anchor_{anchor:02d}", **stats(by_anchor[anchor], halfway)}
        row["solo_activations"] = len(solo[anchor])
        row["solo_win_rate"] = stats(solo[anchor], halfway)["win_rate"] if solo[anchor] else math.nan
        row["win_rate_vs_all"] = row["win_rate"] / overall["win_rate"]
        table.append(row)
    table.sort(key=lambda r: -r["win_rate"])
    summary = [{"anchor": "All anchor entries", **overall, "solo_activations": "", "solo_win_rate": math.nan,
                "win_rate_vs_all": 1.0}]
    control_file = args.control / "random_entry_targets.csv"
    if control_file.exists():
        with control_file.open(encoding="utf-8-sig", newline="") as stream:
            random_rows = sorted(csv.DictReader(stream), key=lambda r: r["entry_at"])
        reference = stats(random_rows, halfway)
        summary.append({"anchor": "Random times, no anchor", **reference, "solo_activations": "",
                        "solo_win_rate": math.nan, "win_rate_vs_all": reference["win_rate"] / overall["win_rate"]})

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "anchor_statistics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([title for _, title in COLUMNS])
        for row in table + summary:
            writer.writerow([fmt(key, row.get(key)) for key, _ in COLUMNS])

    print(f"Wrote {args.output / 'anchor_statistics.csv'} ({len(table)} anchors, halves split at {halfway[:10]})")


if __name__ == "__main__":
    main()
