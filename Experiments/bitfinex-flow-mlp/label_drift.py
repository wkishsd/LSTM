"""Measure trade-label rates within earlier-defined ATR bands as time advances."""

from datetime import datetime, timezone
import json

import numpy as np

from train_flow import HERE, fixed_fold, view


def main():
    data = view(HERE / "features.npz")
    earlier, _ = fixed_fold(data, .4, .55)
    atr = data["atr"]
    edges = np.quantile(atr[earlier], [.25, .5, .75])
    strata = np.searchsorted(edges, atr, side="right")
    stats = []
    for start, stop in ((.40, .55), (.55, .70), (.70, .85), (.85, 1.0)):
        _, ix = fixed_fold(data, start, stop)
        bands = []
        for band in range(4):
            y = data["y"][ix[strata[ix] == band]]
            bands.append({"entries": len(y), "winner_rate": float(y.mean())})
        first, last = (datetime.fromtimestamp(int(t), timezone.utc).date().isoformat()
                       for t in data["time"][ix[[0, -1]]])
        stats.append({"from_utc": first, "to_utc": last, "entries": len(ix),
                      "winner_rate": float(data["y"][ix].mean()), "atr_bands": bands})
    result = {"atr5_over_price_cutpoints": edges.tolist(), "periods": stats}
    text = json.dumps(result, indent=2) + "\n"
    (HERE / "label_drift.json").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
