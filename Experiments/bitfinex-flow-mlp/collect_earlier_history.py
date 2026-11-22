"""Build a separate timeline export before the dates used for model development."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "Pipelines/Bitfinex/bitfinex_indicators_v2.1.py"
spec = spec_from_file_location("bitfinex_research_collector", SOURCE)
collector = module_from_spec(spec)
sys.modules[spec.name] = collector
spec.loader.exec_module(collector)


def main():
    output = HERE / "earlier_export"
    output.mkdir(exist_ok=True)
    start, end, cutoff = map(pd.Timestamp, ("2023-03-01T00:00:00Z", "2023-11-01T00:00:00Z", "2023-11-02T00:00:00Z"))
    window = collector.ML_WINDOW_MINUTES
    settings = collector.ExitSettings()
    native, cutoff = collector.collect_training_history(start, cutoff, HERE / "earlier_candle_cache", window)
    entries, _ = collector.filter_main_anchor_entries(native, 0, start, end, settings.initial_entry_offset_down)
    print(f"Collected {len(entries):,} earlier entries", flush=True)
    audit = collector.label_mlp_entries(entries, native, settings, output, workers=4).reindex(entries.index)
    timeline, names, rows = collector.build_mlp_feature_timeline(native, entries, window)
    np.save(output / "feature_timeline.npy", timeline)
    audit["feature_end_row"] = rows
    audit.to_csv(output / "entry_targets.csv")
    manifest = {
        "ticker": collector.TICKER,
        "start_inclusive": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "snapshot_cutoff": cutoff.isoformat(),
        "window_minutes": window,
        "feature_names_per_minute": names,
        "flatten_order": "oldest minute first, features in listed order",
        "macd_normalization": collector.MACD_FEATURE_TRANSFORM,
        "feature_timeline_first_close": (entries.entry_at.iloc[0] - pd.Timedelta(minutes=window - 1)).isoformat(),
        "features_per_example": window * len(names),
        "entry_commission_pct_per_fill": settings.entry_commission_pct,
        "exit_commission_pct_per_fill": settings.exit_commission_pct,
        "exit_settings": vars(settings),
        "main_anchor_ids": list(collector.MAIN_ANCHOR_IDS),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved separate early timeline export at {output}", flush=True)


if __name__ == "__main__":
    main()
