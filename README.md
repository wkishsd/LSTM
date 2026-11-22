# LSTM workspace

## Start here

- **Train the trade MLP** with the [training panel](Training/rocm-9070/Training%20Panel.cmd) or follow the [training guide](Training/rocm-9070/README.md). The training package, input datasets, and run history are all under [Training/rocm-9070](Training/rocm-9070/).
- **Collect Bitfinex data and export training sets** with the [Bitfinex pipeline](Pipelines/Bitfinex/README.md). The [anchor-frequency script](Pipelines/Bitfinex/anchor_frequency_analysis.py) lives beside the collector.
- **Analyze 30-minute MACD cycles** with the [MACD pipeline](Pipelines/MACD/README.md).
- **Run the shared LSTM example and tests** using the [model guide](Models/SharedLSTM/README.md).

## Data and results

- [Exports](Exports/) contains the [indicator workbook](Exports/indicators/bitfinex_indicators_v2.1.xlsx), [anchor-frequency reports](Exports/anchor_frequency/), [training datasets and workbooks](Exports/training/), [MACD reports](Exports/macd_30m/), and [candle cache](Exports/candle_cache/).
- [Experiments](Experiments/) contains the Bitfinex research scripts, diagnostics, and experimental runs grouped by study.
- [Documentation](Documentation/) contains diagrams and the [Bitfinex data flow](Documentation/BITFINEX_DATA_FLOW.html).
- [Handoff](Handoff/) contains the [November MLP prototype](Handoff/2026-11-16-mlp-prototype/) and the [revised September entry-rule handoff](Handoff/2026-10-17-entry-rule/). The [original September handoff](Handoff/2026-10-16/) is available for comparison.
- [Audit](Audit/) contains dated reviews, evidence, reports, and packaged snapshots.

## Archived work

- [November scratch files and intermediate artifacts](Archive/Intermediate/2026-11-21/) include temporary handoff work, smoke tests, old display-control builds, and the previous root cache.
- [October cleanup archive](Archive/cleanup-2026-10-16/) contains older script versions and temporary files. The original move log is in [moves.json](Archive/cleanup-2026-10-16/moves.json).

Run the guide commands from this workspace root. Local support folders for Python, Node, and older audit runners are hidden in Explorer.

