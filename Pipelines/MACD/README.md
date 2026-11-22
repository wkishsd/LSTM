# Native 30-minute MACD cycle analysis

## Files

- `macd_30m_cycle_analysis.py`: native candle collector, MACD/signal/histogram
  calculator and interval analysis. This is separate from the existing scripts.
- `macd_30m_report.mjs`: optional Excel report writer using the bundled Artifact
  Tool runtime. It receives the already calculated analysis, not exchange data.

The collector calls only Bitfinex `trade:30m:tETHUSD/hist`. It never fetches
one-minute candles and never packs smaller candles into 30-minute candles.
Defaults are ETH/USD spot and MACD 12/26/9, matching the main script.

## Run

```powershell
python Pipelines/MACD/macd_30m_cycle_analysis.py --candles 100000 --output-dir Exports/macd_30m/latest
```

Add `--xlsx` to also produce the Excel workbook. The companion exporter uses
`@oai/artifact-tool` through the bundled runtime junction in
`node_modules`. It requires Node;
the Python collector produces CSV/JSON/NumPy files independently. The workbook and previews go into the selected output directory.

```powershell
python Pipelines/MACD/macd_30m_cycle_analysis.py --candles 100000 --output-dir Exports/macd_30m/latest --xlsx
```

Use `--node` to specify a Node executable if it is not found automatically.
Python dependencies are `numpy`, `pandas` and `requests`. No LSTM package is
required. The main indicator and trading scripts are not imported or modified.

The saved 7 November 2026 data can be replayed without API calls:

```powershell
python Pipelines/MACD/macd_30m_cycle_analysis.py --candles 100000 --cache Exports/macd_30m/2026-11-07_100000/native_30m_snapshot.json --output-dir Exports/macd_30m/2026-11-07_100000 --xlsx
```

## Counting rule

`hist = MACD - signal`. Green means positive histogram and red means negative.
Exact zeros retain the previous nonzero sign, matching the existing half-wave
helper. Colour intensity and whether a bar is growing do not define a new wave.

Each event is the first negative histogram candle after a positive half-wave.
A completed cycle runs from one such event to the next and contains one red
half-wave and one green half-wave. Its length is the difference between their
chronological 30-minute candle numbers.

For example, red starts at candles 10 and 30 give:

- Cycle distance: 20 half-hour periods, or 10 hours.
- Candles strictly between those starts: 19.
- A window containing both start candles: 21 candles.

Both event timestamps are candle OPEN times; the colour is confirmed 30 minutes
later. The cycle table also provides confirmation times. Display times are fixed
UTC+3, with no daylight saving adjustment. CSV files retain epoch milliseconds.

## History and missing candles

The analysis retains exactly 100,000 actual API candles plus at least 260 older
native candles for EMA warm-up. Open candles are excluded. To preserve the main
script's chart-aligned clock, missing internal 30-minute slots are filled with
the preceding close as OHLC and zero volume. These are explicit assumptions,
not recovered trades, and every such row has `is_synthetic = 1`.

The saved run has 100,000 native candles and 82 flagged internal slots, making
100,082 sequential histogram rows. The 260 warm-up rows are excluded from the
analysis and report. No gaps are filled using future prices.

EMA initialization and recurrence match the main script:
`ewm(span=period, adjust=False, min_periods=period)`, with the signal EMA applied
to finite MACD values. Values are not rounded before crossing detection.

Only intervals whose two red starts occur inside the analysis range enter the
percentile array. Leading and unfinished trailing intervals are reported but
excluded from that distribution.

## Percentile

For N completed cycles, sort the integer interval lengths and select the value
at one-based rank `ceil(0.90 * N)`. This is the smallest whole-period length
covering at least 90% of observed completed cycles. Ties can make coverage
slightly greater than 90%. The report also provides the linearly interpolated
90th percentile and the achieved coverage with one fewer period.

Saved result: 3,866 completed cycles; P90 distance 47 periods, or 23.5 hours.
This covers 3,489 cycles, 90.2483%. A 46-period distance covers only 89.4723%.
A window containing both red-start candles needs 48 candles.

## Monthly cycle lengths

Each completed cycle is counted once, in the fixed UTC+3 calendar month when
the ending red-start candle closes. This also applies to cycles that cross a
month boundary. The monthly mean is total cycle periods divided by completed
cycles. Months with incomplete candle coverage at the start or end of the run
are marked partial. A month with no completions has no mean or percentile.

The report includes each month's nearest-rank P90 and the share of that month's
cycles covered by the overall P90. These show how the typical length and the
longer cycles vary over time. Comparing monthly means does not establish a
normal distribution or a stable distribution over time.

## Outputs

The workbook is saved to
`Exports/macd_30m/legacy-reports/macd_30m_100000.xlsx`.
It contains a compact summary, monthly cycle metrics, the complete cycle table
and every sequential MACD/signal/histogram row. Filtering and frozen headers are
enabled on detail.

The analysis directory also contains:

- `candles_30m.csv`: OHLCV, MACD, signal, histogram, crossings and gap flags.
- `red_start_cycles.csv`: completed cycle lengths, red/green components and times.
- `monthly_cycle_lengths.csv`: monthly counts, mean duration, P90 and coverage
  at the overall P90.
- `length_distribution.csv`: frequency and cumulative coverage by length.
- `summary.json` / `summary.csv`: results, settings, range and provenance.
- `native_30m_snapshot.json`: the original API candles and fixed cutoff.
- `analysis_arrays.npz`: separate `hist_values`, `red_start_indices`,
  `interval_lengths` and `timestamp_ms` arrays. Indices are zero-based; the
  report's candle numbers are one-based.

Example in Python:

```python
import numpy as np
data = np.load('Exports/macd_30m/2026-11-07_100000/analysis_arrays.npz')
hist_values = data['hist_values']
interval_lengths = data['interval_lengths']
```

Source and documented rate limit:
https://docs.bitfinex.com/reference/rest-public-candles
The collector uses a 30-second timeout, at least 2.1 seconds between request
starts, pages of at most 10,000 rows and bounded retry/backoff handling.
