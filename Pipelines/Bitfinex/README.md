# Main-anchor MLP training data

The main collector now creates a chronological sample for every decision time
with at least one of the 18 main anchors. Multiple anchors on the same closed
M5 candle share an entry ID and retain their individual flags. Each decision is
simulated independently, including decisions made during another trade series.

```powershell
python Pipelines/Bitfinex/bitfinex_indicators_v2.1.py --workers 6
```

The default entry period is the last three calendar years ending at the most
recent midnight in fixed UTC+3. `--start` and `--end` set the entry period
(start inclusive, end exclusive). `--cutoff` sets the data availability time
and can be later than the entry period to allow trades to finish. Use
`--history-cache` to specify the directory for monthly native candles and
`--output` for the dataset directory. Label batches are checkpointed.

Each X row contains 1,410 closed one-minute slots (23.5 hours), with 36
normalized market features and 18 main-anchor flags per slot. Higher
timeframes use their last closed candle as of each minute. Anchor flags are
set only on their decision minute. Values are flattened oldest first for
76,140 FP32 inputs. See `manifest.json` for the exact feature order.

MACD and its signal use the signed transform `x / sqrt(1 + x^2)` on all
four timeframes. Zero maps to zero, and large values approach −1 or +1.
New exports record the formula as `macd_normalization` in the manifest.
Exports created before this change used MACD divided by raw ATR.

The binary target is 1 when the sum of the short position returns, including
re-entries, meets or exceeds the total fee. A canceled unfilled initial order
has target 0. The commission defaults to 0.05 percentage points per filled
entry and per filled exit. Edit `ENTRY_COMMISSION_PCT` and
`EXIT_COMMISSION_PCT` near the top of the collector to change the defaults.
For a single run, override either with `--entry-commission-pct` or
`--exit-commission-pct`. Wave 1 is the negative M5 histogram half-wave at
order placement, or the next negative half-wave if the order is placed in
a positive one. The order and M1 trade simulation start on the usual clock,
including the time before wave 1 starts. A position still open at the start of
M5 half-wave 6 closes at the first actual M1 Open at or after the cutoff. The earlier ADX
market exit takes priority when a closed M5 candle in histogram wave 2 or 4
has ADX > 25 and the M5 MACD line is positive after a zero cross up confirmed
at or after initial order placement. The cross may occur on an earlier
candle. A series that needs data beyond the snapshot retains a blank label
in the audit CSV.
The wave-6 opening fill is a historical convention. The histogram transition
itself is only confirmed at that M5 candle's close, which is used when checking
whether the label was known before the test period.
The audit's legacy `target_available` field refers to completed filled series.
Use the presence of `binary_target` to determine whether an entry is labeled.

The oldest 80% of labeled decision times form the training candidates. The
newest 20% form the test set. Training candidates whose outcomes are known
at or after the first test decision are excluded from training. The audit CSV
retains their features' decision time, label, and exclusion reason.

The four output arrays are `X_train.npy`, `y_train.npy`, `X_test.npy`, and
`y_test.npy`. X contains FP32 values and y contains int64 class indices
suitable for a two-output MLP using cross-entropy. `entry_targets.csv`
maps each exported row to its entry ID and records the trades, fees, returns,
and target. The `feature_end_row` column locates each entry's last minute in
`feature_timeline.npy`, including entries outside the supervised splits.
Use a memory-mapped read and copy only the current batch:

```python
from pathlib import Path
import numpy as np
import torch

root = Path("Exports/training/mlp")
X = np.load(root / "X_train.npy", mmap_mode="r")
y = np.load(root / "y_train.npy", mmap_mode="r")
batch_x = torch.from_numpy(X[:64].copy())
batch_y = torch.from_numpy(y[:64].copy())
```

The remaining sections describe the prior snapshot-based review workbook
and its original gross-return audit. Its in-code helpers still support the
review workflow. The main CLI now produces the NumPy splits described above.

Workspace organization: see [README.md](README.md). Saved training CSVs and
settings are in `Exports/training/`. Audit packages, reports and loose evidence
are in `Audit/`.

The 20,000-candle indicator workbook is available through:

```powershell
python Pipelines/Bitfinex/bitfinex_indicators_v2.1.py --indicators-only
```

Dependencies: `numpy`, `pandas`, `requests`, `openpyxl`. The existing local
`.python-deps` folder is used if `requests` is not installed in the interpreter.

## Random-minute control

`random_minute_control.py` tests whether the main anchors add an edge for the current exit rules. It samples M5 decision times on which no main anchor fires, places the same initial short stop order, and labels each with the unchanged exit simulator, settings and fees read from the anchor export's manifest. It then compares fill rate, winner rate and net return with the anchor entries overall, by time period and by M5 ATR / close quartile.

```powershell
python Pipelines/Bitfinex/random_minute_control.py --samples 20000 --workers 8
```

It runs offline from `Exports/candle_cache` and stops if a cache file is missing rather than calling the API. Results go to `Exports/training/random_minute_control/` (`comparison.json` and `random_entry_targets.csv`). Label batches are checkpointed there, so a repeated run is fast. Without openpyxl installed, the control stubs the Excel-only import.

Passing `--stop-timeframe 1m|5m` also relabels the anchor entries with the current exit code (`--anchor-samples N` to sample them) and writes `anchor_entry_targets.csv`. Use it whenever the rules differ from the export's: the 18 Nov 2026 export predates the current ADX / wave-6 rules, so its labels do not match what the code produces today.

```powershell
python Pipelines/Bitfinex/random_minute_control.py --samples 12000 --anchor-samples 12000 --stop-timeframe 5m --output Exports/training/random_minute_control_5m
```

## Stop timeframe

`--stop-timeframe` (`ExitSettings.stop_timeframe`, default `1m`) chooses the candles that **set and move** the trailing exit stop. With `5m`, the M1 rules are applied unchanged to M5 candles built from actual M1 candles (a leading partial bucket is dropped):

- A closed red M5 candle joins the trade's red history. Up to 20 red M5 candles closed at or before the trade entry seed it, from one day of pre-entry M1 history.
- A closed green M5 candle searches left for the nearest red M5 candle whose High is at least its High. The candidate stop is that High plus `stop_x` ($0.30). It sets the first stop or lowers the current one; it never raises it.
- The stop is acted on at the M5 close and is **triggered by every M1 High** at or above it, as a resting order would be.

Entries, re-entries, support searches, ADX and wave-6 exits are unchanged. Stop events carry `candle_timeframe = 5m`. The verified v9 source is not edited; the change is applied through the same named adapter as the initial-fill timing.

## Anchor membership

The confirmed workbook is **Основные и допополнительные якоря 15.10.2026.xlsx**.
Its decimal-looking cells `10.31` and `25.26` are interpreted as two IDs each.
The code uses the current 35-anchor numbering; former anchor 23 was removed.

- Main: **1, 3, 5, 7, 8, 9, 13, 14, 15, 16, 19, 22, 23, 24, 25, 26, 28, 33**.
- Additional, movement start: **11, 30, 32**.
- Additional, movement strengthening: **12, 17, 18, 20**.
- Additional, possible trend change: **10, 31**.
- Additional, potential reversals: **2, 4, 6, 29**.
- Additional, trend end: **21**.
- Additional, volume: **27, 34, 35**.

Only the main group creates entry samples. Multiple main anchors at one time
produce one row with multiple flags equal to 1. Additional groups are retained
as separate `group_events` datasets and never create rows on their own.

## Input timing and targets

Anchors are detected on closed M5 candles. The next M5 opening is **order placement**, not an assumed fill. For a 10:00–10:05 signal candle, place an initial short stop order at 10:05 using **Low of M1 10:04–10:05 minus $0.10**. The absolute offset is provisional and configurable with `--initial-entry-offset-down` / `ExitSettings.initial_entry_offset_down`.

An unfilled initial order expires at the first subsequent CLOSED M5 MACD histogram transition from negative to nonnegative, or at the existing ADX / wave-6 series boundary if earlier. Only actual API M1 candles can execute the order. If M1 opens at or below the order level, use that Open; otherwise the first Low touch fills at the level. Intrabar touches are represented by the M1 closing timestamp, with no exit processing inside that fill candle. This timestamp is a proxy, not the known exact trade time.

Features and M5 exit-wave numbering remain anchored to the signal/order time even if execution is delayed. In-code `entries.entry_at` is placement and legacy `entry_price` is the order LEVEL; actual execution is in the separate `initial_orders` dataset. `initial_fill_*` fields are future outcomes and are excluded from `model_features`. Unfilled orders have no position, no re-entries, and no return target. Chronological counts distinguish admitted orders from executed initial entries and reserve capacity during the initial wait.

Market features describe only candles closed by placement. At 13:05 they use M1 13:04–13:05 and M5 13:00–13:05, never the future 13:05 candle's close or Low.

The main table includes raw OHLCV for M1/M5/M15/M30, normalized price, volume
and indicators, and the 18 main-anchor activation flags. Numeric model inputs
are explicitly listed in the settings and stored separately from targets.
Workbook and CSV display timestamps use fixed UTC+3; internal calculations use UTC.

The required v9 exit functions and v5 MACD wave/ADX helpers are embedded inside
the main Python file in isolated namespaces. Retained function bodies are
unchanged. MACD wave boundaries and the ADX > 25 termination rule remain active;
unrelated original collectors, command-line interfaces and reports are omitted.
Original-file and embedded-subset SHA-256 hashes are recorded. No external copy
of either supplied script is needed to run this file.

The chart audit found two inherited indicator-input differences in v5: missing
no-trade minutes were skipped by MACD, and binary floating-point ties could
alter directional movement. Exit inputs now fill empty indicator slots and use
the main script's decimal-safe M5 DMI/ADX calculation, as explicitly confirmed.
The original MACD wave helpers, ADX > 25 condition, stops, support rules and
re-entry rules remain active. Stop/support/fill processing still uses actual
API candles. Corrected inputs can change historical wave boundaries and targets;
they are compared separately with the unedited original-script baseline.

Settings confirmed for this run:

- Exit-stop absolute offset: **0.3**.
- Support-cluster relative tolerance: **0.0003**, or **0.03%**.
- Re-entry downward absolute offset: **0**.
- ADX threshold: **25**, using the original numbered M5 waves 2 and 4.

`target_gross_return_pct` is the sum of the original per-trade
`short_result_pct` values, including re-entries. It is expressed in percentage
points: `1.5` means 1.5%, before leverage, fees and slippage. It is not a
loss/multiprofit class; class boundaries have not been supplied.

In the original v9 baseline, an active position was left open at the start of
the sixth half-wave. Such rows had `target_available = 0` and an empty target.
`realized_gross_return_pct` records only the already closed trades and must
not be substituted for an unfinished series' target. Flat series with an
unfilled re-entry order at the final boundary retain their realized target.

If the preceding source M1 has no API candle, its existing flat previous-close slot supplies the reference Low and `entry_reference_is_synthetic = 1` (also retained in legacy `entry_price_is_synthetic`). This does not create a synthetic execution: fills always require a subsequent actual API M1. The initial-order adapter leaves the hashed original v9 source intact and changes only initial fill-candle timing; historical immediate-entry parity reports predate this entry-rule revision.

## In-code usage

```python
datasets = collect_training_datasets(case_limit=2000)
X = datasets.model_features
y = datasets.targets
available = y["target_available"].eq(1)
X_trainable = X.loc[available]
y_trainable = y.loc[available, "target_gross_return_pct"]
```

Each row is an entry sample endpoint. A future LSTM window builder should use
consecutive candle histories from the underlying indicator timeline, rather
than treating these irregularly spaced entry rows as consecutive candles.

The original `collect_datasets()` API remains available for the full raw,
normalized and anchor timeline. `TrainingDatasets` keeps features, model inputs,
targets, groups, trades, stop events, support searches and entry orders separate.
The spreadsheet joins features and targets for inspection. CSV counterparts
preserve those separate datasets for later code use.

## Original-run comparison

The unmodified supplied v9 CLI is run on 20 input workbooks of 100 real entries
each. These batches avoid oversized repeated-candle input workbooks. The
embedded simulator is compared against every shared field in all five core
outputs: trades, stop events, support searches, entry orders and entry summaries.

Comparison details are stored under `.indicator-audit/exit-integration/`.
The original files in Downloads/Telegram Desktop are never modified.

The initial cleanup reproduced all 2,000 original exit traces exactly. After
the approved chart-input correction, 22 of 2,000 return targets changed; 54
model input columns, 4,305 trades and 2,305 re-entries were retained. The current
`comparison` sheet identifies changed outputs explicitly instead of claiming
exact original-input parity. 1,985 targets are available and 15 unfinished.

## Completed 40-case chart audit

The 40 cases span 17 September–7 October 2026 (UTC+3), covering 87 positions,
47 re-entries and 71 downward stop moves. The audit checked 840 M1 and 391 M5
chart candles, 7,386 displayed MACD/DMI values, 240 MACD wave boundaries and
32 ADX termination decisions. All agree within chart display precision after
loading sufficient chart history. One entry minute is absent from the API;
the chart displays its previous-close flat candle, matching the explicit
synthetic-entry policy. Case 936's target changed after correcting MACD inputs.

An independent replay passed 2,984 rule assertions, including first stop/entry
crossings, stop priority, support clustering, stop resets, ADX close-at-open,
unfilled orders and the unfinished position at the research cutoff. The saved
workbook passed 287,080 trade/summary field comparisons against the new results.

- `bitfinex_exit_audit_40_cases.csv`: one audit summary per case.
- `bitfinex_exit_chart_readings_40_cases.csv`: every chart reading and script value.
- `bitfinex_exit_audit_40_cases.json`: tolerances, maximum differences and counts.
- `bitfinex_exit_chart_verified.png`: representative verified ADX decision.

## Static-audit follow-up (16 October 2026)

See `STATIC_AUDIT_RESPONSE.md` for measurements, fixes and strategy questions.
Embedded-source hashes are now verified at import and loading. Training/exit
indicator warm-up is enforced. Empty wave outputs retain their schema, ambiguous
numeric membership lists are rejected, and metadata records workbook validation.

`closed_indicator_history()` advances already-closed empty slots to the known
entry clock without future candles. This fixes an inference/historical-feature
disagreement for seven entries immediately after missing M1 minutes. The first
100 entries and all seven edge entries now match exactly after future removal.
All saved 2,000 feature rows and targets remain unchanged after these fixes.

This historical audit used the entry M5 candle's half-wave as wave 1. The
current collector starts counting at the negative half-wave as described above.
