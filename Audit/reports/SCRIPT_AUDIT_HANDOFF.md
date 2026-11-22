# Bitfinex v2.1 — implementation and independent-audit handoff

Snapshot date: 16 October 2026. This document describes the current implementation
and records the checks performed by the implementing agent. It is **not an
independent audit verdict**. Verify the code and underlying evidence rather than
treating the reported PASS results as proof of correctness.

Follow-up: `STATIC_AUDIT_RESPONSE.md` records the subsequent static-review
measurements and fixes. The current runtime verifies embedded-source hashes,
enforces warm-up, rejects ambiguous numeric anchor lists, preserves empty-wave
schemas and advances trailing empty feature slots to the known decision clock.
These changes preserve all saved 2,000-entry features and targets. Historical
test descriptions below identify their original validation stage.

## 1. Which file to audit

Workspace: `C:/Users/wkishsd/Downloads/otcheti/LSTM`.

**Active implementation: `bitfinex_indicators_v2.1.py`.**

`bitfinex_indicators (1).py` is an older, separate script. It still contains the
1,000-row lookback and large M1 indicator periods. The current work was written
to v2.1, not that older file. They are not interchangeable entry points.

Primary output: `bitfinex_training_dataset_v2.1.xlsx`.

Original supplied files, retained outside the workspace and not edited:

- `C:/Users/wkishsd/Downloads/Telegram Desktop/macd_stop_trades_from_dataset_v9.py`
- `C:/Users/wkishsd/Downloads/Telegram Desktop/macd_halfwaves_adx_minutes_v5.py`
- Anchor membership source:
  `C:/Users/wkishsd/Downloads/Основные и допополнительные якоря 15.10.2026.xlsx`

The main Python file is self-contained with respect to the two supplied scripts:
it does not require them at runtime. It still needs its Python dependencies and
either public candle API access or the saved candle cache.

## 2. Intended result and confirmed decisions

The implementation creates short-only, hypothetical entry samples triggered by
the main anchor group, with already-available market features and future gross
return targets produced by the supplied exit strategy.

Confirmed user decisions implemented:

- Detect anchors on **closed 5m candles**; assign entry to the next 5m opening.
- Former anchor 23 was removed; former 24–36 became current 23–35.
- Use all 35 current rules, split into one main group and six additional groups.
  The user explicitly accepted seven groups after the original six-group request.
- Interpret membership workbook cells `10.31` and `25.26` as two IDs each.
- Only main anchors generate training rows. Additional groups remain available
  in code but do not independently generate training rows or model flag columns.
- Retain the required MACD wave helper and ADX > 25 termination helper.
- Absolute exit-stop offset X = **0.3**; support tolerance = **0.0003**;
  downward re-entry offset = **0**.
- Targets are original-style gross returns, without fees or leverage. Unfinished
  position series have unavailable targets. No class thresholds were supplied.
- Fill missing indicator time slots and use decimal-safe DMI for exit inputs.
  The user explicitly approved this correction after the chart audit found
  differences from the original helper outputs.
- Display timestamps in fixed **UTC+3**, not a daylight-saving regional zone.

The strategy is not a live trading bot, a fitted LSTM, or a chronological
single-position portfolio backtest. No orders were sent to the exchange.

## 3. Main changes made

### Existing market-data / indicator work

- Four native timeframes: M1, M5, M15 and M30. The current production path
  requests the exchange's native OHLCV for each timeframe; it does not construct
  higher candles by multiplying M1 indicator periods or resampling M1 locally.
- Shared indicator settings per timeframe: RSI 14, ATR 14, MACD 12/26 with signal
  9, DMI length 14 with ADX smoothing 4.
- Separate raw, normalized and anchor datasets for Python use.
- Closed-candle alignment, source-candle timestamps, UTC+3 exports, numeric
  feature formatting, explicit zero-log encoding and anchor overlap metrics.
- Public candle requests have a 30-second timeout, 2.1-second minimum spacing,
  four attempts and retry/backoff handling for timeouts, connection errors,
  HTTP 429 and server errors. These are configured controls, not a guarantee
  against rate limiting across multiple processes or changing API policies.

### Entry/target integration

- Added `ExitSettings`, `TrainingDatasets`, main-anchor filtering, features at
  entry, original-input construction, exit target calculation and workbook/CSV
  export.
- Initially embedded both supplied scripts in full. Subsequently trimmed them
  to the required functions, classes, constants and dependencies, removing their
  unrelated collectors, standalone CLIs and report/statistics routines.
- Retained original function bodies inside `ORIGINAL_EXIT_SOURCE` and
  `ORIGINAL_WAVE_SOURCE`, executed in separate module namespaces. These are
  Python source strings: ordinary top-level AST inspection alone will miss the
  embedded functions. Inspect/parse the string values as well.
- M5 production exit indicators now come from shared `calculate_native()`.
  The original v5 `calculate_adx()` definition remains in the embedded subset
  for provenance, but it is not the production M5 ADX calculation path.
- M1 exit MACD wave markers still use the original v5 MACD/wave functions, now
  supplied with a gap-filled indicator history.
- Added a pandas timestamp-unit correction: convert explicitly to nanoseconds
  before converting datetime integers to milliseconds. Added a local
  `.python-deps` fallback when importing `requests`.

Relevant current code locations (search by name; line numbers can change):

- `request_candles`, `fetch_candles`: API collection and closed-candle cutoff.
- `fill_candle_gaps`, `directional_movement`, `calculate_native`: market inputs.
- `normalize_log`, `normalize`: numeric transformations and invalid-value rules.
- `detect_anchor_events`, `filter_anchors`: full timeline anchor events.
- `filter_main_anchor_entries`, `features_at_entry`: training row selection/timing.
- `LegacyExitInputBuilder`: critical adapter between market data and original exits.
- Embedded `process_entry`: position / stop / re-entry state machine.
- `calculate_exit_targets`: aggregation and availability of targets.
- `build_training_dataset`, `export_training_dataset`, `main`: public run path.

## 4. Data and timing contracts

### Two run modes

- Default training run: `--cases 2000`; `--cases 0` means all matching entries.
  It selects the **first chronological** matching entries after the warm-up
  within the fetched snapshot / requested entry range, not the latest 2,000.
  Default fetched M1 history is 45,000 actual API candle records.
- `--indicators-only`: separate all-candle indicator/anchor export with
  `LOOKBACK_CANDLES = 20000`. This is not the default training-row count.

Training entries start at least three days after the earliest fetched M1
timestamp. The adapter uses M5 history from three days before the first selected
entry; each M1 wave calculation uses three days before its entry. The market
indicator path also has `warmup_candles()` (currently 260 per timeframe).

### Candle identity versus availability

A candle timestamp identifies its **opening**, while `available_at` identifies
when its completed OHLCV and indicators become usable.

Example: a main anchor on the 13:00–13:05 M5 candle creates an entry at 13:05.
That entry's features use the completed 13:04–13:05 M1 candle, completed
13:00–13:05 M5 candle, and last completed M15/M30 candles. The eventual
13:05 M1 high/low/close/volume must not be inputs to that entry.

`features_at_entry()` chooses the last M1 candle with availability <= entry time,
then aligns all timeframes by availability. `source_candle_*` columns identify
the source candle opening, not the entry time or its closing time.

The older full `anchor_data` timeline has different semantics: it marks a
candidate at an M1 opening but carries that same M1 candle's eventual closed
features. Its own docstring warns against using that row as an opening-time
training snapshot. Audit the default training pipeline separately.

### Missing candles

- Indicator history fills internal missing slots with previous-close OHLC and
  zero volume, advancing EMA/RMA by one timeframe period. The general gap filler
  does not extend fetched endpoints. For training/inference features,
  `closed_indicator_history()` additionally advances trailing empty slots only
  through an explicitly known decision time, excluding forming/future candles.
- Exit stops, support clusters and simulated fills still process actual API M1
  candles. Gap-filled bars are used for indicator/wave history, not supplied as
  executed-price observations to the original state machine.
- If an anchor entry opening has no API M1 candle, entry reference price is the
  preceding observed close, explicitly marked `entry_price_is_synthetic = 1`.
  This is an assumption about an executable entry price, not evidence of a trade.
- The original 2,000-case preparation recorded eight synthetic entry references.
  The 40-case sample includes case 936, whose chart shows the flat empty candle.

Audit the intentional distinction between a clock-based indicator history and
an actual-record execution history, including whether empty-minute order
placement timing should have different strategy semantics.

## 5. Indicators, normalization and actual model inputs

RSI and ATR use Wilder-style smoothing; RMA is seeded from an initial mean.
Main market MACD uses pandas `ewm(adjust=False, min_periods=period)`; the retained
v5 M1 MACD helper uses an SMA-seeded EMA. These initialization methods are not
identical at the start of history. Long warm-up made the checked readings agree;
do not assume exact equivalence at every possible warm-up length.

DMI compares price differences using `Decimal(str(price))`, so mathematically
equal upward/downward changes do not become unequal from binary subtraction.
Strict ties contribute neither +DM nor -DM. The ADX > 25 decision itself is
unchanged.

For each timeframe the model has nine numeric market features:

1. `ln(close)`.
2. `ln(candle volume)`.
3. `(RSI14 - 50) / 50`.
4. `ATR14 / that timeframe's close`.
5. `MACD12/26 / that timeframe's raw ATR14`.
6. `Signal9 / that timeframe's raw ATR14`.
7. `ADX14/4 / 100`.
8. `DI+14 / 100`.
9. `DI-14 / 100`.

Zero is explicitly encoded as **0** for every logarithm-normalized field;
`ln(0)` is never evaluated. Negative, missing and infinite log inputs are
rejected. This encoding makes zero input and positive input 1 share the same
normalized value. It was a requested convention, not a mathematical identity.

MACD/ATR and Signal/ATR encode 0/0 as zero; nonzero/0 remains undefined and
causes the finite-feature validation to fail. Source RSI/DI/DX zero-denominator
branches should be checked in the actual functions as well.

**36 market features + 18 main-anchor flags = 54 model inputs.**
The 18 are anchor activation flags, not the indicator count. Raw OHLCV for all
four timeframes is present in the inspection/features table, but raw OHLCV is
not automatically included in `model_features`. Only `norm_*` and `anchor_*`
columns are selected there. Targets and future outcome timestamps are excluded.

No LSTM sequence/window constructor, training procedure, train/test split,
feature scaler fitted on a training partition, loss/multiprofit classification
or production flat-position gate has been implemented here.

## 6. Anchor events and memberships

Rules evaluate unnormalized completed M5 indicators. Downward crossing means
previous >= previous boundary and current < current boundary; upward crossing
uses previous <= and current >. Rolling reference windows exclude the current
candle. Reversals require strict direction changes. "First" means a false-to-true
transition. The 35 current rule descriptions are reproduced in the appendix;
the predicate implementation remains the source to inspect for exact semantics.

Main IDs: **1, 3, 5, 7, 8, 9, 13, 14, 15, 16, 19, 22, 23, 24, 25, 26, 28, 33**.

Six additional categories:

- Movement start: 11, 30, 32.
- Movement strength: 12, 17, 18, 20.
- Possible trend change: 10, 31.
- Potential reversals: 2, 4, 6, 29.
- Trend end: 21.
- Volume: 27, 34, 35.

Simultaneous main anchors produce one row with multiple flags set to 1 and
semicolon-separated `main_anchor_numbers`. All 35 flags exist in the event/entry
data, but only the 18 main flags are model features. `group_events` retains seven
event datasets; category groups are not modeled or trained separately here.

Entries are hypothetical and evaluated independently. Anchors inside another
sample's hypothetical position are not removed. A chronological one-position
execution policy would be a separate evaluation stage.

## 7. Exact exit strategy retained

### Initial entry and red-candle context

- Short entry at the main-anchor entry timestamp and reference price.
- Each new position, including a re-entry, seeds its red history with up to the
  last 20 red candles before or including its entry/trigger candle.
- Red means open > close, green means open < close; doji creates no stop candidate.
- Later red candles are appended. The ongoing red history is not capped at 20.

### Exit stops

- After a green candle closes, search backward for the nearest retained red
  candle with red high >= green high.
- Candidate exit stop = red high + 0.3.
- Set the first stop when a candidate exists; subsequently lower it only when
  the candidate is strictly below the existing stop. Never raise it.
- Check an existing stop against every actual candle's high **before** processing
  that candle for a stop update. A new stop applies to later candles.
- High >= stop fills at the exact stop level; the event is timestamped at that
  M1 candle's close. There may be no initial stop until a qualifying green candle.

### Re-entry after an ordinary stop

- The series may continue after a stop. Search only while in a positive M1 MACD
  histogram half-wave, using its immediately preceding negative half-wave.
- A negative wave becomes usable after the first positive candle closes. An
  eligible negative wave is checked once, rather than repeatedly searching older
  waves. Pending orders persist until filled or terminated.
- Sort actual negative-wave candle lows. Relative cluster tolerance is
  `abs(low - cluster_base) / abs(cluster_base) <= 0.0003`.
- Type 1: at least three lows in the lowest cluster.
- Type 2: a single lowest low plus an upper cluster of at least three lows.
- Type 3: a lowest pair plus an upper cluster of at least three lows.
- When valid, support is the wave's global minimum low, not the upper cluster's
  base. Re-entry level = support - 0; it must be below the current close.
- A pending short order triggers on the first subsequent actual M1 low <= level,
  fills at that exact level and records the entry at the trigger candle's close.
- Reset red history for the new trade. Do not set or trigger an exit stop inside
  that same re-entry trigger candle.

### MACD wave horizon and ADX termination

- M5 half-waves are defined by the **histogram** sign, not the MACD line sign.
  Positive/negative is strict; histogram zero carries the previous wave sign.
- Wave 1 is the M5 half-wave containing the entry; the research interval ends at
  the **start of numbered wave 6**, exclusive. This is the precise code behavior
  behind the informal "three-wave" terminology.
- An active position at that boundary is recorded as open/unclosed, not silently
  force-closed. A pending order is recorded unfilled. If a sixth wave does not
  exist in the available snapshot, that entry has no valid wave-6 target horizon.
- ADX > 25 is considered in numbered M5 waves **2 and 4**, exactly as the supplied
  collector specifies. They are not explicitly filtered for positive histogram
  sign. A short entry does not itself guarantee wave 1 is negative, so waves 2/4
  are not guaranteed to be positive. No extra entry sign filter was added.
- ADX is actionable only at the qualifying M5 candle's close. At/after that
  detection time, the first available M1 opening terminates the series: close an
  active position at that M1 open and cancel a pending re-entry order.
- This ADX check takes priority over ordinary stop/fill processing.

### Target aggregation

- Per closed position: `(entry_price - exit_price) / entry_price * 100`, retaining
  the original engine's rounded `short_result_pct` output.
- Target: sum those original rounded per-position percentages across the series,
  including re-entries, rounded to six decimals. This is not compounded equity.
- A valid ended series with all positions closed can have an available target
  even when a final re-entry order was left unfilled.
- An active unfinished position makes the target unavailable. Partial
  `realized_gross_return_pct` must not be substituted for its complete target.

## 8. Why current results differ from the original scripts

There are two different comparison stages. Do not conflate them:

1. Original-input integration: unedited v9 CLI, 20 batches of 100 entries,
   compared with the embedded engine on the same constructed inputs. All 2,000
   matched. Trimming the embedded source also preserved all original output rows.
2. Approved production correction: fill missing indicator slots for M1 MACD and
   use shared decimal-safe M5 DMI/ADX. Exit state-machine rules were retained,
   but original-input numerical parity is no longer the intended result.

For the saved 2,000-entry snapshot:

- Entry range: 16 September 2026 23:25 to 8 October 2026 04:40, fixed UTC+3.
- 421 entries have changed trace fields; **22 have changed return targets**.
  Trace differences include indicator/metadata values, not necessarily a changed
  execution. Current comparison uses DataFrame equality; inspect field differences.
- 4,305 positions and 2,305 re-entries; 1,985 available targets, 15 unavailable.
- Available statuses: 1,660 complete and 325 complete with a final unfilled order.
- The observed M5 ADX detection times and wave-6 horizon times did not change.
- All 54 model-feature columns remained exactly unchanged in the correction run.

`bitfinex_training_comparison_2000.csv` and the workbook's `comparison` sheet
identify these changes. Old `comparison_summary.json` records stage 1, not a
claim that the current corrected inputs reproduce the original outputs exactly.

## 9. What was checked, and what was not established

### Original-code parity evidence

Saved stage-1 report: 677,854 numeric and 317,786 string comparisons across five
outputs, tolerance 1e-9, maximum numeric difference about 4.55e-13. These checks
establish implementation parity on shared inputs, not that the original strategy
or adapter is causally correct. The adapter was written by the same agent.

### Forty-case chart and replay audit

- Case dates: 17 September–7 October 2026, UTC+3.
- 87 positions, 47 re-entries, 71 downward stop moves.
- 840 M1 and 391 M5 chart candles; 7,386 MACD/histogram/signal/DI/ADX readings.
- 240 M5 wave-boundary checks and 32 ADX termination decisions.
- 1,230 native OHLC matches and one empty chart candle matching previous-close
  synthetic OHLC.
- Display tolerances: 0.00000051 for six-decimal MACD fields; 0.000051 for
  four-decimal DMI/ADX. These are display-precision comparisons, not bitwise
  comparisons with the exchange's hidden full-precision indicator state.
- 2,984 independent rule assertions; 287,080 exported trade/summary field checks.

The chart work used browser UI navigation and visible legend text, automated by
the agent. It was not a separate human manually placing every hypothetical
order. The chart sometimes selected neighbouring candles after date jumps;
readings were corrected using adjacent OHLC fingerprints and timestamp views.
Earlier history was loaded to resolve chart initialization differences.

The sample is **not random**: 32 cases were chosen across the timeline from
two-position, stop-moving, relatively short ADX-ended series, plus seven selected
wave-cutoff cases and one unfinished case. Only case 936 among the 22 changed
targets is in the 40-case sample. Audit the other 21 changes independently.

The "independent replay" is a separately coded check, but it was also written by
the implementing agent. It uses the same saved candles and the existing trade,
order and summary records to identify paths to check. It is not a completely
blind strategy implementation generated solely from raw candles and anchors;
it does not by itself rule out omitted searches or entries.

These results do not establish predictive quality, profitability, a valid
train/test split, complete absence of leakage, or realistic execution costs.

## 10. Priority questions for an independent reviewer

1. Reconstruct events and targets independently from raw candle JSON, without
   trusting existing entry-order/trade outputs as an oracle.
2. Trace feature availability at every timeframe boundary, particularly M5/M15/
   M30 closes, sparse periods and initial warm-up. Never use inspection-table
   OHLCV or target metadata indiscriminately as model inputs.
3. Verify all 35 predicates, renumbering and the anchor-workbook membership.
   Compare the current document definitions separately if needed; this handoff
   does not claim a fresh Google Doc synchronization.
4. Assess original numbered waves 2/4 versus the description "positive waves,"
   including positive-histogram short entries and the wave-6 research horizon.
5. Examine indicator EMA initialization differences and sensitivity to the
   three-day/260-candle warm-up. Check that future truncation leaves inputs fixed.
6. Review synthetic entry prices and indicator-only gap filling versus
   actual-candle execution, support clustering and order-placement clocks.
7. Check all 22 changed targets and explain each changed support/order/fill;
   stage-1 parity is not sufficient evidence for the approved corrections.
8. Check first versus subsequent stop crossings, red-history selection, doji,
   tied highs/lows, support-cluster grouping, pending-order persistence, and
   priority when ADX, stop and entry conditions coincide.
9. Assess exact-level fills when prices gap through a level, latency, liquidity,
   no initial stop, and the skipped exit evaluation inside a re-entry trigger
   candle. These are inherited simulation assumptions; no intraminute path,
   bid/ask execution, fees, funding or slippage is modeled.
10. Review overlapping labels and chronological split leakage. Each anchor is a
    separate hypothetical opportunity, including ones inside another sample's
    position. A portfolio flat gate is not applied during target construction.
11. Do not treat irregular event rows as consecutive one-minute LSTM timesteps.
    Build sequence windows and chronological validation explicitly in future work.
12. Validate that raw features, normalized features and targets remain separate
    in Python, not merely in workbook tabs, and that unavailable targets stay
    excluded from training without silently becoming zero returns.

## 11. Reproduction and evidence files

From the workspace, using Python with numpy, pandas, requests and openpyxl:

```powershell
python bitfinex_indicators_v2.1.py --cases 2000 --cache-dir .indicator-audit/exit-integration/candles --output audit_reproduction.xlsx
```

Use a different output filename to preserve the audited workbook. A fresh API
run without the cache uses a different snapshot and need not reproduce these
case IDs, counts or targets. Default CLI export does not automatically attach
the historical comparison/audit reports; those were added by audit utilities.

Current production audit utilities (read before running):

```powershell
python .indicator-audit/exit-integration/independent_exit_audit.py --chart-aligned
python .indicator-audit/exit-integration/compare_chart_readings.py
python .indicator-audit/exit-integration/final_chart_audit.py
```

The last two commands recheck **saved browser readings**; they do not open the
chart and recollect readings. `final_chart_audit.py` also writes audit reports.

Important evidence:

- Main XLSX tabs: `training_data`, `model_features`, `targets`, `anchor_groups`,
  `trades`, `entry_summary`, `comparison`, `parameters`.
- `bitfinex_training_dataset_v2.1_*`: separate feature, model-feature, target,
  training, stop-event, support-search and entry-order CSVs; settings JSON.
- `bitfinex_training_comparison_2000.csv`, `bitfinex_training_validation.json`.
- `bitfinex_exit_audit_40_cases.csv`, `bitfinex_exit_audit_40_cases.json`.
- `bitfinex_exit_chart_readings_40_cases.csv`: chart and script values side by side,
  including explicit UTC and UTC+3 timestamps.
- `.indicator-audit/exit-integration/candles/{1m,5m,15m,30m}.json`: frozen API
  arrays in `[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]` order. Do not accidentally read
  the API's second price field as HIGH.
- `.indicator-audit/exit-integration/reference/`: original-input case tables,
  input/output workbooks and logs for unedited v9 CLI baseline batches.
- `.indicator-audit/exit-integration/integrated_*.pkl`: original-input embedded
  results. `chart_aligned_*.pkl` and `chart_aligned_reference/`: corrected results.
- `.indicator-audit/exit-integration/chart_observations.json`: raw visible legend
  text, requested points and some neighbouring-candle selection attempts.
  `chart_check_points.json` defines the final 1,231 points. The raw observations
  include two additional superseded-plan points excluded from final comparisons.
- `.indicator-audit/exit-integration/selection-*.png` and
  `bitfinex_exit_chart_verified.png`: saved chart views. Not every point has an
  individual screenshot; visible legend text is the bulk of the saved evidence.
- `trimmed_helpers_validation.json`: historical pre-correction parity report.
  The current `check_trimmed_helpers.py` checks current helper input schemas
  separately; it no longer asserts corrected inputs equal original inputs.

The main file is generated by
`.indicator-audit/exit-integration/build_embedded_main.py`, using
`bitfinex_indicators_v2.1.before-targets.py`, `training_extension.py.txt` and the
two original supplied files. This developer rebuild path needs those files;
normal execution of the resulting main file does not. Avoid overwriting a code
fix by rebuilding from stale extension/backup sources.

For a portable audit, provide this handoff, the active script, both originals,
the membership workbook, the output XLSX/CSVs, validation JSON and four cached
candle JSONs. Saved local pickle files are optional convenience artifacts, not
a required portable input format.

## Appendix A. Snapshot fingerprints

The following fingerprint/manifest sections were appended from local files when
this handoff was created. Hashes identify artifacts; they do not validate them.


```json
[
  {
    "file": "bitfinex_indicators_v2.1.py",
    "exists": true,
    "bytes": 170604,
    "sha256": "fb24e14f09b509bf50d367a679b44a93111131971fdd183b4129726979a34add"
  },
  {
    "file": "bitfinex_indicators (1).py",
    "exists": true,
    "bytes": 14304,
    "sha256": "cf85c8ade6736078a10d7d16c9d9afcf604521a917d6bd12fb2ccd82fe66014c"
  },
  {
    "file": "bitfinex_training_dataset_v2.1.xlsx",
    "exists": true,
    "bytes": 3957751,
    "sha256": "abee49d01b1c3cf8272b3065a48304230b169f4925fefc148714cea73b271388"
  },
  {
    "file": "bitfinex_training_dataset_v2.1_settings.json",
    "exists": true,
    "bytes": 6561,
    "sha256": "ddc266bef66f85110a414412461bd9acfdde99a4f5f1a4c2caa6f1d48f737edf"
  },
  {
    "file": "bitfinex_training_validation.json",
    "exists": true,
    "bytes": 4052,
    "sha256": "b41fb04e743b07a386ffb0ed1529a51adda71a23051ac29e2d512ccf29c8a119"
  },
  {
    "file": "bitfinex_exit_audit_40_cases.json",
    "exists": true,
    "bytes": 1121,
    "sha256": "c86ca46b8efb945cbd7bbcd60c76e7a6349c1875f7be6e16d06e9cd19cd1ffe2"
  },
  {
    "file": "C:/Users/wkishsd/Downloads/Telegram Desktop/macd_stop_trades_from_dataset_v9.py",
    "exists": true,
    "bytes": 125368,
    "sha256": "e9d28fd0199668b4e61bf64d8fca112f30a0b30afc65cc1f3654c47e7cbb3eb1"
  },
  {
    "file": "C:/Users/wkishsd/Downloads/Telegram Desktop/macd_halfwaves_adx_minutes_v5.py",
    "exists": true,
    "bytes": 51887,
    "sha256": "c8ae9bb6ffec2aae49750c6d51059c42051be9fb0d822c32c9fbbf5c67765b1b"
  },
  {
    "file": "C:/Users/wkishsd/Downloads/Основные и допополнительные якоря 15.10.2026.xlsx",
    "exists": true,
    "bytes": 11416,
    "sha256": "97f18afe52af1093c07b7a23021e58cd5a24163a61a897101de8ea0737eeb6ac"
  },
  {
    "file": ".indicator-audit/exit-integration/candles/1m.json",
    "exists": true,
    "bytes": 2455973,
    "sha256": "913e22a5d9bd19b15691ba085683f5926bcbbb43a64cdfd041271a701d1276fe"
  },
  {
    "file": ".indicator-audit/exit-integration/candles/5m.json",
    "exists": true,
    "bytes": 616279,
    "sha256": "51fb68f48e7a4ca15b36d7e5b8037d4d0c335a06919e2239deefd3de7ad525ea"
  },
  {
    "file": ".indicator-audit/exit-integration/candles/15m.json",
    "exists": true,
    "bytes": 616336,
    "sha256": "113c619a06548544c01d30abe884038bc2b48f06e54bde4783b6b87299a22be9"
  },
  {
    "file": ".indicator-audit/exit-integration/candles/30m.json",
    "exists": true,
    "bytes": 618260,
    "sha256": "7de8e5271707ace28b236012efd771f94e6e0321de534e8e1ac746d0844221d9"
  }
]
```

## Appendix B. Exact model column order

```text
norm_price
norm_volume_1m
norm_rsi_14_1m
norm_atr_14_1m
norm_macd_12_26_1m
norm_macd_signal_12_26_9_1m
norm_adx_14_4_1m
norm_di_plus_14_1m
norm_di_minus_14_1m
norm_price_5m
norm_volume_5m
norm_rsi_14_5m
norm_atr_14_5m
norm_macd_12_26_5m
norm_macd_signal_12_26_9_5m
norm_adx_14_4_5m
norm_di_plus_14_5m
norm_di_minus_14_5m
norm_price_15m
norm_volume_15m
norm_rsi_14_15m
norm_atr_14_15m
norm_macd_12_26_15m
norm_macd_signal_12_26_9_15m
norm_adx_14_4_15m
norm_di_plus_14_15m
norm_di_minus_14_15m
norm_price_30m
norm_volume_30m
norm_rsi_14_30m
norm_atr_14_30m
norm_macd_12_26_30m
norm_macd_signal_12_26_9_30m
norm_adx_14_4_30m
norm_di_plus_14_30m
norm_di_minus_14_30m
anchor_01
anchor_03
anchor_05
anchor_07
anchor_08
anchor_09
anchor_13
anchor_14
anchor_15
anchor_16
anchor_19
anchor_22
anchor_23
anchor_24
anchor_25
anchor_26
anchor_28
anchor_33
```

## Appendix C. Current anchor descriptions from the code

These are source descriptions, not an independent interpretation of the predicates.

1. RSI пересекает 50 вниз.
2. RSI пересекает 30 вниз.
3. RSI пересекает 70 вниз.
4. RSI пересекает 60 вниз.
5. RSI меняет рост на снижение; предыдущее значение >70.
6. RSI пересекает вниз среднее своих предыдущих 5 значений.
7. MACD пересекает ноль вниз.
8. MACD пересекает Signal вниз.
9. Hist меняет рост на снижение, оставаясь >0.
10. Hist меняет рост на снижение, оставаясь <0.
11. MACD меняет рост на снижение, оставаясь >0.
12. Signal пересекает ноль вниз.
13. DI− пересекает DI+ вверх.
14. ADX пересекает 25 вверх; DI− > DI+.
15. ADX меняет снижение на рост; предыдущее значение ≤20, DI− > DI+.
16. Разница DI− − DI+ пересекает 10 пунктов вверх.
17. DI− пересекает 25 вверх; DI− > DI+.
18. DI+ пересекает 20 вниз; DI− > DI+.
19. ADX меняет рост на снижение; предыдущее значение >35, DI+ > DI−.
20. Впервые DI− растет, а DI+ уменьшается 3 последовательных периода.
21. ADX пересекает 25 вниз; DI+ > DI−.
22. MACD пересекает Signal вниз при ADX >25.
23. Впервые DI− > DI+ на протяжении 3 последовательных периодов.
24. Hist опускается ниже минимума своих предыдущих 10 значений; MACD <0.
25. ATR пересекает вверх 75-й процентиль своих предыдущих 80 значений; DI− > DI+.
26. Объем впервые >2 × средний объем предыдущих 20 периодов; DI− > DI+.
27. ATR пересекает вверх среднее своих предыдущих 20 значений; DI− > DI+.
28. RSI опускается ниже минимума своих предыдущих 10 значений, оставаясь <50.
29. RSI пересекает 70 вверх.
30. Signal меняет рост на снижение, оставаясь >0.
31. Впервые Hist уменьшается 3 последовательных периода; все 3 значения >0.
32. DI+ меняет рост на снижение, оставаясь выше DI−.
33. Прирост DI− за период впервые >5 пунктов; DI− > DI+.
34. Объем впервые >3 × средний объем предыдущих 20 периодов; RSI >70.
35. ATR уменьшается после 3 последовательных увеличений; DI+ > DI−.

## Appendix D. Case IDs for targeted review

Forty chart-audited entry IDs:

```text
61, 97, 182, 203, 374, 403, 466, 510, 566, 619, 744, 792, 870, 882, 936, 938, 980, 1039, 1067, 1072, 1085, 1144, 1146, 1248, 1305, 1333, 1345, 1394, 1456, 1459, 1487, 1678, 1679, 1765, 1835, 1884, 1898, 1953, 1956, 1968
```

Twenty-two entries with changed return targets after the approved input correction:

```text
919, 920, 921, 922, 923, 924, 925, 926, 928, 929, 930, 931, 932, 933, 934, 935, 936, 1618, 1662, 1694, 1695, 1696
```
