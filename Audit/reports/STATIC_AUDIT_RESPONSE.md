# Response to the static audit — 16 October 2026

This follow-up used the local original scripts, frozen candle JSONs and saved
2,000-entry output. It checks the supplied review against those artifacts and
documents the changes made. It remains an implementation-side investigation,
not an independent assessment of trading performance.

Active script: `bitfinex_indicators_v2.1.py`.
Measurements and reproduction utilities:
`.indicator-audit/static-review-2026-10-16/`.

## Summary

Confirmed robustness weaknesses have been fixed. A further missing-minute
decision-time consistency bug was found and fixed. The fixes leave all saved
2,000-entry features and targets unchanged; they make validation stricter and
make inference on a data prefix agree with the historical feature table.

The entry-candle versus signal-candle wave-1 definition, censoring policy and
wave-1 ADX exemption are strategy choices. Their existing behavior remains in
production, consistent with the instruction to retain the original exit rules.

## 1. Embedded source integrity — confirmed and fixed

Previously the embedded-source SHA constants were not enforced at runtime.
The current script calls `_verify_embedded_sources()` at import and before
loading an embedded module. A mismatch raises `RuntimeError`; checks therefore
remain active under `python -O`, unlike ordinary assertions.

The local comparisons also verified:

- The two original files match their recorded original-file hashes.
- Both embedded source strings match their recorded embedded-subset hashes.
- All 26 retained v9 functions/classes and 14 retained v5 functions/classes
  have identical AST bodies to the supplied originals.
- A deliberately changed source string is rejected by the new guard.

Hashes stored next to source detect accidental inconsistency; an editor could
change source and hashes together. They are not proof of authenticity. The
original files are included in the independent audit bundle for direct review.

## 2. Wave 1 uses the entry candle — confirmed; production unchanged

For **151 / 2,000 entries (7.55%)**, the M5 histogram changes sign between the
last closed signal candle and the M5 candle opening at entry. That entry
candle's histogram is known only five minutes after entry.

This does not enter `model_features`, but it does select a different future
label horizon. It is inherited from the supplied collector's entry-candle
lookup; whether it is the desired target definition needs an explicit decision.

A separate counterfactual simulation anchored wave 1 to the last closed signal
candle while retaining prices and all other exit rules. Among the 151 affected
entries it changed **87 return targets** and **three availability flags**.
This scenario was not applied to the main script or workbook.

Evidence: `wave1_and_duration_by_entry.csv`,
`signal_wave_definition_scenario.csv`, and their JSON summaries in the follow-up
directory. These counterfactual figures are separate from the earlier 22 target
changes caused by approved indicator-input corrections.

## 3. Wave-1 ADX exemption / anchor-specific durations — confirmed

Production retains the original rule: ADX > 25 is checked only in numbered
waves 2 and 4. Anchor 14 and anchor 22 do not override that rule.

Measured on the saved run:

- Anchor 14: 87 entries; 43 terminate from ADX on the first candle of wave 2;
  median series duration 80 minutes.
- Anchor 22: 91 entries; 48 terminate from ADX on the first candle of wave 2;
  median series duration 75 minutes.
- Union of those anchors: 174 distinct entries; 87 hit ADX on the first wave-2
  candle, and 133 somewhere in wave 2; median duration 77.5 minutes.

Series duration is measured from entry to ADX detection or the research cutoff,
not necessarily the holding time of a continuously open position. Per-anchor
rows overlap when several anchors fire together; do not add their counts as
independent samples.

All 18 main-anchor duration distributions are saved in
`series_duration_by_main_anchor.csv`. No target-horizon standardization or
additional ADX entry filter was introduced.

## 4. Target exclusion / censoring — real modeling issue; unchanged

All **15 unavailable targets in the selected 2,000 rows** are unfinished
positions at the wave-6 cutoff. There are **zero no-wave-6 cases** in those
2,000 rows. The unavailable rows remain present with `target_available = 0`
and `target_status`; they are not silently assigned zero return.

The entire frozen snapshot contains 2,353 main-anchor opportunities. Outside
the first 2,000, the final 34 opportunities have no sixth wave in the snapshot.
Their IDs are recorded in `findings_measurements.json`.

Excluding unfinished targets from training can be informative censoring even
when uncommon. Mark-to-market targets would redefine the labels; they were
not added because the user explicitly requested original gross returns and
unavailable unfinished targets. The current availability/status fields expose
the distinction but do not statistically correct selection bias.

## 5. ADX detection versus actual fill — measured

There are 1,611 ADX-ended series and 144 actual forced closes of active positions.
One series and active forced close are delayed by a missing M1 interval:

- Entry ID **927**, trade 3.
- Detection: **26 September 2026, 22:45 UTC+3**.
- First observed M1 opening / simulated forced close: **22:47 UTC+3**.
- Delay: **two minutes**; fill price 2681.9.

All other observed ADX-ended series have no such delay. Full series-level
measurements are in `adx_detection_to_actual_open.csv`; actual forced fills
are in `adx_forced_fill_delays.csv`.

The recorded detection clock and actual trade exit clock have different
meanings. The simulation continues to fill at the first observed opening,
rather than inventing a trade price at an empty minute. No strategy timing
change was made.

## 6. Warm-up — assertion missing; claimed cache shortage not present

The old code checked finite features but did not enforce the configured
minimum history. The current code explicitly requires at least 260 closed
indicator slots on every feature timeframe before the first entry, plus
sufficient M5 and M1 exit-indicator warm-up.

The saved cache does not start every timeframe at the same timestamp. Actual
closed candles before the first selected entry are:

- M1: **4,313** actual records / 4,320 slots after filling gaps.
- M5: **2,807** actual records and slots.
- M15: **7,602** actual records and slots.
- M30: **8,801** actual records / 8,823 slots after filling gaps.

The estimate of 144 M30 bars applies to a hypothetical three-day-only M30
history, not this cache. A separate three-day truncation experiment did show
M30 sensitivity: maximum normalized-feature difference approximately
0.000154620748 across the first 100 entries. The new guard rejects that
insufficient M30 input. Retaining only 260 slots produced differences up to
approximately 4.36e-8 across tested normalized features; 260 slots is not a
mathematical guarantee of exact initialization invariance.

Removing every future candle from the first 100 entries gave exactly identical
model features. That test alone missed the seven sparse-minute cases below.

## 7. Modeling risks — acknowledged; no unrequested feature/label redesign

- Four log close levels can be highly correlated and can encode price regime
  or time. Log volume is not automatically stationary either. No statistical
  stationarity or predictive-value claim has been established.
- Main-anchor-only rows intentionally define a conditional opportunity dataset.
  There is no no-signal class. This is separate from predicting loss/profit for
  an opportunity that exists.
- Overlapping hypothetical trades do not provide 2,000 independent examples.
  Chronological splitting, label-interval purging, embargo and LSTM sequence
  construction remain future work.
- The actual last label-information time may be earlier than `series_cutoff_at`
  when ADX terminates a series; the latter is the wave-6 horizon. Check actual
  exit/order times and feature-window overlap when defining splits.

No anchors were dropped, no no-signal rows were added, and no model was trained.

## 8. Smaller robustness findings

- **Empty M1 wave results:** `pd.DataFrame(wave_rows)` could lose its schema when
  empty. It now supplies `MACD_1M_WAVE_COLUMNS` explicitly. An empty-wave fixture
  successfully normalized and passed through the exit engine.
- **Membership parsing:** only the two specifically confirmed numeric lists
  10.31 and 25.26 are accepted as numeric multi-ID cells. Other numeric decimals
  are rejected because Excel may have discarded trailing zeros. Text `10.30`
  or `10,30` parses as IDs 10 and 30. Membership validation against the actual
  supplied workbook passes.
- **Optional workbook validation:** default runs still use the embedded confirmed
  configuration. Metadata now explicitly records whether a workbook was
  validated. An explicit workbook records its path and SHA. The refreshed
  audited export was validated against the supplied workbook.
- **Spot feed / costs:** the feed remains ETH/USD spot OHLCV. Short borrow,
  funding, fees and slippage are not modeled. Gross simulated returns should
  not be read as executable net returns.
- **Absolute X:** 0.3 remains an absolute price offset, as the user specified.
  No automatic price-, tick- or volatility-scaling was introduced.

## 9. Additional confirmed bug: trailing empty-minute feature clock

Beyond the static findings, prefix tests found seven affected entries:
**1278, 1584, 1608, 1636, 1643, 1658, 1666**.

Each immediately follows a minute missing from the API. With a full historical
snapshot, internal gap filling supplies a flat candle at that minute. With
only data available at entry, the old gap filler stopped at the last observed
candle. `features_at_entry()` then selected an older M1 feature row, and could
also align higher timeframes to that older availability clock.

This was an inference/historical-feature consistency defect. Some normalized
volume differences exceeded five log units, so it was not display rounding.

The new `closed_indicator_history(candles, cutoff)`:

1. Excludes every candle not yet closed at the explicitly known decision time.
2. Fills internal gaps and advances trailing empty, already-closed slots with
   previous-close OHLC and zero volume.
3. Never fills a forming candle or uses a future candle to obtain the price.

`features_at_entry()` uses this clock-based history before feature alignment.
The first 100 entries plus all seven edge entries now match the saved features
**exactly** after all future candles are removed: **107/107**, max difference 0.

This assumes omitted slots in a successfully retrieved candle history are empty
trading intervals. It does not distinguish no trades from an upstream data
outage. Stop/support/fill processing still uses actual API candles only.

## 10. Regression results and audit bundle

- All **2,000 model-feature rows** remain exactly unchanged.
- All **2,000 targets** remain exactly unchanged.
- Full exit outputs also match exactly: 4,305 trades, 16,868 stop-event rows,
  4,488 support searches, 3,775 entry-order rows and 2,000 summaries.
- The original embedded function bodies remain unchanged. Fixes are in the
  surrounding adapter, validation, membership parser and feature-clock code.
- The output workbook/CSV metadata was refreshed to identify the current code
  and checks; the saved market features and targets were preserved.

`BITFINEX_INDEPENDENT_AUDIT_BUNDLE.zip` supplies the four candle JSONs, current
script, both supplied originals, membership workbook, output workbook/CSVs,
handoff and measurements. Its manifest records SHA-256 hashes of included files.
The new reviewer can independently reconstruct entries and exits from raw data.

No browser audit was repeated for these fixes: chart-derived market values and
saved targets did not change. The new evidence is regression, provenance,
history-length and decision-time-prefix checking. The earlier 40-case chart
audit remains historical evidence with the limitations described in the handoff.
