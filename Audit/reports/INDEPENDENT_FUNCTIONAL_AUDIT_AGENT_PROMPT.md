# Prompt for an independent reviewing agent

Perform a bounded, independent review of the Python script and the small functional audit described below. Verify the evidence and conclusions; treat all prior PASS claims, including this handoff, as unverified. Return a short report with concrete expected versus actual values and one verdict: **“sample checks passed,” “confirmed defect,” or “inconclusive.”**

## Files

Workspace: `C:\Users\wkishsd\Downloads\otcheti\LSTM`

- Active script: `C:\Users\wkishsd\Downloads\otcheti\LSTM\bitfinex_indicators_v2.1.py`
- Original frozen audit bundle: `C:\Users\wkishsd\Downloads\otcheti\LSTM\BITFINEX_INDEPENDENT_AUDIT_BUNDLE.zip`
- Implementation handoff: `C:\Users\wkishsd\Downloads\otcheti\LSTM\SCRIPT_AUDIT_HANDOFF.md`
- Newly written independent audit: `C:\Users\wkishsd\Downloads\otcheti\LSTM\.independent-functional-audit`
- Extracted bundle: `C:\Users\wkishsd\Downloads\otcheti\LSTM\.independent-functional-audit\bundle`

This is a single-file prompt, not a self-contained data archive. It references files on the shared filesystem. If those files are unavailable, identify that limitation instead of claiming to have verified the audit.

## Authorized scope and constraints

Inspect the active execution path for:

1. Closed-candle timing and accidental future-data use in features or anchor decisions.
2. Main-anchor filtering, simultaneous activations and the explicit default 2,000-entry cap.
3. Separation of model features from targets.
4. Missing-candle handling and sufficient warm-up.

Independently calculate expectations from frozen raw candles for these **12 entries**:

`1, 500, 1000, 1069, 1278, 1584, 1608, 1636, 1643, 1658, 1666, 1900`

Entries `1278, 1584, 1608, 1636, 1643, 1658, 1666` are mandatory because they follow missing minutes. The other five cover earlier/later dataset regions, stop placement, downward stop movement, stop exits, re-entry, ADX termination and an unavailable unfinished target.

Compare anchor IDs and assignment times; source OHLCV and normalized features across M1/M5/M15/M30; stop levels, event ordering, support and re-entry levels, exit times and final gross-return targets.

**Do not call production calculation functions to generate expected answers.** Running production functions to obtain actual results is allowed. The prior independent calculator is evidence to inspect, not an oracle to trust. Check its interpretation and independence, especially whether it accidentally consumes saved outputs as expected prices, events or returns. It uses saved entry IDs/timestamps to locate cases, then derives expected entry prices and calculations from raw candles.

Use the existing rules, including entry-candle wave numbering. Chart-aligned indicator inputs intentionally differ from the supplied original scripts; exact numerical parity with original indicator inputs is not required. Flag ambiguous rules separately from implementation errors.

Keep the work bounded: **no live API requests, browser checks, model training, strategy optimization, reconstruction of all 2,000 target paths, or production-code edits.** Expand the sample only to investigate a discrepancy. Computing indicator history and inspecting the entry-selection cap is permitted; do not replay all 2,000 exit series.

## Data and rules to verify against the sources

- Frozen candle arrays are `[MTS, OPEN, CLOSE, HIGH, LOW, VOLUME]` in `bundle\candles\{1m,5m,15m,30m}.json`.
- Native higher-timeframe candles are supplied separately; do not resample M1 to substitute for them.
- Main anchors: `1,3,5,7,8,9,13,14,15,16,19,22,23,24,25,26,28,33`. Membership comes from the bundled workbook, with the agreed `10.31` and `25.26` cells interpreted as pairs of IDs.
- Evaluate anchors on closed M5 candles and assign entry at the next M5 opening. Simultaneous main anchors create one row with multiple flags.
- Features use only completed candles available at entry. Fill empty indicator slots with previous-close OHLC and zero volume, including trailing empty slots through the known decision clock. Execution and support searches use actual M1 records.
- Per-timeframe indicators: RSI14, ATR14, MACD12/26 signal9, DMI14 and ADX smoothing4. DMI ties use decimal-safe price differences.
- Nine normalized features per timeframe: ln(close), ln(volume), (RSI−50)/50, ATR/close, MACD/ATR, signal/ATR, ADX/100, DI+/100 and DI−/100. Zero log inputs are explicitly encoded as zero.
- Model inputs comprise 36 normalized market features plus 18 main-anchor flags. Targets/outcome metadata are excluded.
- Initial and re-entry positions seed up to the last 20 red candles before or including the entry/trigger candle. Later red history is retained.
- On a green candle close, find the nearest retained red candle with high >= green high. Candidate stop is red high + 0.3. Set the first stop or lower an existing stop; never raise it.
- Check an existing stop against the actual candle high before processing a stop update. Stop fills occur at the exact stop level and are timestamped at the candle close.
- While flat in a positive M1 MACD histogram half-wave, the immediately preceding negative wave can be searched once, after the first positive candle closes. Support uses actual wave lows and relative cluster tolerance 0.0003. A valid support is the global minimum low; re-entry offset is zero, and the order level must be below current close.
- Pending orders persist. A subsequent actual low <= level triggers re-entry at that exact level, timestamped at candle close. Do not evaluate a new exit stop inside that trigger candle.
- M5 wave 1 contains the entry candle. The research horizon ends at the start of numbered wave 6, exclusive. Histogram zero carries the prior sign.
- ADX >25 in numbered M5 waves 2 and 4 becomes actionable only when that M5 candle closes. The first available M1 opening at/after detection terminates the series, closing an active position at its open and canceling a pending order. ADX takes priority over other processing.
- An open position at the wave-six cutoff leaves the complete target unavailable. Partial realized return must not be substituted.
- Gross target is the sum of six-decimal per-position short returns, rounded to six decimals; it is not compounded and excludes costs.
- Display times below are fixed **UTC+3**.

## Prior audit implementation and evidence

The active script matched the bundle's script byte-for-byte at audit time. Recorded SHA-256: `fb24e14f09b509bf50d367a679b44a93111131971fdd183b4129726979a34add`. Recheck if files may have changed.

Within `.independent-functional-audit`:

- `independent.py`: separate raw-candle indicator, scalar-anchor and execution calculator. It does not import production or original helpers.
- `compare.py`: imports the active production script to obtain actual results, compares against independent expectations and bundled outputs, checks future truncation, inspects original embedded bodies and the membership workbook, and checks the cap. It disables socket connection and requests execution during its run.
- `expected_features.csv`, `expected_anchors.csv`, `expected_targets.csv`, `expected_trades.csv`, `expected_stops.csv`, `expected_orders.csv`, `expected_searches.csv`.
- `actual_entries.csv`, `actual_features.csv`, `actual_model.csv`, `actual_targets.csv`, `actual_trades.csv`, `actual_stops.csv`, `actual_orders.csv`, `actual_searches.csv`, `actual_summaries.csv`.
- `comparison_summary.csv`: column comparison counts, pass flags and maximum absolute errors; tolerance 1e-9 absolute, zero relative tolerance, with NaN equality.
- `future_truncation.csv`: per-case model differences and anchor equality after removing all candles closing after entry.
- `inspection.json`: source-body provenance comparisons, selection checks, model columns and warm-up counts.
- `run_log.txt`: last comparator output.

The bundle also contains original scripts, anchor membership workbook, full saved outputs and earlier audit evidence. Earlier audit reports are contextual claims, not independent proof.

The prior audit used local Python for `independent.py`, and the bundled interpreter for `compare.py` because local Python lacked openpyxl:

```powershell
Set-Location 'C:\Users\wkishsd\Downloads\otcheti\LSTM'
$env:PYTHONIOENCODING = 'utf-8'
python .independent-functional-audit/independent.py
& 'C:\Users\wkishsd\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .independent-functional-audit/compare.py
```

Read both audit utilities before running. They overwrite only their audit evidence files. Prefer another audit output directory if preserving the prior evidence matters. Runtime paths may change; locate an available interpreter if necessary, without making live market-data requests.

## Prior findings — all subject to verification

Prior verdict: **sample checks passed**, with no confirmed implementation defect in the inspected path/sample.

- 215 column-level comparison checks passed; expected values matched both current runtime results and bundled outputs for the compared fields.
- All 240 OHLCV values, 48 source timestamps and 432 normalized market values matched across the 12 cases. Maximum normalized absolute difference: **1.3322676295501878e-15**.
- All sampled anchor flags and assignment times matched. Example: entry **1584**, main anchors **8;16;23**, assigned **3 October 2026 17:10 UTC+3**, following the 17:05 M5 candle.
- Removing future candles changed none of the 54 model values or anchor flags in any sampled case.
- All seven gap cases used expected previous-close OHLC and zero-volume indicator slots.
- Reconstructed paths comprised **30 positions, 135 stop events, 27 support searches and 25 re-entry orders**, including **31 downward stop moves, 28 stop exits and 18 executed re-entries**. There was one ADX forced position close; other ADX terminations could occur while flat or with a pending order.
- Initial closed-history counts were M1 **4,320**, M5 **2,807**, M15 **7,602**, M30 **8,823**, above the enforced 260-slot minimum. These counts alone do not prove initialization insensitivity.
- Default entry selection returned exactly 2,000 unique chronological main-anchor entries. Selecting 2,001 preserved the first 2,000 unchanged.
- Embedded retained function/class AST bodies matched their supplied original counterparts. This proves provenance, not functional correctness.

Expected and actual final gross-return targets reported by the prior audit:

- **1:** +0.440573%; **500:** −0.259565%; **1000:** +0.273571%.
- **1069:** complete target unavailable; partial realized return +0.376578%.
- **1278:** +0.082292%; **1584:** −0.041065%; **1608:** −0.145455%.
- **1636:** +0.089070%; **1643:** +0.011133%; **1658:** −0.003700%.
- **1666:** −0.063170%; **1900:** −0.233609%.

Concrete trace examples:

- **Entry 1:** First stop **2416.7** at **16 September 23:30**; lowered to **2413.6 at 23:34**; stop exit **2413.6 at 23:35**. Re-entry order **2406.1 at 23:49**, filled **17 September 00:39**. Final target **+0.440573%**.
- **Entry 1278:** Missing **30 September 07:44** M1 slot has OHLC **2674.8**, volume **0**, normalized RSI **−0.052530004830**, MACD/ATR **−0.322332900291**. The separate **07:45** entry open is **2674.9**.
- **Entry 1643:** ADX forced close at the **4 October 09:15 M1 open**, price **2694.3**, target **+0.011133%**. Its 09:00 entry features use source openings **08:59 / 08:55 / 08:45 / 08:30** for M1/M5/M15/M30.
- **Entry 1069:** Fourth position remains open at the **28 September 11:45 wave-six boundary**. Complete target is unavailable despite **+0.376578%** realized from closed positions.

## Interpretation and limits

Numbered M5 waves 2/4 are not necessarily positive histogram waves. Preserve the requested entry-candle numbering and report this semantic qualification separately from defects. Empty-slot filling, exact-level fills and skipped exit evaluation inside the re-entry trigger candle are existing simulation assumptions; do not silently redesign them.

The previous sample was targeted, not random. It did not establish correctness of every entry, exercise every anchor branch or rare tie, establish initialization stability under different history lengths, validate real-world execution, or demonstrate profitability. Do not claim the whole script is proven correct because it runs, preserves original bodies or matches its own saved outputs.

## Required response

Return a short report covering:

- Cases and behaviors you actually checked.
- Matches and discrepancies with concrete expected versus actual values.
- Any confirmed bug, its code location and practical effect.
- Ambiguous rules separately from implementation errors.
- One verdict: **“sample checks passed,” “confirmed defect,” or “inconclusive.”**
- What the sample cannot establish.

State clearly whether you independently recomputed results, inspected existing audit evidence, or did both. Do not present this handoff's claims as newly verified facts.
