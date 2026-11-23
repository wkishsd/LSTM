# Exit model: ideas and possible execution

Status: **paused** (23 Nov 2026). This note keeps the reasoning and a plan for when we come back to it. Nothing here is implemented yet, apart from the tools listed under "Already in place".

## Why exits matter

- With the current rule-based exits, no setup is profitable after fees. On filled series, 15m stop, 5m anchors: win rate 34.5% against a break-even of 36.1%, net −0.024% per series.
- Anchors do not beat random entry times on net return (0.9 SE with the 1m stop, 1.5 SE with the 5m stop). With the 5m stop, anchor winners run further than random ones (+0.788% vs +0.714%, 2.9 SE, overstated because series overlap), while losers are the same size.
- Exit timing changes results a lot. In the wave-horizon study (15m anchors, 15m stop), moving the series cutoff by one M5 half-wave changes net return by about 0.05–0.07 points. Cutoffs at even waves (start of a positive half-wave) do better than odd ones. That swing is larger than the anchor-vs-random gap.
- The current MLP labels are made at entry: "will the whole series, under fixed rules, end net positive?" A model trained on them can only skip entries; it cannot change how a trade ends.

## The idea: a tandem entry + exit model

- **Entry model:** at each anchor, enter or skip (as now).
- **Exit model:** while a position is open, at specific decision points, hold or close now. Optionally, skip a re-entry the rules would place.
- **Current exit logic stays as the safety net.** Stops, ADX closes and re-entries keep running. The model may only close earlier or skip a re-entry; it never widens a stop or holds past one, so it cannot lose more than the rules already do.
- **Hard cap at M5 half-wave H.** Everything closes at the start of half-wave H. A cap on re-entries per series is optional. This keeps the model from churning in circles. The horizon study picks H; an even wave looks preferable.

## Decision points: events, not every candle

Deciding at every M5 close gives roughly 100 strongly correlated samples per series, which a model can easily memorize. Use the same discipline as entries, where anchors are the only decision points:

1. **Mirror anchors.** The upside counterparts of the 18 main anchors, evaluated on the same timeframe, for example:
   - RSI crossing back up through 30 or 50 (mirror of 1, 3);
   - MACD crossing above its signal (mirror of 8, 22);
   - a negative histogram starting to rise (mirror of 9, 10);
   - +DI crossing above −DI (mirror of 13);
   - ADX turning down in a bearish trend (mirror of 19).
2. **Events the exit logic already produces.**
   - The stop moved down (green candle with a basis red candle).
   - A support level found (before a re-entry order).
   - An M5 half-wave boundary.
3. **A minimum gap.** One decision per event, none within about 15 minutes of the previous one.

Training guards:
- Weight each series equally, so long series with many decisions don't dominate.
- Keep all decisions of a series in the same fold.
- Use time-ordered folds only.

## How the exit model learns (without full reinforcement learning)

- **Label at a decision point** = net return if we close now (paying the exit fee) minus net return if we continue under the current rules up to the cap. The simulator replays the "continue" branch from that exact moment; with the fast exit path this is cheap.
- **Inputs at a decision point:**
  - the same 23.5-hour feature window ending at the decision time;
  - trade state: unrealized P&L, time in trade, current half-wave number, distance to stop, re-entries so far, fees paid.

  All of it is known at decision time.
- **Iteration:** once an exit model exists, the "continue" branch can follow the model instead of the rules; relabel and retrain. One or two rounds usually settles it.

## Coupling entries and exits

- An entry's label should be the outcome under the learned exit policy, not under the fixed rules. Train the exit model first, then relabel entries with it, then train the entry model.
- **Avoid leakage.** Exit predictions used for entry labels must come from models that did not see those trades. Fit on separate time blocks and use each block's predictions on the others. Otherwise entry labels look better than reality.
- **Time order throughout.** Train on earlier periods, test on later ones. The final test is Feb–Nov 2026, the weak period.

## What would count as a real result

- **Beat the rules.** The combined policy beats the rule-based baseline on the 2026 test, after fees.
- **Attribution.** The same exit model applied to **random** entries shows how much of the gain comes from exits and how much from anchors. Exit skill that works on random entries is still useful, but it would mean the anchors add nothing.
- **Ceiling.** The hindsight-best single exit per series (entry to the lowest low before the cap, minus fees) is an upper bound, never achievable. If even this barely covers fees, an exit model has nothing to work with.
- **Simple baseline.** A simple exit rule is the bar to beat, for example close at the first mirror anchor after the price has moved in our favour.

## Risks

- Memorizing noise from many correlated decision points (mitigated by events, equal series weights, grouped time folds).
- 2026 behaves differently from 2023–2025; models fit on the earlier years may not carry over.
- Early closes that each look reasonable can lose the gain to extra fees; the label includes the exit fee for this reason.
- Hindsight leakage: every feature must be known at the decision time; trade-state features must be computed causally.

## Possible execution, in order

1. **Ceiling and event study (go/no-go).** For filled series up to wave 20, anchors and random entries:
   - the hindsight-best exit return;
   - when it happens (which half-wave, how long after entry);
   - how many decision points each event source produces per series;
   - how close each source's events are to the best exit.
2. **Decision hook in `fast_exit`.** Let a policy return "close now" or "skip re-entry" at decision events. Check that a policy that never acts reproduces today's labels exactly (`tests_fast_exit.py`).
3. **Exit dataset.** Decision events, state features, and close-now vs continue labels for every filled series.
4. **Exit model.** Train it and evaluate it as a policy in the simulator against the rules, the simple baseline and the ceiling, on anchors and on random entries.
5. **Entry relabel and entry model.** Relabel entries under the exit policy (leakage-safe), train the entry model, evaluate the pair on the 2026 test. Optionally iterate once.

## Already in place

- `Pipelines/Bitfinex/fast_exit.py`, `run_fast.py`, `tests_fast_exit.py`: an array-based exit loop that gives results identical to v9, about 4× faster labeling, and an equivalence test.
- `--stop-timeframe 1m|5m|15m`, `--anchor-timeframe 5m|15m`, `--wave-horizon H` in the collector.
- `Pipelines/Bitfinex/wave_horizon_study.py`: labels one set of entries at several cutoffs. The output of the first run (15m anchors, 15m stop, waves 6–20) is in `Exports/training/wave_horizon_15manchor_15mstop/`.
- `random_minute_control.py`: random-entry control (currently uses the plain collector, not the fast path).
