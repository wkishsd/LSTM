# Bitfinex profitable-entry MLP

This research checkpoint predicts the original profitable-entry target at the anchor decision. It uses a two-hidden-layer MLP trained to rank candidates with elevated ATR and applies a bounded adjustment to the ATR ranking. The recipe was selected on three development periods from the original export. The separate 2023 reproduction refit that fixed recipe on earlier outcomes before each block.

On 12,536 later decisions from June 17 to October 31, 2023, the development-selected 400-update model found 254 profitable labels in 797 calls. ATR found 248 at the same weekly call quota. The paired four-week resampling interval for the precision difference was −4.83 to +5.42 percentage points. The observed improvement was +0.75 points. The interval includes zero.

The 2023 model's winning-label count across 20, 40, 120, 240, and 400 optimizer updates was 245, 257, 253, 249, and 254. The 2026 comparison gave 214 versus 242 over 1,009 calls. The separate six-day 2026 slice gave 10 versus 10 at 37 calls. These are retrospective rankings within each calendar week. The 797-call result does not establish repeatable improvement across market periods.

## Replay the saved model

Extract the ZIP and run this command with NumPy and PyTorch installed.

```text
python replay_trade_checkpoint.py
```

The replay uses the archived inputs and both trained checkpoints per date block. It verifies 12,536 predictions and prints the 254 profitable labels. The included weights cover this 2023 replay period. New dates require a refit with outcomes known at the time.

`earlier_trade_curve.json` contains the per-step 2023 results. `tail_pair_mlp_results.json` contains the historical and recent 2026 comparisons.
