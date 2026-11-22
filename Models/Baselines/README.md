# Baseline models for the trade MLP

`trade_baselines.py` holds three non-neural models that the trainer runs alongside the MLP when you tick them under **Also train and test** in the training panel, or pass `--baselines atr,logistic,gbt` to `python -m mlp_trades train`. They answer one question: does the MLP learn anything a much simpler model cannot?

They use NumPy only and run on the CPU, so the MLP keeps the GPU. They need no extra packages in the `bitfinex-mlp` environment.

## Same data, same split

The trainer prepares the data once, then forks the baselines before PyTorch starts. The baselines call the trainer's own `training_split`, so they see exactly the same train, validation and test entries, including the unfilled-order exclusion and any data-period filter. Test metrics use the trainer's definitions: average precision, winners among the 1,000 highest-scored test entries against the number expected at random, and the mean net return per trade of those 1,000 entries against taking every test trade.

## Models

| Key | Model | Fitting |
|---|---|---|
| `atr` | Rank by M5 ATR / close at the decision minute | None. Binary datasets only. |
| `logistic` | L2 logistic regression on standardized features (one-vs-rest for profit buckets) | Newton's method. The L2 strength is picked on validation average precision. |
| `gbt` | Histogram gradient-boosted trees, depth 4, learning rate 0.05, 70% row subsampling | Early stopping on validation log loss, at most 600 rounds (300 for profit buckets). |

## Decision-time features (85)

Every value uses only rows at or before the entry's last closed minute.

- Per timeframe (M1, M5, M15, M30): RSI, ATR / close, ADX, DI+, DI−, MACD / close, signal / close, histogram / close, and ln(volume) minus its 60-row mean.
- M1 log returns over 1, 5, 15, 60, 240, 720 and 1,409 minutes, and realized volatility over 60 and 240 minutes.
- Each of the 18 anchors: whether it is active now, and whether it fired in the last 120 minutes.
- Hour of day and day of week as sine and cosine pairs.

MACD is divided by close rather than ATR, because MACD / ATR blows up when ATR approaches zero.

## Output

`baselines.json` in the run folder holds, per model, the settings, validation and test metrics, and the largest logistic coefficients or the features the trees split on most. Partial results are saved after each model, and an error is recorded instead of stopping the MLP.
