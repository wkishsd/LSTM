# Next-hour ETH volatility MLP

This model forecasts how much ETH will move over the next 60 minutes at Bitfinex anchor decision times. The target is log realized volatility from Binance ETHUSDT minute closes. The model uses 158 Bitfinex and Binance inputs computed at or before each decision.

| Optimizer updates per refit | Historical forecast MAE | New six-day forecast MAE |
| --- | ---: | ---: |
| ATR forecast | 0.30931 | 0.26978 |
| 20 | 0.29467 | 0.25460 |
| 120 | 0.26034 | 0.24487 |
| 240 | 0.25774 | 0.24171 |

Errors are in log-volatility units. The historical evaluation covers 15,832 anchor decisions across 24 calendar weeks. The new six-day evaluation covers 586 decisions. The 240-update MLP reduced MAE by 16.7% historically and 10.4% on the new slice. It beat the ATR forecast in each of the 24 historical weeks. The evaluation used four-week refits and required both training outcomes to be available before the evaluation feature windows began.

The same training recipe reproduced on 12,536 separate 2023 anchor decisions. Forecast MAE fell from 0.31413 after 20 updates to 0.30453 after 120 and 0.30161 after 240. ATR forecast MAE was 0.32184. The 240-update MLP beat the ATR forecast in 20 of 21 calendar weeks. All fits in this replication used labels from before their evaluation block. The early training data contains one 80-minute gap in the spot archives, which was advanced using the last available price and zero activity.

The same model's trade ranking at the development-selected recipe found 201 of 1,009 winners versus 242 for ATR on the historical comparison, and 10 of 37 for both on the new slice. The improvement in volatility forecasting does not establish an improvement in classifying profitable Bitfinex setups.

## Replay the saved model

The package contains the 240-update checkpoints for both seeds in each of the two date blocks intersecting the new slice, the precomputed inputs, and the saved forecasts. With Python, NumPy, and PyTorch available, run this from the extracted folder.

```text
python replay_volatility_checkpoint.py --verify-against verified_volatility_fresh.npz
```

This command runs inference on CPU, checks all 586 forecasts against the saved values, and prints `Log-volatility MAE 0.241706`. Add `--output forecasts.npz` to write the forecast values. Inference uses only decision-time inputs. The outcome array in the example data serves to score forecast error.

`volatility_verification.json` contains the forecast error by update count. `volatility_mlp_results.json` contains the trade-ranking metrics and the development-selected recipe. This snapshot contains only the checkpoints applicable to the included dates. Forecasting additional date blocks requires the same feature pipeline and a new refit.
