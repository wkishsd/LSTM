# Bitfinex MLP restart brief

## Active task

Find a reproducible cause of weak validation ranking on the original binary Bitfinex ETHUSD dataset. The user suspects a trainer or data-handling defect and wants a fresh diagnosis grounded in the actual data. The profit-bucket experiment is outside the current task. The launcher is `Training/rocm-9070/Training Panel.cmd` and the trainer is in `Training/rocm-9070/mlp_trades`.

## Starting evidence

- The active export is `Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd`. It contains 105,545 signals. The trainer uses the original net-positive binary target and chronological partitions.
- Run `Training/rocm-9070/runs/rx9070-20261120-024355-b25e` used 1,410 minutes, `32 → 16`, ReLU, dropout 0.4, batch 2,048, seed 42, and learning rate `1e-7`. Validation calls went from 15,614 in epoch 1 to 138 in epoch 320. An exact replay reproduced early and final call counts. The untrained model already made 15,650 calls with median entry score 53.8%, just above the 50% decision threshold. Changing only the output biases to the training class prior removed the initial flood in a 12-epoch control. This establishes what drove the call-count swing in this run. The validation plateau remains unresolved. See `Experiments/bitfinex-low-lr-audit/results.json`.
- The previous audit matched all 105,545 exported IDs, timestamps, and labels, 384 shuffled CPU/GPU windows, and 120 initial fills sampled against raw candles. A five-minute score shift did not improve ranking when the same outcomes were compared. These checks covered the tested paths and leave the upstream outcome simulator open to further investigation. See `Experiments/bitfinex-alignment-audit/`.
- A separate target issue was confirmed. The frozen export priced 862 wave-6 market exits before the confirming M5 close. Repricing 132 single-trade cases flipped eight labels, all in training. Current label availability is delayed to confirmation, while the exit price stays historical. The measured effect does not establish the cause of the plateau. See `Experiments/bitfinex-alignment-audit/raw_execution_checks.json`.

## Fresh diagnostic

Start from the original binary run and actual checkpoints. Recheck the active data, optimizer and evaluation paths for a fault that could explain weak held-out ranking. Keep score ranking, the 50% entry rule, and checkpoint selection distinct. Use a controlled test with a result that could disprove the suspected problem. Require direct evidence before calling a cause or changing production behavior.
