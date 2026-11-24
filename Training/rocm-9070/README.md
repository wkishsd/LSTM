# Bitfinex MLP training on the Radeon RX 9070

## Use the training panel

Double-click `Training Panel.cmd` in Windows Explorer. Edit the settings on the left and click **Start training**. The panel creates a new output folder for each run. It uses the same three-year export and Radeon launcher as the PowerShell commands below.

## Bitfinex profit buckets

Select **Bitfinex · Profit buckets**, set **Hold after fill (min)**, and click **Start training**. The default hold is 60 minutes. The collector reuses the original export's confirmed ETHUSD short fills and the frozen M1 candle cache. Features end at the last M1 close before the fill candle opened, so the candle that filled the order never reaches the model. (The source export stamps level-touch fills at their candle's close and open fills at their candle's open without saying which, so the window ends one minute before the stamp.) The hold is measured from the recorded fill time. Each example exits at the first actual M1 open at or after the timed exit. Missing candles postpone that exit by at most 5 minutes (`--max-exit-delay-minutes`); longer gaps leave the trade unlabeled. Exports collected before this change are rejected and must be re-collected. Split purging treats the profit as known at the exit candle's close. The mode excludes unfilled orders and leaves trades without an available exit unlabeled.

The collector deducts the entry fee on entry notional and the exit fee on exit notional. The five classes are negative, 0 to under 1%, 1 to under 2%, 2 to under 3%, and 3% or more of entry notional. Break-even belongs to the second class. Each example has one entry and one exit. The timed exit replaces the legacy simulator's stop and re-entry sequence.

The Results tab shows precision, recall, and counts for each class. Epoch history records a five-class confusion matrix, the majority-class accuracy baseline, macro recall, and macro average precision. The collector saves its labels and hold settings alongside the run's launcher files.

## UCI HAR stair benchmark

Select **UCI HAR · Stairs** under **Dataset**. The task is to classify upstairs versus downstairs walking from 128 sequential sensor samples at 50 Hz. The trainer receives the nine published sensor channels in order. Zero padding fills the remaining input columns. The benchmark uses the same models, optimizer, checkpointing, and progress panel.

The packaged stair subset contains 2,950 windows. With the default fractions it uses 1,610 for training, 449 for validation, and 891 for test. Participants stay in one split each, including the official test subjects. In this mode the train and validation fractions divide the official training subject pool. Lookback counts sensor samples. Keep the 128-sample preset for the full sequence.

The data are adapted from Human Activity Recognition Using Smartphones by Reyes-Ortiz, Anguita, Ghio, Oneto, and Parra, UCI Machine Learning Repository, 2013. The adaptation selects classes 2 and 3 and keeps the original sensor sequences and split assignments. [Dataset and CC BY 4.0 license](https://doi.org/10.24432/C54S4K)

From `Training/rocm-9070`, the data-only CLI check is `python -m mlp_trades validate --config train.json --dataset uci_har --lookback-minutes 128 --output-dir ./har-data-check`.

## Beijing PM2.5 forecasting benchmark

Select **Beijing PM2.5** under **Dataset**. The model sees up to 168 hourly observations at the Aotizhongxin station, including pollution, weather, calendar, and missing-reading indicators. It predicts whether PM2.5 will be higher six hours after the last input. Missing features carry forward from the past, and the label requires an actual PM2.5 reading at both endpoints.

The benchmark contains 33,606 labeled windows. The default chronological split retains 23,357 training, 4,900 validation, and 5,041 test windows after purging 308 around the boundaries. The launcher defaults to 168 hours and selects the checkpoint by validation average precision.

The local data are adapted from Song Chen's Beijing Multi-Site Air Quality dataset, UCI Machine Learning Repository, 2017. The adaptation keeps one station, adds time and missingness features, and derives six-hour direction labels. [Dataset and CC BY 4.0 license](https://doi.org/10.24432/C5RK5G)

From `Training/rocm-9070`, the data-only CLI check is `python -m mlp_trades validate --config train.json --dataset uci_air --lookback-minutes 168 --output-dir ./air-data-check`.

## Bitfinex dataset and run controls

With **Bitfinex ETHUSD** selected, **Data period → Last 6 months** takes the six calendar months ending at the frozen export's exclusive end date. It filters signal dates before the chronological split and retains the original timeline for lookback and outcome validation. Launch `Training Panel.cmd --bitfinex-six-months` to open the matched preset from the full-export average-precision pilot. The CLI option is `--data-period last_6_months`.

The **Model** dropdown offers **Dense MLP**, **Shared encoder**, and **Per-minute encoder**. Selecting an encoder fills in a 720-minute lookback and a `128 → 64 → 2` classifier. Both encoders map each minute through `54 → 16 → 8`. The classifier receives the resulting 5,760 values in chronological order. The shared encoder uses the same weights at every minute. The per-minute encoder learns a separate set of weights for each position in the window. The selected activation applies throughout, and dropout applies to the classifier's hidden layers. Run reports and checkpoints record the selected model.

The default is a chronological **70% train / 15% validation / 15% test** split before purging. **Keep best by** defaults to **Average precision**, which ranks all validation entries by score across thresholds. **Results** charts it against the validation winner rate. **Epochs** reports accuracy, loss, average precision, recall, entry precision, balanced accuracy, entry counts, and timings. The final test evaluates the selected checkpoint once. Set validation to 0 to evaluate the last checkpoint. Use **Metric guide** for definitions, **Console** for trainer messages, or **Open folder** for the run files.

**Feature input** offers three experiments. **Stationary** combines relative price and relative MACD and also replaces each timeframe's ln(volume) with its difference from the window mean, so no input reveals the absolute price or volume level of the period. The first two are one-change experiments. **Relative price** uses each candle's log price relative to the last closed candle in the window on the same timeframe, in percent units. **Relative MACD** reverses the bounded transform in the frozen export, then divides MACD and its signal line by that candle's close before fitting the training-only standardization. Reversal uses the stored float32 values. **Exported features** uses the original columns.

The average epoch time appears above the tabs. New runs record the duration and running average for each epoch. Completed older runs show an approximate average based on file times when available, with per-epoch times left blank.

**Reload panel** restarts the panel on the current code without closing running trainers. It keeps the form values and the selected run. If the updated panel fails to start, the old one stays open and shows the error.

**Force pause** suspends the selected run. **Resume** continues it. Closing the panel leaves the trainer in its current state. Open it again to continue monitoring.

**Also train and test** runs CPU baselines alongside the MLP: an ATR rule, logistic regression, and boosted trees. They use the same entries and chronological split, start in a forked process before the GPU work begins, and write `baselines.json` to the run folder. The Results tab compares each model's test average precision, the precision of its top 1,000 test entries, and the mean net return per trade of those entries with the MLP, with a random ranking, and with taking every test trade. Net return is the series' gross return minus all fees, in percent. The CLI option is `--baselines atr,logistic,gbt`. The models are in [Models/Baselines](../../Models/Baselines/README.md). **Force pause** pauses the baselines together with the trainer.

**Train labels** changes only what the model trains on, using `label_agreement.csv` in the export folder. That file records, for each filled entry, in how many of 15 exit configurations (stop 1m, 5m, 15m × cutoff at M5 half-wave 6, 10, 15, 20, 30, ADX close off) its series ends with a net return above zero. Build it with `python Pipelines/Bitfinex/label_agreement.py --anchor-timeframe 5m --export <export folder>` from the exit strategy study runs; the trainer refuses a file whose entry times differ from the export.

| Option | CLI value | Training split |
|---|---|---|
| Export labels | `off` | The export's own labels (default) |
| Weight by agreement | `weight` | Same labels; each epoch draws entries with weight \|2 × wins / 15 − 1\|, so unanimous entries count fully and a 7–8 split barely counts |
| All 15 win vs rest | `unanimous` | Class 1 only for entries every configuration wins |
| All 15 win vs all lose | `unanimous_vs_lose` | Class 1 for unanimous winners, class 0 for unanimous losers; the entries in between are left out |

Validation and test keep every entry and the export labels in every mode, so results compare directly with an `off` run. The baselines train on the same labels (weight mode draws their training set once with the same weights). `split_report.json` and `baselines.json` record the mode, the entries kept and the class-1 rate before and after. The CLI options are `--label-agreement` and `--agreement-file`.

**Export run** saves a Markdown report for the selected run with its summary, saved launch parameters, data split, and available epoch and test metrics.

**Dropout** controls the fraction of hidden activations randomly masked during training after each hidden activation. The launcher default `0.2` masks 20%, and `0` disables it. A fresh mask is sampled on each forward pass. Frozen evaluation, validation, test and inference use all activations. The run parameters and checkpoint save this setting. Open **Metric guide** for the explanation.

The three-year ETHUSD export with bounded MACD is in `Exports/training/mlp_2023-11-13_to_2026-11-13_bounded_macd`. For training, the loader excludes the 13,261 signals whose initial order never filled: the export labels them 0, but an unfilled order is not a losing trade. It accepts the remaining 92,284 filled signals. The default split retains 64,519 training, 13,747 validation, and 13,843 test entries after purging 175 overlaps. Predictions are therefore conditional on the order filling. The model fits standardization on training minutes and saves those statistics in its checkpoint.

On the RX 9070, the launcher caches the shared minute timeline in VRAM when there is room. The trainer assembles windows there from the same shuffled entry indices. The Console and `run.json` report which loader was used. Use `--timeline-cache off` to use the CPU loader for a run.

## Environment

- Windows 11 with WSL2 and Ubuntu 24.04.
- AMD Adrenalin with WSL GPU support. On this machine, Adrenalin 26.10.1 exposes the RX 9070 to ROCm through WSL.
- ROCm 7.2.1 and ROCDXG 1.2.0 in Ubuntu. AMD's 7.2.1 stack needs `HSA_ENABLE_DXG_DETECTION=1`, which the launcher sets.
- The AMD wheel uses Ubuntu's `libgomp1` and the shared libraries from the ROCm package repository.
- The Ubuntu user `mlp-runner` has a Python 3.12 virtual environment at `~/.venvs/bitfinex-mlp`. It uses AMD's PyTorch 2.9.1 ROCm wheel and NumPy 1.26.4.
- PyTorch uses `cuda:0` as the device name for the Radeon through AMD HIP. The launcher checks that the installed build uses HIP and the selected device is the RX 9070.

AMD documents this setup in its [WSL guide](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/howto_wsl.html) and [compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html).

## Launch from PowerShell

Run the GPU check alone to verify a forward pass, backward pass and optimizer step with the configured MLP shape.

```powershell
wsl.exe -d Ubuntu-24.04 -u mlp-runner -- bash /mnt/c/Users/wkishsd/Downloads/otcheti/LSTM/Training/rocm-9070/train.sh --probe-only
```

Start each run with a fresh output directory. From PowerShell, the following commands switch to the setup folder, create a unique run name, and launch training in the current terminal. The console prints one record per completed epoch.

```powershell
Set-Location 'C:\Users\wkishsd\Downloads\otcheti\LSTM\Training\rocm-9070'
$launcher = '/mnt/c/Users/wkishsd/Downloads/otcheti/LSTM/Training/rocm-9070/train.sh'
$runName = "rx9070-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
Write-Host "Run folder: runs/$runName"
wsl.exe -d Ubuntu-24.04 -u mlp-runner -- bash $launcher --output-dir "runs/$runName"
```

## Change the settings

Edit `train.json` for persistent defaults. Override settings for a single run with flags. This example changes the epoch count and batch size. Use a fresh `$runName` before each launch.

```powershell
$runName = "rx9070-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
wsl.exe -d Ubuntu-24.04 -u mlp-runner -- bash $launcher --output-dir "runs/$runName" --epochs 50 --batch-size 64
```

- `--epochs` controls the number of passes through the training data. The default is 20.
- To run an encoder from PowerShell, add `--model shared --lookback-minutes 720` or `--model per_minute --lookback-minutes 720`.
- **Lookback (min)** defaults to 360 in the panel and takes the latest 360 rows ending at the entry minute. Use `--lookback-minutes` to change it in a launcher command. Checkpoints store the lookback for inference. The trainer uses the export's original eligibility and split boundaries.
- `--batch-size` controls how many examples the optimizer processes per update. The default is 32.
- `--learning-rate` controls the size of each weight update. The launcher default is 0.0001.
- `--selection` applies when validation is enabled. Choose `accuracy` for highest validation accuracy or `loss` for lowest validation loss.
- `--selection average_precision` selects the highest validation ranking score. `--feature-input relative_price` or `--feature-input relative_macd` changes one input family.
- Choose **Above baseline** in the panel, or `--selection profitable_above_baseline`, to maximize profitable calls minus (entry calls × validation positive rate). **Min. entry calls** defaults to 30 and can be set with `--min-entry-calls`. Lower validation loss breaks ties. If every epoch falls below the minimum, the run records the reason and finishes without testing a checkpoint.
- The first two hidden layers default to 128 and 64 neurons. Check **Third hidden layer** to add one more hidden layer with an editable width of 32 neurons. For command-line runs, use `--hidden3 32` to enable it or `--hidden3 0` to disable it. The extra layer uses the selected activation and dropout settings.
- The launcher defaults for `--train-fraction` and `--validation-fraction` are 0.70 and 0.15. Test receives the remaining 0.15. The split removes earlier examples whose windows or outcome intervals overlap the next partition.

## Monitor progress

Open another PowerShell tab in the setup folder. Use the run name printed at launch. For the original session, use `first-rx9070-run`.

```powershell
$run = 'runs/first-rx9070-run'
Get-Content "$run/history.jsonl" -Tail 1 -Wait
```

Each line has the epoch number, online training metrics, frozen evaluation metrics, and GPU memory use. The epoch timing includes training, evaluation, and checkpoint writes. There can be a pause before the first record while the loader checks the data and the first epoch finishes. Ctrl+C in the monitoring tab stops the log view.

## Check the results

```powershell
$result = Get-Content "$run/test_metrics.json" -Raw | ConvertFrom-Json
$result.checkpoint_epoch
$result.test | Format-List accuracy,balanced_accuracy,precision_enter,recall_enter,count
$result.test.confusion_matrix
```

`test_metrics.json` records the final test result or the reason no checkpoint qualified. `last.pt` holds the latest completed epoch. Runs with validation save `best.pt` when an epoch qualifies. The actual settings are in `run.json`.
