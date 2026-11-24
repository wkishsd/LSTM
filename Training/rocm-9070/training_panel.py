"""Windows launcher and viewer for the existing Bitfinex MLP trainer."""

import ctypes
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tkinter as tk
from datetime import datetime, timezone
from tkinter import filedialog, messagebox, ttk
from uuid import uuid4

from mlp_trades.schema import (
    BENCHMARK_LOOKBACKS, DATA_PERIOD_LABELS, DATASET_LABELS, DEFAULT_PROFIT_HOLD, ENCODER_PRESET,
    ENCODER_WIDTHS, FEATURE_INPUT_LABELS, INPUT_TIMEFRAME_LABELS, LABEL_AGREEMENT_LABELS, LOSS_LABELS, MODEL_LABELS, PROFIT_CLASSES, PROFIT_DATASET, STATES, WIDTH,
    window_input_size,
)


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PANEL_LOGS = RUNS / ".panel"
LEGACY = "first-rx9070-run"
WSL = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "wsl.exe"
WSL_SESSION = [str(WSL), "-d", "Ubuntu-24.04", "-u", "mlp-runner"]
OPTIONS = (
    ("dataset", "Dataset", str, "bitfinex"),
    ("bitfinex_dir", "Export", str, ""),
    ("data_period", "Data period", str, "all"),
    ("profit_hold_minutes", "Hold after fill (min)", int, DEFAULT_PROFIT_HOLD),
    ("model", "Model", str, "dense"),
    ("input_timeframe", "Input candles", str, "1m"),
    ("feature_input", "Feature input", str, "exported"),
    ("epochs", "Epochs", int, 20),
    ("batch_size", "Batch size", int, 32),
    ("learning_rate", "Learning rate", float, 0.0001),
    ("loss", "Loss", str, "cross_entropy"),
    ("label_agreement", "Train labels", str, "off"),
    ("selection", "Keep best by", str, "accuracy"),
    ("min_entry_calls", "Min. entry calls", int, 30),
    ("hidden1", "First hidden layer", int, 128),
    ("hidden2", "Second hidden layer", int, 64),
    ("hidden3", "Third hidden layer", int, 0),
    ("lookback_minutes", "Lookback (min)", int, 360),
    ("activation", "Activation", str, "leaky_relu"),
    ("initialization", "Initialization", str, "default"),
    ("dropout", "Dropout (0 to <1)", float, 0.2),
    ("train_fraction", "Train fraction", float, 0.80),
    ("validation_fraction", "Validation fraction", float, 0.0),
    ("seed", "Random seed", int, 42),
)
SELECTION_LABELS = {
    "accuracy": "accuracy",
    "loss": "loss",
    "average_precision": "Average precision",
    "profitable_above_baseline": "Above baseline",
}
OPTION_LABELS = {
    "initialization": {"default": "Default", "ones_zero_bias": "Weights 1, biases 0"},
    "dataset": DATASET_LABELS, "data_period": DATA_PERIOD_LABELS, "model": MODEL_LABELS,
    "feature_input": FEATURE_INPUT_LABELS, "selection": SELECTION_LABELS,
    "loss": LOSS_LABELS, "label_agreement": LABEL_AGREEMENT_LABELS, "input_timeframe": INPUT_TIMEFRAME_LABELS,
}
CHOICES = {
    # "Weights 1, biases 0" stays labeled for old run reports but can no longer be launched.
    "initialization": (OPTION_LABELS["initialization"]["default"],),
    "dataset": tuple(DATASET_LABELS.values()),
    "data_period": tuple(DATA_PERIOD_LABELS.values()),
    "model": tuple(MODEL_LABELS.values()),
    "feature_input": tuple(FEATURE_INPUT_LABELS.values()),
    "selection": tuple(SELECTION_LABELS.values()),
    "activation": ("relu", "leaky_relu"),
    "loss": tuple(LOSS_LABELS.values()),
    "label_agreement": tuple(LABEL_AGREEMENT_LABELS.values()),
    "input_timeframe": tuple(INPUT_TIMEFRAME_LABELS.values()),
}
BASELINE_LABELS = {"atr": "ATR rule", "logistic": "Logistic regression", "gbt": "Boosted trees", "hour": "Hour-of-day rule"}
BENCHMARK_CLASSES = {"uci_har": ("downstairs", "Upstairs"), "uci_air": ("no rise", "PM2.5 rise")}
BENCHMARK_DESCRIPTIONS = {
    "uci_har": "UCI HAR · Upstairs vs. downstairs · 50 samples/sec",
    "uci_air": "Beijing PM2.5 · Forecast a rise six hours ahead",
}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


EXPORTS = ROOT.parents[1] / "Exports" / "training"


def bitfinex_exports(current=None):
    """--bitfinex-dir value (relative to this folder, as train.json writes it) -> label, for each binary export.

    The label names the anchor and stop timeframes from the manifest. Exports built before the stop-timeframe
    option follow the old ADX / wave-6 rules and are marked as such.
    """
    exports = {}
    for path in sorted(EXPORTS.glob("mlp_*")):
        if not all((path / name).is_file() for name in ("manifest.json", "feature_timeline.npy", "entry_targets.csv")):
            continue
        manifest = read_json(path / "manifest.json")
        if manifest.get("profit_target"):
            continue
        stop = manifest.get("exit_settings", {}).get("stop_timeframe")
        if stop:
            label = f"{manifest.get('anchor_timeframe', '5m')} anchors · {stop} stop"
        else:
            suffix = path.name.split("2026-11-13")[-1].strip("_").replace("_", " ") or "original"
            label = f"Old rules · {suffix.replace('macd', 'MACD')}"
        if label in exports.values():
            label += f" · {path.name}"
        exports[f"../../Exports/training/{path.name}"] = label
    if current and current not in exports:
        exports[current] = Path(current).name
    minutes = {"1m": 1, "5m": 5, "15m": 15}

    def order(item):
        parts = item[1].split(" ")
        if parts[0] in minutes and len(parts) > 3:
            return (0, minutes[parts[0]], minutes.get(parts[3], 99))
        return (1, 0, 0)
    return dict(sorted(exports.items(), key=order))


def read_history(path):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except ValueError:
            # The trainer may still be writing the last line.
            continue
    return records


def write_state(directory, **data):
    pending = directory / "status.tmp"
    pending.write_text(json.dumps(data), encoding="utf-8")
    pending.replace(directory / "status.json")


def decoded_tail(path):
    try:
        with path.open("rb") as source:
            size = source.seek(0, os.SEEK_END)
            source.seek(max(0, size - 12_000))
            data = source.read()
    except OSError:
        return ""
    # WSL itself may emit UTF-16; Python running inside Ubuntu emits UTF-8.
    if data.startswith(b"\xff\xfe"):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def run_worker(name):
    job = PANEL_LOGS / name
    settings = read_json(job / "settings.json")
    with (job / "console.log").open("wb") as log:
        try:
            write_state(job, phase="starting", pid=os.getpid())
            path_result = subprocess.run(
                [*WSL_SESSION, "--", "wslpath", "-a", (ROOT / "train.sh").as_posix()],
                capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW,
            )
            launcher = path_result.stdout.decode("utf-8").strip()
            command = [*WSL_SESSION, "--", "bash", launcher]
            if settings.get("dataset") == PROFIT_DATASET:
                hold = settings.pop("profit_hold_minutes", DEFAULT_PROFIT_HOLD)
                command += ["--collect-profit", str(hold), f"runs/.panel/{name}/profit_export"]
            command += ["--output-dir", f"runs/{name}"]
            for key, value in settings.items():
                command.extend((f"--{key.replace('_', '-')}", str(value)))
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            write_state(job, phase="running", pid=os.getpid())
            code = child.wait()
            write_state(job, phase="exited", returncode=code)
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            message = str(error)
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                message += "\n" + error.stderr.decode("utf-8", errors="replace").replace("\x00", "")
            log.write(f"{message}\n".encode("utf-8"))
            write_state(job, phase="exited", returncode=1)


def training_control(action, name=None):
    command = [*WSL_SESSION, "--", "python3", "training_control.py", action]
    if name is not None:
        command.append(name)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, encoding="utf-8",
                            errors="replace", timeout=3, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise ValueError(result.stderr.strip() or "Training control failed.")
    return json.loads(result.stdout)


def process_running(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x00100000, False, pid)
    if not handle:
        return False
    try:
        return kernel.WaitForSingleObject(handle, 0) == 258
    finally:
        kernel.CloseHandle(handle)


def percent(value):
    return f"{value:.2%}" if value is not None else "—"


def accuracy_baseline(values):
    if "majority_accuracy" in values:
        return values["majority_accuracy"]
    if "always_skip_accuracy" in values:
        return values["always_skip_accuracy"]
    return values["class_counts"]["0"] / values["count"]


def seconds(value):
    return f"{value:.2f}" if value is not None else "—"


def metrics_text(values, dataset="bitfinex"):
    if not values:
        return "Awaiting first epoch."
    if dataset == PROFIT_DATASET:
        lines = [
            f"Accuracy  {percent(values['accuracy'])}    Majority class  {percent(accuracy_baseline(values))}",
            f"Macro recall  {percent(values['balanced_accuracy'])}    "
            f"Macro average precision  {percent(values.get('average_precision'))}",
        ]
        for name, item in zip(PROFIT_CLASSES, values["per_class"]):
            lines.append(f"{name}   {item['support']:,} actual · {item['calls']:,} predicted · "
                         f"Recall {percent(item['recall'])} · Precision {percent(item['precision'])}")
        return "\n".join(lines)
    matrix = values["confusion_matrix"]
    known = matrix["TP"] + matrix["FN"]
    negative, positive = BENCHMARK_CLASSES.get(dataset, ("skip", "Entry"))
    baseline = f"Always {negative}"
    precision = f"{positive} precision"
    caught = f"{positive} detected" if dataset in BENCHMARK_CLASSES else "Profitable signals caught"
    calls = f"{positive} predictions"
    false = f"False {positive.lower()} predictions" if dataset in BENCHMARK_CLASSES else "False entries"
    return (
        f"Accuracy  {percent(values['accuracy'])}    {baseline}  {percent(accuracy_baseline(values))}\n"
        f"Balanced accuracy  {percent(values['balanced_accuracy'])}    "
        f"{precision}  {percent(values['precision_enter'])}\n"
        f"Average precision  {percent(values.get('average_precision'))}\n"
        f"{caught}  {matrix['TP']:,} / {known:,}  ({percent(values['recall_enter'])})\n"
        f"{calls}  {values['allowed']:,}    {false}  {matrix['FP']:,}"
    )


def comparison_row(label, test):
    row = f"{label:<20} AP {percent(test.get('average_precision')):>7}"
    if quota := test.get("fixed_quota"):
        hits = quota["positives"] / quota["calls"] if quota["calls"] else None
        lift = quota["positives"] / quota["baseline_positives"] if quota["baseline_positives"] else None
        row += f"   top {quota['calls']:,} precision {percent(hits):>7}   lift {lift:.2f}×" if lift else ""
        if "net_return_pct_per_trade" in quota:
            row += f"   net/trade {quota['net_return_pct_per_trade']:+.3f}%"
    if "accuracy" in test:
        row += f"   accuracy {percent(test['accuracy'])}"
    return row


def baseline_comparison(run, config, mlp_test):
    requested = config.get("baselines", "")
    if not requested:
        return "Not requested for this run. Tick models under “Also train and test” before starting."
    report = read_json(run / "baselines.json")
    if report.get("error"):
        return f"Baselines failed: {report['error']}"
    lines = []
    if mlp_test:
        rate = mlp_test.get("positive_rate")
        if rate is None and "class_counts" in mlp_test and len(mlp_test["class_counts"]) == 2:
            rate = mlp_test["class_counts"]["1"] / mlp_test["count"]
        if rate is not None:
            every = mlp_test.get("fixed_quota", {}).get("all_net_return_pct_per_trade")
            net = f"   net/trade {every:+.3f}% taking every test trade" if every is not None else ""
            lines.append(f"{'Random ranking':<20} AP {percent(rate):>7}   (winner rate on test){net}")
        lines.append(comparison_row("MLP (checkpoint)", mlp_test))
    else:
        lines.append("MLP · test pending")
    models = report.get("models", {})
    for key in requested.split(","):
        entry = models.get(key)
        label = BASELINE_LABELS.get(key, key)
        if entry is None:
            lines.append(f"{label:<20} running…")
        elif "skipped" in entry:
            lines.append(f"{label:<20} skipped · {entry['skipped']}")
        else:
            lines.append(comparison_row(label, entry["test"]))
    return "\n".join(lines)


def average_epoch_text(run, history, metadata, complete):
    average = history[-1].get("average_epoch_seconds") if history else None
    if average is None:
        average = metadata.get("timing", {}).get("average_epoch_seconds")
    if average is not None:
        return f"Average epoch  {seconds(average)} s"
    if complete and history:
        try:
            elapsed = (run / "history.jsonl").stat().st_mtime - (run / "run.json").stat().st_mtime
        except OSError:
            elapsed = 0
        if elapsed > 0:
            return f"Average epoch  ≈{seconds(elapsed / len(history))} s · estimated from file times"
    return "Average epoch  —"


def run_report(run, status):
    result = read_json(run / "test_metrics.json")
    metadata = read_json(run / "run.json")
    config = metadata.get("config")
    if not config:
        raise ValueError("The trainer has not saved the run parameters yet.")
    history = read_history(run / "history.jsonl")
    split = read_json(run / "split_report.json")
    has_validation = config.get("validation_fraction", 0.15) > 0
    evaluation_key = "validation" if has_validation else "train_evaluation"
    evaluated_label = "Validation" if has_validation else "Frozen training"
    model = config.get("model", "dense")
    dataset = config.get("dataset", "bitfinex")
    har = dataset == "uci_har"
    units = BENCHMARK_LOOKBACKS.get(dataset, (None, "minutes per entry"))[1]
    baseline = ("majority class" if dataset == PROFIT_DATASET else
                "always " + BENCHMARK_CLASSES.get(dataset, ("skip", "Entry"))[0])
    features = config.get("feature_input", "exported")
    widths = [config.get("hidden1", 128), config.get("hidden2", 64)]
    if config.get("hidden3"):
        widths.append(config["hidden3"])
    if result:
        status = "Complete"
    lines = [
        f"# Bitfinex MLP run {run.name}", "", f"- Status · {status}",
        f"- Completed epochs · {len(history)} of {config['epochs']}",
        f"- Dataset · {DATASET_LABELS.get(dataset, dataset)}",
        f"- Lookback · {config.get('lookback_minutes', STATES):,} {units}",
        f"- Model · {MODEL_LABELS.get(model, model)}",
        f"- Initialization · {OPTION_LABELS['initialization'][config.get('initialization', 'default')]}",
        f"- Feature input · {FEATURE_INPUT_LABELS.get(features, features)}",
        f"- Loss · {LOSS_LABELS[config.get('loss', 'cross_entropy')]}",
        "- Hidden layers · " + " → ".join(map(str, widths)),
        f"- {average_epoch_text(run, history, metadata, bool(result))}",
        "", "## Summary", "",
    ]
    if split.get("data_period"):
        period = split["data_period"]
        lines.insert(4, f"- Signal period · {period['start_inclusive']} to {period['end_exclusive']} (exclusive)")
    if target := metadata.get("profit_target"):
        lines.insert(5, f"- Hold after confirmed short fill · {target['hold_minutes']} minutes")
    if has_validation:
        criterion = metadata.get("selection") or config.get("selection", "accuracy")
        if criterion == "profitable_above_baseline":
            minimum = config.get("min_entry_calls", 30)
            lines.append(f"Checkpoint selection maximizes profitable calls above baseline with at least {minimum} validation calls.")
        else:
            lines.append(f"Validation scores the fixed weights after each epoch. The checkpoint is selected by validation {criterion}.")
    else:
        lines.append("Frozen training metrics score the full training set after each epoch. The final test uses the last checkpoint.")
    dropout = config.get("dropout", 0.0)
    if model != "dense":
        encoder = " → ".join(map(str, (WIDTH, *ENCODER_WIDTHS)))
        lines.append(f"The {encoder} encoder processes each step before the classifier combines the sequence.")
    lines.append(f"Dropout masks {dropout:.0%} of classifier hidden activations during training. "
                 "Frozen evaluation, validation and test use all activations.")
    if history:
        last = history[-1]
        lines.extend([
            "",
            f"- Online training accuracy at epoch {last['epoch']} · {percent(last['train']['accuracy'])}",
        ])
        if has_validation:
            best = next((row for row in reversed(history) if row.get("best")), None)
            if best:
                ev = best["validation"]
                lines.append(f"- Selected epoch {best['epoch']} · validation accuracy {percent(ev['accuracy'])}, "
                             f"loss {ev['loss']:.4f}, average precision {percent(ev.get('average_precision'))}")
                if criterion == "profitable_above_baseline":
                    lines.append(f"- Profitable above baseline · {ev['profitable_above_baseline']:+.2f} on {ev['allowed']:,} calls")
        ev = last[evaluation_key]
        lines.append(f"- {evaluated_label} at epoch {last['epoch']} · accuracy {percent(ev['accuracy'])}, "
                     f"{baseline} {percent(accuracy_baseline(ev))}, loss {ev['loss']:.4f}")
    if result.get("test"):
        test = result["test"]
        tested_epoch = result.get("checkpoint_epoch") or result.get("best_epoch") or len(history)
        lines.append(f"- Final test at epoch {tested_epoch} · accuracy {percent(test['accuracy'])}, "
                     f"{baseline} {percent(accuracy_baseline(test))}, balanced accuracy {percent(test['balanced_accuracy'])}, "
                     f"loss {test['loss']:.4f}")
    else:
        lines.append(f"- {result.get('test_reason', 'Final test metrics are pending.')}")

    if split.get("splits"):
        lines.extend(["", "## Subject split" if har else "## Chronological split", ""])
        for name, part in split["splits"].items():
            if har:
                subjects = ", ".join(map(str, part["subjects"]))
                lines.append(f"- {name.title()} · {part['count']:,} windows, subject IDs {subjects}")
                continue
            first = datetime.fromtimestamp(part["first_signal"], timezone.utc).strftime("%Y-%m-%d %H:%M")
            last = datetime.fromtimestamp(part["last_signal"], timezone.utc).strftime("%Y-%m-%d %H:%M")
            lines.append(f"- {name.title()} · {part['count']:,} entries from {first} to {last} UTC")
            if dataset == PROFIT_DATASET and "class_counts" in part:
                lines.append("  - " + ", ".join(f"{label} {count:,}" for label, count in
                                                zip(PROFIT_CLASSES, part["class_counts"])))
        if not har:
            lines.append(f"- Entries removed at split boundaries · {split['purged_count']:,}")

    if config.get("baselines"):
        lines.extend(["", "## Test comparison with baselines", "", "```",
                      baseline_comparison(run, config, result.get("test")), "```"])
    sections = [
        ("Launch parameters", config),
        ("Runtime and input files", {key: value for key, value in metadata.items() if key != "config"}),
    ]
    if result:
        sections.append(("Final test record", result))
    if baselines := read_json(run / "baselines.json"):
        sections.append(("Baseline records", baselines))
    for title, values in sections:
        lines.extend(["", f"## {title}", "", "```json", json.dumps(values, indent=2, ensure_ascii=False), "```"])
    if history:
        lines.extend(["", "## Epoch history", "", "```jsonl"])
        lines.extend(json.dumps(row, ensure_ascii=False) for row in history)
        lines.append("```")
    return "\n".join(lines) + "\n"


class TrainingPanel:
    def __init__(self, window, dataset="bitfinex", six_months=False):
        self.window = window
        self.name = tk.StringVar()
        self.status = tk.StringVar()
        self.speed = tk.StringVar()
        self.selection = tk.StringVar()
        self.chart_title = tk.StringVar()
        self.chart_legend = tk.StringVar()
        self.checkpoint_title = tk.StringVar()
        self.test_title = tk.StringVar()
        self.latest = tk.StringVar()
        self.test = tk.StringVar()
        self.dataset_description = tk.StringVar()
        self.lookback_label = tk.StringVar(value="Lookback (min)")
        self.fields = {}
        self.history = []
        self.table_key = None
        self.log_key = None
        self.pending_worker = None
        self.trainers = {}
        self.has_validation = False
        self.result_dataset = "bitfinex"
        self.plot_average_precision = False

        window.title("Bitfinex MLP · Training Panel")
        window.geometry("1240x740")
        window.minsize(850, 620)
        style = ttk.Style(window)
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("TButton", padding=(10, 6))

        outer = ttk.Frame(window, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Bitfinex MLP training", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, textvariable=self.dataset_description).pack(anchor="w", pady=(0, 14))
        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        self.build_settings(body)
        self.fields["dataset"].set(DATASET_LABELS[dataset])
        self.apply_dataset_preset()
        if six_months and dataset == "bitfinex":
            self.apply_six_month_preset()

        main = ttk.Frame(body, padding=(14, 0, 0, 0))
        main.pack(side="left", fill="both", expand=True)
        controls = ttk.Frame(main)
        controls.pack(fill="x")
        ttk.Label(controls, text="Run").pack(side="left", padx=(0, 8))
        self.runs = ttk.Combobox(controls, state="readonly", textvariable=self.name)
        self.runs.pack(side="left", fill="x", expand=True)
        self.runs.bind("<<ComboboxSelected>>", lambda _: self.refresh())
        self.open_button = ttk.Button(controls, text="Open folder", command=self.open_folder)
        self.open_button.pack(side="left", padx=(8, 0))
        self.export_button = ttk.Button(controls, text="Export run", command=self.export_run, state="disabled")
        self.export_button.pack(side="left", padx=(8, 0))
        self.pause_button = ttk.Button(controls, text="Force pause", command=self.toggle_training)
        self.pause_button.pack(side="left", padx=(8, 0))
        self.reload_button = ttk.Button(controls, text="Reload panel", command=self.reload_panel)
        self.reload_button.pack(side="left", padx=(8, 0))
        ttk.Label(main, textvariable=self.status, style="Section.TLabel").pack(anchor="w", pady=(12, 5))
        self.progress = ttk.Progressbar(main, mode="determinate")
        self.progress.pack(fill="x")
        ttk.Label(main, textvariable=self.speed).pack(anchor="w", pady=(5, 0))

        tabs = ttk.Notebook(main)
        tabs.pack(fill="both", expand=True, pady=(12, 0))
        self.summary_tab(tabs)
        self.epochs_tab(tabs)
        diagnostics = ttk.Frame(tabs, padding=10)
        tabs.add(diagnostics, text="Optimizer")
        diagnostic_controls = ttk.Frame(diagnostics)
        diagnostic_controls.pack(fill="x")
        self.diagnostic_steps = tk.StringVar(value="100")
        ttk.Label(diagnostic_controls, text="Sample every N steps (0 = off)").pack(side="left")
        ttk.Entry(diagnostic_controls, textvariable=self.diagnostic_steps, width=7).pack(side="left", padx=8)
        ttk.Button(diagnostic_controls, text="Load 10,000-epoch Bitfinex preset",
                   command=self.apply_long_run_preset).pack(side="left", padx=8)
        self.diagnostics = tk.StringVar(value="Select a run to inspect optimizer diagnostics.")
        ttk.Label(diagnostics, textvariable=self.diagnostics, font=("Consolas", 9),
                  justify="left").pack(anchor="nw", pady=(12, 0))
        logs = ttk.Frame(tabs, padding=10)
        tabs.add(logs, text="Console")
        self.console = tk.Text(logs, wrap="none", font=("Consolas", 9), state="disabled")
        scrollbar = ttk.Scrollbar(logs, command=self.console.yview)
        self.console.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.console.pack(side="left", fill="both", expand=True)
        self.poll()

    def build_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="New run", padding=12)
        frame.pack(side="left", anchor="n", fill="y")
        defaults = read_json(ROOT / "train.json")
        self.default_lookback = defaults.get("lookback_minutes", 360)
        OPTION_LABELS["bitfinex_dir"] = bitfinex_exports(defaults.get("bitfinex_dir"))
        CHOICES["bitfinex_dir"] = tuple(OPTION_LABELS["bitfinex_dir"].values())
        for row, (key, title, kind, default) in enumerate(OPTIONS):
            value = defaults.get(key, default)
            if key == "hidden3":
                self.enable_hidden3 = tk.BooleanVar(value=value > 0)
                ttk.Checkbutton(frame, text=title, variable=self.enable_hidden3).grid(
                    row=row, column=0, sticky="w", padx=(0, 8), pady=4,
                )
                self.fields[key] = tk.StringVar(value=str(value or 32))
                third_width = ttk.Entry(frame, textvariable=self.fields[key], width=12)
                third_width.grid(row=row, column=1, sticky="ew")

                def update_hidden3(*_):
                    third_width.configure(state="normal" if self.enable_hidden3.get() else "disabled")

                self.enable_hidden3.trace_add("write", update_hidden3)
                update_hidden3()
                continue
            label = ttk.Label(frame, text=title)
            if key == "lookback_minutes":
                label.configure(textvariable=self.lookback_label)
            label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            if key in OPTION_LABELS:
                value = OPTION_LABELS[key][value]
            field = tk.StringVar(value=str(value))
            self.fields[key] = field
            if key in CHOICES:
                width = {"dataset": 23, "data_period": 18, "model": 18, "feature_input": 18,
                         "selection": 17, "activation": 10, "initialization": 19, "loss": 19,
                         "label_agreement": 19, "bitfinex_dir": 23,
                         "input_timeframe": 21}[key]
                entry = ttk.Combobox(frame, values=CHOICES[key], textvariable=field,
                                     state="readonly", width=width)
            else:
                entry = ttk.Entry(frame, textvariable=field, width=12)
            entry.grid(row=row, column=1, sticky="ew")
            if key == "selection":
                checkpoint_choice = entry
            elif key == "min_entry_calls":
                minimum_entry = entry
            elif key == "model":
                entry.bind("<<ComboboxSelected>>", self.apply_encoder_preset)
            elif key == "dataset":
                entry.bind("<<ComboboxSelected>>", self.apply_dataset_preset)
            elif key == "feature_input":
                self.feature_input_box = entry
            elif key == "bitfinex_dir":
                self.export_box = entry
            elif key == "input_timeframe":
                self.input_timeframe_box = entry
            elif key == "data_period":
                self.data_period_box = entry
                entry.bind("<<ComboboxSelected>>", self.apply_bitfinex_period)
            elif key == "profit_hold_minutes":
                self.profit_hold_fields = (label, entry)

        def update_selection(*_):
            try:
                has_validation = float(self.fields["validation_fraction"].get()) > 0
            except ValueError:
                has_validation = False
            binary_trades = self.fields["dataset"].get() == DATASET_LABELS["bitfinex"]
            checkpoint_choice["values"] = CHOICES["selection"] if binary_trades else CHOICES["selection"][:-1]
            checkpoint_choice.configure(state="readonly" if has_validation else "disabled")
            use_minimum = (has_validation and binary_trades
                           and self.fields["selection"].get() == SELECTION_LABELS["profitable_above_baseline"])
            minimum_entry.configure(state="normal" if use_minimum else "disabled")

        self.fields["validation_fraction"].trace_add("write", update_selection)
        self.fields["selection"].trace_add("write", update_selection)
        self.fields["dataset"].trace_add("write", update_selection)
        update_selection()
        baselines = ttk.LabelFrame(frame, text="Also train and test (CPU, same split)", padding=(8, 4))
        baselines.grid(row=len(OPTIONS), columnspan=2, sticky="ew", pady=(10, 0))
        self.baseline_vars, self.baseline_boxes = {}, []
        for column, (key, label) in enumerate(BASELINE_LABELS.items()):
            self.baseline_vars[key] = tk.BooleanVar(value=False)
            box = ttk.Checkbutton(baselines, text=label, variable=self.baseline_vars[key])
            box.grid(row=column // 2, column=column % 2, sticky="w", padx=(0, 8))
            self.baseline_boxes.append(box)
        self.launch = ttk.Button(frame, text="Start training", command=self.start)
        self.launch.grid(row=len(OPTIONS) + 1, columnspan=2, sticky="ew", pady=(14, 4))
        ttk.Label(frame, text="Validation fraction 0 saves the final epoch for test.", wraplength=195).grid(
            row=len(OPTIONS) + 2, columnspan=2, sticky="w"
        )

    def apply_encoder_preset(self, _event):
        if self.fields["model"].get() == MODEL_LABELS["dense"]:
            return
        for key, value in ENCODER_PRESET.items():
            self.fields[key].set(str(value))
        dataset = self.selected_dataset()
        if dataset in BENCHMARK_LOOKBACKS:
            self.fields["lookback_minutes"].set(str(BENCHMARK_LOOKBACKS[dataset][0]))
        self.enable_hidden3.set(False)

    def selected_dataset(self):
        label = self.fields["dataset"].get()
        return next(key for key, value in DATASET_LABELS.items() if value == label)

    def apply_long_run_preset(self):
        self.fields["dataset"].set(DATASET_LABELS["bitfinex"])
        self.apply_dataset_preset()
        preset = {
            "data_period": DATA_PERIOD_LABELS["all"], "model": MODEL_LABELS["dense"],
            "feature_input": FEATURE_INPUT_LABELS["exported"], "loss": LOSS_LABELS["cross_entropy"],
            "initialization": OPTION_LABELS["initialization"]["default"],
            "epochs": 10000, "batch_size": 2048, "learning_rate": 1e-5,
            "selection": SELECTION_LABELS["average_precision"], "hidden1": 32, "hidden2": 16,
            "lookback_minutes": 1410, "activation": "relu", "dropout": 0.4,
            "train_fraction": 0.7, "validation_fraction": 0.15, "seed": 42,
        }
        for field, value in preset.items():
            self.fields[field].set(str(value))
        self.enable_hidden3.set(False)
        self.diagnostic_steps.set("100")
        self.apply_bitfinex_period()

    def apply_bitfinex_period(self, _event=None):
        period = self.fields["data_period"].get()
        market = "ETHUSD · Timed short exits" if self.selected_dataset() == PROFIT_DATASET else "ETHUSD"
        self.dataset_description.set(f"{market} · {period} · Radeon RX 9070")

    def apply_six_month_preset(self):
        preset = {
            "data_period": DATA_PERIOD_LABELS["last_6_months"],
            "model": MODEL_LABELS["per_minute"], "feature_input": FEATURE_INPUT_LABELS["exported"],
            "activation": "relu", "dropout": 0.2, "epochs": 40, "batch_size": 4096,
            "learning_rate": 0.0001, "train_fraction": 0.7, "validation_fraction": 0.15,
            "selection": SELECTION_LABELS["average_precision"], "seed": 42,
        }
        for key, value in preset.items():
            self.fields[key].set(str(value))
        self.apply_encoder_preset(None)
        self.apply_bitfinex_period()

    def apply_dataset_preset(self, _event=None):
        dataset = self.selected_dataset()
        title = "Profit buckets" if dataset == PROFIT_DATASET else "Training Panel"
        self.window.title(f"Bitfinex MLP · {title}")
        for widget in self.profit_hold_fields:
            if dataset == PROFIT_DATASET:
                widget.grid()
            else:
                widget.grid_remove()
        benchmark = dataset in BENCHMARK_LOOKBACKS
        if benchmark:
            lookback, unit = BENCHMARK_LOOKBACKS[dataset]
            self.lookback_label.set(f"Lookback ({unit})")
            self.dataset_description.set(BENCHMARK_DESCRIPTIONS[dataset])
            self.fields["data_period"].set(DATA_PERIOD_LABELS["all"])
            self.fields["lookback_minutes"].set(str(lookback))
            self.fields["feature_input"].set(FEATURE_INPUT_LABELS["exported"])
            criterion = "accuracy" if dataset == "uci_har" else "average_precision"
            self.fields["selection"].set(SELECTION_LABELS[criterion])
        else:
            self.lookback_label.set("Lookback (min)")
            self.apply_bitfinex_period()
            lookback = self.default_lookback
            if self.fields["model"].get() != MODEL_LABELS["dense"]:
                lookback = ENCODER_PRESET["lookback_minutes"]
            self.fields["lookback_minutes"].set(str(lookback))
            if dataset == PROFIT_DATASET and self.fields["selection"].get() == SELECTION_LABELS["profitable_above_baseline"]:
                self.fields["selection"].set(SELECTION_LABELS["average_precision"])
        self.feature_input_box.configure(state="disabled" if benchmark else "readonly")
        self.data_period_box.configure(state="disabled" if benchmark else "readonly")
        # Profit buckets collect from train.json's export; only the binary dataset trains on the chosen one.
        self.export_box.configure(state="readonly" if dataset == "bitfinex" else "disabled")
        # Only the binary export has every entry on an M5 close; profit fills and benchmarks use 1m rows.
        if dataset != "bitfinex":
            self.fields["input_timeframe"].set(INPUT_TIMEFRAME_LABELS["1m"])
        self.input_timeframe_box.configure(state="readonly" if dataset == "bitfinex" else "disabled")
        for box in getattr(self, "baseline_boxes", []):
            box.configure(state="disabled" if benchmark else "normal")

    def summary_tab(self, tabs):
        summary = ttk.Frame(tabs, padding=12)
        tabs.add(summary, text="Results")
        ttk.Label(summary, textvariable=self.chart_title, style="Section.TLabel").pack(anchor="w")
        ttk.Label(summary, textvariable=self.chart_legend).pack(anchor="w")
        self.chart = tk.Canvas(summary, height=145, background="white", highlightthickness=0)
        self.chart.pack(fill="x", pady=(6, 10))
        self.chart.bind("<Configure>", lambda _: self.draw_chart())
        for title, content in ((self.checkpoint_title, self.latest), (self.test_title, self.test)):
            ttk.Label(summary, textvariable=title, style="Section.TLabel").pack(anchor="w")
            ttk.Label(summary, textvariable=content, justify="left").pack(anchor="w", pady=(2, 12))
        self.baseline_text = tk.StringVar()
        ttk.Label(summary, text="Test comparison with baselines", style="Section.TLabel").pack(anchor="w")
        ttk.Label(summary, textvariable=self.baseline_text, justify="left",
                  font=("Consolas", 9)).pack(anchor="w", pady=(2, 0))

    def epochs_tab(self, tabs):
        epochs = ttk.Frame(tabs, padding=10)
        tabs.add(epochs, text="Epochs")
        toolbar = ttk.Frame(epochs)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(toolbar, textvariable=self.selection, style="Section.TLabel").pack(side="left")
        ttk.Button(toolbar, text="Metric guide", command=self.show_epoch_guide).pack(side="right")
        grid = ttk.Frame(epochs)
        grid.pack(fill="both", expand=True)
        grid.columnconfigure(0, weight=1)
        grid.rowconfigure(0, weight=1)
        columns = (
            ("epoch", "Epoch", 52), ("seconds", "Epoch (s)", 70), ("average", "Avg. (s)", 70),
            ("train", "Train acc.", 101), ("evaluation", "Val. acc.", 106), ("baseline", "Always skip", 82),
            ("val_loss", "Val. loss", 70), ("val_recall", "Val. recall", 77), ("precision", "Entry precision", 99),
            ("average_precision", "Avg. precision", 99),
            ("balanced", "Balanced acc.", 96), ("caught", "Profitable caught", 112),
            ("calls", "Entry calls", 75), ("false", "False entries", 86),
            ("above_baseline", "Above baseline", 126), ("best", "New best", 70),
        )
        self.table = ttk.Treeview(grid, columns=[column[0] for column in columns], show="headings")
        for col, title, width in columns:
            self.table.heading(col, text=title)
            self.table.column(col, width=width, minwidth=width, stretch=False, anchor="center")
        vertical = ttk.Scrollbar(grid, command=self.table.yview)
        horizontal = ttk.Scrollbar(grid, orient="horizontal", command=self.table.xview)
        self.table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.table.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        ttk.Label(epochs, textvariable=self.test_title, style="Section.TLabel").pack(anchor="w", pady=(8, 2))
        ttk.Label(epochs, textvariable=self.test, justify="left").pack(anchor="w")

    def show_epoch_guide(self):
        messagebox.showinfo(
            "Epoch metrics",
            "Dropout is the fraction of hidden activations randomly masked during training. "
            "0.2 masks 20% after each hidden activation, with a fresh random mask on every forward pass. "
            "The remaining activations are scaled to keep their expected magnitude. Dropout is disabled "
            "for frozen training evaluation, validation and test. It can slow memorization, and may improve "
            "validation results. 0 disables it.\n\n"
            "Online training accuracy uses predictions made during weight updates. Frozen training accuracy "
            "scores the full training set using the weights at the end of the epoch. When validation is enabled, "
            "the fixed weights score the validation set. Always skip is the share of evaluated signals with target 0.\n\n"
            "For UCI HAR, target 0 is downstairs and target 1 is upstairs. The train and validation fractions "
            "divide the official training subjects. The official test subjects are held apart.\n\n"
            "For Beijing PM2.5, target 1 means pollution is higher at the reading six hours ahead. "
            "Lookback counts hourly samples, and the split follows time order.\n\n"
            "For Bitfinex profit buckets, a confirmed short fill starts one timed trade. The final return includes "
            "both fees and falls into one of five classes. The majority baseline is the largest class's share. "
            "Macro recall and macro average precision give equal weight to each class observed in that split.\n\n"
            "Recall is the share of profitable signals caught. Entry precision is the share of entry "
            "calls that were profitable. False entries are entry calls with target 0.\n\n"
            "Average precision measures how well the model ranks winners across all thresholds. "
            "It can select an epoch with zero calls at the default threshold. "
            "The baseline for a tied ranking is the share of winners in the split.\n\n"
            "Above baseline is profitable calls minus (entry calls × validation positive rate). "
            "With this selection criterion, an epoch must meet Min. entry calls to qualify. "
            "Higher scores win, with lower validation loss breaking ties.\n\n"
            "Balanced accuracy averages the share of profitable signals entered and the share of nonprofitable "
            "signals skipped. Always skipping gives (0% + 100%) / 2 = 50%. Loss measures the quality "
            "of the probability scores and penalizes confident mistakes. Lower loss is better.\n\n"
            "The final test runs once, using the last checkpoint for a train/test split or the best validation "
            "checkpoint when validation is enabled.\n\n"
            "Epoch (s) measures training, evaluation, and checkpoint saves. Avg. (s) averages all epochs completed "
            "so far in the run.",
            parent=self.window,
        )

    def settings(self):
        values = {}
        try:
            values["diagnostic_steps"] = int(self.diagnostic_steps.get())
        except ValueError:
            raise ValueError("Enter a whole number for optimizer diagnostic steps.") from None
        if values["diagnostic_steps"] < 0:
            raise ValueError("Optimizer diagnostic steps must be 0 or greater.")
        for key, title, kind, _ in OPTIONS:
            if key == "profit_hold_minutes" and self.selected_dataset() != PROFIT_DATASET:
                continue
            if key == "hidden3" and not self.enable_hidden3.get():
                values[key] = 0
                continue
            try:
                values[key] = kind(self.fields[key].get())
            except ValueError:
                raise ValueError(f"Enter a valid value for {title.lower()}.") from None
        if self.enable_hidden3.get() and values["hidden3"] <= 0:
            raise ValueError("Third hidden layer width must be positive when enabled.")
        for key in ("epochs", "batch_size", "hidden1", "hidden2", "learning_rate", "min_entry_calls"):
            if not math.isfinite(values[key]) or values[key] <= 0:
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be positive.")
        window_input_size(values["lookback_minutes"])
        if not math.isfinite(values["dropout"]) or not 0 <= values["dropout"] < 1:
            raise ValueError("Dropout must be a fraction from 0 up to but excluding 1, such as 0.2.")
        train, validation = values["train_fraction"], values["validation_fraction"]
        if not (0 < train < 1 and 0 <= validation < 1 and train + validation < 1):
            raise ValueError("Use a positive train fraction, validation of 0 or more, and a sum below 1.")
        if not 0 <= values["seed"] < 2**32:
            raise ValueError("Seed must be between 0 and 4,294,967,295.")
        for key, labels in OPTION_LABELS.items():
            chosen = next((value for value, label in labels.items() if label == values[key]), None)
            if chosen is None:
                raise ValueError(f"Choose a listed value for {key.replace('_', ' ')}.")
            values[key] = chosen
        dataset = values["dataset"]
        if dataset != "bitfinex":
            values.pop("bitfinex_dir")
        if values["input_timeframe"] == "5m" and values["lookback_minutes"] % 5:
            raise ValueError("With 5m input candles, use a lookback that is a multiple of 5 minutes.")
        selected = [key for key, chosen in self.baseline_vars.items() if chosen.get()]
        if selected and dataset not in BENCHMARK_LOOKBACKS:
            values["baselines"] = ",".join(selected)
        if dataset == PROFIT_DATASET and values["profit_hold_minutes"] <= 0:
            raise ValueError("Hold after fill must be at least one minute.")
        if dataset in BENCHMARK_LOOKBACKS:
            limit, unit = BENCHMARK_LOOKBACKS[dataset]
            if values["lookback_minutes"] > limit:
                raise ValueError(f"{DATASET_LABELS[dataset]} has at most {limit} {unit} per window.")
        return values

    def start(self):
        self.check_trainers()
        if self.refresh():
            messagebox.showinfo("Training in progress", "Pause the active run or wait for it to finish.",
                                parent=self.window)
            return
        try:
            values = self.settings()
            name = f"rx9070-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:4]}"
            job = PANEL_LOGS / name
            job.mkdir(parents=True)
            (job / "settings.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
            worker = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--worker", name],
                creationflags=subprocess.CREATE_NO_WINDOW,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            self.pending_worker = (name, worker)
            self.name.set(name)
            self.refresh()
        except (OSError, ValueError, OverflowError) as error:
            messagebox.showerror("Could not start training", str(error), parent=self.window)

    def available_runs(self):
        existing = {
            item.name for item in RUNS.glob("*")
            if item.is_dir() and ((item / "history.jsonl").exists() or (item / "run.json").exists())
        }
        existing.update(item.name for item in PANEL_LOGS.glob("*") if item.is_dir())
        if (RUNS / "first-rx9070-wsl.pid").exists():
            existing.add(LEGACY)
        return sorted(existing, key=lambda name: max(
            (folder.stat().st_mtime for folder in (RUNS / name, PANEL_LOGS / name) if folder.exists()),
            default=0,
        ), reverse=True)

    def running(self, name, state):
        if state.get("phase") in ("starting", "running"):
            return process_running(state.get("pid"))
        if name == LEGACY and not (RUNS / name / "test_metrics.json").exists():
            try:
                pid = int((RUNS / "first-rx9070-wsl.pid").read_text(encoding="ascii"))
            except (OSError, ValueError):
                return False
            return process_running(pid)
        return False

    def open_folder(self):
        folder = RUNS / self.name.get()
        if not folder.is_dir():
            folder = PANEL_LOGS / self.name.get()
        if folder.is_dir():
            os.startfile(folder)

    def export_run(self):
        name = self.name.get()
        try:
            report = run_report(RUNS / name, self.status.get())
            destination = filedialog.asksaveasfilename(
                parent=self.window, title="Export run report", initialfile=f"{name}.md",
                defaultextension=".md", filetypes=[("Markdown report", "*.md")],
            )
            if destination:
                path = Path(destination)
                if path.suffix.lower() != ".md":
                    raise ValueError("Choose a .md filename for the run report.")
                path.write_text(report, encoding="utf-8")
                messagebox.showinfo("Run exported", f"Saved {path.name}", parent=self.window)
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not export run", str(error), parent=self.window)

    def check_trainers(self):
        try:
            self.trainers = training_control("status")
        except (OSError, ValueError, subprocess.SubprocessError):
            self.trainers = {}

    def toggle_training(self):
        name = self.name.get()
        try:
            self.trainers.update(training_control("resume" if self.trainers.get(name) else "pause", name))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            messagebox.showerror("Training control failed", str(error), parent=self.window)
            self.check_trainers()
        self.refresh()

    def refresh(self):
        names = self.available_runs()
        self.runs["values"] = names
        if self.name.get() not in names:
            self.name.set(names[0] if names else "")
        states = {name: read_json(PANEL_LOGS / name / "status.json") for name in names}
        active = {name for name, state in states.items() if self.running(name, state)} | self.trainers.keys()
        paused = {name for name, is_paused in self.trainers.items() if is_paused}
        active -= paused
        pending_name = None
        if self.pending_worker:
            name, worker = self.pending_worker
            if worker.poll() is None and name not in paused:
                pending_name = name
        self.launch.configure(state="disabled" if active or pending_name else "normal")
        self.open_button.configure(state="normal" if names else "disabled")
        name = self.name.get()
        can_control = name in self.trainers and (name not in paused or not (active or pending_name))
        self.pause_button.configure(text="Resume" if name in paused else "Force pause",
                                    state="normal" if can_control else "disabled")
        if not name:
            self.export_button.configure(state="disabled")
            self.status.set("Choose settings and start a run.")
            self.speed.set("Average epoch  —")
            self.selection.set("")
            self.diagnostics.set("Select a run to inspect optimizer diagnostics.")
            return bool(active or pending_name)
        run = RUNS / name
        job = PANEL_LOGS / name
        state = states[name]
        self.history = read_history(run / "history.jsonl")
        metadata = read_json(run / "run.json")
        self.export_button.configure(state="normal" if metadata.get("config") else "disabled")
        config = metadata.get("config", {}) or read_json(job / "settings.json")
        self.result_dataset = config.get("dataset", "bitfinex")
        self.has_validation = config.get("validation_fraction", 0.15) > 0
        total = config.get("epochs", 20)
        epoch = self.history[-1]["epoch"] if self.history else 0
        result = read_json(run / "test_metrics.json")
        self.progress.configure(maximum=total, value=epoch)
        if result:
            status = f"Complete · {epoch} of {total} epochs"
        elif name in paused:
            status = f"Paused · {epoch} of {total} epochs"
        elif name in active or name == pending_name:
            starting = ("Starting · preparing profit buckets and GPU" if self.result_dataset == PROFIT_DATASET
                        else "Starting · checking GPU and data")
            status = f"Training · {epoch} of {total} epochs" if epoch else starting
            if epoch >= total:
                status = "Evaluating the test set"
        elif state.get("returncode"):
            status = f"Stopped after {epoch} of {total} epochs · see Console"
        elif state.get("phase") in ("starting", "running"):
            # The worker died without recording an exit (killed, logoff, reboot).
            status = f"Interrupted after {epoch} of {total} epochs · trainer is no longer running"
        else:
            status = f"{epoch} of {total} epochs recorded · final test pending"
        self.status.set(status)
        self.speed.set(average_epoch_text(run, self.history, metadata, bool(result)))
        activation = {"relu": "ReLU", "leaky_relu": "LeakyReLU"}.get(config.get("activation", "relu"))
        features = config.get("feature_input", "exported")
        feature_label = FEATURE_INPUT_LABELS.get(features, features)
        dropout_label = f" · Dropout {config.get('dropout', 0):.0%}"
        criterion = config.get("selection", "accuracy")
        self.plot_average_precision = self.has_validation and criterion == "average_precision"
        if self.result_dataset in BENCHMARK_CLASSES:
            negative, positive = BENCHMARK_CLASSES[self.result_dataset]
            baseline_label = f"{positive} rate" if self.plot_average_precision else f"Always {negative}"
        elif self.result_dataset == PROFIT_DATASET:
            baseline_label = "Macro AP baseline" if self.plot_average_precision else "Majority class"
        else:
            baseline_label = "Winner rate" if self.plot_average_precision else "Always skip"
        self.chart_legend.set(f"Blue · Model     Gray dashed · {baseline_label}")
        if self.has_validation:
            shown = next((row for row in reversed(self.history) if row["best"]), None)
            best_label = f" · Best epoch {shown['epoch']}" if shown else ""
            detail = f"Keep best by validation {SELECTION_LABELS.get(criterion, criterion).lower()}{best_label}"
            metric = "average precision" if self.plot_average_precision else "accuracy"
            if self.plot_average_precision and self.result_dataset == PROFIT_DATASET:
                metric = "macro average precision"
            self.chart_title.set(f"Validation {metric} by epoch")
            self.checkpoint_title.set("Selected checkpoint")
            self.test_title.set("Final test · selected checkpoint")
        else:
            shown = self.history[-1] if self.history else None
            train_fraction = config.get("train_fraction", 0.8)
            detail = ("Official test subjects" if self.result_dataset == "uci_har"
                      else f"{train_fraction:.0%} train / {1-train_fraction:.0%} test")
            if config.get("feature_scaling") == "standardize":
                detail += " · Standardized"
            self.chart_title.set("Frozen training accuracy by epoch")
            self.checkpoint_title.set("Latest training evaluation")
            self.test_title.set("Final test · last checkpoint")
        model = config.get("model", "dense")
        input_label = DATASET_LABELS[self.result_dataset] if self.result_dataset in BENCHMARK_CLASSES else feature_label
        if self.result_dataset == PROFIT_DATASET:
            hold = metadata.get("profit_target", {}).get("hold_minutes", config.get("profit_hold_minutes", DEFAULT_PROFIT_HOLD))
            input_label = f"Profit buckets · {hold} min · {feature_label}"
        if config.get("data_period") == "last_6_months":
            input_label = f"{input_label} · Last 6 months"
        loss_label = LOSS_LABELS[config.get("loss", "cross_entropy")]
        self.selection.set(
            f"{MODEL_LABELS.get(model, model)} · {activation}{dropout_label} · {input_label} · {loss_label} · {detail}"
        )
        if shown:
            evaluated = shown[self.evaluation_key]
            caption = f"Epoch {shown['epoch']} · Loss {evaluated['loss']:.4f}"
            if shown["selection"] == "profitable_above_baseline":
                caption += f" · Above baseline {evaluated['profitable_above_baseline']:+.2f}"
            self.latest.set(f"{caption}\n{metrics_text(evaluated, self.result_dataset)}")
        elif self.history and config.get("selection") == "profitable_above_baseline":
            minimum = config.get("min_entry_calls", 30)
            message = f"An epoch needs at least {minimum} validation calls to qualify."
            self.latest.set(result.get("test_reason", message))
        else:
            self.latest.set("Awaiting first epoch.")
        final_test = result.get("test")
        self.test.set(metrics_text(final_test, self.result_dataset) if final_test else result.get("test_reason", "Available after training finishes."))
        self.baseline_text.set(baseline_comparison(run, config, final_test))
        self.update_diagnostics(config, final_test)
        self.draw_chart()
        self.update_table(name)
        self.update_console(name, job)
        return bool(active or pending_name)

    def form_state(self):
        return {"fields": {key: field.get() for key, field in self.fields.items()},
                "hidden3": self.enable_hidden3.get(),
                "baselines": {key: chosen.get() for key, chosen in self.baseline_vars.items()},
                "diagnostic_steps": self.diagnostic_steps.get(), "run": self.name.get(),
                "geometry": self.window.geometry()}

    def restore_form(self, state):
        fields = state.get("fields", {})
        if fields.get("dataset") in DATASET_LABELS.values():
            self.fields["dataset"].set(fields["dataset"])
            self.apply_dataset_preset()
        for key, value in fields.items():
            if key in self.fields and key != "dataset":
                self.fields[key].set(value)
        self.enable_hidden3.set(bool(state.get("hidden3")))
        for key, chosen in state.get("baselines", {}).items():
            if key in self.baseline_vars:
                self.baseline_vars[key].set(bool(chosen))
        self.diagnostic_steps.set(state.get("diagnostic_steps", self.diagnostic_steps.get()))
        if state.get("geometry"):
            self.window.geometry(state["geometry"])
        if state.get("run"):
            self.name.set(state["run"])
        self.apply_bitfinex_period()
        self.refresh()

    def reload_panel(self):
        """Restart the panel on the current code, keeping the form and selected run.

        Trainers run in their own processes and keep running; only this window restarts.
        """
        PANEL_LOGS.mkdir(parents=True, exist_ok=True)
        state_file, log_file = PANEL_LOGS / "panel-reload.json", PANEL_LOGS / "panel-reload.log"
        state_file.write_text(json.dumps(self.form_state(), indent=2), encoding="utf-8")
        self.reload_button.configure(state="disabled", text="Reloading…")
        with log_file.open("wb") as log:
            child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--restore", str(state_file)],
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                     creationflags=subprocess.CREATE_NO_WINDOW)

        def finish():
            if child.poll() is None:  # the new panel is up; hand over to it
                self.window.destroy()
                return
            self.reload_button.configure(state="normal", text="Reload panel")
            messagebox.showerror("Could not reload the panel",
                                 "The updated panel failed to start, so this one stays open.\n\n"
                                 + decoded_tail(log_file)[-2000:], parent=self.window)

        self.window.after(2500, finish)

    def poll(self):
        try:
            self.check_trainers()
            self.refresh()
        finally:
            self.window.after(1800, self.poll)

    def update_diagnostics(self, config, final_test):
        interval = config.get("diagnostic_steps", 0)
        if not interval:
            self.diagnostics.set("Optimizer diagnostics were off for this run.")
            return
        lines = [f"Saved diagnostics every {interval} optimizer steps"]
        if self.history:
            latest = self.history[-1][self.evaluation_key]
            if "average_precision" in latest:
                lines.append(f"Latest {'validation' if self.has_validation else 'training'} average precision"
                             f" · {latest['average_precision']:.2%}")
            for label, values in (("Validation", latest), ("Test", final_test or {})):
                if quota := values.get("fixed_quota"):
                    lines.append(f"{label} top {quota['calls']:,} · {quota['positives']:,.1f} winners"
                                 f" · {quota['baseline_positives']:,.1f} expected at base rate")
        fixed = [row for row in self.history if "fixed_train" in row]
        if fixed:
            lines.append("")
            lines.append(f"Fixed training sample · {fixed[-1]['fixed_train']['count']:,} entries")
            for row in fixed[:1] + fixed[max(1, len(fixed) - 4):]:
                evaluation = row[self.evaluation_key]
                line = f"Epoch {row['epoch']:>6,}  loss {row['fixed_train']['loss']:.6g}"
                if "average_precision" in evaluation:
                    line += f"  AP {evaluation['average_precision']:.2%}"
                if quota := evaluation.get("fixed_quota"):
                    line += f"  top {quota['calls']:,} winners {quota['positives']:,.1f}"
                lines.append(line)
        sample = next((row["train"]["optimizer_samples"][-1] for row in reversed(self.history)
                       if row["train"].get("optimizer_samples")), None)
        if sample:
            lines.extend(("", f"Optimizer step {sample['step']:,}",
                          f"{'Layer weights':<22}{'Gradient norm':>15}{'Update norm':>15}{'Update / weight':>17}"))
            for name, norms in sample["weights"].items():
                relative = norms["relative_update"]
                relative_text = f"{relative:.3g}" if relative is not None else "—"
                lines.append(f"{name:<22}{norms['gradient_norm']:>15.3g}{norms['update_norm']:>15.3g}"
                             f"{relative_text:>17}")
        self.diagnostics.set("\n".join(lines))

    @property
    def evaluation_key(self):
        return "validation" if self.has_validation else "train_evaluation"

    def update_table(self, name):
        key = (name, len(self.history), self.has_validation, self.result_dataset)
        if self.table_key == key:
            return
        self.table_key = key
        self.table.heading("train", text="Train acc." if self.has_validation else "Online train acc.")
        self.table.heading("evaluation", text="Val. acc." if self.has_validation else "Frozen train acc.")
        self.table.heading("val_loss", text="Val. loss" if self.has_validation else "Train loss")
        self.table.heading("val_recall", text="Val. recall" if self.has_validation else "Train recall")
        benchmark = self.result_dataset in BENCHMARK_CLASSES
        negative, positive = BENCHMARK_CLASSES.get(self.result_dataset, ("skip", "Entry"))
        self.table.heading("baseline", text=f"Always {negative}")
        self.table.heading("precision", text=f"{positive} precision")
        self.table.heading("caught", text=f"{positive} detected" if benchmark else "Profitable caught")
        self.table.heading("calls", text=f"{positive} calls")
        self.table.heading("false", text=f"False {positive.lower()}" if benchmark else "False entries")
        profit = self.result_dataset == PROFIT_DATASET
        self.table.heading("average_precision", text="Macro avg. precision" if profit else "Avg. precision")
        self.table.heading("balanced", text="Macro recall" if profit else "Balanced acc.")
        columns = self.table["columns"]
        if profit:
            self.table.heading("baseline", text="Majority class")
            hidden = {"val_recall", "precision", "caught", "calls", "false", "above_baseline"}
            if not self.has_validation:
                hidden.add("best")
            columns = tuple(column for column in columns if column not in hidden)
        elif benchmark:
            columns = tuple(column for column in columns if column != "above_baseline")
        elif not self.has_validation:
            columns = columns[:-2]
        self.table["displaycolumns"] = columns
        self.table.delete(*self.table.get_children())
        for row in self.history:
            val = row[self.evaluation_key]
            first = (row["epoch"], seconds(row.get("epoch_seconds")), seconds(row.get("average_epoch_seconds")),
                     percent(row["train"]["accuracy"]), percent(val["accuracy"]), percent(accuracy_baseline(val)),
                     f"{val['loss']:.3f}")
            if profit:
                values = (*first, "", "", percent(val.get("average_precision")),
                          percent(val["balanced_accuracy"]), "", "", "", "", "Yes" if row["best"] else "")
                self.table.insert("", "end", values=values)
                continue
            matrix = val["confusion_matrix"]
            score = val.get("profitable_above_baseline")
            above_baseline = "—" if score is None else f"{score:+.2f}"
            if row.get("selection") == "profitable_above_baseline" and not row["selection_eligible"]:
                above_baseline += " (too few calls)"
            self.table.insert("", "end", values=(
                *first, percent(val["recall_enter"]), percent(val["precision_enter"]),
                percent(val.get("average_precision")),
                percent(val["balanced_accuracy"]), f"{matrix['TP']:,} / {(matrix['TP'] + matrix['FN']):,}",
                f"{val['allowed']:,}", f"{matrix['FP']:,}", above_baseline, "Yes" if row["best"] else "",
            ))

    def update_console(self, name, job):
        log = decoded_tail(job / "console.log")
        if name == LEGACY:
            log = "\n".join(decoded_tail(RUNS / file) for file in (
                "first-rx9070-console.log", "first-rx9070-errors.log"
            )).strip()
        key = (name, log)
        if self.log_key == key:
            return
        follow = self.log_key is None or self.log_key[0] != name or self.console.yview()[1] >= 0.98
        self.log_key = key
        self.console.configure(state="normal")
        self.console.delete("1.0", "end")
        self.console.insert("end", log or "Waiting for trainer output.")
        self.console.configure(state="disabled")
        if follow:
            self.console.see("end")

    def draw_chart(self):
        chart = self.chart
        chart.delete("all")
        width, height = chart.winfo_width(), chart.winfo_height()
        if not self.history or width < 100 or height < 50:
            return
        metrics = [row[self.evaluation_key] for row in self.history]
        if self.plot_average_precision:
            values = [row["average_precision"] for row in metrics]
            baseline = metrics[0].get("average_precision_baseline", 1 - accuracy_baseline(metrics[0]))
        else:
            values = [row["accuracy"] for row in metrics]
            baseline = accuracy_baseline(metrics[0])
        low = max(0, min(baseline, *values) - 0.02)
        high = min(1, max(baseline, *values) + 0.02)
        left, right, top, bottom = 49, width - 18, 10, height - 26

        def y(value):
            return bottom - (value - low) / (high - low) * (bottom - top)

        for value in (low, (low + high) / 2, high):
            chart.create_line(left, y(value), right, y(value), fill="#e0e4ea")
            chart.create_text(left - 6, y(value), text=f"{value:.0%}", anchor="e", fill="#4b5565")
        chart.create_line(left, y(baseline), right, y(baseline), fill="#687480", dash=(5, 4))
        span = max(1, self.history[-1]["epoch"] - 1)
        previous = None
        for row, value in zip(self.history, values):
            x = left + (row["epoch"] - 1) * (right - left) / span
            point = (x, y(value))
            if previous:
                chart.create_line(*previous, *point, fill="#255aa7", width=2)
            chart.create_oval(x - 3, point[1] - 3, x + 3, point[1] + 3, fill="#255aa7", outline="")
            previous = point
        chart.create_text(left, bottom + 13, text="Epoch 1", anchor="w", fill="#4b5565")
        if len(self.history) > 1:
            chart.create_text(right, bottom + 13, text=f"Epoch {self.history[-1]['epoch']}",
                              anchor="e", fill="#4b5565")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2])
    else:
        window = tk.Tk()
        dataset = next((key for key in DATASET_LABELS if f"--{key.replace('_', '-')}" in sys.argv), "bitfinex")
        panel = TrainingPanel(window, dataset=dataset, six_months="--bitfinex-six-months" in sys.argv)
        if "--bitfinex-long-run" in sys.argv:
            panel.apply_long_run_preset()
        if "--restore" in sys.argv[:-1]:
            panel.restore_form(read_json(Path(sys.argv[sys.argv.index("--restore") + 1])))
        window.mainloop()
