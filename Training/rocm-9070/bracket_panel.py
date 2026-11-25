"""Bracket study panel: steps 3 and 5 of the bracket plan in one window.

Pick a bracket export pair (built by Pipelines/Bitfinex/bracket_targets.py: every 5m candle labelled
"take profit before stop", plus a twin with the anchor inputs zeroed) and the MLP settings. One study then runs,
in order and unattended:

1. train the MLP on the twin without anchors (step 3),
2. score every candle with its best checkpoint,
3. train the same MLP on the export with anchors (step 5),
4. score every candle with it,
5. Models/Baselines/bracket_ranking.py on the test period: both MLPs, logistic regression and boosted trees
   against a random ranking, and anchors on against anchors off.

Training and scoring run in WSL on the RX 9070 through train.sh, at lower CPU priority with 2 PyTorch
threads. The ranking runs on Windows at below-normal priority with 2 BLAS threads, so other jobs keep running.
A study lives in runs/.bracket/<study>; its two training runs are ordinary runs/<study>-noanchors and
runs/<study>-anchors folders, which Training Panel.cmd lists too. A stopped study resumes at its first
unfinished step.
"""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tkinter as tk
from datetime import datetime
from tkinter import messagebox, ttk

from training_panel import (
    EXPORTS, ROOT, RUNS, WSL_SESSION, decoded_tail, process_running, read_history, read_json,
    training_control, write_state,
)

STUDIES = RUNS / ".bracket"
REPO = ROOT.parents[1]
RANKING = REPO / "Models" / "Baselines" / "bracket_ranking.py"
SITE_PACKAGES = REPO / ".venv-shared-lstm" / "Lib" / "site-packages"
CPU_THREADS = "2"
INPUTS = {"compact": "85 decision features", "window": "1m candle window"}
DEFAULTS = {
    "model_input": "compact", "lookback_minutes": 360, "hidden1": 32, "hidden2": 16, "dropout": 0.2,
    "epochs": 100, "batch_size": 2048, "learning_rate": 0.0003, "seed": 42,
}
FIELDS = (
    ("hidden1", "First hidden layer", int), ("hidden2", "Second hidden layer", int),
    ("dropout", "Dropout (0 to <1)", float), ("epochs", "Epochs", int), ("batch_size", "Batch size", int),
    ("learning_rate", "Learning rate", float), ("lookback_minutes", "Lookback (min, window only)", int),
    ("seed", "Random seed", int),
)
# (key, title, run suffix or None)
STEPS = (
    ("train_off", "Train MLP · anchors off", "noanchors"),
    ("predict_off", "Score every candle · anchors off", "noanchors"),
    ("train_on", "Train MLP · anchors on", "anchors"),
    ("predict_on", "Score every candle · anchors on", "anchors"),
    ("ranking", "Rank against random (test period)", None),
)
SCORER_LABELS = {
    "random": "Random ranking", "mlp_no_anchors": "MLP · anchors off", "mlp_anchors": "MLP · anchors on",
    "logistic": "Logistic regression", "gbt": "Boosted trees",
}


def bracket_pairs():
    """Key -> export pair for every bracket export that has its zeroed-anchor twin."""
    pairs = {}
    for path in sorted(EXPORTS.glob("mlp_*")):
        manifest = read_json(path / "manifest.json")
        bracket = manifest.get("bracket")
        twin = path.with_name(path.name + "_noanchors")
        if not bracket or manifest.get("anchor_inputs") == "zeroed" or not (twin / "manifest.json").is_file():
            continue
        summary = read_json(path / "summary.json")
        tp, sl = bracket["take_profit_pct"], bracket["stop_pct"]
        label = f"{bracket['side'].capitalize()} +{tp:g}% / −{sl:g}%  ·  {tp / sl:g}:1"
        pairs[path.name] = {
            "label": label, "on": path.name, "off": twin.name, "break_even": summary.get("break_even_win_rate"),
            "periods": {name: item.get("win_rate") for name, item in summary.get("periods", {}).items()},
        }
    return pairs


def run_folder(study, suffix):
    return RUNS / f"{study}-{suffix}"


def step_done(study, key, suffix):
    if key.startswith("train"):
        return (run_folder(study, suffix) / "test_metrics.json").is_file()
    if key.startswith("predict"):
        return (run_folder(study, suffix) / "predictions" / "predictions.csv").is_file()
    return (STUDIES / study / "ranking" / "ranking.json").is_file()


def wsl_launcher():
    result = subprocess.run([*WSL_SESSION, "--", "wslpath", "-a", (ROOT / "train.sh").as_posix()],
                            capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    return result.stdout.decode("utf-8").strip()


def step_command(study, config, key, suffix, launcher):
    export = config["off" if suffix == "noanchors" else "on"]
    relative = f"../../Exports/training/{export}"
    run = f"runs/{study}-{suffix}"
    wsl = [*WSL_SESSION, "--", "nice", "-n", "10", "bash", launcher]
    model = config["model"]
    if key.startswith("train"):
        command = [
            "--output-dir", run, "--bitfinex-dir", relative, "--dataset", "bitfinex", "--data-period", "all",
            "--model", "dense", "--model-input", model["model_input"], "--input-timeframe", "1m",
            "--feature-input", "exported", "--target", "net", "--loss", "cross_entropy",
            "--selection", "average_precision", "--min-entry-calls", "30", "--activation", "leaky_relu",
            "--hidden3", "0", "--train-fraction", "0.7", "--validation-fraction", "0.15",
            "--torch-threads", CPU_THREADS,
        ]
        for name in ("hidden1", "hidden2", "dropout", "epochs", "batch_size", "learning_rate", "lookback_minutes",
                     "seed"):
            command += [f"--{name.replace('_', '-')}", str(model[name])]
        return wsl + command
    return wsl + ["--predict", "--checkpoint", f"{run}/best.pt", "--bitfinex-dir", relative,
                  "--output-dir", f"{run}/predictions", "--batch-size", "4096", "--torch-threads", CPU_THREADS]


def ranking_command(study, config):
    command = [str(Path(sys.executable).with_name("python.exe")), str(RANKING),
               "--export", str(EXPORTS / config["on"]),
               "--scores", f"mlp_no_anchors={run_folder(study, 'noanchors') / 'predictions' / 'predictions.csv'}",
               "--scores", f"mlp_anchors={run_folder(study, 'anchors') / 'predictions' / 'predictions.csv'}",
               "--compare", "mlp_anchors,mlp_no_anchors", "--output", str(STUDIES / study / "ranking")]
    if config.get("baselines"):
        command += ["--baselines", "logistic,gbt"]
    env = dict(os.environ, PYTHONPATH=str(SITE_PACKAGES), PYTHONIOENCODING="utf-8")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = CPU_THREADS
    return command, env


def clear_partial(study, key, suffix):
    """A step that stopped halfway leaves a non-empty output folder the trainer refuses to reuse."""
    if key.startswith("train"):
        folder = run_folder(study, suffix)
        if folder.exists() and any(folder.iterdir()):
            folder.rename(folder.with_name(f"{folder.name}-stopped-{datetime.now():%H%M%S}"))
    elif key.startswith("predict"):
        shutil.rmtree(run_folder(study, suffix) / "predictions", ignore_errors=True)


def run_study(study):
    folder = STUDIES / study
    config = read_json(folder / "study.json")
    with (folder / "console.log").open("ab") as log:
        launcher = None
        for key, title, suffix in STEPS:
            if step_done(study, key, suffix):
                continue
            write_state(folder, phase="running", step=key, pid=os.getpid())
            log.write(f"\n===== {title} · {datetime.now():%Y-%m-%d %H:%M:%S} =====\n".encode("utf-8"))
            log.flush()
            try:
                clear_partial(study, key, suffix)
                if key == "ranking":
                    command, env = ranking_command(study, config)
                    flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
                else:
                    launcher = launcher or wsl_launcher()
                    command, env = step_command(study, config, key, suffix, launcher), None
                    flags = subprocess.CREATE_NO_WINDOW
                log.write(("> " + " ".join(command) + "\n").encode("utf-8"))
                log.flush()
                code = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                        creationflags=flags).wait()
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                log.write(f"{error}\n".encode("utf-8"))
                code = 1
            if code or not step_done(study, key, suffix):
                write_state(folder, phase="failed", step=key, returncode=code or 1)
                return
        write_state(folder, phase="done")


def fmt_pct(value, digits=1):
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}%}"


def fmt_interval(pair, scale=1.0, digits=3, suffix=""):
    if not pair:
        return "—"
    return f"[{pair[0] * scale:.{digits}f}, {pair[1] * scale:.{digits}f}]{suffix}"


def training_progress(run):
    history = [record for record in read_history(run / "history.jsonl") if "epoch" in record]
    if not history:
        return ""
    best = None
    for record in history:
        if record.get("best"):
            best = record
    text = f"epoch {history[-1]['epoch']}"
    if best:
        ap = best.get("validation", {}).get("average_precision")
        text += f" · best validation AP {ap:.4f} at epoch {best['epoch']}" if ap is not None else ""
    return text


class BracketPanel:
    def __init__(self, window):
        self.window = window
        window.title("Bracket study · MLP against random rankings")
        window.geometry("1220x820")
        window.minsize(980, 640)
        self.pairs = bracket_pairs()
        self.trainers = {}
        self.pending = None
        STUDIES.mkdir(parents=True, exist_ok=True)

        outer = ttk.Frame(window, padding=12)
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer)
        top.pack(fill="x")
        self.build_setup(top)
        self.build_study(top)

        tabs = ttk.Notebook(outer)
        tabs.pack(fill="both", expand=True, pady=(12, 0))
        results = ttk.Frame(tabs, padding=10)
        console = ttk.Frame(tabs, padding=6)
        tabs.add(results, text="Results")
        tabs.add(console, text="Console")
        self.summary = tk.Text(results, height=9, wrap="word", relief="flat", font=("Segoe UI", 10))
        self.summary.pack(fill="x")
        columns = ("scorer", "auc", "auc95", "top10", "lift95", "p5", "p10", "p20", "verdict")
        self.table = ttk.Treeview(results, columns=columns, show="headings", height=6)
        heads = ("Scorer", "AUC", "AUC 95%", "Top-10% win", "Top-10% lift 95% (pts)", "Beats random · 5%",
                 "· 10%", "· 20%", "Beats random")
        widths = (170, 70, 130, 100, 160, 120, 70, 70, 100)
        for column, head, width in zip(columns, heads, widths):
            self.table.heading(column, text=head)
            self.table.column(column, width=width, anchor="w" if column == "scorer" else "center")
        self.table.pack(fill="both", expand=True, pady=(8, 0))
        self.console = tk.Text(console, wrap="none", font=("Consolas", 9))
        scroll = ttk.Scrollbar(console, command=self.console.yview)
        self.console.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.console.pack(fill="both", expand=True)

        self.check_trainers()
        self.refresh()
        self.poll()

    def build_setup(self, parent):
        box = ttk.LabelFrame(parent, text="New study", padding=10)
        box.pack(side="left", fill="y")
        ttk.Label(box, text="Bracket").grid(row=0, column=0, sticky="w", pady=2)
        labels = [pair["label"] for pair in self.pairs.values()]
        self.pair = tk.StringVar(value=labels[0] if labels else "")
        combo = ttk.Combobox(box, textvariable=self.pair, values=labels, state="readonly", width=30)
        combo.grid(row=0, column=1, sticky="w", pady=2)
        combo.bind("<<ComboboxSelected>>", lambda _: self.show_pair())
        self.pair_info = tk.StringVar()
        ttk.Label(box, textvariable=self.pair_info, foreground="#555").grid(row=1, column=0, columnspan=2,
                                                                          sticky="w", pady=(0, 6))
        ttk.Label(box, text="Model input").grid(row=2, column=0, sticky="w", pady=2)
        self.model_input = tk.StringVar(value=INPUTS[DEFAULTS["model_input"]])
        ttk.Combobox(box, textvariable=self.model_input, values=list(INPUTS.values()), state="readonly",
                     width=30).grid(row=2, column=1, sticky="w", pady=2)
        self.fields = {}
        for row, (key, title, _) in enumerate(FIELDS, start=3):
            ttk.Label(box, text=title).grid(row=row, column=0, sticky="w", pady=2)
            variable = tk.StringVar(value=str(DEFAULTS[key]))
            ttk.Entry(box, textvariable=variable, width=12).grid(row=row, column=1, sticky="w", pady=2)
            self.fields[key] = variable
        self.baselines = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="Also rank with logistic regression and boosted trees",
                        variable=self.baselines).grid(row=row + 1, column=0, columnspan=2, sticky="w", pady=(6, 2))
        ttk.Label(box, text="Split 70 / 15 / 15 by time, kept best by validation AP.\n"
                            "Uses the GPU in WSL, 2 CPU threads, lower priority.",
                  foreground="#555").grid(row=row + 2, column=0, columnspan=2, sticky="w", pady=(4, 6))
        self.start_button = ttk.Button(box, text="Start study", command=self.start)
        self.start_button.grid(row=row + 3, column=0, columnspan=2, sticky="we")
        self.show_pair()

    def build_study(self, parent):
        box = ttk.LabelFrame(parent, text="Study", padding=10)
        box.pack(side="left", fill="both", expand=True, padx=(12, 0))
        bar = ttk.Frame(box)
        bar.pack(fill="x")
        self.study = tk.StringVar()
        self.studies = ttk.Combobox(bar, textvariable=self.study, state="readonly", width=44)
        self.studies.pack(side="left")
        self.studies.bind("<<ComboboxSelected>>", lambda _: self.refresh(force=True))
        self.open_button = ttk.Button(bar, text="Open folder", command=self.open_folder)
        self.open_button.pack(side="right")
        self.pause_button = ttk.Button(bar, text="Pause training", command=self.toggle_pause)
        self.pause_button.pack(side="right", padx=6)
        self.resume_button = ttk.Button(bar, text="Resume study", command=self.resume)
        self.resume_button.pack(side="right")
        self.study_info = tk.StringVar()
        ttk.Label(box, textvariable=self.study_info, foreground="#555").pack(fill="x", pady=(6, 4))
        self.steps = ttk.Treeview(box, columns=("step", "status", "detail"), show="headings", height=5)
        for column, head, width in (("step", "Step", 230), ("status", "Status", 90), ("detail", "Detail", 400)):
            self.steps.heading(column, text=head)
            self.steps.column(column, width=width, anchor="w")
        self.steps.pack(fill="both", expand=True)

    def show_pair(self):
        pair = self.selected_pair()
        if not pair:
            self.pair_info.set("No bracket export pair found in Exports/training.")
            return
        periods = pair["periods"]
        rates = " / ".join(fmt_pct(periods.get(name)) for name in ("train", "validation", "test"))
        self.pair_info.set(f"Win rate train / validation / test: {rates}\nBreak-even: {fmt_pct(pair['break_even'])}")

    def selected_pair(self):
        return next((pair for pair in self.pairs.values() if pair["label"] == self.pair.get()), None)

    def model_settings(self):
        values = {"model_input": next(k for k, v in INPUTS.items() if v == self.model_input.get())}
        for key, title, kind in FIELDS:
            try:
                values[key] = kind(self.fields[key].get())
            except ValueError:
                raise ValueError(f"Enter a valid value for {title.lower()}.") from None
        for key in ("hidden1", "hidden2", "epochs", "batch_size", "learning_rate", "lookback_minutes"):
            if not math.isfinite(values[key]) or values[key] <= 0:
                raise ValueError(f"{key.replace('_', ' ').capitalize()} must be positive.")
        if not 0 <= values["dropout"] < 1:
            raise ValueError("Dropout must be from 0 up to but excluding 1.")
        if values["lookback_minutes"] > 1409:
            raise ValueError("The export keeps at most 1,409 minutes of history.")
        return values

    # Study state

    def study_names(self):
        folders = [path for path in STUDIES.glob("*") if (path / "study.json").is_file()]
        return [path.name for path in sorted(folders, key=lambda path: path.stat().st_mtime, reverse=True)]

    def worker_alive(self, name):
        state = read_json(STUDIES / name / "status.json")
        if state.get("phase") == "running" and process_running(state.get("pid")):
            return True
        return bool(self.pending and self.pending[0] == name and self.pending[1].poll() is None)

    def busy_study(self):
        return next((name for name in self.study_names() if self.worker_alive(name)), None)

    def check_trainers(self):
        try:
            self.trainers = training_control("status")
        except (OSError, ValueError, subprocess.SubprocessError):
            self.trainers = {}

    def launch(self, name):
        worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", name],
                                  creationflags=subprocess.CREATE_NO_WINDOW, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.pending = (name, worker)
        self.study.set(name)
        self.refresh(force=True)

    def gpu_clear(self, name):
        self.check_trainers()
        others = sorted(run for run, paused in self.trainers.items() if not paused and not run.startswith(name))
        if not others:
            return True
        return messagebox.askyesno(
            "Other training is running",
            "These runs are training on the GPU now:\n\n" + "\n".join(others) +
            "\n\nStart anyway? Both will share the GPU and run slower.", parent=self.window)

    def start(self):
        busy = self.busy_study()
        if busy:
            messagebox.showinfo("Study in progress", f"{busy} is still running.", parent=self.window)
            return
        pair = self.selected_pair()
        try:
            if not pair:
                raise ValueError("No bracket export pair to train on.")
            model = self.model_settings()
        except ValueError as error:
            messagebox.showerror("Check the settings", str(error), parent=self.window)
            return
        key = pair["on"].split("bracket_")[-1]
        name = f"bracket-{key}-{datetime.now():%Y%m%d-%H%M%S}"
        if not self.gpu_clear(name):
            return
        folder = STUDIES / name
        folder.mkdir(parents=True)
        config = {"label": pair["label"], "on": pair["on"], "off": pair["off"], "break_even": pair["break_even"],
                  "model": model, "baselines": self.baselines.get(), "created": datetime.now().isoformat(timespec="seconds")}
        (folder / "study.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        self.launch(name)

    def resume(self):
        name = self.study.get()
        if not name or self.busy_study():
            return
        if self.gpu_clear(name):
            self.launch(name)

    def current_run(self, name):
        state = read_json(STUDIES / name / "status.json")
        suffix = next((s for key, _, s in STEPS if key == state.get("step")), None)
        if state.get("phase") == "running" and suffix and state.get("step", "").startswith("train"):
            return f"{name}-{suffix}"
        return None

    def toggle_pause(self):
        run = self.current_run(self.study.get())
        if not run:
            return
        self.check_trainers()
        try:
            self.trainers.update(training_control("resume" if self.trainers.get(run) else "pause", run))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            messagebox.showerror("Training control failed", str(error), parent=self.window)
        self.refresh(force=True)

    def open_folder(self):
        folder = STUDIES / self.study.get()
        if self.study.get() and folder.is_dir():
            os.startfile(folder)

    # Display

    def poll(self):
        self.refresh()
        self.window.after(3000, self.poll)

    def refresh(self, force=False):
        names = self.study_names()
        self.studies["values"] = names
        if self.study.get() not in names:
            self.study.set(names[0] if names else "")
        busy = self.busy_study()
        self.start_button.configure(state="disabled" if busy or not self.pairs else "normal")
        name = self.study.get()
        if not name:
            self.study_info.set("Start a study to train both MLPs and rank them against random.")
            for button in (self.resume_button, self.pause_button, self.open_button):
                button.configure(state="disabled")
            return
        folder = STUDIES / name
        config = read_json(folder / "study.json")
        state = read_json(folder / "status.json")
        alive = self.worker_alive(name)
        model = config.get("model", {})
        shape = f"{INPUTS.get(model.get('model_input'), '?')}"
        if model.get("model_input") == "window":
            shape += f" · {model.get('lookback_minutes')} min"
        self.study_info.set(
            f"{config.get('label', '')}  ·  {shape}  ·  {model.get('hidden1')}/{model.get('hidden2')} · dropout "
            f"{model.get('dropout')} · {model.get('epochs')} epochs · batch {model.get('batch_size')} · "
            f"lr {model.get('learning_rate')}  ·  started {config.get('created', '')}")
        run = self.current_run(name) if alive else None
        if run and run not in self.trainers:
            self.check_trainers()
        paused = bool(run and self.trainers.get(run))
        self.steps.delete(*self.steps.get_children())
        for key, title, suffix in STEPS:
            detail = ""
            if step_done(name, key, suffix):
                status = "done"
                if key.startswith("train"):
                    metrics = read_json(run_folder(name, suffix) / "test_metrics.json")
                    detail = f"best epoch {metrics.get('best_epoch')} · test AP " \
                             f"{metrics.get('test', {}).get('average_precision', float('nan')):.4f}"
                elif key.startswith("predict"):
                    count = read_json(run_folder(name, suffix) / "predictions" / "prediction_summary.json").get("count")
                    detail = f"{count:,} candles scored" if count else ""
            elif state.get("step") == key and alive:
                status = "paused" if paused else "running"
                if key.startswith("train"):
                    detail = training_progress(run_folder(name, suffix))
                    detail = f"{detail} of {model.get('epochs')}" if detail else "starting…"
            elif state.get("step") == key and state.get("phase") == "failed":
                status = "failed"
                detail = f"exit code {state.get('returncode')} · see Console"
            elif state.get("step") == key:
                status = "stopped"
                detail = "Resume study continues from here"
            else:
                status = "waiting"
            self.steps.insert("", "end", values=(title, status, detail))
        unfinished = not step_done(name, "ranking", None)
        self.resume_button.configure(state="normal" if unfinished and not alive and not busy else "disabled")
        self.pause_button.configure(text="Resume training" if paused else "Pause training",
                                    state="normal" if run else "disabled")
        self.open_button.configure(state="normal")
        self.show_console(folder)
        self.show_results(folder, config, force)

    def show_console(self, folder):
        text = decoded_tail(folder / "console.log")
        if text != self.console.get("1.0", "end-1c"):
            at_end = self.console.yview()[1] > 0.99
            self.console.delete("1.0", "end")
            self.console.insert("1.0", text)
            if at_end:
                self.console.see("end")

    def show_results(self, folder, config, force):
        path = folder / "ranking" / "ranking.json"
        stamp = (str(folder), path.stat().st_mtime if path.is_file() else None)
        if not force and getattr(self, "_shown", None) == stamp:
            return
        self._shown = stamp
        self.table.delete(*self.table.get_children())
        self.summary.configure(state="normal")
        self.summary.delete("1.0", "end")
        report = read_json(path)
        if not report:
            self.summary.insert("1.0", "Results appear here when the ranking step has finished.")
            self.summary.configure(state="disabled")
            return
        scorers = report["scorers"]
        for name, result in scorers.items():
            top = result["tops"]["top_10%"]
            beats = [result["one_position"][f"pass_{rate}%"]["beats_random_share"] for rate in (5, 10, 20)]
            self.table.insert("", "end", values=(
                SCORER_LABELS.get(name, name), f"{result['auc']:.4f}", fmt_interval(result["auc_95"], digits=4),
                fmt_pct(top["win_rate"]), fmt_interval(top["lift_95"], 100, 2), *(fmt_pct(b, 0) for b in beats),
                "yes" if result["verdict_beats_random"] else "no"))
        base = next(iter(scorers.values()))
        lines = [f"Test period: {report['candles']:,} candles over {report['days']} days · base win rate "
                 f"{fmt_pct(base['base_win_rate'])} · break-even {fmt_pct(config.get('break_even'))} · mean net "
                 f"{base['base_mean_net']:.3f}% per trade"]
        off, on = scorers.get("mlp_no_anchors"), scorers.get("mlp_anchors")
        if off:
            lines.append(f"Step 3 · MLP without anchors beats random: {'PASS' if off['verdict_beats_random'] else 'FAIL'}")
        if on:
            lines.append(f"Step 5 · MLP with anchors beats random: {'PASS' if on['verdict_beats_random'] else 'FAIL'}")
        anchors_help = False
        for item in report.get("comparisons", []):
            anchors_help = item["auc_difference_95"][0] > 0
            lines.append(
                f"Anchors on − off: AUC {item['auc_difference']:+.4f} {fmt_interval(item['auc_difference_95'], digits=4)}"
                f" · top-10% win rate {item['top10_win_rate_difference'] * 100:+.2f} pts "
                f"{fmt_interval(item['top10_difference_95'], 100, 2)}")
        if off and on:
            go = off["verdict_beats_random"] and on["verdict_beats_random"] and anchors_help
            lines.append(f"Step 6 · move to the ATR exit: {'GO' if go else 'NOT YET'} (needs both MLPs to beat "
                         "random and the anchors' AUC gain interval above 0)")
        lines.append("Beats random at a pass rate = share of 500 random gates of that pass rate whose one-position "
                     "net the model beats; the rule needs 95% at two of the three rates, an AUC interval above 0.5 "
                     "and a top-10% lift interval above 0.")
        self.summary.insert("1.0", "\n".join(lines))
        self.summary.configure(state="disabled")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_study(sys.argv[2])
    else:
        window = tk.Tk()
        BracketPanel(window)
        window.mainloop()
