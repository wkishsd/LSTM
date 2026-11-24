import argparse
import json
import math
import sys
from dataclasses import fields
from pathlib import Path

from .data import DataPaths, prepare
from .io_utils import new_output_dir, write_json
from .schema import (
    BENCHMARK_LOOKBACKS, DATA_PERIOD_LABELS, DATASET_LABELS, FEATURE_INPUT_LABELS,
    LABEL_AGREEMENT_LABELS, LOSS_LABELS, MODEL_LABELS, PROFIT_DATASET, STATES, window_input_size,
)

PATH_OPTIONS = {field.name for field in fields(DataPaths)} | {
    "checkpoint", "output_dir", "bitfinex_dir", "bitfinex_entries", "agreement_file",
}
# The baseline scripts live in the workspace's Models/Baselines folder.
BASELINES_DIR = Path(__file__).resolve().parents[3] / "Models" / "Baselines"
BASELINE_MODELS = ("atr", "logistic", "gbt", "hour")


def start_baselines(data, args, out):
    """Fork the CPU baselines before torch is imported, so the child never inherits a CUDA context."""
    import multiprocessing

    if str(BASELINES_DIR) not in sys.path:
        sys.path.insert(0, str(BASELINES_DIR))
    from trade_baselines import child_main

    process = multiprocessing.get_context("fork").Process(target=child_main, args=(data, args, out),
                                                          name="trade-baselines")
    process.start()
    print(f"Baselines started in process {process.pid}: {args.baselines}", flush=True)
    return process


def build_parser():
    parser = argparse.ArgumentParser(description="MLP for trade classification")
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("synthetic", help="Generate independent shuffled NumPy arrays")
    synthetic.add_argument("--output-dir", required=True)
    synthetic.add_argument("--samples", type=int, default=80)
    synthetic.add_argument("--seed", type=int, default=42)
    for command in ("validate", "train", "predict"):
        p = commands.add_parser(command)
        p.add_argument("--config", help="JSON config; paths inside resolve relative to the JSON file")
        p.add_argument("--output-dir")
        p.add_argument("--bitfinex-dir", help="Read feature_timeline.npy, entry_targets.csv and manifest.json from this export")
        if command == "predict":
            p.add_argument("--bitfinex-entries", help="Optional new-signal CSV with entry_id, entry_at and feature_end_row")
        for name in ("current", "current_ids", "current_times", "history", "history_ids", "history_times", "schema"):
            p.add_argument("--" + name.replace("_", "-"))
        if command != "predict":
            p.add_argument("--dataset", choices=DATASET_LABELS, default="bitfinex",
                           help="Use the Bitfinex export or a local UCI benchmark")
            p.add_argument("--data-period", choices=DATA_PERIOD_LABELS, default="all",
                           help="Select all Bitfinex entries or the export's last six calendar months")
            for name in ("gross", "commission", "outcome_ids", "close_times"):
                p.add_argument("--" + name.replace("_", "-"))
            for prefix in ("gross", "commission"):
                p.add_argument(f"--{prefix}-unit", choices=("fraction", "percent", "bps", "absolute"))
                p.add_argument(f"--{prefix}-currency")
            p.add_argument("--max-holding-minutes", type=int)
            p.add_argument("--train-fraction", type=float, default=0.7)
            p.add_argument("--validation-fraction", type=float, default=0.15,
                           help="Set to 0 to evaluate training each epoch and test the final checkpoint")
            # Also accepted by validate so the exact training config can be checked first.
            p.add_argument("--model", choices=MODEL_LABELS, default="dense",
                           help="Model architecture. The encoder models map each minute from 54 to 16 to 8 features.")
            p.add_argument("--feature-input", choices=FEATURE_INPUT_LABELS, default="exported",
                           help="Optional input transformation for the Bitfinex timeline")
            p.add_argument("--hidden1", type=int, default=128)
            p.add_argument("--initialization", choices=("default",), default="default",
                           help="Random PyTorch initialization; constant initializations were removed")
            p.add_argument("--loss", choices=LOSS_LABELS, default="cross_entropy",
                           help="Training and evaluation loss. MSE compares probabilities to one-hot labels.")
            p.add_argument("--diagnostic-steps", type=int, default=0,
                           help="Sample gradients and updates every N optimizer steps. 0 disables diagnostics.")
            p.add_argument("--hidden2", type=int, default=64)
            p.add_argument("--hidden3", type=int, default=0,
                           help="Third hidden-layer width; 0 disables it")
            p.add_argument("--activation", choices=("relu", "leaky_relu"), default="relu",
                           help="Hidden-layer activation; leaky_relu uses a negative slope of 0.01")
            p.add_argument("--dropout", type=float, default=0.0,
                           help="Fraction of hidden activations masked during training; 0 disables dropout")
            p.add_argument("--epochs", type=int, default=20)
            p.add_argument("--lookback-minutes", type=int, default=STATES,
                           help=f"Number of most recent minute rows per entry, from 1 to {STATES}")
            p.add_argument("--learning-rate", type=float, default=0.001)
            p.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adam")
            p.add_argument("--selection", choices=("accuracy", "loss", "profitable_above_baseline", "average_precision"),
                           default="accuracy")
            p.add_argument("--min-entry-calls", type=int, default=30,
                           help="Minimum validation entry calls to qualify for profitable_above_baseline")
            p.add_argument("--feature-scaling", choices=("none", "standardize"), default="none",
                           help="For a Bitfinex timeline, fit feature standardization on training minutes")
            p.add_argument("--timeline-cache", choices=("off", "auto"), default="off",
                           help="Auto builds Bitfinex windows on GPU when the timeline fits in VRAM")
            if command == "train":
                p.add_argument("--label-agreement", choices=LABEL_AGREEMENT_LABELS, default="off",
                               help="Train-split labels from label_agreement.csv: weight entries by agreement, or "
                                    "make class 1 the entries every exit configuration wins (against the rest, or "
                                    "against entries every configuration loses). Validation and test are unchanged.")
                p.add_argument("--agreement-file",
                               help="label_agreement.csv to use; defaults to the one in --bitfinex-dir")
                p.add_argument("--baselines", default="",
                               help="Comma-separated CPU baselines trained alongside the MLP on the same split: "
                                    + ", ".join(BASELINE_MODELS))
        else:
            p.add_argument("--checkpoint")
        p.add_argument("--batch-size", type=int, default=32)
        p.add_argument("--device", default="auto")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--num-workers", type=int, default=0)
        p.add_argument("--torch-threads", type=int, default=0, help="0 uses PyTorch default")
    return parser


def parse_args(argv=None):
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    first = parser.parse_args(argv)
    if getattr(first, "config", None):
        path = Path(first.config).resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object")
        actions = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction)).choices[first.command]._actions
        by_name = {action.dest: action for action in actions}
        unknown = config.keys() - by_name.keys()
        if unknown or "config" in config or "help" in config:
            raise ValueError(f"Unknown or reserved config options: {sorted(unknown | (config.keys() & {'config', 'help'}))}")
        # Convert config to CLI tokens so argparse applies the SAME type and choice checks.
        injected = []
        for name, value in config.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, bool)):
                raise ValueError(f"Invalid config scalar: {name}")
            if name in PATH_OPTIONS:
                value = str((path.parent / str(value)).resolve())
            injected.append(f"--{name.replace('_', '-')}={value}")
        first = parser.parse_args([argv[0], *injected, *argv[1:]])
    for name in PATH_OPTIONS:
        value = getattr(first, name, None)
        if value:
            setattr(first, name, str(Path(value).resolve()))
    if first.command == "synthetic":
        return first
    required = ["output_dir"]
    if first.command == "predict":
        required.append("checkpoint")
    dataset = getattr(first, "dataset", "bitfinex")
    if getattr(first, "data_period", "all") != "all" and (dataset in BENCHMARK_LOOKBACKS or not first.bitfinex_dir):
        raise ValueError("The data-period filter requires the Bitfinex timeline export")
    if dataset == PROFIT_DATASET:
        if not first.bitfinex_dir:
            raise ValueError("Profit buckets require a collected Bitfinex profit export")
        if first.selection == "profitable_above_baseline":
            raise ValueError("Use accuracy, loss, or macro average precision for profit buckets")
    if dataset in BENCHMARK_LOOKBACKS:
        limit, unit = BENCHMARK_LOOKBACKS[dataset]
        if not 1 <= first.lookback_minutes <= limit:
            raise ValueError(f"{DATASET_LABELS[dataset]} lookback must be between 1 and {limit} {unit}")
        if first.feature_input != "exported" or first.selection == "profitable_above_baseline":
            raise ValueError("Use the dataset features and select checkpoints by accuracy, loss, or average precision")
    elif first.bitfinex_dir:
        separate_options = {field.name for field in fields(DataPaths)} | {
            "gross_unit", "commission_unit", "gross_currency", "commission_currency", "max_holding_minutes",
        }
        if any(getattr(first, key, None) is not None for key in separate_options):
            raise ValueError("Use either --bitfinex-dir or separate NumPy input and outcome options")
    else:
        if getattr(first, "bitfinex_entries", None):
            raise ValueError("--bitfinex-entries requires --bitfinex-dir")
        if getattr(first, "feature_input", "exported") != "exported":
            raise ValueError("--feature-input transformations require --bitfinex-dir")
        required += ["current", "current_ids", "current_times", "history", "history_ids", "history_times", "schema"]
        if first.command != "predict":
            required += ["gross", "commission", "outcome_ids", "gross_unit", "commission_unit"]
    missing = [name for name in required if not getattr(first, name, None)]
    if missing:
        raise ValueError("Missing options: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    for name in ("batch_size", "hidden1", "hidden2", "epochs", "min_entry_calls"):
        if hasattr(first, name) and getattr(first, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if getattr(first, "diagnostic_steps", 0) < 0:
        raise ValueError("Diagnostic steps must be nonnegative")
    if getattr(first, "hidden3", 0) < 0:
        raise ValueError("hidden3 must be nonnegative (0 disables the third hidden layer)")
    if not 0 <= first.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if first.num_workers < 0 or first.torch_threads < 0:
        raise ValueError("Worker/thread counts must be nonnegative")
    if hasattr(first, "learning_rate") and (not math.isfinite(first.learning_rate) or first.learning_rate <= 0):
        raise ValueError("learning_rate must be finite and positive")
    if hasattr(first, "dropout") and (not math.isfinite(first.dropout) or not 0 <= first.dropout < 1):
        raise ValueError("dropout must be a finite fraction from 0 up to but excluding 1")
    if hasattr(first, "lookback_minutes"):
        window_input_size(first.lookback_minutes)
    if getattr(first, "label_agreement", "off") != "off":
        if dataset != "bitfinex" or not first.bitfinex_dir:
            raise ValueError("Label agreement needs the binary Bitfinex dataset from --bitfinex-dir")
        from .agreement import default_path
        agreement = Path(first.agreement_file or default_path(first.bitfinex_dir))
        if not agreement.is_file():
            raise ValueError(f"{agreement} not found; build it with Pipelines/Bitfinex/label_agreement.py")
    if getattr(first, "baselines", ""):
        names = [name.strip() for name in first.baselines.split(",") if name.strip()]
        unknown = sorted(set(names) - set(BASELINE_MODELS))
        if unknown:
            raise ValueError(f"Unknown baselines {unknown}; choose from {', '.join(BASELINE_MODELS)}")
        if dataset in BENCHMARK_LOOKBACKS or not first.bitfinex_dir:
            raise ValueError("Baselines require a Bitfinex timeline export")
        first.baselines = ",".join(dict.fromkeys(names))
    return first


def main(argv=None):
    try:
        args = parse_args(argv)
        if args.command == "synthetic":
            from .synthetic import generate
            print(generate(args.output_dir, args.samples, args.seed))
            return
        out = new_output_dir(args.output_dir)
        print("Validating memory-mapped arrays...", flush=True)
        training = args.command != "predict"
        if getattr(args, "dataset", "bitfinex") == "uci_har":
            from .har import prepare_har
            data = prepare_har()
        elif getattr(args, "dataset", "bitfinex") == "uci_air":
            from .air_quality import prepare_air
            data = prepare_air()
        elif args.bitfinex_dir:
            from .bitfinex import export_paths, prepare_bitfinex, select_bitfinex_period
            paths = export_paths(args.bitfinex_dir, getattr(args, "bitfinex_entries", None))
            data = prepare_bitfinex(paths, training=training)
            if training and ("profit_target" in data.schema) != (args.dataset == PROFIT_DATASET):
                raise ValueError("The Bitfinex export does not match the selected dataset")
            data = select_bitfinex_period(data, getattr(args, "data_period", "all"))
        else:
            paths = DataPaths(**{field.name: getattr(args, field.name, None) for field in fields(DataPaths)})
            kwargs = {name: getattr(args, name) for name in (
                "gross_unit", "commission_unit", "gross_currency", "commission_currency", "max_holding_minutes",
            )} if training else {}
            data = prepare(paths, training=training, **kwargs)
        write_json(out / "validation_report.json", data.report)
        print(json.dumps({key: value for key, value in data.report.items() if key != "excluded"}), flush=True)
        if args.command == "validate":
            from .split import training_split
            _, report = training_split(data, args.train_fraction, args.validation_fraction)
            write_json(out / "split_report.json", report)
            print(json.dumps({"split_counts": {name: item["count"] for name, item in report["splits"].items()}, "purged": report["purged_count"]}), flush=True)
        else:
            baselines = start_baselines(data, args, out) if getattr(args, "baselines", "") else None
            try:
                import torch
                from .engine import predict, train
                if args.torch_threads:
                    torch.set_num_threads(args.torch_threads)
                (predict if args.command == "predict" else train)(data, args, out)
            except BaseException:
                if baselines is not None:
                    baselines.terminate()
                raise
            if baselines is not None:
                if baselines.is_alive():
                    print("MLP finished; waiting for the baselines to finish...", flush=True)
                baselines.join()
                if baselines.exitcode:
                    print(f"Baselines exited with code {baselines.exitcode}; see baselines.json", flush=True)
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        message = str(exc)
        if "out of memory" in message.lower():
            message += " | Reduce --batch-size (down to 1), hidden widths, or use --device cpu."
        print(f"ERROR: {message}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
