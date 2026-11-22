"""Pause or resume the WSL training process for a panel run."""

import json
import os
from pathlib import Path
import signal
import sys
import time


PAUSED_STATES = {"T", "t"}


def process_state(process):
    return (process / "stat").read_text().rsplit(") ", 1)[1].split()[0]


def training_processes():
    runs = Path(__file__).resolve().parent / "runs"
    processes = {}
    for process in Path("/proc").iterdir():
        if not process.name.isdecimal():
            continue
        try:
            args = (process / "cmdline").read_bytes().decode(errors="replace").split("\0")
            if args[1:4] != ["-m", "mlp_trades", "train"] or "--output-dir" not in args:
                continue
            output = Path(args[args.index("--output-dir") + 1])
            output = ((process / "cwd").readlink() / output).resolve()
            if output.parent == runs:
                state = process_state(process)
                if state not in {"Z", "X"}:
                    # The forked baseline process shares the trainer's command line.
                    processes.setdefault(output.name, []).append((process, state))
        except (OSError, IndexError):
            # A process can exit between the /proc reads.
            continue
    return processes


def paused(group):
    return all(state in PAUSED_STATES for _, state in group)


def control(action, name=None):
    processes = training_processes()
    if action == "status":
        return {run: paused(group) for run, group in processes.items()}
    if name not in processes:
        raise ValueError("The trainer is not running for this run.")
    pause = action == "pause"
    if not pause and any(not paused(group) for run, group in processes.items() if run != name):
        raise ValueError("Pause the other training run before resuming this one.")
    group = processes[name]
    current = {process for process, _ in training_processes().get(name, [])}
    for process, _ in group:
        # A pidfd targets the discovered process even if its numeric PID is reused.
        fd = os.pidfd_open(int(process.name))
        try:
            if process not in current:
                raise ValueError("The training process has changed.")
            signal.pidfd_send_signal(fd, signal.SIGSTOP if pause else signal.SIGCONT)
        finally:
            os.close(fd)
    for _ in range(30):
        if all((process_state(process) in PAUSED_STATES) == pause for process, _ in group):
            return {name: pause}
        time.sleep(0.05)
    raise ValueError("The trainer has not acknowledged the signal. Check its status again.")


if __name__ == "__main__":
    try:
        action, *names = sys.argv[1:]
        if action not in {"status", "pause", "resume"} or len(names) != int(action != "status"):
            raise ValueError("Expected status, pause RUN, or resume RUN.")
        print(json.dumps(control(action, *names)))
    except (ValueError, OSError) as error:
        sys.exit(str(error))
