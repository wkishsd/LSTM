#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

python="${MLP_PYTHON:-$HOME/.venvs/bitfinex-mlp/bin/python}"
export HSA_ENABLE_DXG_DETECTION=1
export LD_LIBRARY_PATH="/opt/rocm/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [[ "${1:-}" == --collect-profit ]]; then
    hold_minutes="$2"
    profit_export="$3"
    shift 3
    source_export="$("$python" -c 'import json; print(json.load(open("train.json"))["bitfinex_dir"])')"
    "$python" -m mlp_trades.profit_collector --source "$source_export" \
        --candles ../../Exports/candle_cache/ETHUSD/1m --output "$profit_export" --hold-minutes "$hold_minutes"
    set -- --bitfinex-dir "$profit_export" "$@"
fi

"$python" - <<'PY'
import torch

from mlp_trades.engine import resolve_device, seed_everything
from mlp_trades.model import TradeMLP
from mlp_trades.schema import INPUT_SIZE

if not torch.version.hip or not torch.cuda.is_available():
    raise SystemExit("Use the WSL environment with Radeon ROCm PyTorch. GPU is unavailable.")

device = resolve_device("cuda:0")
name = torch.cuda.get_device_name(device)
if "RX 9070" not in name.upper():
    raise SystemExit(f"Expected RX 9070 on device 0, found {name}.")

seed_everything(42)
model = TradeMLP().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
features = torch.randn(32, INPUT_SIZE, device=device)
labels = torch.arange(32, device=device) % 2
loss = torch.nn.functional.cross_entropy(model(features), labels)
if not torch.isfinite(loss):
    raise SystemExit("GPU probe produced a nonfinite loss.")
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
optimizer.step()
torch.cuda.synchronize(device)
print(f"GPU training step passed on {name}, HIP {torch.version.hip}", flush=True)
PY

if [[ "${1:-}" == --probe-only ]]; then
    exit 0
fi

exec "$python" -m mlp_trades train --config train.json "$@"
