"""Run bitfinex_indicators_v2.1.py with the faster, result-identical exit path.

Same command line as the collector, for example:

    python Pipelines/Bitfinex/run_fast.py --workers 11 --stop-timeframe 5m --output Exports/training/mlp_...

The collector file is loaded unchanged; fast_exit is installed at import, so
spawned label workers (which re-import this file) use it too. Labels are
identical to the v9 path (see tests_fast_exit.py); only speed differs, so
label checkpoints from either path are interchangeable.
"""
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("bitfinex_indicators_v2_1", HERE / "bitfinex_indicators_v2.1.py")
pipeline = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = pipeline  # label workers unpickle functions by this module name
_spec.loader.exec_module(pipeline)

sys.path.insert(0, str(HERE))
import fast_exit  # noqa: E402

fast_exit.install(pipeline.filled_initial_order_engine(), pipeline.original_wave_engine())

if __name__ == "__main__":
    pipeline.main()
