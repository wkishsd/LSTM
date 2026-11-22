"""Memory mapped arrays, explicit ID joins and streaming row validation."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .schema import HISTORY, STATES, WIDTH, read_schema, window_input_size


@dataclass
class DataPaths:
    current: str
    current_ids: str
    current_times: str
    history: str
    history_ids: str
    history_times: str
    schema: str
    gross: str | None = None
    commission: str | None = None
    outcome_ids: str | None = None
    close_times: str | None = None


@dataclass
class TimelinePaths:
    timeline: str
    entries: str
    manifest: str


def mmap_array(path, name, tail, kind="numeric"):
    if not path or Path(path).suffix.lower() != ".npy":
        raise ValueError(f"{name}: a .npy file is required")
    arr = np.load(path, mmap_mode="r", allow_pickle=False)
    if arr.ndim != len(tail) + 1 or arr.shape[1:] != tail:
        raise ValueError(f"{name}: expected (N, {', '.join(map(str, tail))}), got {arr.shape}")
    if kind == "numeric" and arr.dtype.kind not in "fiu":
        raise ValueError(f"{name}: expected real numeric dtype, got {arr.dtype}")
    if kind == "time" and arr.dtype != np.dtype("int64"):
        raise ValueError(f"{name}: expected int64 Unix seconds")
    if kind == "ids" and arr.dtype.kind not in "iuU":
        raise ValueError(f"{name}: IDs must be integers or fixed-width Unicode (no pickle)")
    return arr


def id_index(arr, name):
    values = arr.tolist()
    if any(isinstance(x, str) and not x.strip() for x in values):
        raise ValueError(f"{name}: empty ID")
    result = {value: i for i, value in enumerate(values)}
    if len(result) != len(values):
        raise ValueError(f"{name}: duplicate entry_id; join would be ambiguous")
    return result


def unit_scales(gross_unit, commission_unit, gross_currency=None, commission_currency=None):
    relative = {"fraction": 1.0, "percent": 0.01, "bps": 0.0001}
    if gross_unit in relative and commission_unit in relative:
        return relative[gross_unit], relative[commission_unit]
    if gross_unit == commission_unit == "absolute":
        if not gross_currency or gross_currency != commission_currency:
            raise ValueError("Absolute amounts require identical explicit currency/unit tags")
        return 1.0, 1.0
    raise ValueError("Cannot mix absolute and relative units without conversion data")


def feature_errors(array):
    errors = []
    if not np.isfinite(array).all():
        errors.append("nonfinite_features")
    elif np.any(np.abs(array.astype(np.float64, copy=False)) > np.finfo(np.float32).max):
        errors.append("features_overflow_float32")
    anchors = array[..., 36:]
    if not np.all((anchors == 0) | (anchors == 1)):
        errors.append("nonbinary_anchors")
    return errors


@dataclass
class Prepared:
    paths: DataPaths | TimelinePaths
    schema: dict
    ids: np.ndarray
    times: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    history_rows: np.ndarray
    labels: np.ndarray | None
    valid: np.ndarray
    report: dict
    # Net return in percent of entry notional after fees, where the export records it, else None.
    returns: np.ndarray | None


def prepare(paths, *, training=True, gross_unit=None, commission_unit=None,
            gross_currency=None, commission_currency=None, max_holding_minutes=None):
    schema = read_schema(paths.schema)
    arrays = {
        "current": mmap_array(paths.current, "current", (WIDTH,)),
        "current_ids": mmap_array(paths.current_ids, "current_ids", (), "ids"),
        "current_times": mmap_array(paths.current_times, "current_times", (), "time"),
        "history": mmap_array(paths.history, "history", (HISTORY, WIDTH)),
        "history_ids": mmap_array(paths.history_ids, "history_ids", (), "ids"),
        "history_times": mmap_array(paths.history_times, "history_times", (HISTORY,), "time"),
    }
    groups = [["current_ids", "current", "current_times"], ["history_ids", "history", "history_times"]]
    if training:
        scales = unit_scales(gross_unit, commission_unit, gross_currency, commission_currency)
        arrays.update({
            "outcome_ids": mmap_array(paths.outcome_ids, "outcome_ids", (), "ids"),
            "gross": mmap_array(paths.gross, "gross", ()),
            "commission": mmap_array(paths.commission, "commission", ()),
        })
        groups.append(["outcome_ids", "gross", "commission"])
        if paths.close_times:
            arrays["close_times"] = mmap_array(paths.close_times, "close_times", (), "time")
            groups[-1].append("close_times")
        elif not isinstance(max_holding_minutes, int) or max_holding_minutes <= 0:
            raise ValueError("Supply close_times or a positive --max-holding-minutes upper bound")
    for group in groups:
        if len({len(arrays[name]) for name in group}) != 1:
            raise ValueError(f"Row counts do not match within {group}")
    indexes = {group[0]: id_index(arrays[group[0]], group[0]) for group in groups}
    families = {"string" if arrays[name].dtype.kind == "U" else "integer" for name in indexes}
    if len(families) != 1:
        raise ValueError("All ID arrays must use the same type family; no implicit coercion")
    # Small metadata stays in RAM; do not retain mmap handles to these files.
    ids = np.array(arrays["current_ids"], copy=True)
    times = np.array(arrays["current_times"], copy=True)
    n = len(ids)
    starts = np.zeros(n, dtype=np.int64)
    ends = times.copy()
    history_rows = np.full(n, -1, dtype=np.int64)
    labels = np.zeros(n, dtype=np.int64) if training else None
    valid, excluded = [], []
    counts = Counter()
    current_set = set(indexes["current_ids"])
    # Missing IDs on either side are reported; matched rows are joined by ID, not position.
    union = set().union(*(set(index) for index in indexes.values()))
    for entry_id in sorted(union - current_set):
        reasons = [f"missing_{name}" for name, index in indexes.items() if entry_id not in index]
        counts.update(reasons)
        excluded.append({"entry_id": entry_id, "reasons": reasons})
    for i, entry_id in enumerate(ids.tolist()):
        reasons = [f"missing_{name}" for name, index in indexes.items() if entry_id not in index]
        h = indexes["history_ids"].get(entry_id)
        t = int(times[i])
        # Restrict timestamps to a practical range, avoiding int64 arithmetic overflow.
        if not 0 <= t <= 32_503_680_000 or t % 60:
            reasons.append("invalid_current_time")
        reasons.extend(feature_errors(arrays["current"][i]))
        if not np.any(arrays["current"][i, 36:] == 1):
            reasons.append("no_current_anchor")
        if h is not None:
            history_rows[i] = h
            ht = arrays["history_times"][h]
            starts[i] = ht[0]
            if not np.array_equal(ht, t - np.arange(HISTORY, 0, -1, dtype=np.int64) * 60):
                reasons.append("invalid_history_timeline")
            reasons.extend(feature_errors(arrays["history"][h]))
        if training:
            o = indexes["outcome_ids"].get(entry_id)
            if o is not None:
                gross, fee = float(arrays["gross"][o]), float(arrays["commission"][o])
                net = gross * scales[0] - fee * scales[1]
                if not all(np.isfinite(x) for x in (gross, fee, net)):
                    reasons.append("nonfinite_outcome")
                elif fee < 0:
                    reasons.append("negative_commission")
                else:
                    labels[i] = int(net > 0)
                close = int(arrays["close_times"][o]) if paths.close_times else t + max_holding_minutes * 60
                if close < t or close > 32_503_680_000:
                    reasons.append("invalid_close_time")
                else:
                    ends[i] = close
        if reasons:
            reasons = sorted(set(reasons))
            counts.update(reasons)
            excluded.append({"entry_id": entry_id, "reasons": reasons})
        else:
            valid.append(i)
    report = {
        "current_rows": n, "union_ids": len(union), "accepted": len(valid),
        "excluded_count": len(excluded), "reason_counts": dict(counts), "excluded": excluded,
        "outcome_end_source": ("observed" if paths.close_times else "max_holding_upper_bound") if training else None,
    }
    return Prepared(paths, schema, ids, times, starts, ends, history_rows, labels,
                    np.asarray(valid, dtype=np.int64), report, None)


class TradeDataset:
    """Only a single sample is copied from mmap; DataLoader stacks CPU minibatches."""

    def __init__(self, prepared, indices, lookback_minutes=STATES):
        self.paths = prepared.paths
        self.indices = np.asarray(indices, dtype=np.int64)
        self.history_rows = prepared.history_rows
        self.labels = prepared.labels
        self.input_size = window_input_size(lookback_minutes)
        self.lookback_minutes = lookback_minutes
        self._current = self._history = None

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_current"] = state["_history"] = None
        return state

    def __getitem__(self, item):
        i = self.indices[item]
        row = self.history_rows[i]
        if isinstance(self.paths, TimelinePaths):
            if self._history is None:
                self._history = mmap_array(self.paths.timeline, "timeline", (WIDTH,))
            x = self._history[row + 1 - self.lookback_minutes:row + 1].astype(np.float32, copy=True)
        else:
            if self._current is None:
                self._current = mmap_array(self.paths.current, "current", (WIDTH,))
                self._history = mmap_array(self.paths.history, "history", (HISTORY, WIDTH))
            x = np.empty((self.lookback_minutes, WIDTH), dtype=np.float32)
            x[:-1] = self._history[row, STATES - self.lookback_minutes:]
            x[-1] = self._current[i]
        target = self.labels[i] if self.labels is not None else i
        return x.reshape(self.input_size), int(target)
