"""Array-based replacement for the v9 exit loop used in labeling.

`process_filled_initial_order(entry, adx_hits, minutes, waves, args)` has the
same signature and the same trade/summary results as the engine's
`process_filled_initial_order` (the v9 `process_entry` plus the collector's
adapter: initial fill-candle timing, wave-6 market close, and 5m/15m stop
movement). It avoids pandas row access inside the per-minute loop, which is
where the v9 code spends most of its time.

Scope: it reproduces everything the labels depend on (trades with entry/exit
times, prices, reasons, short_result_pct and status; stop events' type, time
and levels; the summary counters and final state). Display-only audit fields
(local time strings, trade path statistics, support-search text) are not
produced. Use the v9 engine when those audit tables are needed.

Not wired in by default. `install(engine)` swaps it into an engine returned
by `filled_initial_order_engine()`; `uninstall(engine)` restores v9.
`tests_fast_exit.py` checks equality against v9 on sampled entries.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TF_MS_1M = 60_000
TF_MS_5M = 300_000
STOP_CANDLE_MS = {"5m": 300_000, "15m": 900_000}
RED_HISTORY_LIMIT = 20  # v9 PRE_ENTRY_RED_HISTORY_LIMIT


def _f(value: Any) -> float:
    """v9 safe_float: NaN for missing/unparseable."""
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _i(value: Any) -> Optional[int]:
    """v9 safe_int: None for missing/unparseable."""
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _color(op: float, cl: float) -> str:
    if op != op or cl != cl:
        return "unknown"
    if op > cl:
        return "red"
    if op < cl:
        return "green"
    return "neutral"


def _is_close(a: float, base: float, threshold: float) -> bool:
    if base == 0 or a != a or base != base:
        return False
    return abs(float(a) - float(base)) / abs(float(base)) <= float(threshold)


def _support_level(lows: List[float], threshold: float) -> Tuple[bool, Optional[float], str, str]:
    """v9 find_support_level_from_candles on already sorted lows: (found, level, type, reason)."""
    n = len(lows)
    if n == 0:
        return False, None, "", "no_candles"
    if n < 3:
        return False, None, "", "less_than_3_lows"

    def collect(start: int) -> Tuple[int, int]:
        base, i = lows[start], start
        while i < n and _is_close(lows[i], base, threshold):
            i += 1
        return i - start, i

    def search_base(start: int) -> bool:
        i = start
        while i < n:
            size, next_i = collect(i)
            if size >= 3:
                return True
            i = max(next_i, i + 1)
        return False

    if _is_close(lows[1], lows[0], threshold):
        if _is_close(lows[2], lows[0], threshold):
            return True, float(lows[0]), "type1", "lower_cluster_3_plus"
        if search_base(2):
            return True, float(lows[0]), "type3", "lower_pair_plus_upper_base_cluster"
        return False, None, "", "type3_no_upper_base_cluster"
    if search_base(1):
        return True, float(lows[0]), "type2", "single_low_below_upper_base_cluster"
    return False, None, "", "type2_no_upper_base_cluster"


class _Minutes:
    """Per-entry candle arrays: all rows (support candles), pre-range context and the live range."""

    def __init__(self, minutes: pd.DataFrame, entry_id: int, range_start: int, range_end: int):
        own = minutes[minutes["entry_id"].astype("Int64") == int(entry_id)]
        mts_all = pd.to_numeric(own["minute_mts_utc"], errors="coerce")
        # Candles of each 1m MACD wave, for support searches (v9 uses every row of the entry).
        self.wave_lows: Dict[int, List[float]] = {}
        if "macd_1m_wave_no" in own.columns:
            wave_no = pd.to_numeric(own["macd_1m_wave_no"], errors="coerce")
            low = pd.to_numeric(own["low"], errors="coerce")
            ok = wave_no.notna() & low.notna() & mts_all.notna()
            frame = pd.DataFrame({"w": wave_no[ok].astype("int64"), "low": low[ok].astype(float),
                                  "mts": mts_all[ok]})
            for w, group in frame.sort_values(["low", "mts"]).groupby("w", sort=False):
                self.wave_lows[int(w)] = group["low"].tolist()
        keep = mts_all.notna() & (mts_all < range_end)
        frame = own.loc[keep].assign(minute_mts_utc=mts_all[keep]).sort_values("minute_mts_utc", kind="mergesort")
        self.all_mts = frame["minute_mts_utc"].to_numpy()
        self.all_open = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
        self.all_high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
        self.all_low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
        self.all_close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
        self.all_wave_no = (pd.to_numeric(frame["macd_1m_wave_no"], errors="coerce").to_numpy(dtype=float)
                            if "macd_1m_wave_no" in frame.columns else np.full(len(frame), np.nan))
        self.all_wave_sign = (pd.to_numeric(frame["macd_1m_wave_sign"], errors="coerce").to_numpy(dtype=float)
                              if "macd_1m_wave_sign" in frame.columns else np.full(len(frame), np.nan))
        self.live = np.flatnonzero(self.all_mts >= range_start)
        self._reds = (self.all_open > self.all_close)  # NaN compares False, like v9 "unknown"

    def red_history(self, upto_ms: int) -> List[Tuple[int, float]]:
        """v9 build_initial_red_history: last 20 red candles starting at or before upto_ms."""
        end = int(np.searchsorted(self.all_mts, upto_ms, side="right"))
        idx = np.flatnonzero(self._reds[:end])[-RED_HISTORY_LIMIT:]
        return [(int(self.all_mts[i]), float(self.all_high[i])) for i in idx]

    def stop_candles(self, tf_ms: int):
        """_m5_candles: aggregate complete rows into tf_ms buckets; drop a leading partial bucket."""
        ok = ~(np.isnan(self.all_open) | np.isnan(self.all_high) | np.isnan(self.all_low) | np.isnan(self.all_close))
        mts = self.all_mts[ok].astype("int64")
        if not len(mts):
            return (np.empty(0, "int64"),) * 2 + (np.empty(0),) * 4
        bucket = mts // tf_ms * tf_ms
        starts = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
        ends = np.r_[starts[1:], len(mts)]
        op, hi, lo, cl = self.all_open[ok], self.all_high[ok], self.all_low[ok], self.all_close[ok]
        b = bucket[starts]
        o = op[starts]
        h = np.maximum.reduceat(hi, starts)
        lw = np.minimum.reduceat(lo, starts)
        c = cl[ends - 1]
        if mts[0] > b[0]:
            b, o, h, lw, c = b[1:], o[1:], h[1:], lw[1:], c[1:]
        return b, b + tf_ms, o, h, lw, c


def _adx_stop_ms(adx_hits: pd.DataFrame, entry_id: int, range_start: int, range_end: int) -> Optional[int]:
    if adx_hits is None or adx_hits.empty or "candle_mts_utc" not in adx_hits.columns:
        return None
    own = adx_hits[adx_hits["entry_id"].astype("Int64") == int(entry_id)]
    candle = pd.to_numeric(own["candle_mts_utc"], errors="coerce").dropna().astype("int64")
    detection = candle + TF_MS_5M
    detection = detection[(detection >= int(range_start)) & (detection < int(range_end))]
    return int(detection.min()) if len(detection) else None


def _negative_waves(waves: pd.DataFrame, entry_id: int) -> Dict[int, Optional[int]]:
    """First row per wave number with sign -1 -> its wave_end_ms (v9 takes df.iloc[0])."""
    if waves is None or waves.empty:
        return {}
    own = waves[waves["entry_id"].astype("Int64") == int(entry_id)]
    out: Dict[int, Optional[int]] = {}
    for no, sign, end in zip(own["macd_1m_wave_no"], own["macd_1m_wave_sign"], own["wave_end_ms_utc"]):
        n = _i(no)
        if n is None or n in out or not (sign == -1):
            continue
        out[n] = _i(end)
    return out


class _Trade:
    __slots__ = ("trade_no", "source", "entry_ms", "entry_price", "stop", "stop_set_ms", "first_stop",
                 "first_stop_ms", "reds", "order_no", "moves", "candidates", "ignored")

    def __init__(self, trade_no, source, entry_ms, entry_price, reds, order_no=None):
        self.trade_no, self.source, self.entry_ms, self.entry_price = trade_no, source, entry_ms, entry_price
        self.stop = None
        self.stop_set_ms = None
        self.first_stop = None
        self.first_stop_ms = None
        self.reds = reds  # list of (mts, high)
        self.order_no = order_no
        self.moves = self.candidates = self.ignored = 0


def process_filled_initial_order(entry: pd.Series, adx_hits: pd.DataFrame, minutes: pd.DataFrame,
                                 waves: pd.DataFrame, args) -> Tuple[list, list, list, list, dict]:
    entry_id = int(entry["entry_id"])
    entry_price = _f(entry.get("entry_price"))
    entry_ms = _i(entry.get("entry_ms_utc"))
    wave6_ms = _i(entry.get("wave6_start_ms_utc"))
    trades: List[Dict[str, Any]] = []
    stops: List[Dict[str, Any]] = []
    supports: List[Dict[str, Any]] = []
    orders: List[Dict[str, Any]] = []
    base = {"entry_id": entry_id, "entry_ms_utc": entry_ms if entry_ms is not None else np.nan,
            "wave6_start_ms_utc": wave6_ms if wave6_ms is not None else np.nan}
    summary = dict(base, final_reason="", adx_stop_ms_utc=np.nan, trades_count=0, closed_trades_count=0,
                   stops_triggered_count=0, support_searches_count=0, entry_orders_count=0,
                   reentries_executed_count=0, last_state="")
    if entry_ms is None or entry_price != entry_price:
        summary["final_reason"] = "bad_entry_data"
        return trades, stops, supports, orders, summary
    if wave6_ms is None:
        summary["final_reason"] = "no_wave6"
        return trades, stops, supports, orders, summary

    fill_candle = _i(entry.get("initial_fill_candle_ms"))
    range_start = fill_candle if fill_candle is not None else int(entry_ms) - int(entry_ms) % TF_MS_1M
    range_end = int(wave6_ms)
    m = _Minutes(minutes, entry_id, range_start, range_end)
    if not len(m.live):
        summary["final_reason"] = "no_minute_data"
        return trades, stops, supports, orders, summary
    adx_ms = _adx_stop_ms(adx_hits, entry_id, range_start, range_end)
    if adx_ms is not None:
        summary["adx_stop_ms_utc"] = adx_ms
    neg_waves = _negative_waves(waves, entry_id)
    intrabar = bool(entry.get("initial_fill_intrabar", False))
    stop_tf = getattr(args, "stop_timeframe", "1m")
    stop_x = float(args.stop_x)
    threshold = float(args.support_threshold)
    offset = float(args.reentry_offset_down)

    tf = None
    if stop_tf != "1m":
        tf = stop_tf
        c_start, c_close, c_open, c_high, _c_low, c_cl = m.stop_candles(STOP_CANDLE_MS[tf])
        c_red = c_open > c_cl
        c_green = c_open < c_cl
    tf_state = {"trade_no": None, "done": None}

    checked: set = set()
    counters = {"support": 0, "order": 0}
    pending: Optional[Dict[str, Any]] = None
    trade_no = 1
    stops_triggered = 0
    reentries = 0
    first_reds = m.red_history(range_start)
    active: Optional[_Trade] = _Trade(1, "initial", int(entry_ms), float(entry_price), first_reds)

    def event(trade, kind, ms, minute_ms, old, new, candidate, timeframe="1m"):
        stops.append({"entry_id": entry_id, "stop_event_no": len(stops) + 1, "trade_no": trade.trade_no,
                      "event_type": kind, "event_ms_utc": ms, "minute_mts_utc": minute_ms,
                      "candle_timeframe": timeframe,
                      "old_stop_level": old if old is not None else np.nan,
                      "new_stop_level": new if new is not None else np.nan,
                      "candidate_stop_level": candidate})
        return len(stops)

    def close(trade, reason, exit_ms, exit_price, event_no=None):
        short = np.nan
        if exit_price is not None and exit_price == exit_price and trade.entry_price:
            short = round(((float(trade.entry_price) - float(exit_price)) / float(trade.entry_price)) * 100.0, 6)
        trades.append(dict(base, trade_no=trade.trade_no, trade_source=trade.source,
                           trade_entry_ms_utc=int(trade.entry_ms), trade_entry_price=float(trade.entry_price),
                           entry_order_no=trade.order_no if trade.order_no is not None else np.nan,
                           first_exit_stop_level=trade.first_stop if trade.first_stop is not None else np.nan,
                           last_exit_stop_level=trade.stop if trade.stop is not None else np.nan,
                           stop_moves_count=trade.moves, stop_candidates_count=trade.candidates,
                           stop_ignored_count=trade.ignored,
                           exit_ms_utc=int(exit_ms) if exit_ms is not None else np.nan,
                           exit_price=float(exit_price) if exit_price is not None and exit_price == exit_price else np.nan,
                           exit_reason=reason, stop_trigger_event_no=event_no if event_no is not None else np.nan,
                           short_result_pct=short,
                           trade_status="closed" if exit_price is not None else "open_or_unclosed"))

    def search(at_ms, price, k):
        nonlocal pending
        wave_no, sign = m.all_wave_no[k], m.all_wave_sign[k]
        if wave_no != wave_no or sign != sign or int(sign) != 1:
            return
        prev = int(wave_no) - 1
        if prev in checked or prev not in neg_waves:
            return
        wave_end = neg_waves[prev]
        if wave_end is None or int(at_ms) < int(wave_end) + TF_MS_1M:
            return
        checked.add(prev)
        counters["support"] += 1
        found, level, kind, reason = _support_level(m.wave_lows.get(prev, []), threshold)
        usable = False
        order_level = None
        if found and level is not None:
            order_level = float(level) - offset
            usable = float(order_level) < float(price)
        supports.append({"entry_id": entry_id, "support_search_no": counters["support"], "search_ms_utc": int(at_ms),
                         "source_macd_1m_wave_no": prev, "support_found": found,
                         "support_level": level if level is not None else np.nan, "support_is_usable": usable})
        if not usable:
            return
        counters["order"] += 1
        pending = {"order_no": counters["order"], "set_ms": int(at_ms), "level": float(order_level)}
        orders.append({"entry_id": entry_id, "entry_order_no": counters["order"], "order_status": "pending",
                       "order_set_ms_utc": int(at_ms), "entry_order_level": float(order_level)})

    def order_status(status, trigger_ms=None):
        for row in reversed(orders):
            if row["entry_order_no"] == pending["order_no"]:
                row["order_status"] = status
                if trigger_ms is not None:
                    row["trigger_ms_utc"] = trigger_ms
                return

    def candidate_from(trade, green_high, set_ms, minute_ms, timeframe):
        basis = None
        for mts_red, high_red in reversed(trade.reds):
            if high_red >= float(green_high):
                basis = high_red
                break
        trade.candidates += 1
        if basis is None:
            trade.ignored += 1
            event(trade, "candidate_not_found", set_ms, minute_ms, trade.stop, trade.stop, np.nan, timeframe)
            return
        candidate = float(basis) + stop_x
        old = trade.stop
        if old is None:
            kind = "exit_stop_set"
            trade.stop = trade.first_stop = candidate
            trade.stop_set_ms = trade.first_stop_ms = set_ms
        elif candidate < float(old):
            kind = "exit_stop_moved_down"
            trade.stop, trade.stop_set_ms = candidate, set_ms
            trade.moves += 1
        else:
            kind = "candidate_ignored_not_lower"
            trade.ignored += 1
        event(trade, kind, set_ms, minute_ms, old, trade.stop, candidate, timeframe)

    def advance_tf(trade, upto_ms):
        """_advance_m5_stop for the 5m/15m stop timeframe."""
        if tf_state["trade_no"] != trade.trade_no:
            n_before = int(np.searchsorted(c_close, trade.entry_ms, side="right"))
            reds_idx = np.flatnonzero(c_red[:n_before])[-RED_HISTORY_LIMIT:]
            trade.reds = [(int(c_start[i]), float(c_high[i])) for i in reds_idx]
            tf_state["trade_no"], tf_state["done"] = trade.trade_no, int(trade.entry_ms)
        lo = int(np.searchsorted(c_close, tf_state["done"], side="right"))
        hi = int(np.searchsorted(c_close, int(upto_ms), side="right"))
        for i in range(lo, hi):
            tf_state["done"] = close_ms = int(c_close[i])
            if c_red[i]:
                trade.reds.append((int(c_start[i]), float(c_high[i])))
            elif c_green[i]:
                candidate_from(trade, float(c_high[i]), close_ms, int(c_start[i]), tf)

    final_reason = "wave6_reached"
    last_state = "position_active"
    broke = False
    for k in m.live:
        m_start = int(m.all_mts[k])
        m_end = m_start + TF_MS_1M
        op, hi, lo, cl = m.all_open[k], m.all_high[k], m.all_low[k], m.all_close[k]
        if intrabar and m_start == fill_candle:
            continue
        if adx_ms is not None and m_start >= adx_ms:
            final_reason = "adx_gt_25"
            if active is not None:
                no = event(active, "forced_close_adx_gt_25", m_start, m_start, active.stop, active.stop, np.nan)
                close(active, "study_ended_adx_gt_25", m_start, _f(op), no)
                active = None
            if pending is not None:
                order_status("canceled")
                pending = None
            last_state = "terminated_by_adx"
            broke = True
            break
        if active is None and pending is not None:
            if lo <= float(pending["level"]):
                order_status("triggered", m_end)
                trade_no += 1
                reentries += 1
                active = _Trade(trade_no, "reentry", m_end, float(pending["level"]), m.red_history(m_start),
                                pending["order_no"])
                pending = None
                last_state = "position_active_after_reentry"
                continue
            last_state = "pending_reentry_order"
            continue
        if active is None and pending is None:
            search(m_end, _f(cl), k)
            last_state = "flat_waiting_for_support_or_entry"
            continue
        if tf is not None:
            advance_tf(active, m_start)
        if active.stop is not None and hi >= float(active.stop):
            stops_triggered += 1
            level = float(active.stop)
            no = event(active, "exit_stop_triggered", m_end, m_start, active.stop, active.stop, np.nan)
            close(active, "exit_stop_triggered", m_end, level, no)
            active = None
            last_state = "flat_after_exit_stop"
            search(m_end, _f(cl), k)
            continue
        if tf is not None:
            continue
        color = _color(op, cl)
        if color == "red":
            if not any(mts == m_start for mts, _ in active.reds):
                active.reds.append((m_start, float(hi)))
            continue
        if color != "green":
            continue
        candidate_from(active, _f(hi), m_end, m_start, "1m")
    if not broke:
        final_reason = "wave6_reached"

    if active is not None:
        cutoff_open = _f(entry.get("cutoff_market_open"))
        if cutoff_open == cutoff_open:
            close(active, "market_close_wave6", int(entry["cutoff_market_ms"]), cutoff_open)
            last_state = "closed_at_market_cutoff"
        else:
            close(active, f"study_ended_{final_reason}", int(range_end), None)
            last_state = "position_open_at_study_end"
    if pending is not None:
        order_status("unfilled")
        last_state = "pending_order_unfilled_at_study_end"

    summary.update(final_reason=final_reason, trades_count=len(trades),
                   closed_trades_count=sum(1 for t in trades if t["trade_status"] == "closed"),
                   stops_triggered_count=stops_triggered, support_searches_count=len(supports),
                   entry_orders_count=len(orders), reentries_executed_count=reentries, last_state=last_state)
    return trades, stops, supports, orders, summary


def _make_ema_sma_seed():
    def ema_sma_seed(values: pd.Series, length: int) -> pd.Series:
        """v5 ema_sma_seed with the recursion on Python floats (same IEEE doubles, same results)."""
        alpha = 2.0 / (length + 1.0)
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        n = len(arr)
        out = np.full(n, np.nan, dtype=float)
        nan = np.isnan(arr)
        start = None
        if n >= length:
            # First i whose window arr[i:i+length] has no NaN (v5 scans windows in order).
            bad = np.convolve(nan.astype(np.int64), np.ones(length, dtype=np.int64), "valid")
            hits = np.flatnonzero(bad == 0)
            start = int(hits[0]) if len(hits) else None
        if start is None:
            return pd.Series(out, index=values.index)
        seed_i = start + length - 1
        prev = float(np.mean(arr[start:seed_i + 1]))
        out[seed_i] = prev
        values_list = arr.tolist()
        result = out.tolist()
        for i in range(seed_i + 1, n):
            v = values_list[i]
            if v != v:
                result[i] = prev
                continue
            prev = (v - prev) * alpha + prev
            result[i] = prev
        return pd.Series(np.asarray(result, dtype=float), index=values.index)
    return ema_sma_seed


def _make_build_waves(wave_class):
    def build_waves(hist: pd.Series):
        """v5 build_waves on a Python list (same strict-sign rules, same output)."""
        arr = pd.to_numeric(hist, errors="coerce").to_numpy(dtype=float).tolist()
        n = len(arr)
        wave_id = [-1] * n
        waves = []
        first = None
        cur_sign = 0
        for i, x in enumerate(arr):
            if x != x:
                continue
            si = 1 if x > 0.0 else -1 if x < 0.0 else 0
            if si != 0:
                first, cur_sign = i, si
                break
        if first is None:
            return [], wave_id
        cur_start, cur_wave_idx, prev_sign = first, 0, cur_sign
        for i in range(first, n):
            x = arr[i]
            if x != x:
                wave_id[i] = cur_wave_idx
                continue
            si = 1 if x > 0.0 else -1 if x < 0.0 else 0
            if si == 0:
                wave_id[i] = cur_wave_idx
                continue
            if si != prev_sign:
                waves.append(wave_class(idx=cur_wave_idx, sign=prev_sign, start_i=cur_start, end_i=i - 1))
                cur_wave_idx += 1
                cur_start = i
                prev_sign = si
            wave_id[i] = cur_wave_idx
        waves.append(wave_class(idx=cur_wave_idx, sign=prev_sign, start_i=cur_start, end_i=n - 1))
        return waves, wave_id
    return build_waves


def install(engine, wave=None) -> None:
    """Use the array-based loop for filled initial orders in this engine object and,
    when the v5 wave module is given, its faster identical MACD EMA and wave builder.

    Apply before labeling in each process (label workers need it in their initializer).
    """
    if not hasattr(engine, "_v9_process_filled_initial_order"):
        engine._v9_process_filled_initial_order = engine.process_filled_initial_order
    engine.process_filled_initial_order = process_filled_initial_order
    if wave is not None:
        if not hasattr(wave, "_v5_ema_sma_seed"):
            wave._v5_ema_sma_seed, wave._v5_build_waves = wave.ema_sma_seed, wave.build_waves
        wave.ema_sma_seed = _make_ema_sma_seed()
        wave.build_waves = _make_build_waves(wave.Wave)


def uninstall(engine, wave=None) -> None:
    if hasattr(engine, "_v9_process_filled_initial_order"):
        engine.process_filled_initial_order = engine._v9_process_filled_initial_order
    if wave is not None and hasattr(wave, "_v5_ema_sma_seed"):
        wave.ema_sma_seed, wave.build_waves = wave._v5_ema_sma_seed, wave._v5_build_waves
