#!/usr/bin/env python3
"""Solo/shared anchor frequencies and pair coverage on a saved timeline.

Run: python anchor_frequency_analysis.py
     python Pipelines/Bitfinex/anchor_frequency_analysis.py --input Exports/indicators/bitfinex_indicators_v2.1.xlsx --step 50

Outputs a separate XLSX and reusable CSV/JSON datasets. No API requests.
Dependencies: numpy, pandas, openpyxl; the indicator module also imports requests.

Python use:
    report = analyze_workbook(Path("Exports/indicators/bitfinex_indicators_v2.1.xlsx"))
    frequency_data, band_data, pair_data = report.frequency, report.bands, report.pairs
    # For a freshly collected in-memory IndicatorDatasets object:
    report = analyze_anchor_dataset(datasets.anchors)

Frequency bands: zero separately, then 1-50, 51-100, etc. They group anchors
by observed activation count and do not define trading strategy groups.
"""

import argparse
from dataclasses import dataclass
from datetime import timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.series import SeriesLabel
from openpyxl.formatting.rule import ColorScaleRule, CellIsRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT.parents[1] / "Exports/indicators/bitfinex_indicators_v2.1.xlsx"
DEFAULT_OUTPUT = ROOT.parents[1] / "Exports/anchor_frequency/anchor_frequency_analysis.xlsx"
UTC3 = timezone(timedelta(hours=3))
ANCHOR_IDS = tuple(range(1, 36))
FLAG_COLUMNS = tuple(f"anchor_{n:02d}" for n in ANCHOR_IDS)


@dataclass
class AnchorFrequencyReport:
    frequency: pd.DataFrame
    bands: pd.DataFrame
    pairs: pd.DataFrame
    coverage: pd.DataFrame
    exclusive_pairs: pd.DataFrame
    events: pd.DataFrame
    metadata: dict


def indicator_module():
    """Load the versioned script without running its CLI or fetching data."""
    name = "_bitfinex_anchor_frequency_source"
    if name in sys.modules:
        return sys.modules[name]
    local_dependencies = ROOT / ".python-deps"
    if local_dependencies.exists():
        sys.path.insert(0, str(local_dependencies))
    spec = importlib.util.spec_from_file_location(name, ROOT / "bitfinex_indicators_v2.1.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_anchor_ids(value, legacy=False):
    if value is None or pd.isna(value):
        raise ValueError("Blank anchor assignment: use 0 explicitly for no anchor")
    if isinstance(value, (int, float, np.number)):
        if not np.isfinite(value) or int(value) != value:
            raise ValueError(f"Invalid anchor assignment: {value}")
        ids = () if value == 0 else (int(value),)
    else:
        tokens = str(value).strip().split(";")
        ids = () if tokens == ["0"] else tuple(int(t) for t in tokens)
    allowed_ids = tuple(range(1, 37)) if legacy else ANCHOR_IDS
    if len(set(ids)) != len(ids) or any(n not in allowed_ids for n in ids):
        raise ValueError(f"Unknown or repeated anchor IDs: {value}")
    if legacy:
        mapping = indicator_module().LEGACY_ANCHOR_ID_MAP
        ids = tuple(mapping[n] for n in ids if n in mapping)
    return tuple(sorted(ids))


def local_times_to_utc(values):
    times = pd.DatetimeIndex(pd.to_datetime(values))
    return times.tz_localize(UTC3).tz_convert("UTC") if times.tz is None else times.tz_convert("UTC")


def event_grid(anchors):
    """The same closed-5m observation universe used by the indicator script."""
    times = pd.DatetimeIndex(pd.to_datetime(anchors.timestamp, utc=True))
    if not len(times) or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Need nonempty, unique, chronological M1 timestamps")
    if not anchors[list(FLAG_COLUMNS)].isin([0, 1]).all().all():
        raise ValueError("Anchor flags must contain numeric 0/1 values")
    grid = pd.date_range(times[0].ceil("5min"), times[-1].floor("5min"), freq="5min", name="entry_at")
    observed = anchors[list(FLAG_COLUMNS)].copy()
    observed.index = times
    flags = observed.reindex(grid).fillna(0).astype(np.int64)
    for entry, ids in anchors.attrs.get("unmapped_entries", {}).items():
        entry = pd.Timestamp(entry).tz_convert("UTC")
        if entry in flags.index:
            for n in ids:
                if n not in ANCHOR_IDS:
                    raise ValueError(f"Unknown unmapped anchor: {n}")
                flags.loc[entry, f"anchor_{n:02d}"] = 1
    return flags


def saved_overlap_counts(sheet):
    rows = list(sheet.iter_rows(values_only=True))
    title = next((i for i, row in enumerate(rows) if row and row[0] == "Joint occurrence counts"), None)
    if title is None:
        raise ValueError("Workbook has no joint occurrence matrix to reconcile missing entry minutes")
    labels = tuple(value for value in rows[title + 2][1:] if value is not None)
    legacy = labels == tuple(range(1, 37))
    if labels != ANCHOR_IDS and not legacy:
        raise ValueError("Overlap matrix must contain the current 35 IDs or the original 36 IDs in order")
    count = len(labels)
    matrix = np.array([row[1:count + 1] for row in rows[title + 3:title + 3 + count]], dtype=np.int64)
    if legacy:
        keep = [i for i, n in enumerate(labels) if n != 23]
        matrix = matrix[np.ix_(keep, keep)]
    summary = {row[0]: row[8] for row in rows if len(row) > 8 and isinstance(row[0], str) and row[8] is not None}
    return matrix, summary, legacy


def analyze_workbook(path=DEFAULT_INPUT, step=50):
    """Read exported assignments, recover missing entry minutes, reconcile.

    M5 indicators in anchor_data provide the history for recovering events
    on missing M1 rows. Existing assignments are retained. Recovery only uses
    contiguous histories with at least 82 prior indicator candles. Missing
    intervals without sufficient history start as zero, and are accepted only
    if all current joint counts match the saved full-timeline matrix. If any
    active event cannot be recovered, stop rather than publish partial counts.
    """
    path = Path(path).resolve()
    module = indicator_module()
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        if not {"anchor_data", "anchor_overlap"}.issubset(workbook.sheetnames):
            raise ValueError("Use the v2.1 workbook with anchor_data and anchor_overlap")
        expected_counts, saved_summary, legacy = saved_overlap_counts(workbook["anchor_overlap"])
        source_rows = iter(workbook["anchor_data"].values)
        headers = next(source_rows)
        source = pd.DataFrame(source_rows, columns=headers)
        source.timestamp = local_times_to_utc(source.timestamp)
        ids = source.anchor_numbers.map(lambda value: parse_anchor_ids(value, legacy=legacy))
        for n, column in zip(ANCHOR_IDS, FLAG_COLUMNS):
            source[column] = ids.map(lambda active: int(n in active))
        flags = event_grid(source)
        missing = flags.index.difference(pd.DatetimeIndex(source.timestamp))
        native_columns = ["open", "high", "low", "close", "volume", *module.INDICATOR_COLUMNS]
        native = source[["source_candle_5m", *[f"{c}_5m" for c in native_columns]]].copy()
        native.columns = ["timestamp", *native_columns]
        native.timestamp = local_times_to_utc(native.timestamp)
        if native.groupby("timestamp")[native_columns].nunique().gt(1).any().any():
            raise ValueError("Repeated M5 source candles contain inconsistent values")
        native = native.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        native["available_at"] = native.timestamp + pd.Timedelta(minutes=5)
        recovered_times = set()
        runs = native.timestamp.diff().ne(pd.Timedelta(minutes=5)).cumsum()
        for _, run in native.groupby(runs):
            detected = module.detect_anchor_events(run)
            recoverable = detected.iloc[82:].index.intersection(missing)
            flags.loc[recoverable] = detected.loc[recoverable, list(FLAG_COLUMNS)].to_numpy()
            recovered_times.update(recoverable)
        values = flags.to_numpy(dtype=np.int64)
        if not np.array_equal(values.T @ values, expected_counts):
            raise ValueError("Reconstructed events do not match the saved overlap matrix. "
                             "Use analyze_anchor_dataset(datasets.anchors) on the original in-memory dataset "
                             "and export its complete events; do not infer missing signals as zero.")
        if len(flags) != saved_summary.get("Eligible closed 5m observations"):
            raise ValueError("Anchor dataset and overlap tab cover different windows")
        recovered_positive = int(flags.loc[missing].any(axis=1).sum())
        if not legacy and recovered_positive != saved_summary.get("Empty M1 entry minutes recovered"):
            raise ValueError("Recovered missing-entry count differs from the saved overlap summary")
        metadata = {"source_file": str(path), "source_window_utc3": saved_summary.get("Window (M1 openings, UTC+3)"),
                    "source_m1_rows": len(source), "missing_m1_entry_minutes": len(missing),
                    "recovered_active_entry_minutes": recovered_positive,
                    "insufficient_history_missing_entries_certified_zero": len(set(missing) - recovered_times),
                    "anchor_schema_version": module.ANCHOR_SCHEMA_VERSION,
                    "legacy_36_id_input_remapped": legacy,
                    "overlap_matrix_reconciled": True}
        return calculate_anchor_frequency(flags, step, module.ANCHOR_RULES, metadata)
    finally:
        workbook.close()


def analyze_anchor_dataset(anchors, step=50):
    """Analyze the production dataset directly, including its unmapped events."""
    metadata = {"source_file": "In-memory datasets.anchors", "source_m1_rows": len(anchors),
                "source_window_utc3": f"{anchors.timestamp.iloc[0].tz_convert(UTC3):%Y-%m-%d %H:%M} to "
                                      f"{anchors.timestamp.iloc[-1].tz_convert(UTC3):%Y-%m-%d %H:%M}"}
    return calculate_anchor_frequency(event_grid(anchors), step, indicator_module().ANCHOR_RULES, metadata)


def calculate_anchor_frequency(flags, step=50, rules=None, metadata=None):
    """Count each anchor once per 5m close, irrespective of extra companions.

    A solo activation has exactly one active anchor. Shared includes any
    other active anchor. Always-together pairs have identical nonempty
    activation sets: count(A)=count(B)=count(A and B). It does NOT require
    A and B to be the only anchors on those candles. Zero-count anchors never
    qualify. One-way inclusion is reported separately.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("The activation-band step must be a positive integer")
    flags = flags.loc[:, list(FLAG_COLUMNS)].copy()
    if flags.empty or not flags.isin([0, 1]).all().all():
        raise ValueError("Need nonempty numeric 0/1 event flags for all 35 anchors")
    times = pd.DatetimeIndex(flags.index)
    if times.tz is None or times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError("Event timestamps must be chronological, unique and timezone-aware")
    if not (times == times.floor("5min")).all() or (len(times) > 1 and not (times[1:] - times[:-1] == pd.Timedelta(minutes=5)).all()):
        raise ValueError("Events must cover every consecutive 5m entry opening, including zero-signal periods")
    flags = flags.astype(np.int64)
    values = flags.to_numpy(); n = len(flags); simultaneous = values.sum(axis=1)
    totals = values.sum(axis=0); solo = values[simultaneous == 1].sum(axis=0); shared = totals - solo
    fraction = lambda a, b: np.divide(a, b, out=np.full(np.shape(a), np.nan, dtype=float), where=b > 0)
    frequency = pd.DataFrame({"anchor_id": ANCHOR_IDS, "total_activations": totals,
                              "solo_activations": solo, "shared_activations": shared,
                              "solo_share": fraction(solo, totals), "shared_share": fraction(shared, totals),
                              "activation_rate": totals / n, "solo_rate": solo / n, "shared_rate": shared / n})
    frequency["band_index"] = np.where(totals > 0, (totals - 1) // step, -1)
    frequency["activation_band"] = frequency.band_index.map(lambda i: "0" if i == -1 else f"{i * step + 1}-{(i + 1) * step}")
    frequency["rule"] = frequency.anchor_id.map(rules or {}).fillna("")
    band_rows = []
    for index in range(-1, int(frequency.band_index.max()) + 1):
        group = frequency.loc[frequency.band_index.eq(index)]
        total, single, multiple = (int(group[c].sum()) for c in ("total_activations", "solo_activations", "shared_activations"))
        band_rows.append({"activation_band": "0" if index == -1 else f"{index * step + 1}-{(index + 1) * step}", "anchor_count": len(group),
                          "anchor_ids": ";".join(map(str, group.anchor_id.tolist())),
                          "total_activations": total, "solo_activations": single, "shared_activations": multiple,
                          "solo_share": single / total if total else np.nan, "shared_share": multiple / total if total else np.nan})
    joint = values.T @ values
    coverage = np.divide(joint, totals[:, None], out=np.full(joint.shape, np.nan), where=totals[:, None] > 0)
    np.fill_diagonal(coverage, np.nan)
    pair_rows = []
    for ia, a in enumerate(ANCHOR_IDS):
        for ib in range(ia + 1, len(ANCHOR_IDS)):
            b = ANCHOR_IDS[ib]; na, nb, both = int(totals[ia]), int(totals[ib]), int(joint[ia, ib])
            mutual = na > 0 and nb > 0 and na == nb == both
            a_with_b = na > 0 and both == na
            b_with_a = nb > 0 and both == nb
            relation = ("always together" if mutual else "inactive anchor" if not na or not nb
                        else "A always with B" if a_with_b else "B always with A" if b_with_a
                        else "partial overlap" if both else "never together")
            pair_mask = values[:, ia].astype(bool) & values[:, ib].astype(bool)
            alone = int((pair_mask & (simultaneous == 2)).sum())
            pair_rows.append({"anchor_a": a, "anchor_b": b, "a_total": na, "b_total": nb,
                              "joint_activations": both, "a_without_b": na - both, "b_without_a": nb - both,
                              "b_given_a": both / na if na else np.nan, "a_given_b": both / nb if nb else np.nan,
                              "pair_only_activations": alone, "pair_with_extra_anchors": both - alone,
                              "always_together": mutual, "relationship": relation})
    pairs = pd.DataFrame(pair_rows).sort_values(["always_together", "joint_activations", "anchor_a", "anchor_b"], ascending=[False, False, True, True]).reset_index(drop=True)
    metadata = {**(metadata or {}), "band_step": step, "eligible_5m_observations": n,
                "first_entry_utc3": times[0].tz_convert(UTC3).isoformat(), "last_entry_utc3": times[-1].tz_convert(UTC3).isoformat(),
                "total_anchor_activations": int(totals.sum()), "solo_event_candles": int((simultaneous == 1).sum()),
                "shared_event_candles": int((simultaneous >= 2).sum()),
                "always_together_pairs": int(pairs.always_together.sum()),
                "one_way_always_pairs": int(pairs.relationship.isin(["A always with B", "B always with A"]).sum()),
                "inactive_anchors": frequency.loc[frequency.total_activations.eq(0), "anchor_id"].tolist()}
    return AnchorFrequencyReport(frequency.sort_values(["total_activations", "anchor_id"], ascending=[False, True]).reset_index(drop=True),
                                 pd.DataFrame(band_rows), pairs,
                                 pd.DataFrame(coverage, index=pd.Index(ANCHOR_IDS, name="anchor_a"), columns=ANCHOR_IDS),
                                 pairs.loc[pairs.always_together].copy(), flags, metadata)


LABELS = {"anchor_id": "Anchor ID", "total_activations": "Total activations", "solo_activations": "Solo activations",
          "shared_activations": "Shared activations", "solo_share": "Solo / anchor total", "shared_share": "Shared / anchor total",
          "activation_rate": "Total / all 5m periods", "solo_rate": "Solo / all 5m periods", "shared_rate": "Shared / all 5m periods",
          "activation_band": "Activation band", "rule": "Anchor rule", "anchor_count": "Anchors in band", "anchor_ids": "Anchor IDs",
          "anchor_a": "Anchor A", "anchor_b": "Anchor B", "a_total": "A total", "b_total": "B total", "joint_activations": "A and B together",
          "a_without_b": "A without B", "b_without_a": "B without A", "b_given_a": "B present when A fires", "a_given_b": "A present when B fires",
          "pair_only_activations": "Pair alone", "pair_with_extra_anchors": "Pair with extra anchors", "always_together": "Always together", "relationship": "Relationship"}
RATE_COLUMNS = {"solo_share", "shared_share", "activation_rate", "solo_rate", "shared_rate", "b_given_a", "a_given_b"}


def write_table(workbook, name, title, description, frame):
    ws = workbook.create_sheet(name)
    ws.append([title]); ws.append([description]); ws.append([])
    ws.append([LABELS.get(c, c) for c in frame.columns])
    for row in frame.itertuples(index=False, name=None):
        ws.append([None if pd.isna(v) else int(v) if isinstance(v, np.integer) else bool(v) if isinstance(v, np.bool_) else v for v in row])
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(frame.columns))
    ws.merge_cells(start_row=2, start_column=1, end_row=2, end_column=len(frame.columns))
    ws.row_dimensions[1].height = 28; ws.row_dimensions[2].height = 38; ws.row_dimensions[4].height = 46
    ws.freeze_panes = "B5"; ws.auto_filter.ref = f"A4:{get_column_letter(len(frame.columns))}{4 + len(frame)}"
    ws.sheet_view.showGridLines = False
    for row in ws.iter_rows():
        for cell in row:
            cell.font = Font(name="Arial", size=10, bold=cell.row in (1, 4), color="172A3A")
            cell.alignment = Alignment(vertical="center", horizontal="center" if cell.row == 4 else None, wrap_text=cell.row in (2, 4))
            if cell.row == 4: cell.fill = PatternFill("solid", fgColor="DFE8EF")
    for i, column in enumerate(frame.columns, 1):
        width = 74 if column == "rule" else 46 if column == "anchor_ids" else 24 if column == "relationship" else 19
        ws.column_dimensions[get_column_letter(i)].width = width
        for row in range(5, 5 + len(frame)):
            ws.cell(row, i).number_format = "0.0%" if column in RATE_COLUMNS else "#,##0" if pd.api.types.is_numeric_dtype(frame[column]) else "General"
            if column in ("activation_band", "anchor_ids", "relationship"):
                ws.cell(row, i).alignment = Alignment(horizontal="left", indent=1, vertical="center")
    for row in range(5, 5 + len(frame)):
        ws.row_dimensions[row].height = 23
        if "rule" in frame.columns:
            cell = ws.cell(row, list(frame.columns).index("rule") + 1)
            cell.alignment = Alignment(wrap_text=True, vertical="center")
            ws.row_dimensions[row].height = 36
    return ws


def export_report(report, output=DEFAULT_OUTPUT):
    output = Path(output).resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    window = report.metadata.get("source_window_utc3") or f"{report.metadata['first_entry_utc3']} to {report.metadata['last_entry_utc3']}"
    caption = f"UTC+3: {window}. {report.metadata['eligible_5m_observations']:,} closed 5m observations. Counts use all anchors, including missing M1 entry minutes."
    workbook = Workbook(); workbook.remove(workbook.active)
    frequency = report.frequency.drop(columns="band_index")
    ws = write_table(workbook, "anchor_frequency", "Anchor activations: solo versus shared", caption, frequency)
    chart = BarChart(); chart.type = "bar"; chart.grouping = "stacked"; chart.overlap = 100
    chart.title = "Solo and shared activations by anchor"
    chart.y_axis.title = "Activations (closed 5m candles)"; chart.x_axis.title = "Anchor ID"
    chart.add_data(Reference(ws, min_col=3, max_col=4, min_row=4, max_row=4 + len(frequency)), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=5, max_row=4 + len(frequency)))
    chart.width = 28; chart.height = 27; chart.legend.position = "b"
    chart.y_axis.scaling.min = 0; chart.y_axis.majorUnit = report.metadata["band_step"]
    chart.x_axis.scaling.orientation = "maxMin"
    for series, color, label in zip(chart.series, ("ADBCC7", "315F80"), ("Solo activations", "Shared activations")):
        series.tx = SeriesLabel(v=label)
        series.graphicalProperties.solidFill = color; series.graphicalProperties.line.noFill = True
    ws.add_chart(chart, "M5")
    ws.cell(43, 1, "Source: " + report.metadata.get("source_file", "In-memory anchor events"))
    ws.merge_cells("A43:K43"); ws.cell(43, 1).alignment = Alignment(wrap_text=True); ws.row_dimensions[43].height = 32
    ws.cell(45, 1, "Solo/shared shares use each anchor's own total; rates use all eligible 5m periods. Blank shares mean zero activations.")
    ws.merge_cells("A45:K45"); ws.cell(45, 1).alignment = Alignment(wrap_text=True); ws.row_dimensions[45].height = 32
    ws.cell(47, 1, f"Always-together pairs: {report.metadata['always_together_pairs']}. One-way always-accompanied pairs: {report.metadata['one_way_always_pairs']}.")
    ws.merge_cells("A47:K47")
    bands = write_table(workbook, "frequency_bands", f"Activation frequency bands (step {report.metadata['band_step']})",
                        "Bands group anchors by individual total. Band sums count individual signals; several signals may occur on one candle.", report.bands)
    for row in range(5, 5 + len(report.bands)):
        bands.cell(row, 3).alignment = Alignment(wrap_text=True, horizontal="left", indent=1, vertical="center"); bands.row_dimensions[row].height = 32
    pair_display = report.pairs.copy()
    pair_display.always_together = pair_display.always_together.map({True: "Yes", False: "No"})
    write_table(workbook, "anchor_pairs", "Anchor pair appearances, including larger combinations",
                "Always together: both active sets are identical and nonempty. One-way coverage is shown separately. Extra anchors do not disqualify a pair.", pair_display)
    matrix = workbook.create_sheet("pair_coverage")
    last_column = get_column_letter(len(ANCHOR_IDS) + 1)
    last_row = len(ANCHOR_IDS) + 4
    matrix_range = f"B5:{last_column}{last_row}"
    matrix.cell(1, 1, "Directional companion coverage: P(column anchor | row anchor)")
    matrix.merge_cells(f"A1:{last_column}1"); matrix.row_dimensions[1].height = 26
    matrix.cell(2, 1, "100% in one direction = containment. 100% in both directions = always together in this timeline. Additional anchors are allowed. Blank = inactive row anchor or diagonal.")
    matrix.merge_cells(f"A2:{last_column}2"); matrix.row_dimensions[2].height = 38; matrix.cell(2, 1).alignment = Alignment(wrap_text=True)
    matrix.cell(3, 1, "Column: companion anchor ID. Row: triggering anchor ID. " + caption)
    matrix.merge_cells(f"A3:{last_column}3"); matrix.row_dimensions[3].height = 38; matrix.cell(3, 1).alignment = Alignment(wrap_text=True)
    matrix.cell(4, 1, "Anchor ID")
    for j, n in enumerate(ANCHOR_IDS, 2): matrix.cell(4, j, n)
    for i, a in enumerate(ANCHOR_IDS, 5):
        matrix.cell(i, 1, a)
        for j, b in enumerate(ANCHOR_IDS, 2):
            value = report.coverage.loc[a, b]
            cell = matrix.cell(i, j, None if pd.isna(value) else float(value)); cell.number_format = "0.0%"
            if a != b and value == 1 and report.coverage.loc[b, a] == 1:
                matrix.conditional_formatting.add(cell.coordinate, CellIsRule(operator="equal", formula=["1"],
                    font=Font(bold=True), fill=PatternFill("solid", fgColor="B8D8C1"), stopIfTrue=True))
    matrix.conditional_formatting.add(matrix_range, ColorScaleRule(start_type="num", start_value=0, start_color="FFFFFF", end_type="num", end_value=1, end_color="91B5CF"))
    matrix.conditional_formatting.add(matrix_range, CellIsRule(operator="equal", formula=["1"], font=Font(bold=True), stopIfTrue=True))
    matrix.freeze_panes = "B5"; matrix.column_dimensions["A"].width = 13
    for j in range(2, len(ANCHOR_IDS) + 2): matrix.column_dimensions[get_column_letter(j)].width = 8
    matrix.sheet_view.showGridLines = False
    for row in matrix.iter_rows(min_row=4):
        matrix.row_dimensions[row[0].row].height = 22
        for cell in row:
            cell.font = Font(name="Arial", size=10, bold=cell.column == 1 or cell.row == 4)
            cell.alignment = Alignment(horizontal="center", vertical="center")
    if not report.exclusive_pairs.empty:
        write_table(workbook, "always_together", "Pairs that never appear without each other in this timeline", caption, report.exclusive_pairs)
    workbook.save(output)
    prefix = output.with_suffix("")
    for label, frame in (("frequency", report.frequency), ("bands", report.bands), ("pairs", report.pairs)):
        frame.to_csv(prefix.with_name(prefix.name + "_" + label + ".csv"), index=False, encoding="utf-8-sig")
    events = report.events.copy(); events.index = events.index.tz_convert(UTC3)
    events.index.name = "entry_at"
    events.to_csv(prefix.with_name(prefix.name + "_events.csv"), encoding="utf-8-sig")
    prefix.with_name(prefix.name + "_summary.json").write_text(json.dumps(report.metadata, indent=2, ensure_ascii=False), encoding="utf8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--step", type=int, default=50)
    args = parser.parse_args()
    report = analyze_workbook(args.input, args.step)
    output = export_report(report, args.output)
    print(json.dumps(report.metadata, indent=2, ensure_ascii=False))
    print(f"Wrote {output}")
    if report.exclusive_pairs.empty:
        print("No nonempty anchor pairs always occur together in this timeline.")
    else:
        print(report.exclusive_pairs[["anchor_a", "anchor_b", "joint_activations", "pair_with_extra_anchors"]].to_string(index=False))
    one_way = report.pairs.loc[report.pairs.relationship.isin(["A always with B", "B always with A"])]
    if not one_way.empty:
        print("One-way always-accompanied pairs:")
        print(one_way[["anchor_a", "anchor_b", "a_total", "b_total", "joint_activations", "relationship"]].to_string(index=False))


if __name__ == "__main__":
    main()
