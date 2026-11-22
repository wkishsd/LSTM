"""Beijing air-quality forecast using one station on a chronological hourly timeline."""

import csv
from datetime import datetime, timedelta, timezone
from io import BytesIO, TextIOWrapper
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .data import Prepared, TimelinePaths, mmap_array
from .schema import AIR_STEPS, WIDTH


AIR_DIR = Path(__file__).resolve().parent.parent / "datasets" / "uci_air"
HORIZON = 6
MEASUREMENTS = ("PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP", "PRES", "DEWP", "RAIN", "WSPM")
COMPASS = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
BEIJING = timezone(timedelta(hours=8))
CHANNELS = [
    *MEASUREMENTS, "wind_sin", "wind_cos",
    *(f"missing_{name}" for name in (*MEASUREMENTS, "wind")),
    "hour_sin", "hour_cos", "year_sin", "year_cos",
]
SCHEMA = {
    "dataset": "uci_beijing_pm25_rise", "version": 1, "station": "Aotizhongxin",
    "step_seconds": 3600, "lookback_steps": AIR_STEPS, "horizon_steps": HORIZON,
    "columns": [*CHANNELS, *(f"padding_{i}" for i in range(WIDTH - len(CHANNELS)))],
    "labels": {"0": "no rise", "1": "PM2.5 rises in six hours"},
    "source": "https://doi.org/10.24432/C5RK5G", "license": "CC BY 4.0",
}


def convert_air(archive_path, output=AIR_DIR):
    """Build the frozen benchmark without fitting or running a model."""
    with ZipFile(archive_path) as outer:
        inner = next(name for name in outer.namelist()
                     if Path(name).name == "PRSA2017_Data_20130301-20170228.zip")
        with ZipFile(BytesIO(outer.read(inner))) as archive:
            name = next(name for name in archive.namelist()
                        if Path(name).name == "PRSA_Data_Aotizhongxin_20130301-20170228.csv")
            with TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline="") as source:
                records = list(csv.DictReader(source))

    count = len(records)
    timestamps = np.empty(count, dtype=np.int64)
    measurements = np.full((count, len(MEASUREMENTS) + 2), np.nan, dtype=np.float32)
    clocks = np.empty((count, 4), dtype=np.float32)
    for i, row in enumerate(records):
        time = datetime(*(int(row[field]) for field in ("year", "month", "day", "hour")), tzinfo=BEIJING)
        timestamps[i] = int(time.timestamp())
        for column, field in enumerate(MEASUREMENTS):
            if row[field] not in ("", "NA"):
                measurements[i, column] = float(row[field])
        if row["wd"] in COMPASS:
            angle = 2 * np.pi * COMPASS.index(row["wd"]) / len(COMPASS)
            measurements[i, -2:] = np.sin(angle), np.cos(angle)
        hour = 2 * np.pi * time.hour / 24
        year = 2 * np.pi * (time.timetuple().tm_yday - 1) / 365.2425
        clocks[i] = np.sin(hour), np.cos(hour), np.sin(year), np.cos(year)
    if not np.all(np.diff(timestamps) == 3600):
        raise ValueError("The air-quality station must have one row per hour")

    observed = np.isfinite(measurements)
    candidates = np.arange(AIR_STEPS - 1, count - HORIZON, dtype=np.int64)
    ends = candidates[observed[candidates, 0] & observed[candidates + HORIZON, 0]]
    labels = (measurements[ends + HORIZON, 0] > measurements[ends, 0]).astype(np.int64)

    # Carry measurements forward only. A missing flag distinguishes replacements.
    last_seen = np.maximum.accumulate(np.where(observed, np.arange(count)[:, None], -1), axis=0)
    filled = np.take_along_axis(measurements, np.maximum(last_seen, 0), axis=0)
    filled[last_seen < 0] = 0
    features = np.zeros((count, WIDTH), dtype=np.float32)
    features[:, :len(CHANNELS)] = np.column_stack((filled, ~observed[:, :-1], clocks))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "timeline.npy", features)
    np.savez(output / "entries.npz", timestamps=timestamps, ends=ends, labels=labels)
    (output / "manifest.json").write_text(json.dumps(SCHEMA, indent=2) + "\n", encoding="utf-8")
    return prepare_air(output)


def prepare_air(directory=AIR_DIR):
    root = Path(directory)
    paths = TimelinePaths(str(root / "timeline.npy"), str(root / "entries.npz"), str(root / "manifest.json"))
    schema = json.loads(Path(paths.manifest).read_text(encoding="utf-8"))
    if schema != SCHEMA:
        raise ValueError("Unexpected Beijing air-quality schema")
    timeline = mmap_array(paths.timeline, "air-quality timeline", (WIDTH,))
    with np.load(paths.entries, allow_pickle=False) as entries:
        times, ends, labels = (entries[key] for key in ("timestamps", "ends", "labels"))
    if (times.shape != (len(timeline),) or times.dtype != np.int64 or ends.dtype != np.int64
            or ends.ndim != 1 or not len(ends) or ends[0] < AIR_STEPS - 1
            or ends[-1] + HORIZON >= len(timeline) or not np.all(np.diff(ends) > 0)
            or labels.shape != ends.shape or not np.isin(labels, (0, 1)).all()
            or not np.all(np.diff(times) == 3600) or not np.isfinite(timeline).all()):
        raise ValueError("Invalid air-quality windows or labels")
    eligible_hours = len(timeline) - (AIR_STEPS - 1) - HORIZON
    report = {"accepted": len(ends), "candidate_hours": eligible_hours,
              "excluded_for_missing_pm25": eligible_hours - len(ends), "rises": int(labels.sum()),
              "lookback_hours": AIR_STEPS, "horizon_hours": HORIZON}
    return Prepared(paths, schema, ends + 1, times[ends], times[ends - AIR_STEPS + 1],
                    times[ends + HORIZON], ends, labels, np.arange(len(ends), dtype=np.int64), report, None)
