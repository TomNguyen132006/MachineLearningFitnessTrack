"""Generate synthetic MetaMotion CSVs with the same file names and columns as the real dataset.

The real MetaMotion recordings are not included in this repository. This script creates a
stand-in dataset so the full pipeline can run: 5 participants (A-E), 5 barbell exercises
(3 heavy sets of 5 reps and 3 medium sets of 10 reps each), plus standing and sitting rest.
Each set is a sine wave at the rep rate plus Gaussian noise, so results computed from it
do not reflect real-world performance.

The output is deterministic (fixed random seed and fixed timezone), so every run produces
identical files.

Usage (from any directory):
    python src/data/generate_synthetic_data.py [output_dir]
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "raw" / "MetaMotion"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

# Recording timezone (UTC-6), fixed so the timestamps do not depend on the machine's timezone
TZ = timezone(timedelta(hours=-6))

# Relative signal strength on the x, y, z axes for each exercise
exercises = {"bench": (0.0, 1.0, 0.1), "squat": (0.9, 0.2, 0.3), "row": (0.1, 0.3, 0.9),
             "ohp": (0.6, 0.8, 0.0), "dead": (0.3, 0.1, 1.0)}
t = datetime(2019, 1, 11, 15, 0, 0, tzinfo=TZ)

# make_dataset.py reads this exact file name from the real dataset, so this set keeps it
RENAMED = {"A-bench-heavy2-rpe8": "2019-01-11T16.10.08.270"}


def write(name, start, dur, sensor, hz, sig):
    n = int(dur * hz)
    epoch = (start.timestamp() * 1000 + np.arange(n) * 1000 / hz).astype(np.int64)
    el = np.arange(n) / hz
    unit = "g" if sensor == "Accelerometer" else "deg/s"
    df = pd.DataFrame({"epoch (ms)": epoch,
                       "time (01:00)": [datetime.fromtimestamp(e / 1000, timezone.utc).strftime("%Y-%m-%dT%H.%M.%S.%f")[:-3] for e in epoch],
                       "elapsed (s)": el.round(3)})
    for i, ax in enumerate("xyz"):
        df[f"{ax}-axis ({unit})"] = sig(el, i) + rng.normal(0, 0.05 if unit == "g" else 5, n)
    ts = RENAMED.get(name, start.strftime("%Y-%m-%dT%H.%M.%S.000"))
    hzs = "12.500Hz" if sensor == "Accelerometer" else "25.000Hz"
    df.to_csv(out / f"{name}_MetaWear_{ts}_C42732BE255C_{sensor}_{hzs}_1.4.4.csv", index=False)


for p in "ABCDE":
    for ex, w in exercises.items():
        for cat, reps in (("heavy", 5), ("medium", 10)):
            for s in (1, 2, 3):
                period = 2.5 if cat == "heavy" else 2.0
                dur = reps * period + 2
                name = f"{p}-{ex}-{cat}{s}-rpe8"
                acc = lambda e, i: w[i] * np.sin(2 * np.pi * e / period) + (1.0 if i == 1 else 0)
                gyr = lambda e, i: 40 * w[(i + 1) % 3] * np.cos(2 * np.pi * e / period)
                write(name, t, dur, "Accelerometer", 12.5, acc)
                write(name, t, dur, "Gyroscope", 25.0, gyr)
                t += timedelta(minutes=2)
    for cat in ("standing", "sitting"):
        name = f"{p}-rest-{cat}"
        write(name, t, 30, "Accelerometer", 12.5, lambda e, i: (1.0 if i == 1 else 0) + 0 * e)
        write(name, t, 30, "Gyroscope", 25.0, lambda e, i: 0 * e)
        t += timedelta(minutes=2)

print(f"Wrote {len(os.listdir(out))} files to {out}")
