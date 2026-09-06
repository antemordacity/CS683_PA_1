#!/usr/bin/env python3
"""parse_perf_csv.py  Aggregate raw `perf stat -x, -o <file>` output (one file per
run, named "<label>_<size>.csv" or "<label>_<size>_<n2>.csv") into one tidy CSV.

`perf stat -x,` emits one line per event: "value,unit,event,...". Some events may be
unavailable on a given CPU/perf build and show up as "<not counted>" or similar; those
are left blank in the output rather than crashing the aggregation.

Usage:
  python3 parse_perf_csv.py <raw_dir> <out_csv>
"""
import csv
import glob
import os
import sys

FIELDS = [
    "instructions", "cycles",
    "l1d_loads", "l1d_misses",
    "llc_loads", "llc_misses",
    "l2_refs", "l2_misses",
]

EVENT_MAP = {
    "instructions": "instructions",
    "cycles": "cycles",
    "l1d_loads": "L1-dcache-loads",
    "l1d_misses": "L1-dcache-load-misses",
    "llc_loads": "LLC-loads",
    "llc_misses": "LLC-load-misses",
    "l2_refs": "l2_rqsts.references",
    "l2_misses": "l2_rqsts.miss",
}


def parse_one(path):
    vals = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            raw_val, event = parts[0], parts[2]
            try:
                vals[event] = float(raw_val)
            except ValueError:
                pass  # e.g. "<not counted>" / "<not supported>"
    return vals


def main():
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <raw_dir> <out_csv>", file=sys.stderr)
        sys.exit(1)
    raw_dir, out_csv = sys.argv[1], sys.argv[2]

    files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    rows = []
    for path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        if base.startswith("_"):
            continue  # metadata files (e.g. _events.txt-like markers)
        # label is everything except the trailing "_<size>"
        if "_" not in base:
            continue
        label, size = base.rsplit("_", 1)
        vals = parse_one(path)
        row = {"label": label, "size": size}
        for f in FIELDS:
            row[f] = vals.get(EVENT_MAP[f], "")
        rows.append(row)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "size"] + FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
