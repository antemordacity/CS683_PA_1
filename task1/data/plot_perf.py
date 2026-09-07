#!/usr/bin/env python3
"""
plot_perf.py — grouped bar chart of perf metrics normalized to a "naive" baseline.

Each input stat file looks like:

    stage=optimized H=2048 W=2048 K=11 seed=1234 iters=10  time(median)=493.064ms  GFLOP/s=2.06
    --- per-run average over 10 iterations (stage=optimized) ---
                                 <not supported>
    task-clock                   total=1271.87          avg/run=127.19
    ...
    --- derived metrics ---
    IPC (instr/cycle)            2.592
    L1D-load MPKI                13.2977
    ...

USAGE
-----
Single matrix size (all files are one group, naive.txt required as baseline):

    python3 plot_perf.py --metric gflops naive.txt unroll.txt reorder.txt tile.txt simd.txt optimized.txt

Multiple matrix sizes (one directory per size, each containing one stat file per
stage including a naive baseline file):

    python3 plot_perf.py --metric gflops size1/ size2/ size3/ size4/

Discover every metric name available in your files:

    python3 plot_perf.py --list-metrics naive.txt

Any metric printed by --list-metrics can be passed to --metric, e.g.:

    --metric gflops
    --metric ipc
    --metric l1d_mpki          (alias -> l1d_load_mpki)
    --metric l1i_mpki          (alias -> l1i_load_mpki)
    --metric llc_load_mpki
    --metric llc_miss_rate     (alias -> llc_load_miss_rate)
    --metric l1d_miss_rate     (alias -> l1d_load_miss_rate)
    --metric cache_misses
    --metric branch_misses
    --metric time              (alias -> time_median_ms)

Every bar shows stage_value / naive_value for that group (so 1.0 == the naive
baseline, drawn as a solid horizontal reference line). A value like 0.93
therefore renders as a bar dipping BELOW that reference line, exactly like a
value above 1.0 rises above it. Use --invert for metrics where *lower* is
better (e.g. MPKI, miss rates, time) if you want improvements to point up too.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Known stage ordering / labels / colors, matching the reference figure style.
# Anything not listed here still works -- it just falls back to a generic
# label and the default matplotlib color cycle.
# ----------------------------------------------------------------------------
STAGE_ORDER = ["unroll", "reorder", "tile", "simd", "optimized"]

STAGE_LABELS = {
    "unroll": "Loop unrolling",
    "reorder": "Loop reordering",
    "tile": "Tiling",
    "simd": "SIMD",
    "optimized": "Optimized (all)",
}

STAGE_COLORS = {
    "unroll": "#4472C4",
    "reorder": "#ED7D31",
    "tile": "#A5A5A5",
    "simd": "#FFC000",
    "optimized": "#5B9BD5",
}

# Friendly shorthands -> normalized metric key produced by the parser below.
METRIC_ALIASES = {
    "gflops": "gflops",
    "gflop_s": "gflops",
    "flops": "gflops",
    "time": "time_median_ms",
    "time_ms": "time_median_ms",
    "runtime": "time_median_ms",
    "ipc": "ipc",
    "l1d_mpki": "l1d_load_mpki",
    "l1_mpki": "l1d_load_mpki",
    "l1i_mpki": "l1i_load_mpki",
    "llc_mpki": "llc_load_mpki",
    "dtlb_mpki": "dtlb_load_mpki",
    "itlb_mpki": "itlb_load_mpki",
    "l1d_miss_rate": "l1d_load_miss_rate",
    "l1_miss_rate": "l1d_load_miss_rate",
    "llc_miss_rate": "llc_load_miss_rate",
}


def normalize_key(name: str) -> str:
    """Turn a raw label like 'L1D-load MPKI' or 'IPC' into 'l1d_load_mpki' / 'ipc'."""
    name = re.sub(r"\(.*?\)", "", name)  # drop "(instr/cycle)" etc.
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


HEADER_RE = re.compile(
    r"stage=(\S+)\s+H=(\d+)\s+W=(\d+)\s+K=(\d+)\s+seed=(\d+)\s+iters=(\d+)\s+"
    r"time\(median\)=([\d.]+)ms\s+GFLOP/s=([\d.]+)"
)
AVG_RUN_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_\-/]*)\s+total=([-\d.]+)\s+avg/run=([-\d.]+)")
DERIVED_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_\-\(\)/ ]*?)\s{2,}([-\d.]+)\s*%?\s*$")


def parse_stat_file(path: Path) -> dict:
    """Parse one stat file into a flat dict of normalized_metric_name -> float."""
    text = path.read_text()
    data = {}

    m = HEADER_RE.search(text)
    if m:
        data["stage"] = m.group(1)
        data["h"] = int(m.group(2))
        data["w"] = int(m.group(3))
        data["k"] = int(m.group(4))
        data["seed"] = int(m.group(5))
        data["iters"] = int(m.group(6))
        data["time_median_ms"] = float(m.group(7))
        data["gflops"] = float(m.group(8))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or "not supported" in line:
            continue

        m2 = AVG_RUN_RE.match(line)
        if m2:
            data[normalize_key(m2.group(1))] = float(m2.group(3))
            continue

        m3 = DERIVED_RE.match(line)
        if m3:
            data[normalize_key(m3.group(1))] = float(m3.group(2))
            continue

    return data


def resolve_metric(user_metric: str, sample: dict) -> str:
    """Map a user-supplied metric string to an actual key present in `sample`."""
    key = normalize_key(user_metric)
    key = re.sub(r"^(relative_|normalized_|rel_)", "", key)
    key = METRIC_ALIASES.get(key, key)

    if key in sample:
        return key

    candidates = sorted(k for k in sample if key in k or k in key)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        sys.exit(f"Metric '{user_metric}' is ambiguous. Candidates: {candidates}")
    sys.exit(
        f"Unknown metric '{user_metric}'.\nAvailable metrics:\n  "
        + "\n  ".join(sorted(sample))
    )


def collect_group_files(dir_path: Path):
    return sorted(f for f in dir_path.iterdir() if f.is_file())


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Plot a perf metric, normalized to a naive baseline, across optimization stages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Either stat files (one group / matrix size) or directories "
        "(one directory per matrix size, each holding one stat file per stage).",
    )
    ap.add_argument("--metric", default="gflops", help="Metric to plot (default: gflops). See --list-metrics.")
    ap.add_argument("--baseline-stage", default="naive", help="Stage name used as the 1.0x baseline (default: naive).")
    ap.add_argument(
        "--invert",
        action="store_true",
        help="Plot baseline/value instead of value/baseline. Use for metrics where "
        "lower is better (MPKI, miss rates, time) so improvements still point up.",
    )
    ap.add_argument("--out", default="perf_comparison.png", help="Output image path.")
    ap.add_argument("--title", default=None, help="Custom chart title.")
    ap.add_argument("--list-metrics", action="store_true", help="Print all metric names found and exit.")
    return ap


def main():
    args = build_arg_parser().parse_args()
    paths = [Path(p) for p in args.inputs]

    dirs_mode = all(p.is_dir() for p in paths)
    files_mode = all(p.is_file() for p in paths)
    if not dirs_mode and not files_mode:
        sys.exit("Pass either all directories (one per matrix size) or all files (one group), not a mix.")

    if dirs_mode:
        raw_groups = [(p.name, collect_group_files(p)) for p in paths]
    else:
        raw_groups = [(None, paths)]

    parsed_groups = []
    for label, files in raw_groups:
        stage_data = {}
        for f in files:
            d = parse_stat_file(f)
            if "stage" in d:
                stage_data[d["stage"]] = d
        if args.baseline_stage not in stage_data:
            sys.exit(
                f"Group '{label or files}' is missing baseline stage "
                f"'{args.baseline_stage}'. Found stages: {sorted(stage_data)}"
            )
        if label is None:
            base = stage_data[args.baseline_stage]
            label = f'{base.get("h", "?")}x{base.get("w", "?")}'
        parsed_groups.append((label, stage_data))

    sample = next(iter(parsed_groups[0][1].values()))

    if args.list_metrics:
        print("Available metrics:")
        for k in sorted(sample):
            print(" ", k)
        return

    metric_key = resolve_metric(args.metric, sample)

    all_stages = []
    for _, stage_data in parsed_groups:
        for s in stage_data:
            if s != args.baseline_stage and s not in all_stages:
                all_stages.append(s)
    ordered_stages = [s for s in STAGE_ORDER if s in all_stages] + [
        s for s in all_stages if s not in STAGE_ORDER
    ]

    n_groups = len(parsed_groups)
    n_stages = len(ordered_stages)
    x = np.arange(n_groups)
    width = 0.8 / max(n_stages, 1)

    fig, ax = plt.subplots(figsize=(max(6, 2.2 * n_groups + 3), 5))

    all_values = []
    for i, stage in enumerate(ordered_stages):
        values = []
        for _, stage_data in parsed_groups:
            base_val = stage_data[args.baseline_stage].get(metric_key)
            stage_val = stage_data.get(stage, {}).get(metric_key)
            if not base_val or stage_val is None:
                values.append(np.nan)
                continue
            ratio = stage_val / base_val
            if args.invert:
                ratio = 1.0 / ratio
            values.append(ratio)

        all_values.extend(v for v in values if not np.isnan(v))
        offset = (i - (n_stages - 1) / 2) * width
        heights = [(v - 1.0) if not np.isnan(v) else 0.0 for v in values]
        ax.bar(
            x + offset,
            heights,
            width,
            bottom=1.0,
            label=STAGE_LABELS.get(stage, stage.replace("_", " ").title()),
            color=STAGE_COLORS.get(stage),
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )

    # Baseline reference line acts as the chart's visual "x-axis": bars below
    # 1.0 (e.g. a 0.93x result) render dipping below this line.
    ax.axhline(1.0, color="black", linewidth=1.2, zorder=3)

    if all_values:
        vmin, vmax = min(all_values + [1.0]), max(all_values + [1.0])
        pad = max((vmax - vmin) * 0.15, 0.05)
        ax.set_ylim(vmin - pad, vmax + pad)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in parsed_groups])
    if n_groups > 1:
        ax.set_xlabel("matrix size")

    metric_label = args.metric.replace("_", " ")
    direction = f"{args.baseline_stage} / value" if args.invert else f"value / {args.baseline_stage}"
    ax.set_ylabel(f"Normalized {metric_label}\n({direction})")
    ax.set_title(args.title or f"Normalized {metric_label} vs {args.baseline_stage} baseline")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
