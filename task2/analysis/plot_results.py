#!/usr/bin/env python3
"""plot_results.py  Reads results/*.csv (produced by the run_*.sh scripts) and writes
every plot the assignment asks for into results/plots/. Skips a plot with a clear
warning if its input CSV isn't there yet, instead of crashing -- so you can run this
after each sweep script individually, or once at the end.

Usage:
  python3 plot_results.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "results")
PLOTS = os.path.join(RESULTS, "plots")


def load_csv(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        print(f"skip: {path} not found (run the matching sweep script first)")
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def savefig(fig, name):
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---- Task 1: speedup vs. matrix size ---------------------------------------------
def plot_size_sweep():
    rows = load_csv("size_sweep.csv")
    if rows is None:
        return
    by_stage = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(
            (int(r["M"]), float(r["speedup"]))
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    for stage in ("simd", "prefetch", "optimized"):
        if stage not in by_stage:
            continue
        pts = sorted(by_stage[stage])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=stage)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size (M=N=K)")
    ax.set_ylabel("Speedup vs. matmul_naive")
    ax.set_title("Speedup vs. matrix size")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "01_speedup_vs_size.png")

    # GFLOP/s vs size, all 4 stages including naive -- useful supporting plot
    by_stage_gf = {}
    for r in rows:
        by_stage_gf.setdefault(r["stage"], []).append(
            (int(r["M"]), float(r["gflops"]))
        )
    fig, ax = plt.subplots(figsize=(7, 5))
    for stage in ("naive", "simd", "prefetch", "optimized"):
        if stage not in by_stage_gf:
            continue
        pts = sorted(by_stage_gf[stage])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=stage)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size (M=N=K)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Absolute throughput vs. matrix size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "01b_gflops_vs_size.png")


# ---- Task 2: speedup vs. prefetch distance ----------------------------------------
def plot_prefetch_distance():
    rows = load_csv("prefetch_distance.csv")
    if rows is None:
        return
    by_size = {}
    for r in rows:
        by_size.setdefault(int(r["M"]), []).append(
            (int(r["distance"]), float(r["speedup"]))
        )

    fig, ax = plt.subplots(figsize=(7, 5))
    for size, pts in sorted(by_size.items()):
        pts = sorted(pts)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                label=f"{size}x{size}")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Prefetch distance (elements ahead)")
    ax.set_ylabel("Speedup vs. matmul_naive")
    ax.set_title("Speedup vs. prefetch distance")
    ax.legend(title="Matrix size")
    ax.grid(True, alpha=0.3)
    savefig(fig, "02_speedup_vs_prefetch_distance.png")


# ---- Task 3: speedup vs. prefetch locality hint -----------------------------------
def plot_prefetch_hint():
    rows = load_csv("prefetch_hint.csv")
    if rows is None:
        return
    hints = ["_MM_HINT_NTA", "_MM_HINT_T2", "_MM_HINT_T1", "_MM_HINT_T0"]
    by_size = {}
    for r in rows:
        by_size.setdefault(int(r["M"]), {})[r["hint"]] = float(r["speedup"])

    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.8 / max(len(by_size), 1)
    x_base = range(len(hints))
    for i, (size, vals) in enumerate(sorted(by_size.items())):
        ys = [vals.get(h, 0.0) for h in hints]
        xs = [x + i * width for x in x_base]
        ax.bar(xs, ys, width=width, label=f"{size}x{size}")
    ax.set_xticks([x + width * (len(by_size) - 1) / 2 for x in x_base])
    ax.set_xticklabels([h.replace("_MM_HINT_", "") for h in hints])
    ax.set_xlabel("Prefetch locality hint")
    ax.set_ylabel("Speedup vs. matmul_naive")
    ax.set_title("Speedup vs. prefetch locality hint")
    ax.legend(title="Matrix size")
    ax.grid(True, alpha=0.3, axis="y")
    savefig(fig, "03_speedup_vs_prefetch_hint.png")


# ---- Task 4: speedup vs. SIMD width -----------------------------------------------
def plot_simd_width():
    rows = load_csv("simd_width.csv")
    if rows is None:
        return
    by_width = {}
    for r in rows:
        if r["width"] == "naive":
            continue
        by_width.setdefault(r["width"], []).append(
            (int(r["M"]), float(r["speedup"]))
        )
    order = ["128", "256", "512"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for w in order:
        if w not in by_width:
            continue
        pts = sorted(by_width[w])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                label=f"{w}-bit")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size (M=N=K)")
    ax.set_ylabel("Speedup vs. matmul_naive")
    ax.set_title("Speedup vs. SIMD width")
    if "512" not in by_width:
        ax.text(0.5, 0.02, "AVX-512 not available on the machine this ran on",
                 transform=ax.transAxes, ha="center", fontsize=8, color="gray")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "04_speedup_vs_simd_width.png")


# ---- Cache-miss trend plots (perf) -------------------------------------------------
def _miss_rate(loads, misses):
    try:
        loads, misses = float(loads), float(misses)
        return misses / loads if loads > 0 else None
    except (TypeError, ValueError):
        return None


def plot_cache_misses_vs_size():
    rows = load_csv("perf_metrics_size.csv")
    if rows is None:
        return
    stages = ["naive", "simd", "prefetch", "optimized"]
    metrics = [
        ("l1d_loads", "l1d_misses", "L1D miss rate"),
        ("llc_loads", "llc_misses", "LLC miss rate"),
        ("l2_refs", "l2_misses", "L2 miss rate"),
    ]
    for loads_key, misses_key, title in metrics:
        fig, ax = plt.subplots(figsize=(7, 5))
        any_data = False
        for stage in stages:
            pts = []
            for r in rows:
                if r["label"] != stage:
                    continue
                rate = _miss_rate(r[loads_key], r[misses_key])
                if rate is not None:
                    pts.append((int(r["size"]), rate))
            if not pts:
                continue
            any_data = True
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=stage)
        if not any_data:
            print(f"skip: no data for {title} (event likely unsupported on that CPU)")
            plt.close(fig)
            continue
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Matrix size (M=N=K)")
        ax.set_ylabel("Miss rate (misses / loads)")
        ax.set_title(f"{title} vs. matrix size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fname = f"05_{misses_key}_vs_size.png"
        savefig(fig, fname)


def plot_cache_misses_vs_prefetch_distance():
    rows = load_csv("perf_metrics_prefetch.csv")
    if rows is None:
        return
    metrics = [
        ("l1d_loads", "l1d_misses", "L1D miss rate"),
        ("llc_loads", "llc_misses", "LLC miss rate"),
        ("l2_refs", "l2_misses", "L2 miss rate"),
    ]
    for loads_key, misses_key, title in metrics:
        pts = []
        for r in rows:
            if not r["label"].startswith("dist"):
                continue
            dist = int(r["label"][len("dist"):])
            rate = _miss_rate(r[loads_key], r[misses_key])
            if rate is not None:
                pts.append((dist, rate))
        if not pts:
            print(f"skip: no data for {title} vs distance")
            continue
        pts.sort()
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Prefetch distance (elements ahead)")
        ax.set_ylabel("Miss rate (misses / loads)")
        ax.set_title(f"{title} vs. prefetch distance")
        ax.grid(True, alpha=0.3)
        savefig(fig, f"06_{misses_key}_vs_prefetch_distance.png")


# ---- Final comparison: best of each technique across sizes ------------------------
def plot_final_comparison():
    rows = load_csv("size_sweep.csv")
    if rows is None:
        return
    # "prefetch" and "simd" columns in size_sweep.csv already reflect whatever
    # distance/hint/width you've settled on in src/matmul_prefetch.cpp and
    # src/matmul_simd.cpp -- re-run run_size_sweep.sh after tuning those files with
    # your best parameters so this plot reflects your best configuration, not
    # whatever was hardcoded when you first wrote them.
    by_stage = {}
    for r in rows:
        by_stage.setdefault(r["stage"], []).append(
            (int(r["M"]), float(r["speedup"]))
        )
    labels = {
        "prefetch": "Software Prefetching (best params)",
        "simd": "SIMD (best width)",
        "optimized": "Prefetching + SIMD (matmul_optimized)",
    }
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for stage, label in labels.items():
        if stage not in by_stage:
            continue
        pts = sorted(by_stage[stage])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size (M=N=K)")
    ax.set_ylabel("Speedup vs. matmul_naive")
    ax.set_title("Best performance by technique vs. matrix size")
    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend()
    ax.grid(True, alpha=0.3)
    savefig(fig, "07_final_comparison.png")


def main():
    plot_size_sweep()
    plot_prefetch_distance()
    plot_prefetch_hint()
    plot_simd_width()
    plot_cache_misses_vs_size()
    plot_cache_misses_vs_prefetch_distance()
    plot_final_comparison()


if __name__ == "__main__":
    main()
