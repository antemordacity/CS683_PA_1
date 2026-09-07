# Matmul optimization analysis toolkit

Drop this whole `analysis/` folder in as a **sibling of `src/` and `include/`** in your
repo (i.e. `your-repo/analysis/...`, next to `your-repo/src/` and `your-repo/include/`).
Nothing here touches `src/`, `include/`, `main.cpp`, or the `Makefile` — it only reads
your real `matmul_naive.cpp` / `matmul_simd.cpp` / `matmul_prefetch.cpp` /
`matmul_optimized.cpp` and compiles its own analysis-only binaries against them.

```
your-repo/
├── Makefile
├── include/        (matmul.h, timer.h, utils.h)
├── src/             (matmul_naive.cpp, matmul_simd.cpp, matmul_prefetch.cpp,
│                      matmul_optimized.cpp, main.cpp  ← your graded submission)
└── analysis/        ← this folder
```

## What's in here

| File | Purpose |
|---|---|
| `bench_common.h` | Shared timing/correctness helper used by every `bench_*.cpp` driver. |
| `bench_size.cpp` | Task 1: speedup vs. matrix size, all three stages vs. naive. |
| `matmul_prefetch_param.cpp` | Analysis-only copy of `matmul_prefetch.cpp` with distance/hint as compile-time `-D`s. **Not graded, not `src/matmul_prefetch.cpp`.** |
| `bench_prefetch_distance.cpp`, `bench_prefetch_hint.cpp` | Task 2 / Task 3 drivers, link against the param copy above. |
| `matmul_simd_sse.cpp`, `matmul_simd_avx512.cpp` | Analysis-only 128-bit / 512-bit variants of your SIMD kernel (same 4-column register-tile structure as `src/matmul_simd.cpp`, just a different vector width), so width is the only thing that changes. **Not graded.** |
| `bench_simd_width.cpp` | Task 4 driver: SSE + your real AVX2 `src/matmul_simd.cpp` + AVX-512 linked into one binary. |
| `bench_perf_workload.cpp` | Minimal fixed-iteration harness meant to be wrapped by `perf stat`/`perf record` (no internal timer, so perf counts pure compute). |
| `parse_perf_csv.py` | Aggregates raw `perf stat -x,` output files into one tidy CSV. |
| `run_*.sh` | One script per experiment — build + run + write `results/*.csv`. |
| `plot_results.py` | Reads every `results/*.csv` and writes every required plot to `results/plots/`. Skips a plot with a clear message if its CSV isn't there yet, so you can run it after each step or once at the end. |

## Exact run order

All commands assume `cd analysis/` first, on an x86_64 Linux machine.

```bash
cd analysis

# --- Part 1: speedup vs. matrix size (all 3 stages vs. naive) --------------------
./run_size_sweep.sh
# custom sizes, e.g. to skip the slowest naive runs while iterating:
#   ./run_size_sweep.sh 128 256 512 1024 2048

# --- Part 2: speedup vs. prefetch distance ---------------------------------------
./run_prefetch_distance_sweep.sh                       # defaults: 1 2 4 8 16 32 64 128, sizes 512/1024/2048
# custom: ./run_prefetch_distance_sweep.sh "1 4 16 64" "512 1024"
# -> look at results/prefetch_distance.csv / the plot for the distance that peaks

# --- Part 3: speedup vs. prefetch locality hint -----------------------------------
# pass the best distance you found above so the two analyses use the same distance
./run_prefetch_hint_sweep.sh 8                          # replace 8 with your best distance

# --- Part 4: speedup vs. SIMD width (SSE / AVX2 / AVX-512) -----------------------
./run_simd_width_sweep.sh
# skips AVX-512 automatically (with a note) if this CPU doesn't support AVX512F

# --- perf: cache-miss / instruction-count trends ----------------------------------
# requires `perf`; if missing:
#   sudo apt-get install linux-tools-common linux-tools-$(uname -r) linux-tools-generic
# if you get "Permission denied":
#   sudo sysctl -w kernel.perf_event_paranoid=-1      (or run the script with sudo)
./run_perf_metrics_size.sh                              # L1D/L2/LLC + instructions/cycles vs. size, per stage
./run_perf_metrics_prefetch.sh                          # same metrics vs. prefetch distance

# --- hardware prefetcher on/off (Intel + root + bare metal only) ------------------
# many VMs/containers block MSR access entirely -- that's expected, not a bug here.
sudo ./run_hw_prefetch_toggle.sh optimized 1024 5

# --- plot everything ---------------------------------------------------------------
python3 plot_results.py
# -> results/plots/*.png
```

`results/` and `results/plots/` are created automatically; re-running any script
overwrites its own CSV (each script owns exactly one output file, so partial re-runs
don't clobber unrelated results).

## Mapping outputs to the assignment's required plots/analyses

- **1. Speedup vs. matrix size** → `results/plots/01_speedup_vs_size.png` (+ `01b_gflops_vs_size.png` as supporting evidence).
- **2. Speedup vs. prefetch distance** → `02_speedup_vs_prefetch_distance.png`.
- **3. Speedup vs. prefetch level (T0/T1/T2/NTA)** → `03_speedup_vs_prefetch_hint.png`.
- **4. Speedup vs. SIMD width** → `04_speedup_vs_simd_width.png`.
- **Cache-miss trends (L1D/L2/LLC) vs. size and vs. prefetch params** → `05_*_vs_size.png`, `06_*_vs_prefetch_distance.png`. The distance where these bottom out is your answer to "optimal prefetch distance"; the *rate at which extra distance stops helping/starts hurting* (e.g. evicting useful lines too early, or issuing more prefetch instructions than the core can absorb) is worth calling out explicitly in the report.
- **Instruction count vs. SIMD width/size** → the `instructions` column in `results/perf_metrics_size.csv` (this script doesn't currently sweep SIMD width through perf — if you want that too, copy `run_perf_metrics_size.sh`'s pattern using the three widths built the same way `run_simd_width_sweep.sh` does, and add a `07_instructions_vs_width.png` — I kept this out by default since it's a straightforward extension of an existing script rather than new machinery, and the assignment marks it as optional table-filling.)
- **Hardware prefetcher on/off** → `results/hw_prefetch_toggle.csv`.
- **Final comparison (best of each technique)** → `07_final_comparison.png`. **Important**: this plot reads `results/size_sweep.csv`'s existing `prefetch`/`simd`/`optimized` columns, i.e. whatever distance/hint/width is *currently hardcoded in your actual `src/matmul_prefetch.cpp` and `src/matmul_simd.cpp`*. If your distance/hint/width sweeps find a better setting than what's currently in those files, update them and re-run `./run_size_sweep.sh` before generating this final plot, or it'll compare against a stale configuration.

## Notes / known constraints

- **Runtime**: `matmul_naive` is O(n³) scalar; a single 4096³ call can take on the order
  of a minute or more depending on the machine. `bench_size.cpp` already reduces
  timed repetitions at larger sizes (see `reps_for_size` in `bench_common.h`) to keep
  this tolerable, but the largest sizes will still be the slowest part of the whole
  toolkit by far. Shrink the size list first if you're iterating.
- **AVX-512**: `run_simd_width_sweep.sh` checks `/proc/cpuinfo` for `avx512f` and skips
  that variant with a clear note if unsupported — the SSE/AVX2 comparison still runs.
- **perf events**: the L2-specific raw event (`l2_rqsts.miss`) is Intel-specific and not
  guaranteed to exist on every CPU/perf build; the scripts check for it and leave those
  CSV columns blank rather than failing if it's missing. L1D/LLC use perf's portable
  generalized hardware-cache event names and should work on both Intel and AMD.
- **Hardware-prefetcher toggle**: Intel-only (MSR `0x1A4`), needs root and often needs
  bare metal — most cloud VMs and containers block MSR access by design. If it fails
  with a permission error, that's the platform, not a bug; there's no software
  workaround for that from inside the guest.
- **`matmul_prefetch_param.cpp` staleness**: it's a **manually kept-in-sync** mirror of
  `src/matmul_prefetch.cpp`'s structure (tile size, loop order), with only the prefetch
  distance/hint parameterized. If you change anything else about the real prefetch
  kernel later, update this file too, or the distance/hint sweep will be characterizing
  a kernel shape you no longer submit.
