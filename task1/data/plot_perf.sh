#!/usr/bin/env bash
#
# plot_perf.sh — thin wrapper around plot_perf.py
#
# Makes sure python3/matplotlib/numpy are available, then forwards all
# arguments straight to plot_perf.py.
#
# EXAMPLES
#   # one matrix size, files passed directly (naive.txt is the baseline)
#   ./plot_perf.sh --metric gflops naive.txt unroll.txt reorder.txt tile.txt simd.txt optimized.txt
#
#   # multiple matrix sizes, one directory per size
#   ./plot_perf.sh --metric gflops --out gflops.png size1/ size2/ size3/ size4/
#
#   # see every metric name you can pass to --metric
#   ./plot_perf.sh --list-metrics naive.txt
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/plot_perf.py"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but was not found on PATH." >&2
    exit 1
fi

if ! python3 -c "import matplotlib, numpy" >/dev/null 2>&1; then
    echo "Installing missing dependencies (matplotlib, numpy)..." >&2
    if ! python3 -m pip install --user matplotlib numpy >/dev/null 2>&1; then
        # Some distros (Debian/Ubuntu "externally managed" Python) need this flag.
        python3 -m pip install --user --break-system-packages matplotlib numpy
    fi
fi

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 [--metric NAME] [--out FILE.png] [--invert] [--list-metrics] <files...|dirs...>" >&2
    echo "Run '$0 --help' for full details." >&2
    exit 1
fi

python3 "${PY_SCRIPT}" "$@"
