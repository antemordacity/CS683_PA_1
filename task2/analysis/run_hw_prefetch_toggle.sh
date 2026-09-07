#!/usr/bin/env bash
# run_hw_prefetch_toggle.sh  Studies the effect of Intel's hardware prefetchers on
# matmul performance by toggling MSR 0x1A4 (Intel SDM Vol. 4, "MSR_MISC_FEATURE_CONTROL"):
#   bit 0: disable L2 hardware prefetcher
#   bit 1: disable L2 adjacent-cache-line prefetcher
#   bit 2: disable DCU (L1 data) prefetcher
#   bit 3: disable DCU IP prefetcher
#
# REQUIREMENTS (all of these commonly fail in VMs/cloud/lab containers -- that is
# expected, not a bug in this script):
#   - an actual Intel CPU (checked below; AMD has no equivalent standard MSR)
#   - root (re-run with sudo)
#   - msr-tools installed:        sudo apt-get install msr-tools
#   - the msr kernel module:      sudo modprobe msr
#   - bare-metal or a VM that passes through MSR access (most cloud VMs and
#     containers block this outright -- if `rdmsr` reports "Operation not
#     permitted", there is nothing further to try short of a different machine)
#
# Usage:
#   sudo ./run_hw_prefetch_toggle.sh [stage] [size] [iters]
#   sudo ./run_hw_prefetch_toggle.sh optimized 1024 5
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results

if [[ $EUID -ne 0 ]]; then
    echo "Re-run with sudo: sudo $0 $*"
    exit 1
fi

if ! grep -qi "GenuineIntel" /proc/cpuinfo; then
    echo "This script's MSR 0x1A4 trick is Intel-specific."
    echo "On AMD there is no standard documented MSR for this; check your BIOS for"
    echo "a 'Hardware Prefetcher' setting instead, or consult AMD uProf / your CPU's"
    echo "model-specific register documentation."
    exit 1
fi

if ! command -v rdmsr >/dev/null 2>&1; then
    echo "install msr-tools: sudo apt-get install msr-tools"
    exit 1
fi
modprobe msr 2>/dev/null || true

CPU=0     # toggle + pin the benchmark to this logical CPU
MSR=0x1a4

ORIG=$(rdmsr -p $CPU $MSR 2>&1) || {
    echo "rdmsr failed: $ORIG"
    echo "MSR access is blocked on this machine (common in VMs/containers)."
    echo "This script needs bare-metal (or a VM configured to pass MSRs through)."
    exit 1
}
echo "original MSR $MSR on CPU $CPU: $ORIG"

STAGE=${1:-optimized}
SIZE=${2:-1024}
ITERS=${3:-5}

g++ -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include \
    bench_perf_workload.cpp matmul_prefetch_param.cpp \
    ../src/matmul_naive.cpp ../src/matmul_simd.cpp ../src/matmul_prefetch.cpp \
    ../src/matmul_optimized.cpp \
    -o /tmp/bench_perf

OUT=results/hw_prefetch_toggle.csv
echo "config,stage,M,N,K,iters,total_wall_ms,per_call_ms" > "$OUT"

run_once () {
    local label=$1
    local t0 t1 total per_call
    t0=$(date +%s.%N)
    taskset -c $CPU /tmp/bench_perf "$STAGE" "$SIZE" "$SIZE" "$SIZE" "$ITERS" >/dev/null
    t1=$(date +%s.%N)
    total=$(python3 -c "print(($t1-$t0)*1000)")
    per_call=$(python3 -c "print(($t1-$t0)*1000/$ITERS)")
    echo "$label,$STAGE,$SIZE,$SIZE,$SIZE,$ITERS,$total,$per_call" >> "$OUT"
    echo "$label: ${per_call} ms/call"
}

wrmsr -p $CPU $MSR 0x0   # all four documented prefetchers ON (typical default)
run_once "hw_prefetch_enabled"

wrmsr -p $CPU $MSR 0xF   # bits 0-3 set -> all four documented prefetchers OFF
run_once "hw_prefetch_disabled"

wrmsr -p $CPU $MSR "$ORIG"   # restore whatever was there before this script ran
echo "restored original MSR value ($ORIG)"
echo "wrote $OUT"
