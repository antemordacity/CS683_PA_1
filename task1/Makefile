# CS683 PA-1 Task 1  build file.
#
# The compiler flags are PINNED and identical for every stage so that all speedups
# are attributable to the techniques you implement, not to compiler options:
#
#   -O2                 realistic scalar optimization (inlining, register allocation)
#   -fno-tree-vectorize compiler auto-vectorization is OFF, so the SIMD stage is YOUR job
#   -mavx2 -mfma        enable the AVX2 + FMA intrinsics your conv_simd/optimized use
#
# Do NOT add -march=native or -ftree-vectorize: the autograder uses exactly these flags,
# and the assignment PDF explicitly deducts points for extra compiler optimizations
# beyond loop reorder/unroll, tiling, and SIMD (see "NOTE" on p.1 of the PA1 handout).
CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall
LDFLAGS  :=
BIN          := bin
MAIN         := src/main.cpp
STUDENT_SRC  := src/conv_naive.cpp src/conv_reorder.cpp src/conv_unroll.cpp \
                src/conv_tile.cpp src/conv_simd.cpp src/conv_optimized.cpp
# Instructor-only: reference implementations used to calibrate/regress the tiers.
SOLUTION_SRC := src/conv_naive.cpp solution/conv_reorder.cpp solution/conv_unroll.cpp \
                solution/conv_tile.cpp solution/conv_simd.cpp solution/conv_optimized.cpp

.PHONY: all run solution clean \
        perf perf-record perf-report perf-annotate \
        perf-cache mpki instructions cacheinfo perf-sweep-mpki perf-sweep-speedup

all: $(BIN)/conv

# Student build: edit src/conv_*.cpp, then `make` and `./bin/conv`.
$(BIN)/conv: $(STUDENT_SRC) $(MAIN) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

run: $(BIN)/conv
	./$(BIN)/conv

# Instructor build against solution/ (kept out of the student handout).
solution: $(BIN)/conv_solution
$(BIN)/conv_solution: $(SOLUTION_SRC) $(MAIN) | $(BIN)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN) perf.data perf.data.old *.csv

# =============================================================================
# perf profiling  (Task 1B: L1-D MPKI vs tile size/matrix size;
#                  Task 1C: instruction count, naive vs SIMD)
# =============================================================================
#
# Same pinned flags as the graded build, plus -g and -fno-omit-frame-pointer so
# perf can unwind stacks and attribute samples to functions/lines. Neither flag
# changes codegen vs. the graded build -- this is a separate binary, conv_perf,
# so bin/conv (what gets graded) is never touched by profiling.
PERF_CXXFLAGS := $(CXXFLAGS) -g -fno-omit-frame-pointer
PERF  ?= perf
STAGE ?= optimized     # naive | reorder | unroll | tile | simd | optimized
PH    ?= 2048
PW    ?= 2048
PK    ?= 3
PSEED ?= 1234

$(BIN)/conv_perf: $(STUDENT_SRC) $(MAIN) | $(BIN)
	$(CXX) $(PERF_CXXFLAGS) $^ -o $@ $(LDFLAGS)

# --- sampling profiler (hotspot location within one stage) -----------------
#
# IMPORTANT: main.cpp (the provided, do-not-modify harness) always runs
# conv_naive as its correctness/speedup baseline even in single-stage mode, so
# every recording below contains BOTH conv_naive and conv_$(STAGE) samples.
# perf attributes samples per-function-symbol, so --symbol-filter reliably
# isolates STAGE's own samples from naive's in the *report* step; it does NOT
# mean perf skipped profiling naive during *record*. For a true single-stage
# aggregate count (mpki, instructions, cache-*), see the counting targets
# further below, which use marked regions instead of the sampling profiler.

# make perf-record STAGE=simd
perf-record: $(BIN)/conv_perf
	$(PERF) record -g -o perf.data -- ./$(BIN)/conv_perf $(STAGE) $(PH) $(PW) $(PK) $(PSEED)

# make perf-report STAGE=simd   (plain text, no TUI, no tips.txt nag)
perf-report:
	$(PERF) report -i perf.data --symbol-filter=conv_$(STAGE) --stdio

# make perf-annotate STAGE=simd  -> per-instruction sample attribution inside conv_simd
perf-annotate:
	$(PERF) annotate -i perf.data --stdio "conv_$(STAGE)"

# --- perf stat: quick process-wide counters ---------------------------------
# NOTE: process-wide, so naive+STAGE combined (see caveat above). Useful for a
# fast sanity check or for comparing relative *changes* across stages, but not
# a clean isolated number. Use the counting targets below for that.
perf: $(BIN)/conv_perf
	$(PERF) stat ./$(BIN)/conv_perf $(STAGE) $(PH) $(PW) $(PK) $(PSEED)

PERF_CACHE_EVENTS := cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses,dTLB-loads,dTLB-load-misses

perf-cache: $(BIN)/conv_perf
	$(PERF) stat -e $(PERF_CACHE_EVENTS) ./$(BIN)/conv_perf $(STAGE) $(PH) $(PW) $(PK) $(PSEED)

# --- Task 1B: L1-D MPKI (misses per kilo instructions) ----------------------
# MPKI = (L1-dcache-load-misses / instructions) * 1000
# Emits CSV via -x, so we can awk the two numbers straight out; also prints
# the raw perf stat table for the record. Same naive+STAGE caveat applies --
# report MPKI per STAGE and compare relative trends across tile sizes/matrix
# sizes as the handout asks, rather than treating this as a hermetic number.
mpki: $(BIN)/conv_perf
	@$(PERF) stat -e instructions,L1-dcache-load-misses \
		-x, -o /tmp/mpki_$(STAGE).csv \
		-- ./$(BIN)/conv_perf $(STAGE) $(PH) $(PW) $(PK) $(PSEED) >/dev/null
	@awk -F, '/instructions/{ins=$$1} /L1-dcache-load-misses/{miss=$$1} \
		END{printf "stage=%s H=%s W=%s K=%s  instructions=%s  L1D-load-misses=%s  MPKI=%.4f\n", \
		"$(STAGE)","$(PH)","$(PW)","$(PK)",ins,miss,(miss/ins)*1000}' /tmp/mpki_$(STAGE).csv

# --- Task 1C: raw instruction count (naive vs SIMD) -------------------------
instructions: $(BIN)/conv_perf
	$(PERF) stat -e instructions ./$(BIN)/conv_perf $(STAGE) $(PH) $(PW) $(PK) $(PSEED)

# --- find your L1-D cache size (Task 1B step 1) -----------------------------
cacheinfo:
	@echo "--- lscpu cache summary ---"
	@lscpu | grep -i -E 'L1d|L1i|L2|L3' || true
	@echo "--- getconf sizes (bytes) ---"
	@getconf -a 2>/dev/null | grep -i -E 'LEVEL1_DCACHE|LEVEL1_ICACHE|LEVEL2_CACHE|LEVEL3_CACHE' || true

# --- convenience sweeps for the required plots ------------------------------
# make perf-sweep-mpki STAGE=tile SIZES="256 512 1024 2048"
SIZES ?= 256 512 1024 2048
perf-sweep-mpki: $(BIN)/conv_perf
	@for s in $(SIZES); do \
		$(MAKE) --no-print-directory mpki STAGE=$(STAGE) PH=$$s PW=$$s PK=$(PK) PSEED=$(PSEED); \
	done

# make perf-sweep-speedup STAGE=simd SIZES="256 512 1024 2048"
perf-sweep-speedup: $(BIN)/conv_perf
	@for s in $(SIZES); do \
		echo "== H=W=$$s K=$(PK) =="; \
		./$(BIN)/conv_perf $(STAGE) $$s $$s $(PK) $(PSEED); \
	done