# CS683 PA-1 Task 1  build file.
#
# The compiler flags are PINNED and identical for every stage so that all speedups
# are attributable to the techniques you implement, not to compiler options:
#
#   -O2                 realistic scalar optimization (inlining, register allocation)
#   -fno-tree-vectorize compiler auto-vectorization is OFF, so the SIMD stage is YOUR job
#   -mavx2 -mfma        enable the AVX2 + FMA intrinsics your conv_simd/optimized use
#
# Do NOT add -march=native or -ftree-vectorize: the autograder uses exactly these flags.
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

.PHONY: all run solution clean perf-stat mpki

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
# perf profiling -- uses src/perf_harness.cpp, NOT main.cpp.
#
# main.cpp always runs conv_naive as its baseline, even in single-stage mode,
# so anything built from it will have naive's work mixed into perf's numbers.
# perf_harness.cpp runs ONLY the stage you name -- no naive call, no smoke
# test, nothing else in the timed loop -- so these counters are clean.
# =============================================================================
PERF_MAIN     := src/perf_harness.cpp
PERF_CXXFLAGS := $(CXXFLAGS) -g -fno-omit-frame-pointer
PERF  ?= perf
STAGE ?= optimized     # naive | reorder | unroll | tile | simd | optimized
PH    ?= 2048
PW    ?= 2048
PK    ?= 3
PSEED ?= 1234

$(BIN)/perf_harness: $(STUDENT_SRC) $(PERF_MAIN) | $(BIN)
	$(CXX) $(PERF_CXXFLAGS) $^ -o $@ $(LDFLAGS)

# make perf-stat STAGE=tile   -- raw counters: instructions, L1D loads/misses,
# LLC loads/misses -- for exactly conv_$(STAGE), nothing else.
perf-stat: $(BIN)/perf_harness
	$(PERF) stat -e instructions,L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
		./$(BIN)/perf_harness $(STAGE) $(PH) $(PW) $(PK) $(PSEED)

# make mpki STAGE=tile   -- L1-D MPKI = (L1-dcache-load-misses / instructions) * 1000
mpki: $(BIN)/perf_harness
	@$(PERF) stat -e instructions,L1-dcache-load-misses -x, -o /tmp/mpki_$(STAGE).csv \
		-- ./$(BIN)/perf_harness $(STAGE) $(PH) $(PW) $(PK) $(PSEED) >/dev/null
	@awk -F, '/instructions/{ins=$$1} /L1-dcache-load-misses/{miss=$$1} \
		END{printf "stage=%s H=%s W=%s K=%s  instructions=%s  L1D-load-misses=%s  MPKI=%.4f\n", \
		"$(STAGE)","$(PH)","$(PW)","$(PK)",ins,miss,(miss/ins)*1000}' /tmp/mpki_$(STAGE).csv