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

.PHONY: all run solution clean perf perf-record perf-report

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

# ---- perf profiling ---------------------------------------------------------
# Same pinned optimization flags as the graded build, plus -g and
# -fno-omit-frame-pointer so `perf` can unwind stacks and attribute samples to
# functions/lines. Neither flag changes codegen from the graded build.
PERF_CXXFLAGS := $(CXXFLAGS) -g -fno-omit-frame-pointer
PERF  ?= perf
STAGE ?= optimized     # which conv_* stage to profile: naive|reorder|unroll|tile|simd|optimized
PH    ?= 2048
PW    ?= 2048
PK    ?= 3
PSEED ?= 1234

$(BIN)/conv_perf: $(STUDENT_SRC) $(MAIN) | $(BIN)
	$(CXX) $(PERF_CXXFLAGS) $^ -o $@ $(LDFLAGS)

# make perf STAGE=simd
# make perf STAGE=tile PH=512 PW=512 PK=5
# Uses the harness's single-stage mode, so only naive + STAGE run under perf --
# no overhead/noise from profiling all six stages in one shot.
perf: $(BIN)/conv_perf
	$(PERF) stat ./$(BIN)/conv_perf $(STAGE) $(PH) $(PW) $(PK) $(PSEED)

# make perf-record STAGE=optimized
perf-record: $(BIN)/conv_perf
	$(PERF) record -g -o perf.data -- ./$(BIN)/conv_perf $(STAGE) $(PH) $(PW) $(PK) $(PSEED)

perf-report:
	$(PERF) report -i perf.data

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN) perf.data perf.data.old