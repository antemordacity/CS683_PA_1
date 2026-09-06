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
ITERS ?= 10             # must match perf_harness's --iters -- see kDefaultIters

$(BIN)/perf_harness: $(STUDENT_SRC) $(PERF_MAIN) | $(BIN)
	$(CXX) $(PERF_CXXFLAGS) $^ -o $@ $(LDFLAGS)

# Comprehensive event list -- the perf "default" software/hardware set plus
# every standard cache-hierarchy event (L1D/L1I, LLC, dTLB/iTLB, each split
# into loads and stores where the CPU exposes it). Not every CPU/PMU supports
# every one of these -- unsupported events show up as "<not supported>" or
# "<not counted>" in the CSV rather than failing the whole run, so it's safe
# to ask for all of them and see what your hardware actually reports.
PERF_EVENTS := task-clock,context-switches,cpu-migrations,page-faults,cycles,ref-cycles,instructions,stalled-cycles-frontend,stalled-cycles-backend,branches,branch-misses,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses,L1-icache-load-misses,LLC-loads,LLC-load-misses,LLC-stores,LLC-store-misses,dTLB-loads,dTLB-load-misses,dTLB-stores,dTLB-store-misses,iTLB-loads,iTLB-load-misses

# make perf-stat STAGE=tile ITERS=10
# Raw counters from perf attach to the WHOLE process, which runs conv_$(STAGE)
# ITERS times back-to-back (nothing else). Prints every counter perf reports
# (total + per-run average), then derives IPC and MPKI for each load/store
# cache level perf gave us data for.
perf-stat: $(BIN)/perf_harness
	@$(PERF) stat -e $(PERF_EVENTS) \
		-x, -o /tmp/perfstat_$(STAGE).csv \
		-- ./$(BIN)/perf_harness $(STAGE) $(PH) $(PW) $(PK) $(PSEED) --iters=$(ITERS)
	@echo "--- per-run average over $(ITERS) iterations (stage=$(STAGE)) ---"
	@awk -F, -v n=$(ITERS) ' \
		{ \
			val=$$1; ev=$$3; gsub(/ /,"",ev); \
			if (val !~ /^[0-9.]+$$/) { printf "%-28s %s\n", ev, "<not supported>"; next } \
			printf "%-28s total=%-16s avg/run=%.2f\n", ev, val, val/n; \
			counts[ev]=val; \
		} \
		END{ \
			printf "\n--- derived metrics ---\n"; \
			if (counts["cycles"]>0) \
				printf "%-28s %.3f\n", "IPC (instr/cycle)", counts["instructions"]/counts["cycles"]; \
			if (counts["instructions"]>0) { \
				if ("L1-dcache-load-misses" in counts) \
					printf "%-28s %.4f\n", "L1D-load MPKI",  counts["L1-dcache-load-misses"]/counts["instructions"]*1000; \
				if ("L1-dcache-store-misses" in counts) \
					printf "%-28s %.4f\n", "L1D-store MPKI", counts["L1-dcache-store-misses"]/counts["instructions"]*1000; \
				if ("L1-icache-load-misses" in counts) \
					printf "%-28s %.4f\n", "L1I-load MPKI",  counts["L1-icache-load-misses"]/counts["instructions"]*1000; \
				if ("LLC-load-misses" in counts) \
					printf "%-28s %.4f\n", "LLC-load MPKI",  counts["LLC-load-misses"]/counts["instructions"]*1000; \
				if ("LLC-store-misses" in counts) \
					printf "%-28s %.4f\n", "LLC-store MPKI", counts["LLC-store-misses"]/counts["instructions"]*1000; \
				if ("dTLB-load-misses" in counts) \
					printf "%-28s %.4f\n", "dTLB-load MPKI", counts["dTLB-load-misses"]/counts["instructions"]*1000; \
				if ("dTLB-store-misses" in counts) \
					printf "%-28s %.4f\n", "dTLB-store MPKI",counts["dTLB-store-misses"]/counts["instructions"]*1000; \
				if ("iTLB-load-misses" in counts) \
					printf "%-28s %.4f\n", "iTLB-load MPKI", counts["iTLB-load-misses"]/counts["instructions"]*1000; \
			} \
			if (("L1-dcache-loads" in counts) && ("L1-dcache-load-misses" in counts) && counts["L1-dcache-loads"]>0) \
				printf "%-28s %.4f%%\n", "L1D-load miss rate", counts["L1-dcache-load-misses"]/counts["L1-dcache-loads"]*100; \
			if (("LLC-loads" in counts) && ("LLC-load-misses" in counts) && counts["LLC-loads"]>0) \
				printf "%-28s %.4f%%\n", "LLC-load miss rate", counts["LLC-load-misses"]/counts["LLC-loads"]*100; \
		}' /tmp/perfstat_$(STAGE).csv

# make mpki STAGE=tile ITERS=10
# L1-D MPKI = (L1-dcache-load-misses / instructions) * 1000, computed from the
# process totals -- MPKI is already a ratio, so it doesn't need dividing by
# ITERS separately, but per-run instruction/miss counts are also printed.
# Also reports IPC (instructions per cycle) -- perf only computes this
# derived metric in its human-readable table, not in -x/CSV mode, so we pull
# in "cycles" alongside instructions and compute IPC ourselves here.
mpki: $(BIN)/perf_harness
	@$(PERF) stat -e instructions,cycles,L1-dcache-load-misses -x, -o /tmp/mpki_$(STAGE).csv \
		-- ./$(BIN)/perf_harness $(STAGE) $(PH) $(PW) $(PK) $(PSEED) --iters=$(ITERS) >/dev/null
	@awk -F, -v n=$(ITERS) '/instructions/{ins=$$1} /cycles/{cyc=$$1} /L1-dcache-load-misses/{miss=$$1} \
		END{printf "stage=%s H=%s W=%s K=%s iters=%d\n  instructions(total)=%s (avg/run=%.0f)\n  cycles(total)=%s (avg/run=%.0f)\n  IPC=%.3f\n  L1D-load-misses(total)=%s (avg/run=%.0f)\n  MPKI=%.4f\n", \
		"$(STAGE)","$(PH)","$(PW)","$(PK)",n,ins,ins/n,cyc,cyc/n,ins/cyc,miss,miss/n,(miss/ins)*1000}' /tmp/mpki_$(STAGE).csv