# Makefile: Complex Networks analysis pipeline
#
# Usage:
#   make            -> run all Part 1 assignments (a1 -> a2 -> a3 -> a4)
#   make a1         -> structural metrics (no plots)
#   make a2         -> degree distribution, k_nn(k), c(k) + plots
#   make a3         -> Louvain community detection + community plots
#   make a4         -> CM random networks
#   make run_test   -> quick Part 2 validation run
#   make run2       -> Part 2 production run
#   make binning    -> optional Part 2 post-processing if binning.py exists
#   make clean      -> remove results and Part 2 binary

PYTHON = python3
FC     = gfortran
FFLAGS = -O0 -std=f2008 -Wall -Wextra

SRC1   = part1/src
RES1   = part1/results
SRC2   = part2/src
RES2   = part2/results
BIN2   = part2/bin
FIG2   = part2/figures
EXE2   = $(BIN2)/sis_lifespan_cm
F90_2  = $(SRC2)/sis_lifespan_cm.f90

# Default Part 2 parameters, can be overridden from command line:
# make run2 N=30000 GAMMA=3.5 NRUNS=1000 LMIN=0.02 LMAX=0.10 NLAM=25
N      ?= 100000
GAMMA  ?= 3.5
NRUNS  ?= 2000
LMIN   ?= 0.02
LMAX   ?= 0.10
NLAM   ?= 25
OUT2   ?= $(RES2)/raw/part2_N$(N)_g$(GAMMA).dat

.PHONY: all a1 a2 a3 a4 a4-plots part2-build run_test run2 binning clean

all: a1 a2 a3 a4 a4-plots

a1:
	mkdir -p $(RES1)
	$(PYTHON) $(SRC1)/assignment1.py

a2: a1
	$(PYTHON) $(SRC1)/assignment2.py

a3: a1
	$(PYTHON) $(SRC1)/assignment3.py

a4: a1
	$(PYTHON) $(SRC1)/assignment4.py

a4-plots: a4
	$(PYTHON) $(SRC1)/assignment4-plots.py

build2:
	mkdir -p $(BIN2) $(RES2)/raw $(RES2)/processed $(FIG2)
	$(FC) $(FFLAGS) -o $(EXE2) $(F90_2)

runtest2: build2
	$(EXE2) 1000 3.5 50 0.02 0.08 5 $(RES2)/raw/test_g35_N1000.dat

run2: build2
	$(EXE2) $(N) $(GAMMA) $(NRUNS) $(LMIN) $(LMAX) $(NLAM) $(OUT2)

binning:
	@if [ -f $(SRC2)/binning.py ]; then \
		mkdir -p $(RES2)/processed $(FIG2); \
		$(PYTHON) $(SRC2)/binning.py; \
	else \
		echo "No $(SRC2)/binning.py found; skipping binning."; \
	fi

clean:
	rm -rf *.mod