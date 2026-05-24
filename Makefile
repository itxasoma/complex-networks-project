# Makefile: Complex Networks analysis pipeline
#
# Usage:
#   make        -> run all assignments (a1 → a2 → a3 → a4)
#   make a1     -> structural metrics (no plots)
#   make a2     -> degree distribution, k_nn(k), c(k) + all plots
#   make a3     -> Louvain community detection + community plots
#   make a4     -> CM random networks
#   make clean  -> remove results/

PYTHON = python3
SRC    = src
RES    = results

.PHONY: all a1 a2 a3 a4 a4-plots clean

all: a1 a2 a3 a4 a4-plots

a1:
	mkdir -p $(RES)
	$(PYTHON) $(SRC)/assignment1.py

a2: a1
	$(PYTHON) $(SRC)/assignment2.py

a3: a1
	$(PYTHON) $(SRC)/assignment3.py

a4: a1
	$(PYTHON) $(SRC)/assignment4.py

a4-plots: a4
	$(PYTHON) $(SRC)/assignment4_plots.py

clean:
	rm -rf $(RES)
