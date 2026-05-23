# Makefile — Complex Networks structural analysis (Python)
# Usage:
#   make              -> full pipeline (download, clean, stats, plots)
#   make download     -> fetch raw edge list into data/
#   make clean_net    -> remove self-loops/multi-edges, extract GCC
#   make stats        -> compute degree statistics
#   make plots        -> generate degree-distribution figures
#   make clean        -> remove results/ and data/gcc* files
#   make distclean    -> remove ALL generated files including raw data

PYTHON  = python3
SRCDIR  = src
DATADIR = data
RESDIR  = results

.PHONY: all download clean_net stats plots clean distclean

all: plots

download:
	mkdir -p $(DATADIR)
	$(PYTHON) $(SRCDIR)/load.py

clean_net: download
	$(PYTHON) $(SRCDIR)/gcc.py

stats: clean_net
	mkdir -p $(RESDIR)
	$(PYTHON) $(SRCDIR)/stats.py

plots: stats
	$(PYTHON) $(SRCDIR)/plots.py

clean:
	rm -rf $(RESDIR)
	rm -f $(DATADIR)/gcc_edgelist.txt $(DATADIR)/label_to_idx.json $(DATADIR)/degree_sequence.npy

distclean: clean
	rm -rf $(DATADIR)
