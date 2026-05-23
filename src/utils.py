"""utils.py — shared helpers: paths, timer, logging."""

import time
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
DATADIR = ROOT / "data"
RESDIR  = ROOT / "results"

RAW_FILE   = DATADIR / "ca-HepTh.txt"
CLEAN_FILE = DATADIR / "gcc_edgelist.txt"
MAP_FILE   = DATADIR / "label_to_idx.json"
DEG_FILE   = DATADIR / "degree_sequence.npy"
STATS_CSV  = RESDIR  / "stats.csv"


class Timer:
    def __enter__(self):
        self._t = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._t
        print(f"  done in {self.elapsed:.2f}s")
