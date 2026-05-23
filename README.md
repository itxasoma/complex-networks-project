# Complex Networks — Structural Analysis

A clean Python implementation of **structural network analysis** for the *Complex Systems* course (MASM, 2025–2026). The pipeline downloads a real-world network, preprocesses it into an unweighted undirected simple graph, extracts the Giant Connected Component (GCC), and computes all Assignment 1 statistics.

Network used: **ca-HepTh** (arXiv High-Energy Physics – Theory collaboration), sourced from [SNAP](https://snap.stanford.edu/data/ca-HepTh.html).  
After GCC extraction: **6 951 nodes · 41 197 edges**.

---

## Repository structure

```
complex-networks/
├── src/
│   ├── load.py          # Download, parse edge list, remove self-loops & multi-edges
│   ├── gcc.py           # Extract GCC, report isolates, build index mapping
│   ├── stats.py         # Degree sequence, k_avg, k_max, P(k), moments
│   ├── plots.py         # Degree distribution plots (lin-lin, log-log, log-lin)
│   └── utils.py         # Shared helpers (timer, path constants)
├── data/                # Raw and cleaned edge lists (generated at runtime)
├── results/             # Output figures and CSV statistics (generated at runtime)
├── Makefile
├── LICENSE
└── README.md
```

---

## Requirements

### Python ≥ 3.9

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib pandas scipy networkx requests
```

---

## How to run

### Full pipeline (download → clean → GCC → stats → plots)

```bash
make
```

### Step by step

```bash
make download   # fetch raw edge list into data/
make clean_net  # remove self-loops, duplicates → data/gcc_edgelist.txt
make stats      # compute degree statistics → results/stats.csv
make plots      # generate degree-distribution figures → results/
```

### Clean generated files

```bash
make clean
```

To restart from scratch (including downloaded data):

```bash
make distclean
```

---

## Preprocessing pipeline

All steps follow the Assignment 1 specification:

1. **Load edge list** — read pairs $(u, v)$; ignore weights if present.  
2. **Remove self-loops** — drop any edge where $u = v$.  
3. **Remove multi-edges** — for each unordered pair $\{u, v\}$ keep at most one edge; sort $(u, v)$ with $u < v$ and deduplicate.  
4. **Node label → node index** — remap original labels $\ell$ to contiguous integers $0, 1, \ldots, N-1$; store the bijection in `data/label_to_idx.json`.  
5. **Report degree-zero nodes** — count and list nodes with $k_i = 0$ before GCC extraction; exclude them from all statistics.  
6. **Extract GCC** — keep only the largest connected component; record $N_\text{GCC}$ and $E_\text{GCC}$.  
7. **Adjacency structure** — build degree vector $\mathbf{k}$ and neighbour list of total length $2E$ with pointer array, consistent with the lecture-note data structure.

---

## Statistics computed (Assignment 1)

| Quantity | Symbol | Description |
|---|---|---|
| Number of nodes (GCC) | $N$ | After isolate removal |
| Number of edges (GCC) | $E$ | Simple, undirected |
| Average degree | $\langle k \rangle = 2E/N$ | First moment |
| Degree variance | $\langle k^2 \rangle - \langle k \rangle^2$ | Second central moment |
| Maximum degree | $k_\text{max}$ | Hub identification |
| Degree distribution | $P(k)$ | Empirical, normalised |

Degree distribution is plotted on three scales: linear–linear, log–log, and log–linear, to identify power-law or exponential behaviour.

---

## Key results

| Quantity | Value |
|---|---|
| $N$ (GCC) | 6 951 |
| $E$ (GCC) | 41 197 |
| $\langle k \rangle$ | ≈ 11.86 |
| $k_\text{max}$ | (computed at runtime) |

---

## References

- A.-L. Barabási, *Network Science*, Cambridge University Press (2016)  
- M. E. J. Newman, *Networks: An Introduction*, Oxford University Press (2010)  
- J. Leskovec, A. Krevl, *SNAP Datasets: Stanford Large Network Dataset Collection*, [snap.stanford.edu/data](https://snap.stanford.edu/data) (2014)  
- M. Á. Serrano, *Complex Networks I — Structural properties and models*, MASM lecture notes (2025–2026)

---

## Author

**Itxaso Muñoz-Aldalur** — MASM, Universitat de Barcelona / Universitat Politècnica de Catalunya, 2025–2026.
