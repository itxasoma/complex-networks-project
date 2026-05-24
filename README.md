# Complex Networks — Structural Analysis

Python implementation of structural network analysis for the *Complex Systems* course (MASM, 2025–2026).  
The pipeline loads a real-world network, preprocesses it into an unweighted undirected simple graph, extracts the Giant Connected Component (GCC), and runs three analysis assignments.

Network used: **LastFM Asia** social network, sourced from [SNAP / Benedek Rozemberczki](https://snap.stanford.edu/data/feather-lastfm-social.html).  
After GCC extraction: **7 624 nodes · 27 806 edges**.

---

## Repository structure

```
complex-networks-project/
├── src/
│   ├── assignment1.py   # Pointer-based adjacency structure, degree sequence,
│   │                    # P(k), k_nn(k), c(k), assortativity, path length
│   ├── assignment2.py   # Manual P(k), CCDF, k_nn(k), c(k) + all plots
│   ├── assignment3.py   # Louvain community detection + community plots
│   ├── __init__.py
│   └── mplstyle/
│       └── science.mplstyle   # Shared Matplotlib style
├── lastfm_asia/
│   └── lastfm_asia_edges.csv  # Raw edge list (node_1, node_2)
├── results/             # Output figures (PDF) and CSV summaries (generated)
├── Makefile
├── LICENSE
└── README.md
```

---

## Requirements

Python ≥ 3.9

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib networkx python-louvain
```

> `python-louvain` provides the `community` package used in Assignment 3.

---

## How to run

### Full pipeline

```bash
make
```

Runs all three assignments in order and saves all figures and CSVs to `results/`.

### Individual assignments

```bash
make a1   # structural metrics only (no plots)
make a2   # degree distribution, k_nn(k), c(k) + all plots
make a3   # Louvain community detection + community plots
```

### Clean generated files

```bash
make clean
```

---

## Pipeline overview

### Assignment 1 — `assignment1.py`

Builds the pointer-based adjacency structure described in the lecture notes and computes all structural metrics. **No plots** — plotting is handled by Assignment 2.

**Data structures:**
- `D[i]` — degree vector of length $N$
- `P1[i]` — frozen first pointer: start of node $i$'s neighbour block in `V`
- `P2[i]` — write-head pointer: advances during the second pass
- `V` — flat neighbour list of length $2E$

**Metrics computed:**

| Quantity | Symbol |
|---|---|
| Nodes / edges (GCC) | $N$, $E$ |
| Average degree | $\langle k \rangle$ |
| Second moment | $\langle k^2 \rangle$ |
| Degree distribution | $P(k)$, $P_c(k)$ |
| Avg. nearest-neighbour degree | $k_{\mathrm{nn}}(k)$ |
| Clustering spectrum | $\bar{c}(k)$ |
| Degree assortativity | $r$ |
| Average path length / diameter | $\langle \ell \rangle$, $D$ |

### Assignment 2 — `assignment2.py`

Replicates the Assignment 1 data structures, then computes and plots all distributions using **manual implementations** (no NetworkX shortcuts for the core quantities).

**Plots produced (`results/`):**

| File | Content |
|---|---|
| `a1_degree_dist_linear.pdf` | $P(k)$ linear, with $\langle k \rangle$ reference |
| `a1_degree_dist_loglog.pdf` | $P(k)$ log-log |
| `a1_degree_dist_ccdf.pdf` | Complementary CDF $P_c(k)$ log-log |
| `a2_knn.pdf` | $k_{\mathrm{nn}}(k)$ with $\langle k^2 \rangle / \langle k \rangle$ reference |
| `a2_ck.pdf` | Clustering spectrum $\bar{c}(k)$ with $\langle c \rangle$ reference |

### Assignment 3 — `assignment3.py`

Detects the community structure of the GCC using the **Louvain algorithm** (greedy modularity maximisation). Communities are coloured with the `plasma_r` perceptually uniform scale, mapped on $\log(\text{size})$ for contrast even under a heavy-tailed size distribution.

**Key results:**

| Metric | Value |
|---|---|
| Communities | 28 |
| Modularity $Q$ | 0.8148 |
| Largest community | 1 103 nodes (14.5 %) |
| Intra-community edges | 91.2 % |

**Plots produced (`results/`):**

| File | Content |
|---|---|
| `a3_community_sizes.pdf` | Community sizes ranked, plasma_r colourbar |
| `a3_network_communities.pdf` | Spring-layout network coloured by community |
| `a3_edge_pie.pdf` | Intra- vs inter-community edge fractions |
| `a3_comm_degree_dists.pdf` | $P(k)$ for the 5 largest communities |

---

## Key results

| Quantity | Value |
|---|---|
| $N$ (GCC) | 7 624 |
| $E$ (GCC) | 27 806 |
| $\langle k \rangle$ | 7.2976 |
| $\langle k^2 \rangle$ | 112.5466 |
| $k_{\mathrm{max}}$ | 216 |
| $\langle c \rangle$ | 0.2194 |
| Assortativity $r$ | 0.1752 |
| $\langle \ell \rangle$ | ≈ 4.57 |
| Communities (Louvain) | 28 |
| Modularity $Q$ | 0.8148 |

---

## References

- A.-L. Barabási, *Network Science*, Cambridge University Press (2016)  
- V. D. Blondel et al., *Fast unfolding of communities in large networks*, J. Stat. Mech. (2008)  
- B. Rozemberczki, R. Sarkar, *Characteristic Functions on Graphs*, CIKM (2020)  
- M. Á. Serrano, *Complex Networks I & II — Structural properties, models, community structure*, MASM lecture notes (2025–2026)

---

## Author

**Itxaso Muñoz-Aldalur** — MASM, Universitat de Barcelona / Universitat Politècnica de Catalunya, 2025–2026.
