# Complex Networks — Final Project

Python and Fortran implementation of structural network analysis and SIS spreading dynamics for the *Complex Networks* course (2025–2026).

Network used: **LastFM Asia** social network, sourced from [SNAP / Benedek Rozemberczki](https://snap.stanford.edu/data/feather-lastfm-social.html).  
After GCC extraction: **7 624 nodes · 27 806 edges**.

Full results and discussion are available in the [project report](REPORT_COMPLEX_NETWORKS.pdf).

***

## Repository structure

```
complex-networks-project/
├── part1/
│   ├── src/
│   │   ├── assignment1.py        # Pointer-based adjacency structure, degree sequence
│   │   ├── assignment2.py        # P(k), CCDF, k_nn(k), c(k) + all plots
│   │   ├── assignment3.py        # Louvain community detection + community plots
│   │   ├── assignment4.py        # Configuration Model generation and comparison
│   │   ├── assignment4-plots.py  # CM comparison plots
│   │   ├── network_utils.py      # Shared utilities
│   │   └── mplstyle/
│   │       └── science.mplstyle  # Matplotlib style
│   └── results/                  # Output figures (PDF) and CSV summaries (generated)
├── part2/
│   ├── src/
│   │   ├── sis_lifespan_cm.f90   # Gillespie SIS simulation on CM power-law networks
│   │   ├── scaling2.py           # FSS analysis: peak extraction, exponent fitting
│   │   ├── plots2.py             # All Part 2 figures
│   │   ├── run_part2_scan.sh     # Shell script to launch parameter scans
│   │   ├── run_part2.slm         # SLURM job script
│   │   ├── run_lc_window_g35.slm # SLURM script for refined λ window scan (γ=3.5)
│   │   ├── submit_scan.slm       # SLURM array submission script
│   │   └── submit_scan_window.slm
│   ├── bin/                      # Compiled Fortran binary (generated)
│   ├── figures/                  # Output figures (generated)
│   └── results/                  # CSV summaries (generated)
├── lastfm_asia.zip               # Raw edge list
├── Makefile
├── REPORT_COMPLEX_NETWORKS.pdf
├── LICENSE
└── README.md
```

***

## Requirements

### Part 1 — Python ≥ 3.9

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy matplotlib networkx python-louvain pandas scipy
```

> `python-louvain` provides the `community` package used in Assignment 3.

### Part 2 — gfortran

```bash
gfortran -O2 -o part2/bin/sis_lifespan_cm part2/src/sis_lifespan_cm.f90
```

Simulations were run on a SLURM cluster using the `.slm` job scripts in `part2/src/`. For local runs, use `run_part2_scan.sh` directly.

***

## How to run

### Part 1 — Structural analysis

```bash
make        # full pipeline: runs all four assignments
make a1     # structural metrics only (no plots)
make a2     # degree distribution, k_nn(k), c(k) + all plots
make a3     # Louvain community detection + community plots
make a4     # Configuration Model generation and comparison
make clean  # remove generated files
```

### Part 2 — SIS dynamics

```bash
# Compile
gfortran -O2 -o part2/bin/sis_lifespan_cm part2/src/sis_lifespan_cm.f90

# Run scans (edit parameters in the script as needed)
bash part2/src/run_part2_scan.sh

# FSS analysis and figures
cd part2/src
python3 scaling2.py
python3 plots2.py
```

***

## Pipeline overview

### Part 1 — Structural analysis of the LastFM Asia network

**Assignment 1** builds the pointer-based adjacency structure (`D`, `P1`, `P2`, `V`) described in the lecture notes, preprocesses the raw edge list (self-loops, multi-edges, GCC extraction), and computes global structural metrics.

**Assignment 2** computes and plots $P(k)$, $P_c(k)$, $k_{\mathrm{nn}}(k)$, and $\bar{c}(k)$ using manual implementations (no NetworkX shortcuts for core quantities).

**Assignment 3** detects community structure using the Louvain algorithm (greedy modularity maximisation) and produces network visualisations coloured by community membership.

**Assignment 4** generates an ensemble of 100 Configuration Model realisations preserving the degree sequence, computes all structural metrics, and compares them to the real network to isolate higher-order correlations beyond the degree sequence.

### Part 2 — SIS spreading dynamics on power-law networks

Simulates the SIS model via the Gillespie algorithm on Configuration Model networks with $P(k) \propto k^{-\gamma}$, $k_{\min}=4$, for $\gamma \in \{3.5,\ 2.5\}$ at seven system sizes from $N=10^4$ to $10^6$.

The lifespan method extracts the epidemic threshold $\lambda_c$ and critical exponents $1/\nu$, $\beta/\nu$, and $\gamma_1/\nu$ via finite-size scaling of the average outbreak lifespan $\langle\tau(\lambda,N)\rangle$ and the endemic fraction $P_{\mathrm{end}}(\lambda,N)$.

***

## References

- M. E. J. Newman, *Networks: An Introduction*, Oxford University Press (2010)
- A.-L. Barabási, *Network Science*, Cambridge University Press (2016)
- V. D. Blondel et al., *Fast unfolding of communities in large networks*, J. Stat. Mech. (2008)
- A. S. Mata et al., *Lifespan method as a tool to study criticality in absorbing-state phase transitions*, Phys. Rev. E **91**, 052117 (2015)
- B. Rozemberczki, C. Allen, R. Sarkar, *Multi-Scale Attributed Node Embedding*, J. Complex Networks **9**, cnab014 (2021)
- M. Á. Serrano, *Complex Networks I & II — lecture notes*, MASM (2025–2026)

***

## Author

**Itxaso Muñoz-Aldalur** — Universitat de Barcelona 2025–2026.