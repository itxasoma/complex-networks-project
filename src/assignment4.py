"""
assignment4.py: Configuration Model (CM) random network generation and analysis.

Uses the degree sequence from the GCC (Assignment 1) to:
  1. Generate a single CM realization and compute its topological properties.
  2. Generate 100 CM realizations and compute ensemble averages.

CM algorithm: stub-matching with random shuffle.
  - Create 2E stubs: node i contributes D[i] stubs.
  - Shuffle stubs randomly.
  - Pair stubs sequentially: (stub[0], stub[1]), (stub[2], stub[3]), ...
  - Discard self-loops and multi-edges (simple graph enforcement).

Outputs (results/):
  a4_comparison.csv   — scalar metrics: original vs CM single vs CM ensemble
  a4_ensemble.csv     — per-realization scalar values for all 100 runs
"""

import csv
import os
import time

import networkx as nx
import numpy as np

try:
    import community as community_louvain
except ImportError:
    raise ImportError("Install with: pip install python-louvain")


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS    = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

N_REALIZATIONS = 100


# ── 1. Load & clean ───────────────────────────────────────────────────────────
def load_edgelist(path):
    edges = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            edges.add((min(u, v), max(u, v)))
    return edges


edge_file = os.path.join(REPO_ROOT, "lastfm_asia", "lastfm_asia_edges.csv")
raw_edges = load_edgelist(edge_file)

G_full = nx.Graph()
G_full.add_edges_from(raw_edges)
G_full.remove_nodes_from(list(nx.isolates(G_full)))

gcc_nodes = max(nx.connected_components(G_full), key=len)
G_orig    = G_full.subgraph(gcc_nodes).copy()
mapping   = {n: i for i, n in enumerate(sorted(G_orig.nodes()))}
G_orig    = nx.relabel_nodes(G_orig, mapping)

N = G_orig.number_of_nodes()
E = G_orig.number_of_edges()
print(f"Original GCC  N={N}  E={E}")

D_orig = np.array([d for _, d in sorted(G_orig.degree())], dtype=np.int64)
print(f"Degree sequence: min={D_orig.min()}  max={D_orig.max()}  "
      f"mean={D_orig.mean():.4f}  sum={D_orig.sum()}")


# ── 2. Configuration Model generator ─────────────────────────────────────────
def configuration_model(degree_seq, rng):
    """
    Generate a simple undirected CM random graph from degree_seq.

    Algorithm (stub matching):
      1. Build stub list: node i repeated D[i] times.
      2. Shuffle stubs uniformly at random.
      3. Pair consecutive stubs: edge (stubs[2k], stubs[2k+1]).
      4. Discard self-loops (u == v) and multi-edges.

    Returns a nx.Graph. The resulting graph may have slightly lower
    degrees than requested because discarded stubs are not re-matched
    (erased CM). This is standard practice.
    """
    stubs = np.repeat(np.arange(len(degree_seq), dtype=np.int64), degree_seq)
    rng.shuffle(stubs)

    edge_set = set()
    for k in range(0, len(stubs) - 1, 2):
        u, v = int(stubs[k]), int(stubs[k + 1])
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in edge_set:
            continue
        edge_set.add(key)

    G = nx.Graph()
    G.add_nodes_from(range(len(degree_seq)))
    G.add_edges_from(edge_set)
    return G


# ── 3. Analysis function ──────────────────────────────────────────────────────
def analyse(G):
    """
    Compute all A2 topological properties on graph G using the
    pointer-based adjacency structure from Assignment 1.
    Returns a dict of scalar metrics and distribution arrays.
    """
    N_ = G.number_of_nodes()
    E_ = G.number_of_edges()
    if E_ == 0:
        return None

    edge_list = list(G.edges())
    D = np.zeros(N_, dtype=np.int64)
    for u, v in edge_list:
        D[u] += 1
        D[v] += 1

    P1 = np.zeros(N_, dtype=np.int64)
    for i in range(1, N_):
        P1[i] = P1[i-1] + D[i-1]
    P2 = P1.copy()

    V = np.zeros(2 * E_, dtype=np.int64)
    for u, v in edge_list:
        V[P2[u]] = v;  P2[u] += 1
        V[P2[v]] = u;  P2[v] += 1

    adj_set = [set(V[P1[i]:P1[i] + D[i]]) for i in range(N_)]

    k_max  = int(D.max())
    k_avg  = float(D.mean())
    k2_avg = float((D**2).mean())

    # P(k) and CCDF
    nk = np.zeros(k_max + 1, dtype=np.int64)
    for i in range(N_):
        nk[D[i]] += 1
    Pk = nk / N_
    Pc = np.zeros(k_max + 1)
    Pc[k_max] = Pk[k_max]
    for k in range(k_max - 1, -1, -1):
        Pc[k] = Pc[k + 1] + Pk[k]

    # k_nn(k)
    knn_acc = np.zeros(k_max + 1)
    for i in range(N_):
        ki = D[i]
        if ki == 0:
            continue
        knn_acc[ki] += sum(int(D[V[pos]]) for pos in range(P1[i], P1[i] + ki)) / ki
    knn = np.zeros(k_max + 1)
    for k in range(1, k_max + 1):
        if nk[k] > 0:
            knn[k] = knn_acc[k] / nk[k]

    # c(k)
    ck_acc = np.zeros(k_max + 1)
    for i in range(N_):
        ki = int(D[i])
        if ki < 2:
            continue
        neighbours = V[P1[i]: P1[i] + ki]
        tri_i = sum(
            1 for a in range(ki)
            for b in range(a + 1, ki)
            if int(neighbours[b]) in adj_set[int(neighbours[a])]
        )
        ck_acc[ki] += tri_i / (ki * (ki - 1) / 2)
    ck = np.zeros(k_max + 1)
    for k in range(2, k_max + 1):
        if nk[k] > 0:
            ck[k] = ck_acc[k] / nk[k]
    c_avg = float(ck_acc.sum()) / N_

    r = nx.degree_assortativity_coefficient(G)

    partition  = community_louvain.best_partition(G, resolution=1.0, random_state=42)
    modularity = community_louvain.modularity(partition, G)
    n_comm     = len(set(partition.values()))

    return dict(
        N=N_, E=E_, k_avg=k_avg, k2_avg=k2_avg, k_max=k_max,
        c_avg=c_avg, r=r, modularity=modularity, n_comm=n_comm,
        knn_uncorr=k2_avg / k_avg,
        nk=nk, Pk=Pk, Pc=Pc, knn=knn, ck=ck,
    )


# ── 4. Original network ───────────────────────────────────────────────────────
print("\nAnalysing original network …")
t0       = time.time()
orig_res = analyse(G_orig)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  <k>={orig_res['k_avg']:.4f}  <c>={orig_res['c_avg']:.6f}  "
      f"r={orig_res['r']:.4f}  Q={orig_res['modularity']:.4f}  "
      f"n_comm={orig_res['n_comm']}")


# ── 5. Single CM realization ──────────────────────────────────────────────────
print("\nGenerating single CM realization …")
rng   = np.random.default_rng(42)
G_cm1 = configuration_model(D_orig, rng)
E_cm1 = G_cm1.number_of_edges()
print(f"  CM1  N={G_cm1.number_of_nodes()}  E={E_cm1}  "
      f"(discarded {E - E_cm1} edges as self-loops/multi-edges)")

t0      = time.time()
cm1_res = analyse(G_cm1)
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  <k>={cm1_res['k_avg']:.4f}  <c>={cm1_res['c_avg']:.6f}  "
      f"r={cm1_res['r']:.4f}  Q={cm1_res['modularity']:.4f}  "
      f"n_comm={cm1_res['n_comm']}")


# ── 6. Ensemble of 100 CM realizations ────────────────────────────────────────
print(f"\nGenerating {N_REALIZATIONS} CM realizations …")
scalar_keys = ["k_avg", "k2_avg", "c_avg", "r", "modularity", "n_comm",
               "knn_uncorr", "E"]
ensemble = {k: [] for k in scalar_keys}

t0 = time.time()
for i in range(N_REALIZATIONS):
    G_cm = configuration_model(D_orig, rng)
    res  = analyse(G_cm)
    if res is None:
        continue
    for k in scalar_keys:
        ensemble[k].append(res[k])
    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N_REALIZATIONS}  ({time.time()-t0:.0f}s elapsed)")

print(f"  Total time: {time.time()-t0:.1f}s")

ens_mean = {k: np.mean(v) for k, v in ensemble.items()}
ens_std  = {k: np.std(v)  for k, v in ensemble.items()}

print("\nEnsemble averages (mean ± std):")
for k in scalar_keys:
    print(f"  {k:<14} {ens_mean[k]:.4f} ± {ens_std[k]:.4f}")


# ── 7. Comparison table ───────────────────────────────────────────────────────
print("\n── Comparison: Original vs CM single vs CM ensemble ──")
header = f"{'Metric':<16} {'Original':>12} {'CM single':>12} {'CM mean':>12} {'CM std':>10}"
print(header)
print("-" * len(header))
rows = [
    ("E",         orig_res["E"],         cm1_res["E"],         ens_mean["E"],         ens_std["E"]),
    ("<k>",       orig_res["k_avg"],      cm1_res["k_avg"],     ens_mean["k_avg"],     ens_std["k_avg"]),
    ("<k²>",      orig_res["k2_avg"],     cm1_res["k2_avg"],    ens_mean["k2_avg"],    ens_std["k2_avg"]),
    ("<c>",       orig_res["c_avg"],      cm1_res["c_avg"],     ens_mean["c_avg"],     ens_std["c_avg"]),
    ("r",         orig_res["r"],          cm1_res["r"],         ens_mean["r"],         ens_std["r"]),
    ("Q",         orig_res["modularity"], cm1_res["modularity"],ens_mean["modularity"],ens_std["modularity"]),
    ("n_comm",    orig_res["n_comm"],     cm1_res["n_comm"],    ens_mean["n_comm"],    ens_std["n_comm"]),
    ("<k²>/<k>",  orig_res["knn_uncorr"],cm1_res["knn_uncorr"],ens_mean["knn_uncorr"],ens_std["knn_uncorr"]),
]
for label, vo, vc, vm, vs in rows:
    print(f"  {label:<14} {vo:>12.4f} {vc:>12.4f} {vm:>12.4f} {vs:>10.4f}")


# ── 8. Save CSVs ──────────────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "a4_comparison.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "original", "cm_single", "cm_mean", "cm_std"])
    for label, vo, vc, vm, vs in rows:
        w.writerow([label, round(float(vo), 4), round(float(vc), 4),
                    round(float(vm), 4), round(float(vs), 4)])

with open(os.path.join(RESULTS, "a4_ensemble.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(scalar_keys)
    for i in range(len(ensemble["E"])):
        w.writerow([round(ensemble[k][i], 4) for k in scalar_keys])

print("\nDone. Results saved to", RESULTS)
print("Run assignment4-plots.py to generate figures.")