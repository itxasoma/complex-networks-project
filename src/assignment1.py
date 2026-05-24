"""
assignment1.py: Structural characterisation of the network (GCC)
Builds the pointer-based adjacency structure and computes all metrics.
No plots. Plotting is handled by assignment2.py.
"""

import csv
import os

import networkx as nx
import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS    = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


# ── 1. Load & clean ───────────────────────────────────────────────────────────
def load_edgelist(path):
    edges, nodes_raw = set(), set()
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
            nodes_raw.update([u, v])
    return edges, nodes_raw

edge_file = os.path.join(REPO_ROOT, "lastfm_asia", "lastfm_asia_edges.csv")
raw_edges, raw_nodes = load_edgelist(edge_file)

G_full = nx.Graph()
G_full.add_edges_from(raw_edges)

isolates = list(nx.isolates(G_full))
print(f"Nodes with degree 0 (isolates): {len(isolates)}")
G_full.remove_nodes_from(isolates)

# ── 2. Giant Connected Component ─────────────────────────────────────────────
gcc_nodes = max(nx.connected_components(G_full), key=len)
G = G_full.subgraph(gcc_nodes).copy()

mapping = {n: i for i, n in enumerate(sorted(G.nodes()))}
G = nx.relabel_nodes(G, mapping)

N = G.number_of_nodes()
E = G.number_of_edges()
print(f"\nGCC  N={N}  E={E}")


# ── 3. Pointer-based adjacency structure ─────────────────────────────────────
# D[i]  : degree of node i                               (length N)
# P1[i] : first pointer  – start of node i's block in V  (frozen)
# P2[i] : second pointer – current write-head            (advances in pass 2)
# V     : flat neighbour list                            (length 2E)

gcc_edge_list = list(G.edges())

# Pass 1: count degrees
D = np.zeros(N, dtype=np.int64)
for u, v in gcc_edge_list:
    D[u] += 1
    D[v] += 1

# Initialise pointers (P1 = exclusive prefix sum of D; P2 = copy of P1)
P1 = np.zeros(N, dtype=np.int64)
for i in range(1, N):
    P1[i] = P1[i-1] + D[i-1]
P2 = P1.copy()

# Pass 2: fill V
V = np.zeros(2 * E, dtype=np.int64)
for u, v in gcc_edge_list:
    V[P2[u]] = v;  P2[u] += 1
    V[P2[v]] = u;  P2[v] += 1

assert np.all(P2 == P1 + D), "Pointer check failed"
assert int(D.sum()) == 2 * E,  "Degree sum ≠ 2E"

# Adjacency sets for O(1) edge look-up (used by triangle counting)
adj_set = [set(V[P1[i]:P1[i] + D[i]]) for i in range(N)]


# ── 4. Basic metrics ──────────────────────────────────────────────────────────
k_max  = int(D.max())
k_min  = int(D.min())
k_avg  = float(D.mean())
k2_avg = float((D**2).mean())

print(f"\n{'node':>8}  {'degree':>8}")
for i in range(N):
    print(f"{i:>8}  {int(D[i]):>8}")

print(f"\n<k>   = {k_avg:.4f}")
print(f"<k²>  = {k2_avg:.4f}")
print(f"kmax  = {k_max}   kmin = {k_min}")
print(f"Density = {2*E/(N*(N-1)):.6f}")


# ── 5. Degree distribution P(k) and complementary CDF P_c(k) ─────────────────
nk = np.zeros(k_max + 1, dtype=np.int64)
for i in range(N):
    nk[D[i]] += 1

Pk = nk / N

Pc = np.zeros(k_max + 1)
Pc[k_max] = Pk[k_max]
for k in range(k_max - 1, -1, -1):
    Pc[k] = Pc[k + 1] + Pk[k]

assert abs(float(Pk.sum()) - 1.0) < 1e-9, "P(k) does not sum to 1"
assert abs(float(Pc[0])   - 1.0) < 1e-9, "P_c(0) should equal 1"

print(f"\nSum P(k) = {Pk.sum():.6f}  (should be 1.0)")
print(f"P_c(0)   = {Pc[0]:.6f}  (should be 1.0)")


# ── 6. Average nearest-neighbour degree k_nn(k) ──────────────────────────────
knn_acc = np.zeros(k_max + 1)
for i in range(N):
    ki = D[i]
    if ki == 0:
        continue
    neigh_deg_sum = sum(int(D[V[pos]]) for pos in range(P1[i], P1[i] + ki))
    knn_acc[ki] += neigh_deg_sum / ki

knn = np.zeros(k_max + 1)
for k in range(1, k_max + 1):
    if nk[k] > 0:
        knn[k] = knn_acc[k] / nk[k]

knn_uncorr = k2_avg / k_avg
print(f"\n<k²>/<k> = {knn_uncorr:.4f}  (uncorrelated-network reference)")


# ── 7. Clustering spectrum c(k) ──────────────────────────────────────────────
ck_acc          = np.zeros(k_max + 1)
total_triangles = 0

for i in range(N):
    ki = int(D[i])
    if ki < 2:
        continue
    neighbours = V[P1[i]: P1[i] + ki]
    tri_i = 0
    for a in range(ki):
        j1 = int(neighbours[a])
        for b in range(a + 1, ki):
            j2 = int(neighbours[b])
            if j2 in adj_set[j1]:
                tri_i += 1
    total_triangles += tri_i
    ck_acc[ki] += tri_i / (ki * (ki - 1) / 2)

ck = np.zeros(k_max + 1)
for k in range(2, k_max + 1):
    if nk[k] > 0:
        ck[k] = ck_acc[k] / nk[k]

c_avg       = float(ck_acc.sum()) / N
n_triangles = total_triangles // 3

print(f"\n<c>       = {c_avg:.6f}")
print(f"Triangles = {n_triangles:,}")

# Cross-check with NetworkX
c_nx = nx.average_clustering(G)
print(f"NetworkX <c> = {c_nx:.6f}  (cross-check)")


# ── 8. Assortativity ─────────────────────────────────────────────────────────
r = nx.degree_assortativity_coefficient(G)
print(f"\nDegree assortativity r = {r:.4f}")


# ── 9. Path-length & diameter (sampled for N > 5000) ─────────────────────────
if N <= 5000:
    apl  = nx.average_shortest_path_length(G)
    diam = nx.diameter(G)
else:
    rng    = np.random.default_rng(42)
    sample = rng.choice(list(G.nodes()), size=500, replace=False)
    lengths = []
    for src in sample:
        sp = nx.single_source_shortest_path_length(G, src)
        lengths.extend(sp.values())
    apl  = float(np.mean(lengths))
    diam = int(max(lengths))

print(f"Average path length <l> = {apl:.4f}")
print(f"Diameter              D = {diam}")


# ── 10. Save summary CSV ──────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "a1_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for label, val in [
        ("N_gcc",           N),
        ("E_gcc",           E),
        ("<k>",             round(k_avg,    4)),
        ("<k2>",            round(k2_avg,   4)),
        ("kmax",            k_max),
        ("kmin",            k_min),
        ("<k2>/<k>",        round(knn_uncorr, 4)),
        ("<c>",             round(c_avg,    6)),
        ("triangles",       n_triangles),
        ("assortativity_r", round(r,        4)),
        ("<l>",             round(apl,      4)),
        ("diameter",        diam),
    ]:
        w.writerow([label, val])

print("\nDone. Results in", RESULTS)