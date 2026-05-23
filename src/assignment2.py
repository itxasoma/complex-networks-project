"""
assignment2.py: Degree distribution, ANND and clustering spectrum (GCC)
References: NetworkX, numpy, scipy, matplotlib
"""

import csv
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# ── Paths (all relative to this script's location) ───────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS    = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


# Optional matplotlib style
plt.style.use(os.path.join(SCRIPT_DIR, "mplstyle", "science.mplstyle"))


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
                continue          # skip header "node_1,node_2"
            if u == v:
                continue          # drop self-loops
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


# ── 2b. Pointer-based adjacency structure (Assignment 1) ─────────────────────
# D[i]  : degree of node i                              (length N)
# P1[i] : first pointer  – start of node i's block in V (frozen)
# P2[i] : second pointer – current write-head           (advances in pass 2)
# V     : flat neighbour list                           (length 2E)

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


# Build adjacency sets for O(1) edge look-up in triangle counting
adj_set = [set(V[P1[i]:P1[i] + D[i]]) for i in range(N)]


k_max = int(D.max())
k_min = int(D.min())
k_avg = float(D.mean())
print(f"<k>   = {k_avg:.4f}")
print(f"kmax  = {k_max}   kmin = {k_min}")


# ── 3. Degree distribution  P(k)  and complementary CDF  P_c(k) ─────────────
# Slide prescription:
#   • One loop over the degree vector to build the raw histogram  nk  (1 vector).
#   • Normalise by N to obtain P(k).
#   • One loop from k_max down to 0, accumulating, to get P_c(k) = P(K >= k).

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

print(f"Sum P(k) = {Pk.sum():.6f}  (should be 1.0)")
print(f"P_c(0)   = {Pc[0]:.6f}  (should be 1.0)")


# ── 4. Average nearest-neighbour degree  k_nn(k) ─────────────────────────────
# Slide prescription:
#   For each node i (1 loop), increment knn[D[i]] with the local contribution
#   of all neighbours j: add D[j] / D[i]. Then divide by nk[D[i]].

knn_acc = np.zeros(k_max + 1)

for i in range(N):
    ki = D[i]
    if ki == 0:
        continue
    neigh_deg_sum = 0
    for pos in range(P1[i], P1[i] + ki):
        j = V[pos]
        neigh_deg_sum += D[j]
    knn_acc[ki] += neigh_deg_sum / ki

knn = np.zeros(k_max + 1)
for k in range(1, k_max + 1):
    if nk[k] > 0:
        knn[k] = knn_acc[k] / nk[k]

k2_mean    = float((D**2).mean())
knn_uncorr = k2_mean / k_avg
print(f"<k²>         = {k2_mean:.4f}")
print(f"<k²>/<k>     = {knn_uncorr:.4f}  (uncorrelated-network reference)")


# ── 5. Clustering spectrum  c̄(k) ─────────────────────────────────────────────
# Slide prescription:
#   For each node i, check all pairs of neighbours with 3 nested loops.
#   If two neighbours are connected, they form a triangle through i.
#   c_i = triangles_i / (k_i (k_i - 1) / 2)
#   c̄(k) = average of c_i over all nodes of degree k

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
    c_i = tri_i / (ki * (ki - 1) / 2)
    ck_acc[ki] += c_i

ck = np.zeros(k_max + 1)
for k in range(2, k_max + 1):
    if nk[k] > 0:
        ck[k] = ck_acc[k] / nk[k]

c_global    = float(ck_acc.sum()) / N
n_triangles = total_triangles // 3

print(f"<c> (global avg clustering) = {c_global:.6f}")
print(f"Triangles                   = {n_triangles:,}")

c_nx = nx.average_clustering(G)
print(f"NetworkX average_clustering = {c_nx:.6f}  (cross-check)")


# ── 6. Assortativity ─────────────────────────────────────────────────────────
r = nx.degree_assortativity_coefficient(G)
print(f"\nDegree assortativity r = {r:.4f}")


# ── 7. Prepare arrays for plots ──────────────────────────────────────────────
ks_nz  = np.array([k for k in range(1, k_max + 1) if nk[k] > 0])
Pk_nz  = Pk[ks_nz]
Pc_nz  = Pc[ks_nz]
knn_nz = knn[ks_nz]

ks_ck  = np.array([k for k in range(2, k_max + 1) if nk[k] > 0 and ck[k] > 0])
ck_nz  = ck[ks_ck]


# ── 8. Degree distribution plots ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(ks_nz, Pk_nz, width=0.8, color="steelblue", alpha=0.8)
ax.axvline(k_avg, color="tomato", lw=1.5, ls="--",
           label=rf"$\langle k \rangle = {k_avg:.2f}$")
ax.set(xlabel=r"$k$", ylabel=r"$P(k)$", title="Degree distribution (linear)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a2_degree_dist_linear.pdf"))
plt.show()


fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_nz, Pk_nz, "o", ms=4, color="steelblue")
ax.set(xlabel=r"$k$", ylabel=r"$P(k)$", title="Degree distribution (log-log)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a2_degree_dist_loglog.pdf"))
plt.show()


fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_nz, Pc_nz, color="steelblue")
ax.set(xlabel=r"$k$", ylabel=r"$P(K \geq k)$", title="CCDF (log-log)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a2_ccdf.pdf"))
plt.show()


# ── 9. Average nearest-neighbour degree ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_nz, knn_nz, "o", ms=4, color="darkorange")
ax.axhline(knn_uncorr, color="tomato", lw=1.5, ls="--",
           label=rf"$\langle k^2 \rangle / \langle k \rangle = {knn_uncorr:.2f}$")
ax.set(xlabel=r"$k$", ylabel=r"$k_{\mathrm{nn}}(k)$",
       title="Average nearest-neighbour degree")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a2_knn.pdf"))
plt.show()


# ── 10. Clustering spectrum ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_ck, ck_nz, "o", ms=4, color="green")
ax.axhline(c_global, color="tomato", lw=1.5, ls="--",
           label=rf"$\langle c \rangle = {c_global:.4f}$")
ax.set(xlabel=r"$k$", ylabel=r"$c(k)$", title="Clustering spectrum")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a2_ck.pdf"))
plt.show()


# ── 11. Save summary ─────────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "a2_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for label, val in [
        ("N_gcc",           N),
        ("E_gcc",           E),
        ("<k>",             round(k_avg,       4)),
        ("<k2>",            round(k2_mean,     4)),
        ("kmax",            k_max),
        ("kmin",            k_min),
        ("<k2>/<k>",        round(knn_uncorr,  4)),
        ("<c>",             round(c_global,    6)),
        ("triangles",       n_triangles),
        ("assortativity_r", round(r,           4)),
    ]:
        w.writerow([label, val])


with open(os.path.join(RESULTS, "a2_degree_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["k", "nk", "Pk", "Pc_k", "knn_k", "c_k"])
    for k in range(1, k_max + 1):
        w.writerow([k, int(nk[k]), float(Pk[k]),
                    float(Pc[k]), float(knn[k]), float(ck[k])])


print("\nDone. Results in", RESULTS)