"""
assignment1.py: Structural characterisation of the network (GCC)
References: NetworkX, numpy, scipy, matplotlib
"""

import csv
import os
from collections import Counter

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

# ── 3. Basic metrics ──────────────────────────────────────────────────────────
degrees  = np.array([d for _, d in G.degree()])
k_mean   = degrees.mean()
k2_mean  = (degrees**2).mean()
k_max    = int(degrees.max())
k_min    = int(degrees.min())

print(f"<k>   = {k_mean:.4f}")
print(f"<k²>  = {k2_mean:.4f}")
print(f"kmax  = {k_max}   kmin = {k_min}")
print(f"Density = {2*E/(N*(N-1)):.6f}")

cc_global = nx.transitivity(G)
cc_avg    = nx.average_clustering(G)
print(f"Global clustering C_Delta = {cc_global:.4f}")
print(f"Average clustering <c>    = {cc_avg:.4f}")

# ── 4. Degree distribution ────────────────────────────────────────────────────
cnt     = Counter(degrees)
k_vals  = np.array(sorted(cnt))
pk_vals = np.array([cnt[k] for k in k_vals]) / N
ccdf    = np.cumsum(pk_vals[::-1])[::-1]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

axes[0].bar(k_vals, pk_vals, width=0.8, color="steelblue", alpha=0.8)
axes[0].set(xlabel=r"$k$", ylabel=r"$P(k)$", title="Degree distribution (linear)")

axes[1].loglog(k_vals, pk_vals, "o", ms=4, color="steelblue")
axes[1].set(xlabel=r"$k$", ylabel=r"$P(k)$", title="Degree distribution (log-log)")

axes[2].loglog(k_vals, ccdf, color="steelblue")
axes[2].set(xlabel=r"$k$", ylabel=r"$P(K \geq k)$", title="CCDF (log-log)")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a1_degree_dist.pdf"))
plt.show()

# ── 5. Average nearest-neighbour degree k_nn(k) ──────────────────────────────
knn    = nx.average_degree_connectivity(G)
knn_k  = np.array(sorted(knn))
knn_kv = np.array([knn[k] for k in knn_k])

plt.figure(figsize=(5, 4))
plt.loglog(knn_k, knn_kv, "o", ms=4, color="darkorange")
plt.xlabel(r"$k$")
plt.ylabel(r"$k_{nn}(k)$")
plt.title("Average nearest-neighbour degree")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a1_knn.pdf"))
plt.show()

# ── 6. Clustering spectrum c(k) ───────────────────────────────────────────────
ck_raw = {}
for n in G.nodes():
    k = G.degree(n)
    c = nx.clustering(G, n)
    ck_raw.setdefault(k, []).append(c)
ck_k = np.array(sorted(ck_raw))
ck_v = np.array([np.mean(ck_raw[k]) for k in ck_k])

mask = ck_k > 1
plt.figure(figsize=(5, 4))
plt.loglog(ck_k[mask], ck_v[mask], "o", ms=4, color="green")
plt.xlabel(r"$k$")
plt.ylabel(r"$c(k)$")
plt.title("Clustering spectrum")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a1_ck.pdf"))
plt.show()

# ── 7. Assortativity ─────────────────────────────────────────────────────────
r = nx.degree_assortativity_coefficient(G)
print(f"\nDegree assortativity r = {r:.4f}")

# ── 8. Path-length & diameter (sampled for N > 5000) ─────────────────────────
if N <= 5000:
    apl  = nx.average_shortest_path_length(G)
    diam = nx.diameter(G)
else:
    rng     = np.random.default_rng(42)
    sample  = rng.choice(list(G.nodes()), size=500, replace=False)
    lengths = []
    for src in sample:
        sp = nx.single_source_shortest_path_length(G, src)
        lengths.extend(sp.values())
    apl  = float(np.mean(lengths))
    diam = int(max(lengths))

print(f"Average path length <l> = {apl:.4f}")
print(f"Diameter              D = {diam}")

# ── 9. Save summary ───────────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "a1_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for label, val in [
        ("N_gcc",           N),
        ("E_gcc",           E),
        ("<k>",             round(k_mean,   4)),
        ("<k2>",            round(k2_mean,  4)),
        ("kmax",            k_max),
        ("kmin",            k_min),
        ("C_delta",         round(cc_global,4)),
        ("<c>",             round(cc_avg,   4)),
        ("assortativity_r", round(r,        4)),
        ("<l>",             round(apl,      4)),
        ("diameter",        diam),
    ]:
        w.writerow([label, val])

print("\nDone. Results in", RESULTS)