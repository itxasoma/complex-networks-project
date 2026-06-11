"""
assignment4.py: Configuration Model (CM) random network generation and analysis.

Uses the degree sequence from the original GCC to:
1. Generate one CM realization.
2. Generate an ensemble of CM realizations.
3. Compare scalar topological properties with the original network.
"""


import os
import time

import numpy as np

from network_utils import analyze_graph, load_gcc_from_csv, save_csv


try:
    import community as community_louvain
except ImportError:
    raise ImportError("Install with: pip install python-louvain")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
EDGE_FILE = os.path.join(REPO_ROOT, "lastfm_asia", "lastfm_asia_edges.csv")
os.makedirs(RESULTS, exist_ok=True)

N_REALIZATIONS = 100


def configuration_model(degree_seq, rng):
    stubs = np.repeat(np.arange(len(degree_seq), dtype=np.int64), degree_seq)
    rng.shuffle(stubs)

    edge_set = set()
    for i in range(0, len(stubs) - 1, 2):
        u = int(stubs[i])
        v = int(stubs[i + 1])
        if u == v:
            continue
        e = (min(u, v), max(u, v))
        if e in edge_set:
            continue
        edge_set.add(e)

    import networkx as nx

    G = nx.Graph()
    G.add_nodes_from(range(len(degree_seq)))
    G.add_edges_from(edge_set)
    return G


def analyze_with_communities(G):
    res = analyze_graph(G)
    partition = community_louvain.best_partition(G, resolution=1.0, random_state=42)
    modularity = community_louvain.modularity(partition, G)
    n_comm = len(set(partition.values()))
    res["modularity"] = modularity
    res["n_comm"] = n_comm
    return res


G_orig = load_gcc_from_csv(EDGE_FILE)
orig = analyze_with_communities(G_orig)
D_orig = orig["D"].copy()

print(f"Original GCC  N={orig['N']}  E={orig['E']}")
print(f"<k>={orig['k_avg']:.4f}  <c>={orig['c_avg']:.6f}  r={orig['r']:.4f}  Q={orig['modularity']:.4f}")


rng = np.random.default_rng(42)

print("\nGenerating single CM realization...")
G_cm1 = configuration_model(D_orig, rng)
cm1 = analyze_with_communities(G_cm1)
print(f"CM single  N={cm1['N']}  E={cm1['E']}  discarded edges = {orig['E'] - cm1['E']}")
print(f"<k>={cm1['k_avg']:.4f}  <c>={cm1['c_avg']:.6f}  r={cm1['r']:.4f}  Q={cm1['modularity']:.4f}")


scalar_keys = ["E", "k_avg", "k2_avg", "c_avg", "r", "modularity", "n_comm", "knn_uncorr"]
ensemble = {k: [] for k in scalar_keys}

print(f"\nGenerating {N_REALIZATIONS} CM realizations...")
t0 = time.time()
for i in range(N_REALIZATIONS):
    G_cm = configuration_model(D_orig, rng)
    res = analyze_with_communities(G_cm)
    for k in scalar_keys:
        ensemble[k].append(res[k])
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{N_REALIZATIONS}  ({time.time() - t0:.0f}s elapsed)")

ens_mean = {k: float(np.mean(v)) for k, v in ensemble.items()}
ens_std = {k: float(np.std(v)) for k, v in ensemble.items()}

print("\nEnsemble averages:")
for k in scalar_keys:
    print(f"{k:<12} {ens_mean[k]:.4f} ± {ens_std[k]:.4f}")


rows = [
    ["E", orig["E"], cm1["E"], ens_mean["E"], ens_std["E"]],
    ["<k>", orig["k_avg"], cm1["k_avg"], ens_mean["k_avg"], ens_std["k_avg"]],
    ["<k2>", orig["k2_avg"], cm1["k2_avg"], ens_mean["k2_avg"], ens_std["k2_avg"]],
    ["<c>", orig["c_avg"], cm1["c_avg"], ens_mean["c_avg"], ens_std["c_avg"]],
    ["r", orig["r"], cm1["r"], ens_mean["r"], ens_std["r"]],
    ["Q", orig["modularity"], cm1["modularity"], ens_mean["modularity"], ens_std["modularity"]],
    ["n_comm", orig["n_comm"], cm1["n_comm"], ens_mean["n_comm"], ens_std["n_comm"]],
    ["<k2>/<k>", orig["knn_uncorr"], cm1["knn_uncorr"], ens_mean["knn_uncorr"], ens_std["knn_uncorr"]],
]

save_csv(
    os.path.join(RESULTS, "a4_comparison.csv"),
    ["metric", "original", "cm_single", "cm_mean", "cm_std"],
    [[r[0], round(float(r[1]), 4), round(float(r[2]), 4), round(float(r[3]), 4), round(float(r[4]), 4)] for r in rows],
)

save_csv(
    os.path.join(RESULTS, "a4_ensemble.csv"),
    scalar_keys,
    [[round(float(ensemble[k][i]), 4) for k in scalar_keys] for i in range(N_REALIZATIONS)],
)

print("\nDone. Results saved to", RESULTS)
print("Run assignment4-plots.py to generate figures.")