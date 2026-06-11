"""
assignment2.py: Degree distribution, ANND and clustering spectrum.

Computes P(k), Pc(k), k_nn(k), c(k), and produces the required plots.
Also creates an optional GCC visualization for the report.
"""


import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from network_utils import analyze_graph, load_gcc_from_csv, save_csv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
EDGE_FILE = os.path.join(REPO_ROOT, "lastfm_asia", "lastfm_asia_edges.csv")
os.makedirs(RESULTS, exist_ok=True)

plt.style.use(os.path.join(SCRIPT_DIR, "mplstyle", "science.mplstyle"))


def plot_gcc_network(G, outpath):
    deg = dict(G.degree())
    nodes = list(G.nodes())
    node_sizes = [5 + 2.0 * np.sqrt(deg[n]) for n in nodes]
    node_colors = [deg[n] for n in nodes]

    pos = nx.spring_layout(G, seed=42, k=0.15)

    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw_networkx_edges(G, pos, ax=ax, width=0.15, alpha=0.2, edge_color="gray")
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_title("Giant connected component")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


G = load_gcc_from_csv(EDGE_FILE)
res = analyze_graph(G)

ks_pk = np.array([k for k in range(1, res["k_max"] + 1) if res["nk"][k] > 0])
ks_ck = np.array([k for k in range(2, res["k_max"] + 1) if res["nk"][k] > 0 and res["ck"][k] > 0])

print(f"GCC  N={res['N']}  E={res['E']}")
print(f"<k>   = {res['k_avg']:.4f}")
print(f"<k2>  = {res['k2_avg']:.4f}")
print(f"<k2>/<k> = {res['knn_uncorr']:.4f}")
print(f"<c>   = {res['c_avg']:.6f}")
print(f"Triangles = {res['n_triangles']:,}")


# Optional report figure
plot_gcc_network(G, os.path.join(RESULTS, "a2_gcc_network.pdf"))


# P(k) and Pc(k) together
fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_pk, res["Pk"][ks_pk], "o-", ms=4, color="steelblue", label=r"$P(k)$")
ax.loglog(ks_pk, res["Pc"][ks_pk], "s-", ms=4, color="tomato", label=r"$P_c(k)$")
ax.set_xlabel(r"$k$")
ax.set_ylabel("Probability")
ax.set_title("Degree distribution")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a2_degree_distributions.pdf"))
plt.close()


# k_nn(k)
fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_pk, res["knn"][ks_pk], "o", ms=4, color="darkorange")
ax.axhline(
    res["knn_uncorr"],
    color="black",
    lw=1.2,
    ls="--",
    label=rf"$\langle k^2 \rangle / \langle k \rangle = {res['knn_uncorr']:.2f}$",
)
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$k_{\mathrm{nn}}(k)$")
ax.set_title("Average nearest-neighbour degree")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a2_knn.pdf"))
plt.close()


# c(k)
fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_ck, res["ck"][ks_ck], "o", ms=4, color="forestgreen")
ax.axhline(
    res["c_avg"],
    color="black",
    lw=1.2,
    ls="--",
    label=rf"$\langle c \rangle = {res['c_avg']:.4f}$",
)
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$c(k)$")
ax.set_title("Clustering spectrum")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a2_ck.pdf"))
plt.close()


save_csv(
    os.path.join(RESULTS, "a2_summary.csv"),
    ["metric", "value"],
    [
        ["N_gcc", res["N"]],
        ["E_gcc", res["E"]],
        ["<k>", round(res["k_avg"], 4)],
        ["<k2>", round(res["k2_avg"], 4)],
        ["<k2>/<k>", round(res["knn_uncorr"], 4)],
        ["<c>", round(res["c_avg"], 6)],
        ["triangles", res["n_triangles"]],
        ["assortativity_r", round(res["r"], 4)],
    ],
)

save_csv(
    os.path.join(RESULTS, "a2_degree_table.csv"),
    ["k", "nk", "Pk", "Pc_k", "knn_k", "c_k"],
    [
        [k, int(res["nk"][k]), float(res["Pk"][k]), float(res["Pc"][k]), float(res["knn"][k]), float(res["ck"][k])]
        for k in range(1, res["k_max"] + 1)
    ],
)

print("\nDone. Results in", RESULTS)