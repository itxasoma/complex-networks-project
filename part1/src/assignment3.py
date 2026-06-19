"""
assignment3.py: Community structure of the network (GCC).

Uses Louvain community detection and produces:
- community size distribution
- network visualization colored by community
- intra/inter-community edge fractions
- degree distributions of the largest communities
"""


import os
from collections import Counter

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.cm import ScalarMappable

from network_utils import load_gcc_from_csv, save_csv


try:
    import community as community_louvain
except ImportError:
    raise ImportError("Install with: pip install python-louvain")


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
EDGE_FILE = os.path.join(REPO_ROOT, "../lastfm_asia", "lastfm_asia_edges.csv")
os.makedirs(RESULTS, exist_ok=True)

plt.style.use(os.path.join(SCRIPT_DIR, "mplstyle", "science.mplstyle"))


G = load_gcc_from_csv(EDGE_FILE)
N = G.number_of_nodes()
E = G.number_of_edges()

partition = community_louvain.best_partition(G, resolution=1.0, random_state=42)
modularity = community_louvain.modularity(partition, G)
comm_sizes = Counter(partition.values())
n_comm = len(comm_sizes)
sizes = np.array(sorted(comm_sizes.values(), reverse=True))

print(f"GCC  N={N}  E={E}")
print(f"Number of communities = {n_comm}")
print(f"Modularity Q = {modularity:.4f}")
print(f"Largest community = {sizes[0]} nodes")
print(f"Smallest community = {sizes[-1]} nodes")
print(f"Mean community size = {sizes.mean():.1f}")


cmap = plt.get_cmap("inferno_r")
size_values = np.array(list(comm_sizes.values()))
log_min = np.log(size_values).min()
log_max = np.log(size_values).max()


def community_color(cid):
    s = comm_sizes[cid]
    t = (np.log(s) - log_min) / max(log_max - log_min, 1e-12)
    return cmap(0.05 + 0.90 * t)


# Community size distribution
ranked = comm_sizes.most_common()
rank_ids = [cid for cid, _ in ranked]
bar_colors = [community_color(cid) for cid in rank_ids]

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(range(1, n_comm + 1), sizes, color=bar_colors)
ax.axhline(sizes.mean(), color="black", lw=1.2, ls="--", label=f"mean = {sizes.mean():.1f}")
ax.set_xlabel("Community rank")
ax.set_ylabel("Number of nodes")
ax.set_title(f"Community sizes (Q = {modularity:.3f})")
ax.legend()

sm = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=log_min, vmax=log_max))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("log(community size)")
cbar.ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_community_sizes.pdf"))
plt.close()


# Network visualization
VIZ_MAX = 2000
rng = np.random.default_rng(42)

if N > VIZ_MAX:
    sampled = []
    for cid, cnt in comm_sizes.items():
        members = np.array([n for n, c in partition.items() if c == cid])
        k = max(1, round(VIZ_MAX * cnt / N))
        if len(members) <= k:
            sampled.extend(members.tolist())
        else:
            sampled.extend(rng.choice(members, size=k, replace=False).tolist())
    sampled = sorted(set(sampled))[:VIZ_MAX]
    H = G.subgraph(sampled).copy()
    subtitle = f"(sampled {len(H)} / {N} nodes)"
else:
    H = G
    subtitle = f"({N} nodes)"

node_colors = [community_color(partition[n]) for n in H.nodes()]
print(f"Computing spring layout for {len(H)} nodes...")
pos = nx.spring_layout(H, seed=42, k=1 / np.sqrt(len(H)))

fig, ax = plt.subplots(figsize=(5, 4))
nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.20, width=0.35, edge_color="#000000")
nx.draw_networkx_nodes(H, pos, ax=ax, node_color=node_colors, node_size=6, linewidths=0)
ax.set_title(f"Louvain communities {subtitle}\nQ = {modularity:.4f}, {n_comm} communities", fontsize=8)
ax.axis("off")

sm2 = ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=log_min, vmax=log_max))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax, shrink=0.65, pad=0.01)
cbar2.set_label("log(community size)", fontsize=7)
cbar2.ax.tick_params(labelsize=6)
cbar2.ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_network_communities.pdf"), dpi=150)
plt.close()


# Intra vs inter edges
intra = sum(1 for u, v in G.edges() if partition[u] == partition[v])
inter = E - intra

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie(
    [intra, inter],
    labels=["Intra-community", "Inter-community"],
    colors=["steelblue", "tomato"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.2},
)
ax.set_title("Edge distribution by community membership")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_edge_pie.pdf"))
plt.close()


# Degree distributions of top 5 communities
top5_ids = [cid for cid, _ in comm_sizes.most_common(5)]

fig, ax = plt.subplots(figsize=(5, 4))
for cid in top5_ids:
    members = [n for n, c in partition.items() if c == cid]
    degs = [G.degree(n) for n in members]
    cnt = Counter(degs)
    ks = np.array(sorted(cnt))
    pk = np.array([cnt[k] / len(members) for k in ks])
    ax.loglog(ks, pk, "o", ms=3.5, label=f"C{cid} ({len(members)} nodes)")

ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k)$")
ax.set_title("Degree distribution of top 5 communities")
ax.legend(fontsize=7, loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_comm_degree_dists.pdf"))
plt.close()


save_csv(
    os.path.join(RESULTS, "a3_summary.csv"),
    ["metric", "value"],
    [
        ["N_gcc", N],
        ["E_gcc", E],
        ["n_communities", n_comm],
        ["modularity_Q", round(modularity, 4)],
        ["largest_comm", int(sizes[0])],
        ["smallest_comm", int(sizes[-1])],
        ["mean_comm_size", round(float(sizes.mean()), 1)],
        ["intra_edges", intra],
        ["inter_edges", inter],
        ["intra_frac", round(intra / E, 4)],
    ],
)

save_csv(
    os.path.join(RESULTS, "a3_community_table.csv"),
    ["community_id", "size", "frac_of_N"],
    [[cid, cnt, round(cnt / N, 4)] for cid, cnt in comm_sizes.most_common()],
)

print("\nDone. Results in", RESULTS)