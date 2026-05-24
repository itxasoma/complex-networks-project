"""
assignment3.py: Community structure of the network (GCC)
Method: Louvain algorithm (via python-louvain / community package)
Produces community plots and a summary CSV.
"""

import csv
import os
import random
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
                continue
            if u == v:
                continue
            edges.add((min(u, v), max(u, v)))
            nodes_raw.update([u, v])
    return edges, nodes_raw


edge_file = os.path.join(REPO_ROOT, "lastfm_asia", "lastfm_asia_edges.csv")
raw_edges, _ = load_edgelist(edge_file)

G_full = nx.Graph()
G_full.add_edges_from(raw_edges)
G_full.remove_nodes_from(list(nx.isolates(G_full)))


# ── 2. Giant Connected Component ─────────────────────────────────────────────
gcc_nodes = max(nx.connected_components(G_full), key=len)
G = G_full.subgraph(gcc_nodes).copy()

mapping = {n: i for i, n in enumerate(sorted(G.nodes()))}
G = nx.relabel_nodes(G, mapping)

N = G.number_of_nodes()
E = G.number_of_edges()
print(f"GCC  N={N}  E={E}")


# ── 3. Louvain community detection ───────────────────────────────────────────
partition  = community_louvain.best_partition(G, resolution=1.0, random_state=42)
n_comm     = len(set(partition.values()))
modularity = community_louvain.modularity(partition, G)

print(f"\nNumber of communities : {n_comm}")
print(f"Modularity Q          : {modularity:.4f}")

comm_sizes = Counter(partition.values())
sizes      = np.array(sorted(comm_sizes.values(), reverse=True))

print(f"Largest community     : {sizes[0]} nodes  ({100*sizes[0]/N:.1f}%)")
print(f"Smallest community    : {sizes[-1]} nodes")
print(f"Mean community size   : {sizes.mean():.1f}")
print(f"Median community size : {np.median(sizes):.1f}")


# ── Shared colour map: plasma mapped on log(size) ────────────────────────────
# Mapping by log(size) instead of linear rank gives much better colour spread
# when the size distribution is heavy-tailed (many small, few large communities).
cmap_base  = matplotlib.colormaps["plasma_r"]

all_sizes  = np.array([comm_sizes[cid] for cid in comm_sizes])
log_sizes  = np.log(all_sizes)
log_min, log_max = log_sizes.min(), log_sizes.max()

def comm_color(cid):
    """Plasma colour mapped on log(size): large=yellow, small=purple."""
    s = comm_sizes[cid]
    t = (np.log(s) - log_min) / max(log_max - log_min, 1e-9)
    return cmap_base(0.05 + 0.90 * t)


# ── 4. Community-size distribution ───────────────────────────────────────────
ranked_ids = [cid for cid, _ in comm_sizes.most_common()]
bar_colors = [comm_color(cid) for cid in ranked_ids]

fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(range(1, n_comm + 1), sizes, color=bar_colors, alpha=0.9)
ax.set(xlabel="Community rank (by size)", ylabel="Number of nodes",
       title=f"Community sizes  (Q = {modularity:.3f},  {n_comm} communities)")
ax.axhline(sizes.mean(), color="black", lw=1.2, ls="--",
           label=rf"mean = {sizes.mean():.1f}")
ax.legend(fontsize=9)

sm = cm.ScalarMappable(cmap=cmap_base,
                       norm=mcolors.Normalize(vmin=log_min, vmax=log_max))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("log(community size)", fontsize=8)
cbar.ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_community_sizes.pdf"))
plt.show()


# ── 5. Network visualisation ──────────────────────────────────────────────────
VIZ_MAX = 2000
random.seed(42)

if N > VIZ_MAX:
    sampled = []
    for cid, cnt in comm_sizes.items():
        members = [n for n, c in partition.items() if c == cid]
        k       = max(1, round(VIZ_MAX * cnt / N))
        sampled.extend(random.sample(members, min(k, len(members))))
    sampled = list(set(sampled))[:VIZ_MAX]
    H       = G.subgraph(sampled).copy()
    subtitle = f"(sampled {len(H)} / {N} nodes)"
else:
    H        = G
    subtitle  = f"({N} nodes)"

sub_partition = {n: partition[n] for n in H.nodes()}
node_colors   = [comm_color(sub_partition[n]) for n in H.nodes()]

print(f"\nComputing spring layout for {len(H)} nodes …")
pos = nx.spring_layout(H, seed=42, k=1 / np.sqrt(len(H)))

fig, ax = plt.subplots(figsize=(5, 4))

# Edges: solid, visible but not dominant
nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.25, width=0.4, edge_color="#000000")
nx.draw_networkx_nodes(H, pos, ax=ax, node_color=node_colors,
                       node_size=6, linewidths=0)

ax.set_title(
    f"Louvain communities  {subtitle}\n"
    f"Q = {modularity:.4f},  {n_comm} communities",
    fontsize=8
)
ax.axis("off")

sm2 = cm.ScalarMappable(cmap=cmap_base,
                        norm=mcolors.Normalize(vmin=log_min, vmax=log_max))
sm2.set_array([])
cbar2 = plt.colorbar(sm2, ax=ax, shrink=0.6, pad=0.01)
cbar2.set_label("log(community size)", fontsize=7)
cbar2.ax.tick_params(labelsize=6)
cbar2.ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_network_communities.pdf"), dpi=150)
plt.show()


# ── 6. Intra- vs inter-community edge fractions ───────────────────────────────
intra = sum(1 for u, v in G.edges() if partition[u] == partition[v])
inter = E - intra
print(f"\nIntra-community edges : {intra}  ({100*intra/E:.1f}%)")
print(f"Inter-community edges : {inter}  ({100*inter/E:.1f}%)")

fig, ax = plt.subplots(figsize=(4, 4))
ax.pie([intra, inter],
       labels=["Intra-community", "Inter-community"],
       colors=["steelblue", "tomato"],
       autopct="%1.1f%%", startangle=90,
       wedgeprops={"edgecolor": "white", "linewidth": 1.2})
ax.set_title("Edge distribution by community membership")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_edge_pie.pdf"))
plt.show()


# ── 7. Degree distribution: top 5 communities (overlaid, default colors) ─────
top5_ids = [cid for cid, _ in comm_sizes.most_common(5)]

fig, ax = plt.subplots(figsize=(5, 4))
for cid in top5_ids:
    members = [n for n, c in partition.items() if c == cid]
    degs    = [G.degree(n) for n in members]
    cnt     = Counter(degs)
    ks      = np.array(sorted(cnt))
    pk      = np.array([cnt[k] / len(members) for k in ks])
    ax.loglog(ks, pk, "o", ms=3.5,
              label=f"C{cid} ({len(members)} nodes)")

ax.set(xlabel=r"$k$", ylabel=r"$P(k)$",
       title="Degree distribution: top 5 communities")
ax.legend(fontsize=7, loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_comm_degree_dists.pdf"))
plt.show()


# ── 8. Save summary CSV ───────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "a3_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for label, val in [
        ("N_gcc",            N),
        ("E_gcc",            E),
        ("n_communities",    n_comm),
        ("modularity_Q",     round(modularity, 4)),
        ("largest_comm",     int(sizes[0])),
        ("smallest_comm",    int(sizes[-1])),
        ("mean_comm_size",   round(float(sizes.mean()), 1)),
        ("intra_edges",      intra),
        ("inter_edges",      inter),
        ("intra_frac",       round(intra / E, 4)),
    ]:
        w.writerow([label, val])

with open(os.path.join(RESULTS, "a3_community_table.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["community_id", "size", "frac_of_N"])
    for cid, cnt in comm_sizes.most_common():
        w.writerow([cid, cnt, round(cnt / N, 4)])

print("\nDone. Results in", RESULTS)