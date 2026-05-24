"""
assignment3.py: Community structure of the network (GCC)
Method: Louvain algorithm (via python-louvain / community package)
Produces community plots and a summary CSV.
"""

import csv
import os
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

try:
    import community as community_louvain          # python-louvain
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
# Resolution gamma=1.0 (default). Set random_state for reproducibility.
partition = community_louvain.best_partition(G, resolution=1.0, random_state=42)

# partition: dict  node -> community_id
n_communities = len(set(partition.values()))
modularity    = community_louvain.modularity(partition, G)

print(f"\nNumber of communities : {n_communities}")
print(f"Modularity Q          : {modularity:.4f}")

# Community sizes
from collections import Counter
comm_sizes = Counter(partition.values())
sizes      = np.array(sorted(comm_sizes.values(), reverse=True))

print(f"Largest community     : {sizes[0]} nodes  ({100*sizes[0]/N:.1f}%)")
print(f"Smallest community    : {sizes[-1]} nodes")
print(f"Mean community size   : {sizes.mean():.1f}")
print(f"Median community size : {np.median(sizes):.1f}")


# ── 4. Community-size distribution ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, len(sizes) + 1), sizes, color="steelblue", alpha=0.8)
ax.set(xlabel="Community rank (by size)", ylabel="Number of nodes",
       title=f"Community sizes  (Q = {modularity:.3f},  {n_communities} communities)")
ax.axhline(sizes.mean(), color="tomato", lw=1.5, ls="--",
           label=rf"mean = {sizes.mean():.1f}")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_community_sizes.pdf"))
plt.show()


# ── 5. Network visualisation (spring layout, coloured by community) ───────────
# For large N we sample a subgraph to keep the plot readable.
VIZ_MAX = 2000
random.seed(42)

if N > VIZ_MAX:
    # Sample nodes while keeping community proportions
    sampled = []
    for cid, cnt in comm_sizes.items():
        members = [n for n, c in partition.items() if c == cid]
        k       = max(1, round(VIZ_MAX * cnt / N))
        sampled.extend(random.sample(members, min(k, len(members))))
    sampled = list(set(sampled))[:VIZ_MAX]
    H       = G.subgraph(sampled).copy()
    subtitle = f"(sampled {len(H)} / {N} nodes)"
else:
    H       = G
    subtitle = f"({N} nodes)"

sub_partition = {n: partition[n] for n in H.nodes()}

# Assign a colour to each community
cmap        = plt.get_cmap("tab20")
comm_ids    = sorted(set(sub_partition.values()))
color_map   = {cid: cmap(i % 20) for i, cid in enumerate(comm_ids)}
node_colors = [color_map[sub_partition[n]] for n in H.nodes()]

print(f"\nComputing spring layout for {len(H)} nodes …")
pos = nx.spring_layout(H, seed=42, k=1/np.sqrt(len(H)))

fig, ax = plt.subplots(figsize=(10, 10))
nx.draw_networkx_edges(H, pos, ax=ax, alpha=0.08, width=0.4, edge_color="gray")
nx.draw_networkx_nodes(H, pos, ax=ax, node_color=node_colors,
                       node_size=12, linewidths=0)
ax.set_title(f"Community structure — Louvain  {subtitle}\n"
             f"Q = {modularity:.4f},  {n_communities} communities", fontsize=11)
ax.axis("off")

# Legend: only the largest communities to avoid clutter
top_n   = min(12, n_communities)
top_ids = [cid for cid, _ in comm_sizes.most_common(top_n)]
patches = [mpatches.Patch(color=color_map[cid],
                          label=f"C{cid}  ({comm_sizes[cid]} nodes)")
           for cid in top_ids]
ax.legend(handles=patches, loc="lower left", fontsize=7,
          title="Top communities", title_fontsize=8, framealpha=0.7)

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


# ── 7. Degree distribution per community (top 5 communities) ─────────────────
top5_ids = [cid for cid, _ in comm_sizes.most_common(5)]
fig, axes = plt.subplots(1, 5, figsize=(14, 3), sharey=False)

for ax, cid in zip(axes, top5_ids):
    members  = [n for n, c in partition.items() if c == cid]
    degs     = [G.degree(n) for n in members]
    cnt      = Counter(degs)
    ks       = sorted(cnt)
    pk       = [cnt[k] / len(members) for k in ks]
    ax.bar(ks, pk, color=color_map[cid], alpha=0.85, width=0.8)
    ax.set_title(f"C{cid}\n({len(members)} nodes)", fontsize=8)
    ax.set_xlabel(r"$k$", fontsize=8)
    ax.set_ylabel(r"$P(k)$", fontsize=8)
    ax.tick_params(labelsize=7)

plt.suptitle("Degree distribution — top 5 communities", fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a3_comm_degree_dists.pdf"), bbox_inches="tight")
plt.show()


# ── 8. Save summary CSV ───────────────────────────────────────────────────────
with open(os.path.join(RESULTS, "a3_summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["metric", "value"])
    for label, val in [
        ("N_gcc",            N),
        ("E_gcc",            E),
        ("n_communities",    n_communities),
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