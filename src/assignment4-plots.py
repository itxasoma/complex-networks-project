"""
assignment4-plots.py: Plots for Assignment 4 (CM analysis).

Reads the CSVs produced by assignment4.py and the original edge list,
then generates all comparison figures. Run after assignment4.py.

Reads:
  results/a4_comparison.csv
  results/a4_ensemble.csv

Writes:
  results/a4_pk_comparison.pdf
  results/a4_ccdf_comparison.pdf
  results/a4_knn_comparison.pdf
  results/a4_ck_comparison.pdf
  results/a4_ensemble_distributions.pdf
"""

import csv
import os

import matplotlib.pyplot as plt
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

plt.style.use(os.path.join(SCRIPT_DIR, "mplstyle", "science.mplstyle"))


# ── 1. Load comparison CSV ────────────────────────────────────────────────────
comparison = {}
with open(os.path.join(RESULTS, "a4_comparison.csv")) as f:
    for row in csv.DictReader(f):
        comparison[row["metric"]] = {
            "original":  float(row["original"]),
            "cm_single": float(row["cm_single"]),
            "cm_mean":   float(row["cm_mean"]),
            "cm_std":    float(row["cm_std"]),
        }

# ── 2. Load ensemble CSV ──────────────────────────────────────────────────────
ensemble = {}
with open(os.path.join(RESULTS, "a4_ensemble.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        for k, v in row.items():
            ensemble.setdefault(k, []).append(float(v))

ens_mean = {k: np.mean(v) for k, v in ensemble.items()}
ens_std  = {k: np.std(v)  for k, v in ensemble.items()}
N_REALIZATIONS = len(ensemble["E"])

print(f"Loaded {N_REALIZATIONS} ensemble realizations from CSV.")


# ── 3. Rebuild original graph and single CM for distribution plots ────────────
# We need the full Pk/knn/ck arrays, which are not stored in the CSVs.
# Re-running analyse() on the original and one CM realization is fast (~5s).

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


def build_adjacency(G):
    """Return D, P1, V, adj_set for graph G."""
    N_ = G.number_of_nodes()
    E_ = G.number_of_edges()
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
    return D, P1, V, adj_set


def compute_distributions(G):
    """Return nk, Pk, Pc, knn, ck, k_max, k_avg, k2_avg, c_avg, knn_uncorr."""
    D, P1, V, adj_set = build_adjacency(G)
    N_ = G.number_of_nodes()
    k_max  = int(D.max())
    k_avg  = float(D.mean())
    k2_avg = float((D**2).mean())

    nk = np.zeros(k_max + 1, dtype=np.int64)
    for i in range(N_):
        nk[D[i]] += 1
    Pk = nk / N_
    Pc = np.zeros(k_max + 1)
    Pc[k_max] = Pk[k_max]
    for k in range(k_max - 1, -1, -1):
        Pc[k] = Pc[k + 1] + Pk[k]

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

    return dict(nk=nk, Pk=Pk, Pc=Pc, knn=knn, ck=ck,
                k_max=k_max, k_avg=k_avg, k2_avg=k2_avg,
                c_avg=c_avg, knn_uncorr=k2_avg / k_avg)


edge_file = os.path.join(REPO_ROOT, "lastfm_asia", "lastfm_asia_edges.csv")
raw_edges = load_edgelist(edge_file)

G_full = nx.Graph()
G_full.add_edges_from(raw_edges)
G_full.remove_nodes_from(list(nx.isolates(G_full)))
gcc_nodes = max(nx.connected_components(G_full), key=len)
G_orig    = G_full.subgraph(gcc_nodes).copy()
mapping   = {n: i for i, n in enumerate(sorted(G_orig.nodes()))}
G_orig    = nx.relabel_nodes(G_orig, mapping)

D_orig = np.array([d for _, d in sorted(G_orig.degree())], dtype=np.int64)

print("Computing distributions for original network …")
orig = compute_distributions(G_orig)

print("Generating single CM realization (seed=42) …")
rng   = np.random.default_rng(42)
stubs = np.repeat(np.arange(len(D_orig), dtype=np.int64), D_orig)
rng.shuffle(stubs)
edge_set = set()
for k in range(0, len(stubs) - 1, 2):
    u, v = int(stubs[k]), int(stubs[k + 1])
    if u == v:
        continue
    key = (min(u, v), max(u, v))
    if key not in edge_set:
        edge_set.add(key)
G_cm1 = nx.Graph()
G_cm1.add_nodes_from(range(len(D_orig)))
G_cm1.add_edges_from(edge_set)

print("Computing distributions for CM single realization …")
cm1 = compute_distributions(G_cm1)
print("Ready. Generating plots …\n")


# ── Helper: non-zero index array ─────────────────────────────────────────────
def nz(res, start=1):
    return np.array([k for k in range(start, res["k_max"] + 1) if res["nk"][k] > 0])

def nz_knn(res):
    return np.array([k for k in range(1, res["k_max"] + 1)
                     if res["nk"][k] > 0 and res["knn"][k] > 0])

def nz_ck(res):
    return np.array([k for k in range(2, res["k_max"] + 1)
                     if res["nk"][k] > 0 and res["ck"][k] > 0])


# ── Plot 1: P(k) comparison ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ks_o = nz(orig)
ks_c = nz(cm1)
ax.loglog(ks_o, orig["Pk"][ks_o], "o", ms=3.5, label="Original")
ax.loglog(ks_c, cm1["Pk"][ks_c],  "s", ms=3.5, label="CM (single)")
ax.set(xlabel=r"$k$", ylabel=r"$P(k)$", title=r"Degree distribution $P(k)$")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_pk_comparison.pdf"))
plt.show()
print("Saved a4_pk_comparison.pdf")


# ── Plot 2: CCDF comparison ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_o, orig["Pc"][ks_o], color="steelblue",  label="Original")
ax.loglog(ks_c, cm1["Pc"][ks_c],  color="darkorange", ls="--", label="CM (single)")
ax.set(xlabel=r"$k$", ylabel=r"$P_c(k)$", title="CCDF")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_ccdf_comparison.pdf"))
plt.show()
print("Saved a4_ccdf_comparison.pdf")


# ── Plot 3: k_nn(k) comparison ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ks_knn_o = nz_knn(orig)
ks_knn_c = nz_knn(cm1)
ax.loglog(ks_knn_o, orig["knn"][ks_knn_o], "o", ms=3.5, label="Original")
ax.loglog(ks_knn_c, cm1["knn"][ks_knn_c],  "s", ms=3.5, label="CM (single)")
ax.axhline(orig["knn_uncorr"], color="tomato", lw=1.2, ls="--",
           label=rf"$\langle k^2\rangle/\langle k\rangle = {orig['knn_uncorr']:.2f}$")
ax.set(xlabel=r"$k$", ylabel=r"$k_{\mathrm{nn}}(k)$",
       title="Average nearest-neighbour degree")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_knn_comparison.pdf"))
plt.show()
print("Saved a4_knn_comparison.pdf")


# ── Plot 4: c(k) comparison ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
ks_ck_o = nz_ck(orig)
ks_ck_c = nz_ck(cm1)
ax.loglog(ks_ck_o, orig["ck"][ks_ck_o], "o", ms=3.5, label="Original")
ax.loglog(ks_ck_c, cm1["ck"][ks_ck_c],  "s", ms=3.5, label="CM (single)")
ax.axhline(orig["c_avg"], color="steelblue", lw=1.2, ls="--",
           label=rf"Original $\langle c\rangle = {orig['c_avg']:.4f}$")
ax.axhline(ens_mean["c_avg"], color="darkorange", lw=1.2, ls="--",
           label=rf"CM ensemble $\langle c\rangle = {ens_mean['c_avg']:.4f}$")
ax.set(xlabel=r"$k$", ylabel=r"$c(k)$", title="Clustering spectrum")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_ck_comparison.pdf"))
plt.show()
print("Saved a4_ck_comparison.pdf")


# ── Plot 5: Ensemble distributions ───────────────────────────────────────────
# For metrics where the original value is far outside the CM range
# (<c> and Q), the original is shown as a text annotation rather than
# an off-screen vline, so the histogram remains readable.

plot_metrics = [
    ("c_avg",      r"$\langle c \rangle$",                comparison["<c>"]["original"]),
    ("r",          r"Assortativity $r$",                  comparison["r"]["original"]),
    ("modularity", r"Modularity $Q$",                     comparison["Q"]["original"]),
    ("n_comm",     r"$n_{\mathrm{comm}}$",                comparison["n_comm"]["original"]),
    ("knn_uncorr", r"$\langle k^2\rangle/\langle k\rangle$", comparison["<k²>/<k>"]["original"]),
    ("E",          r"Edges $E$",                          comparison["E"]["original"]),
]

fig, axes = plt.subplots(2, 3, figsize=(11, 6))
axes = axes.flatten()

for ax, (key, label, orig_val) in zip(axes, plot_metrics):
    data     = ensemble[key]
    data_min = min(data)
    data_max = max(data)
    ax.hist(data, bins=15, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(ens_mean[key], color="black", lw=1.2, ls=":",
               label=f"CM mean: {ens_mean[key]:.3f}")

    # Only draw the original vline if it falls within the data range
    # (with 20% margin); otherwise annotate as text to avoid empty plots
    margin    = 0.20 * (data_max - data_min + 1e-9)
    in_range  = (data_min - margin) <= orig_val <= (data_max + margin)

    if in_range:
        ax.axvline(orig_val, color="tomato", lw=1.5, ls="--",
                   label=f"Original: {orig_val:.3f}")
        ax.legend(fontsize=6)
    else:
        ax.legend(fontsize=6)
        ax.annotate(
            f"Original:\n{orig_val:.3f}",
            xy=(0.97, 0.95), xycoords="axes fraction",
            ha="right", va="top", fontsize=6,
            color="tomato",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tomato",
                      alpha=0.8, lw=0.8),
        )

    ax.set(xlabel=label, ylabel="Count",
           title=f"{label}  (n={N_REALIZATIONS})")

plt.suptitle(f"CM ensemble distributions ({N_REALIZATIONS} realizations)",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_ensemble_distributions.pdf"),
            bbox_inches="tight")
plt.show()
print("Saved a4_ensemble_distributions.pdf")

print("\nAll plots saved to", RESULTS)