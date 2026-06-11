"""
assignment4-plots.py: Plots for Assignment 4 (CM analysis).

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
import numpy as np

from network_utils import analyze_graph, load_gcc_from_csv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
EDGE_FILE = os.path.join(REPO_ROOT, "lastfm_asia", "lastfm_asia_edges.csv")

plt.style.use(os.path.join(SCRIPT_DIR, "mplstyle", "science.mplstyle"))


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


comparison = {}
with open(os.path.join(RESULTS, "a4_comparison.csv")) as f:
    for row in csv.DictReader(f):
        comparison[row["metric"]] = {
            "original": float(row["original"]),
            "cm_single": float(row["cm_single"]),
            "cm_mean": float(row["cm_mean"]),
            "cm_std": float(row["cm_std"]),
        }

ensemble = {}
with open(os.path.join(RESULTS, "a4_ensemble.csv")) as f:
    for row in csv.DictReader(f):
        for k, v in row.items():
            ensemble.setdefault(k, []).append(float(v))

ens_mean = {k: float(np.mean(v)) for k, v in ensemble.items()}
N_REALIZATIONS = len(ensemble["E"])

print(f"Loaded {N_REALIZATIONS} ensemble realizations.")


G_orig = load_gcc_from_csv(EDGE_FILE)
orig = analyze_graph(G_orig)

rng = np.random.default_rng(42)
G_cm1 = configuration_model(orig["D"], rng)
cm1 = analyze_graph(G_cm1)


def nz(res, start=1):
    return np.array([k for k in range(start, res["k_max"] + 1) if res["nk"][k] > 0])


def nz_knn(res):
    return np.array([k for k in range(1, res["k_max"] + 1) if res["nk"][k] > 0 and res["knn"][k] > 0])


def nz_ck(res):
    return np.array([k for k in range(2, res["k_max"] + 1) if res["nk"][k] > 0 and res["ck"][k] > 0])


ks_o = nz(orig)
ks_c = nz(cm1)


# P(k)
fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_o, orig["Pk"][ks_o], "o", ms=3.5, label="Original")
ax.loglog(ks_c, cm1["Pk"][ks_c], "s", ms=3.5, label="CM (single)")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P(k)$")
ax.set_title(r"Degree distribution $P(k)$")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_pk_comparison.pdf"))
plt.close()


# CCDF
fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_o, orig["Pc"][ks_o], color="steelblue", label="Original")
ax.loglog(ks_c, cm1["Pc"][ks_c], color="darkorange", ls="--", label="CM (single)")
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$P_c(k)$")
ax.set_title("CCDF")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_ccdf_comparison.pdf"))
plt.close()


# k_nn(k)
ks_knn_o = nz_knn(orig)
ks_knn_c = nz_knn(cm1)

fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_knn_o, orig["knn"][ks_knn_o], "o", ms=3.5, label="Original")
ax.loglog(ks_knn_c, cm1["knn"][ks_knn_c], "s", ms=3.5, label="CM (single)")
ax.axhline(
    orig["knn_uncorr"],
    color="black",
    lw=1.2,
    ls="--",
    label=rf"$\langle k^2 \rangle / \langle k \rangle = {orig['knn_uncorr']:.2f}$",
)
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$k_{\mathrm{nn}}(k)$")
ax.set_title("Average nearest-neighbour degree")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_knn_comparison.pdf"))
plt.close()


# c(k)
ks_ck_o = nz_ck(orig)
ks_ck_c = nz_ck(cm1)

fig, ax = plt.subplots(figsize=(5, 4))
ax.loglog(ks_ck_o, orig["ck"][ks_ck_o], "o", ms=3.5, label="Original")
ax.loglog(ks_ck_c, cm1["ck"][ks_ck_c], "s", ms=3.5, label="CM (single)")
ax.axhline(
    orig["c_avg"],
    color="steelblue",
    lw=1.2,
    ls="--",
    label=rf"Original $\langle c \rangle = {orig['c_avg']:.4f}$",
)
ax.axhline(
    ens_mean["c_avg"],
    color="darkorange",
    lw=1.2,
    ls="--",
    label=rf"CM ensemble $\langle c \rangle = {ens_mean['c_avg']:.4f}$",
)
ax.set_xlabel(r"$k$")
ax.set_ylabel(r"$c(k)$")
ax.set_title("Clustering spectrum")
ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_ck_comparison.pdf"))
plt.close()


# Ensemble histograms
plot_metrics = [
    ("c_avg", r"$\langle c \rangle$", comparison["<c>"]["original"]),
    ("r", r"Assortativity $r$", comparison["r"]["original"]),
    ("modularity", r"Modularity $Q$", comparison["Q"]["original"]),
    ("n_comm", r"$n_{\mathrm{comm}}$", comparison["n_comm"]["original"]),
    ("knn_uncorr", r"$\langle k^2 \rangle / \langle k \rangle$", comparison["<k2>/<k>"]["original"]),
    ("E", r"Edges $E$", comparison["E"]["original"]),
]

fig, axes = plt.subplots(2, 3, figsize=(11, 6))
axes = axes.flatten()

for ax, (key, label, orig_val) in zip(axes, plot_metrics):
    data = ensemble[key]
    data_min = min(data)
    data_max = max(data)

    ax.hist(data, bins=15, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(ens_mean[key], color="black", lw=1.2, ls=":", label=f"CM mean: {ens_mean[key]:.3f}")

    margin = 0.20 * (data_max - data_min + 1e-9)
    in_range = (data_min - margin) <= orig_val <= (data_max + margin)

    if in_range:
        ax.axvline(orig_val, color="tomato", lw=1.5, ls="--", label=f"Original: {orig_val:.3f}")
        ax.legend(fontsize=6)
    else:
        ax.legend(fontsize=6)
        ax.annotate(
            f"Original:\n{orig_val:.3f}",
            xy=(0.97, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=6,
            color="tomato",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="tomato", alpha=0.8, lw=0.8),
        )

    ax.set_xlabel(label)
    ax.set_ylabel("Count")
    ax.set_title(f"{label}  (n={N_REALIZATIONS})")

plt.suptitle(f"CM ensemble distributions ({N_REALIZATIONS} realizations)", fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "a4_ensemble_distributions.pdf"), bbox_inches="tight")
plt.close()

print("\nAll plots saved to", RESULTS)