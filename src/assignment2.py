"""
assignment2.py: Null models: Erdős-Rényi and Configuration Model
Compares synthetic graphs with the empirical GCC from Assignment 1.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# ── Load GCC from assignment 1 output ────────────────────────────────────────
# Re-run the loading + GCC extraction (or import from a1 if structured as module)
exec(open("assignment1.py").read())   # gives G, N, E, degrees, k_mean

# ── Helpers ───────────────────────────────────────────────────────────────────
def degree_dist(H):
    degs = np.array([d for _, d in H.degree()])
    cnt  = Counter(degs)
    k    = np.array(sorted(cnt))
    pk   = np.array([cnt[ki] for ki in k]) / H.number_of_nodes()
    return k, pk, degs

def metrics(H):
    H = H.subgraph(max(nx.connected_components(H), key=len)).copy()
    degs = np.array([d for _, d in H.degree()])
    return {
        "N":   H.number_of_nodes(),
        "E":   H.number_of_edges(),
        "<k>": degs.mean(),
        "C_delta": nx.transitivity(H),
        "<c>":     nx.average_clustering(H),
        "r":       nx.degree_assortativity_coefficient(H),
    }

n_realiz = 10   # number of realisations to average

# ── Erdős-Rényi  G(N, p) ─────────────────────────────────────────────────────
p_er  = 2 * E / (N * (N - 1))
print(f"\nER model:  N={N}, p={p_er:.6f}, expected <k>={p_er*(N-1):.3f}")

er_metrics = []
er_kvals, er_pkvals = [], []
for _ in range(n_realiz):
    Ger = nx.erdos_renyi_graph(N, p_er, seed=None)
    Ger = Ger.subgraph(max(nx.connected_components(Ger), key=len)).copy()
    er_metrics.append(metrics(Ger))
    k, pk, _ = degree_dist(Ger)
    er_kvals.append(k); er_pkvals.append(pk)

# ── Configuration Model (preserves degree sequence) ──────────────────────────
deg_seq = list(degrees)
if sum(deg_seq) % 2 != 0:
    deg_seq[0] += 1   # fix parity

cm_metrics = []
cm_kvals, cm_pkvals = [], []
for _ in range(n_realiz):
    Gcm = nx.configuration_model(deg_seq, seed=None)
    Gcm = nx.Graph(Gcm)          # collapse multi-edges
    Gcm.remove_edges_from(nx.selfloop_edges(Gcm))
    Gcm = Gcm.subgraph(max(nx.connected_components(Gcm), key=len)).copy()
    cm_metrics.append(metrics(Gcm))
    k, pk, _ = degree_dist(Gcm)
    cm_kvals.append(k); cm_pkvals.append(pk)

# ── Print comparison table ────────────────────────────────────────────────────
def avg_metric(lst, key):
    return np.mean([m[key] for m in lst])

headers = ["metric", "empirical", "ER (avg)", "CM (avg)"]
rows = []
for key, label in [("<k>","<k>"),("C_delta","C_Δ"),("<c>","<c>"),("r","r")]:
    rows.append([label,
                 round(float({"<k>":k_mean,"C_delta":cc_global,"<c>":cc_avg,"r":r}[key]),4),
                 round(avg_metric(er_metrics, key),4),
                 round(avg_metric(cm_metrics, key),4)])

print(f"\n{'metric':<10} {'empirical':>12} {'ER':>12} {'CM':>12}")
for row in rows:
    print(f"{row[0]:<10} {row[1]:>12} {row[2]:>12} {row[3]:>12}")

# ── Plot degree distributions ─────────────────────────────────────────────────
k_emp, pk_emp, _ = degree_dist(G)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, title, kv_list, pkv_list in [
    (axes[0], "Empirical vs ER",  er_kvals,  er_pkvals),
    (axes[1], "Empirical vs CM",  cm_kvals,  cm_pkvals),
]:
    # plot all realisations lightly
    for k_, pk_ in zip(kv_list, pkv_list):
        ax.loglog(k_, pk_, color="grey", alpha=0.3, lw=1)
    ax.loglog(k_emp, pk_emp, "o", ms=4, color="steelblue", label="Empirical", zorder=5)
    ax.set(xlabel="k", ylabel="P(k)", title=title)
    ax.legend()

plt.tight_layout()
plt.savefig("../results/a2_degree_dists.png", dpi=150)
plt.show()

# ── Clustering spectrum comparison ───────────────────────────────────────────
def ck_spectrum(H):
    ck = {}
    for n in H.nodes():
        k = H.degree(n)
        ck.setdefault(k, []).append(nx.clustering(H, n))
    ks = np.array(sorted(ck))
    cv = np.array([np.mean(ck[k]) for k in ks])
    return ks, cv

ck_k_emp, ck_v_emp = ck_spectrum(G)

# use one CM realisation for the spectrum
Gcm_one = nx.configuration_model(deg_seq, seed=42)
Gcm_one = nx.Graph(Gcm_one); Gcm_one.remove_edges_from(nx.selfloop_edges(Gcm_one))
Gcm_one = Gcm_one.subgraph(max(nx.connected_components(Gcm_one), key=len)).copy()
ck_k_cm, ck_v_cm = ck_spectrum(Gcm_one)

plt.figure(figsize=(5, 4))
plt.loglog(ck_k_emp[ck_k_emp>1], ck_v_emp[ck_k_emp>1], "o", ms=4,
           label="Empirical", color="steelblue")
plt.loglog(ck_k_cm[ck_k_cm>1],  ck_v_cm[ck_k_cm>1],  "s", ms=4,
           label="CM", color="darkorange", alpha=0.7)
plt.xlabel("k"); plt.ylabel("c(k)")
plt.title("Clustering spectrum: empirical vs CM")
plt.legend(); plt.tight_layout()
plt.savefig("../results/a2_ck_comparison.png", dpi=150)
plt.show()

import csv
with open("../results/a2_summary.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(headers)
    w.writerows(rows)
print("\nDone. Results in ../results/")
