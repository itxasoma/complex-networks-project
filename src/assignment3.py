"""
assignment3.py: Site percolation: random failure and targeted attack
Measures the relative size of the GCC (S) and mean component size (chi)
as nodes are removed, for both the empirical network and the CM null model.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

exec(open("assignment1.py").read())   # gives G, N, E, degrees

# ── Percolation helpers ───────────────────────────────────────────────────────
def gcc_size(H):
    if H.number_of_nodes() == 0:
        return 0
    cc = sorted(nx.connected_components(H), key=len, reverse=True)
    return len(cc[0]) / H.number_of_nodes()

def mean_finite_size(H):
    """Mean component size  χ = Σ_s s²·n_s / Σ_s s·n_s  (excluding GCC)."""
    if H.number_of_nodes() == 0:
        return 0.0
    cc = sorted([len(c) for c in nx.connected_components(H)], reverse=True)
    if len(cc) < 2:
        return 0.0
    finite = cc[1:]
    total  = sum(finite)
    if total == 0:
        return 0.0
    return sum(s**2 for s in finite) / total

def percolation(H0, strategy="random", n_steps=50, seed=None):
    """
    Remove nodes one fraction step at a time.
    strategy: 'random' | 'degree' | 'betweenness'
    Returns arrays f (fraction removed), S (gcc/N), chi (mean finite size).
    """
    rng = np.random.default_rng(seed)
    H   = H0.copy()
    N0  = H.number_of_nodes()
    step_size = max(1, N0 // n_steps)

    f_arr, S_arr, chi_arr = [0.0], [gcc_size(H)], [mean_finite_size(H)]

    removed = 0
    while H.number_of_nodes() > step_size:
        if strategy == "random":
            victims = rng.choice(list(H.nodes()), size=step_size, replace=False).tolist()
        elif strategy == "degree":
            # always attack highest-degree nodes
            sorted_nodes = sorted(H.degree(), key=lambda x: x[1], reverse=True)
            victims = [n for n, _ in sorted_nodes[:step_size]]
        elif strategy == "betweenness":
            bc = nx.betweenness_centrality(H, normalized=True, seed=int(rng.integers(1e6)))
            sorted_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)
            victims = [n for n, _ in sorted_nodes[:step_size]]

        H.remove_nodes_from(victims)
        removed += step_size
        f_arr.append(removed / N0)
        S_arr.append(gcc_size(H))
        chi_arr.append(mean_finite_size(H))

    return np.array(f_arr), np.array(S_arr), np.array(chi_arr)

# ── Run on empirical network ──────────────────────────────────────────────────
n_realiz  = 5   # realisations for random removal (targeted is deterministic)
n_steps   = 60

print("Running random percolation on empirical network…")
rand_S, rand_chi = [], []
for i in range(n_realiz):
    f, S, chi = percolation(G, strategy="random", n_steps=n_steps, seed=i)
    rand_S.append(S); rand_chi.append(chi)
rand_S_mean   = np.mean(rand_S,   axis=0)
rand_chi_mean = np.mean(rand_chi, axis=0)

print("Running degree-targeted attack on empirical network…")
f_t, S_t, chi_t = percolation(G, strategy="degree", n_steps=n_steps, seed=0)

# ── Run on one CM null-model realisation ──────────────────────────────────────
deg_seq = list(degrees)
if sum(deg_seq) % 2 != 0:
    deg_seq[0] += 1
Gcm = nx.configuration_model(deg_seq, seed=0)
Gcm = nx.Graph(Gcm); Gcm.remove_edges_from(nx.selfloop_edges(Gcm))
Gcm = Gcm.subgraph(max(nx.connected_components(Gcm), key=len)).copy()

print("Running random percolation on CM…")
cm_rand_S = []
for i in range(n_realiz):
    f_cm, S_cm, _ = percolation(Gcm, strategy="random", n_steps=n_steps, seed=i+100)
    cm_rand_S.append(S_cm)
cm_rand_S_mean = np.mean(cm_rand_S, axis=0)

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# GCC size vs fraction removed
ax = axes[0]
ax.plot(f,   rand_S_mean,   label="Random (empirical)", color="steelblue")
ax.plot(f_t, S_t,           label="Targeted (empirical)", color="tomato", ls="--")
ax.plot(f,   cm_rand_S_mean,label="Random (CM)", color="darkorange", ls=":")
ax.set(xlabel="Fraction of nodes removed f",
       ylabel="Relative GCC size S",
       title="Percolation: GCC size")
ax.legend()

# Mean finite component size (susceptibility)
ax = axes[1]
ax.plot(f,   rand_chi_mean, label="Random (empirical)", color="steelblue")
ax.plot(f_t, chi_t,         label="Targeted (empirical)", color="tomato", ls="--")
ax.set(xlabel="Fraction of nodes removed f",
       ylabel=r"$\chi$  (mean finite component size)",
       title="Percolation: mean finite size")
ax.legend()

plt.tight_layout()
plt.savefig("../results/a3_percolation.png", dpi=150)
plt.show()

# ── Approximate percolation threshold ────────────────────────────────────────
# f_c ~ where S drops below 0.05
for threshold in [0.05, 0.01]:
    idx_r = np.where(rand_S_mean < threshold)[0]
    idx_t = np.where(S_t          < threshold)[0]
    fc_r = f[idx_r[0]]   if len(idx_r) else ">1"
    fc_t = f_t[idx_t[0]] if len(idx_t) else ">1"
    print(f"f_c (S<{threshold}):  random={fc_r}  targeted={fc_t}")

# theoretical f_c for ER / annealed (κ = <k²>/<k>):
k_mean_v  = degrees.mean()
k2_mean_v = (degrees**2).mean()
kappa     = k2_mean_v / k_mean_v
fc_theory = 1 - 1 / (kappa - 1)
print(f"\nTheoretical f_c (heterogeneous mean-field) = {fc_theory:.4f}")
print(f"κ = <k²>/<k> = {kappa:.4f}")

import csv, os
os.makedirs("../results", exist_ok=True)
with open("../results/a3_summary.csv", "w", newline="") as f_:
    w = csv.writer(f_)
    w.writerow(["metric", "value"])
    w.writerow(["kappa", round(kappa, 4)])
    w.writerow(["fc_theory_hmf", round(fc_theory, 4)])
print("\nDone. Results in ../results/")
