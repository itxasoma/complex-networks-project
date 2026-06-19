"""
assignment1.py: Network reading and basic statistics.

Reads the edge list, removes self-loops and duplicates, extracts the GCC,
stores it with D, P1, V, and prints the degree list and basic metrics.
"""


import os

from network_utils import analyze_graph, load_gcc_from_csv, sampled_distances, save_csv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS = os.path.join(REPO_ROOT, "results")
EDGE_FILE = os.path.join(REPO_ROOT, "../lastfm_asia", "lastfm_asia_edges.csv")
os.makedirs(RESULTS, exist_ok=True)


G = load_gcc_from_csv(EDGE_FILE)
res = analyze_graph(G)
apl, diam = sampled_distances(G)


print(f"GCC  N={res['N']}  E={res['E']}")
print(f"<k>   = {res['k_avg']:.4f}")
print(f"<k2>  = {res['k2_avg']:.4f}")
print(f"kmax  = {res['k_max']}   kmin = {res['k_min']}")
print(f"Density = {res['density']:.6f}")
print(f"Global clustering C_Delta = {res['c_global']:.6f}")
print(f"Degree assortativity r = {res['r']:.4f}")
print(f"Average path length <l> = {apl:.4f}")
print(f"Diameter D = {diam}")

print(f"\n{'node':>8}  {'degree':>8}")
for i, k in enumerate(res["D"]):
    print(f"{i:>8}  {int(k):>8}")


save_csv(
    os.path.join(RESULTS, "a1_summary.csv"),
    ["metric", "value"],
    [
        ["N_gcc", res["N"]],
        ["E_gcc", res["E"]],
        ["<k>", round(res["k_avg"], 4)],
        ["<k2>", round(res["k2_avg"], 4)],
        ["kmin", res["k_min"]],
        ["kmax", res["k_max"]],
        ["density", round(res["density"], 6)],
        ["C_delta", round(res["c_global"], 6)],
        ["assortativity_r", round(res["r"], 4)],
        ["<l>", round(apl, 4)],
        ["diameter", diam],
    ],
)

save_csv(
    os.path.join(RESULTS, "a1_degree_list.csv"),
    ["node", "degree"],
    [[i, int(k)] for i, k in enumerate(res["D"])],
)

print("\nDone. Results in", RESULTS)