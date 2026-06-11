import csv

import networkx as nx
import numpy as np


def load_edgelist_csv(path):
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


def load_gcc_from_csv(path):
    G = nx.Graph()
    G.add_edges_from(load_edgelist_csv(path))
    G.remove_nodes_from(list(nx.isolates(G)))
    gcc_nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(gcc_nodes).copy()
    mapping = {n: i for i, n in enumerate(sorted(G.nodes()))}
    return nx.relabel_nodes(G, mapping)


def build_pointer_structure(G):
    N = G.number_of_nodes()
    E = G.number_of_edges()
    edge_list = list(G.edges())

    D = np.zeros(N, dtype=np.int64)
    for u, v in edge_list:
        D[u] += 1
        D[v] += 1

    P1 = np.zeros(N, dtype=np.int64)
    for i in range(1, N):
        P1[i] = P1[i - 1] + D[i - 1]

    P2 = P1.copy()
    V = np.zeros(2 * E, dtype=np.int64)
    for u, v in edge_list:
        V[P2[u]] = v
        P2[u] += 1
        V[P2[v]] = u
        P2[v] += 1

    return D, P1, V


def degree_distribution(D):
    k_max = int(D.max())
    nk = np.zeros(k_max + 1, dtype=np.int64)
    for k in D:
        nk[k] += 1

    Pk = nk / len(D)

    Pc = np.zeros(k_max + 1)
    Pc[k_max] = Pk[k_max]
    for k in range(k_max - 1, -1, -1):
        Pc[k] = Pc[k + 1] + Pk[k]

    return nk, Pk, Pc


def knn_by_degree(D, P1, V, nk):
    k_max = int(D.max())
    knn_acc = np.zeros(k_max + 1)

    for i in range(len(D)):
        ki = int(D[i])
        if ki == 0:
            continue
        neigh_deg_sum = 0
        for pos in range(P1[i], P1[i] + ki):
            neigh_deg_sum += int(D[V[pos]])
        knn_acc[ki] += neigh_deg_sum / ki

    knn = np.zeros(k_max + 1)
    for k in range(1, k_max + 1):
        if nk[k] > 0:
            knn[k] = knn_acc[k] / nk[k]

    return knn


def clustering_by_degree(D, P1, V, nk):
    N = len(D)
    k_max = int(D.max())
    adj_set = [set(V[P1[i]:P1[i] + D[i]]) for i in range(N)]

    ck_acc = np.zeros(k_max + 1)
    total_triangles = 0

    for i in range(N):
        ki = int(D[i])
        if ki < 2:
            continue

        neighbours = V[P1[i]:P1[i] + ki]
        tri_i = 0
        for a in range(ki):
            u = int(neighbours[a])
            for b in range(a + 1, ki):
                v = int(neighbours[b])
                if v in adj_set[u]:
                    tri_i += 1

        total_triangles += tri_i
        ck_acc[ki] += tri_i / (ki * (ki - 1) / 2)

    ck = np.zeros(k_max + 1)
    for k in range(2, k_max + 1):
        if nk[k] > 0:
            ck[k] = ck_acc[k] / nk[k]

    c_avg = float(ck_acc.sum()) / N
    n_triangles = total_triangles // 3

    return ck, c_avg, n_triangles


def analyze_graph(G):
    N = G.number_of_nodes()
    E = G.number_of_edges()

    D, P1, V = build_pointer_structure(G)
    nk, Pk, Pc = degree_distribution(D)
    knn = knn_by_degree(D, P1, V, nk)
    ck, c_avg, n_triangles = clustering_by_degree(D, P1, V, nk)

    k_min = int(D.min())
    k_max = int(D.max())
    k_avg = float(D.mean())
    k2_avg = float((D ** 2).mean())
    density = 2 * E / (N * (N - 1)) if N > 1 else 0.0
    c_global = nx.transitivity(G)
    r = nx.degree_assortativity_coefficient(G)
    knn_uncorr = k2_avg / k_avg

    return {
        "N": N,
        "E": E,
        "D": D,
        "P1": P1,
        "V": V,
        "nk": nk,
        "Pk": Pk,
        "Pc": Pc,
        "knn": knn,
        "ck": ck,
        "k_min": k_min,
        "k_max": k_max,
        "k_avg": k_avg,
        "k2_avg": k2_avg,
        "density": density,
        "c_avg": c_avg,
        "c_global": c_global,
        "n_triangles": n_triangles,
        "r": r,
        "knn_uncorr": knn_uncorr,
    }


def sampled_distances(G, sample_size=500, seed=42):
    N = G.number_of_nodes()

    if N <= 5000:
        apl = nx.average_shortest_path_length(G)
        diam = nx.diameter(G)
        return float(apl), int(diam)

    rng = np.random.default_rng(seed)
    sample = rng.choice(list(G.nodes()), size=min(sample_size, N), replace=False)

    lengths = []
    for src in sample:
        sp = nx.single_source_shortest_path_length(G, src)
        lengths.extend(sp.values())

    return float(np.mean(lengths)), int(max(lengths))


def save_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)