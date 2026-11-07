
import csv
import numpy as np
import networkx as nx
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

CLUSTER_EDGES = [(1,2),(1,3),(1,4),(1,5),(2,5),(2,6),(2,7),(3,4),(4,5),(5,6),(6,7),(8,3),(8,4),(8,5),(9,5),(9,6),(9,7),(8,9)]
CLUSTER_SIZE = 9
TOPOLOGY = "linear"        # "ring" або "linear"
# Регулярні зв'язки: вузол i кластера k -- вузол i кластера k+1
REGULAR = [(7, 3), (2, 1), (9, 8)]   # вузол a кластера k -- вузол b кластера k+1
# Нерегулярні зв'язки (резервні): кластер k -- кластер k+SKIP
SKIP = 2
IRREGULAR = [(5, 5)]   # вузол a кластера k -- вузол b кластера k+SKIP
MAX_NODES = 1000



def node(k, i):
    """Глобальний номер вузла i (1..6) кластера k (0..)."""
    return k * CLUSTER_SIZE + i - 1


def build(num_clusters, irregular=True):
    n = num_clusters * CLUSTER_SIZE
    edges = set()

    def add(u, v):
        if u != v:
            edges.add((min(u, v), max(u, v)))

    for k in range(num_clusters):
        for a, b in CLUSTER_EDGES:
            add(node(k, a), node(k, b))

    def neighbor(k, step):
        if TOPOLOGY == "ring":
            return (k + step) % num_clusters
        return k + step if k + step < num_clusters else None

    for k in range(num_clusters):
        m = neighbor(k, 1)
        if m is not None and m != k:
            for a, b in REGULAR:
                add(node(k, a), node(m, b))
        m = neighbor(k, SKIP)
        if irregular and m is not None and m != k:
            for a, b in IRREGULAR:
                add(node(k, a), node(m, b))
    return n, edges


def metrics(n, edges):
    A = lil_matrix((n, n), dtype=np.int8)
    for u, v in edges:
        A[u, v] = A[v, u] = 1
    A = A.tocsr()
    dist = shortest_path(A, method="D", directed=False, unweighted=True)
    finite = dist[np.isfinite(dist)]
    D = int(finite.max())
    aD = finite.sum() / (n * (n - 1))
    S = int(np.asarray(A.sum(axis=1)).max())
    C = len(edges)
    T = 2 * aD / S            # як у прикладі ЛР1
    return D, aD, S, C, T


def main():
    rows = []
    max_clusters = MAX_NODES // CLUSTER_SIZE
    steps = sorted(set(list(range(1, 11)) + list(range(15, max_clusters + 1, 10)) + [max_clusters]))
    for k in steps:
        n, edges = build(k)
        D, aD, S, C, T = metrics(n, edges)
        # для порівняння: лише регулярні зв'язки
        n0, edges0 = build(k, irregular=False)
        D0, aD0, S0, C0, T0 = metrics(n0, edges0)
        rows.append((n, k, D, round(aD, 3), S, C, round(T, 3),
                     D0, round(aD0, 3), S0, C0, round(T0, 3)))
        print(f"N={n:4d} n={k:3d} | D={D:3d} aD={aD:7.3f} S={S:2d} C={C:5d} T={T:6.3f}"
              f" | regular only: D={D0:3d} aD={aD0:7.3f} S={S0:2d} C={C0:5d} T={T0:6.3f}")
    with open("results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "n", "D", "avgD", "S", "C", "T",
                    "D_reg", "avgD_reg", "S_reg", "C_reg", "T_reg"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
