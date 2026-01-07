"""Graph utilities for MaxCut experiments.

Provides graph generation, classical greedy MaxCut heuristics, and evaluation helpers.
Designed for small graphs (n <= 12) used for research/prototyping.
"""

from typing import Tuple, List, Optional
import networkx as nx
import random
import numpy as np


def set_seed(seed: int):
    """Set Python, NumPy and NetworkX seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def generate_erdos_renyi(n: int, p: float = 0.5, seed: int = None) -> nx.Graph:
    """Generate an undirected Erdos-Renyi graph with unit weights.

    Args:
        n: number of nodes
        p: edge probability
        seed: optional random seed

    Returns:
        networkx.Graph with integer nodes 0..n-1 and edge attribute 'weight' = 1
    """
    if seed is not None:
        set_seed(seed)
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def evaluate_cut(graph: nx.Graph, bitstring: List[int]) -> float:
    """Compute the MaxCut value for a given bitstring.

    bitstring: list of 0/1 assignments, length==n
    Returns total weight of edges crossing the cut.
    """
    total = 0.0
    for u, v, data in graph.edges(data=True):
        if bitstring[u] != bitstring[v]:
            total += data.get("weight", 1.0)
    return float(total)


def greedy_maxcut(graph: nx.Graph, seed: int = None, restarts: int = 10) -> Tuple[List[int], float]:
    """Simple greedy MaxCut with random restarts.

    Heuristic: iterate nodes in random order; assign each node to side that
    increases the cut most given previous assignments (ties broken randomly).

    Returns best found bitstring and its cut value.
    """
    if seed is not None:
        set_seed(seed)

    n = graph.number_of_nodes()
    nodes = list(graph.nodes())

    best_bs = None
    best_val = -1.0

    for r in range(restarts):
        random.shuffle(nodes)
        assignment = {node: 0 for node in nodes}
        # build assignment incrementally
        for node in nodes:
            # compute gain for assigning 0 or 1
            gain0 = 0.0
            gain1 = 0.0
            for nbr in graph.neighbors(node):
                w = graph[node][nbr].get("weight", 1.0)
                # if neighbor not assigned, treat as 0
                nbr_val = assignment.get(nbr, 0)
                if 0 != nbr_val:
                    gain0 += w
                if 1 != nbr_val:
                    gain1 += w
            # pick best
            if gain0 > gain1:
                assignment[node] = 0
            elif gain1 > gain0:
                assignment[node] = 1
            else:
                assignment[node] = random.choice([0, 1])
        bs = [assignment[i] for i in range(n)]
        val = evaluate_cut(graph, bs)
        if val > best_val:
            best_val = val
            best_bs = bs
    return best_bs, best_val


def gw_style_maxcut(graph: nx.Graph, repeats: int = 50, embed_dim: Optional[int] = None, seed: int = None) -> Tuple[List[int], float]:
    """A Goemans–Williamson-style heuristic using randomized hyperplane rounding on a spectral embedding.

    Notes:
    - This is a heuristic *in the spirit of* GW: we compute a low-dimensional
      spectral embedding (top eigenvectors of adjacency matrix) and perform
      randomized hyperplane rounding to produce cuts. It avoids heavy SDP
      solvers and is suitable for small research experiments.

    Args:
        graph: networkx graph
        repeats: number of random hyperplane trials
        embed_dim: embedding dimension (default min(n, 10))
        seed: optional seed for randomness

    Returns:
        best_bitstring, best_cut_value
    """
    if seed is not None:
        set_seed(seed)
    n = graph.number_of_nodes()
    if embed_dim is None:
        embed_dim = min(n, 10)

    # build adjacency matrix
    A = nx.to_numpy_array(graph, nodelist=range(n), weight='weight')
    # symmetric eigen-decomposition
    vals, vecs = np.linalg.eigh(A)
    # take top eigenvectors corresponding to largest eigenvalues
    top_idx = np.argsort(vals)[-embed_dim:]
    emb = vecs[:, top_idx]

    best_bs = None
    best_val = -1.0
    rng = np.random.RandomState(seed)
    for _ in range(repeats):
        r = rng.normal(size=(embed_dim,))
        projections = emb.dot(r)
        bs = [1 if x > 0 else 0 for x in projections]
        val = evaluate_cut(graph, bs)
        if val > best_val:
            best_val = val
            best_bs = bs
    return best_bs, best_val


if __name__ == "__main__":
    # quick smoke test
    G = generate_erdos_renyi(8, 0.4, seed=42)
    bs_greedy, val_greedy = greedy_maxcut(G, seed=42, restarts=20)
    bs_gw, val_gw = gw_style_maxcut(G, repeats=100, seed=42)
    print(f"Greedy found cut {val_greedy} on graph with {G.number_of_edges()} edges")
    print(f"GW-style heuristic found cut {val_gw}")
