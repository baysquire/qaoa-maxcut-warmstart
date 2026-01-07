"""Warm-start strategies for QAOA based on classical solutions.

Provides two warm-start approaches:
1. state_warmstart: initialize QAOA initial state to the classical bitstring (product basis)
2. param_warmstart: initialize parameters (gammas/betas) around small perturbations informed by classical solution

We focus on clarity and keeping the design choices explicit.
"""
from typing import List, Tuple, Optional
import numpy as np
from .qaoa_baseline import QAOAOptimizer


def state_warmstart(classical_bitstring: List[int]) -> List[int]:
    """Return an initial product basis state for the QAOA circuit (list of 0/1)."""
    return classical_bitstring.copy()


def param_warmstart(p: int, seed: Optional[int] = 0, perturbation: float = 0.1) -> np.ndarray:
    """Produce initial params (gammas, betas) by small randomized perturbation around heuristic values.

    Heuristic: set small gamma values and betas near pi/4 to mix from classical solution.
    """
    rng = np.random.RandomState(seed)
    gammas = rng.normal(loc=0.2, scale=perturbation, size=p)
    betas = rng.normal(loc=np.pi / 4, scale=perturbation, size=p)
    return np.concatenate([gammas, betas])


def run_warmstarted_qaoa(graph, classical_bitstring: List[int], p: int = 1, strategy: str = "state", optimizer_method: str = "COBYLA", maxiter: int = 200):
    """Run QAOA with warm-start.

    strategy: 'state' or 'params' or 'both'
    returns same outputs as baseline optimize
    """
    n = graph.number_of_nodes()
    edges = list(graph.edges())
    qopt = QAOAOptimizer(edges, n, p)

    initial_state = None
    init_params = None
    if strategy in ("state", "both"):
        initial_state = state_warmstart(classical_bitstring)
    if strategy in ("params", "both"):
        init_params = param_warmstart(p=p, seed=42)

    final_params, final_value, history = qopt.optimize(initial_params=init_params, initial_state=initial_state, method=optimizer_method, maxiter=maxiter)
    return final_params, final_value, history


if __name__ == "__main__":
    # quick smoke test that uses a classical solution
    import networkx as nx
    from .graph_utils import greedy_maxcut
    G = nx.erdos_renyi_graph(6, 0.5, seed=1)
    bs, val = greedy_maxcut(G, seed=1, restarts=20)
    p = 1
    params, v, hist = run_warmstarted_qaoa(G, bs, p=p, strategy='state', maxiter=30)
    print(f"Warm-start final cut approx: {v}, classical cut: {val}")
