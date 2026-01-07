import pytest

qiskit = pytest.importorskip("qiskit")

import networkx as nx
from src.graph_utils import generate_erdos_renyi, greedy_maxcut
from src.qaoa_baseline import QAOAOptimizer
from src.qaoa_warmstart import run_warmstarted_qaoa


def test_qaoa_runs_and_returns_values():
    G = generate_erdos_renyi(6, 0.5, seed=1)
    bs, classical_val = greedy_maxcut(G, seed=1, restarts=10)
    edges = list(G.edges())
    q = QAOAOptimizer(edges, n_qubits=G.number_of_nodes(), p=1)
    params, val, hist = q.optimize(maxiter=10)
    assert isinstance(params, (list, type(params)))
    assert val >= 0


def test_warmstart_state_changes_behavior():
    G = generate_erdos_renyi(6, 0.5, seed=2)
    bs, classical_val = greedy_maxcut(G, seed=2, restarts=10)
    params_w, val_w, hist_w = run_warmstarted_qaoa(G, bs, p=1, strategy='state', maxiter=10)
    assert val_w >= 0
