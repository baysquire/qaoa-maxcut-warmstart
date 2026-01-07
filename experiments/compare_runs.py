"""Experiment driver to compare random initialization vs warm-start QAOA.

Produces console output summarizing final cut approximation and convergence traces.
Minimal plotting is included for qualitative comparison (matplotlib), but experiments are kept small for clarity.
"""
import sys
import os
# allow running this script directly from repository root by adding project root to PYTHONPATH
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from src.graph_utils import generate_erdos_renyi, greedy_maxcut, evaluate_cut
from src.qaoa_baseline import QAOAOptimizer
from src.qaoa_warmstart import run_warmstarted_qaoa, state_warmstart, param_warmstart


def run_experiment(n=8, p_edge=0.5, seed=42, p_level=1):
    G = generate_erdos_renyi(n, p_edge, seed=seed)
    classical_bs, classical_val = greedy_maxcut(G, seed=seed, restarts=50)
    print(f"Classical greedy cut value: {classical_val}")

    edges = list(G.edges())
    qopt = QAOAOptimizer(edges, n, p_level)

    # baseline (random init)
    params_b, val_b, hist_b = qopt.optimize(maxiter=60)
    print(f"Baseline QAOA final approx cut: {val_b}")

    # warm-start: state strategy
    params_w, val_w, hist_w = run_warmstarted_qaoa(G, classical_bs, p=p_level, strategy='state', maxiter=60)
    print(f"State warm-start QAOA final approx cut: {val_w}")

    # warm-start: param strategy
    init_params = param_warmstart(p=p_level, seed=seed)
    params_wp, val_wp, hist_wp = qopt.optimize(initial_params=init_params, maxiter=60)
    print(f"Param warm-start QAOA final approx cut: {val_wp}")

    # plot (values were stored as negative expectation in history; convert)
    plt.figure(figsize=(6,4))
    for name, hist in [("baseline", hist_b), ("state", hist_w), ("param", hist_wp)]:
        vals = -np.array(hist['values']) if 'values' in hist and len(hist['values'])>0 else np.array([])
        if vals.size>0:
            plt.plot(vals, label=name)
    plt.hlines(classical_val, 0, 60, colors='k', linestyles='dashed', label='classical')
    plt.xlabel('Optimizer callback step')
    plt.ylabel('Cut value (approx)')
    plt.legend()
    plt.title('Baseline vs Warm-start QAOA (qualitative)')
    plt.tight_layout()
    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'compare_runs.png'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved comparison plot to: {out_path}")
    # plt.show()  # disabled for non-interactive runs


if __name__ == "__main__":
    run_experiment(n=8, p_edge=0.45, seed=1, p_level=1)
