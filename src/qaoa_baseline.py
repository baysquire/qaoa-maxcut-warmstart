"""Standard QAOA implementation for MaxCut using statevector simulator.

Key design choices (p=1 or 2 supported):
- Cost unitary implemented with `RZZGate(2*gamma)` per edge (corresponds to exp(-i gamma Z_i Z_j) up to global phase)
- Mixer implemented as single-qubit RX(2*beta)
- Expectation computed exactly from statevector probabilities * classical cost
- Optimizer: SciPy COBYLA (deterministic) by default, with callback logging

This module focuses on clarity and reproducibility for small graphs.
"""
from typing import List, Tuple, Optional
import numpy as np
# Robust Qiskit imports: Aer may be provided separately (qiskit-aer). We fall back to BasicAer if necessary.
try:
    from qiskit import QuantumCircuit, transpile
    try:
        from qiskit import Aer
    except Exception:
        # Try several common locations for Aer across packaging variants
        try:
            from qiskit.providers.aer import Aer
        except Exception:
            try:
                # Some distributions expose Aer as qiskit_aer
                from qiskit_aer import Aer
            except Exception:
                # Fall back to BasicAer if nothing else is available
                try:
                    from qiskit.providers.basicaer import BasicAer as Aer
                except Exception:
                    raise ImportError(
                        "Aer backend not available. Install with 'pip install qiskit-aer' or use conda-forge: 'conda install -c conda-forge qiskit-aer'."
                    )
except Exception as e:
    raise ImportError("qiskit is required to run QAOA routines. Install with 'pip install qiskit' or see README.") from e
from qiskit.circuit.library import RZZGate
from scipy.optimize import minimize
from .utils import probabilities_from_statevector, bitstring_from_int


def build_qaoa_circuit(n_qubits: int, graph_edges: List[Tuple[int,int]], p: int,
                       gammas: List[float], betas: List[float], initial_state: Optional[List[int]] = None) -> QuantumCircuit:
    """Construct QAOA circuit for given params.

    initial_state: optional product basis state as list of 0/1 (for warm-starting state)
    """
    qc = QuantumCircuit(n_qubits)
    # initial state
    if initial_state is None:
        # uniform superposition
        for q in range(n_qubits):
            qc.h(q)
    else:
        # prepare classical basis state by X gates where bit == 1
        for i, b in enumerate(initial_state):
            if int(b) == 1:
                qc.x(i)
    # QAOA layers
    for layer in range(p):
        gamma = gammas[layer]
        # cost unitary via RZZ gates per edge
        for (i, j) in graph_edges:
            qc.append(RZZGate(2 * gamma), [i, j])
        # mixer
        beta = betas[layer]
        for q in range(n_qubits):
            qc.rx(2 * beta, q)
    return qc


def compute_expectation_from_statevector(statevector: np.ndarray, graph_edges: List[Tuple[int,int]]) -> float:
    """Compute expected MaxCut value from statevector by summing probs * classical cut cost."""
    n = int(np.log2(len(statevector)))
    probs = probabilities_from_statevector(statevector)
    exp = 0.0
    for idx, p in enumerate(probs):
        if p < 1e-12:
            continue
        bs = bitstring_from_int(idx, n)
        val = 0.0
        for (i, j) in graph_edges:
            if bs[i] != bs[j]:
                val += 1.0
        exp += p * val
    return float(exp)


class QAOAOptimizer:
    """Encapsulate optimization routine and logging for QAOA."""
    def __init__(self, graph_edges: List[Tuple[int,int]], n_qubits: int, p: int,
                 backend=None):
        self.graph_edges = graph_edges
        self.n_qubits = n_qubits
        self.p = p
        if backend is None:
            backend = Aer.get_backend("statevector_simulator")
        self.backend = backend

    def expectation(self, params: np.ndarray, initial_state: Optional[List[int]] = None) -> float:
        gammas = params[: self.p]
        betas = params[self.p :]
        qc = build_qaoa_circuit(self.n_qubits, self.graph_edges, self.p, gammas.tolist(), betas.tolist(), initial_state)
        qc = transpile(qc, self.backend)
        result = self.backend.run(qc).result()
        sv = result.get_statevector(qc)
        exp = compute_expectation_from_statevector(sv, self.graph_edges)
        # we will *minimize* negative expectation to maximize cut
        return -exp

    def optimize(self, initial_params: Optional[np.ndarray] = None, initial_state: Optional[List[int]] = None, method: str = "COBYLA", maxiter: int = 200) -> Tuple[np.ndarray, float, dict]:
        """Optimize QAOA parameters.

        Args:
            initial_params: optional initial guess for [gammas, betas]
            initial_state: optional product-basis state to initialize the circuit (warm-start)
            method: optimizer name
            maxiter: maximum optimizer iterations
        """
        if initial_params is None:
            # random init small angles
            rng = np.random.RandomState(42)
            initial_params = rng.uniform(-0.5, 0.5, size=2 * self.p)
        history = {"params": [], "values": []}

        def callback(xk):
            history["params"].append(xk.copy())
            history["values"].append(self.expectation(xk, initial_state=initial_state))

        res = minimize(lambda x: self.expectation(x, initial_state=initial_state), x0=initial_params, method=method,
                       options={"maxiter": maxiter}, callback=callback)
        final_params = res.x
        final_value = -self.expectation(final_params, initial_state=initial_state)  # turn back to maximized value
        return final_params, final_value, history


if __name__ == "__main__":
    # quick smoke run on small graph
    edges = [(0,1),(1,2),(2,0),(2,3)]
    q = QAOAOptimizer(edges, n_qubits=4, p=1)
    params, val, hist = q.optimize(maxiter=20)
    print(f"Final cut (approx) = {val}")
