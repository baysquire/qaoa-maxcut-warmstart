"""Smoke test to check imports and report missing dependencies gracefully."""
import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print("Checking imports for project modules...")
try:
    from src import graph_utils, qaoa_baseline, qaoa_warmstart
    print("Imported src modules OK.")
except Exception as e:
    print("Failed to import project modules:", e)

try:
    import qiskit
    print("Qiskit available:", qiskit.__version__)
except Exception as e:
    print("Qiskit not available. Install with: pip install -r requirements.txt")

print("Done.")
