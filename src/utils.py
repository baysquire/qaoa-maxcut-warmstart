"""Utility helpers for QAOA MaxCut experiments.

Includes bitstring helpers, seeding helpers, and a small Result dataclass.
"""
from typing import List, Tuple
import numpy as np


def bitstring_from_int(x: int, n: int) -> List[int]:
    return [(x >> i) & 1 for i in range(n)][::-1]


def int_from_bitstring(bs: List[int]) -> int:
    x = 0
    for b in bs:
        x = (x << 1) | int(b)
    return x


def probabilities_from_statevector(statevector: np.ndarray) -> np.ndarray:
    probs = np.abs(statevector) ** 2
    # ensure normalization up to numerical noise
    probs = probs / probs.sum()
    return probs
