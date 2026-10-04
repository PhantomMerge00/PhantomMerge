"""Shared statistics helpers for CCER mechanism verification."""
from __future__ import annotations

import math


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value; b=mask-only flip, c=unmask-only flip."""
    n = int(b) + int(c)
    if n <= 0:
        return 1.0
    k = min(int(b), int(c))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))
