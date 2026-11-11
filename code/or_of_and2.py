"""or_of_and2.py KMAX : E_1/Inf_1 for the OR of k disjoint AND2 (the dual of tribes with k blocks of width 2), k <= KMAX.

All variables are equivalent, so one coordinate is enough. Exact truth table and Walsh-Hadamard transform.
"""
import sys
import numpy as np
from check_small_trees import walsh, phi

kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10
for k in range(1, kmax + 1):
    n = 2 * k
    T = np.arange(1 << n, dtype=np.int64)
    val = np.zeros(1 << n, dtype=bool)
    for j in range(k):
        val |= ((T >> (2 * j)) & 1).astype(bool) & ((T >> (2 * j + 1)) & 1).astype(bool)
    mu = walsh(np.where(val, -1.0, 1.0)) ** 2
    lo = T[(T & 1) == 0]
    Inf = mu[lo | 1].sum()
    E = phi(mu[lo], mu[lo | 1]).sum()
    print(f"k={k:2d} n={n:2d} Inf {Inf:.10f} (exact {0.5 * 0.75 ** (k - 1):.10f}) E/Inf {E / Inf:.12f} Pr[f=1] {val.mean():.6f}",
          flush=True)
