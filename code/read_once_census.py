"""read_once_census.py NMAX : max over read-once monotone functions on n variables of min_i E_i/Inf_i, for n <= NMAX.

A read-once monotone function is a tree whose leaves are distinct variables and whose internal nodes are AND or OR
gates with at least two children; adjacent gates may be taken of opposite type. The value of min_i E_i/Inf_i depends
only on the shape of the tree (relabelling variables permutes the ratios) and is the same for a tree and its dual
(the dual has the same Fourier weights), so it is enough to list shapes with an OR gate at the root.
Prints, for each n, the number of shapes, the best value and the best formula.
"""
import sys
from functools import lru_cache
import numpy as np
from check_small_trees import walsh, phi

LEAF = ()


@lru_cache(maxsize=None)
def shapes(n):
    """All shapes with n leaves, as sorted tuples of child shapes (a leaf is the empty tuple)."""
    if n == 1:
        return (LEAF,)
    out = set()

    def rec(left, minkey, acc):
        if left == 0:
            if len(acc) >= 2:
                out.add(tuple(acc))
            return
        for k in range(1, left + 1):
            if k == n:
                continue
            for s in shapes(k):
                key = (k, s)
                if key < minkey:
                    continue
                rec(left - k, key, acc + [s])

    rec(n, (0, ()), [])
    return tuple(sorted(out, key=lambda s: (size(s), s)))


@lru_cache(maxsize=None)
def size(s):
    return 1 if s == LEAF else sum(size(c) for c in s)


def evaluate(s, is_or, cols, pos):
    if s == LEAF:
        v = cols[pos[0]]
        pos[0] += 1
        return v
    vals = [evaluate(c, not is_or, cols, pos) for c in s]
    out = vals[0].copy()
    for v in vals[1:]:
        out = (out | v) if is_or else (out & v)
    return out


def formula(s, is_or, pos):
    if s == LEAF:
        pos[0] += 1
        return f"x{pos[0]}"
    parts = [formula(c, not is_or, pos) for c in s]
    if is_or:
        return " v ".join(parts)
    return "".join(p if len(p) <= 3 else f"({p})" for p in parts) if all("v" not in p for p in parts) \
        else "".join(p if "v" not in p else f"({p})" for p in parts)


def min_ratio(n, s):
    T = np.arange(1 << n)
    cols = [((T >> i) & 1).astype(bool) for i in range(n)]
    val = evaluate(s, True, cols, [0])
    mu = walsh(np.where(val, -1.0, 1.0)) ** 2
    best = np.inf
    rats = []
    for i in range(n):
        lo = T[(T >> i) & 1 == 0]
        Inf = mu[lo | (1 << i)].sum()
        r = phi(mu[lo], mu[lo | (1 << i)]).sum() / Inf
        rats.append(float(r))
        best = min(best, r)
    return float(best), rats


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    for n in range(2, nmax + 1):
        res = sorted(((min_ratio(n, s)[0], s) for s in shapes(n)), key=lambda t: -t[0])
        v, s = res[0]
        second = next((w for w, _ in res if w < v - 1e-9), None)
        print(f"n={n:2d} shapes {len(shapes(n)):6d} best {v:.12f} by {formula(s, True, [0])}"
              f"  (next value {second:.12f})" if second is not None else
              f"n={n:2d} shapes {len(shapes(n)):6d} best {v:.12f} by {formula(s, True, [0])}", flush=True)
