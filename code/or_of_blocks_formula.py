"""or_of_blocks_formula.py : E_i/Inf_i for an OR of disjoint blocks from one block's spectrum, no full truth table.

Write a_b for the 0/1 indicator that block b is false. Then f = -1 + 2 prod_b a_b (f = -1 when the OR is true), so for
T nonempty fhat(T) = 2 prod_b ahat_b(T_b), and fhat(empty) = -1 + 2 prod_b q_b with q_b = Pr[a_b = 1]. Phi is
homogeneous of degree one and sum_T ahat_b(T)^2 = q_b, so for a variable i of block 1, with A = ahat_1^2 and
R0 = prod_{b>1} ahat_b(empty)^2, Q = prod_{b>1} q_b:
    Inf_i = 4 Q sum_{S} A(S+i)
    E_i   = 4 Q sum_{S not i} Phi(A(S), A(S+i)) - 4 R0 Phi(A(empty), A(i)) + Phi(fhat(empty)^2, 4 R0 A(i)).
Blocks: T(j) = (x1 v x2)(x3 v x4)...(x_{2j-1} v x_{2j}) (AND of j OR2), and A(j) = x1 x2 ... xj.
Usage: python or_of_blocks_formula.py   (checks the formula against truth tables, then searches).
"""
import itertools
from functools import lru_cache
import numpy as np
from check_small_trees import walsh, phi


@lru_cache(None)
def block(kind, j):
    n = 2 * j if kind == "T" else j
    T = np.arange(1 << n)
    if kind == "T":
        val = np.ones(1 << n, dtype=bool)
        for t in range(j):
            val &= (((T >> (2 * t)) & 1) | ((T >> (2 * t + 1)) & 1)).astype(bool)
    else:
        val = (T == (1 << n) - 1)
    a = (~val).astype(float)                     # indicator that the block is false
    ah = walsh(a)                                # ahat(S) = E[a(x) chi_S(x)], chi_S = prod (-1)^{x_i}
    A = ah ** 2
    i = 0                                        # every variable of these blocks is equivalent to x1
    lo = T[(T >> i) & 1 == 0]
    return dict(n=n, q=float(a.mean()), A0=float(A[0]), Ai=float(A[1]), sumInf=float(A[lo | 1].sum()),
                sumPhi=float(phi(A[lo], A[lo | 1]).sum()))


def ratio(blocks, which):
    """blocks: list of (kind, j); which: index of the block holding the variable."""
    b1 = block(*blocks[which])
    rest = [block(*b) for k, b in enumerate(blocks) if k != which]
    Q = float(np.prod([r["q"] for r in rest])) if rest else 1.0
    R0 = float(np.prod([r["A0"] for r in rest])) if rest else 1.0
    f0 = -1 + 2 * b1["q"] * Q
    Inf = 4 * Q * b1["sumInf"]
    E = 4 * Q * b1["sumPhi"] - 4 * R0 * float(phi(np.array([b1["A0"]]), np.array([b1["Ai"]]))[0]) \
        + float(phi(np.array([f0 * f0]), np.array([4 * R0 * b1["Ai"]]))[0])
    return E / Inf, Inf


def min_ratio(blocks):
    return min(ratio(blocks, k)[0] for k in {b: k for k, b in enumerate(blocks)}.values())


def truth_table_ratio(blocks):
    n = sum(block(*b)["n"] for b in blocks)
    T = np.arange(1 << n, dtype=np.int64)
    val = np.zeros(1 << n, dtype=bool)
    pos = 0
    for kind, j in blocks:
        m = block(kind, j)["n"]
        if kind == "T":
            v = np.ones(1 << n, dtype=bool)
            for t in range(j):
                v &= (((T >> (pos + 2 * t)) & 1) | ((T >> (pos + 2 * t + 1)) & 1)).astype(bool)
        else:
            v = np.ones(1 << n, dtype=bool)
            for t in range(j):
                v &= ((T >> (pos + t)) & 1).astype(bool)
        val |= v
        pos += m
    mu = walsh(np.where(val, -1.0, 1.0)) ** 2
    out, pos = [], 0
    for kind, j in blocks:
        lo = T[(T >> pos) & 1 == 0]
        out.append(float(phi(mu[lo], mu[lo | (1 << pos)]).sum() / mu[lo | (1 << pos)].sum()))
        pos += block(kind, j)["n"]
    return min(out)


if __name__ == "__main__":
    print("formula against truth tables")
    for bl in ([("A", 2)] * 4, [("T", 3)] * 3, [("T", 2)] * 2 + [("A", 2)], [("A", 2)] * 2 + [("T", 3)] * 2,
               [("A", 3), ("T", 2), ("A", 2)], [("T", 4), ("A", 2), ("A", 2)]):
        print(f"  {bl}: formula {min_ratio(bl):.12f}  truth table {truth_table_ratio(bl):.12f}")
    print("m copies of T(j) (n = 2jm)")
    best = []
    for j in range(1, 9):
        row = []
        for m in range(1, 41):
            r = min_ratio([("T", j)] * m)
            row.append((r, m))
            best.append((r, 2 * j * m, f"{m} x T({j})"))
        r, m = max(row)
        print(f"  j={j}: best m={m:2d} n={2 * j * m:3d} ratio {r:.12f}")
    print("mixtures: a copies of T(j1) and b copies of T(j2), j1 < j2 <= 8, a, b <= 30")
    for j1, j2 in itertools.combinations(range(1, 9), 2):
        for a, b in itertools.product(range(1, 31), repeat=2):
            bl = [("T", j1)] * a + [("T", j2)] * b
            best.append((min_ratio(bl), 2 * (j1 * a + j2 * b), f"{a} x T({j1}) + {b} x T({j2})"))
    best.sort(reverse=True)
    print("BEST 12")
    for r, n, name in best[:12]:
        print(f"  {r:.12f} n={n} {name}")
    print("smallest n reaching each threshold")
    for thr in (3.70, 3.71, 3.72, 3.73, 3.74, 3.75):
        c = [(n, r, name) for r, n, name in best if r >= thr]
        print(f"  >= {thr}: " + (f"n={min(c)[0]} {min(c)[2]} {min(c)[1]:.12f}" if c else "none"))
