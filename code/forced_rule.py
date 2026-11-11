"""A coordinate forced on the minterm-and-maxterm rule with large ratio.

tilde f_k = M(y) OR (x AND [y >= y0]) (check_gk.build with remove=True). Replace y_j, j <= m (the ones of y0), by
T_t = [at least t of 2t fresh variables] (shortest minterm t, shortest maxterm t+1, bias P = Pr[Bin(2t,1/2) >= t])
and y_j, j > m, by its dual T'_t = [at least t+1 of 2t] (minterm t+1, maxterm t, bias 1-P). Then the only shortest
minterm through x is x + the y0 blocks and the only shortest maxterm through x is x + the other blocks, so A & B = {x}.
By the composition theorem the ratio at x of the composed function is the ratio at x of tilde f_k under the product
measure with these biases. Part 1 checks both statements on the full composed truth table for small (k, t); part 2
tabulates the biased ratio. Usage: python forced_rule.py
"""
import sys
from math import comb

import numpy as np

from check_gk import analyse, build, xlog


def biased_weights(f, n, mus):
    """Squared coefficients of the +-1 function f in the orthonormal basis of the product measure with Pr[bit i] = mus[i]."""
    v = np.where(f, -1.0, 1.0)
    for i in range(n):          # bit i is axis n-1-i after reshape
        mu = mus[i]
        s = np.sqrt(mu * (1 - mu))
        v = v.reshape(1 << (n - 1 - i), 2, 1 << i)
        v = np.stack([(1 - mu) * v[:, 0] + mu * v[:, 1], s * (v[:, 1] - v[:, 0])], axis=1).reshape(-1)
    return v ** 2


def ratio_x(w, n):
    s = np.arange(1 << n)
    lo = s[(s & 1) == 0]
    p, q = w[lo], w[lo | 1]
    return float((xlog(p) + xlog(q) - xlog(p + q)).sum() / q.sum()), float(q.sum())


def bias(t):
    return sum(comb(2 * t, j) for j in range(t, 2 * t + 1)) / 4 ** t


def composed(k, t):
    f, nf = build(k, True)
    m = (k - 1) // 2
    nb = 2 * t
    n = 1 + k * nb
    z = np.arange(1 << n)
    pc = np.array([bin(v).count("1") for v in range(1 << nb)])
    idx = z & 1
    for j in range(k):
        c = pc[(z >> (1 + j * nb)) & ((1 << nb) - 1)]
        yj = (c >= t) if j < m else (c >= t + 1)
        idx = idx | (yj.astype(np.int64) << (1 + j))
    return f[idx], n


if __name__ == "__main__":
    print("part 1: full composed truth tables")
    print("k t  n   A&B            ratio_x(full)  ratio_x(biased tilde f_k)  min_i ratio")
    for k, t in ((3, 1), (5, 1), (7, 1), (9, 1), (3, 2), (5, 2), (3, 3)):
        F, n = composed(k, t)
        A, B, inf, E, ratio, _, mn = analyse(F, n)
        ab = [i for i in range(n) if A[i] and B[i]]
        f, nf = build(k, True)
        P = bias(t)
        rb, _ = ratio_x(biased_weights(f, nf, [0.5] + [P] * ((k - 1) // 2) + [1 - P] * ((k + 1) // 2)), nf)
        print(f"{k} {t} {n:<3} {str(ab):<14} {ratio[0]:12.9f}  {rb:12.9f}               {np.nanmin(ratio):.6f}")
        sys.stdout.flush()
    print("part 2: ratio at x of tilde f_k under the gadget biases (t = inf is the uniform measure)")
    ts = (1, 2, 5, 20, 100, 1000)
    print("k   " + "".join(f"t={t:<9}" for t in ts) + "t=inf")
    for k in range(3, 18, 2):
        f, nf = build(k, True)
        row = []
        for P in [bias(t) for t in ts] + [0.5]:
            row.append(ratio_x(biased_weights(f, nf, [0.5] + [P] * ((k - 1) // 2) + [1 - P] * ((k + 1) // 2)), nf)[0])
        print(f"{k:<3} " + "".join(f"{r:<11.5f}" for r in row))
        sys.stdout.flush()
