"""f_k = MAJ_k(y) OR (x AND [y >= y0]) and g_k = M(y) OR (x AND [y >= y0]), where y0 = {y_1..y_m}, m = (k-1)/2, and
M is MAJ_k with the minterm y1 = y0 + {y_{m+1}} removed. For odd k, prints whether x lies in a shortest minterm (A) and
in a shortest maxterm (B), Inf_x, E_x/Inf_x, min_i E_i/Inf_i and the worst E_i/Inf_i over A & B. Exact Walsh-Hadamard.
Bit 0 of a point is x; bits 1..k are y_1..y_k. Usage: python check_gk.py [kmax]   (default 13)
"""
import sys

import numpy as np


def wht(v):
    v = v.astype(np.float64).copy()
    h = 1
    while h < len(v):
        v = v.reshape(-1, 2, h)
        v = np.stack([v[:, 0] + v[:, 1], v[:, 0] - v[:, 1]], axis=1).reshape(-1)
        h *= 2
    return v


def xlog(t):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0
    out[m] = -t[m] * np.log2(t[m])
    return out


def build(k, remove):
    n = k + 1
    s = np.arange(1 << n)
    x = s & 1
    y = s >> 1
    wy = np.array([bin(v).count("1") for v in range(1 << k)])[y]
    m = (k - 1) // 2
    y0 = (1 << m) - 1
    y1 = y0 | (1 << m)
    M = wy >= (k + 1) // 2
    if remove:
        M = M & (y != y1)
    f = M | ((x == 1) & ((y & y0) == y0))
    return f, n


def analyse(f, n):
    s = np.arange(1 << n)
    w = np.array([bin(v).count("1") for v in s])
    # monotone check
    for i in range(n):
        lo = s[(s >> i & 1) == 0]
        assert not np.any(f[lo] & ~f[lo | 1 << i]), "not monotone"
    tw = w[f].min()
    fw = w[~f].max()
    A = [bool(np.any(f & (w == tw) & (s >> i & 1 == 1))) for i in range(n)]
    B = [bool(np.any(~f & (w == fw) & (s >> i & 1 == 0))) for i in range(n)]
    mu = (wht(np.where(f, -1.0, 1.0)) / (1 << n)) ** 2
    inf, E = np.zeros(n), np.zeros(n)
    for i in range(n):
        lo = s[(s >> i & 1) == 0]
        p, q = mu[lo], mu[lo | 1 << i]
        inf[i] = q.sum()
        E[i] = (xlog(p) + xlog(q) - xlog(p + q)).sum()
    ratio = E / inf
    ab = [ratio[i] for i in range(n) if A[i] and B[i]]
    return A, B, inf, E, ratio, (max(ab), min(ab), sum(1 for i in range(n) if A[i] and B[i])), ratio.min()


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 13
    print("k  fn  x_in_A x_in_B  Inf_x          E_x/Inf_x   min_i E_i/Inf_i  |A&B|  worst over A&B  best over A&B")
    for k in range(3, kmax + 1, 2):
        for name, rem in (("f", False), ("g", True)):
            f, n = build(k, rem)
            A, B, inf, E, ratio, (abw, abb, nab), mn = analyse(f, n)
            print(f"{k:<2} {name}   {str(A[0]):<6} {str(B[0]):<6} {inf[0]:.6e}  {ratio[0]:10.6f}  {mn:10.6f}"
                  f"       {nab:<5}  {abw:10.6f}      {abb:10.6f}")
        sys.stdout.flush()
