"""Exact min_i E_i/Inf_i for symmetric read-once AND/OR towers, by recursion on 0/1 indicators.

For a 0/1 function c of its own variables write C(T) = c^(T)^2. A node carries
  p   = Pr[c = 1] = sum_T C(T)
  C0  = C(empty) = p^2
and for each coordinate class:  Ci = C({i}),  Ii = sum_{S} C(S u i),  Ph = sum_{S not ni} Phi(C(S), C(S u i)).
AND of independent indicators multiplies spectra exactly (no empty-set correction); Phi is 1-homogeneous,
so the other children enter through p (for Ii, Ph) and C0 (for Ci).  NOT changes only C(empty).
Final f = 1 - 2c: W(T) = 4 C(T) for T nonempty, W(empty) = (1-2p)^2.
"""
import itertools, math, sys
from functools import lru_cache


def Phi(p, q):
    s = p + q
    return (p * math.log(s / p) if p > 0 else 0.0) + (q * math.log(s / q) if q > 0 else 0.0)


def leaf():
    return (0.5, 0.25, ((0.25, 0.25, Phi(0.25, 0.25), 1),))   # classes: (Ci, Ii, Ph, count)


def NOT(node):
    p, C0, cl = node
    q = 1 - p
    return (q, q * q, tuple((Ci, Ii, Ph - Phi(C0, Ci) + Phi(q * q, Ci), k) for Ci, Ii, Ph, k in cl))


def AND(children):
    """children: list of (node, multiplicity)."""
    P = 1.0; C0 = 1.0
    for (p, c0, _), m in children:
        P *= p ** m; C0 *= c0 ** m
    out = []
    for j, ((p, c0, cl), m) in enumerate(children):
        op = p ** (m - 1); oc = c0 ** (m - 1)          # the other copies and children, formed
        for jj, ((pp, cc, _), mm) in enumerate(children):  # directly so an underflowed p cannot divide
            if jj != j:
                op *= pp ** mm; oc *= cc ** mm
        for Ci, Ii, Ph, k in cl:
            out.append((Ci * oc, Ii * op, Ph * op, k * m))
    return (P, C0, tuple(out))


def OR(children):
    return NOT(AND([(NOT(n), m) for n, m in children]))


def ratio(node):
    p, C0, cl = node
    w0 = (1 - 2 * p) ** 2
    best = math.inf
    for Ci, Ii, Ph, k in cl:
        if not Ii > 1e-280:            # bias so extreme that double precision cannot see the influence
            return 0.0
        E = 4 * Ph - 4 * Phi(C0, Ci) + Phi(w0, 4 * Ci)
        best = min(best, E / (4 * Ii))
    return best / math.log(2)          # entropy in bits, as in the paper


def nvars(node):
    return sum(k for *_, k in node[2])


def tower(fans, top="OR"):
    """fans listed from the bottom gate up; gates alternate, the top one is `top`."""
    gates = []
    g = top
    for _ in fans:
        gates.append(g); g = "AND" if g == "OR" else "OR"
    gates.reverse()
    node = leaf()
    for k, g in zip(fans, gates):
        node = (AND if g == "AND" else OR)([(node, k)])
    return node


# ---------- independent check: truth table ----------
def tt_ratio(fans, top="OR"):
    gates = []
    g = top
    for _ in fans:
        gates.append(g); g = "AND" if g == "OR" else "OR"
    gates.reverse()
    n = 1
    for k in fans:
        n *= k
    N = 1 << n
    import numpy as np
    x = ((np.arange(N)[:, None] >> np.arange(n)) & 1).astype(bool)
    vals = x
    for k, g in zip(fans, gates):
        vals = vals.reshape(N, -1, k)
        vals = vals.all(axis=2) if g == "AND" else vals.any(axis=2)
    c = vals.reshape(N).astype(float)
    f = 1 - 2 * c
    # Walsh-Hadamard with chi_S(x) = (-1)^{|S cap x|}
    h = f.copy()
    s = 1
    while s < N:
        h = h.reshape(-1, 2, s)
        h = np.stack([h[:, 0] + h[:, 1], h[:, 0] - h[:, 1]], axis=1).reshape(N)
        s *= 2
    W = (h / N) ** 2
    best = math.inf
    for i in range(n):
        b = 1 << i
        Inf = sum(W[T] for T in range(N) if T & b)
        E = sum(Phi(W[T], W[T | b]) for T in range(N) if not T & b)
        best = min(best, E / Inf)
    return best / math.log(2)


if __name__ == "__main__":
    # 1. truth-table checks
    for fans, top in [((2, 2), "OR"), ((2, 3), "AND"), ((3, 2, 2), "OR"), ((2, 2, 3), "AND"), ((2, 2, 2, 2), "OR")]:
        a, b = ratio(tower(fans, top)), tt_ratio(fans, top)
        print("check", fans, top, f"{a:.12f} {b:.12f}", "ok" if abs(a - b) < 1e-10 else "MISMATCH")
    # 2. the P8 value: OR of 4 copies of AND of 3 OR2 = fans (2,3,4), top OR
    print("P8 j=3 m=4:", f"{ratio(tower((2, 3, 4), 'OR')):.12f}  (paper 3.783969373526)")
    print("P8 j=5 m=8:", f"{ratio(tower((2, 5, 8), 'OR')):.12f}  (paper 3.949934743097)")
    # 3. towers by depth: best ratio with fan-ins up to K at every level
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    for d in range(2, 7):
        best = (0, None, None)
        rng = range(2, K + 1) if d <= 4 else range(2, min(K, 8) + 1)
        for fans in itertools.product(rng, repeat=d):
            for top in ("OR", "AND"):
                r = ratio(tower(fans, top))
                if r > best[0]:
                    best = (r, fans, top)
        r, fans, top = best
        n = math.prod(fans)
        print(f"depth {d}: {r:.12f} fans {fans} top {top} n {n:.4g}")
