"""Read-once towers whose bottom level is a small monotone block B (MAJ3, MAJ5, other thresholds) instead of a variable.

The tower recursion of towers_mp.py needs, for each coordinate class of a child, only (Ci, Ii, Pn, count) and the
child's log p, log q. For a block B on k variables these come from its truth table: C(S) = (Fourier coefficient of the
0/1 indicator at S)^2, Ci = C({i}), Ii = sum over S not containing i of C(S u i), Pn = sum over nonempty S not
containing i of Phi(C(S), C(S u i)). Every level above the blocks is an AND or OR of disjoint copies, as before.
Checked against full truth tables for n <= 18, then the fan-ins are climbed for each block.
"""
import itertools, math, random, sys
import numpy as np
import mpmath as mp

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from towers_mp import Phi, comp, AND, OR, ratio

KMAX = 40


def threshold(k, t):
    """Truth table (index = bitmask) of [x_1 + ... + x_k >= t]."""
    return np.array([bin(m).count("1") >= t for m in range(1 << k)])


def wht_indicator(tb):
    N = len(tb)
    h = tb.astype(float).copy()                 # Fourier of the 0/1 indicator, chi_S = (-1)^{|S cap x|}
    s = 1
    while s < N:
        h = h.reshape(-1, 2, s)
        h = np.stack([h[:, 0] + h[:, 1], h[:, 0] - h[:, 1]], axis=1).reshape(N)
        s *= 2
    return h / N


def block_leaf(tb):
    N = len(tb); k = N.bit_length() - 1
    cnt = int(tb.sum())
    a = wht_indicator(tb)
    C = [mp.mpf(int(round(x * N))) ** 2 / N ** 2 for x in a]   # exact: N*a is an integer
    lp = mp.log(mp.mpf(cnt) / N); lq = mp.log(mp.mpf(N - cnt) / N)
    groups = {}
    for i in range(k):
        b = 1 << i
        Ci = C[b]
        Ii = sum(C[T | b] for T in range(N) if not T & b)
        Pn = sum(Phi(C[T], C[T | b]) for T in range(1, N) if not T & b)
        key = (mp.nstr(Ci, 30), mp.nstr(Ii, 30), mp.nstr(Pn, 30))
        if key in groups:
            groups[key][3] += 1
        else:
            groups[key] = [Ci, Ii, Pn, 1]
    return (lp, lq, tuple(tuple(v) for v in groups.values()))


def tower(base, fans, top="OR"):
    gates = []
    g = top
    for _ in fans:
        gates.append(g); g = "AND" if g == "OR" else "OR"
    gates.reverse()
    node = base
    for k, g in zip(fans, gates):
        node = (AND if g == "AND" else OR)([(node, k)])
    return node


def tt_ratio(tb, fans, top="OR"):
    k = len(tb).bit_length() - 1
    gates = []
    g = top
    for _ in fans:
        gates.append(g); g = "AND" if g == "OR" else "OR"
    gates.reverse()
    n = k * math.prod(fans)
    N = 1 << n
    x = ((np.arange(N)[:, None] >> np.arange(n)) & 1)
    idx = (x.reshape(N, -1, k) * (1 << np.arange(k))).sum(axis=2)
    vals = tb[idx]
    for kk, g in zip(fans, gates):
        vals = vals.reshape(N, -1, kk)
        vals = vals.all(axis=2) if g == "AND" else vals.any(axis=2)
    a = wht_indicator(vals.reshape(N))
    W = (2 * a) ** 2                            # f = 1 - 2c: f^(S) = -2 c^(S) for S nonempty
    W[0] = (1 - 2 * a[0]) ** 2
    best = math.inf
    for i in range(n):
        b = 1 << i
        T = np.arange(N)[(np.arange(N) & b) == 0]
        p, q = W[T], W[T | b]
        s = p + q
        with np.errstate(divide="ignore", invalid="ignore"):
            E = np.where(p > 0, p * np.log(s / np.where(p > 0, p, 1)), 0) + np.where(q > 0, q * np.log(s / np.where(q > 0, q, 1)), 0)
        best = min(best, E.sum() / q.sum())
    return best / math.log(2)


def climb(base, fans, top):
    fans = list(fans); best = ratio(tower(base, fans, top))
    improved = True
    while improved:
        improved = False
        for i in range(len(fans)):
            for k in range(2, KMAX + 1):
                if k == fans[i]:
                    continue
                t = fans[:]; t[i] = k
                r = ratio(tower(base, t, top))
                if r > best + 1e-13:
                    best, fans, improved = r, t, True
    return best, tuple(fans)


if __name__ == "__main__":
    bases = {"MAJ3": threshold(3, 2), "MAJ5": threshold(5, 3), "T(4,2)": threshold(4, 2), "T(4,3)": threshold(4, 3),
             "T(5,2)": threshold(5, 2), "T(5,4)": threshold(5, 4)}
    print("truth-table checks")
    for name, fans, top in [("MAJ3", (), "OR"), ("MAJ3", (2,), "OR"), ("MAJ3", (2,), "AND"), ("MAJ3", (3,), "OR"),
                            ("MAJ3", (2, 2), "OR"), ("MAJ3", (3, 2), "AND"), ("T(4,2)", (2, 2), "OR"),
                            ("MAJ5", (3,), "AND"), ("T(5,4)", (2,), "OR")]:
        tb = bases[name]
        a = ratio(tower(block_leaf(tb), fans, top)); b = tt_ratio(tb, fans, top)
        print(" ", name, fans, top, mp.nstr(a, 14), f"{b:.12f}", "ok" if abs(a - b) < 1e-9 else "MISMATCH", flush=True)
    random.seed(1)
    depths = [int(d) for d in sys.argv[1:]] or [1, 2, 3, 4, 5, 6]
    for name, tb in bases.items():
        base = block_leaf(tb)
        prev = None
        for d in depths:
            res = []
            for top in ("OR", "AND"):
                starts = [[2] * d, [3] * d, [4] * d] + [[random.randint(2, 10) for _ in range(d)] for _ in range(3)]
                if prev:
                    starts.append(list(prev) + [2]); starts.append([2] + list(prev))
                for s in starts:
                    res.append(climb(base, (s + [2] * d)[:d], top) + (top,))
            r, fans, top = max(res)
            prev = fans
            n = (len(tb).bit_length() - 1) * math.prod(fans)
            print(f"{name:7s} depth {d}: {float(r):.12f} fans {fans} top {top} n {n}", flush=True)
