"""For f = x AND g (g monotone on other variables, p = Pr[g = 1]) the erasure entropy of x depends only on p:
    E_x / Inf_x = 2(1-p) + [eta((1-p)^2) + eta(p^2) - eta((1-p)^2 + p^2)] / p,   Inf_x = p.
Since h2(q) <= 2 sqrt(q(1-q)), the bracket is at most 2p(1-p), so E_x / Inf_x <= 4(1-p); both are checked on a grid.
Checked here against exact transforms for random monotone g, and the right side maximised over p.
"""
import numpy as np

from check_gk import analyse, xlog


def formula(p):
    return 2 * (1 - p) + (xlog((1 - p) ** 2) + xlog(p ** 2) - xlog((1 - p) ** 2 + p ** 2)) / p


def random_monotone(k, rng):
    """Up-closure of a few random points."""
    s = np.arange(1 << k)
    g = np.zeros(1 << k, dtype=bool)
    for _ in range(rng.integers(1, 6)):
        v = rng.integers(0, 1 << k)
        g |= (s & v) == v
    return g


rng = np.random.default_rng(1)
worst = 0.0
for trial in range(300):
    k = int(rng.integers(2, 10))
    g = random_monotone(k, rng)
    s = np.arange(1 << (k + 1))
    f = (s & 1 == 1) & g[s >> 1]
    A, B, inf, E, ratio, _, _ = analyse(f, k + 1)
    p = g.mean()
    worst = max(worst, abs(ratio[0] - formula(p)), abs(inf[0] - p))
print("max |exact - formula| over 300 random g:", worst)
ps = np.linspace(1e-6, 1, 2000001)
v = formula(ps)
i = int(np.argmax(v))
print("max over p of the formula: %.9f at p = %.6f; limit p->0: %.6f; p = 1/2: %.6f" % (v[i], ps[i], formula(1e-9), formula(0.5)))
q = ps ** 2 / ((1 - ps) ** 2 + ps ** 2)
h = xlog(q) + xlog(1 - q)
print("max over p of formula / (4(1-p)) for p < 1: %.6f" % np.max(v[:-1] / (4 * (1 - ps[:-1]))))
print("max over q of h2(q) - 2 sqrt(q(1-q)): %.3e" % np.max(h - 2 * np.sqrt(q * (1 - q))))
