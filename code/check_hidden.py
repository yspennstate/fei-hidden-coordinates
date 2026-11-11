"""Exact checks of every identity and inequality in Sections 3-6 of the paper, on explicit truth tables.

Bit j of an input index is variable j. Boolean functions are 0/1 arrays of length 2^N; the +-1 version is 1 - 2F.
All entropies in bits. Usage: python check_hidden.py   (about a minute; the largest table has 2^19 entries)
"""
import math

import numpy as np

ETA = 1 / math.log(2)


def h2(t):
    return 0.0 if t <= 0 or t >= 1 else -t * math.log2(t) - (1 - t) * math.log2(1 - t)


def spectrum(F):
    """Squared Fourier coefficients of 1 - 2F by a fast Walsh-Hadamard transform."""
    a = (1.0 - 2.0 * np.asarray(F, dtype=float)).copy()
    n = a.size
    h = 1
    while h < n:
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], axis=1).reshape(-1)
        h *= 2
    return (a / n) ** 2


def xlog(t):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    m = t > 0
    out[m] = -t[m] * np.log2(t[m])
    return out


def Phi(p, q):
    return xlog(p) + xlog(q) - xlog(np.asarray(p) + np.asarray(q))


def split(mu, i):
    m = mu.reshape(-1, 2, 1 << i)
    return m[:, 0, :].reshape(-1), m[:, 1, :].reshape(-1)


def erasure(mu, i):
    p, q = split(mu, i)
    return float(Phi(p, q).sum())


def influence(mu, i):
    return float(split(mu, i)[1].sum())


def level_two(mu, i, N):
    return float(sum(mu[(1 << i) | (1 << j)] for j in range(N) if j != i))


def bits(N):
    idx = np.arange(1 << N)
    return [(idx >> j) & 1 for j in range(N)]


def even_mass(mu):
    size = np.array([bin(s).count("1") for s in range(mu.size)])
    return float(mu[size % 2 == 0].sum())


def hidden(k):
    """f(y, x) = MAJ_k(y) OR (x AND y >= y0), y0 = ones on the first (k-1)/2 coordinates; x is variable k."""
    N = k + 1
    b = bits(N)
    w = sum(b[:k])
    cover = np.ones(1 << N, dtype=bool)
    for j in range((k - 1) // 2):
        cover &= b[j] == 1
    return ((w >= (k + 1) // 2) | ((b[k] == 1) & cover)).astype(np.int8)


def report(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" " + detail if detail else ""))
    return ok


ALL = []

# Section 3: the hidden coordinate.
print("== Section 3: hidden coordinate, exact identities and sum E_i <= 2 I")
for k in (1, 3, 5, 7, 9, 11, 13):
    N = k + 1
    mu = spectrum(hidden(k))
    d = 2.0 ** -k
    a = math.comb(k - 1, (k - 1) // 2) / 2 ** (k - 1)
    infs = [influence(mu, i) for i in range(N)]
    I = sum(infs)
    E = [erasure(mu, i) for i in range(N)]
    ok = abs(infs[k] - d) < 1e-12 and abs(I - k * a) < 1e-12 and abs(even_mass(mu) - d) < 1e-12
    ok &= all(abs(infs[j] - (a + d if j < (k - 1) // 2 else a - d)) < 1e-12 for j in range(k))
    ok &= all(e <= h2(d) + 1e-12 for e in E) and sum(E) <= 2 * I + 1e-12
    ALL.append(report(f"k={k}", ok, f"I={I:.6f} sumE/I={sum(E) / I:.6f} E_x/Inf_x={E[k] / d:.6f}"))

# Section 4: self-dual substitutions, the leaf formula, the leaf bound and the sum bound.
print("== Section 4: substitutions, leaf formula, E_leaf/Inf_leaf < 4, sum E_i <= 5 I")


def substituted(k, blocks):
    """blocks[j] in {1, 3, 5}: y_j is a single variable or MAJ_3 / MAJ_5 of its own fresh variables."""
    sizes = list(blocks)
    N = sum(sizes) + 1
    b = bits(N)
    ys, pos, leaves = [], 0, []
    for s in sizes:
        ys.append((sum(b[pos:pos + s]) >= (s + 1) // 2).astype(int))
        leaves.append(list(range(pos, pos + s)))
        pos += s
    x = b[N - 1]
    w = sum(ys)
    cover = np.ones(1 << N, dtype=bool)
    for j in range((k - 1) // 2):
        cover &= ys[j] == 1
    return ((w >= (k + 1) // 2) | ((x == 1) & cover)).astype(np.int8), leaves, N


for k, blocks in ((3, (3, 3, 3)), (3, (5, 3, 1)), (5, (3, 3, 3, 3, 1)), (5, (1, 5, 3, 1, 3)), (1, (5,)), (1, (3,))):
    F, leaves, N = substituted(k, blocks)
    muF = spectrum(F)
    muf = spectrum(hidden(k))
    d = 2.0 ** -k
    infF = [influence(muF, i) for i in range(N)]
    EF = [erasure(muF, i) for i in range(N)]
    ok = abs(EF[N - 1] - erasure(muf, k)) < 1e-12 and abs(infF[N - 1] - d) < 1e-12
    ok &= abs(even_mass(muF) - d) < 1e-12
    worst = 0.0
    for j, L in enumerate(leaves):
        dj = influence(muf, j)
        p, q = split(muf, j)
        for ell in L:
            s = len(L)
            tau = math.comb(s - 1, (s - 1) // 2) / 2 ** (s - 1)
            pred = float(Phi(p, tau * tau * q).sum())
            ok &= abs(EF[ell] - pred) < 1e-12 and abs(infF[ell] - tau * dj) < 1e-12
            ok &= EF[ell] <= min(h2(d), h2(tau * tau * dj)) + 1e-12
            worst = max(worst, EF[ell] / infF[ell])
    I = sum(infF)
    ok &= worst < 4 and sum(EF) <= 5 * I + 1e-12
    ALL.append(report(f"k={k} blocks={blocks}", ok, f"N={N} max leaf ratio={worst:.6f} sumE/I={sum(EF) / I:.6f}"))

# The constants in (5.1): B_k decreasing, B_3 < 4; the k = 1 bound.
Bk = [math.sqrt(2 * k * 2.0 ** -k) * (k + ETA) for k in range(3, 60, 2)]
ALL.append(report("B_k decreasing, B_3<4", all(x > y for x, y in zip(Bk, Bk[1:])) and Bk[0] < 4, f"B_3={Bk[0]:.6f}"))
t = np.linspace(1e-6, 1, 200001)
ALL.append(report("k=1 leaf bound < 3", float((t * np.log2(2 * math.e / t ** 2)).max()) < 3,
                  f"max={float((t * np.log2(2 * math.e / t ** 2)).max()):.6f}"))
ALL.append(report("(k+1)h2(2^-k)<=2I, k odd<=99",
                  all((k + 1) * h2(2.0 ** -k) <= 2 * k * math.comb(k - 1, (k - 1) // 2) / 2 ** (k - 1) for k in range(1, 100, 2))))

# Section 5: rules.
print("== Section 5: coordinate rules on the hidden coordinate")
for k in (5, 9, 13, 17):
    N = k + 1
    mu = spectrum(hidden(k))
    infs = [influence(mu, i) for i in range(N)]
    W2 = [level_two(mu, i, N) for i in range(N)]
    E = [erasure(mu, i) for i in range(N)]
    d = 2.0 ** -k
    ok = min(range(N), key=lambda i: infs[i]) == k and all(infs[i] > d + 1e-15 for i in range(k))
    ok &= all(abs(W2[i] - k * d * d) < 1e-15 for i in range(N))
    ALL.append(report(f"k={k} x has least influence, W2 all equal", ok, f"E_x/Inf_x={E[k] / d:.6f}"))

for k in (3, 5):
    F, leaves, N = substituted(k, (3,) * k)
    # minterms of the substituted function: x with a minterm of MAJ_3 in each of the first (k-1)/2 blocks (width k),
    # or minterms of MAJ_3 in (k+1)/2 blocks (width k+1). Check that every minimal true point of width k contains x.
    idx = np.arange(1 << N)
    wt = np.array([bin(v).count("1") for v in idx])
    true = F.astype(bool)
    minimal = [v for v in idx[true] if all(not true[v & ~(1 << j)] for j in range(N) if v >> j & 1)]
    shortest = min(wt[v] for v in minimal)
    sh = [v for v in minimal if wt[v] == shortest]
    count = [sum(v >> j & 1 for v in sh) for j in range(N)]
    ok = shortest == k and count[N - 1] == len(sh) and all(c < len(sh) for c in count[:N - 1])
    ALL.append(report(f"k={k} MAJ3 blocks: x in every shortest minterm", ok, f"{len(sh)} shortest minterms"))

for r in (1, 2, 3, 5, 8, 13, 20):
    tau = math.comb(2 * r, r) / 4 ** r
    ALL.append(report(f"tau(MAJ_{2 * r + 1})<=1/sqrt(r+1)", tau <= 1 / math.sqrt(r + 1) + 1e-15, f"{tau:.6f}"))

# Section 5 continued: complementary pivotal points.
print("== Complementary pivotal points: W2_x = 0, every other W2 > 0, E_x/Inf_x grows")
for k in (6, 8, 10, 12, 14, 16, 18):
    N = k + 1
    b = bits(N)
    w = sum(b[:k])
    y = sum(b[j] << j for j in range(k))
    y0 = (1 << (k // 2)) - 1
    y1 = ((1 << k) - 1) ^ y0
    P = (y == y0) | (y == y1)
    F = np.where(P, b[k], (w >= k // 2).astype(int)).astype(np.int8)
    mu = spectrum(F)
    infs = [influence(mu, i) for i in range(N)]
    W2 = [level_two(mu, i, N) for i in range(N)]
    d = 2.0 ** (1 - k)
    a0 = math.comb(k, k // 2) / 2 ** k
    ok = abs(infs[k] - d) < 1e-15 and W2[k] < 1e-28 and all(abs(infs[j] - a0) < 1e-12 and W2[j] > 1e-12 for j in range(k))
    # monotone check
    for j in range(N):
        lo = F.reshape(-1, 2, 1 << j)
        ok &= bool(np.all(lo[:, 0, :] <= lo[:, 1, :]))
    Ex = erasure(mu, k)
    ALL.append(report(f"k={k}", ok, f"E_x/Inf_x={Ex / d:.6f} min_i E_i/Inf_i={min(erasure(mu, i) / infs[i] for i in range(N)):.6f}"))

# Section 6: the exchangeable law with mu({i}) = Pr[i in U]^2 and E_i/Pr[i in U] -> infinity.
print("== The law of Section 6: E_i / delta exactly, against log2(n)/288")
for n in (100, 1000, 10000, 100000):
    p = 1.0 / n
    b = 0.25
    lp = [math.lgamma(n + 1) - math.lgamma(s + 1) - math.lgamma(n - s + 1) + s * math.log(p) + (n - s) * math.log1p(-p)
          for s in range(n + 1)]
    q = 1 - math.exp(lp[0]) - math.exp(lp[1])
    d = sum(math.exp(lp[s]) * s / n for s in range(2, n + 1)) / q        # rho[i in U]
    delta = (1 - math.sqrt(1 - 4 * b * d)) / 2
    # per-set mass of a set of size s >= 2 under b*rho: b p^s (1-p)^(n-s) / q
    def m(s):
        return 0.0 if s < 2 else b * math.exp(s * math.log(p) + (n - s) * math.log1p(-p)) / q
    # T not containing i, |T| = s. s = 0: (empty, {i}); s = 1: ({j}, {i,j}); s >= 2: both in the product part,
    # where Phi(m(s), m(s+1)) = (m(s) + m(s+1)) h2(p) and there are C(n-1, s) such T.
    Ei = float(Phi(np.array([1 - b - n * delta ** 2]), np.array([delta ** 2]))[0])
    Ei += (n - 1) * float(Phi(np.array([delta ** 2]), np.array([m(2)]))[0])
    for s in range(2, n):
        logc = math.lgamma(n) - math.lgamma(s + 1) - math.lgamma(n - s)
        logm = math.log(b) + s * math.log(p) + (n - s) * math.log1p(-p) - math.log(q)   # log m(s)
        Ei += math.exp(logc + logm - math.log1p(-p)) * h2(p)                         # m(s) + m(s+1) = m(s)/(1-p)
    marg = delta ** 2 + b * d
    ok = abs(marg - delta) < 1e-15 and Ei / delta >= math.log2(n) / 288
    ALL.append(report(f"n={n}", ok, f"E_i/delta={Ei / delta:.4f} bound={math.log2(n) / 288:.4f}"))

print("ALL PASS" if all(ALL) else "SOME CHECK FAILED")
