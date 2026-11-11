"""Every monotone Boolean function on n <= 6 variables: maxima of min_i E_i/Inf_i, H/I and sum_i E_i/I, and the
worst E_i/Inf_i chosen by four coordinate rules (ties resolved against the rule).

Monotone functions on n variables are generated as pairs f0 <= f1 of monotone functions on n-1 variables
(f = f0 on x_{n-1}=0, f1 on x_{n-1}=1). Fourier coefficients come from a dense Hadamard product, entropies in bits.
Usage: python census_mono.py [nmax]   (default 6; n = 6 takes a few minutes and about 2 GB of memory)
"""
import functools
import json
import sys

import numpy as np

TIE = 1e-11


def monotone_tables(n):
    """All monotone f: {0,1}^n -> {0,1} as boolean arrays of shape (count, 2^n); bit j of the index is x_j."""
    if n == 0:
        return np.array([[False], [True]])
    prev = monotone_tables(n - 1)
    out = []
    for f1 in prev:
        below = ~np.any(prev & ~f1, axis=1)          # f0 <= f1 pointwise
        f0s = prev[below]
        out.append(np.concatenate([f0s, np.repeat(f1[None, :], len(f0s), axis=0)], axis=1))
    return np.concatenate(out, axis=0)


@functools.lru_cache(None)
def hadamard(n):
    idx = np.arange(1 << n)
    pc = np.vectorize(lambda v: bin(v).count("1"))(idx[:, None] & idx[None, :])
    return np.where(pc % 2 == 0, 1.0, -1.0)


def xlog(t):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, -t * np.log2(np.where(t > 0, t, 1.0)), 0.0)


def stats(tables, n):
    Hm = hadamard(n)
    f = np.where(tables, -1.0, 1.0)
    mu = (f @ Hm / (1 << n)) ** 2                     # spectral distribution, rows sum to 1
    size = np.array([bin(s).count("1") for s in range(1 << n)])
    H = xlog(mu).sum(axis=1)
    inf = np.stack([mu[:, [s for s in range(1 << n) if s >> i & 1]].sum(axis=1) for i in range(n)], axis=1)
    E = np.zeros_like(inf)
    W2 = np.zeros_like(inf)
    for i in range(n):
        lo = [s for s in range(1 << n) if not s >> i & 1]
        hi = [s | 1 << i for s in lo]
        p, q = mu[:, lo], mu[:, hi]
        E[:, i] = (xlog(p) + xlog(q) - xlog(p + q)).sum(axis=1)
        two = [s for s in range(1 << n) if size[s] == 2 and s >> i & 1]
        W2[:, i] = mu[:, two].sum(axis=1)
    return H, inf, E, W2


def chunks(n, size=50000):
    """The monotone tables on n variables in pieces, built from pairs f0 <= f1 on n-1 variables."""
    prev = monotone_tables(n - 1)
    buf, count = [], 0
    for f1 in prev:
        f0s = prev[~np.any(prev & ~f1, axis=1)]
        buf.append(np.concatenate([f0s, np.repeat(f1[None, :], len(f0s), axis=0)], axis=1))
        count += len(f0s)
        if count >= size:
            yield np.concatenate(buf, axis=0)
            buf, count = [], 0
    if buf:
        yield np.concatenate(buf, axis=0)


def run(n):
    res = {"n": n, "labelled": 0}
    best = {}

    def keep(name, vals, rows):
        k = int(np.argmax(vals))
        if name not in best or vals[k] > best[name][0]:
            best[name] = (float(vals[k]), "".join("1" if b else "0" for b in rows[k]))

    for rows in chunks(n):
        res["labelled"] += len(rows)
        H, inf, E, W2 = stats(rows, n)
        I = inf.sum(axis=1)
        ok = I > 1e-15
        rows, H, inf, E, W2, I = rows[ok], H[ok], inf[ok], E[ok], W2[ok], I[ok]
        rel = inf > 1e-15
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(rel, E / np.where(rel, inf, 1.0), np.nan)
            w2n = np.where(rel, W2 / np.where(rel, inf, 1.0), np.nan)
        keep("min_ratio", np.nanmin(ratio, axis=1), rows)
        keep("H_over_I", H / I, rows)
        keep("sumE_over_I", E.sum(axis=1) / I, rows)

        def rule(score, pick_max):
            s = np.where(rel, score, -np.inf if pick_max else np.inf)
            ext = (s.max if pick_max else s.min)(axis=1)[:, None]
            tied = rel & (np.abs(s - ext) <= TIE)
            return np.where(tied, ratio, -np.inf).max(axis=1)

        keep("rule_maxInf", rule(inf, True), rows)
        keep("rule_minInf", rule(inf, False), rows)
        keep("rule_minW2", rule(w2n, False), rows)
        keep("rule_maxW2", rule(w2n, True), rows)
    for k, (v, tt) in best.items():
        res[k] = {"value": round(v, 12), "truth_table_x0_first": tt}
    return res


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    out = [run(n) for n in range(1, nmax + 1)]
    print(json.dumps(out, indent=1))
