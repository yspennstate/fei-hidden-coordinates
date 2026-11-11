"""min_i E_i/Inf_i, H and I for a few read-once monotone formulas on 8 and 10 variables.

E_i and Inf_i are computed from the Fourier weights mu(T) = fhat(T)^2 by formula (eq:Ei) of the paper,
E_i = sum_{T not containing i} Phi(mu(T), mu(T+i)). Inputs are 0/1, output +-1 with f = -1 when the formula is true.
"""
import itertools
import numpy as np


def walsh(v):
    a = v.astype(float).copy()
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2 * h)
        x, y = a[:, :h].copy(), a[:, h:].copy()
        a[:, :h], a[:, h:] = x + y, x - y
        a = a.reshape(-1)
        h *= 2
    return a / len(a)


def phi(p, q):
    s = p + q
    out = np.zeros_like(s)
    m = (p > 0) & (q > 0)
    out[m] = p[m] * np.log2(s[m] / p[m]) + q[m] * np.log2(s[m] / q[m])
    return out


def stats(n, formula):
    pts = np.array(list(itertools.product([0, 1], repeat=n)))[:, ::-1]   # row t has bit i = x_i
    f = np.where(np.array([formula(x) for x in pts]), -1.0, 1.0)
    mu = walsh(f) ** 2
    T = np.arange(1 << n)
    nz = mu[mu > 1e-300]
    H = float(-(nz * np.log2(nz)).sum())
    I = float((mu * np.array([bin(t).count("1") for t in T])).sum())
    rat = []
    for i in range(n):
        lo = T[(T >> i) & 1 == 0]
        Inf = mu[lo | (1 << i)].sum()
        E = phi(mu[lo], mu[lo | (1 << i)]).sum()
        rat.append((E / Inf, Inf))
    return H, I, rat


cases = {
    "n=8  (x1x2 v x3x4)(x5x6 v x7x8)": (8, lambda x: ((x[0] & x[1]) | (x[2] & x[3])) & ((x[4] & x[5]) | (x[6] & x[7]))),
    "n=8  same, search labelling (x0x7 v x1x4)(x2x6 v x3x5)":
        (8, lambda x: ((x[0] & x[7]) | (x[1] & x[4])) & ((x[2] & x[6]) | (x[3] & x[5]))),
    "n=8  tribes 4x2 (AND of four OR2)": (8, lambda x: all(x[2 * j] | x[2 * j + 1] for j in range(4))),
    "n=10 tribes 5x2 (AND of five OR2)": (10, lambda x: all(x[2 * j] | x[2 * j + 1] for j in range(5))),
    "n=10 tribes 2x5 (OR of two AND5)": (10, lambda x: all(x[:5]) | all(x[5:])),
    "n=10 (x1 v x2)(x3 v x4) v (x5 v x6)(x7 v x8) v x9x10":
        (10, lambda x: ((x[0] | x[1]) & (x[2] | x[3])) | ((x[4] | x[5]) & (x[6] | x[7])) | (x[8] & x[9])),
    "n=12 OR of six AND2 (dual of tribes 6x2)": (12, lambda x: any(x[2 * j] & x[2 * j + 1] for j in range(6))),
    "n=18 OR of three (x1 v x2)(x3 v x4)(x5 v x6)":
        (18, lambda x: any(all(x[6 * b + 2 * j] | x[6 * b + 2 * j + 1] for j in range(3)) for b in range(3))),
}
for name, (n, fm) in cases.items() if __name__ == "__main__" else []:
    H, I, rat = stats(n, fm)
    r = sorted(set(round(float(a), 12) for a, _ in rat))
    infs = sorted(set(round(float(b), 10) for _, b in rat))
    print(f"{name}: min E_i/Inf_i {min(a for a, _ in rat):.12f}  ratios {r}  influences {infs}  H {H:.9f}  I {I:.8f}")
