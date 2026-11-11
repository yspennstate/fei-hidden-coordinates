"""Exact ratio E_x/Inf_x of tilde f_k = M(y) OR (x AND [y >= y0]) under the product measure in which y_1..y_m have
bias P and y_{m+1}..y_k have bias 1-P (x uniform), k = 2m+1. This is the ratio at x of the composed function F_{k,t}
when P = Pr[Bin(2t,1/2) >= t]; P = 1/2 is tilde f_k itself.

Write the +-1 form as A(y) + D(y) chi_x. Then D = 1[y0] + 1[y1] (the two points of [y >= y0] outside M), so for
S inside a = {1..m} and b = {m+2..k}, with i = |S & a|, j = |S & b|, and s = sqrt(P(1-P)):
    D^(S) = s^(i+j) (-1)^j P^(2m-i-j),       D^(S) = 0 when m+1 is in S,
    A^(S) = [S empty] - 2 (MAJ^(S) - 1[y1]^(S)) - D^(S),   1[y1]^(S) = (1-P) D^(S),
    MAJ^(S) = s^(i+j) * sum_{c >= m+1} [z^c] (z-1)^(i+j) ((1-P)+Pz)^(m-i) (P+(1-P)z)^(m-j+1).
Every quantity is s^(i+j) times a rational, the s-powers cancel inside the entropy term, and Inf_x = P^(2m).
All arithmetic is exact (Fractions); only the final logs are floats.
Usage: python forced_exact.py [kmax] [t ...]     (t = 0 means P = 1/2)
"""
import sys
from fractions import Fraction
from math import comb, log, log1p, log2

LN2 = log(2)


def bias(t):
    if t == 0:
        return Fraction(1, 2)
    return Fraction(sum(comb(2 * t, j) for j in range(t, 2 * t + 1)), 4 ** t)


def pmul(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def ppow(a, n):
    out = [1]
    for _ in range(n):
        out = pmul(out, a)
    return out


def flog(fr):
    """log2 of a positive Fraction, safe for huge numerators and denominators."""
    return log2(fr.numerator) - log2(fr.denominator)


def log2_1p(x):
    """log2(1 + x) for a Fraction x >= 0. For small x the difference of two large logs in flog loses every digit
    (at k = 121 the ratio a^2/d^2 is about 1e17 and the weight of the a-term is as large), so use log1p."""
    try:
        xf = float(x)
    except OverflowError:
        return flog(1 + x)
    return log1p(xf) / LN2


def ratio(k, P):
    m = (k - 1) // 2
    u, w = P.numerator, P.denominator
    X = [w - u, u]          # w * ((1-P) + P z)
    Y = [u, w - u]          # w * (P + (1-P) z)
    Xp = [ppow(X, e) for e in range(m + 1)]
    Yp = [ppow(Y, e) for e in range(m + 2)]
    Zp = [ppow([-1, 1], e) for e in range(2 * m + 1)]
    Inf = P ** (2 * m)
    E = 0.0
    for i in range(m + 1):
        for j in range(m + 1):
            poly = pmul(pmul(Xp[m - i], Yp[m - j + 1]), Zp[i + j])
            maj = Fraction(sum(poly[m + 1:]), w ** (2 * m - i - j + 1))
            d = (-1) ** j * P ** (2 * m - i - j)
            a = (1 if i == j == 0 else 0) - 2 * (maj - (1 - P) * d) - d
            a2, d2 = a * a, d * d
            sc = (P * (1 - P)) ** (i + j) * comb(m, i) * comb(m, j) / Inf   # weight relative to Inf_x
            term = 0.0
            if a2:
                term += float(a2 * sc) * log2_1p(d2 / a2)
            term += float(d2 * sc) * log2_1p(a2 / d2)
            E += term
    return E, float(Inf)


if __name__ == "__main__":
    kmax = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    ts = [int(a) for a in sys.argv[2:]] or [1, 0]
    print("k    " + "".join(f"t={t:<10}" if t else "P=1/2       " for t in ts))
    for k in range(3, kmax + 1, 2):
        print(f"{k:<4} " + "".join(f"{ratio(k, bias(t))[0]:<12.6f}" for t in ts))
        sys.stdout.flush()
