"""Cancellation-free version of read_once_towers.py, in mpmath.

Node = (lp, lq, classes), lp = log Pr[c=1], lq = log Pr[c=0], both exact (one is computed with expm1/log1p of the other).
Class = (Ci, Ii, Pn, count) with Pn = sum over NONEMPTY S not containing i of Phi(C(S), C(S u i)); the empty-set
term Phi(p^2, Ci) is added only where needed, so NOT (which changes C(empty) alone) leaves the class untouched.
AND, child j with others' product op = prod p (so the others' C(empty) product is op^2):
    Ci -> Ci op^2,  Ii -> Ii op,  Pn -> op Pn + op (1 - op) Phi(p_j^2, Ci).
Final f = 1 - 2c:  E_i = 4 Pn + Phi((p - q)^2, 4 Ci),  Inf_i = 4 Ii.
"""
import math
import mpmath as mp

mp.mp.dps = 40


def Phi(a, b):
    s = a + b
    t = mp.mpf(0)
    if a > 0:
        t += a * (mp.log(s) - mp.log(a))
    if b > 0:
        t += b * (mp.log(s) - mp.log(b))
    return t


def comp(l):
    """log(1 - e^l) for l < 0, accurate at both ends."""
    return mp.log(-mp.expm1(l)) if l > -mp.log(2) else mp.log1p(-mp.exp(l))


def leaf():
    h = mp.mpf(1) / 4
    return (mp.log(0.5), mp.log(0.5), ((h, h, mp.mpf(0), 1),))


def NOT(n):
    return (n[1], n[0], n[2])


def AND(children):
    tot = sum(m * n[0] for n, m in children)
    out = []
    for n, m in children:
        lop = tot - n[0]                       # log of the others' product of p (other copies included)
        op = mp.exp(lop)
        one_minus = -mp.expm1(lop)
        p2 = mp.exp(2 * n[0])
        for Ci, Ii, Pn, k in n[2]:
            out.append((Ci * op * op, Ii * op, op * Pn + op * one_minus * Phi(p2, Ci), k * m))
    return (tot, comp(tot), tuple(out))


def OR(children):
    return NOT(AND([(NOT(n), m) for n, m in children]))


def tower(fans, top="OR"):
    gates = []
    g = top
    for _ in fans:
        gates.append(g); g = "AND" if g == "OR" else "OR"
    gates.reverse()
    node = leaf()
    for k, g in zip(fans, gates):
        node = (AND if g == "AND" else OR)([(node, k)])
    return node


def ratio(node):
    lp, lq, cl = node
    w0 = (mp.exp(lp) - mp.exp(lq)) ** 2
    best = None
    for Ci, Ii, Pn, k in cl:
        r = (4 * Pn + Phi(w0, 4 * Ci)) / (4 * Ii) / mp.log(2)
        best = r if best is None else min(best, r)
    return best


if __name__ == "__main__":
    from read_once_towers import tower as tower_f, ratio as ratio_f
    for fans, top in [((2, 2), "OR"), ((2, 3), "AND"), ((3, 2, 2), "OR"), ((2, 5, 8), "OR"), ((2, 4, 4, 9), "OR"),
                      ((2, 4, 3, 4, 8), "OR"), ((2, 3, 2, 3, 4, 8), "OR")]:
        a, b = ratio(tower(fans, top)), ratio_f(tower_f(fans, top))
        print(fans, top, mp.nstr(a, 15), f"{b:.12f}", "ok" if abs(a - b) < 1e-9 else "DIFF")
    for fans, top in [((53, 2, 10), "OR"), ((17, 5, 3, 3, 12), "OR")]:
        print("was inf:", fans, mp.nstr(ratio(tower(fans, top)), 15))
