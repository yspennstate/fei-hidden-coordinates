"""Coordinate ascent over the fan-ins of alternating read-once towers, depths 2..10, fan-ins 2..40."""
import math, random, sys
sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from towers_mp import tower, ratio

KMAX = 40
random.seed(1)


def climb(fans, top):
    fans = list(fans); best = ratio(tower(fans, top))
    improved = True
    while improved:
        improved = False
        for i in range(len(fans)):
            for k in range(2, KMAX + 1):
                if k == fans[i]:
                    continue
                t = fans[:]; t[i] = k
                r = ratio(tower(t, top))
                if r > best + 1e-13:
                    best, fans, improved = r, t, True
    return best, tuple(fans)


prev = None
for d in range(2, 11):
    res = []
    for top in ("OR", "AND"):
        starts = [[2] * d, [3] * d, [4] * d]
        if prev:
            starts.append([2] + list(prev[1:]) if len(prev) == d - 1 else [2] * d)
            starts.append(list(prev) + [2])
        starts += [[random.randint(2, 10) for _ in range(d)] for _ in range(6)]
        for s in starts:
            s = (s + [2] * d)[:d]
            res.append(climb(s, top) + (top,))
    r, fans, top = max(res)
    prev = fans
    print(f"depth {d:2d}: {float(r):.12f} fans {fans} top {top} n {math.prod(fans):.4g}", flush=True)
