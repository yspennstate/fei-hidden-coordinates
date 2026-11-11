"""Coordinate ascent over the fan-ins of plain read-once AND/OR towers with an OR at the top.

Each sweep tries every fan-in from 2 to 40 at one level, then every pair of levels moved by up to +-3 each, and moves to the
best improvement. The starts at depth d are the towers in KNOWN, the best three towers of depth d-1 with one level of fan-in 2
inserted at every position, and four random fan-in sets.
"""
import itertools, math, random, sys, time

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from towers_mp import tower, ratio
import mpmath as mp

KMAX = 40
random.seed(2)
cache = {}


def r(fans, top="OR"):
    key = (tuple(fans), top)
    if key not in cache:
        cache[key] = ratio(tower(list(fans), top))
    return cache[key]


def climb(fans, top="OR"):
    fans = list(fans); best = r(fans, top)
    while True:
        cand = None
        for i in range(len(fans)):
            for k in range(2, KMAX + 1):
                if k != fans[i]:
                    t = fans[:]; t[i] = k
                    v = r(t, top)
                    if v > best + 1e-13 and (cand is None or v > cand[0]):
                        cand = (v, t)
        if cand is None:
            for i, j in itertools.combinations(range(len(fans)), 2):
                for a in range(-3, 4):
                    for b in range(-3, 4):
                        if a == 0 or b == 0:
                            continue
                        t = fans[:]; t[i] += a; t[j] += b
                        if min(t) < 2 or max(t) > KMAX:
                            continue
                        v = r(t, top)
                        if v > best + 1e-13 and (cand is None or v > cand[0]):
                            cand = (v, t)
        if cand is None:
            return best, tuple(fans)
        best, fans = cand


KNOWN = {2: [(2, 8)], 3: [(2, 5, 8)], 4: [(2, 4, 4, 9)], 5: [(2, 4, 3, 4, 9), (2, 4, 3, 4, 8)],
         6: [(2, 3, 2, 3, 4, 8), (2, 4, 3, 3, 3, 6)], 7: [(2, 3, 2, 2, 2, 4, 9)], 8: [(2, 3, 2, 2, 2, 4, 6, 9)],
         9: [(2, 3, 2, 2, 2, 2, 2, 5, 10)], 10: [(2, 3, 2, 2, 2, 2, 2, 5, 7, 10)]}

if __name__ == "__main__":
    depths = [int(d) for d in sys.argv[1:]] or list(range(4, 11))
    prev = []
    for d in depths:
        t0 = time.time()
        starts = [list(s) for s in KNOWN.get(d, [])] + [list(s) for s in KNOWN.get(d - 1, [])]
        starts = [(s + [2] * d)[:d] if len(s) < d else s for s in starts]
        for p in prev:
            starts.append([2] + list(p)); starts.append(list(p) + [2])
            for i in range(d - 1):
                starts.append(list(p[:i]) + [2] + list(p[i:]))
        starts += [[random.randint(2, 10) for _ in range(d)] for _ in range(4)]
        seen = set(); res = []
        for s in starts:
            if tuple(s) in seen:
                continue
            seen.add(tuple(s))
            res.append(climb(s))
        res.sort(reverse=True)
        best = res[0]
        prev = [f for _, f in res[:3]]
        n = math.prod(best[1])
        print(f"depth {d:2d}: {mp.nstr(best[0], 13)} fans {best[1]} n {n}  ({len(seen)} starts, {time.time()-t0:.0f} s)",
              flush=True)
