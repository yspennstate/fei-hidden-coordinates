"""Depth-6 threshold-block towers searched as in climb2.py: single moves, then pair moves of +-3, from several starts."""
import itertools, math, random, sys, time

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from towers_mp import ratio
from mixed_towers import threshold, block_leaf, tower
import mpmath as mp

KMAX = 40
random.seed(3)


def climb(base, fans, top, cache):
    def r(t):
        k = tuple(t)
        if k not in cache:
            cache[k] = ratio(tower(base, list(t), top))
        return cache[k]
    fans = list(fans); best = r(fans)
    while True:
        cand = None
        for i in range(len(fans)):
            for k in range(2, KMAX + 1):
                if k != fans[i]:
                    t = fans[:]; t[i] = k; v = r(t)
                    if v > best + 1e-13 and (cand is None or v > cand[0]):
                        cand = (v, t)
        if cand is None:
            for i, j in itertools.combinations(range(len(fans)), 2):
                for a in range(-3, 4):
                    for b in range(-3, 4):
                        if a and b:
                            t = fans[:]; t[i] += a; t[j] += b
                            if 2 <= min(t) and max(t) <= KMAX:
                                v = r(t)
                                if v > best + 1e-13 and (cand is None or v > cand[0]):
                                    cand = (v, t)
        if cand is None:
            return best, tuple(fans)
        best, fans = cand


if __name__ == "__main__":
    blocks = {"MAJ3": threshold(3, 2), "MAJ5": threshold(5, 3), "T(4,2)": threshold(4, 2)}
    known = [(2, 4, 3, 3, 3, 6), (3, 3, 3, 3, 4, 8), (2, 3, 2, 3, 4, 8), (2, 4, 3, 4, 8, 2), (2, 2, 2, 2, 2, 2)]
    for name, tb in blocks.items():
        t0 = time.time(); base = block_leaf(tb); k = len(tb).bit_length() - 1
        res = []
        for top in ("OR", "AND"):
            cache = {}
            starts = [list(s) for s in known] + [[random.randint(2, 10) for _ in range(6)] for _ in range(3)]
            for s in starts:
                res.append(climb(base, s, top, cache) + (top,))
        v, fans, top = max(res)
        print(f"{name:7s} depth 6: {mp.nstr(v, 13)} fans {fans} top {top} n {k * math.prod(fans)} ({time.time()-t0:.0f} s)",
              flush=True)
