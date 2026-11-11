"""or_of_blocks.py NMAX : min_i E_i/Inf_i for ORs of disjoint blocks drawn from a few small read-once shapes, n <= NMAX.

Blocks: A2 = x1x2, A3 = x1x2x3, C4 = (x1 v x2)(x3 v x4), T6 = (x1 v x2)(x3 v x4)(x5 v x6). Variables inside one block
type are equivalent, so one coordinate per block type is evaluated. Prints the ten best combinations.
"""
import sys
import itertools
import numpy as np
from check_small_trees import walsh, phi

BLOCKS = {"A2": 2, "A3": 3, "C4": 4, "T6": 6}


def block_value(name, cols):
    if name == "A2":
        return cols[0] & cols[1]
    if name == "A3":
        return cols[0] & cols[1] & cols[2]
    if name == "C4":
        return (cols[0] | cols[1]) & (cols[2] | cols[3])
    return (cols[0] | cols[1]) & (cols[2] | cols[3]) & (cols[4] | cols[5])


def evaluate(combo):
    n = sum(BLOCKS[b] for b in combo)
    T = np.arange(1 << n, dtype=np.int64)
    cols = [((T >> i) & 1).astype(bool) for i in range(n)]
    val = np.zeros(1 << n, dtype=bool)
    first = {}
    pos = 0
    for b in combo:
        first.setdefault(b, pos)
        val |= block_value(b, cols[pos:pos + BLOCKS[b]])
        pos += BLOCKS[b]
    mu = walsh(np.where(val, -1.0, 1.0)) ** 2
    rats = {}
    for b, i in first.items():
        lo = T[(T >> i) & 1 == 0]
        rats[b] = float(phi(mu[lo], mu[lo | (1 << i)]).sum() / mu[lo | (1 << i)].sum())
    return n, min(rats.values()), rats


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    names = list(BLOCKS)
    res = []
    for counts in itertools.product(range(11), range(7), range(6), range(4)):
        n = sum(c * BLOCKS[b] for c, b in zip(counts, names))
        if n > nmax or sum(counts) < 2:
            continue
        combo = [b for c, b in zip(counts, names) for _ in range(c)]
        n, m, rats = evaluate(combo)
        res.append((m, n, counts, rats))
        print(f"n={n:2d} {dict(zip(names, counts))} min {m:.12f}", flush=True)
    res.sort(reverse=True)
    print("BEST")
    for m, n, counts, rats in res[:10]:
        print(f"{m:.12f} n={n} {dict(zip(names, counts))} {rats}")
