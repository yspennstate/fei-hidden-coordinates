"""check_table.py n HEX : min_i E_i/Inf_i, each ratio, H, I and monotonicity of the function with truth table HEX.

Bit t of int(HEX, 16) is the value at the point whose coordinate i is bit i of t (reading the string the other way
round complements every input, which changes no ratio).
"""
import sys
import numpy as np
from check_small_trees import walsh, phi

n, hx = int(sys.argv[1]), sys.argv[2]
N = int(hx, 16)
T = np.arange(1 << n)
val = np.array([(N >> int(t)) & 1 for t in T])
mono = all(val[t] <= val[t | (1 << i)] for t in T for i in range(n))
f = np.where(val == 1, -1.0, 1.0)
mu = walsh(f) ** 2
nz = mu[mu > 1e-300]
H = float(-(nz * np.log2(nz)).sum())
I = float((mu * np.array([bin(t).count("1") for t in T])).sum())
rat = []
for i in range(n):
    lo = T[(T >> i) & 1 == 0]
    Inf = mu[lo | (1 << i)].sum()
    rat.append(phi(mu[lo], mu[lo | (1 << i)]).sum() / Inf)
print(f"n {n} monotone {mono} min {min(rat):.12f} H {H:.9f} I {I:.8f} mean {f.mean():.6f}")
print("ratios", " ".join(f"{r:.9f}" for r in rat))
ones = [int(t) for t in T if val[t]]
mins = [t for t in ones if not any(val[t & ~(1 << i)] for i in range(n) if (t >> i) & 1)]
print("minimal true points", len(mins), ":", " ".join("".join(str(i) for i in range(n) if (t >> i) & 1) for t in mins))
zeros = [int(t) for t in T if not val[t]]
maxs = [t for t in zeros if not any(val[t | (1 << i)] == 0 for i in range(n) if not (t >> i) & 1)]
print("maximal false points", len(maxs), ": complements", " ".join("".join(str(i) for i in range(n) if not (t >> i) & 1) for t in maxs))
