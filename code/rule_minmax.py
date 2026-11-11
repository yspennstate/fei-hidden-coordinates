"""Coordinate rules built from shortest minterms and shortest maxterms, on every monotone function of n <= 6 variables.

For monotone f the shortest minterms are the true points of least weight, and the shortest maxterms are the
complements of the false points of greatest weight. For each f we form
    A = coordinates in some shortest minterm,  B = coordinates in some shortest maxterm,
and report the best (smallest) E_i/Inf_i over A, over B and over A & B (the friendly rules), and, ties and choices
resolved against the rule, the worst E_i/Inf_i over A, over B, over A & B (when it is
non-empty), and over A | B. Also counted: functions with A & B empty. Uses census_mono.py from the same folder.
Usage: python rule_minmax.py [nmax]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from census_mono import chunks, stats  # noqa: E402


def term_sets(rows, n):
    w = np.array([bin(s).count("1") for s in range(1 << n)])
    big = n + 1
    tw = np.where(rows, w, big).min(axis=1)                 # least weight of a true point
    fw = np.where(~rows, w, -1).max(axis=1)                 # greatest weight of a false point
    tmask = rows & (w[None, :] == tw[:, None])
    fmask = ~rows & (w[None, :] == fw[:, None])
    A = np.stack([tmask[:, [s for s in range(1 << n) if s >> i & 1]].any(axis=1) for i in range(n)], axis=1)
    B = np.stack([fmask[:, [s for s in range(1 << n) if not s >> i & 1]].any(axis=1) for i in range(n)], axis=1)
    return A, B, tw, n - fw


def run(n):
    res = {"n": n, "labelled": 0, "AB_empty": 0}
    best = {}

    def keep(name, vals, rows, extra=None):
        good = np.isfinite(vals)
        if not good.any():
            return
        k = int(np.argmax(np.where(good, vals, -np.inf)))
        if name not in best or vals[k] > best[name][0]:
            best[name] = (float(vals[k]), "".join("1" if b else "0" for b in rows[k]),
                          None if extra is None else extra(k))

    for rows in chunks(n):
        res["labelled"] += len(rows)
        H, inf, E, W2 = stats(rows, n)
        I = inf.sum(axis=1)
        ok = I > 1e-15
        rows, inf, E = rows[ok], inf[ok], E[ok]
        rel = inf > 1e-15
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(rel, E / np.where(rel, inf, 1.0), np.nan)
        A, B, minlen, maxlen = term_sets(rows, n)
        AB = A & B & rel
        res["AB_empty"] += int((~AB.any(axis=1)).sum())

        def worst(mask):
            return np.where(mask & rel, ratio, -np.inf).max(axis=1)

        def info(mask):
            return lambda k: {"minterm_len": int(minlen[k]), "maxterm_len": int(maxlen[k]),
                              "set": [int(i) for i in np.flatnonzero(mask[k])],
                              "ratios": [None if np.isnan(r) else round(float(r), 6) for r in ratio[k]]}

        def friendly(mask):
            return np.where(mask & rel, ratio, np.inf).min(axis=1)

        keep("best_in_A", friendly(A), rows, info(A))
        keep("best_in_B", friendly(B), rows, info(B))
        keep("best_in_A_and_B", friendly(AB), rows, info(AB))
        keep("rule_minterm", worst(A), rows, info(A))
        keep("rule_maxterm", worst(B), rows, info(B))
        keep("rule_min_and_max", worst(AB), rows, info(AB))
        keep("rule_min_or_max", worst(A | B), rows, info(A | B))
        glob = np.nanmin(ratio, axis=1)
        keep("min_ratio", glob, rows)
        gap = friendly(AB) - glob                     # cost of restricting the choice to A & B
        res["AB_misses_best"] = res.get("AB_misses_best", 0) + int((gap > 1e-9).sum())
        keep("AB_gap", gap, rows, info(AB))
        keep("AB_quotient", friendly(AB) / np.where(glob > 0, glob, np.nan), rows, info(AB))
    for k, (v, tt, ex) in best.items():
        res[k] = {"value": round(v, 12), "truth_table_x0_first": tt}
        if ex is not None:
            res[k]["witness"] = ex
    return res


if __name__ == "__main__":
    nmax = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    nmin = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    out = [run(n) for n in range(nmin, nmax + 1)]
    print(json.dumps(out, indent=1))
