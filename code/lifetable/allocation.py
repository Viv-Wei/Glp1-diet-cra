import numpy as np
from .settings import SEXES, ST


def allocate(X, F, G, Tm, first, C, mode, alloc, select, tier2, prio):

    P1 = sum(X[(sx, st, "E")].sum() for sx, st in prio)
    Et = Tm + (
        P1 if select else sum(X[(sx, st, "E")].sum() for sx in SEXES for st in ST)
    )
    K = C * Et
    if mode in ("baseline",) and not first:
        return Et, K, 0.0

    def move(keys, frac, part="all"):
        for key in keys:
            kT = (key[0], key[1], "T")
            if part == "all":
                mv = X[key] * frac
                mvf = F[key] * frac if F is not None else None
            elif part == "dm":
                mv = (X[key] - F[key]) * frac
                mvf = None
            else:
                mv = F[key] * frac
                mvf = mv
            if G is not None:
                if part == "all":
                    mvg = G[key] * frac
                elif part == "dm":
                    mvg = np.maximum(G[key] - F[key], 0.0) * frac
                else:
                    mvg = np.minimum(F[key], G[key]) * frac
                G[key] = G[key] - mvg
                G[kT] = G[kT] + mvg
            X[key] = X[key] - mv
            X[kT] = X[kT] + mv
            if F is not None and mvf is not None:
                F[key] = F[key] - mvf
                F[kT] = F[kT] + mvf

    if mode == "dynamic" and not first:
        move([(sx, st, "E") for sx, st in prio], 1.0)
        return Et, K, P1
    A = max(K - Tm, 0.0)
    if alloc == "uniform" and not select:
        PE = sum(X[(sx, st, "E")].sum() for sx in SEXES for st in ST)
        fr = min(A / PE, 1.0) if PE > 0 else 0.0
        move([(sx, st, "E") for sx in SEXES for st in ST], fr)
        return Et, K, min(A, PE)
    m1 = min(A, P1)
    if P1 > 0 and m1 > 0:
        move([(sx, st, "E") for sx, st in prio], m1 / P1)
    if select:
        return Et, K, m1
    A2 = A - m1
    keysN = [(sx, "N", "E") for sx in SEXES]
    done = m1
    if tier2 and A2 > 0:
        PD = sum((X[k_] - F[k_]).sum() for k_ in keysN)
        if PD > 0:
            m2 = min(A2, PD)
            move(keysN, m2 / PD, "dm")
            A2 -= m2
            done += m2
        PF = sum(F[k_].sum() for k_ in keysN)
        if PF > 0 and A2 > 0:
            m3 = min(A2, PF)
            move(keysN, m3 / PF, "free")
            done += m3
    else:
        PN = sum(X[k_].sum() for k_ in keysN)
        if PN > 0 and A2 > 0:
            m3 = min(A2, PN)
            move(keysN, m3 / PN)
            done += m3
    return Et, K, done
