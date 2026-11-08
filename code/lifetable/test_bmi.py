# -*- coding: utf-8 -*-
"""Tests of the BMI-class structure (code/lifetable/bmi.py).

Run from the project root:  python code/lifetable/test_bmi.py           (unit tests, a few seconds)
                            python code/lifetable/test_bmi.py --model   (also the full-model equivalence, about 1 minute)
"""
import os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lifetable import bmi                                                   # noqa: E402
from lifetable.settings import SEXES, ST, AGES                              # noqa: E402

OK = []
def check(name, cond, detail=""):
    assert cond, f"FAILED: {name} {detail}"; OK.append(name); print(f"  ok  {name}")

KS8 = list(bmi.KT + bmi.KU)
def toy(seed=1):
    """Random compartments X with F <= X (people without diabetes) and G = F."""
    rng = np.random.default_rng(seed); n = len(AGES)
    X = {(sx, st, k): rng.uniform(10, 100, n) * (0.0 if k.startswith("T") else 1.0) for sx in SEXES for st in ST for k in KS8}
    F = {key: v * rng.uniform(0.6, 0.95, n) for key, v in X.items()}
    return X, F, {key: v.copy() for key, v in F.items()}

def total(D): return sum(v.sum() for v in D.values())

def test_pools():
    X, F, G = toy()
    for rule in ("new", "legacy"):
        t1, t2, t3 = bmi.pools(X, F, rule)
        k1 = {(k[1], k[2], p) for k, p in t1}; k2 = {(k[1], k[2], p) for k, p in t2}; k3 = {(k[1], k[2], p) for k, p in t3}
        check(f"pools ({rule}): tier 1 has everyone with IHD or IS at BMI >=27", {(st, b, "all") for st in ("I", "S", "B") for b in ("U2", "U3")} <= k1)
        check(f"pools ({rule}): tier 1 has diabetes at BMI <27 only under the new rule",
              ({(st, b, "dm") for st in ("I", "S", "B") for b in ("U0", "U1")} <= k1) == (rule == "new") and not any(b in ("U0", "U1") and p == "all" for _, b, p in k1))
        check(f"pools ({rule}): tier 2 has diabetes at BMI <27 only under the new rule", ({("N", "U0", "dm"), ("N", "U1", "dm")} <= k2) == (rule == "new"))
        check(f"pools ({rule}): tier 3 is BMI >=30 without diabetes", k3 == {("N", "U3", "free")})
        check(f"pools ({rule}): no enrolled compartment in any pool", all(k[2].startswith("U") for k, _ in t1 + t2 + t3))
    t1, t2, t3 = bmi.pools(X, F, "new", thr_cvd=25.0, thr_ob=27.0)
    check("pools (thresholds 25 and 27): IHD or IS from BMI 25, tier 3 from BMI 27",
          {b for (_, st, b), p in t1 if st == "I" and p == "all"} == {"U1", "U2", "U3"} and {k[2] for k, _ in t3} == {"U2", "U3"})
    t1, t2, t3 = bmi.pools(X, F, "new", thr_cvd=30.0)
    check("pools (threshold 30): IHD or IS without diabetes from BMI 30 only", {b for (_, st, b), p in t1 if st == "S" and p == "all"} == {"U3"})

def test_allocation():
    for alloc, tier2 in (("risk", True), ("risk", False), ("uniform", True)):
        X, F, G = toy(2); n0 = total(X); f0 = total(F)
        Et, K, done = bmi.allocate_bmi(X, F, G, 0.0, True, 0.2, "maintain", alloc, tier2, "new")
        T = sum(X[(sx, st, k)].sum() for sx in SEXES for st in ST for k in bmi.KT)
        check(f"allocation ({alloc}, tier2={tier2}): enrolled equals the target", abs(T / K - 1) < 1e-12, f"{T} vs {K}")
        check(f"allocation ({alloc}, tier2={tier2}): people and people without diabetes conserved",
              abs(total(X) / n0 - 1) < 1e-12 and abs(total(F) / f0 - 1) < 1e-12)
        check(f"allocation ({alloc}, tier2={tier2}): no negative compartments, F <= X",
              all((X[k] >= -1e-9).all() and (F[k] <= X[k] + 1e-9).all() for k in X))
    X, F, G = toy(3); t1, _, _ = bmi.pools(X, F, "new"); P1 = sum(bmi._size(X, F, k, p) for k, p in t1)
    Et, K, done = bmi.allocate_bmi(X, F, G, 0.0, True, 0.01, "maintain", "risk", True, "new")
    T_cvd = sum(X[(sx, st, k)].sum() for sx in SEXES for st in ("I", "S", "B") for k in bmi.KT)
    check("allocation (risk): a small target is filled from tier 1 only", K < P1 and abs(T_cvd / K - 1) < 1e-12)
    X, F, G = toy(5); bmi.allocate_bmi(X, F, G, 0.0, True, 0.6, "maintain", "risk", True, "new")
    T_ul_nodm = sum(F[(sx, st, b)].sum() for sx in SEXES for st in ST for b in ("T0", "T1"))
    check("allocation: nobody without diabetes enrolled at BMI <27", T_ul_nodm < 1e-9)

def test_rebalance():
    X, F, G = toy(4); n = len(AGES); rng = np.random.default_rng(7)
    for key in list(X):                                               # enrolled group with a different BMI mix (selective)
        if key[2].startswith("U"):
            w = {"0": 0.6, "1": 0.4, "2": 0.15, "3": 0.05}[key[2][1]]
            X[(key[0], key[1], "T" + key[2][1])] = X[key] * w; F[(key[0], key[1], "T" + key[2][1])] = F[key] * w
    G = {k: v.copy() for k, v in F.items()}
    tg = {(sx, st): {"0": np.full(n, 0.4), "1": np.full(n, 0.1), "2": np.full(n, 0.3), "3": np.full(n, 0.2)} for sx in SEXES for st in ST}
    n0, f0 = total(X), total(F)
    grp0 = {(sx, st, g): sum(X[(sx, st, g + b)] for b in bmi.BC) for sx in SEXES for st in ST for g in "TU"}
    bmi.rebalance(X, F, G, tg)
    check("rebalance: people and people without diabetes conserved", abs(total(X) / n0 - 1) < 1e-12 and abs(total(F) / f0 - 1) < 1e-12)
    check("rebalance: enrolled and not enrolled each keep their size at every age",
          all(np.allclose(sum(X[(sx, st, g + b)] for b in bmi.BC), grp0[(sx, st, g)]) for sx, st, g in grp0))
    both = lambda sx, st, b: X[(sx, st, "T" + b)] + X[(sx, st, "U" + b)]
    check("rebalance: combined class shares equal the targets at every age",
          all(np.allclose(both(sx, st, b) / (grp0[(sx, st, "T")] + grp0[(sx, st, "U")]), tg[(sx, st)][b]) for sx in SEXES for st in ST for b in bmi.BC))
    lowT = sum(X[(sx, st, "T0")].sum() + X[(sx, st, "T1")].sum() for sx in SEXES for st in ST) / sum(grp0[(sx, st, "T")].sum() for sx in SEXES for st in ST)
    lowP = sum(tg[(sx, st)]["0"].sum() + tg[(sx, st)]["1"].sum() for sx in SEXES for st in ST) / (len(SEXES) * len(ST) * n)
    check("rebalance: the enrolled keep a BMI mix different from the population (selection not undone)", lowT > lowP + 0.05,
          f"enrolled share below 27: {lowT:.2f}, population {lowP:.2f}")
    check("rebalance: F <= X and G <= X everywhere", all((F[k] <= X[k] + 1e-9).all() and (G[k] <= X[k] + 1e-9).all() for k in X))

def test_t2d_nonnegative():
    """T2D cases averted must not be negative in any country, arm or year (cumulatively)."""
    env = dict(os.environ, LT_DP_OVERRIDE=os.environ.get("LT_DP_OVERRIDE", "data/results/v3/diet_policy_arm_v2_tailored.csv"))
    f = "/tmp/test_bmi_t2d.csv"
    subprocess.run([sys.executable, "code/lifetable_v3.py", "--years", "30", "--coverage", "0.20", "--diet-scenario", "feasible", "--out", f],
                   env=env, check=True, capture_output=True)
    import pandas as pd
    t = pd.read_csv(f[:-4] + "_t2d.csv"); yr = [c for c in t.columns if c.startswith("yr")]
    low = float(t[yr].cumsum(axis=1).min().min())
    check("model: cumulative T2D cases averted never negative (any country, arm, year)", low >= -1e-6, f"lowest {low:.1f}")

def test_model_equivalence():
    """With no new IHD, IS or T2D, the old rule, diabetes shared within BMI >=27 and no rebalancing, nobody's eligibility
    can change, so the BMI structure must reproduce the original structure."""
    env = dict(os.environ, SENS_TEST_FREEZE="1", SENS_TREND="none", LT_DP_OVERRIDE=os.environ.get("LT_DP_OVERRIDE", "data/results/v3/diet_policy_arm_v2_tailored.csv"))
    out = {}
    for tag, extra in (("old", dict(SENS_ELIG_STRUCT="legacy")),
                       ("new", dict(SENS_ELIG_STRUCT="bmi", SENS_ELIG_RULE="legacy", SENS_BMI_REBAL="0", SENS_DM_BYCLASS="0"))):
        f = f"/tmp/test_bmi_{tag}.csv"
        subprocess.run([sys.executable, "code/lifetable_v3.py", "--years", "30", "--coverage", "0.20", "--diet-scenario", "feasible",
                        "--out", f], env=dict(env, **extra), check=True, capture_output=True)
        out[tag] = f
    import pandas as pd
    a, b = pd.read_csv(out["old"]), pd.read_csv(out["new"]); num = a.select_dtypes("number").columns
    rel = float(((a[num] - b[num]).abs() / a[num].abs().where(a[num].abs() > 1e-9)).max().max())
    check("model: frozen BMI structure reproduces the original structure", rel < 1e-8, f"largest relative difference {rel:.2e}")

if __name__ == "__main__":
    test_pools(); test_allocation(); test_rebalance()
    if "--model" in sys.argv: test_model_equivalence(); test_t2d_nonnegative()
    print(f"{len(OK)} checks passed")
