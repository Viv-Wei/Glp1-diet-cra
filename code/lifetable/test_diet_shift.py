# -*- coding: utf-8 -*-
"""Tests of the direction constraint in the dietary arm's shift towards the TMREL (diet_policy_arm.shifted_exposure).
Run from the project root: python -m pytest code/lifetable/test_diet_shift.py -q"""
import os, importlib.util, numpy as np, pandas as pd

def _load():
    p = "code/preprocessing/diet_policy_arm.py" if os.path.exists("code/preprocessing/diet_policy_arm.py") else "src/diet_policy_arm.py"
    spec = importlib.util.spec_from_file_location("dpa", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

dpa = _load()
CURVES = next(f for f in ("data/params/dose_response_conv_fixed.csv", "v2_conv_fixed/params/dose_response_conv_fixed.csv") if os.path.exists(f))
EXPO = next(d for d in ("data/input", "data") if os.path.exists(f"{d}/exposure_gbd_male.csv"))
curves = pd.read_csv(CURVES)
expo = pd.concat(pd.read_csv(f"{EXPO}/exposure_gbd_{s}.csv") for s in ("male", "female")).reset_index(drop=True)
tm = curves.groupby("gdd_var").tmrel.first()
protective = curves.groupby("gdd_var").curve_id.first().str.startswith("low_")

def _old(e, scenario):
    """The shift before the direction constraint, for comparison."""
    e = e.copy()
    for gv, (mode, x) in dpa.SCENARIOS[scenario].items():
        m = e.gdd_var == gv
        if mode == "pct": e.loc[m, "intake_median"] *= (1 + x)
        else:
            t = 3.0 if gv == "v37" else tm.get(gv, np.nan)
            if pd.notna(t): e.loc[m, "intake_median"] += x * (t - e.loc[m, "intake_median"])
    return e

def test_feasible_package_unchanged():
    for sc in ("feasible", "sodium30"):
        assert np.allclose(dpa.shifted_exposure(expo, curves, sc).intake_median, _old(expo, sc).intake_median, rtol=0, atol=0)

def test_no_shift_away_from_favourable_side():
    for sc in ["aspirational"] + [f"single_{g}" for g in dpa.SINGLE_FACTORS]:
        new = dpa.shifted_exposure(expo, curves, sc)
        d = new.intake_median.values - expo.intake_median.values
        prot = expo.gdd_var.map(protective).fillna(False).astype(bool).values
        assert (d[prot] >= -1e-12).all(), (sc, "a protective food was lowered")
        assert (d[~prot] <= 1e-12).all(), (sc, "a harmful factor was raised")

def test_strata_on_the_unfavourable_side_move_as_before():
    new, old = dpa.shifted_exposure(expo, curves, "aspirational"), _old(expo, "aspirational")
    t = expo.gdd_var.map(lambda g: 3.0 if g == "v37" else tm.get(g, np.nan))
    prot = expo.gdd_var.map(protective).fillna(False).astype(bool)
    below = (prot & (expo.intake_median <= t)) | (~prot & (expo.intake_median >= t))
    assert np.allclose(new.intake_median[below], old.intake_median[below], rtol=0, atol=1e-12)
    changed = ~np.isclose(new.intake_median, old.intake_median)
    assert changed.sum() == (~below & t.notna()).sum()          # only the strata already past the TMREL differ

if __name__ == "__main__":
    for f in (test_feasible_package_unchanged, test_no_shift_away_from_favourable_side, test_strata_on_the_unfavourable_side_move_as_before):
        f(); print("passed:", f.__name__)
