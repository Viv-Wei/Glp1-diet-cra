# -*- coding: utf-8 -*-
"""Tests of the dietary effect on deaths after onset (SENS_DIET_POST, lifetable.effects.make_multipliers).
Run from the project root: python -m pytest tests/test_diet_post.py -q"""
import os, sys, importlib
for p in ("src", "code"):
    if os.path.isdir(os.path.join(p, "lifetable")): sys.path.insert(0, p)
from lifetable import settings, effects
from lifetable.transitions import ramp

def _setup(env):
    for k in ("SENS_DIET_POST",): os.environ.pop(k, None)
    os.environ.update(env); importlib.reload(settings)
    s = settings.read_settings(0.2); return s, effects.hr_sets(s)

T, RD, RG = 30, 8, 1                     # a year after full phase-in
RT = {"cv": 0.8, "t2d": 0.7}             # effect retained this year

def test_default_unchanged():
    s, H = _setup({})
    m0 = effects.make_multipliers(s, H, 0.10, 0.08, RD, RG)
    m1 = effects.make_multipliers(s, H, 0.10, 0.08, RD, RG, 0.0, 0.0)
    for arm in ("diet", "glp1", "combo"):
        for k in ("T", "U"):
            a, b = m0(arm, T, k, RT), m1(arm, T, k, RT)
            assert a == b and "post_I" not in a and "post_S" not in a, (arm, k)

def test_diet_arm_acts_after_onset():
    s, H = _setup({"SENS_DIET_POST": "post_na"})
    m = effects.make_multipliers(s, H, 0.10, 0.08, RD, RG, 0.04, 0.05)("diet", T, "U", RT)
    assert abs(m["post_I"] - (1 - 0.04 * ramp(T, RD))) < 1e-12 and abs(m["post_S"] - (1 - 0.05 * ramp(T, RD))) < 1e-12
    assert "post_I" not in m["_N"] and "post_S" not in m["_N"]          # not applied to people without IHD or IS

def test_combination_multiplies():
    s, H = _setup({"SENS_DIET_POST": "post_na"})
    mm = effects.make_multipliers(s, H, 0.10, 0.08, RD, RG, 0.04, 0.05)
    c, g = mm("combo", T, "T", RT), mm("glp1", T, "T", RT)
    assert abs(c["post_I"] - (1 - 0.04 * ramp(T, RD)) * g["post"]) < 1e-12
    assert abs(c["post_S"] - (1 - 0.05 * ramp(T, RD)) * g["post"]) < 1e-12
    u = mm("combo", T, "U", RT)                                            # not enrolled: dietary effect only
    assert abs(u["post_I"] - (1 - 0.04 * ramp(T, RD))) < 1e-12

def test_treatment_arm_unchanged():
    s0, H0 = _setup({}); g0 = effects.make_multipliers(s0, H0, 0.10, 0.08, RD, RG)("glp1", T, "T", RT)
    s1, H1 = _setup({"SENS_DIET_POST": "post_na"}); g1 = effects.make_multipliers(s1, H1, 0.10, 0.08, RD, RG, 0.04, 0.05)("glp1", T, "T", RT)
    assert g0 == g1 and "post_I" not in g1

if __name__ == "__main__":
    for f in (test_default_unchanged, test_diet_arm_acts_after_onset, test_combination_multiplies, test_treatment_arm_unchanged):
        f(); print("passed:", f.__name__)
