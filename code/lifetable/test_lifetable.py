import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lifetable.settings import AGES, SEXES, ST, KS
from lifetable.transitions import hazards, step, step_vec, age_one_year, ramp
from lifetable.calibration import or_joint, calibrate, ckd_calibrate, ckd_events
from lifetable import persistence as ps
from lifetable.allocation import allocate
from lifetable.eligibility import eligible_by_age

HZ = dict(
    lam_I=0.004,
    lam_S=0.002,
    mu_I=0.03,
    mu_S=0.05,
    mu_0=0.01,
    q_i=0.2,
    q_s=0.15,
    rr_cond=1.0,
)


def test_step_conserves_people():
    x = dict(N=0.9, I=0.05, S=0.03, B=0.02)
    new, ev, py = step(x, hazards(**HZ))
    dead = ev["fat_I"] + ev["fat_S"] + ev["post_I"] + ev["post_S"] + ev["bg"]
    assert sum(new.values()) + dead == pytest.approx(1.0, abs=1e-14)
    assert 0 < py < 1


def test_step_vec_matches_step():
    x = dict(N=0.9, I=0.05, S=0.03, B=0.02)
    new, ev, _ = step(x, hazards(**HZ))
    newv, evv = step_vec({k: np.array([v, v]) for k, v in x.items()}, hazards(**HZ))
    for st in ST:
        assert newv[st][0] == pytest.approx(new[st], rel=1e-12)
    for k in ("fat_I", "fat_S", "post_I", "post_S", "bg"):
        assert evv[k][1] == pytest.approx(ev[k], rel=1e-12)


def test_multipliers_reduce_events():
    x = {st: np.full(3, 0.25) for st in ST}
    _, ev0 = step_vec(x, hazards(**HZ))
    _, ev1 = step_vec(x, hazards(**HZ, m=dict(I_fatal=0.8, I_nonfatal=0.8, post=0.9)))
    assert (ev1["fat_I"] < ev0["fat_I"]).all() and (ev1["post_I"] < ev0["post_I"]).all()


def test_ageing_keeps_open_age_group():
    v = np.arange(1.0, len(AGES) + 1)
    a = age_one_year(v)
    assert a[0] == 0 and a[-1] == v[-1] + v[-2] and a.sum() == v.sum()


def test_ramp_reaches_half_and_most():
    assert ramp(8 / 1.6, 8) == pytest.approx(0.5)
    assert 0.8 < ramp(8, 8) < 0.85


def test_or_joint():
    assert or_joint(0.1, 0.2, 1.0) == pytest.approx(0.02)
    pb = or_joint(0.1, 0.2, 3.0)
    odds = (pb * (1 - 0.1 - 0.2 + pb)) / ((0.1 - pb) * (0.2 - pb))
    assert odds == pytest.approx(3.0, rel=1e-9)


def test_calibrate_reproduces_targets():
    shares = [0.9, 0.05, 0.03, 0.02]
    tgt = dict(inc_I=0.005, inc_S=0.003, d_I=0.002, d_S=0.0015, d_all=0.012)
    hz, ev = calibrate(shares, tgt, 0.3, 0.15, 1.0)
    assert ev["inc_I"] / ev["py"] == pytest.approx(tgt["inc_I"], rel=1e-6)
    assert (ev["fat_I"] + ev["post_I"]) / ev["py"] == pytest.approx(
        tgt["d_I"], rel=1e-6
    )
    assert (ev["fat_I"] + ev["post_I"] + ev["fat_S"] + ev["post_S"] + ev["bg"]) / ev[
        "py"
    ] == pytest.approx(tgt["d_all"], rel=1e-6)


def test_ckd_calibrate():
    tgt = dict(inc=0.004, d=0.0005, d_all=0.01)
    r = ckd_calibrate(0.1, tgt)
    ev = ckd_events(0.1, r["lam"], r["mu"], r["mb"])
    assert not r["fallback"] and ev["inc"] / ev["py"] == pytest.approx(
        tgt["inc"], rel=1e-6
    )


ONOFF = dict(
    s1_nd=0.648,
    s2_nd=0.557,
    r1_nd=0.363,
    r2_nd=0.159,
    s1_d=0.465,
    s2_d=0.329,
    r1_d=0.473,
    r2_d=0.190,
    res_cv=1 / 3,
    res_t2d=0.25,
    itt_cv=1.0,
    itt_t2d=1.0,
    late=1.0,
)


def test_reconstruction_matches_published():
    for r in ps.reconstruction_check(ONOFF):
        assert abs(r["diff"]) < 0.002


def test_states_stay_a_distribution():
    q = np.array([1.0, 0, 0, 0, 0])
    for _ in range(30):
        o = ps.retained(q, ONOFF, "nd")["o"]
        q = ps.next_states(q, o)
        assert q.sum() == pytest.approx(1.0) and (q >= 0).all()


def test_retained_effect_bounds():
    r = ps.retained(np.array([1.0, 0, 0, 0, 0]), ONOFF, "nd")
    assert 0 < r["on"] < 1 and r["on"] <= r["cv"] <= 1


def test_carry_and_mix_keep_a_distribution():
    qd, qnd = np.array([0.2, 0.5, 0.1, 0.1, 0.1]), np.array([0.1, 0.2, 0.2, 0.2, 0.3])
    c = ps.carry_converters(
        qd, qnd, enrolled_dm=100.0, converters=10.0, enrolled_dm_start=95.0
    )
    assert c.sum() == pytest.approx(1.0) and (c >= 0).all()
    m = ps.mix_new_enrolment(qd, before=90.0, after=100.0)
    assert m.sum() == pytest.approx(1.0) and m[0] == pytest.approx(
        (0.2 * 90 + 10) / 100
    )


def population():
    n = len(AGES)
    X = {
        (sx, st, k): np.full(
            n,
            {"N": 100.0, "I": 10.0, "S": 5.0, "B": 2.0}[st]
            * (1.0 if k != "T" else 0.0),
        )
        for sx in SEXES
        for st in ST
        for k in KS
    }
    F = {key: v * 0.8 for key, v in X.items()}
    return X, F, {key: v.copy() for key, v in F.items()}


def test_allocation_fills_target_and_priority():
    X, F, G = population()
    prio = [(sx, st) for sx in SEXES for st in ("I", "S", "B")]
    tot = sum(v.sum() for v in X.values())
    Et, K, done = allocate(
        X, F, G, 0.0, True, 0.10, "maintain", "risk", False, True, prio
    )
    T = sum(X[(sx, st, "T")].sum() for sx in SEXES for st in ST)
    assert T == pytest.approx(K) and done == pytest.approx(K)
    assert sum(X[(sx, st, "T")].sum() for sx, st in prio) == pytest.approx(K)
    assert all(X[(sx, "N", "T")].sum() == 0 for sx in SEXES)
    assert sum(v.sum() for v in X.values()) == pytest.approx(tot)


def test_allocation_reaches_tier_3_when_tiers_1_2_are_full():
    X, F, G = population()
    prio = [(sx, st) for sx in SEXES for st in ("I", "S", "B")]
    Et, K, done = allocate(
        X, F, G, 0.0, True, 0.50, "maintain", "risk", False, True, prio
    )
    assert all(X[(sx, st, "E")].sum() == pytest.approx(0) for sx, st in prio)
    assert all(
        (X[(sx, "N", "E")] - F[(sx, "N", "E")]).sum() == pytest.approx(0)
        for sx in SEXES
    )
    assert sum(X[(sx, st, "T")].sum() for sx in SEXES for st in ST) == pytest.approx(K)


def test_uniform_allocation_equal_proportions():
    X, F, G = population()
    prio = [(sx, st) for sx in SEXES for st in ("I", "S", "B")]
    E0 = {key: X[key].sum() for key in X if key[2] == "E"}
    allocate(X, F, G, 0.0, True, 0.10, "maintain", "uniform", False, True, prio)
    shares = {key: X[(key[0], key[1], "T")].sum() / E0[key] for key in E0}
    assert max(shares.values()) == pytest.approx(min(shares.values()))


def test_eligible_by_age_keeps_state_totals():
    import pandas as pd

    base = {
        sx: {st: np.linspace(1, 3, len(AGES)) * (10 if st == "N" else 1) for st in ST}
        for sx in SEXES
    }
    g = dict(N=0.3, I=0.4, S=0.35, B=0.45)
    selb = pd.DataFrame(
        [
            dict(iso3="AAA", sex=sx, a0=a0, p27=0.2 + 0.004 * (a0 - 25))
            for sx in SEXES
            for a0 in list(range(25, 95, 5)) + [95]
        ]
    )
    eg = eligible_by_age("AAA", g, base, selb, "p27", {})
    for st in ST:
        tot = sum(base[sx][st].sum() for sx in SEXES)
        assert sum((base[sx][st] * eg[sx][st]).sum() for sx in SEXES) == pytest.approx(
            g[st] * tot, rel=1e-10
        )
        assert all((eg[sx][st] <= 1).all() for sx in SEXES)
