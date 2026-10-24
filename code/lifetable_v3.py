import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifetable.settings import (
    E,
    AGE5,
    G0,
    AGES,
    GIDX,
    SEXES,
    ST,
    ARMS,
    KS,
    read_settings,
)
from lifetable.inputs import (
    ISO,
    cm,
    gbd,
    _gbd_read,
    load_pop,
    a0_,
    _load_diet_module,
    sodium_drop,
    dietary_pif,
)
from lifetable.transitions import DEST, ramp, hazards, step, step_vec, age_one_year
from lifetable.calibration import (
    CKD_PREV_CAP,
    or_joint,
    calibrate,
    q_frac,
    expected_events,
    ckd_calibrate,
    ckd_events,
    baseline_country,
)
from lifetable.eligibility import (
    overweight_by_state,
    eligible_by_age,
    coverage_target,
    initial_states,
    entrants,
)
from lifetable.effects import hr_sets, make_multipliers, add_t2d_multiplier
from lifetable.diabetes import rr_t2d, t2d_overlay, dm_params
from lifetable.persistence import onoff_params, reconstruction_check
from lifetable.projection import run_country
from lifetable.ckd import run_ckd, ckd_country
from lifetable.outputs import (
    full_model_check,
    allocation_update,
    country_results,
    write_all,
)


def main(a):
    s = read_settings(a.coverage)
    s.years = a.years
    HRS = hr_sets(s)
    pop = load_pop()
    rates = dict(
        inc_I=gbd("queryB_ihd.csv", "Incidence"),
        prv_I=gbd("queryB_ihd.csv", "Prevalence"),
        dth_I=gbd("queryB_ihd.csv", "Deaths"),
        inc_S=gbd("queryB_istroke.csv", "Incidence"),
        prv_S=gbd("queryB_istroke.csv", "Prevalence"),
        dth_S=gbd("queryB_istroke.csv", "Deaths"),
        allc=gbd("queryD_All_causes_deaths.csv", "Deaths"),
    )
    gi_ = [
        (
            float(rates["inc_S"].get(("GLB", sx, ag), 0))
            * float(pop.get(("GLB", sx, ag), 0)),
            float(rates["dth_S"].get(("GLB", sx, ag), 0))
            * float(pop.get(("GLB", sx, ag), 0)),
        )
        for sx in SEXES
        for ag in AGE5
    ]
    s_S = s.is_cf28 * sum(i for i, _ in gi_) / sum(d for _, d in gi_)
    suf = "_bmi27" if s.bmi_min == "27" else ""
    OW = dict(
        pd.read_csv(s.ow_file or f"data/params/overweight_prevalence{suf}.csv")[
            ["iso3", "overweight_prev"]
        ].values
    )
    owc = pd.read_csv(
        s.ow_cvd_file or f"data/params/overweight_cvd{suf}.csv"
    ).set_index("iso3")
    ELIG = dict(pd.read_csv(s.elig_file).values) if s.elig_file else {}
    afford = (
        dict(pd.read_csv("data/params/coverage_afford_cap.csv").values)
        if a.coverage == "afford"
        else {}
    )
    dp = dietary_pif(s.dp_file, a.diet_scenario)
    conv = lambda g0: s.k_dur * g0 / (1 - g0 + s.k_dur * g0)
    seltab = pd.read_csv(s.select_file) if s.select_like else None
    selb = (
        pd.read_csv("data/params/select_elig.csv")
        if os.path.exists("data/params/select_elig.csv")
        else None
    )
    wp_all = None
    if s.open_pop:
        wp_all = pd.read_csv(
            "data/wpp/WPP2024_PopulationByAge5GroupSex_Medium.csv",
            low_memory=False,
            usecols=["Location", "ISO3_code", "Time", "AgeGrp", "PopMale", "PopFemale"],
        )
        wp_all = wp_all[
            (wp_all.AgeGrp == "25-29") & wp_all.Time.between(2023, 2023 + a.years)
        ]
    ck = None
    if s.ckd:
        from importlib.machinery import SourceFileLoader

        pv2 = SourceFileLoader(
            "pv2", "code/preprocessing/paf_deficiency_v2.py"
        ).load_module()
        cv = pd.read_csv(s.ckd_curves)
        ck = dict(
            curve=pv2.Curve(cv[cv.curve_id == "high_sbp_ckd"]),
            inc=gbd("queryB_ckd_3years.csv", "Incidence"),
            prv=gbd("queryB_ckd_3years.csv", "Prevalence"),
            dth=gbd("queryB_ckd_3years.csv", "Deaths"),
            sbp={
                sx: pd.read_csv(f"data/input/sbp_gbd2023_{sx.lower()}.csv")
                for sx in SEXES
            },
            expo={
                sx: pd.read_csv(f"data/input/exposure_gbd_{sx.lower()}.csv")
                for sx in SEXES
            },
            dmod=_load_diet_module() if a.diet_scenario not in ("feasible",) else None,
        )
        for sx in SEXES:
            ck["sbp"][sx]["a0"] = a0_(ck["sbp"][sx].age)
            e_ = ck["expo"][sx]
            e_ = e_[(e_.gdd_var == "v37") & (e_.year == 2023)].copy()
            e_["a0"] = a0_(e_.age)
            ck["expo"][sx] = e_
    t2d_on = not s.select_like and os.path.exists("data/params/select_elig.csv")
    if t2d_on:
        t2_inc, t2_prv = gbd("queryB_t2d.csv", "Incidence"), gbd(
            "queryB_t2d.csv", "Prevalence"
        )
        bmit = pd.read_csv("data/params/select_elig.csv").set_index(
            ["iso3", "sex", "a0"]
        )
        tp = pd.read_csv(s.t2d_params)
        hr_t2d = float(tp[tp.param == "t2d_incidence_hr"].value.iloc[0])
    oo = onoff_params(s)

    rows, t2rows, ckrows, ckd_flags, agerows = [], [], [], [], []
    checks = dict(
        feasibility=[], calibration=[], conservation=[], marginal=[], allocation=[]
    )
    if oo is not None:
        checks["onoff_reconstruction"] = reconstruction_check(oo)
    isos = sorted(set(ISO.values()))
    if s.cap_ref:
        isos = [s.cap_ref] + [i for i in isos if i != s.cap_ref]
    R_ref = None
    for iso in isos:
        sc = ELIG.get(iso, 1.0)
        ow = OW.get(iso, np.nan) * sc
        if not np.isfinite(ow):
            continue
        gI, gS = (
            conv(owc.loc[iso, "ow_cvd_ihd"]) * sc,
            conv(owc.loc[iso, "ow_cvd_is"]) * sc,
        )
        base, R_tabs, cnt = baseline_country(
            iso, pop, rates, s.ihd_first, s_S, s.or_comorb, s.rr_cond, checks
        )
        g = overweight_by_state(iso, cnt, ow, gI, gS, checks)
        eg = eligible_by_age(iso, g, base, selb, s.elig_col, checks)
        C = coverage_target(iso, g, cnt, ow, a.coverage, s.alloc, afford, checks)
        mult = make_multipliers(
            s,
            HRS,
            float(dp.get(("IHD", iso), 0.0)),
            float(dp.get(("IschStroke", iso), 0.0)),
            a.ramp_diet,
            a.ramp_glp1,
        )
        init, C_sel = initial_states(
            iso, base, eg, a.coverage, s.select_like, seltab, s.bmi_min, s.dm_pr, checks
        )
        if C_sel is not None:
            C = C_sel
        if s.cap_ref and iso == s.cap_ref:
            R_ref = R_tabs
        ent = (
            entrants(iso, wp_all, pop, base, eg, a.years)
            if (s.open_pop and not s.select_like)
            else None
        )
        t2d = None
        if t2d_on:
            t2d = t2d_overlay(
                iso,
                base,
                init,
                ent,
                eg,
                ow,
                t2_inc,
                t2_prv,
                bmit,
                s,
                s.tier3_bmi30 and not ELIG,
            )
            mult = add_t2d_multiplier(
                mult,
                s,
                float(dp.get(("T2D", iso), 0.0)),
                hr_t2d,
                a.ramp_diet,
                a.ramp_glp1,
            )
        dmp = dm_params(init, t2d, s) if t2d is not None else None
        res = run_country(
            R_tabs,
            init,
            mult,
            a.years,
            C,
            s.treat,
            alloc=s.alloc,
            R_ref=R_ref if s.cap_ref else None,
            select=s.select_like,
            t2d=t2d,
            entrants=ent,
            onoff=oo,
            dm=dmp,
            track_t2d=s.t2d_out,
        )
        if not s.select_like:
            full_model_check(iso, R_tabs, init, res, checks)
        allocation_update(res, checks)
        if ck is not None:
            ckrows.extend(
                ckd_country(
                    iso,
                    res,
                    ck,
                    pop,
                    rates["allc"],
                    dp,
                    s,
                    a.diet_scenario,
                    a.ramp_diet,
                    a.ramp_glp1,
                    C,
                    ckd_flags,
                )
            )
        country_results(iso, res, a.years, C, t2d, checks, rows, t2rows, agerows)
    write_all(a.out, s, a, rows, agerows, ckrows, ckd_flags, t2rows, checks, s_S)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=30)
    p.add_argument("--ramp-glp1", type=int, default=1)
    p.add_argument("--ramp-diet", type=int, default=8)
    p.add_argument("--diet-scenario", default="feasible")
    p.add_argument("--coverage", default="0.20")
    p.add_argument("--out", default="data/results/lifetable_v3.csv")
    main(p.parse_args())
