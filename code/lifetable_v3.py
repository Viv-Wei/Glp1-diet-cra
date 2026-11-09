# -*- coding: utf-8 -*-
"""Multi-state life table comparing population dietary policy with GLP-1RA treatment programmes over 30 years
(19 countries and the GBD 2023 global aggregate). Code in code/lifetable/; this file is the command-line entry point.

Population and states
  Adults aged 25 years or older in 2023 (GBD 2023), by country, sex and single year of age (95 = 95+), followed as a
  closed cohort in annual cycles (SENS_OPEN_POP=1: people turning 25 enter each year, UN WPP 2024). Disease states
  N (neither IHD nor IS), I (IHD only), S (IS only), B (both), from GBD prevalence assuming independence at baseline
  (SENS_OR_COMORB: odds ratio of having both). Each state is split into enrolled in the treatment programme (T),
  untreated eligible (E) and ineligible (O), and by diabetes status.
Transitions
  Fatal and non-fatal first IHD and IS events, death after onset, death from other causes; annual competing-risk
  conversion P_i = (1 - exp(-H)) h_i / H; non-fatal new cases face the death hazards of their new state for half a year.
  Hazards are calibrated so that, without intervention, year-1 incidence and deaths per person-year reproduce GBD 2023,
  and are held at their 2023 values. Fatal first events: IHD SENS_IHD_FIRST (0.40) x IHD deaths; IS chosen so that the
  28-day case fatality of first IS is SENS_IS_CF28 (0.15) in the global population.
Diabetes
  First IHD and IS events 2.00 and 2.27 times as frequent with diabetes (SENS_DM_HR_IHD, SENS_DM_HR_IS; SENS_DM_CVD=0
  removes the difference); the split is set in the baseline arm each year so that its average equals the GBD rate, and
  applied in every arm. Cardiovascular risk, treatment effects, persistence and priority follow diabetes status under the
  no-intervention natural history; new T2D cases under each arm are counted separately (<out>_t2d.csv).
Dietary policy
  First IHD and IS events x (1 - PIF x phase-in x SENS_POLICY_MAINTAIN), everyone, from the dietary policy file
  (LT_DP_OVERRIDE, scenario --diet-scenario); phase-in over --ramp-diet years (8).
Treatment programme (SENS_TREAT)
  maintain (main): coverage is the share of eligible adults enrolled (BMI >=27 with IHD, IS or diabetes; BMI >=30
  otherwise; SENS_BMI_MIN, SENS_TIER3_BMI), kept for 30 years by filling places freed by death; places go first to
  people with IHD or IS, then with diabetes, then others (SENS_ALLOC=uniform: equal proportions). Eligible shares follow
  the sex and age pattern of the fitted BMI distribution within each disease state. --coverage: a share, "afford" (the
  country's maximum affordable coverage, data/params/coverage_afford_cap.csv) or "cstar". baseline: no new enrolment after
  year 1; dynamic: every eligible adult with IHD or IS enrolled. SENS_SELECT_LIKE=1: adults resembling SELECT only.
  Enrolled people stop and restart treatment in five states ON1, ON2, OFF1 (residual effect SENS_RESID_CV, SENS_RESID_T2D),
  OFF2, OFF3, separately without and with diabetes (Rodriguez et al 2025; SENS_STOP_MULT, SENS_RESTART_MULT,
  SENS_RESTART_LATE); SENS_ONOFF=0 uses a constant retention SENS_RETAIN instead. Trial intention-to-treat HRs are applied
  on drug (SENS_ITT_CV, SENS_ITT_T2D: on-treatment equivalents); HRs by diabetes status (SENS_HR_*), effect on fatal first
  events SENS_FATAL_MODE (same | cvd | none), on deaths after onset (SENS_POST_NONE=1: none), in people without CVD
  (SENS_PRIMARY_SCALE, SENS_TIER3_SCALE, SENS_TIER2_MI, SENS_TIER2_STROKE); SENS_CAP_REF caps the absolute benefit after
  onset at the reference country's risk. Treatment phase-in over --ramp-glp1 years (1).
CKD (SENS_CKD=1, <out>_ckd.csv)
  Separate two-state model; dietary PIF on incidence; treatment lowers SBP on drug (SENS_SBP_SHIFT, SENS_SBP_SHIFT_DM)
  on the GBD SBP-CKD curve at the diet-lowered SBP.
Outputs
  <out>.csv (IHD and IS deaths averted by country, arm and year), <out>_byage.csv, <out>_t2d.csv, <out>_ckd.csv, and
  <out>_check_*.csv (calibration, full-model calibration, feasibility, conservation, allocation, eligibility by age,
  persistence reconstruction, programme trace).

Usage (project root): python code/lifetable_v3.py --years 30 --coverage 0.20 --diet-scenario feasible --out data/results/run.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lifetable.settings import E, AGE5, G0, AGES, GIDX, SEXES, ST, ARMS, KS, read_settings     # noqa: F401 (re-exported)
from lifetable.inputs import ISO, cm, gbd, _gbd_read, load_pop, a0_, _load_diet_module, sodium_drop, dietary_pif
from lifetable.transitions import DEST, ramp, hazards, step, step_vec, age_one_year
from lifetable.calibration import (CKD_PREV_CAP, or_joint, calibrate, q_frac, expected_events, ckd_calibrate, ckd_events,
                                   baseline_country)
from lifetable.eligibility import overweight_by_state, eligible_by_age, coverage_target, initial_states, entrants
from lifetable.effects import hr_sets, make_multipliers, add_t2d_multiplier
from lifetable.diabetes import rr_t2d, t2d_overlay, dm_params
from lifetable import bmi as bmimod
from lifetable.persistence import onoff_params, reconstruction_check
from lifetable.projection import run_country
from lifetable.ckd import run_ckd, ckd_country
from lifetable.outputs import full_model_check, allocation_update, country_results, write_all


def main(a):
    s = read_settings(a.coverage); s.years = a.years
    HRS = hr_sets(s)
    pop = load_pop()
    rates = dict(inc_I=gbd("queryB_ihd.csv", "Incidence"), prv_I=gbd("queryB_ihd.csv", "Prevalence"), dth_I=gbd("queryB_ihd.csv", "Deaths"),
                 inc_S=gbd("queryB_istroke.csv", "Incidence"), prv_S=gbd("queryB_istroke.csv", "Prevalence"),
                 dth_S=gbd("queryB_istroke.csv", "Deaths"), allc=gbd("queryD_All_causes_deaths.csv", "Deaths"))
    # s_S: share of IS deaths that are fatal first events, chosen so that the incidence-weighted 28-day case fatality of
    # first ischaemic stroke in the GBD global population equals SENS_IS_CF28 (Feigin et al 2009); shared by all countries
    gi_ = [(float(rates["inc_S"].get(("GLB", sx, ag), 0)) * float(pop.get(("GLB", sx, ag), 0)),
            float(rates["dth_S"].get(("GLB", sx, ag), 0)) * float(pop.get(("GLB", sx, ag), 0))) for sx in SEXES for ag in AGE5]
    s_S = s.is_cf28 * sum(i for i, _ in gi_) / sum(d for _, d in gi_)
    suf = "_bmi27" if s.bmi_min == "27" else ""
    OW = dict(pd.read_csv(s.ow_file or f"data/params/overweight_prevalence{suf}.csv")[["iso3", "overweight_prev"]].values)
    owc = pd.read_csv(s.ow_cvd_file or f"data/params/overweight_cvd{suf}.csv").set_index("iso3")
    ELIG = dict(pd.read_csv(s.elig_file).values) if s.elig_file else {}
    afford = dict(pd.read_csv("data/params/coverage_afford_cap.csv").values) if a.coverage == "afford" else {}
    dp = dietary_pif(s.dp_file, a.diet_scenario)
    # Dietary effect on deaths after onset (SENS_DIET_POST): the sodium component of the dietary package being run.
    # post_na is the sodium part of the feasible package; other packages use their own sodium change: the aspirational
    # package and the sodium single factor move sodium 20% toward its TMREL (single_v37), the 30% sodium scenario is
    # sodium only, other single factors contain no sodium, and the tailored package contains sodium (15% lower, as in
    # the feasible package) only in the countries where sodium is among its three factors.
    dpp = None
    if s.diet_post:
        if s.diet_post != "post_na" or a.diet_scenario == "feasible":
            dpp = dietary_pif(s.dp_file, s.diet_post)
        elif a.diet_scenario in ("aspirational", "single_v37"):
            dpp = dietary_pif(s.dp_file, "single_v37")
        elif a.diet_scenario == "sodium30":
            dpp = dietary_pif(s.dp_file, "sodium30")
        elif a.diet_scenario == "tailored":
            tf = next(f for f in ("params/tailored_top3.csv", "data/params/tailored_top3.csv") if os.path.exists(f))
            tt = pd.read_csv(tf).sort_values(["iso3", "rank"]).groupby("iso3").head(3)
            with_na = set(tt[tt.gdd_var == "v37"].iso3)
            dpp = dietary_pif(s.dp_file, "post_na"); dpp = dpp[dpp.index.get_level_values("iso3").isin(with_na)]
        elif a.diet_scenario.startswith("single_"):
            dpp = None                                                 # single factors other than sodium
        else:
            raise SystemExit(f"[error] no rule for the dietary effect after onset with scenario '{a.diet_scenario}'")
    # SENS_CAP_REF caps the treatment part of the effect on deaths after onset; the dietary part (SENS_DIET_POST) is kept
    conv = lambda g0: s.k_dur * g0 / (1 - g0 + s.k_dur * g0)         # incident -> prevalent cases, duration ratio K
    seltab = pd.read_csv(s.select_file) if s.select_like else None
    selb = pd.read_csv("data/params/select_elig.csv") if os.path.exists("data/params/select_elig.csv") else None
    wp_all = None
    if s.open_pop:
        wp_all = pd.read_csv("data/wpp/WPP2024_PopulationByAge5GroupSex_Medium.csv", low_memory=False,
                             usecols=["Location", "ISO3_code", "Time", "AgeGrp", "PopMale", "PopFemale"])
        wp_all = wp_all[(wp_all.AgeGrp == "25-29") & wp_all.Time.between(2023, 2023 + a.years)]
    ck = None
    if s.ckd:
        from importlib.machinery import SourceFileLoader
        pv2 = SourceFileLoader("pv2", "code/preprocessing/paf_deficiency_v2.py").load_module()
        cv = pd.read_csv(s.ckd_curves)
        ck = dict(curve=pv2.Curve(cv[cv.curve_id == "high_sbp_ckd"]), inc=gbd("queryB_ckd_3years.csv", "Incidence"),
                  prv=gbd("queryB_ckd_3years.csv", "Prevalence"), dth=gbd("queryB_ckd_3years.csv", "Deaths"),
                  sbp={sx: pd.read_csv(f"data/input/sbp_gbd2023_{sx.lower()}.csv") for sx in SEXES},
                  expo={sx: pd.read_csv(f"data/input/exposure_gbd_{sx.lower()}.csv") for sx in SEXES},
                  dmod=_load_diet_module() if a.diet_scenario not in ("feasible",) else None)
        for sx in SEXES:
            ck["sbp"][sx]["a0"] = a0_(ck["sbp"][sx].age)
            e_ = ck["expo"][sx]; e_ = e_[(e_.gdd_var == "v37") & (e_.year == 2023)].copy(); e_["a0"] = a0_(e_.age); ck["expo"][sx] = e_
    t2d_on = not s.select_like and os.path.exists("data/params/select_elig.csv")   # diabetes dimension
    if t2d_on:
        t2_inc, t2_prv = gbd("queryB_t2d.csv", "Incidence"), gbd("queryB_t2d.csv", "Prevalence")
        bmit = pd.read_csv("data/params/select_elig.csv").set_index(["iso3", "sex", "a0"])
        tp = pd.read_csv(s.t2d_params); hr_t2d = float(tp[tp.param == "t2d_incidence_hr"].value.iloc[0])
    oo = onoff_params(s)
    TRF = {}                                                       # trend scenario inputs (code/preprocessing/build_trends.py)
    if s.trend:
        assert s.elig_struct == "bmi", "trends need the BMI structure (SENS_ELIG_STRUCT=bmi)"
        if "cvd" in s.trend: TRF["cvd"] = pd.read_csv("data/params/trends/cvd_factor.csv").set_index(["iso3", "sex", "a0", "t"])
        if "bmi" in s.trend: TRF["bmi"] = pd.read_csv("data/params/trends/bmi_target.csv").set_index(["iso3", "sex", "a0", "t"])
        if "dm" in s.trend: TRF["dm"] = pd.read_csv("data/params/trends/dm_factor.csv").set_index(["iso3", "t"]).r
        if "oth" in s.trend:
            assert os.path.exists("data/params/trends/oth_factor.csv"), "other-cause trend: run code/preprocessing/build_trends.py with the all-cause forecasts"
            TRF["oth"] = pd.read_csv("data/params/trends/oth_factor.csv").set_index(["iso3", "sex", "a0", "t"])

    rows, t2rows, ckrows, ckd_flags, agerows = [], [], [], [], []
    checks = dict(feasibility=[], calibration=[], conservation=[], marginal=[], allocation=[])
    if oo is not None: checks["onoff_reconstruction"] = reconstruction_check(oo)
    isos = sorted(set(ISO.values()))
    if s.cap_ref: isos = [s.cap_ref] + [i for i in isos if i != s.cap_ref]
    R_ref = None
    for iso in isos:
        sc = ELIG.get(iso, 1.0)
        ow = OW.get(iso, np.nan) * sc
        if not np.isfinite(ow): continue
        gI, gS = conv(owc.loc[iso, "ow_cvd_ihd"]) * sc, conv(owc.loc[iso, "ow_cvd_is"]) * sc
        base, R_tabs, cnt = baseline_country(iso, pop, rates, s.ihd_first, s_S, s.or_comorb, s.rr_cond, checks)
        g = overweight_by_state(iso, cnt, ow, gI, gS, checks)
        eg = eligible_by_age(iso, g, base, selb, s.elig_col, checks)
        C = coverage_target(iso, g, cnt, ow, a.coverage, s.alloc, afford, checks)
        mult = make_multipliers(s, HRS, float(dp.get(("IHD", iso), 0.0)), float(dp.get(("IschStroke", iso), 0.0)), a.ramp_diet, a.ramp_glp1,
                                *((float(dpp.get(("IHD", iso), 0.0)), float(dpp.get(("IschStroke", iso), 0.0))) if dpp is not None else ()))
        bmi_targets = None
        if s.elig_struct == "bmi":                                  # BMI classes, eligibility evaluated each year
            assert t2d_on and not s.select_like, "the BMI structure needs the diabetes dimension and excludes SELECT-like runs"
            init, t2d, bmi_targets = bmimod.initial_states_bmi(iso, base, eg, ow, t2_inc, t2_prv, bmit, s, rr_t2d,
                                                               legacy_dm=not s.dm_byclass)
            if s.cap_ref and iso == s.cap_ref: R_ref = R_tabs
            ent = bmimod.entrants_bmi(entrants(iso, wp_all, pop, base, eg, a.years), bmi_targets) if s.open_pop else None
            mult = add_t2d_multiplier(mult, s, float(dp.get(("T2D", iso), 0.0)), hr_t2d, a.ramp_diet, a.ramp_glp1)
        else:
            init, C_sel = initial_states(iso, base, eg, a.coverage, s.select_like, seltab, s.bmi_min, s.dm_pr, checks)
            if C_sel is not None: C = C_sel
            if s.cap_ref and iso == s.cap_ref: R_ref = R_tabs
            ent = entrants(iso, wp_all, pop, base, eg, a.years) if (s.open_pop and not s.select_like) else None
            t2d = None
            if t2d_on:
                t2d = t2d_overlay(iso, base, init, ent, eg, ow, t2_inc, t2_prv, bmit, s, s.tier3_bmi30 and not ELIG)
                mult = add_t2d_multiplier(mult, s, float(dp.get(("T2D", iso), 0.0)), hr_t2d, a.ramp_diet, a.ramp_glp1)
        dmp = dm_params(init, t2d, s) if t2d is not None else None
        TRd = dict(thr=(s.thr_cvd, s.thr_ob)) if s.elig_struct == "bmi" else None
        if s.trend:
            TRd.update(cvd_mode=s.trend_cvd_mode)
            if "cvd" in TRF:                                       # factors by sex and single-year age, each year
                band = np.where(AGES < 50, 25, np.where(AGES < 75, 50, 75)); cf = TRF["cvd"].loc[iso].reset_index()
                TRd["cvd"] = {}
                for sx in SEXES:                                    # arrays (year x single-year age), built once
                    pv = {c: cf[cf.sex == sx].pivot(index="t", columns="a0", values=c) for c in ("f_ihd", "f_is")}
                    idx = [pv["f_ihd"].columns.get_loc(int(b_)) for b_ in band]
                    TRd["cvd"][sx] = [(pv["f_ihd"].values[t - 1, idx], pv["f_is"].values[t - 1, idx]) for t in range(1, a.years + 1)]
            if "oth" in TRF:                                       # other-cause mortality, by sex and single-year age
                band = np.where(AGES < 50, 25, np.where(AGES < 75, 50, 75)); of = TRF["oth"].loc[iso].reset_index(); TRd["oth"] = {}
                for sx in SEXES:
                    pv = of[of.sex == sx].pivot(index="t", columns="a0", values="f_oth"); idx = [pv.columns.get_loc(int(b_)) for b_ in band]
                    TRd["oth"][sx] = [pv.values[t - 1, idx] for t in range(1, a.years + 1)]
            if "bmi" in TRF:
                TRd["bmi"] = bmimod.year_targets(iso, eg, bmit, TRF["bmi"].loc[iso], ow, legacy_dm=not s.dm_byclass, years=a.years)
            if "dm" in TRF:                                        # reference: the baseline arm without the diabetes trend
                ref = run_country(R_tabs, {sx: {k: v.copy() for k, v in init[sx].items()} for sx in SEXES}, mult, a.years, C, s.treat,
                                  alloc=s.alloc, t2d=t2d, entrants=ent, onoff=oo, dm=dmp, track_t2d=False,
                                  bmi_targets=bmi_targets, trend={k: v for k, v in TRd.items() if k != "dm"}, arms=["baseline"])
                TRd["dm"] = dict(ref=ref["baseline"]["dm_share"], r=np.array([TRF["dm"].loc[(iso, t)] for t in range(1, a.years + 1)]))
        res = run_country(R_tabs, init, mult, a.years, C, s.treat, alloc=s.alloc, R_ref=R_ref if s.cap_ref else None,
                          select=s.select_like, t2d=t2d, entrants=ent, onoff=oo, dm=dmp, track_t2d=s.t2d_out,
                          bmi_targets=bmi_targets, trend=TRd)
        if TRd is not None and "dm" in TRd:                       # calibration of the diabetes trend
            got = res["baseline"]["dm_share"]; ref, rr_ = TRd["dm"]["ref"], TRd["dm"]["r"]
            err = max(float(np.nanmax(np.abs(np.divide(got[t][sx], ref[t][sx] * rr_[t], out=np.ones_like(got[t][sx]),
                                                        where=ref[t][sx] * rr_[t] > 1e-6) - 1))) for t in range(a.years) for sx in SEXES)
            checks.setdefault("dm_trend", []).append(dict(iso3=iso, max_rel_error=err, r_final=float(rr_[-1])))
        if not s.select_like: full_model_check(iso, R_tabs, init, res, checks)
        allocation_update(res, checks)
        if ck is not None:
            ckrows.extend(ckd_country(iso, res, ck, pop, rates["allc"], dp, s, a.diet_scenario, a.ramp_diet, a.ramp_glp1, C, ckd_flags))
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
