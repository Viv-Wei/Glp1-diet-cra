# -*- coding: utf-8 -*-
"""30-year projection of one country under the four arms (baseline, treatment, dietary policy, both).

Each year: incidence split by diabetes status (set in the baseline arm), effect retained by enrolled people (five-state
persistence), new T2D cases, the annual competing-risk step for people without and with diabetes, ageing, entrants
(open population), stopping and restarting, and the top-up of programme places.
"""
import numpy as np
import os
from .settings import AGES, AGE5, GIDX, SEXES, ST, ARMS, KS, ELIG_STRUCT
from . import bmi as bmimod
from .transitions import hazards, step_vec, age_one_year
from .allocation import allocate
from .diabetes import incidence_split
from . import persistence as ps


def run_country(R_tabs, init, arms_mult, years, C, mode, alloc="risk", R_ref=None, select=False, t2d=None,
                entrants=None, onoff=None, dm=None, track_t2d=True, bmi_targets=None, trend=None, arms=None):
    """Inputs: R_tabs[sex], calibrated hazards per five-year group; init[sex][(state, k)], initial counts by single-year
    age (nobody enrolled); arms_mult(arm, t, k, Rt, dmf), the hazard multipliers; years; C, coverage of eligible adults;
    mode, maintain (target-coverage maintenance with risk-prioritised initiation) | baseline (no initiation after year 1)
    | dynamic (every eligible person with IHD or IS enrolled, uncapped); alloc, risk | uniform; R_ref, hazards of the
    reference country for the absolute-benefit cap; select, strict SELECT-like pool; t2d, the T2D overlay (None: no
    diabetes dimension); entrants, people turning 25; onoff, persistence probabilities (None: constant retention);
    dm, diabetes parameters; track_t2d, count new T2D cases by arm.
    Output: by arm, yearly IHD and IS deaths, conservation error, programme trace, new T2D cases, enrolled share by
    sex and age, deaths by age group, share on drug, diabetes share among the enrolled, year-1 deaths and the year-1
    allocation."""
    RK = ["lam_I", "lam_S", "mu_I", "mu_S", "mu_0", "q_i", "q_s", "rr_cond"]
    RA = {sx: {k_: np.array([R_tabs[sx][GIDX[i]][k_] for i in range(len(AGES))]) for k_ in RK} for sx in SEXES}
    FREEZE = os.environ.get("SENS_TEST_FREEZE", "0") == "1"                # tests only: no new IHD, IS or T2D
    if FREEZE:
        for sx in SEXES: RA[sx]["lam_I"] = RA[sx]["lam_I"] * 0.0; RA[sx]["lam_S"] = RA[sx]["lam_S"] * 0.0
    BMI = ELIG_STRUCT == "bmi"
    RULE = os.environ.get("SENS_ELIG_RULE", "new"); REBAL = os.environ.get("SENS_BMI_REBAL", "1") == "1"
    T2D_SURV = os.environ.get("SENS_T2D_SURV", "baseline")              # survival in the diabetes tracker: baseline | arm
    THR = (trend or {}).get("thr", (27.0, 30.0))                         # BMI thresholds (settings.thr_cvd, thr_ob)
    TK = [k for k in KS if k.startswith("T")]                             # enrolled compartments
    # compartments sharing the same hazards are stepped together: the BMI classes of the enrolled and of the not enrolled
    GROUPS = [list(bmimod.KT), list(bmimod.KU)] if BMI else [[k] for k in KS]
    sumT = lambda D, keys: sum(D[(sx, st, k)].sum() for sx, st in keys for k in TK)
    ALL = [(sx, st) for sx in SEXES for st in ST]
    def alloc_(X, F, G, Tm, first):
        if BMI: return bmimod.allocate_bmi(X, F, G, Tm, first, C, mode, alloc, tier2, RULE, THR[0], THR[1])
        return allocate(X, F, G, Tm, first, C, mode, alloc, select, tier2, prio)
    RF = ({sx: {k_: np.array([R_ref[sx][GIDX[i]][k_] for i in range(len(AGES))]) for k_ in ("mu_I", "mu_S")} for sx in SEXES}
          if R_ref is not None else None)
    tier2 = bool(dm and dm.get("tier")) and t2d is not None and not select
    prio = [(sx, st) for sx in SEXES for st in ("I", "S", "B")]
    res = {}
    assert ARMS[0] == "baseline", "the baseline arm must run first (it sets the diabetes split of incidence)"
    aY_ref = {}                                                    # diabetes split by year, from the baseline arm
    TR = trend or {}                                                     # trend scenario: cvd, bmi, dm (see lifetable_v3)
    mY = {}                                                              # T2D incidence multipliers, set in the baseline arm
    def en_of(sx, st, k):
        if BMI: return t2d["e"][sx][k[1]]
        nN = st == "N" and sx in t2d.get("e_ow_N", {})
        return t2d["e_ow_N" if nN else "e_ow"][sx] if k in ("T", "E") else t2d["e_no_N" if nN else "e_no"][sx]
    for arm in (arms or ARMS):
        X = {(sx, st, k): init[sx][(st, k)].copy() for sx in SEXES for st in ST for k in KS}
        F = ({key: X[key] * t2d["free0"][key] for key in X} if t2d else None)   # without diabetes, counterfactual (risk, tiers)
        G = ({key: v.copy() for key, v in F.items()} if t2d else None)           # without diabetes in this arm (T2D outcome)
        tot0 = sum(v.sum() for v in X.values()); cum_dead = 0.0
        dI, dS, trace, cT2D, tau, fbp = [], [], [], [], [], []
        cT2Dp = []                                                     # T2D cases prevented in the people at risk, by year
        PQ = {"nd": np.array([1.0, 0.0, 0.0, 0.0, 0.0]), "d": np.array([1.0, 0.0, 0.0, 0.0, 0.0])}   # ON1 ON2 OFF1 OFF2 OFF3
        wdm = []                                                       # diabetes share among the enrolled, by year
        dm_share = []                                                  # diabetes share by sex and age after the T2D step
        by_age = np.zeros(len(AGE5))                                   # cumulative IHD + IS deaths by age group
        y1 = {sx: {c_: np.zeros(len(AGES)) for c_ in ("I", "S", "all")} for sx in SEXES}   # year-1 deaths (calibration check)
        if arm in ("glp1", "combo"):                                   # enrolment at the start of year 1
            Et0, K0, n0 = alloc_(X, F, G, 0.0, True)
            T0 = sumT(X, ALL)
            T0c = sumT(X, prio)
            T0d = (sumT(X, [(sx, "N") for sx in SEXES]) - sumT(F, [(sx, "N") for sx in SEXES])) if F is not None else np.nan
            init_alloc = dict(eligible=Et0, target=K0, treated=T0, share_prior_cvd=T0c / T0 if T0 else np.nan,
                              share_dm_no_cvd=T0d / T0 if T0 else np.nan)
        for t in range(1, years + 1):
            tot_sx = {sx: sum(X[(sx, st, k_)] for st in ST for k_ in KS) for sx in SEXES}
            tau.append({sx: np.divide(sum(X[(sx, st, k_)] for st in ST for k_ in TK), tot_sx[sx], out=np.zeros(len(AGES)),
                                      where=tot_sx[sx] > 0) for sx in SEXES})
            new = {key: np.zeros_like(v) for key, v in X.items()}; yI = yS = 0.0
            newF = {key: np.zeros_like(v) for key, v in X.items()} if t2d else None; yT = 0.0; yP = 0.0
            newG = {key: np.zeros_like(v) for key, v in X.items()} if t2d else None
            convT = 0.0                                                       # enrolled people developing diabetes this year
            d0T = (sumT(X, ALL) - sumT(F, ALL)) if F is not None else 0.0
            aY = {}
            if dm and t2d and arm != "baseline":                              # same split as the baseline arm
                aY = aY_ref[t]
            elif dm and t2d:                                                  # baseline arm: average kept at the GBD rate
                aY = incidence_split(X, F, dm); aY_ref[t] = aY
            Rt = None; RtD = None
            if onoff is not None and arm in ("glp1", "combo"):              # effect retained by the enrolled this year
                Rt, RtD = ps.retained(PQ["nd"], onoff, "nd"), ps.retained(PQ["d"], onoff, "d")
                XT = sumT(X, ALL)
                wd = (1 - sumT(F, ALL) / XT) if (F is not None and XT > 0) else 0.0
                Rt = dict(Rt, on_all=(1 - wd) * Rt["on"] + wd * RtD["on"], on_nd=Rt["on"], on_d=RtD["on"], wd=wd)
            fbp.append(Rt["on_all"] if Rt else None); wdm.append((Rt["wd"], Rt["on_nd"], Rt["on_d"]) if Rt else None)
            dmrow = {}
            for sx in SEXES:
                mdm = 1.0
                if t2d and "dm" in TR:                                        # T2D incidence calibrated to the forecast
                    if arm == "baseline":
                        Dn = sum(X[(sx, st, k_)] - F[(sx, st, k_)] for st in ST for k_ in KS); Nn = sum(X[(sx, st, k_)] for st in ST for k_ in KS)
                        Fsum = sum(F[(sx, st, k_)] for st in ST for k_ in KS)
                        need = np.clip(TR["dm"]["ref"][t - 1][sx] * TR["dm"]["r"][t - 1] * Nn - Dn, 0.0, 0.999 * Fsum)
                        Fs = np.stack([F[(sx, st, k_)] for st in ST for k_ in KS])                     # compartments x ages
                        LE = np.stack([t2d["lam"][sx] * en_of(sx, st, k_) for st in ST for k_ in KS])
                        # Newton's method on the multiplier, by age: new cases are increasing and concave in it, so
                        # starting from 0 the iterates rise monotonically to the root (no overshoot); capped at 50
                        m_ = np.zeros(len(AGES))
                        for _ in range(50):
                            e_ = np.exp(-LE * m_); f_ = (Fs * (1 - e_)).sum(axis=0) - need; fp = (Fs * LE * e_).sum(axis=0)
                            nxt = np.clip(m_ - np.divide(f_, fp, out=np.zeros_like(f_), where=fp > 0), 0.0, 50.0)
                            if np.all(np.abs(nxt - m_) <= 1e-13 * (1 + m_)): m_ = nxt; break
                            m_ = nxt
                        mY.setdefault(t, {})[sx] = m_
                    mdm = mY[t][sx]
                for grp in GROUPS:
                    if grp[0] in TK and not any(X[(sx, st, k)].any() for st in ST for k in grp): continue   # nobody enrolled
                    kk = "T" if grp[0] in TK else "U"
                    mult = arms_mult(arm, t, kk, Rt)
                    if t2d:                                                   # new T2D: counterfactual (F) and in this arm (G)
                      for k in grp:
                        for st in ST:
                            if BMI:                                               # relative incidence by BMI class
                                en = t2d["e"][sx][k[1]]
                            else:
                                nN = st == "N" and sx in t2d.get("e_ow_N", {})    # tier 3 partition at BMI 30
                                en = (t2d["e_ow_N" if nN else "e_ow"][sx] if k in ("T", "E") else t2d["e_no_N" if nN else "e_no"][sx])
                            lam0 = t2d["lam"][sx] * en * mdm * (0.0 if FREEZE else 1.0)
                            p0 = 1 - np.exp(-lam0); p1 = 1 - np.exp(-lam0 * mult.get("T2D", 1.0))
                            if k in TK: convT += float((F[(sx, st, k)] * p0).sum())
                            F[(sx, st, k)] = F[(sx, st, k)] * (1 - p0)
                            yP += float((G[(sx, st, k)] * (p0 - p1)).sum())      # same people, without vs with the effect
                            cases = G[(sx, st, k)] * p1; yT += float(cases.sum()); G[(sx, st, k)] = G[(sx, st, k)] - cases
                    r = RA[sx]
                    if "cvd" in TR:                                           # projected change in IHD and IS rates
                        fI, fS = TR["cvd"][sx][t - 1]; r = dict(r, mu_I=r["mu_I"] * fI, mu_S=r["mu_S"] * fS)
                        if TR.get("cvd_mode", "both") == "both": r.update(lam_I=r["lam_I"] * fI, lam_S=r["lam_S"] * fS)
                    if "oth" in TR:                                           # projected change in other-cause mortality
                        r = dict(r, mu_0=r["mu_0"] * TR["oth"][sx][t - 1])
                    def capped(m_):
                        if RF is not None and kk == "T" and "post" in m_:              # absolute benefit capped at the
                            red = 1 - m_["post"]                                        # reference country's risk level
                            # dietary part of the multiplier (combination arm, SENS_DIET_POST), kept uncapped
                            dI = m_["post_I"] / m_["post"] if ("post_I" in m_ and m_["post"] > 0) else 1.0
                            dS = m_["post_S"] / m_["post"] if ("post_S" in m_ and m_["post"] > 0) else 1.0
                            return dict(m_, post_I=dI * np.where(r["mu_I"] > 0, 1 - red * np.minimum(r["mu_I"], RF[sx]["mu_I"]) / np.where(r["mu_I"] > 0, r["mu_I"], 1), 1.0),
                                        post_S=dS * np.where(r["mu_S"] > 0, 1 - red * np.minimum(r["mu_S"], RF[sx]["mu_S"]) / np.where(r["mu_S"] > 0, r["mu_S"], 1), 1.0))
                        return m_
                    if t2d:                                                   # without (F) and with (X - F) diabetes
                        aI, aS = aY[sx] if dm else (1.0, 1.0)
                        hI, hS = (dm["hr_I"], dm["hr_S"]) if dm else (1.0, 1.0)
                        hz0 = hazards(r["lam_I"] * aI, r["lam_S"] * aS, r["mu_I"], r["mu_S"], r["mu_0"], r["q_i"], r["q_s"], r["rr_cond"], capped(mult))
                        mult1 = arms_mult(arm, t, kk, RtD if RtD is not None else Rt, True) if dm else arms_mult(arm, t, kk, RtD if RtD is not None else Rt)
                        hz1 = hazards(r["lam_I"] * aI * hI, r["lam_S"] * aS * hS, r["mu_I"], r["mu_S"], r["mu_0"], r["q_i"], r["q_s"], r["rr_cond"], capped(mult1))
                        xf = {st: np.stack([F[(sx, st, k)] for k in grp]) for st in ST}
                        xd = {st: np.stack([np.maximum(X[(sx, st, k)] - F[(sx, st, k)], 0.0) for k in grp]) for st in ST}
                        nf, ev0 = step_vec(xf, hz0); nd, ev1 = step_vec(xd, hz1)
                        for st in ST:
                            for i_, k in enumerate(grp): new[(sx, st, k)] += nf[st][i_] + nd[st][i_]; newF[(sx, st, k)] += nf[st][i_]
                        if track_t2d:
                            # diabetes tracker: survival as without any intervention (SENS_T2D_SURV=baseline, default), so
                            # that T2D cases averted reflect the effect on incidence and not longer survival; arm: the arm's
                            # own survival (cases averted then include the extra cases among people kept alive)
                            hzG = hz0 if T2D_SURV == "arm" else hazards(r["lam_I"] * aI, r["lam_S"] * aS, r["mu_I"], r["mu_S"], r["mu_0"],
                                                                         r["q_i"], r["q_s"], r["rr_cond"], None)
                            ng, _ = step_vec({st: np.stack([G[(sx, st, k)] for k in grp]) for st in ST}, hzG)
                            for st in ST:
                                for i_, k in enumerate(grp): newG[(sx, st, k)] += ng[st][i_]
                        ev = {k_: (ev0[k_] + ev1[k_]).sum(axis=0) for k_ in ev0}
                    else:
                        hz = hazards(r["lam_I"], r["lam_S"], r["mu_I"], r["mu_S"], r["mu_0"], r["q_i"], r["q_s"], r["rr_cond"], capped(mult))
                        nx, ev = step_vec({st: np.stack([X[(sx, st, k)] for k in grp]) for st in ST}, hz)
                        for st in ST:
                            for i_, k in enumerate(grp): new[(sx, st, k)] += nx[st][i_]
                        ev = {k_: v.sum(axis=0) for k_, v in ev.items()}
                    dcv = ev["fat_I"] + ev["post_I"] + ev["fat_S"] + ev["post_S"]
                    if t == 1:
                        y1[sx]["I"] += ev["fat_I"] + ev["post_I"]; y1[sx]["S"] += ev["fat_S"] + ev["post_S"]; y1[sx]["all"] += dcv + ev["bg"]
                    yI += float(np.sum(ev["fat_I"] + ev["post_I"])); yS += float(np.sum(ev["fat_S"] + ev["post_S"]))
                    by_age += np.bincount(GIDX, weights=dcv, minlength=len(AGE5))
                    cum_dead += float(np.sum(dcv + ev["bg"]))
                if t2d:                                                       # diabetes share after this year's T2D step
                    Nn = sum(X[(sx, st, k_)] for st in ST for k_ in KS)
                    dmrow[sx] = np.divide(sum(X[(sx, st, k_)] - F[(sx, st, k_)] for st in ST for k_ in KS), Nn, out=np.zeros(len(AGES)), where=Nn > 0)
            dm_share.append(dmrow)
            if t2d:
                for D_ in (newF, newG):
                    for key in D_: D_[key] = age_one_year(D_[key])
                cT2D.append(yT); cT2Dp.append(yP)
            for key in new: new[key] = age_one_year(new[key])              # ageing
            if entrants is not None and t < years:                    # open population: people turning 25 enter
                for sx in SEXES:
                    n_t = float(entrants[sx]["n"][t - 1])
                    for st in ST:
                        for k_ in KS:
                            add = n_t * entrants[sx]["share"][(st, k_)]
                            new[(sx, st, k_)][0] += add; tot0 += add
                            if t2d:
                                fr0 = float(np.atleast_1d(t2d["free0"][(sx, st, k_)])[0])
                                newF[(sx, st, k_)][0] += add * fr0; newG[(sx, st, k_)][0] += add * fr0
            X = new; dI.append(yI); dS.append(yS)
            if t2d: F = newF; G = newG
            if BMI and REBAL and bmi_targets is not None:                       # BMI class shares by age kept at targets
                bmimod.rebalance(X, F, G, TR["bmi"][t - 1] if "bmi" in TR else bmi_targets)
            # ---- on/off transitions among enrolled people over the year, then top-up (new courses start in ON1)
            if Rt is not None:
                for g_, R_ in (("nd", Rt), ("d", RtD)):
                    PQ[g_] = ps.next_states(PQ[g_], R_["o"])
                if F is not None and convT > 0:                           # converters bring the states of the group without diabetes
                    Dm_ = sumT(X, ALL) - sumT(F, ALL)
                    PQ["d"] = ps.carry_converters(PQ["d"], PQ["nd"], Dm_, convT, d0T)
            if arm in ("glp1", "combo"):
                Tm = sumT(X, ALL)
                Fm = sumT(F, ALL) if F is not None else Tm
                Et, K, enrolled = alloc_(X, F, G, Tm, False)
                Tn = sumT(X, ALL)
                if onoff is not None and Tn > 0:                     # mix in new enrolment at ON1, by diabetes status
                    Fn = sumT(F, ALL) if F is not None else Tn
                    for g_, before, after in (("nd", Fm, Fn), ("d", Tm - Fm, Tn - Fn)):
                        PQ[g_] = ps.mix_new_enrolment(PQ[g_], before, after)
                Tcvd = sumT(X, prio)
                alive_t = sum(v.sum() for v in X.values())
                trace.append(dict(arm=arm, year=t, alive=alive_t, eligible=Et, target=K, enrolled_before=Tm, new_enrolment=enrolled,
                                  enrolled_after=Tn, coverage_before=Tm / Et if Et else np.nan,
                                  coverage_after=Tn / Et if Et else np.nan, share_prior_cvd=Tcvd / Tn if Tn else np.nan,
                                  on_share=fbp[-1] if fbp[-1] is not None else np.nan))
        alive = sum(v.sum() for v in X.values())
        res[arm] = dict(dm_share=dm_share, IHD=dI, IschStroke=dS, conservation=(alive + cum_dead) / tot0 - 1, trace=trace, T2D=cT2D, T2Dp=cT2Dp, tau=tau,
                        by_age=by_age, fbp=fbp, wdm=wdm, y1=y1, init_alloc=init_alloc if arm in ("glp1", "combo") else None)
    return res
