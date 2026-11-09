import numpy as np
from .settings import AGES, AGE5, G0, GIDX, SEXES, ST, KS


def rr_t2d(iso, age, mode):
    """Relative risks of T2D for overweight and obesity versus normal weight (Yu et al, Int J Obes 2022, by age;
    alternatives by region or Abdullah et al)."""
    if mode == "abdullah": return 2.99, 7.19
    if mode == "region": return (1.60, 2.97) if iso in ("CHN", "JPN", "IND", "BGD") else (2.51, 5.70)
    return (2.28, 5.37) if age < 45 else (1.96, 3.70) if age < 65 else (1.40, 1.93)


def t2d_overlay(iso, base, init, ent, eg, ow, t2_inc, t2_prv, bmit, s, tier3_bmi30):
    """T2D incidence and the initial share without diabetes in every compartment.
    Inputs: base and init (initial counts, modified in place when tier 3 needs BMI >=30), ent (entrants, modified in
    place), eg (eligible shares), ow (national share above the BMI threshold), GBD T2D incidence and prevalence, bmit
    (BMI distribution by country, sex, age), s (settings), tier3_bmi30.
    Output: t2d dict with lam (incidence among people without diabetes), e_ow and e_no (relative incidence above and
    below the eligibility threshold; e_ow_N, e_no_N for the BMI-30 partition of tier 3) and free0 (share without
    diabetes by compartment)."""
    t2d = dict(lam={}, e_ow={}, e_no={}, free0={}, e_ow_N={}, e_no_N={})
    T3 = tier3_bmi30                                                    # tier 3 (no CVD, no diabetes) needs BMI >=30
    PRD = s.dm_pr; sc30 = {}
    for sx in SEXES:
        lam, eow, eno = np.zeros(len(AGES)), np.zeros(len(AGES)), np.zeros(len(AGES))
        pdm = np.zeros(len(AGES)); rr_ow_ratio = np.zeros(len(AGES))
        eowN, enoN, s30 = np.zeros(len(AGES)), np.zeros(len(AGES)), np.ones(len(AGES))
        for i, ag_ in enumerate(AGES):
            g5 = AGE5[GIDX[i]]; key = (iso, sx, g5); a5 = G0[GIDX[i]]
            pv = float(t2_prv.get(key, 0.0)); ic = float(t2_inc.get(key, 0.0))
            b = bmit.loc[(iso, sx, a5)] if (iso, sx, a5) in bmit.index else None
            p25 = float(b.p25) if b is not None else ow; p30 = float(b.p30) if b is not None else ow / 2
            pth = (float(b.p27) if b is not None else ow) if s.bmi_min == "27" else p25   # eligibility threshold
            r1, r2 = rr_t2d(iso, a5, s.t2d_rr); den = (1 - p25) + (p25 - p30) * r1 + p30 * r2
            S_ow = ((pth - p30) * r1 + p30 * r2) / den                  # share of new T2D in eligible BMI range
            lam[i] = ic / max(1 - pv, 1e-9); eow[i] = S_ow / max(pth, 1e-9); eno[i] = (1 - S_ow) / max(1 - pth, 1e-9)
            pdm[i] = pv; rr_ow_ratio[i] = S_ow / max(pth, 1e-9)
            S30 = p30 * r2 / den                                        # share of new T2D in BMI >=30
            eowN[i] = S30 / max(p30, 1e-9); enoN[i] = (1 - S30) / max(1 - p30, 1e-9)
            s30[i] = min(p30 / max(pth, 1e-9), 1.0)
        t2d["lam"][sx], t2d["e_ow"][sx], t2d["e_no"][sx] = lam, eow, eno
        if T3: t2d["e_ow_N"][sx], t2d["e_no_N"][sx] = eowN, enoN
        sc30[sx] = s30
        tot = sum(base[sx][st] for st in ST); tot = np.where(tot > 0, tot, 1.0)
        p_cvd = pdm * PRD / (pdm * PRD + 1 - pdm)
        sh_cvd = (base[sx]["I"] + base[sx]["S"] + base[sx]["B"]) / tot; sh_N = base[sx]["N"] / tot
        p_N = np.clip((pdm - p_cvd * sh_cvd) / np.where(sh_N > 0, sh_N, 1.0), 0, 1)
        for st in ST:
            p_st = p_N if st == "N" else p_cvd
            q_ow = np.clip(p_st * rr_ow_ratio, 0, 0.999)                 # diabetes among people with overweight
            q_no = np.clip((p_st - eg[sx][st] * q_ow) / np.maximum(1 - eg[sx][st], 1e-9), 0, 0.999)
            t2d["free0"][(sx, st, "T")] = 1 - q_ow; t2d["free0"][(sx, st, "E")] = 1 - q_ow
            t2d["free0"][(sx, st, "O")] = 1 - q_no
    if T3:                                                              # BMI 27-30 without CVD or diabetes: not eligible
        for sx in SEXES:
            f_ = t2d["free0"][(sx, "N", "E")].copy(); fO = t2d["free0"][(sx, "N", "O")].copy(); s3 = sc30[sx]
            E0, O0 = init[sx][("N", "E")].copy(), init[sx][("N", "O")].copy()
            kept = E0 * ((1 - f_) + f_ * s3); moved = E0 * f_ * (1 - s3); On = O0 + moved
            init[sx][("N", "E")], init[sx][("N", "O")] = kept, On
            fe = np.divide(E0 * f_ * s3, kept, out=f_.copy(), where=kept > 0)
            t2d["free0"][(sx, "N", "E")] = fe; t2d["free0"][(sx, "N", "T")] = fe
            t2d["free0"][(sx, "N", "O")] = np.divide(O0 * fO + moved, On, out=fO.copy(), where=On > 0)
            if ent is not None:
                sh = ent[sx]["share"]; e0 = sh[("N", "E")]
                sh[("N", "E")] = e0 * ((1 - f_[0]) + f_[0] * s3[0]); sh[("N", "O")] = sh[("N", "O")] + e0 * f_[0] * (1 - s3[0])
    return t2d


def dm_params(init, t2d, s):
    """Relative first-event incidence with diabetes (hr_I, hr_S; 1 when SENS_DM_CVD=0), the tier-2 priority switch,
    and the baseline split a_I, a_S (kept for reference; the yearly split is set in the baseline arm)."""
    hI = s.dm_hr_ihd if s.dm_cvd else 1.0
    hS = s.dm_hr_is if s.dm_cvd else 1.0
    dmp = dict(tier=s.dm_tier, hr_I=hI, hr_S=hS, a_I={}, a_S={})
    for sx in SEXES:                                            # diabetes share among people at risk at baseline
        def share(states):
            tot = sum(init[sx][(st, k_)] for st in states for k_ in KS)
            dmn = sum(init[sx][(st, k_)] * (1 - t2d["free0"][(sx, st, k_)]) for st in states for k_ in KS)
            return np.divide(dmn, tot, out=np.zeros(len(AGES)), where=tot > 0)
        dmp["a_I"][sx] = 1 / (1 + (hI - 1) * share(("N", "S"))); dmp["a_S"][sx] = 1 / (1 + (hS - 1) * share(("N", "I")))
    return dmp


def incidence_split(X, F, dm):
    """Multipliers of first-onset hazards for people without diabetes, chosen so that, with diabetes multiplying
    incidence by hr_I (IHD) and hr_S (IS), the population average equals the calibrated rate given the current share
    with diabetes. Computed in the baseline arm each year and applied in every arm. Output: {sex: (a_I, a_S)}."""
    aY = {}
    for sx in SEXES:
        def dshare(states):
            tot = sum(X[(sx, st, k_)] for st in states for k_ in KS); dmn = sum(X[(sx, st, k_)] - F[(sx, st, k_)] for st in states for k_ in KS)
            return np.divide(dmn, tot, out=np.zeros(len(AGES)), where=tot > 0)
        aY[sx] = (1 / (1 + (dm["hr_I"] - 1) * dshare(("N", "S"))), 1 / (1 + (dm["hr_S"] - 1) * dshare(("N", "I"))))
    return aY
