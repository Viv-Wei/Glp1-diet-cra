# -*- coding: utf-8 -*-
"""Joint prevalence of IHD and IS, and calibration of baseline hazards to GBD 2023 rates per person-year."""
import math
import numpy as np
from .settings import ST
from .transitions import hazards, step

CKD_PREV_CAP = 0.98


def or_joint(pi, ps, odds):
    """Prevalence of having both conditions for margins pi, ps and odds ratio odds (closed-form 2x2 solution);
    odds = 1 gives pi x ps (independence, main analysis)."""
    if odds == 1.0 or pi <= 0 or ps <= 0: return pi * ps
    a = odds - 1.0; b = 1.0 + (odds - 1.0) * (pi + ps); c = odds * pi * ps
    pb = (b - math.sqrt(max(b * b - 4 * a * c, 0.0))) / (2 * a)
    return min(max(pb, max(0.0, pi + ps - 1.0)), min(pi, ps))


def calibrate(shares, tgt, q_i, q_s, rr_cond, iters=200, tol=1e-7):
    """Baseline hazards of one stratum (country, sex, five-year age group).
    Inputs: shares, the baseline share in N, I, S, B; tgt, GBD rates per person-year (inc_I, inc_S, d_I, d_S, d_all);
    q_i, q_s, fatal shares of first events; rr_cond.
    Output: (hazards lam_I, lam_S, mu_I, mu_S, mu_0; expected year-1 events) such that year-1 incidence, cause-specific
    and all-cause deaths per person-year reproduce the targets under competing risks (to within tol)."""
    n, i, s, b = shares
    lam_I, lam_S = tgt["inc_I"], tgt["inc_S"]
    mu_I = (1 - q_frac(tgt, "I", q_i)) * tgt["d_I"] / max(i + b, 1e-12)
    mu_S = (1 - q_frac(tgt, "S", q_s)) * tgt["d_S"] / max(s + b, 1e-12)
    mu_0 = max(tgt["d_all"] - tgt["d_I"] - tgt["d_S"], 1e-9)
    for _ in range(iters):
        ev = expected_events(shares, lam_I, lam_S, mu_I, mu_S, mu_0, q_i, q_s, rr_cond)
        py = ev["py"]                                         # targets are GBD rates per person-year
        post_I_t = max(tgt["d_I"] - q_i * tgt["inc_I"], 0.0) * py; post_S_t = max(tgt["d_S"] - q_s * tgt["inc_S"], 0.0) * py
        bg_t = max(tgt["d_all"] - tgt["d_I"] - tgt["d_S"], 0.0) * py
        u = [tgt["inc_I"] * py / ev["inc_I"] if ev["inc_I"] > 0 else 1.0, tgt["inc_S"] * py / ev["inc_S"] if ev["inc_S"] > 0 else 1.0,
             post_I_t / ev["post_I"] if ev["post_I"] > 0 and post_I_t > 0 else 1.0,
             post_S_t / ev["post_S"] if ev["post_S"] > 0 and post_S_t > 0 else 1.0,
             bg_t / ev["bg"] if ev["bg"] > 0 and bg_t > 0 else 1.0]
        lam_I *= u[0]; lam_S *= u[1]; mu_I *= u[2]; mu_S *= u[3]; mu_0 *= u[4]
        if max(abs(x - 1) for x in u) < tol: break
    ev = expected_events(shares, lam_I, lam_S, mu_I, mu_S, mu_0, q_i, q_s, rr_cond)
    return dict(lam_I=lam_I, lam_S=lam_S, mu_I=mu_I, mu_S=mu_S, mu_0=mu_0), ev


def q_frac(tgt, which, q):
    inc, d = (tgt["inc_I"], tgt["d_I"]) if which == "I" else (tgt["inc_S"], tgt["d_S"])
    return min(q * inc / d, 1.0) if d > 0 else 0.0


def expected_events(shares, lam_I, lam_S, mu_I, mu_S, mu_0, q_i, q_s, rr_cond):
    _, ev, py = step(dict(zip(ST, shares)), hazards(lam_I, lam_S, mu_I, mu_S, mu_0, q_i, q_s, rr_cond))
    ev.pop("flows", None); ev["py"] = py
    return ev


def ckd_calibrate(prev, tgt, iters=200, tol=1e-7):
    """CKD hazards (incidence among people without CKD, death with CKD, background) reproducing GBD rates per
    person-year in year 1 under competing risks. tgt: inc, d, d_all. Prevalence is capped at CKD_PREV_CAP (GBD
    estimates close to 100% in a few of the oldest strata); the incidence hazard is capped at 20."""
    prev = min(prev, CKD_PREV_CAP)
    lam = min(tgt["inc"] / max(1 - prev, 1e-9), 20.0); mu = tgt["d"] / max(prev, 1e-9); mb = max(tgt["d_all"] - tgt["d"], 1e-9)
    init = dict(lam=lam, mu=mu, mb=mb)
    for _ in range(iters):
        ev = ckd_events(prev, lam, mu, mb)
        u = [tgt["inc"] * ev["py"] / ev["inc"] if ev["inc"] > 0 else 1.0, tgt["d"] * ev["py"] / ev["d"] if ev["d"] > 0 else 1.0,
             max(tgt["d_all"] - tgt["d"], 0) * ev["py"] / ev["bg"] if ev["bg"] > 0 and tgt["d_all"] > tgt["d"] else 1.0]
        lam = min(lam * u[0], 20.0); mu *= u[1]; mb *= u[2]
        if not all(np.isfinite([lam, mu, mb])): return dict(init, fallback=True)
        if max(abs(x - 1) for x in u) < tol: break
    return dict(lam=lam, mu=mu, mb=mb, fallback=False)


def ckd_events(prev, lam, mu, mb, m=1.0):
    w, d = 1 - prev, prev
    Hw, Hd = lam * m + mb, mu + mb
    fw, fd = (1 - math.exp(-Hw)) / Hw, (1 - math.exp(-Hd)) / Hd
    return dict(inc=w * lam * m * fw, d=d * mu * fd, bg=w * mb * fw + d * mb * fd, py=w * fw + d * fd)


def baseline_country(iso, pop, rates, s_first, s_S, or_comorb, rr_cond, checks):
    """Baseline population and calibrated hazards of one country.
    Inputs: pop and rates (GBD 2023 inc_I, prv_I, dth_I, inc_S, prv_S, dth_S, allc, indexed by iso3, sex, age group);
    s_first, the share of IHD deaths that are fatal first events; s_S, the same for IS; or_comorb, rr_cond.
    Output: (base[sex][state], single-year age counts; R_tabs[sex], hazards by five-year group; cnt, totals by state).
    Appends the feasibility and calibration checks of every stratum to checks."""
    from .settings import AGES, AGE5, G0, SEXES
    base, R_tabs, cnt = {}, {}, dict(N=0.0, I=0.0, S=0.0, B=0.0, pop=0.0)
    for sex in SEXES:
        R_tabs[sex], comp = [], {st: np.zeros(len(AGES)) for st in ST}
        for gi, ag in enumerate(AGE5):
            k = (iso, sex, ag); n_g = float(pop.get(k, 0.0))
            pi, ps = float(rates["prv_I"].get(k, 0)), float(rates["prv_S"].get(k, 0))
            pb = or_joint(pi, ps, or_comorb)
            sh = [1 - pi - ps + pb, pi - pb, ps - pb, pb]
            tgt = dict(inc_I=float(rates["inc_I"].get(k, 0)), inc_S=float(rates["inc_S"].get(k, 0)), d_I=float(rates["dth_I"].get(k, 0)),
                       d_S=float(rates["dth_S"].get(k, 0)), d_all=float(rates["allc"].get(k, 0)))
            # fatal first events: IHD s x deaths (as a share of incidence), IS 28-day case fatality
            qi_raw = s_first * tgt["d_I"] / tgt["inc_I"] if tgt["inc_I"] > 0 else 0.0
            qs_raw = s_S * tgt["d_S"] / tgt["inc_S"] if tgt["inc_S"] > 0 else 0.0
            qi = min(qi_raw, 1.0); qs = min(qs_raw, 1.0)
            checks["feasibility"].append(dict(iso3=iso, sex=sex, age=ag, q_IHD_raw=qi_raw, q_IHD=qi,
                                              q_IS_raw=qs_raw, q_IS=qs, capped_IHD=qi_raw > 1.0,
                                              capped_IS=qs_raw > 1.0))
            hz, ev = calibrate(sh, tgt, qi, qs, rr_cond)
            hz.update(q_i=qi, q_s=qs, rr_cond=rr_cond); R_tabs[sex].append(hz)
            rel = lambda m_, t_: (m_ / ev["py"] / t_ - 1) if t_ > 0 else 0.0     # model rate per person-year vs GBD
            checks["calibration"].append(dict(iso3=iso, sex=sex, age=ag, err_inc_I=rel(ev["inc_I"], tgt["inc_I"]),
                err_inc_S=rel(ev["inc_S"], tgt["inc_S"]), err_d_I=rel(ev["fat_I"] + ev["post_I"], tgt["d_I"]),
                err_d_S=rel(ev["fat_S"] + ev["post_S"], tgt["d_S"]),
                err_d_all=rel(ev["fat_I"] + ev["post_I"] + ev["fat_S"] + ev["post_S"] + ev["bg"], tgt["d_all"]),
                fatal_share_IHD_deaths=ev["fat_I"] / ev["py"] / tgt["d_I"] if tgt["d_I"] > 0 else np.nan,
                implied_cf_first_IHD=qi))
            ages_in = (AGES >= G0[gi]) & ((AGES < G0[gi] + 5) if gi < len(AGE5) - 1 else True)
            for st, s_ in zip(ST, sh):
                comp[st][ages_in] = n_g * s_ / ages_in.sum()
                cnt[st] += n_g * s_
            cnt["pop"] += n_g
        base[sex] = comp
    return base, R_tabs, cnt
