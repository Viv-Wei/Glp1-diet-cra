import numpy as np
from .settings import AGES, AGE5, G0, GIDX, SEXES, ARMS
from .calibration import ckd_calibrate, CKD_PREV_CAP
from .transitions import ramp, age_one_year
from .inputs import sodium_drop

NA2SBP = 3.82 / 2.30


def run_ckd(Rc, init_w, init_d, mult_by_year, years):

    W = {sx: init_w[sx].copy() for sx in SEXES}
    D = {sx: init_d[sx].copy() for sx in SEXES}
    out = []
    for t in range(1, years + 1):
        yd = 0.0
        for sx in SEXES:
            lam = np.array([Rc[sx][GIDX[i]]["lam"] for i in range(len(AGES))])
            mu = np.array([Rc[sx][GIDX[i]]["mu"] for i in range(len(AGES))])
            mb = np.array([Rc[sx][GIDX[i]]["mb"] for i in range(len(AGES))])
            m = mult_by_year(t, sx)
            Hw, Hd = lam * m + mb, mu + mb
            pw_inc = (1 - np.exp(-Hw)) * lam * m / Hw
            pw_stay = np.exp(-Hw)
            pd_death = (1 - np.exp(-Hd)) * mu / Hd
            pd_stay = np.exp(-Hd)
            yd += float((D[sx] * pd_death).sum())
            W2, D2 = W[sx] * pw_stay, D[sx] * pd_stay + W[sx] * pw_inc
            for v in (W2, D2):
                v[:] = age_one_year(v)
            W[sx], D[sx] = W2, D2
        out.append(yd)
    return out


def ckd_country(
    iso, res, ck, pop, allc, dp, s, diet_scenario, ramp_diet, ramp_glp1, C, flags
):

    Rc, iw, idd = {}, {}, {}
    for sx in SEXES:
        Rc[sx] = []
        iw[sx] = np.zeros(len(AGES))
        idd[sx] = np.zeros(len(AGES))
        for gi, ag in enumerate(AGE5):
            k_ = (iso, sx, ag)
            pv = float(ck["prv"].get(k_, 0))
            n_g = float(pop.get(k_, 0))
            rc_ = ckd_calibrate(
                pv,
                dict(
                    inc=float(ck["inc"].get(k_, 0)),
                    d=float(ck["dth"].get(k_, 0)),
                    d_all=float(allc.get(k_, 0)),
                ),
            )
            Rc[sx].append(rc_)
            if pv > CKD_PREV_CAP or rc_.get("fallback"):
                flags.append(
                    dict(
                        iso3=iso,
                        sex=sx,
                        age=ag,
                        prevalence=pv,
                        capped=pv > CKD_PREV_CAP,
                        fallback=rc_.get("fallback"),
                    )
                )
            pv = min(pv, CKD_PREV_CAP)
            ages_in = GIDX == gi
            iw[sx][ages_in] = n_g * (1 - pv) / ages_in.sum()
            idd[sx][ages_in] = n_g * pv / ages_in.sum()
    dC = float(dp.get(("CKD", iso), 0.0))
    sh = {
        sx: sodium_drop(ck["expo"][sx], iso, diet_scenario, ck["dmod"]) for sx in SEXES
    }
    sbp0 = {
        sx: ck["sbp"][sx][ck["sbp"][sx].iso3 == iso].set_index("a0").sbp_mean
        for sx in SEXES
    }

    def ratio_arr(sx, d_diet, d_trt):

        out = np.ones(len(AGES))
        for i in range(len(AGES)):
            a5 = G0[GIDX[i]]
            s0 = float(sbp0[sx].get(a5, np.nan))
            if not np.isfinite(s0):
                continue
            dd = sh[sx].get(a5, 0.0) * NA2SBP * d_diet
            out[i] = float(
                ck["curve"].rr(s0 - dd - d_trt, strict=False)
                / ck["curve"].rr(s0 - dd, strict=False)
            )
        return out

    ckd_out = {}
    for arm in ARMS:
        tau_arm = res[arm]["tau"]

        def mby(t, sx, arm=arm, tau_arm=tau_arm):
            rd = (
                ramp(t, ramp_diet) * s.policy_maintain
                if arm in ("diet", "combo")
                else 0.0
            )
            m = np.full(len(AGES), 1 - dC * rd)
            if arm in ("glp1", "combo"):
                tr = tau_arm[t - 1][sx]
                rt = ramp(t, ramp_glp1)
                w_ = res[arm]["wdm"][t - 1]
                if w_ is not None:
                    rr_ = (1 - w_[0]) * (
                        (1 - w_[1]) + w_[1] * ratio_arr(sx, rd, s.sbp_shift * rt)
                    ) + w_[0] * (
                        (1 - w_[2]) + w_[2] * ratio_arr(sx, rd, s.sbp_shift_dm * rt)
                    )
                    m = m * ((1 - tr) + tr * rr_)
                else:
                    m = m * (
                        (1 - tr) + tr * ratio_arr(sx, rd, s.sbp_shift * s.f_bp * rt)
                    )
            return m

        ckd_out[arm] = run_ckd(Rc, iw, idd, mby, s.years)
    b0 = np.array(ckd_out["baseline"])
    rows = []
    for arm in ["glp1", "diet", "combo"]:
        av = b0 - np.array(ckd_out[arm])
        rows.append(
            dict(
                iso3=iso,
                arm=arm,
                deaths_averted_total=float(av.sum()),
                deaths_baseline_total=float(b0.sum()),
                coverage_used=C,
                **{f"yr{i}": float(v) for i, v in enumerate(av, 1)},
            )
        )
    return rows
