import numpy as np
from .settings import AGES, AGE5, GIDX, SEXES, ST, ARMS, KS
from .transitions import hazards, step_vec, age_one_year
from .allocation import allocate
from .diabetes import incidence_split
from . import persistence as ps


def run_country(
    R_tabs,
    init,
    arms_mult,
    years,
    C,
    mode,
    alloc="risk",
    R_ref=None,
    select=False,
    t2d=None,
    entrants=None,
    onoff=None,
    dm=None,
    track_t2d=True,
):

    RK = ["lam_I", "lam_S", "mu_I", "mu_S", "mu_0", "q_i", "q_s", "rr_cond"]
    RA = {
        sx: {
            k_: np.array([R_tabs[sx][GIDX[i]][k_] for i in range(len(AGES))])
            for k_ in RK
        }
        for sx in SEXES
    }
    RF = (
        {
            sx: {
                k_: np.array([R_ref[sx][GIDX[i]][k_] for i in range(len(AGES))])
                for k_ in ("mu_I", "mu_S")
            }
            for sx in SEXES
        }
        if R_ref is not None
        else None
    )
    tier2 = bool(dm and dm.get("tier")) and t2d is not None and not select
    prio = [(sx, st) for sx in SEXES for st in ("I", "S", "B")]
    res = {}
    assert (
        ARMS[0] == "baseline"
    ), "the baseline arm must run first (it sets the diabetes split of incidence)"
    aY_ref = {}
    for arm in ARMS:
        X = {
            (sx, st, k): init[sx][(st, k)].copy()
            for sx in SEXES
            for st in ST
            for k in KS
        }
        F = {key: X[key] * t2d["free0"][key] for key in X} if t2d else None
        G = {key: v.copy() for key, v in F.items()} if t2d else None
        tot0 = sum(v.sum() for v in X.values())
        cum_dead = 0.0
        dI, dS, trace, cT2D, tau, fbp = [], [], [], [], [], []
        PQ = {
            "nd": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            "d": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        }
        wdm = []
        by_age = np.zeros(len(AGE5))
        y1 = {sx: {c_: np.zeros(len(AGES)) for c_ in ("I", "S", "all")} for sx in SEXES}
        if arm in ("glp1", "combo"):
            Et0, K0, n0 = allocate(
                X, F, G, 0.0, True, C, mode, alloc, select, tier2, prio
            )
            T0 = sum(X[(sx, st, "T")].sum() for sx in SEXES for st in ST)
            T0c = sum(X[(sx, st, "T")].sum() for sx, st in prio)
            T0d = (
                sum((X[(sx, "N", "T")] - F[(sx, "N", "T")]).sum() for sx in SEXES)
                if F is not None
                else np.nan
            )
            init_alloc = dict(
                eligible=Et0,
                target=K0,
                treated=T0,
                share_prior_cvd=T0c / T0 if T0 else np.nan,
                share_dm_no_cvd=T0d / T0 if T0 else np.nan,
            )
        for t in range(1, years + 1):
            tot_sx = {
                sx: sum(X[(sx, st, k_)] for st in ST for k_ in KS) for sx in SEXES
            }
            tau.append(
                {
                    sx: np.divide(
                        sum(X[(sx, st, "T")] for st in ST),
                        tot_sx[sx],
                        out=np.zeros(len(AGES)),
                        where=tot_sx[sx] > 0,
                    )
                    for sx in SEXES
                }
            )
            new = {key: np.zeros_like(v) for key, v in X.items()}
            yI = yS = 0.0
            newF = {key: np.zeros_like(v) for key, v in X.items()} if t2d else None
            yT = 0.0
            newG = {key: np.zeros_like(v) for key, v in X.items()} if t2d else None
            convT = 0.0
            d0T = (
                sum(
                    (X[(sx, st, "T")] - F[(sx, st, "T")]).sum()
                    for sx in SEXES
                    for st in ST
                )
                if F is not None
                else 0.0
            )
            aY = {}
            if dm and t2d and arm != "baseline":
                aY = aY_ref[t]
            elif dm and t2d:
                aY = incidence_split(X, F, dm)
                aY_ref[t] = aY
            Rt = None
            RtD = None
            if onoff is not None and arm in ("glp1", "combo"):
                Rt, RtD = ps.retained(PQ["nd"], onoff, "nd"), ps.retained(
                    PQ["d"], onoff, "d"
                )
                XT = sum(X[(sx, st, "T")].sum() for sx in SEXES for st in ST)
                wd = (
                    (1 - sum(F[(sx, st, "T")].sum() for sx in SEXES for st in ST) / XT)
                    if (F is not None and XT > 0)
                    else 0.0
                )
                Rt = dict(
                    Rt,
                    on_all=(1 - wd) * Rt["on"] + wd * RtD["on"],
                    on_nd=Rt["on"],
                    on_d=RtD["on"],
                    wd=wd,
                )
            fbp.append(Rt["on_all"] if Rt else None)
            wdm.append((Rt["wd"], Rt["on_nd"], Rt["on_d"]) if Rt else None)
            for sx in SEXES:
                for k in KS:
                    if k == "T" and not any(X[(sx, st, "T")].any() for st in ST):
                        continue
                    kk = "T" if k == "T" else "U"
                    mult = arms_mult(arm, t, kk, Rt)
                    if t2d:
                        for st in ST:
                            nN = st == "N" and sx in t2d.get("e_ow_N", {})
                            en = (
                                t2d["e_ow_N" if nN else "e_ow"][sx]
                                if k in ("T", "E")
                                else t2d["e_no_N" if nN else "e_no"][sx]
                            )
                            lam0 = t2d["lam"][sx] * en
                            p0 = 1 - np.exp(-lam0)
                            p1 = 1 - np.exp(-lam0 * mult.get("T2D", 1.0))
                            if k == "T":
                                convT += float((F[(sx, st, k)] * p0).sum())
                            F[(sx, st, k)] = F[(sx, st, k)] * (1 - p0)
                            cases = G[(sx, st, k)] * p1
                            yT += float(cases.sum())
                            G[(sx, st, k)] = G[(sx, st, k)] - cases
                    r = RA[sx]

                    def capped(m_):
                        if RF is not None and k == "T" and "post" in m_:
                            red = 1 - m_["post"]
                            return dict(
                                m_,
                                post_I=np.where(
                                    r["mu_I"] > 0,
                                    1
                                    - red
                                    * np.minimum(r["mu_I"], RF[sx]["mu_I"])
                                    / np.where(r["mu_I"] > 0, r["mu_I"], 1),
                                    1.0,
                                ),
                                post_S=np.where(
                                    r["mu_S"] > 0,
                                    1
                                    - red
                                    * np.minimum(r["mu_S"], RF[sx]["mu_S"])
                                    / np.where(r["mu_S"] > 0, r["mu_S"], 1),
                                    1.0,
                                ),
                            )
                        return m_

                    if t2d:
                        aI, aS = aY[sx] if dm else (1.0, 1.0)
                        hI, hS = (dm["hr_I"], dm["hr_S"]) if dm else (1.0, 1.0)
                        hz0 = hazards(
                            r["lam_I"] * aI,
                            r["lam_S"] * aS,
                            r["mu_I"],
                            r["mu_S"],
                            r["mu_0"],
                            r["q_i"],
                            r["q_s"],
                            r["rr_cond"],
                            capped(mult),
                        )
                        mult1 = (
                            arms_mult(arm, t, kk, RtD if RtD is not None else Rt, True)
                            if dm
                            else arms_mult(arm, t, kk, RtD if RtD is not None else Rt)
                        )
                        hz1 = hazards(
                            r["lam_I"] * aI * hI,
                            r["lam_S"] * aS * hS,
                            r["mu_I"],
                            r["mu_S"],
                            r["mu_0"],
                            r["q_i"],
                            r["q_s"],
                            r["rr_cond"],
                            capped(mult1),
                        )
                        xf = {st: F[(sx, st, k)] for st in ST}
                        xd = {
                            st: np.maximum(X[(sx, st, k)] - F[(sx, st, k)], 0.0)
                            for st in ST
                        }
                        nf, ev0 = step_vec(xf, hz0)
                        nd, ev1 = step_vec(xd, hz1)
                        for st in ST:
                            new[(sx, st, k)] += nf[st] + nd[st]
                            newF[(sx, st, k)] += nf[st]
                        if track_t2d:
                            ng, _ = step_vec({st: G[(sx, st, k)] for st in ST}, hz0)
                            for st in ST:
                                newG[(sx, st, k)] += ng[st]
                        ev = {k_: ev0[k_] + ev1[k_] for k_ in ev0}
                    else:
                        hz = hazards(
                            r["lam_I"],
                            r["lam_S"],
                            r["mu_I"],
                            r["mu_S"],
                            r["mu_0"],
                            r["q_i"],
                            r["q_s"],
                            r["rr_cond"],
                            capped(mult),
                        )
                        nx, ev = step_vec({st: X[(sx, st, k)] for st in ST}, hz)
                        for st in ST:
                            new[(sx, st, k)] += nx[st]
                    dcv = ev["fat_I"] + ev["post_I"] + ev["fat_S"] + ev["post_S"]
                    if t == 1:
                        y1[sx]["I"] += ev["fat_I"] + ev["post_I"]
                        y1[sx]["S"] += ev["fat_S"] + ev["post_S"]
                        y1[sx]["all"] += dcv + ev["bg"]
                    yI += float(np.sum(ev["fat_I"] + ev["post_I"]))
                    yS += float(np.sum(ev["fat_S"] + ev["post_S"]))
                    by_age += np.bincount(GIDX, weights=dcv, minlength=len(AGE5))
                    cum_dead += float(np.sum(dcv + ev["bg"]))
            if t2d:
                for D_ in (newF, newG):
                    for key in D_:
                        D_[key] = age_one_year(D_[key])
                cT2D.append(yT)
            for key in new:
                new[key] = age_one_year(new[key])
            if entrants is not None and t < years:
                for sx in SEXES:
                    n_t = float(entrants[sx]["n"][t - 1])
                    for st in ST:
                        for k_ in KS:
                            add = n_t * entrants[sx]["share"][(st, k_)]
                            new[(sx, st, k_)][0] += add
                            tot0 += add
                            if t2d:
                                fr0 = float(
                                    np.atleast_1d(t2d["free0"][(sx, st, k_)])[0]
                                )
                                newF[(sx, st, k_)][0] += add * fr0
                                newG[(sx, st, k_)][0] += add * fr0
            X = new
            dI.append(yI)
            dS.append(yS)
            if t2d:
                F = newF
                G = newG
            if Rt is not None:
                for g_, R_ in (("nd", Rt), ("d", RtD)):
                    PQ[g_] = ps.next_states(PQ[g_], R_["o"])
                if F is not None and convT > 0:
                    Dm_ = sum(
                        (X[(sx, st, "T")] - F[(sx, st, "T")]).sum()
                        for sx in SEXES
                        for st in ST
                    )
                    PQ["d"] = ps.carry_converters(PQ["d"], PQ["nd"], Dm_, convT, d0T)
            if arm in ("glp1", "combo"):
                Tm = sum(X[(sx, st, "T")].sum() for sx in SEXES for st in ST)
                Fm = (
                    sum(F[(sx, st, "T")].sum() for sx in SEXES for st in ST)
                    if F is not None
                    else Tm
                )
                Et, K, enrolled = allocate(
                    X, F, G, Tm, False, C, mode, alloc, select, tier2, prio
                )
                Tn = sum(X[(sx, st, "T")].sum() for sx in SEXES for st in ST)
                if onoff is not None and Tn > 0:
                    Fn = (
                        sum(F[(sx, st, "T")].sum() for sx in SEXES for st in ST)
                        if F is not None
                        else Tn
                    )
                    for g_, before, after in (("nd", Fm, Fn), ("d", Tm - Fm, Tn - Fn)):
                        PQ[g_] = ps.mix_new_enrolment(PQ[g_], before, after)
                Tcvd = sum(X[(sx, st, "T")].sum() for sx, st in prio)
                alive_t = sum(v.sum() for v in X.values())
                trace.append(
                    dict(
                        arm=arm,
                        year=t,
                        alive=alive_t,
                        eligible=Et,
                        target=K,
                        enrolled_before=Tm,
                        new_enrolment=enrolled,
                        enrolled_after=Tn,
                        coverage_before=Tm / Et if Et else np.nan,
                        coverage_after=Tn / Et if Et else np.nan,
                        share_prior_cvd=Tcvd / Tn if Tn else np.nan,
                        on_share=fbp[-1] if fbp[-1] is not None else np.nan,
                    )
                )
        alive = sum(v.sum() for v in X.values())
        res[arm] = dict(
            IHD=dI,
            IschStroke=dS,
            conservation=(alive + cum_dead) / tot0 - 1,
            trace=trace,
            T2D=cT2D,
            tau=tau,
            by_age=by_age,
            fbp=fbp,
            wdm=wdm,
            y1=y1,
            init_alloc=init_alloc if arm in ("glp1", "combo") else None,
        )
    return res
