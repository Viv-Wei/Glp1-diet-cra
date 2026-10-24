import numpy as np
from .settings import AGES, AGE5, G0, GIDX, SEXES, ST


def overweight_by_state(iso, cnt, ow, gI, gS, checks):

    P = {st: cnt[st] / cnt["pop"] for st in ST}
    pI, pS, pB = P["I"] + P["B"], P["S"] + P["B"], P["B"]
    lo, hi = 0.0, 1.0
    if pB > 0:
        lo = max(
            lo,
            (pI * gI - (pI - pB)) / pB,
            (pS * gS - (pS - pB)) / pB,
            (pI * gI + pS * gS - ow) / pB,
        )
        hi = min(
            hi,
            pI * gI / pB,
            pS * gS / pB,
            (pI * gI + pS * gS - (ow - P["N"])) / pB if P["N"] > 0 else 1.0,
        )
    gB_target = max(gI, gS)
    feasible = lo <= hi + 1e-12
    gB = min(max(gB_target, lo), hi) if feasible else gB_target
    g = dict(
        B=gB,
        I=(pI * gI - pB * gB) / max(P["I"], 1e-12),
        S=(pS * gS - pB * gB) / max(P["S"], 1e-12),
    )
    g["N"] = (ow - g["I"] * P["I"] - g["S"] * P["S"] - g["B"] * P["B"]) / max(
        P["N"], 1e-12
    )
    checks["marginal"].append(
        dict(
            iso3=iso,
            gI_in=gI,
            gS_in=gS,
            ow_in=ow,
            gB=gB,
            gB_target=gB_target,
            gB_feasible_lo=lo,
            gB_feasible_hi=hi,
            interval_empty=not feasible,
            **{f"g_{st}": g[st] for st in ST},
            gI_out=(g["I"] * P["I"] + g["B"] * P["B"]) / pI if pI else np.nan,
            gS_out=(g["S"] * P["S"] + g["B"] * P["B"]) / pS if pS else np.nan,
            ow_out=sum(g[st] * P[st] for st in ST),
        )
    )
    return g


def eligible_by_age(iso, g, base, selb, ecol, checks):

    eg = {sx: {st: np.full(len(AGES), g[st]) for st in ST} for sx in SEXES}
    if selb is not None and (selb.iso3 == iso).any():
        sb = selb[selb.iso3 == iso].set_index(["sex", "a0"])[ecol]
        wv = {
            sx: np.array(
                [float(sb.get((sx, G0[GIDX[i]]), np.nan)) for i in range(len(AGES))]
            )
            for sx in SEXES
        }
        if all(np.isfinite(wv[sx]).all() for sx in SEXES):
            for st in ST:
                tot_ = sum(base[sx][st].sum() for sx in SEXES)
                tgt_ = g[st] * tot_
                if tot_ <= 0 or tgt_ <= 0:
                    continue
                e_ = {
                    sx: wv[sx]
                    * g[st]
                    * tot_
                    / sum((base[s_][st] * wv[s_]).sum() for s_ in SEXES)
                    for sx in SEXES
                }
                for _ in range(100):
                    for sx in SEXES:
                        e_[sx] = np.minimum(e_[sx], 1.0)
                    cur = sum((base[sx][st] * e_[sx]).sum() for sx in SEXES)
                    if abs(cur / tgt_ - 1) < 1e-12:
                        break
                    free_ = sum(
                        (base[sx][st] * e_[sx] * (e_[sx] < 1)).sum() for sx in SEXES
                    )
                    if free_ <= 0:
                        break
                    fac = 1 + (tgt_ - cur) / free_
                    for sx in SEXES:
                        e_[sx] = np.where(e_[sx] < 1, e_[sx] * fac, 1.0)
                for sx in SEXES:
                    eg[sx][st] = e_[sx]
    checks.setdefault("eligibility_age", []).extend(
        dict(
            iso3=iso,
            sex=sx,
            state=st,
            age=AGE5[gi],
            eligible_share=float(np.average(eg[sx][st][GIDX == gi])),
            country_share=g[st],
        )
        for sx in SEXES
        for st in ST
        for gi in range(len(AGE5))
    )
    return eg


def coverage_target(iso, g, cnt, ow, coverage, alloc, afford, checks):

    n_ow = ow * cnt["pop"]
    M1 = sum(g[st] * cnt[st] for st in ["I", "S", "B"])
    C = (
        afford.get(iso, 1.0)
        if coverage == "afford"
        else M1 / n_ow if coverage == "cstar" else float(coverage)
    )
    M = C * n_ow
    c1 = min(1.0, M / M1) if M1 > 0 else 0.0
    if alloc == "uniform":
        frac = {st: C * g[st] for st in ST}
    else:
        c1 = min(1.0, M / M1) if M1 > 0 else 0.0
        cN = max(M - c1 * M1, 0.0) / max(g["N"] * cnt["N"], 1e-12)
        frac = dict(
            N=min(cN, 1.0) * g["N"], I=c1 * g["I"], S=c1 * g["S"], B=c1 * g["B"]
        )
    treated = sum(frac[st] * cnt[st] for st in ST)
    prior = sum(frac[st] * cnt[st] for st in ["I", "S", "B"])
    checks["allocation"].append(
        dict(
            iso3=iso,
            coverage=C,
            target=M,
            treated=treated,
            rel_err=treated / M - 1 if M else 0,
            share_treated_with_prior_cvd=prior / treated if treated else np.nan,
        )
    )
    return C


def initial_states(
    iso, base, eg, coverage, select_like, seltab, bmi_min, dm_pr, checks
):

    init = {sx: {} for sx in SEXES}
    C = None
    if select_like:
        se = seltab[seltab.iso3 == iso].set_index(["sex", "a0"])
        PRD = dm_pr
        Cs = 1.0 if coverage in ("cstar", "afford") else float(coverage)
        C = Cs
        pool = 0.0
        tr = 0.0
        for sx in SEXES:
            gi_a = np.array([G0[i] for i in GIDX])
            col = lambda c: np.array(
                [float(se[c].get((sx, int(x)), np.nan)) for x in gi_a]
            )
            s27 = (lambda c: np.ones(len(AGES))) if bmi_min == "27" else col
            pdm = np.nan_to_num(col("p_dm"))
            pdm_c = pdm * PRD / (pdm * PRD + 1 - pdm)
            age45 = (AGES >= 45).astype(float)
            el = dict(
                N=eg[sx]["N"] * np.nan_to_num(s27("share27_ow_pop")) * (1 - pdm),
                I=eg[sx]["I"] * np.nan_to_num(s27("share27_ow_ihd")) * (1 - pdm_c),
                S=eg[sx]["S"] * np.nan_to_num(s27("share27_ow_is")) * (1 - pdm_c),
                B=eg[sx]["B"]
                * np.maximum(
                    np.nan_to_num(s27("share27_ow_ihd")),
                    np.nan_to_num(s27("share27_ow_is")),
                )
                * (1 - pdm_c),
            )
            for st in ST:
                e_ = np.clip(el[st], 0, 1) * age45
                b_ = base[sx][st]
                t_ = b_ * e_ * (min(Cs, 1.0) if st != "N" else 0.0)
                init[sx][(st, "T")] = b_ * 0.0
                init[sx][(st, "E")] = b_ * e_
                init[sx][(st, "O")] = b_ * (1 - e_)
                if st != "N":
                    pool += (b_ * e_).sum()
                    tr += t_.sum()
        checks["allocation"][-1].update(
            coverage=Cs,
            target=Cs * pool,
            treated=tr,
            rel_err=tr / (Cs * pool) - 1 if pool * Cs else 0,
            share_treated_with_prior_cvd=1.0,
            select_like_pool=pool,
        )
    else:
        for sx in SEXES:
            for st in ST:
                init[sx][(st, "T")] = base[sx][st] * 0.0
                init[sx][(st, "E")] = base[sx][st] * eg[sx][st]
                init[sx][(st, "O")] = base[sx][st] * (1 - eg[sx][st])
    return init, C


def entrants(iso, wp_all, pop, base, eg, years):

    wp = (
        wp_all[wp_all.Location == "World"]
        if iso == "GLB"
        else wp_all[wp_all.ISO3_code == iso]
    )
    if not len(wp):
        return None
    ent = {}
    for sx, col in [("Male", "PopMale"), ("Female", "PopFemale")]:
        ser = wp.set_index("Time")[col].sort_index() * 1000.0
        scale = float(pop.get((iso, sx, AGE5[0]), 0.0)) / float(ser.get(2023, np.nan))
        n = np.array(
            [float(ser.get(2023 + t, np.nan)) / 5 * scale for t in range(1, years + 1)]
        )
        tot25 = sum(base[sx][st][0] for st in ST)
        share = {}
        for st in ST:
            sst = base[sx][st][0] / tot25 if tot25 > 0 else (1.0 if st == "N" else 0.0)
            share[(st, "T")] = 0.0
            share[(st, "E")] = sst * eg[sx][st][0]
            share[(st, "O")] = sst * (1 - eg[sx][st][0])
        ent[sx] = dict(n=np.nan_to_num(n), share=share)
    return ent
