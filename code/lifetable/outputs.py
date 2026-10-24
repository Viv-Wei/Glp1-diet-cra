import os
import numpy as np, pandas as pd
from .settings import AGES, AGE5, GIDX, SEXES, ST, ARMS, KS
from .transitions import hazards, step_vec
from .calibration import CKD_PREV_CAP


def full_model_check(iso, R_tabs, init, res, checks):

    for sx in SEXES:
        RA_ = {
            k_: np.array([R_tabs[sx][GIDX[i]][k_] for i in range(len(AGES))])
            for k_ in (
                "lam_I",
                "lam_S",
                "mu_I",
                "mu_S",
                "mu_0",
                "q_i",
                "q_s",
                "rr_cond",
            )
        }
        _, evr = step_vec(
            {st: sum(init[sx][(st, k_)] for k_ in KS) for st in ST},
            hazards(
                RA_["lam_I"],
                RA_["lam_S"],
                RA_["mu_I"],
                RA_["mu_S"],
                RA_["mu_0"],
                RA_["q_i"],
                RA_["q_s"],
                RA_["rr_cond"],
            ),
        )
        ref = dict(
            I=evr["fat_I"] + evr["post_I"],
            S=evr["fat_S"] + evr["post_S"],
            all=evr["fat_I"] + evr["post_I"] + evr["fat_S"] + evr["post_S"] + evr["bg"],
        )
        fy = res["baseline"]["y1"][sx]
        for gi in range(len(AGE5)):
            m_ = GIDX == gi
            checks.setdefault("calibration_full", []).append(
                dict(
                    iso3=iso,
                    sex=sx,
                    age=AGE5[gi],
                    **{
                        f"err_d_{c_}": (
                            (fy[c_][m_].sum() / ref[c_][m_].sum() - 1)
                            if ref[c_][m_].sum() > 0
                            else 0.0
                        )
                        for c_ in ref
                    },
                )
            )


def allocation_update(res, checks):

    ia = res["glp1"]["init_alloc"]
    checks["allocation"][-1].update(
        target=ia["target"],
        treated=ia["treated"],
        rel_err=ia["treated"] / ia["target"] - 1 if ia["target"] else 0.0,
        share_treated_with_prior_cvd=ia["share_prior_cvd"],
        share_treated_dm_no_cvd=ia["share_dm_no_cvd"],
    )


def country_results(iso, res, years, C, t2d, checks, rows, t2rows, agerows):

    if t2d:
        b0 = np.array(res["baseline"]["T2D"])
        for arm in ["glp1", "diet", "combo"]:
            av = b0 - np.array(res[arm]["T2D"])
            t2rows.append(
                dict(
                    iso3=iso,
                    arm=arm,
                    cases_averted_total=float(av.sum()),
                    cases_baseline_total=float(b0.sum()),
                    coverage_used=C,
                    **{f"yr{i}": float(v) for i, v in enumerate(av, 1)},
                )
            )
    checks["conservation"].append(
        dict(iso3=iso, **{f"rel_err_{arm}": res[arm]["conservation"] for arm in ARMS})
    )
    for tr in res["glp1"]["trace"]:
        checks.setdefault("programme", []).append(dict(iso3=iso, **tr))
    for arm in ["glp1", "diet", "combo"]:
        av_age = res["baseline"]["by_age"] - res[arm]["by_age"]
        agerows.extend(
            dict(iso3=iso, arm=arm, age=AGE5[gi], deaths_averted=float(v))
            for gi, v in enumerate(av_age)
        )
    for o in ["IHD", "IschStroke"]:
        b0 = np.array(res["baseline"][o])
        for arm in ["glp1", "diet", "combo"]:
            av = b0 - np.array(res[arm][o])
            row = dict(
                iso3=iso,
                sex="Both",
                outcome=o,
                arm=arm,
                averted_10yr=float(av.sum()),
                averted_yr1=float(av[0]),
                averted_yr10=float(av[min(9, years - 1)]),
                coverage_used=C,
            )
            row.update({f"yr{i}": float(v) for i, v in enumerate(av, 1)})
            rows.append(row)


def write_all(out, s, a, rows, agerows, ckrows, ckd_flags, t2rows, checks, s_S):

    pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(agerows).to_csv(os.path.splitext(out)[0] + "_byage.csv", index=False)
    if ckrows:
        ckdf = pd.DataFrame(ckrows)
        ckdf.to_csv(os.path.splitext(out)[0] + "_ckd.csv", index=False)
        if ckd_flags:
            fl = pd.DataFrame(ckd_flags)
            fl.to_csv(os.path.splitext(out)[0] + "_ckd_flags.csv", index=False)
            print(
                f"  CKD: prevalence capped at {CKD_PREV_CAP} in {int(fl.capped.sum())} strata, calibration fallback in "
                f"{int(fl.fallback.fillna(False).sum())}: "
                + ", ".join(
                    f"{r.iso3} {r.sex[0]} {r.age.split()[0]}" for r in fl.itertuples()
                )
            )
        if ckdf.deaths_averted_total.isna().any():
            print(
                "  [warning] CKD results missing for: "
                + ", ".join(
                    sorted(ckdf[ckdf.deaths_averted_total.isna()].iso3.unique())
                )
            )
        g_ = ckdf[ckdf.iso3 == "GLB"].set_index("arm").deaths_averted_total / 1e6
        print(
            f"  CKD extended (SBP fall {s.sbp_shift} mmHg without and {s.sbp_shift_dm} mmHg with diabetes, while on drug): "
            "global deaths averted (millions) "
            + ", ".join(f"{k} {v:.3f}" for k, v in g_.items())
        )
    if t2rows and s.t2d_out:
        t2df = pd.DataFrame(t2rows)
        t2df.to_csv(os.path.splitext(out)[0] + "_t2d.csv", index=False)
        g_ = t2df[t2df.iso3 == "GLB"].set_index("arm").cases_averted_total / 1e6
        print(
            f"  T2D (RR {s.t2d_rr}): global new cases averted over {a.years} years (millions) "
            + ", ".join(f"{k} {v:.2f}" for k, v in g_.items())
        )
    rep = {k: pd.DataFrame(v) for k, v in checks.items() if len(v)}
    stem = os.path.splitext(out)[0]
    for k, v in rep.items():
        v.to_csv(f"{stem}_check_{k}.csv", index=False)
    f, c, cs, mg, al = (
        rep["feasibility"],
        rep["calibration"],
        rep["conservation"],
        rep["marginal"],
        rep["allocation"],
    )
    print(
        f"[layer 1] s_IHD = {s.ihd_first:.2f}, IS 28-day case fatality {s.is_cf28:.2f} -> s_IS = {s_S:.3f}, fatal mode = {s.fatal_mode}, "
        f"allocation = {s.alloc}, coverage = {a.coverage}"
    )
    print(
        f"  feasibility: IHD strata capped {int(f.capped_IHD.sum())}/{len(f)}, IS strata capped {int(f.capped_IS.sum())}/{len(f)}"
    )
    print(
        f"  calibration: max |error| inc_I {c.err_inc_I.abs().max():.2e}, inc_S {c.err_inc_S.abs().max():.2e}, "
        f"d_I {c.err_d_I.abs().max():.2e}, d_S {c.err_d_S.abs().max():.2e}, all-cause {c.err_d_all.abs().max():.2e}"
    )
    if "calibration_full" in rep:
        cf = rep["calibration_full"]
        print(
            f"  full model (diabetes strata) vs calibrated rates, year-1 deaths: max |error| IHD {cf.err_d_I.abs().max():.2e}, "
            f"IS {cf.err_d_S.abs().max():.2e}, all-cause {cf.err_d_all.abs().max():.2e}"
        )
    print(
        f"  conservation: max |relative error| {cs.filter(like='rel_err').abs().max().max():.2e}"
    )
    print(
        f"  marginal: g_B interval empty in {int(mg.interval_empty.sum())} countries; max |gI_out-gI_in| "
        f"{(mg.gI_out - mg.gI_in).abs().max():.2e}, |ow_out-ow_in| {(mg.ow_out - mg.ow_in).abs().max():.2e}"
    )
    print(f"  allocation: max |treated/target - 1| {al.rel_err.abs().max():.2e}")
    if "programme" in rep:
        pg = rep["programme"]
        gl = pg[pg.iso3 == "GLB"].set_index("year")
        print(
            f"  programme ({s.treat}), global: coverage after top-up years 1/10/20/30 = "
            + ", ".join(
                f"{gl.coverage_after.get(y, np.nan):.3f}" for y in [1, 10, 20, 30]
            )
            + f"; max coverage before top-up across countries {pg.coverage_before.max():.3f} (target {pg.target.div(pg.eligible).max():.3f})"
        )
    print(f"-> {out} and {stem}_check_*.csv")
