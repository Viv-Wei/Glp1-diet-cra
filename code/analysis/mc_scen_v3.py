import argparse, glob, itertools, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

MAIN = os.environ.get("MC_DIR", "data/results/v3/mc")
D = os.environ.get("MC_SCEN_DIR", "data/results/v3/mc_scen")
lnp = lambda m, lo, hi: (np.log(m), (np.log(hi) - np.log(lo)) / (2 * 1.96))
SC_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
SW_GRID = SC_GRID
HR_MAIN = dict(
    mi=(0.77, 0.51, 1.17),
    st=(0.93, 0.75, 1.16),
    cvd=(0.87, 0.73, 1.03),
    mi_dm=(0.87, 0.79, 0.96),
    st_dm=(0.93, 0.80, 1.09),
    cvd_dm=(0.86, 0.79, 0.93),
)


def scenarios():
    S = {"main": dict()}
    S["persist_full"] = dict(env=dict(SENS_STOP_MULT=0))
    S["stop_x0.5"] = dict(env=dict(SENS_STOP_MULT=0.5))
    S["stop_x1.5"] = dict(env=dict(SENS_STOP_MULT=1.5))
    S["no_restart"] = dict(env=dict(SENS_RESTART_MULT=0))
    S["resid_0"] = dict(env=dict(SENS_RESID_CV=0, SENS_RESID_T2D=0))
    S["const_retention"] = dict(env=dict(SENS_ONOFF="0", SENS_RETAIN=0.667 / 0.883))
    S["bmi25"] = dict(env=dict(SENS_BMI_MIN="25"))
    S["no_dm_tier"] = dict(env=dict(SENS_DM_TIER="0"))
    S["class_overall"] = dict(
        hr={
            "": ((0.85, 0.78, 0.94), (0.93, 0.81, 1.06), (0.87, 0.81, 0.92)),
            "_DM": ((0.85, 0.78, 0.94), (0.93, 0.81, 1.06), (0.87, 0.81, 0.92)),
        }
    )
    S["select_only"] = dict(
        hr={"": ((0.72, 0.61, 0.85), (0.93, 0.74, 1.15), (0.85, 0.71, 1.01))}
    )
    S["sattar_dm"] = dict(
        hr={"_DM": ((0.90, 0.83, 0.98), (0.83, 0.76, 0.92), (0.87, 0.80, 0.94))}
    )
    S["tier2_rewind"] = dict(env=dict(SENS_TIER2_MI=0.96, SENS_TIER2_STROKE=0.76))
    S["primary_weak"] = dict(env=dict(SENS_PRIMARY_SCALE=0.4))
    S["tier3_none"] = dict(env=dict(SENS_TIER3_SCALE=0))
    S["tier3_bmi27"] = dict(env=dict(SENS_TIER3_BMI="27"))
    S["itt_select"] = dict(env=dict(SENS_ITT_CV=0.883, SENS_ITT_T2D=0.869))
    S["post_A"] = dict(env=dict(SENS_POST_NONE="1"))
    S["post_C"] = dict(env=dict(SENS_POST_NONE="1", SENS_FATAL_MODE="none"))
    S["restart_late0.5"] = dict(env=dict(SENS_RESTART_LATE=0.5))
    S["restart_late0"] = dict(env=dict(SENS_RESTART_LATE=0))
    S["bmi_gamma"] = dict(
        env=dict(
            SENS_OW_FILE="data/params/overweight_prevalence_bmi27_gamma.csv",
            SENS_OW_CVD_FILE="data/params/overweight_cvd_bmi27_gamma.csv",
        )
    )
    S["dm_pr2"] = dict(env=dict(SENS_DM_PR=2.0))
    S["dm_pr3"] = dict(env=dict(SENS_DM_PR=3.0))
    S["dm_no_risk"] = dict(env=dict(SENS_DM_CVD="0"))
    for pm in (0.8, 0.6, 0.4):
        S[f"maintain_{pm}"] = dict(env=dict(SENS_POLICY_MAINTAIN=pm))
    for rd in (12, 16):
        S[f"phase_{rd}"] = dict(ramp=rd)
    S["bmi30"] = dict(env=dict(SENS_ELIG_FILE="data/params/elig_scale_bmi30_rel27.csv"))
    S["uniform"] = dict(env=dict(SENS_ALLOC="uniform"))
    S["baseline_cohort"] = dict(env=dict(SENS_TREAT="baseline"))
    S["dynamic"] = dict(env=dict(SENS_TREAT="dynamic"), grid=[0.2])
    for dsc in ("tailored", "sodium30", "aspirational"):
        S[dsc] = dict(diet=dsc)
    S["cap"] = dict(env=dict(SENS_CAP_REF="USA"))
    S["tailored_cap"] = dict(env=dict(SENS_CAP_REF="USA"), diet="tailored")
    S["select_like"] = dict(
        env=dict(SENS_SELECT_LIKE="1"),
        hr={"": ((0.72, 0.61, 0.85), (0.93, 0.74, 1.15), (0.85, 0.71, 1.01))},
    )
    for s_, fm in itertools.product((0.25, 0.40, 0.55), ("same", "cvd", "none")):
        S[f"s{s_}_{fm}"] = dict(env=dict(SENS_IHD_FIRST=s_, SENS_FATAL_MODE=fm))
    for k, v in [
        ("is_cf10", dict(SENS_IS_CF28=0.10)),
        ("is_cf20", dict(SENS_IS_CF28=0.20)),
        ("k0.8", dict(SENS_K_DUR=0.8)),
        ("k1.2", dict(SENS_K_DUR=1.2)),
        ("or2", dict(SENS_OR_COMORB=2)),
        ("or4", dict(SENS_OR_COMORB=4)),
        ("rr1.5", dict(SENS_RR_COND=1.5)),
        ("rr2", dict(SENS_RR_COND=2)),
    ]:
        S[k] = dict(env=v)
    S["open_pop"] = dict(env=dict(SENS_OPEN_POP="1"))
    for pm, sm, rd in itertools.product(
        (1.0, 0.8, 0.6, 0.4), (1.5, 1.0, 0.5, 0.0), (8, 12, 16)
    ):
        S[f"sw_pm{pm}_stop{sm}_rd{rd}"] = dict(
            env=dict(SENS_POLICY_MAINTAIN=pm, SENS_STOP_MULT=sm), ramp=rd, grid=SW_GRID
        )
    return S


def prepare(a):
    S = scenarios()
    names = a.only.split(",") if a.only else list(S)
    os.makedirs(D, exist_ok=True)
    n = 0
    with open(f"{D}/jobs.txt", "w") as f:
        for d in range(a.draws):
            for nm in names:
                for c in [f"{g:.2f}" for g in S[nm].get("grid", SC_GRID)] + (
                    [] if "grid" in S[nm] and S[nm]["grid"] == [0.2] else ["afford"]
                ):
                    f.write(f"{nm} {d} {c}\n")
                    n += 1
    print(f"{len(names)} scenarios x {a.draws} draws: {n} jobs -> {D}/jobs.txt")


def worker(a):
    out = f"{D}/out/{a.scen}/d{a.draw}_c{a.cov}.csv"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sc = scenarios()[a.scen]
    r = pd.read_csv(f"{MAIN}/draws.csv").set_index("draw").loc[a.draw]
    env = os.environ | dict(
        SENS_TREAT="maintain",
        SENS_ALLOC="risk",
        SENS_HR_MI=str(r.hr_mi),
        SENS_HR_STROKE=str(r.hr_stroke),
        SENS_HR_CVD=str(r.hr_cvd),
        SENS_IHD_FIRST=str(r.s_ihd),
        SENS_STOP_MULT=str(r.stop_mult),
        SENS_RESTART_MULT=str(r.restart_mult),
        SENS_RESID_CV=str(r.resid_cv),
        SENS_RESID_T2D=str(r.resid_t2d),
        SENS_SBP_SHIFT=str(r.sbp_shift),
        SENS_SBP_SHIFT_DM=str(r.sbp_shift_dm),
        SENS_HR_MI_DM=str(r.hr_mi_dm),
        SENS_HR_STROKE_DM=str(r.hr_stroke_dm),
        SENS_HR_CVD_DM=str(r.hr_cvd_dm),
        SENS_DM_HR_IHD=str(r.dm_hr_ihd),
        SENS_DM_HR_IS=str(r.dm_hr_is),
        LT_DP_OVERRIDE=f"{MAIN}/diet/draw{a.draw}.csv",
        SENS_T2D="0",
        SENS_CKD="0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    for sfx, sets in sc.get("hr", {}).items():
        for (nm_, col_), new_ in zip(
            (("SENS_HR_MI", "mi"), ("SENS_HR_STROKE", "st"), ("SENS_HR_CVD", "cvd")),
            sets,
        ):
            key_ = col_ + ("_dm" if sfx else "")
            val_ = float(
                r[
                    {"mi": "hr_mi", "st": "hr_stroke", "cvd": "hr_cvd"}[col_]
                    + ("_dm" if sfx else "")
                ]
            )
            z = (np.log(val_) - lnp(*HR_MAIN[key_])[0]) / lnp(*HR_MAIN[key_])[1]
            mu, sd = lnp(*new_)
            env[nm_ + sfx] = str(float(np.exp(mu + sd * z)))
    env.update({k: str(v) for k, v in sc.get("env", {}).items()})
    with tempfile.TemporaryDirectory() as td:
        o = f"{td}/lt.csv"
        p = subprocess.run(
            [
                "python",
                "code/lifetable_v3.py",
                "--years",
                "30",
                "--coverage",
                a.cov,
                "--diet-scenario",
                sc.get("diet", "feasible"),
                "--ramp-diet",
                str(sc.get("ramp", 8)),
                "--out",
                o,
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            print(
                f"{a.scen} draw {a.draw} cov {a.cov} failed: {p.stderr[-300:]}",
                file=sys.stderr,
            )
            return
        t = pd.read_csv(o)
        t = t[t.outcome.isin(["IHD", "IschStroke"])]
        yr = [c for c in t.columns if c.startswith("yr")]
        y = t.groupby(["iso3", "arm"])[yr].sum()
        disc = np.array([1.03 ** -(int(c[2:]) - 0.5) for c in yr])
        g = pd.DataFrame(
            {"deaths": y.sum(axis=1), "deaths_disc3": (y.values * disc).sum(axis=1)},
            index=y.index,
        ).reset_index()
    g.insert(0, "cov", a.cov)
    g.insert(0, "draw", a.draw)
    g.to_csv(out, index=False)


def per_draw(x):

    num = x[x["cov"] != "afford"].copy()
    num["c"] = num["cov"].astype(float)
    diet = x[x.arm == "diet"].groupby(["draw", "iso3"]).deaths.first()
    trt = num[num.arm == "glp1"].pivot_table(
        index=["draw", "iso3"], columns="c", values="deaths"
    )
    rows = []
    for (d, iso), y in trt.iterrows():
        cv, v = y.index.values, y.values
        dd = diet.get((d, iso), np.nan)
        if len(cv) == 1:
            rows.append(
                dict(
                    draw=d,
                    iso3=iso,
                    crossover=np.nan,
                    r10=np.nan,
                    r20=dd / v[0],
                    r50=np.nan,
                )
            )
            continue
        k = np.where(v >= dd)[0]
        c_ = (
            np.nan
            if len(k) == 0
            else (
                0.0
                if k[0] == 0
                else cv[k[0] - 1]
                + (dd - v[k[0] - 1])
                / (v[k[0]] - v[k[0] - 1])
                * (cv[k[0]] - cv[k[0] - 1])
            )
        )
        r100 = dd / v[-1] if cv[-1] >= 1.0 and v[-1] > 0 else np.nan
        rows.append(
            dict(
                draw=d,
                iso3=iso,
                crossover=c_,
                r10=dd / np.interp(0.1, cv, v),
                r20=dd / np.interp(0.2, cv, v),
                r50=dd / np.interp(0.5, cv, v),
                r100=r100,
                xext=c_ if np.isfinite(c_) else r100,
                t20=np.interp(0.2, cv, v) / dd if dd else np.nan,
            )
        )
    out = pd.DataFrame(rows)
    af = x[x["cov"] == "afford"].pivot_table(
        index=["draw", "iso3"], columns="arm", values="deaths"
    )
    if len(af):
        out = out.merge(
            (af.diet / af.glp1).rename("raff").reset_index(),
            on=["draw", "iso3"],
            how="left",
        )
    dt = (
        x[(x.arm == "diet")]
        .groupby(["draw", "iso3"])
        .deaths.first()
        .rename("diet_deaths")
        .reset_index()
    )
    return out.merge(dt, on=["draw", "iso3"], how="left")


def aggregate(a):
    _aggregate("deaths", "")
    _aggregate("deaths_disc3", "_disc3")


def _aggregate(col, suf):
    main = pd.concat(pd.read_csv(f) for f in glob.glob(f"{MAIN}/out/d*_c*.csv"))
    if col not in main:
        main = None
    pm_ = per_draw(main).set_index(["draw", "iso3"]) if main is not None else None
    summ, pair = [], []
    for nm in sorted(os.listdir(f"{D}/out")):
        fs = glob.glob(f"{D}/out/{nm}/d*_c*.csv")
        if not fs:
            continue
        x = pd.concat(pd.read_csv(f) for f in fs)
        if col not in x:
            continue
        x = x.assign(deaths=x[col])
        pdw = per_draw(x)
        for iso, g in pdw.groupby("iso3"):
            g = g.copy()
            multi = g.r10.notna().any()
            row = dict(
                scenario=nm,
                iso3=iso,
                draws=g.draw.nunique(),
                share_not_reached=g.crossover.isna().mean() if multi else np.nan,
                P_crossover_lt10=(g.crossover < 0.1).mean() if multi else np.nan,
                P_crossover_lt20=(g.crossover < 0.2).mean() if multi else np.nan,
                P_diet_leads_20=(g.r20 > 1).mean(),
            )
            if multi:
                g["crossover"] = g.crossover.fillna(np.inf)
            for m in [
                "crossover",
                "xext",
                "r10",
                "r20",
                "r50",
                "r100",
                "t20",
                "raff",
                "diet_deaths",
            ]:
                if m in g and g[m].notna().any():
                    row.update(
                        {
                            f"{m}_median": g[m].quantile(0.5, interpolation="lower"),
                            f"{m}_lo": g[m].quantile(0.025, interpolation="lower"),
                            f"{m}_hi": g[m].quantile(0.975, interpolation="higher"),
                        }
                    )
            summ.append(row)
            j = (
                g.set_index(["draw", "iso3"]).join(pm_, rsuffix="_main", how="inner")
                if pm_ is not None
                else []
            )
            if len(j):
                dc = (
                    j.crossover.fillna(np.inf) - j.crossover_main.fillna(np.inf)
                ).replace([np.inf, -np.inf, np.nan], np.nan)
                dr = j.r20 - j.r20_main
                pair.append(
                    dict(
                        scenario=nm,
                        iso3=iso,
                        d_crossover_median=dc.median(),
                        d_crossover_lo=dc.quantile(0.025),
                        d_crossover_hi=dc.quantile(0.975),
                        d_r20_median=dr.median(),
                        d_r20_lo=dr.quantile(0.025),
                        d_r20_hi=dr.quantile(0.975),
                        P_r20_gt1=(j.r20 > 1).mean(),
                    )
                )
    S, P = pd.DataFrame(summ), pd.DataFrame(pair)
    S.to_csv(f"{D}/summary_scenarios{suf}.csv", index=False)
    P.to_csv(f"{D}/summary_paired{suf}.csv", index=False)
    if suf:
        print(f"discounted (3%) summaries -> {D}/summary_scenarios{suf}.csv")
        return
    g = S[S.iso3 == "GLB"].set_index("scenario")
    cols = [
        c
        for c in [
            "draws",
            "crossover_median",
            "crossover_lo",
            "crossover_hi",
            "share_not_reached",
            "r20_median",
            "r20_lo",
            "r20_hi",
        ]
        if c in g
    ]
    print("Global, by scenario (crossover as share of eligible adults):")
    print(g[cols].round(3).to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q1 = sp.add_parser("prepare")
    q1.add_argument("--draws", type=int, default=300)
    q1.add_argument("--only", default="")
    q2 = sp.add_parser("worker")
    q2.add_argument("--scen")
    q2.add_argument("--draw", type=int)
    q2.add_argument("--cov")
    sp.add_parser("aggregate")
    a = p.parse_args()
    {"prepare": prepare, "worker": worker, "aggregate": aggregate}[a.cmd](a)
