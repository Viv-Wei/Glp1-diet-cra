import argparse, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

D = os.environ.get("MC_DIR", "data/results/v3/mc")
GRID = [round(i * 0.025, 3) for i in range(21)] + [0.6, 0.7, 0.8, 0.9, 1.0]
KEY = {0.1, 0.2, 0.5}
lnorm = lambda m, lo, hi: (np.log(m), (np.log(hi) - np.log(lo)) / (2 * 1.96))

HR_MAIN = dict(
    mi=(0.77, 0.51, 1.17),
    st=(0.93, 0.75, 1.16),
    cvd=(0.87, 0.73, 1.03),
    mi_dm=(0.87, 0.79, 0.96),
    st_dm=(0.93, 0.80, 1.09),
    cvd_dm=(0.86, 0.79, 0.93),
    t2d=(0.27, 0.24, 0.31),
)


def prepare(a):
    os.makedirs(f"{D}/diet", exist_ok=True)
    os.makedirs(f"{D}/out", exist_ok=True)
    rng = np.random.default_rng(20261015)
    rows = []
    for d in range(a.draws):
        hr = {k: float(np.exp(rng.normal(*lnorm(*v)))) for k, v in HR_MAIN.items()}
        rows.append(
            dict(
                draw=d,
                hr_mi=hr["mi"],
                hr_stroke=hr["st"],
                hr_cvd=hr["cvd"],
                hr_mi_dm=hr["mi_dm"],
                hr_stroke_dm=hr["st_dm"],
                hr_cvd_dm=hr["cvd_dm"],
                hr_t2d=hr["t2d"],
                s_ihd=rng.uniform(0.25, 0.55),
                stop_mult=rng.uniform(0.75, 1.25),
                restart_mult=rng.uniform(0.5, 1.5),
                resid_cv=rng.uniform(0.0, 2 / 3),
                resid_t2d=rng.uniform(0.0, 0.5),
                dm_hr_ihd=float(np.exp(rng.normal(*lnorm(2.00, 1.83, 2.19)))),
                dm_hr_is=float(np.exp(rng.normal(*lnorm(2.27, 1.95, 2.65)))),
                sbp_shift=rng.normal(3.80, (4.24 - 3.37) / (2 * 1.96)),
                sbp_shift_dm=rng.normal(2.13, (3.27 - 1.00) / (2 * 1.96)),
            )
        )
    dr = pd.DataFrame(rows)
    dp = pd.read_csv(os.environ["LT_DP_OVERRIDE"])
    if os.environ.get("MC_DIET_UNC") == "old":
        for d in range(a.draws):
            x = dp.copy()
            x["averted_frac_alldeaths"] = x.averted_frac_alldeaths * rng.normal(
                1.0, 0.06, len(x)
            )
            x.to_csv(f"{D}/diet/draw{d}.csv", index=False)
    else:
        u = pd.read_csv("data/params/diet_paf_uncertainty.csv").set_index("outcome")
        sd_lo = np.log(u.val / u.lower) / 1.96
        sd_hi = np.log(u.upper / u.val) / 1.96
        rng_d = np.random.default_rng(20261020)
        for d in range(a.draws):
            fac = {}
            for oc in u.index:
                z = rng_d.standard_normal()
                fac[oc] = float(np.exp((sd_lo[oc] if z < 0 else sd_hi[oc]) * z))
                dr.loc[d, f"diet_factor_{oc}"] = fac[oc]
            x = dp.copy()
            x["averted_frac_alldeaths"] = x.averted_frac_alldeaths * x.outcome.map(
                fac
            ).fillna(1.0)
            x.to_csv(f"{D}/diet/draw{d}.csv", index=False)
    dr.to_csv(f"{D}/draws.csv", index=False)
    with open(f"{D}/jobs.txt", "w") as f:
        for d in range(a.draws):
            for c in [f"{g:.3f}" for g in GRID] + ["afford"]:
                f.write(f"{d} {c}\n")
    print(f"{a.draws} draws, {a.draws * (len(GRID) + 1)} jobs -> {D}/jobs.txt")


def worker(a):
    od = os.environ.get("MC_OUT", f"{D}/out")
    os.makedirs(od, exist_ok=True)
    out = f"{od}/d{a.draw}_c{a.cov}.csv"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    r = pd.read_csv(f"{D}/draws.csv").set_index("draw").loc[a.draw]
    key = (
        a.cov == "afford" or float(a.cov) in KEY or os.environ.get("MC_ALL_KEY") == "1"
    )
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
        LT_DP_OVERRIDE=f"{D}/diet/draw{a.draw}.csv",
        SENS_T2D="1" if key else "0",
        SENS_CKD="1" if key else "0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    with tempfile.TemporaryDirectory() as td:
        tp = pd.read_csv("data/params/glp1_trial_params.csv")
        tp.loc[tp.param == "t2d_incidence_hr", "value"] = r.hr_t2d
        tpf = f"{td}/trial.csv"
        tp.to_csv(tpf, index=False)
        env["T2D_TRIAL_PARAMS"] = tpf
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
                "feasible",
                "--out",
                o,
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            print(
                f"draw {a.draw} cov {a.cov} failed: {p.stderr[-300:]}", file=sys.stderr
            )
            return
        t = pd.read_csv(o)
        t = t[t.outcome.isin(["IHD", "IschStroke"])]
        yr = [c for c in t.columns if c.startswith("yr")]
        g = (
            t.groupby(["iso3", "arm"])[yr]
            .sum()
            .sum(axis=1)
            .rename("deaths")
            .reset_index()
        )
        g["coverage_used"] = (
            t.groupby("iso3").coverage_used.first().reindex(g.iso3).values
        )
        for suf, col, name in [
            ("_t2d.csv", "cases_averted_total", "t2d"),
            ("_ckd.csv", "deaths_averted_total", "ckd"),
        ]:
            f = o[:-4] + suf
            if os.path.exists(f):
                x = pd.read_csv(f).set_index(["iso3", "arm"])[col]
                g[name] = [x.get((i, ar), np.nan) for i, ar in zip(g.iso3, g.arm)]
    g.insert(0, "cov", a.cov)
    g.insert(0, "draw", a.draw)
    g.to_csv(out, index=False)


def aggregate(a):
    import glob

    fs = glob.glob(f"{D}/out/d*_c*.csv")
    x = pd.concat(pd.read_csv(f) for f in fs)
    print(f"{len(fs)} job outputs, {x.draw.nunique()} draws")
    q = lambda s: pd.Series(
        {"median": s.median(), "lo": s.quantile(0.025), "hi": s.quantile(0.975)}
    )
    num = x[x["cov"] != "afford"].copy()
    num["c"] = num["cov"].astype(float)
    diet = x[x.arm == "diet"].groupby(["draw", "iso3"]).deaths.first()
    trt = num[num.arm == "glp1"].pivot_table(
        index=["draw", "iso3"], columns="c", values="deaths"
    )
    cross = []
    for (d, iso), y in trt.iterrows():
        cv = y.index.values
        v = y.values
        dd = diet.get((d, iso), np.nan)
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
        cross.append(
            dict(
                draw=d,
                iso3=iso,
                crossover=c_,
                r10=dd / np.interp(0.1, cv, v),
                r20=dd / np.interp(0.2, cv, v),
                r50=dd / np.interp(0.5, cv, v),
                r100=r100,
                xext=c_ if np.isfinite(c_) else r100,
            )
        )
    cr = pd.DataFrame(cross)
    af = x[(x["cov"] == "afford")].pivot_table(
        index=["draw", "iso3"], columns="arm", values="deaths"
    )
    cr = cr.merge(
        (af.diet / af.glp1).rename("raff").reset_index(),
        on=["draw", "iso3"],
        how="left",
    )
    cr["crossover"] = cr.crossover.fillna(np.inf)
    s = cr.groupby("iso3")[[c for c in cr.columns if c not in ("iso3",)]].apply(
        lambda g: pd.Series(
            {
                "crossover_median_%": 100
                * g.crossover.quantile(0.5, interpolation="lower"),
                "crossover_lo_%": 100
                * g.crossover.quantile(0.025, interpolation="lower"),
                "crossover_hi_%": 100
                * g.crossover.quantile(0.975, interpolation="higher"),
                "share_not_reached": np.isinf(g.crossover).mean(),
                "P(crossover<10%)": (g.crossover < 0.1).mean(),
                "P(crossover<20%)": (g.crossover < 0.2).mean(),
                "xext_median_%": 100 * g.xext.median(),
                "xext_lo_%": 100 * g.xext.quantile(0.025),
                "xext_hi_%": 100 * g.xext.quantile(0.975),
                "treat100_pct_of_diet_median": 100 / g.r100.median(),
                "treat100_pct_of_diet_lo": 100 / g.r100.quantile(0.975),
                "treat100_pct_of_diet_hi": 100 / g.r100.quantile(0.025),
                **{
                    f"{k}_{st}": f(g[k])
                    for k in ["r10", "r20", "r50", "raff"]
                    for st, f in [
                        ("median", lambda v: v.median()),
                        ("lo", lambda v: v.quantile(0.025)),
                        ("hi", lambda v: v.quantile(0.975)),
                    ]
                },
            }
        )
    )
    s.to_csv(f"{D}/summary_crossover_ratios.csv")
    dt = x.groupby(["iso3", "arm", "cov"]).deaths.apply(q).unstack()
    dt.to_csv(f"{D}/summary_deaths.csv")
    ex = (
        x[x["cov"].isin(["0.100", "0.200", "0.500", "afford"])]
        .melt(
            id_vars=["draw", "iso3", "arm", "cov"],
            value_vars=["t2d", "ckd"],
            var_name="outcome",
        )
        .dropna()
    )
    ex.groupby(["outcome", "iso3", "arm", "cov"]).value.apply(q).unstack().to_csv(
        f"{D}/summary_t2d_ckd.csv"
    )
    fmt = lambda v: ">100" if not np.isfinite(v) else f"{v:.1f}"
    g = s.loc["GLB"]
    print(
        f"Global crossover coverage {fmt(g['crossover_median_%'])}% ({fmt(g['crossover_lo_%'])} to {fmt(g['crossover_hi_%'])}); "
        f"P(<10%) {g['P(crossover<10%)']:.2f}, P(<20%) {g['P(crossover<20%)']:.2f}"
    )
    for k in ["r10", "r20", "r50", "raff"]:
        print(
            f"  ratio {k}: {g[k+'_median']:.2f} ({g[k+'_lo']:.2f} to {g[k+'_hi']:.2f})"
        )
    print(
        "\nCrossover coverage by country (%; >100 = not reached even at 100% coverage):"
    )
    show = s[
        [
            "crossover_median_%",
            "crossover_lo_%",
            "crossover_hi_%",
            "share_not_reached",
            "P(crossover<10%)",
            "P(crossover<20%)",
            "r20_median",
            "r20_lo",
            "r20_hi",
        ]
    ].sort_values(["crossover_median_%", "share_not_reached"])
    for c in ["crossover_median_%", "crossover_lo_%", "crossover_hi_%"]:
        show[c] = show[c].map(fmt)
    print(show.round(2).to_string())
    print(
        "\nCrossover coverage extended beyond 100% (%): above 100 = size of programme, relative to enrolling all eligible"
    )
    print(
        "adults, needed to match dietary policy (lower bound, assuming extra people benefit as much as the average enrolled):"
    )
    print(
        s[
            [
                "xext_median_%",
                "xext_lo_%",
                "xext_hi_%",
                "treat100_pct_of_diet_median",
                "treat100_pct_of_diet_lo",
                "treat100_pct_of_diet_hi",
            ]
        ]
        .sort_values("xext_median_%")
        .round(1)
        .to_string()
    )
    print("\nProbability that dietary policy averts more deaths (ratio > 1):")
    print(
        cr.groupby("iso3")[["r10", "r20", "r50", "raff"]]
        .agg(lambda v: (v > 1).mean())
        .round(2)
        .rename(columns=lambda c: "P_diet_leads_" + c[1:])
        .loc[show.index]
        .to_string()
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    q1 = sp.add_parser("prepare")
    q1.add_argument("--draws", type=int, default=500)
    q2 = sp.add_parser("worker")
    q2.add_argument("--draw", type=int)
    q2.add_argument("--cov")
    sp.add_parser("aggregate")
    a = p.parse_args()
    {"prepare": prepare, "worker": worker, "aggregate": aggregate}[a.cmd](a)
