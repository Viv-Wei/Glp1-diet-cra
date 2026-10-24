import argparse, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

MC = os.environ.get("MC_DIR", "data/results/v3/mc")
D = os.environ.get("MC_TRAJ_DIR", "data/results/v3/mc_traj")
ALL = os.environ.get("MC_TRAJ_OUTCOMES") == "all"
RUNS = {
    "c0.10": ("0.10", "feasible"),
    "c0.20": ("0.20", "feasible"),
    "cafford": ("afford", "feasible"),
    "asp": ("0.20", "aspirational"),
}


def prepare(a):
    os.makedirs(f"{D}/out", exist_ok=True)
    n = len(pd.read_csv(f"{MC}/draws.csv"))
    with open(f"{D}/jobs.txt", "w") as f:
        for d in range(n):
            for r in RUNS:
                f.write(f"{d} {r}\n")
    print(f"{n} draws x {len(RUNS)} runs -> {D}/jobs.txt")


def worker(a):
    out = f"{D}/out/d{a.draw}_{a.run}.csv"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return
    cov, scen = RUNS[a.run]
    r = pd.read_csv(f"{MC}/draws.csv").set_index("draw").loc[a.draw]
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
        LT_DP_OVERRIDE=f"{MC}/diet/draw{a.draw}.csv",
        SENS_T2D="1" if ALL else "0",
        SENS_CKD="1" if ALL else "0",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    with tempfile.TemporaryDirectory() as td:
        if ALL:
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
                cov,
                "--diet-scenario",
                scen,
                "--out",
                o,
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        if p.returncode != 0:
            print(
                f"draw {a.draw} run {a.run} failed: {p.stderr[-300:]}", file=sys.stderr
            )
            return
        t = pd.read_csv(o)
        t = t[t.outcome.isin(["IHD", "IschStroke"])]
        yr = [c for c in t.columns if c.startswith("yr")]
        g = t.groupby(["iso3", "arm"])[yr].sum().reset_index()
        g.insert(0, "outcome", "IHD+IS")
        if ALL:
            for suf, nm in (("_t2d.csv", "T2D"), ("_ckd.csv", "CKD")):
                f_ = o[:-4] + suf
                if os.path.exists(f_):
                    x = pd.read_csv(f_)
                    x = x[
                        ["iso3", "arm"] + [c for c in x.columns if c.startswith("yr")]
                    ].copy()
                    x.insert(0, "outcome", nm)
                    g = pd.concat([g, x], ignore_index=True)
        if scen == "aspirational":
            g = g[g.arm == "diet"]
    g.insert(0, "run", a.run)
    g.insert(0, "draw", a.draw)
    g.to_csv(out, index=False)


def aggregate(a):
    import glob

    x = pd.concat(pd.read_csv(f) for f in glob.glob(f"{D}/out/d*_*.csv"))
    if "outcome" not in x.columns:
        x.insert(0, "outcome", "IHD+IS")
    yr = [c for c in x.columns if c.startswith("yr")]
    l = x.melt(
        id_vars=["draw", "run", "outcome", "iso3", "arm"],
        value_vars=yr,
        var_name="year",
        value_name="deaths",
    )
    l["year"] = l.year.str[2:].astype(int)
    s_all = (
        l.groupby(["outcome", "iso3", "run", "arm", "year"])
        .deaths.agg(
            median="median",
            lo=lambda v: v.quantile(0.025),
            hi=lambda v: v.quantile(0.975),
        )
        .reset_index()
    )
    s_all.to_csv(f"{D}/trajectories_all_outcomes.csv", index=False)
    s = s_all[s_all.outcome == "IHD+IS"].drop(columns="outcome")
    s.to_csv(f"{D}/trajectories.csv", index=False)
    rows = []
    for (iso, run), g in s[s.run.isin(["c0.10", "c0.20", "cafford"])].groupby(
        ["iso3", "run"]
    ):
        p_ = g.pivot(index="year", columns="arm", values="median")
        yrs = p_.index[p_.diet > p_.glp1]
        rows.append(
            dict(
                iso3=iso,
                run=run,
                first_year_diet_ahead=int(yrs.min()) if len(yrs) else np.nan,
                diet_ahead_in_year30=bool(p_.diet.iloc[-1] > p_.glp1.iloc[-1]),
            )
        )
    y = pd.DataFrame(rows)
    y.to_csv(f"{D}/overtaking_year.csv", index=False)
    print(f"{x.draw.nunique()} draws -> {D}/trajectories.csv, overtaking_year.csv")
    print(
        y.pivot(index="iso3", columns="run", values="first_year_diet_ahead").to_string()
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("prepare")
    q = sp.add_parser("worker")
    q.add_argument("--draw", type=int)
    q.add_argument("--run")
    sp.add_parser("aggregate")
    a = p.parse_args()
    {"prepare": prepare, "worker": worker, "aggregate": aggregate}[a.cmd](a)
