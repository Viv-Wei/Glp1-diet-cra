import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/results/v3/mc_t2dckd")
ap.add_argument("--jobs", type=int, default=32)
A = ap.parse_args()


def read(f):
    return pd.read_csv(
        f, usecols=lambda c: c in ("draw", "cov", "iso3", "arm", "t2d", "ckd")
    )


fs = glob.glob(f"{A.dir}/out/d*_c*.csv")
with ProcessPoolExecutor(A.jobs) as ex:
    x = pd.concat(ex.map(read, fs, chunksize=64))
x["cov"] = x["cov"].astype(str)
q = lambda v, p: float(np.nanquantile(v, p))
curves, summ = [], []
for oc, lab in (("t2d", "T2D"), ("ckd", "CKD")):
    w = x.pivot_table(
        index=["iso3", "draw", "cov"], columns="arm", values=oc
    ).reset_index()
    af = w[w["cov"] == "afford"].set_index(["iso3", "draw"])
    w = w[w["cov"] != "afford"].copy()
    w["c"] = w["cov"].astype(float)
    w["ratio"] = w.diet / w.glp1
    for (iso, c), g in w.groupby(["iso3", "c"]):
        curves.append(
            dict(
                outcome=lab,
                iso3=iso,
                cov=c,
                diet_median=g.diet.median(),
                trt_median=g.glp1.median(),
                ratio_median=g.ratio.median(),
                ratio_lo=q(g.ratio, 0.025),
                ratio_hi=q(g.ratio, 0.975),
            )
        )
    for iso, g in w.groupby("iso3"):
        xo, r10, r20, raf = [], [], [], []
        for dr, h in g.groupby("draw"):
            h = h.sort_values("c")
            cv = np.r_[0, h.c.values]
            v = np.r_[0, h.glp1.values]
            dd = h.diet.iloc[0]
            i = np.where(v >= dd)[0]
            if len(i):
                i = i[0]
                xo.append(
                    cv[i - 1]
                    + (dd - v[i - 1]) / (v[i] - v[i - 1]) * (cv[i] - cv[i - 1])
                )
            else:
                xo.append(dd / v[-1] if v[-1] > 0 else np.nan)
            r10.append(dd / np.interp(0.1, cv, v))
            r20.append(dd / np.interp(0.2, cv, v))
            if (iso, dr) in af.index:
                raf.append(af.loc[(iso, dr), "diet"] / af.loc[(iso, dr), "glp1"])
        xo = np.array(xo)
        summ.append(
            dict(
                outcome=lab,
                iso3=iso,
                draws=len(xo),
                xext_median_pct=100 * np.nanmedian(xo),
                xext_lo_pct=100 * q(xo, 0.025),
                xext_hi_pct=100 * q(xo, 0.975),
                share_not_reached=float(np.mean(xo > 1)),
                P_lt10=float(np.mean(xo < 0.1)),
                P_lt20=float(np.mean(xo < 0.2)),
                r10_median=np.median(r10),
                r20_median=np.median(r20),
                r20_lo=q(r20, 0.025),
                r20_hi=q(r20, 0.975),
                raff_median=np.median(raf) if raf else np.nan,
            )
        )
pd.DataFrame(curves).to_csv(f"{A.dir}/outcome_curves.csv", index=False)
s = pd.DataFrame(summ)
s.to_csv(f"{A.dir}/summary_outcome_crossover.csv", index=False)
pd.set_option("display.width", 200)
print(f"{len(fs)} files")
print(
    s.pivot(index="iso3", columns="outcome", values=["xext_median_pct", "r20_median"])
    .round(2)
    .to_string()
)
