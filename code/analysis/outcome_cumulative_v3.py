import argparse, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="data/results/v3/mc_traj_t2dckd")
ap.add_argument("--jobs", type=int, default=32)
A = ap.parse_args()
RUN = {
    "c0.10": "10%",
    "c0.20": "20%",
    "cafford": "Maximum affordable",
    "asp": "Aspirational 20%",
}


def read(f):
    x = pd.read_csv(f)
    yr = [c for c in x.columns if c.startswith("yr")]
    x["total"] = x[yr].sum(axis=1)
    if "outcome" not in x.columns:
        x["outcome"] = "IHD+IS"
    return x[["draw", "run", "outcome", "iso3", "arm", "total"]]


with ProcessPoolExecutor(A.jobs) as ex:
    x = pd.concat(ex.map(read, glob.glob(f"{A.dir}/out/d*_*.csv"), chunksize=32))
x["coverage"] = x.run.map(RUN)
q = lambda v, p: float(np.nanquantile(v, p))
cum = (
    x.groupby(["outcome", "iso3", "coverage", "arm"])
    .total.agg(
        median="median", lo=lambda v: v.quantile(0.025), hi=lambda v: v.quantile(0.975)
    )
    .reset_index()
)
cum.loc[cum.coverage == "Aspirational 20%", "arm"] = "asp"
cum.to_csv(f"{A.dir}/cumulative_by_outcome.csv", index=False)
rows = []
w = (
    x[x.run != "asp"]
    .pivot_table(
        index=["outcome", "iso3", "coverage", "draw"], columns="arm", values="total"
    )
    .reset_index()
)
for (oc, iso, cov), g in w.groupby(["outcome", "iso3", "coverage"]):
    d, t, c = g.diet.values, g.glp1.values, g.combo.values
    big = np.maximum(d, t)
    for th in (0, 25, 50, 75, 100):
        cc = np.maximum(c - th / 100 * (c - big), big)
        add = np.where(t > 0, 100 * (cc - t) / np.where(t > 0, t, 1), np.nan)
        rows.append(
            dict(
                outcome=oc,
                iso3=iso,
                coverage=cov,
                overlap_pct=th,
                combined_median=np.median(cc),
                combined_lo=q(cc, 0.025),
                combined_hi=q(cc, 0.975),
                add_diet_to_trt_pct_median=np.nanmedian(add),
                diet_median=np.median(d),
                trt_median=np.median(t),
                draws=len(g),
            )
        )
pd.DataFrame(rows).to_csv(f"{A.dir}/overlap_by_outcome.csv", index=False)
print(f"{x.draw.nunique()} draws, outcomes {sorted(x.outcome.unique())}")
s = (
    cum[
        (cum.iso3 == "GLB") & (cum.arm.isin(["diet", "glp1", "combo", "asp"]))
    ].pivot_table(index=["outcome", "coverage"], columns="arm", values="median")
    / 1e3
)
print("Global 30-year totals (thousands), medians:")
print(s.round(1).to_string())
print(f"-> {A.dir}/cumulative_by_outcome.csv, overlap_by_outcome.csv")
