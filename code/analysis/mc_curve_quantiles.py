import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--mc", default="data/results/v3/mc")
ap.add_argument("--jobs", type=int, default=32)
A = ap.parse_args()


def read(f):
    x = pd.read_csv(f, usecols=["draw", "cov", "iso3", "arm", "deaths"])
    return x[x.arm.isin(["diet", "glp1"])]


fs = [
    f for f in glob.glob(f"{A.mc}/out/d*_c*.csv") if "afford" not in os.path.basename(f)
]
with ProcessPoolExecutor(A.jobs) as ex:
    x = pd.concat(ex.map(read, fs, chunksize=64))
w = x.pivot_table(
    index=["draw", "cov", "iso3"], columns="arm", values="deaths"
).reset_index()
w["ratio"] = w.diet / w.glp1
w["t"] = w.glp1 / w.diet
q = (
    w.groupby(["iso3", "cov"])
    .agg(
        median=("ratio", "median"),
        lo=("ratio", lambda v: v.quantile(0.025)),
        hi=("ratio", lambda v: v.quantile(0.975)),
        t_median=("t", "median"),
        t_lo=("t", lambda v: v.quantile(0.025)),
        t_hi=("t", lambda v: v.quantile(0.975)),
    )
    .reset_index()
)
q.to_csv(f"{A.mc}/curve_ratio_quantiles.csv", index=False)
print(f"{len(fs)} files, {w.draw.nunique()} draws -> {A.mc}/curve_ratio_quantiles.csv")
print(q[q.iso3 == "GLB"].round(3).to_string(index=False))
