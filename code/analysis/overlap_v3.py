import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--mc", default="data/results/v3/mc")
ap.add_argument("--out", default="data/results/v3/overlap")
ap.add_argument("--jobs", type=int, default=32)
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)
COVS = {"0.100": "10%", "0.200": "20%", "afford": "Maximum affordable"}
THETA = [0, 25, 50, 75, 100]


def read(f):
    return pd.read_csv(f)


fs = [f for c in COVS for f in glob.glob(f"{A.mc}/out/d*_c{c}.csv")]
with ProcessPoolExecutor(A.jobs) as ex:
    x = pd.concat(ex.map(read, fs, chunksize=32))
x["cov"] = (
    x["cov"].astype(str).map(lambda c: "afford" if c == "afford" else f"{float(c):.3f}")
)
w = x.pivot_table(
    index=["cov", "iso3", "draw"], columns="arm", values="deaths"
).reset_index()
cu = x[x.arm == "glp1"].groupby(["cov", "iso3"]).coverage_used.median()
q = lambda v, p: float(np.nanquantile(v, p))
rows = []
for (cov, iso), g in w.groupby(["cov", "iso3"]):
    d, t, c = g.diet.values, g.glp1.values, g.combo.values
    big = np.maximum(d, t)
    for th in THETA:
        cc = np.maximum(c - th / 100 * (c - big), big)
        add_d = 100 * (cc - t) / t
        add_t = 100 * (cc - d) / d
        rows.append(
            dict(
                coverage=COVS[cov],
                iso3=iso,
                coverage_used_pct=100 * float(cu.get((cov, iso), np.nan)),
                overlap_pct=th,
                combined_median=np.median(cc),
                combined_lo=q(cc, 0.025),
                combined_hi=q(cc, 0.975),
                add_diet_to_trt_pct_median=np.median(add_d),
                add_diet_to_trt_pct_lo=q(add_d, 0.025),
                add_diet_to_trt_pct_hi=q(add_d, 0.975),
                add_trt_to_diet_pct_median=np.median(add_t),
                add_trt_to_diet_pct_lo=q(add_t, 0.025),
                add_trt_to_diet_pct_hi=q(add_t, 0.975),
                diet_median=np.median(d),
                trt_median=np.median(t),
                draws=len(g),
            )
        )
r = pd.DataFrame(rows)
r.to_csv(f"{A.out}/overlap_by_country.csv", index=False)
s = r[r.overlap_pct.isin([0, 50, 100])].pivot_table(
    index=["iso3", "coverage"],
    columns="overlap_pct",
    values=["add_diet_to_trt_pct_median", "add_trt_to_diet_pct_median"],
)
s.columns = [f"{a.replace('_pct_median', '')}_ov{b}" for a, b in s.columns]
s = s.reset_index()
s.to_csv(f"{A.out}/overlap_summary.csv", index=False)
pd.set_option("display.width", 220)
print(f"{len(fs)} files, {w.draw.nunique()} draws")
print(
    "\nAdditional deaths averted by adding dietary policy to treatment (% of treatment alone), median, by overlap"
)
print(
    s.pivot(
        index="iso3",
        columns="coverage",
        values=["add_diet_to_trt_ov0", "add_diet_to_trt_ov100"],
    )
    .round(0)
    .to_string()
)
print(f"-> {A.out}/overlap_by_country.csv, overlap_summary.csv")
