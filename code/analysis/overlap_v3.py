# -*- coding: utf-8 -*-
"""Combined dietary policy and GLP-1RA treatment, with overlap of their effects varied from 0% to 100%
(post hoc on the main Monte Carlo draws; replaces the earlier overlap_sensitivity.py and full-redundancy bound).

Overlap
  0%    the combination arm as simulated, both interventions acting independently on each person's risk (the life table
        applies them multiplicatively and accounts for competing risks, so deaths averted are never double counted)
  theta combination(theta) = combination - theta x (combination - larger single arm)
  100%  the combination averts only as many deaths as the more effective intervention alone
Reported by country, coverage of eligible adults (10%, 20%, maximum affordable) and overlap (0, 25, 50, 75, 100%):
  combined deaths averted; additional deaths averted by adding dietary policy to treatment (% of treatment alone);
  additional deaths averted by adding treatment to dietary policy (% of dietary policy alone). Medians and 95% UIs.

Inputs  data/results/v3/mc/out/d<draw>_c{0.100,0.200,afford}.csv (columns draw, cov, iso3, arm, deaths, coverage_used)
Output  data/results/v3/overlap/overlap_by_country.csv (long), overlap_summary.csv (wide, overlap 0/50/100%)
Usage   python code/analysis/overlap_v3.py [--mc data/results/v3/mc] [--out data/results/v3/overlap] [--jobs 32]
"""
import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(); ap.add_argument("--mc", default="data/results/v3/mc"); ap.add_argument("--out", default="data/results/v3/overlap")
ap.add_argument("--jobs", type=int, default=32); A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
COVS = {"0.100": "10%", "0.200": "20%", "afford": "Maximum affordable"}
THETA = [0, 25, 50, 75, 100]

def read(f): return pd.read_csv(f)
fs = [f for c in COVS for f in glob.glob(f"{A.mc}/out/d*_c{c}.csv")]
with ProcessPoolExecutor(A.jobs) as ex:
    x = pd.concat(ex.map(read, fs, chunksize=32))
x["cov"] = x["cov"].astype(str).map(lambda c: "afford" if c == "afford" else f"{float(c):.3f}")
w = x.pivot_table(index=["cov", "iso3", "draw"], columns="arm", values="deaths").reset_index()
cu = x[x.arm == "glp1"].groupby(["cov", "iso3"]).coverage_used.median()
q = lambda v, p: float(np.nanquantile(v, p))
rows = []
for (cov, iso), g in w.groupby(["cov", "iso3"]):
    d, t, c = g.diet.values, g.glp1.values, g.combo.values
    big = np.maximum(d, t)
    for th in THETA:
        cc = np.maximum(c - th / 100 * (c - big), big)
        add_d = 100 * (cc - t) / t; add_t = 100 * (cc - d) / d
        rows.append(dict(coverage=COVS[cov], iso3=iso, coverage_used_pct=100 * float(cu.get((cov, iso), np.nan)), overlap_pct=th,
                         combined_median=np.median(cc), combined_lo=q(cc, .025), combined_hi=q(cc, .975),
                         add_diet_to_trt_pct_median=np.median(add_d), add_diet_to_trt_pct_lo=q(add_d, .025), add_diet_to_trt_pct_hi=q(add_d, .975),
                         add_trt_to_diet_pct_median=np.median(add_t), add_trt_to_diet_pct_lo=q(add_t, .025), add_trt_to_diet_pct_hi=q(add_t, .975),
                         diet_median=np.median(d), trt_median=np.median(t), draws=len(g)))
r = pd.DataFrame(rows); r.to_csv(f"{A.out}/overlap_by_country.csv", index=False)
s = r[r.overlap_pct.isin([0, 50, 100])].pivot_table(index=["iso3", "coverage"], columns="overlap_pct",
                                                     values=["add_diet_to_trt_pct_median", "add_trt_to_diet_pct_median"])
s.columns = [f"{a.replace('_pct_median', '')}_ov{b}" for a, b in s.columns]; s = s.reset_index(); s.to_csv(f"{A.out}/overlap_summary.csv", index=False)
pd.set_option("display.width", 220)
print(f"{len(fs)} files, {w.draw.nunique()} draws")
print("\nAdditional deaths averted by adding dietary policy to treatment (% of treatment alone), median, by overlap")
print(s.pivot(index="iso3", columns="coverage", values=["add_diet_to_trt_ov0", "add_diet_to_trt_ov100"]).round(0).to_string())
print(f"-> {A.out}/overlap_by_country.csv, overlap_summary.csv")
