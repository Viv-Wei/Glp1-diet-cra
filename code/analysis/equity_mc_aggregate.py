# -*- coding: utf-8 -*-
"""Deaths averted by education group with 95% uncertainty intervals (Supplementary Tables 18 and 19, Figure 6d and
Supplementary Figure 9): summarises equity_v3.py applied to each of the first 300 Monte Carlo draws at 20% coverage
(data/results/v3/equity_mc/alloc/d<draw>/). Education-specific PAFs, affordability and the education-income link are at
central values, so the intervals reflect uncertainty in the deaths averted by each intervention. Medians and 2.5th and
97.5th percentiles; ratios and percentages are computed within each draw.
Outputs: data/results/v3/equity_mc/equity_ui.csv, pooled_ui.csv, mapping_ui.csv
Usage (repository root): python code/analysis/equity_mc_aggregate.py"""
import glob, os
import numpy as np, pandas as pd
B = "data/results/v3/equity_mc"
q = lambda v: pd.Series({"median": np.nanmedian(v), "lo": np.nanpercentile(v, 2.5), "hi": np.nanpercentile(v, 97.5)})
fs = sorted(glob.glob(f"{B}/alloc/d*/tableS_equity_v3.csv")); print(f"{len(fs)} draws")
e = pd.concat(pd.read_csv(f).assign(draw=int(f.split("/")[-2][1:])) for f in fs)
ms = pd.concat(pd.read_csv(f.replace("tableS_equity_v3", "tableS_equity_mapping_sensitivity_v3")).assign(draw=int(f.split("/")[-2][1:])) for f in fs)
k = ["Price scenario", "ISO3", "Education group"]
V = {"diet": "Dietary policy averted (thousands)", "glp1_all": "GLP-1RA averted if affordable to all (thousands)", "glp1": "GLP-1RA averted (thousands)"}
e["ratio"] = np.where(e[V["glp1"]] > 0, e[V["diet"]] / e[V["glp1"]].where(e[V["glp1"]] > 0), np.inf)
const = e.groupby(k)[["Population share (25+)", "PAF ratio", "Affordable fraction"]].first()
parts = [const]
for name, col in list(V.items()) + [("ratio", "ratio")]:
    s = e.groupby(k)[col].apply(q).unstack(); s.columns = [f"{name}_{c}" for c in s.columns]; parts.append(s)
out = pd.concat(parts, axis=1).reset_index(); os.makedirs(B, exist_ok=True); out.to_csv(f"{B}/equity_ui.csv", index=False)
# pooled over the 19 countries, within each draw
p = e.groupby(["draw", "Price scenario", "Education group"])[[V["diet"], V["glp1"], V["glp1_all"]]].sum().reset_index()
p["pct"] = 100 * p[V["glp1"]] / p[V["diet"]]
tot = p.groupby(["draw", "Price scenario"])[[V["diet"], V["glp1_all"]]].transform("sum")
p["share_diet"] = 100 * p[V["diet"]] / tot[V["diet"]]; p["share_glp1_all"] = 100 * p[V["glp1_all"]] / tot[V["glp1_all"]]
pool = []
for (pr, g_), x in p.groupby(["Price scenario", "Education group"]):
    row = {"Price scenario": pr, "Education group": g_}
    for name, col in [("diet", V["diet"]), ("glp1", V["glp1"]), ("pct", "pct"), ("share_diet", "share_diet"), ("share_glp1_all", "share_glp1_all")]:
        for st, v in q(x[col].values).items(): row[f"{name}_{st}"] = v
    pool.append(row)
pool = pd.DataFrame(pool); pool.to_csv(f"{B}/pooled_ui.csv", index=False)
mcol = {"diet": "Dietary policy, low education (thousands)", "glp1": "GLP-1RA, low education (thousands)", "ratio": "Diet-to-treatment ratio"}
mu = []
for lk, x in ms.groupby("Education-income link", sort=False):
    row = {"Education-income link": lk, "Countries with no treatment benefit": x["Countries with no treatment benefit"].mode().iloc[0]}
    for name, col in mcol.items():
        for st, v in q(pd.to_numeric(x[col], errors="coerce").values).items(): row[f"{name}_{st}"] = v
    mu.append(row)
pd.DataFrame(mu).to_csv(f"{B}/mapping_ui.csv", index=False)
pd.set_option("display.width", 200)
g = pool[pool["Price scenario"] == "Generic $140/yr"]
print("\nPooled, generic US$140 (thousands; treatment as % of dietary policy):")
print(g[["Education group", "diet_median", "diet_lo", "diet_hi", "glp1_median", "glp1_lo", "glp1_hi", "pct_median", "pct_lo", "pct_hi"]].round(1).to_string(index=False))
print("\nShare of deaths averted in each education group before affordability (%):")
print(g[["Education group", "share_diet_median", "share_diet_lo", "share_diet_hi", "share_glp1_all_median", "share_glp1_all_lo", "share_glp1_all_hi"]].round(1).to_string(index=False))
print("\n" + pd.DataFrame(mu).round(2).to_string(index=False)); print(f"-> {B}/equity_ui.csv, pooled_ui.csv, mapping_ui.csv")
