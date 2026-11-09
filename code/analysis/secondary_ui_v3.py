# -*- coding: utf-8 -*-
"""Two additions to the supplementary tables, from existing per-draw results (no new model runs).

1. Supplementary Table 8: 95% uncertainty intervals of the combination's multiple of treatment alone for T2D and CKD.
   Within each of the 1000 main Monte Carlo draws (results/v3/mc/out/d<draw>_c<coverage>.csv, columns t2d and ckd for
   the combo, diet and glp1 arms), the combination at overlap w is combo - w x (combo - max(diet, glp1)), and the
   multiple is that divided by treatment alone; medians and 2.5th and 97.5th percentiles are reported. The median
   equals 1 + the added benefit (%) / 100 already in Table 8, which the script checks.
2. Supplementary Table 20: expenditure per death averted with costs and deaths discounted at 3% a year, as in the
   "Discounting 3% a year" scenario (mid-year factors 1.03^-(year - 0.5)). Discounted deaths come from the first 300
   main-analysis draws of the scenario Monte Carlo (results/v3/mc_scen/out/main/), and discounted expenditure from the
   yearly programme records of the deterministic grid, at central values.
Outputs: results/v3/overlap/overlap_ui_t2dckd.csv, results/v3/expenditure_ui/cost_ui_disc3.csv
Usage (repository root): python code/analysis/secondary_ui_v3.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "code")
import lifetable_v3 as L

MC, SC, GRID = "data/results/v3/mc", "data/results/v3/mc_scen/out/main", "data/results/v3/grid"
COV = {"0.100": "10%", "0.200": "20%", "afford": "Maximum affordable"}
q = lambda v: (np.nanmedian(v), np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))

# ---- 1. Table 8 multiples for T2D and CKD
x = pd.concat(pd.read_csv(f, dtype={"cov": str}) for c in COV for f in glob.glob(f"{MC}/out/d*_c{c}.csv"))
print(f"main Monte Carlo: {x.draw.nunique()} draws at {sorted(x['cov'].unique())}")
rows = []
for outcome, col in (("T2D", "t2d"), ("CKD", "ckd")):
    w = x.pivot_table(index=["iso3", "cov", "draw"], columns="arm", values=col)
    for (iso, cov), g in w.groupby(level=[0, 1]):
        c, d, t = g["combo"].values, g["diet"].values, g["glp1"].values
        for ov in (0, 25, 50, 75, 100):
            comb = c - ov / 100 * (c - np.maximum(d, t))
            m = np.where(t > 0, comb / np.where(t > 0, t, 1), np.inf)
            med, lo, hi = q(m)
            rows.append(dict(outcome=outcome, iso3=iso, coverage=COV[cov], overlap_pct=ov, mult_median=med, mult_lo=lo, mult_hi=hi,
                             nonpositive_treatment_draws=int((t <= 0).sum())))
ou = pd.DataFrame(rows); os.makedirs("data/results/v3/overlap", exist_ok=True); ou.to_csv("data/results/v3/overlap/overlap_ui_t2dckd.csv", index=False)
ref = pd.read_csv("data/results/v3/mc_traj_t2dckd/overlap_by_outcome.csv")
chk = ou.merge(ref, on=["outcome", "iso3", "coverage", "overlap_pct"])
d = (chk.mult_median - (1 + chk.add_diet_to_trt_pct_median / 100)).abs()
print(f"Table 8 check: median multiple vs 1 + added benefit in the existing table, largest difference {d.max():.3g} over {len(chk)} rows")
g = ou[(ou.iso3 == "GLB") & (ou.coverage == "20%") & (ou.overlap_pct == 0)]
print(g[["outcome", "mult_median", "mult_lo", "mult_hi"]].round(2).to_string(index=False))

# ---- 2. discounted expenditure per death averted
adults0 = L.load_pop().groupby(level=0).sum()
brand = dict(pd.read_csv("data/params/glp1_prices.csv")[["iso3", "brand_annual_usd"]].values)
def programme(tag):
    prog = pd.read_csv(f"{GRID}/risk_{tag}_check_programme.csv"); prog = prog[prog.arm == "glp1"]
    f = f"{GRID}/risk_{tag}_check_allocation.csv"; alloc = pd.read_csv(f).set_index("iso3") if os.path.exists(f) else None
    out = {}
    for iso, g in prog.groupby("iso3"):
        g = g.sort_values("year"); T0 = float(alloc.loc[iso, "treated"]) if alloc is not None else float(g.enrolled_after.iloc[0])
        enrolled = np.concatenate([[T0], g.enrolled_after.values[:-1]])
        alive0 = np.concatenate([[float(adults0.get(iso, np.nan))], g.alive.values[:-1]])
        v = 1.03 ** -(g.year.values - 0.5)
        out[iso] = dict(paying_disc=float((enrolled * g.on_share.values * v).sum()), cohort_disc=float((((alive0 + g.alive.values) / 2) * v).sum()))
    return out
s = pd.concat(pd.read_csv(f, dtype={"cov": str}) for f in glob.glob(f"{SC}/d*_c*.csv"))
print(f"\nscenario Monte Carlo, main analysis: {s.draw.nunique()} draws at {sorted(s['cov'].unique())}; columns {list(s.columns)}")
assert "deaths_disc3" in s, "no discounted deaths in the scenario Monte Carlo outputs"
rows = []
for cov in [c for c in COV if c in set(s["cov"])]:
    P = programme(cov); P20 = programme("0.200")
    w = s[s["cov"] == cov].pivot_table(index=["iso3", "draw"], columns="arm", values="deaths_disc3")
    for iso, g in w.groupby(level=0):
        dt, dd = g["glp1"].values, g["diet"].values
        diet_cost = 0.55 * P20[iso]["cohort_disc"]
        for label, price in [("Branded", brand.get(iso)), ("Generic $140/yr", 140.0), ("Generic $28/yr", 28.0)]:
            if price is None or pd.isna(price): continue
            tc = price * P[iso]["paying_disc"]
            tpd = np.where(dt > 0, tc / np.where(dt > 0, dt, 1), np.inf); dpd = np.where(dd > 0, diet_cost / np.where(dd > 0, dd, 1), np.inf)
            rows.append(dict(ISO3=iso, price=label, coverage=COV[cov], **dict(zip(["trt_pd", "trt_pd_lo", "trt_pd_hi"], q(tpd))),
                             **dict(zip(["diet_pd", "diet_pd_lo", "diet_pd_hi"], q(dpd))), **dict(zip(["ratio", "ratio_lo", "ratio_hi"], q(tpd / dpd)))))
cd = pd.DataFrame(rows); cd.to_csv("data/results/v3/expenditure_ui/cost_ui_disc3.csv", index=False)
print("\nGlobal expenditure per death averted, costs and deaths discounted at 3% a year (US$):")
print(cd[cd.ISO3 == "GLB"].round(1).to_string(index=False))
u = pd.read_csv("data/results/v3/expenditure_ui/cost_ui.csv"); u = u[(u.ISO3 == "GLB") & (u.outcomes == "IHD + IS")]
print("\nundiscounted, for comparison:"); print(u[["price", "coverage", "trt_pd", "diet_pd", "ratio"]].round(1).to_string(index=False))
print("-> results/v3/overlap/overlap_ui_t2dckd.csv, results/v3/expenditure_ui/cost_ui_disc3.csv")
