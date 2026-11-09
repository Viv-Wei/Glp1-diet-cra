"""
Summaries from the per-draw outputs of the main and scenario Monte Carlo (1000 draws), for the supplementary tables:
  1. data/results/v3/mc_scen/single_factor_mc.csv   each dietary factor shifted 20% toward its TMREL: IHD and IS deaths
                                               averted over 30 years and the factor's share of the country benefit,
                                               computed within each draw (new Supplementary Table 10, Figure 4e)
  2. data/results/v3/mc_scen/sec_t2d_ckd_mc.csv     T2D cases and CKD deaths averted by each intervention in the secondary
                                               scenarios, and the diet-to-treatment ratio within each draw (Table 22)
  3. data/results/v3/expenditure_ui/cost_ui_disc3.csv  expenditure per death averted with costs and deaths discounted at 3%
                                               a year, person-years from each draw (discounted rows of Table 20)
  4. data/results/v3/overlap/overlap_ui_t2dckd.csv  combined effect multiples for T2D and CKD (Table 8)
Medians and 2.5th and 97.5th percentiles. Usage (project root): nice -n 5 python code/analysis/scen_extra.py
"""
import glob, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "code")
import lifetable_v3 as L

S, MC = "data/results/v3/mc_scen", "data/results/v3/mc"
q = lambda v: (np.nanmedian(v), np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))
read = lambda pat: pd.concat((pd.read_csv(f, dtype={"cov": str}) for f in glob.glob(pat)), ignore_index=True)

# ---- 1. single dietary factors
facs = sorted(os.path.basename(d) for d in glob.glob(f"{S}/out/single_v*"))
assert facs, "no single-factor scenarios found"
dd = {}
for nm in facs:
    x = read(f"{S}/out/{nm}/d*_c*.csv"); x = x[x.arm == "diet"].drop_duplicates(["draw", "iso3"])
    dd[nm] = x.set_index(["draw", "iso3"]).deaths
D = pd.DataFrame(dd); tot = D.sum(axis=1)
rows = []
for nm in facs:
    sh = 100 * D[nm] / tot.where(tot > 0)
    for iso in D.index.get_level_values("iso3").unique():
        a, b = D[nm].xs(iso, level="iso3").values, sh.xs(iso, level="iso3").values
        rows.append(dict(scenario=nm, factor=nm.replace("single_", ""), iso3=iso, draws=len(a),
                         **dict(zip(["deaths_median", "deaths_lo", "deaths_hi"], q(a))),
                         **dict(zip(["share_median", "share_lo", "share_hi"], q(b)))))
sf = pd.DataFrame(rows); sf.to_csv(f"{S}/single_factor_mc.csv", index=False)
print(f"1. single factors: {len(facs)} factors, {sf.draws.min()}-{sf.draws.max()} draws -> {S}/single_factor_mc.csv")
print(sf[sf.iso3 == "GLB"].sort_values("deaths_median", ascending=False)[["factor", "deaths_median", "share_median"]].round(0).to_string(index=False))

# ---- 2. secondary scenarios: T2D and CKD
secs = sorted(os.path.basename(d) for d in glob.glob(f"{S}/out/sec_*"))
rows = []
for nm in secs:
    x = read(f"{S}/out/{nm}/d*_c*.csv")
    for col, out in (("t2d", "T2D"), ("ckd", "CKD")):
        if col not in x: continue
        w = x.pivot_table(index=["iso3", "cov", "draw"], columns="arm", values=col)
        for (iso, cov), g in w.groupby(level=[0, 1]):
            t, d = g["glp1"].values, g["diet"].values; r = np.where(t > 0, d / np.where(t > 0, t, 1), np.nan)
            rows.append(dict(scenario=nm, iso3=iso, cov=cov, outcome=out, draws=len(t), **dict(zip(["trt_median", "trt_lo", "trt_hi"], q(t))),
                             **dict(zip(["diet_median", "diet_lo", "diet_hi"], q(d))), **dict(zip(["ratio_median", "ratio_lo", "ratio_hi"], q(r)))))
se = pd.DataFrame(rows); se.to_csv(f"{S}/sec_t2d_ckd_mc.csv", index=False)
print(f"\n2. secondary scenarios: {len(secs)} ({', '.join(secs)}) -> {S}/sec_t2d_ckd_mc.csv")
print(se[(se.iso3 == "GLB")][["scenario", "cov", "outcome", "trt_median", "diet_median", "ratio_median"]].round(2).to_string(index=False))

# ---- 3. discounted expenditure, person-years from each draw (scenario main)
s = read(f"{S}/out/main/d*_c*.csv"); assert {"deaths_disc3", "paying_py_disc3", "cohort_py_disc3_less_a0"} <= set(s.columns), "per-draw person-years missing"
a0 = L.load_pop().groupby(level=0).sum(); w1 = 1.03 ** -0.5
brand = dict(pd.read_csv("data/params/glp1_prices.csv")[["iso3", "brand_annual_usd"]].values)
COV = {"0.10": "10%", "0.20": "20%", "afford": "Maximum affordable"}
g20 = s[(s["cov"] == "0.20") & (s.arm == "glp1")].set_index(["iso3", "draw"])
rows = []
for cov in [c for c in COV if c in set(s["cov"])]:
    w = s[s["cov"] == cov].pivot_table(index=["iso3", "draw"], columns="arm", values="deaths_disc3")
    pay = s[(s["cov"] == cov) & (s.arm == "glp1")].set_index(["iso3", "draw"]).paying_py_disc3
    for iso, g in w.groupby(level=0):
        dr = g.index.get_level_values("draw"); dt, dd_ = g["glp1"].values, g["diet"].values
        coh = g20.loc[iso].reindex(dr).cohort_py_disc3_less_a0.values + w1 * float(a0.get(iso, np.nan)) / 2
        diet_cost = 0.55 * coh; pp = pay.loc[iso].reindex(dr).values
        for label, price in [("Branded", brand.get(iso)), ("Generic $140/yr", 140.0), ("Generic $28/yr", 28.0)]:
            if price is None or pd.isna(price): continue
            tpd = np.where(dt > 0, price * pp / np.where(dt > 0, dt, 1), np.inf); dpd = np.where(dd_ > 0, diet_cost / np.where(dd_ > 0, dd_, 1), np.inf)
            rows.append(dict(ISO3=iso, price=label, coverage=COV[cov], **dict(zip(["trt_pd", "trt_pd_lo", "trt_pd_hi"], q(tpd))),
                             **dict(zip(["diet_pd", "diet_pd_lo", "diet_pd_hi"], q(dpd))), **dict(zip(["ratio", "ratio_lo", "ratio_hi"], q(tpd / dpd)))))
cd = pd.DataFrame(rows); cd.to_csv("data/results/v3/expenditure_ui/cost_ui_disc3.csv", index=False)
print(f"\n3. discounted expenditure: {s.draw.nunique()} draws -> data/results/v3/expenditure_ui/cost_ui_disc3.csv")
print(cd[cd.ISO3 == "GLB"][["price", "coverage", "trt_pd", "diet_pd", "ratio", "ratio_lo", "ratio_hi"]].round(1).to_string(index=False))

# ---- 4. combined effect multiples for T2D and CKD (main Monte Carlo)
C4 = {"0.100": "10%", "0.200": "20%", "afford": "Maximum affordable"}
x = pd.concat((pd.read_csv(f, dtype={"cov": str}) for c in C4 for f in glob.glob(f"{MC}/out/d*_c{c}.csv")), ignore_index=True)
rows = []
for outcome, col in (("T2D", "t2d"), ("CKD", "ckd")):
    w = x.pivot_table(index=["iso3", "cov", "draw"], columns="arm", values=col)
    for (iso, cov), g in w.groupby(level=[0, 1]):
        c, d, t = g["combo"].values, g["diet"].values, g["glp1"].values
        for ov in (0, 25, 50, 75, 100):
            comb = c - ov / 100 * (c - np.maximum(d, t)); m = np.where(t > 0, comb / np.where(t > 0, t, 1), np.inf)
            rows.append(dict(outcome=outcome, iso3=iso, coverage=C4[cov], overlap_pct=ov, **dict(zip(["mult_median", "mult_lo", "mult_hi"], q(m))),
                             nonpositive_treatment_draws=int((t <= 0).sum())))
ou = pd.DataFrame(rows); os.makedirs("data/results/v3/overlap", exist_ok=True); ou.to_csv("data/results/v3/overlap/overlap_ui_t2dckd.csv", index=False)
print(f"\n4. T2D and CKD combined multiples: {x.draw.nunique()} draws -> data/results/v3/overlap/overlap_ui_t2dckd.csv")
print("done")
