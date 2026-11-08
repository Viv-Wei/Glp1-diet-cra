# -*- coding: utf-8 -*-
"""Expenditure per death averted (Supplementary Table 20) and the equal-budget comparison (Supplementary Table 21,
Figure 6c) with 95% uncertainty intervals from the 1000 main Monte Carlo draws.

Deaths averted and person-years both come from each draw (data/results/v3/mc/out/d<draw>_c<coverage>.csv, the same
draws as Supplementary Tables 6 and 7): treatment pays for time on drug only (enrolled at the start of each year times
the share on drug, which varies with the persistence parameters of the draw), dietary policy costs US$0.55 per adult per
year (mid-year adult population), over 30 years without discounting. Prices are fixed. Ratios, shares, coverage bought
and thresholds are computed within each draw and then summarised (median, 2.5th and 97.5th percentiles), so the cost
threshold equals the treatment-to-diet ratio of expenditure per death averted at 20%.
Outputs: data/results/v3/expenditure_ui/cost_ui.csv (Table 20), data/results/v3/expenditure_ui/budget_ui.csv (Table 21)
Usage (repository root): python code/analysis/expenditure_ui_v3.py [--diet-cost 0.55]
"""
import argparse, glob, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "code")
import lifetable_v3 as L

ap = argparse.ArgumentParser(); ap.add_argument("--mc", default="data/results/v3/mc"); ap.add_argument("--grid", default="data/results/v3/grid")
ap.add_argument("--diet-cost", type=float, default=0.55); ap.add_argument("--out", default="data/results/v3/expenditure_ui"); A = ap.parse_args()
PRICES = [("Branded", None), ("Generic $140/yr", 140.0), ("Generic $28/yr", 28.0)]
brand = dict(pd.read_csv("data/params/glp1_prices.csv")[["iso3", "brand_annual_usd"]].values)
adults0 = L.load_pop().groupby(level=0).sum()
q = lambda v: (np.nanmedian(v), np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5))

# ---- deaths averted by draw
fs = glob.glob(f"{A.mc}/out/d*_c*.csv"); x = pd.concat(pd.read_csv(f, dtype={"cov": str}) for f in fs)
x["dck"] = x.deaths + x.ckd
print(f"{len(fs)} per-draw files, {x.draw.nunique()} draws")
assert "paying_py" in x, "per-draw person-years missing: rerun the main Monte Carlo with the current mc_v3.py"
GRIDC = sorted(float(c) for c in x["cov"].unique() if c != "afford")
print(f"coverage levels used for interpolation: {len(GRIDC)} ({min(GRIDC):.3f}-{max(GRIDC):.3f})")
D = {}
for (iso, arm, cov), g in x.groupby(["iso3", "arm", "cov"]):
    D[(iso, arm, cov)] = g.set_index("draw").sort_index()
draws = np.sort(x.draw.unique())
get = lambda iso, arm, cov, col="deaths": D[(iso, arm, cov)][col].reindex(draws).values
covlab = lambda c: f"{c:.3f}"
pay = lambda iso, cov: get(iso, "glp1", cov, "paying_py")                 # person-years on drug, by draw
enr = lambda iso, cov: get(iso, "glp1", cov, "enrolled_py")
cohort = lambda iso: get(iso, "glp1", "0.200", "cohort_py_less_a0") + float(adults0.get(iso, np.nan)) / 2   # adult person-years

cost_rows, bud_rows = [], []
for iso in sorted(x.iso3.unique()):
    diet_cost = A.diet_cost * cohort(iso)                                  # by draw
    d_diet = get(iso, "diet", "0.200"); d_diet_ck = get(iso, "diet", "0.200", "dck")
    T = np.vstack([get(iso, "glp1", covlab(c)) for c in GRIDC])            # coverage x draw
    C = np.vstack([get(iso, "combo", covlab(c)) for c in GRIDC])
    for label, price in PRICES:
        p = brand.get(iso) if price is None else price
        if p is None or pd.isna(p): continue
        # Table 20: expenditure per death averted at 10%, 20% and maximum affordable coverage
        for cov, covname in [("0.100", "10%"), ("0.200", "20%"), ("afford", "Maximum affordable")]:
            if (iso, "glp1", cov) not in D: continue
            tc = p * pay(iso, cov)
            for oname, col, dd in [("IHD + IS", "deaths", d_diet), ("IHD + IS + CKD", "dck", d_diet_ck)]:
                dt = get(iso, "glp1", cov, col)
                tpd = np.where(dt > 0, tc / np.where(dt > 0, dt, 1), np.inf); dpd = np.where(dd > 0, diet_cost / np.where(dd > 0, dd, 1), np.inf)
                cost_rows.append(dict(ISO3=iso, price=label, outcomes=oname, coverage=covname,
                                      enrolled_py_m=np.median(enr(iso, cov)) / 1e6, paying_py_m=np.median(pay(iso, cov)) / 1e6,
                                      **dict(zip(["trt_pd", "trt_pd_lo", "trt_pd_hi"], q(tpd))),
                                      **dict(zip(["diet_pd", "diet_pd_lo", "diet_pd_hi"], q(dpd))),
                                      **dict(zip(["ratio", "ratio_lo", "ratio_hi"], q(tpd / dpd))),
                                      nonpositive_draws=int((dt <= 0).sum() + (dd <= 0).sum())))
        # Table 21: equal-budget comparison
        cost = p * np.vstack([pay(iso, covlab(c)) for c in GRIDC])           # coverage x draw, each draw's cost curve
        budget = p * pay(iso, "0.200"); rem = budget - diet_cost
        cA = np.array([np.interp(diet_cost[k], cost[:, k], GRIDC) if diet_cost[k] <= cost[-1, k] else np.nan for k in range(len(draws))])
        cB = np.array([np.interp(rem[k], cost[:, k], GRIDC) if 0 < rem[k] <= cost[-1, k] else np.nan for k in range(len(draws))])
        at = lambda M, c: np.array([np.interp(c[k], GRIDC, M[:, k]) if np.isfinite(c[k]) else np.nan for k in range(M.shape[1])])
        dA = at(T, cA); t20 = get(iso, "glp1", "0.200"); dB = at(C, cB)
        ok = (t20 > 0) & (d_diet > 0)                                    # as in Table 20: no deaths averted by treatment
        thr = np.where(ok, (budget / np.where(ok, t20, 1)) / (diet_cost / np.where(ok, d_diet, 1)), np.where(t20 <= 0, np.inf, 0.0))
        bud_rows.append(dict(ISO3=iso, price=label, diet_cost_musd=np.median(diet_cost) / 1e6, **dict(zip(["diet", "diet_lo", "diet_hi"], q(d_diet))),
                             A_coverage=np.nanmedian(cA) if np.isfinite(cA).any() else np.nan, **dict(zip(["A_trt", "A_trt_lo", "A_trt_hi"], q(dA))),
                             **dict(zip(["A_ratio", "A_ratio_lo", "A_ratio_hi"], q(d_diet / np.where(dA > 0, dA, np.nan)))),   # diet / treatment, same draw
                             B_budget_musd=np.median(budget) / 1e6, B_diet_share=np.median(diet_cost / budget),
                             **dict(zip(["B_trt", "B_trt_lo", "B_trt_hi"], q(t20))), B_remaining_coverage=np.nanmedian(cB) if np.isfinite(cB).any() else np.nan,
                             **dict(zip(["B_both", "B_both_lo", "B_both_hi"], q(dB))),
                             **dict(zip(["B_gain", "B_gain_lo", "B_gain_hi"], q(np.where(t20 > 0, dB / np.where(t20 > 0, t20, 1), np.inf)))),
                             **dict(zip(["C", "C_lo", "C_hi"], q(thr))), diet_exceeds_budget=bool(np.median(rem) <= 0)))
os.makedirs(A.out, exist_ok=True)
cu = pd.DataFrame(cost_rows); bu = pd.DataFrame(bud_rows)
cu.to_csv(f"{A.out}/cost_ui.csv", index=False); bu.to_csv(f"{A.out}/budget_ui.csv", index=False)
pd.set_option("display.width", 220)
g = cu[(cu.ISO3 == "GLB") & (cu.outcomes == "IHD + IS")]
print("\nGlobal expenditure per death averted (US$, median and 95% UI):")
print(g[["price", "coverage", "trt_pd", "trt_pd_lo", "trt_pd_hi", "diet_pd", "diet_pd_lo", "diet_pd_hi", "ratio", "ratio_lo", "ratio_hi"]].round(1).to_string(index=False))
b = bu[bu.ISO3 == "GLB"]
print("\nGlobal equal-budget comparison (deaths in millions):")
print(pd.DataFrame({"price": b.price, "A coverage %": (100 * b.A_coverage).round(1), "A treatment": (b.A_trt / 1e6).round(2),
                    "A treatment UI": [f"{l/1e6:.2f}-{h/1e6:.2f}" for l, h in zip(b.A_trt_lo, b.A_trt_hi)], "diet": (b.diet / 1e6).round(2),
                    "B both": (b.B_both / 1e6).round(2), "B both UI": [f"{l/1e6:.2f}-{h/1e6:.2f}" for l, h in zip(b.B_both_lo, b.B_both_hi)],
                    "B gain": b.B_gain.round(2), "C": b.C.round(2), "C UI": [f"{l:.2f}-{h:.2f}" for l, h in zip(b.C_lo, b.C_hi)]}).to_string(index=False))
c20 = cu[(cu.outcomes == "IHD + IS") & (cu.coverage == "20%")].set_index(["ISO3", "price"]).ratio
chk = (bu.set_index(["ISO3", "price"]).C - c20.reindex(bu.set_index(["ISO3", "price"]).index)).abs().max()
print(f"\nCheck: cost threshold equals the Table 20 ratio at 20% (largest difference {chk:.2e})")
print(f"-> {A.out}/cost_ui.csv, {A.out}/budget_ui.csv")
