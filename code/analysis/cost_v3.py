import argparse, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, "code")
import lifetable_v3 as L

p = argparse.ArgumentParser()
p.add_argument("--run", default="data/results/v3/equity_cost/main_0.20")
p.add_argument("--outdir", default="data/results/v3/equity_cost")
p.add_argument("--persist", type=float, default=0.50)
p.add_argument("--diet-cost", type=float, default=0.55)
a = p.parse_args()

pop = L.load_pop()
adults0 = pop.groupby(level=0).sum()
prog = pd.read_csv(a.run + "_check_programme.csv")
prog = prog[prog.arm == "glp1"]
alloc = pd.read_csv(a.run + "_check_allocation.csv").set_index("iso3")
m = pd.read_csv(a.run + ".csv")
m = m[m.outcome.isin(["IHD", "IschStroke"])]
yr = [c for c in m.columns if c.startswith("yr")]
D = m.groupby(["iso3", "arm"])[yr].sum().sum(axis=1).unstack("arm")
ck = (
    pd.read_csv(a.run + "_ckd.csv").pivot(
        index="iso3", columns="arm", values="deaths_averted_total"
    )
    if os.path.exists(a.run + "_ckd.csv")
    else None
)
brand = dict(
    pd.read_csv("data/params/glp1_prices.csv")[["iso3", "brand_annual_usd"]].values
)

rows = []
for iso, g in prog.groupby("iso3"):
    g = g.sort_values("year")
    T0 = float(alloc.loc[iso, "treated"])
    enrolled = np.concatenate([[T0], g.enrolled_after.values[:-1]])
    new = np.concatenate([[T0], g.new_enrolment.values[:-1]])
    if "on_share" in g and g.on_share.notna().all():
        payers = enrolled * g.on_share.values
    else:
        payers = new + a.persist * (enrolled - new)
    alive_start = np.concatenate(
        [[float(adults0.get(iso, np.nan))], g.alive.values[:-1]]
    )
    py_cohort = float(((alive_start + g.alive.values) / 2).sum())
    diet_cost = a.diet_cost * py_cohort
    for label, price in [
        ("Branded", brand.get(iso)),
        ("Generic $140/yr", 140.0),
        ("Generic $28/yr", 28.0),
    ]:
        if price is None or pd.isna(price):
            continue
        trt_cost = price * float(payers.sum())
        for scope, dg, dd in [("IHD + IS", D.loc[iso, "glp1"], D.loc[iso, "diet"])] + (
            (
                [
                    (
                        "IHD + IS + CKD",
                        D.loc[iso, "glp1"] + ck.loc[iso, "glp1"],
                        D.loc[iso, "diet"] + ck.loc[iso, "diet"],
                    )
                ]
            )
            if ck is not None
            else []
        ):
            rows.append(
                {
                    "ISO3": iso,
                    "Price scenario": label,
                    "Outcomes": scope,
                    "Enrolled person-years (millions)": round(
                        float(enrolled.sum()) / 1e6, 2
                    ),
                    "Paying person-years (millions)": round(
                        float(payers.sum()) / 1e6, 2
                    ),
                    "GLP-1RA expenditure per death averted (USD)": (
                        round(trt_cost / dg) if dg > 0 else None
                    ),
                    "Dietary policy expenditure per death averted (USD)": (
                        round(diet_cost / dd) if dd > 0 else None
                    ),
                    "Ratio, treatment to diet": (
                        round((trt_cost / dg) / (diet_cost / dd), 1)
                        if dg > 0 and dd > 0
                        else None
                    ),
                }
            )
out = pd.DataFrame(rows)
os.makedirs(a.outdir, exist_ok=True)
out.to_csv(os.path.join(a.outdir, "tableS_cost_v3.csv"), index=False)
x = out[out.Outcomes == "IHD + IS"]
print(x[x.ISO3 == "GLB"].drop(columns="Outcomes").to_string(index=False))
for lab in ["Generic $140/yr", "Generic $28/yr"]:
    y = x[x["Price scenario"] == lab]
    print(
        f"\n{lab}: treatment cheaper per death averted than diet in "
        f"{', '.join(y[y['Ratio, treatment to diet'] < 1].ISO3) or 'no country'}; global ratio "
        f"{float(y[y.ISO3 == 'GLB']['Ratio, treatment to diet'].iloc[0]):.1f}"
    )
