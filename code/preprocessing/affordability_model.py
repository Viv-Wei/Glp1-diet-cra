import pandas as pd

inc = pd.read_csv("data/params/income_quintiles_wb.csv")
prc = pd.read_csv("data/params/glp1_prices.csv")

GENERIC_HI, GENERIC_LO = 140.0, 28.0

m = inc.merge(prc, on="iso3")
m["afford_brand_pct"] = (m["brand_annual_usd"] / m["income_pc_usd"] * 100).round(1)
m["afford_generic_hi_pct"] = (GENERIC_HI / m["income_pc_usd"] * 100).round(2)
m["afford_generic_lo_pct"] = (GENERIC_LO / m["income_pc_usd"] * 100).round(2)

cols = [
    "iso3",
    "quintile",
    "income_pc_usd",
    "brand_annual_usd",
    "verified",
    "afford_brand_pct",
    "afford_generic_hi_pct",
    "afford_generic_lo_pct",
]
out = m[cols].sort_values(["iso3", "quintile"])
out.to_csv("data/results/affordability_matrix.csv", index=False)

piv = out.pivot(index="iso3", columns="quintile", values="afford_brand_pct")
print("Branded price scenario: annual drug cost as % of quintile per-capita income %\n")
print(piv.to_string())
print("\nGeneric upper bound ($140) scenario, quintile Q1:\n")
print(
    out[out.quintile == "Q1"][["iso3", "afford_generic_hi_pct"]].to_string(index=False)
)
print(
    "\nWHO affordability reference: >20% is usually considered catastrophic expenditure (for caption annotation)"
)
flag = out[(out.afford_brand_pct > 100)]
print(f"\nNumber of quintiles where drug cost exceeds annual income: {len(flag)}")
print(flag[["iso3", "quintile", "afford_brand_pct"]].to_string(index=False))
