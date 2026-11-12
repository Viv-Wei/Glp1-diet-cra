import argparse
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--xlsx", default="data/tables/Supplementary_Tables_v5.xlsx")
A = ap.parse_args()
NAME = {
    "procmeat": "Processed meat (high)",
    "redmeat": "Red meat (high)",
    "sodium": "Sodium (high)",
    "ssb": "SSB (high)",
    "transfat": "Trans fat (high)",
    "fiber": "Fibre (low)",
    "fruits": "Fruits (low)",
    "legumes": "Legumes (low)",
    "nutsseeds": "Nuts & seeds (low)",
    "omega6": "Omega-6 PUFA (low)",
    "seafood_omega3": "Seafood omega-3 (low)",
    "vegetables": "Vegetables (low)",
    "wholegrains": "Whole grains (low)",
}
f = pd.read_csv("data/results/factor_burden_by_age.csv")
f["Dietary factor"] = f.factor.map(NAME)
cv = (
    f[f.outcome.isin(["IHD", "IschStroke", "ICH", "SAH"])]
    .groupby(["iso3", "Dietary factor", "outcome"])
    .deaths_attr.sum()
    .clip(lower=0)
    .groupby(["iso3", "Dietary factor"])
    .sum()
    .round(0)
)
t = (
    pd.read_excel(A.xlsx, "Supplementary Table 5", header=1)
    .iloc[:-1]
    .set_index(["ISO3", "Dietary factor"])["Diet-attributable CV deaths"]
)
d = (t - cv.reindex(t.index)).abs()
print(
    f"Supplementary Table 5: {int(d.notna().sum())} of {len(t)} rows recomputed, largest difference {d.max():.0f} deaths"
)
print("IDENTICAL" if d.notna().all() and d.max() <= 1 else "NOT IDENTICAL")
