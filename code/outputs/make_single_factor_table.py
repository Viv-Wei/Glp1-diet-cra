import pandas as pd

NAMES = dict(pd.read_csv("data/tables/table_single_factor_v3.csv")[["GDD variable code", "Dietary factor"]].drop_duplicates().values)
f = pd.read_csv("data/results/v3/mc_scen/single_factor_mc.csv")
x = pd.DataFrame({"Country": f.iso3, "GDD variable code": f.factor, "Dietary factor": f.factor.map(NAMES),
                  "Total CVD deaths averted (30-year)": f.deaths_median.round(), "Share of country CVD benefit (%)": f.share_median.round(1)})
x.dropna(subset=["Dietary factor"]).to_csv("data/tables/table_single_factor_mc.csv", index=False)
print(f"-> data/tables/table_single_factor_mc.csv ({len(x)} rows)")
