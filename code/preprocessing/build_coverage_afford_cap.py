import sys
import pandas as pd

PRICE, THRESHOLD = 140.0, 10.0
m = pd.read_csv("data/results/affordability_matrix.csv")
cap = (m.afford_generic_hi_pct <= THRESHOLD).groupby(m.iso3).mean()
g = pd.read_csv("data/params/income_quintiles_global.csv")
cap["GLB"] = float((100 * PRICE / g.income_pc_usd <= THRESHOLD).mean())
out = cap.rename("afford_cap").rename_axis("iso3").reset_index()
if "--check" in sys.argv:
    old = (
        pd.read_csv("data/params/coverage_afford_cap.csv").set_index("iso3").afford_cap
    )
    c = out.set_index("iso3").afford_cap
    same = (c.reindex(old.index) - old).abs().max() < 1e-12
    print(
        f"19 countries reproduce the existing file: {same}; global (new): {c['GLB']:.1f}; global in the existing file: {'GLB' in old.index}"
    )
else:
    out.to_csv("data/params/coverage_afford_cap.csv", index=False)
    print(out.to_string(index=False))
    print("-> data/params/coverage_afford_cap.csv")
