import glob, os, sys
import pandas as pd

LBL = {
    "v01": "Fruits",
    "v02": "Vegetables",
    "v05": "Legumes",
    "v06": "Nuts & seeds",
    "v08": "Whole grains",
    "v09": "Processed meat",
    "v10": "Red meat",
    "v15": "SSB",
    "v29": "Omega-6",
    "v30": "Seafood omega-3",
    "v34": "Fibre",
    "v37": "Sodium",
}
rows = []
for f in sorted(glob.glob("data/results/v3/single_factor/single_v[0-9][0-9].csv")):
    gv = os.path.basename(f)[7:-4]
    t = pd.read_csv(f)
    t = t[(t.arm == "diet") & t.outcome.isin(["IHD", "IschStroke"])]
    yr = [c for c in t.columns if c.startswith("yr")]
    rows += [
        dict(iso3=i, gdd_var=gv, factor=LBL[gv], deaths_averted=v)
        for i, v in t.groupby("iso3")[yr].sum().sum(axis=1).items()
    ]
r = pd.DataFrame(rows)
r["share_pct"] = (
    100 * r.deaths_averted / r.groupby("iso3").deaths_averted.transform("sum")
)
r = r.sort_values(["iso3", "deaths_averted"], ascending=[True, False])
r["rank"] = r.groupby("iso3").cumcount() + 1
top = r[(r["rank"] <= 3) & (r.deaths_averted > 0)][
    ["iso3", "rank", "gdd_var", "factor", "deaths_averted", "share_pct"]
]
if "--check" in sys.argv:
    old = (
        pd.read_csv("data/params/tailored_top3.csv")
        .sort_values(["iso3", "rank"])
        .groupby("iso3")
        .gdd_var.apply(list)
    )
    new = top.groupby("iso3").gdd_var.apply(list)
    same = sum(old.get(i) == new.get(i) for i in old.index)
    print(f"top three factors identical in {same} of {len(old)} locations")
    sys.exit(0 if same == len(old) else 1)
top.to_csv("data/params/tailored_top3.csv", index=False)
print(top.pivot(index="iso3", columns="rank", values="factor").to_string())
print("-> data/params/tailored_top3.csv")
