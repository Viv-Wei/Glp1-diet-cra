import argparse, glob, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--age", default="All ages")
ap.add_argument("--ckd-fallback", default="IHD")
A = ap.parse_args()
OUT = {
    "Ischemic heart disease": "IHD",
    "Ischemic stroke": "IschStroke",
    "Diabetes mellitus type 2": "T2D",
    "Chronic kidney disease": "CKD",
}
rows = []
for f in glob.glob("data/gbd/**/*.csv", recursive=True):
    try:
        d = pd.read_csv(f, low_memory=False)
    except Exception:
        continue
    need = {
        "measure_name",
        "location_name",
        "sex_name",
        "age_name",
        "cause_name",
        "rei_name",
        "metric_name",
        "year",
        "val",
        "upper",
        "lower",
    }
    if not need.issubset(d.columns):
        continue
    x = d[
        (d.rei_name == "Dietary risks")
        & (d.metric_name == "Percent")
        & (d.measure_name == "Deaths")
        & (d.location_name == "Global")
        & (d.sex_name == "Both")
        & (d.year == 2023)
        & d.cause_name.isin(OUT)
    ]
    for _, r in x.iterrows():
        rows.append(
            dict(
                outcome=OUT[r.cause_name],
                cause_name=r.cause_name,
                age_name=r.age_name,
                val=r.val,
                lower=r.lower,
                upper=r.upper,
                source=os.path.relpath(f),
            )
        )
u = pd.DataFrame(rows).drop_duplicates(["outcome", "age_name"])
if u.empty:
    raise SystemExit("no diet-attributable percentages found")
print("ages available:", sorted(u.age_name.unique()))
u = (
    u[u.age_name == A.age]
    if (u.age_name == A.age).any()
    else u.groupby("outcome").head(1)
)
u = u[(u.lower > 0) & (u.upper > 0)].copy()
u["sd_log"] = (np.log(u.upper) - np.log(u.lower)) / (2 * 1.96)
if "CKD" not in set(u.outcome):
    fb = u[u.outcome == A.ckd_fallback].iloc[0].copy()
    fb["outcome"] = "CKD"
    fb["source"] = f"fallback: {A.ckd_fallback} spread"
    u = pd.concat([u, fb.to_frame().T], ignore_index=True)
os.makedirs("data/params", exist_ok=True)
u.to_csv("data/params/diet_paf_uncertainty.csv", index=False)
pd.set_option("display.width", 200)
print(
    u[["outcome", "age_name", "val", "lower", "upper", "sd_log", "source"]].to_string(
        index=False
    )
)
print(
    "95% range of the factor, exp(+-1.96 sd_log):",
    {
        o: (round(float(np.exp(-1.96 * s)), 2), round(float(np.exp(1.96 * s)), 2))
        for o, s in zip(u.outcome, u.sd_log)
    },
)
print("-> data/params/diet_paf_uncertainty.csv")
