import glob, pandas as pd, numpy as np

cm = pd.read_csv("data/params/country_map.csv")
ISO = dict(zip(cm.gbd_location_name, cm.iso3))
ISO["Global"] = "GLB"
AGE15 = [f"{a}-{a+4} years" for a in range(25, 95, 5)] + ["95+ years"]

SRC = {
    "IHD": glob.glob("data/gbd/Ischemic heart disease/Dietary risks/*/*.csv"),
    "IS": glob.glob("data/gbd/Ischemic stroke/Dietary risks/*/*.csv"),
    "T2D": glob.glob("data/gbd/Diabetes mellitus type 2/Dietary risks/*/*.csv"),
    "ICH": ["data/gbd/queryA_ICH_all_merged.csv"],
    "SAH": ["data/gbd/queryA_SAH_all_merged.csv"],
    "CKD": ["data/gbd/queryA_ckd_3years.csv"],
}

parts = []
for oc, files in SRC.items():
    for f in files:
        d = pd.read_csv(f, low_memory=False)
        d = d[
            (d.rei_name == "Dietary risks")
            & (d.year == 2023)
            & d.measure_name.str.contains("Death", case=False)
            & d.age_name.isin(AGE15)
            & d.sex_name.isin(["Male", "Female"])
            & d.location_name.isin(ISO)
        ]
        d["outcome"] = oc
        parts.append(d)
d = pd.concat(parts, ignore_index=True)
d["iso3"] = d.location_name.map(ISO)
print("metrics:", sorted(d.metric_name.unique()))

key = ["iso3", "outcome", "sex_name", "age_name"]
num = d[d.metric_name == "Number"].set_index(key)[["val", "lower", "upper"]]
rate = d[d.metric_name == "Rate"].set_index(key)["val"].rename("rate")
t = num.join(rate)
if t.rate.isna().all():
    raise SystemExit(
        "No Rate metric in the downloaded data; cannot infer population. Please provide the format of the population file"
    )
t["pop"] = t.val / t.rate * 1e5
t = t.reset_index()

deaths = t.groupby(["iso3", "outcome"])[["val", "lower", "upper"]].sum()

pop_as = t.groupby(["iso3", "sex_name", "age_name"])["pop"].median().reset_index()
pop_a = pop_as.groupby(["iso3", "age_name"])["pop"].sum()
pop_tot = pop_a.groupby("iso3").sum()

std = pop_a.loc["GLB"]
std_w = std / std.sum()

tot = t.groupby(["iso3", "age_name"])["val"].sum()
crude = (tot.groupby("iso3").sum() / pop_tot * 1e5).rename("crude_rate")
asr = (
    (tot / pop_a * 1e5)
    .unstack("age_name")[std_w.index]
    .mul(std_w)
    .sum(axis=1)
    .rename("asr")
)

out = deaths["val"].unstack("outcome").round(0)
out["total"] = out.sum(axis=1)
out = out.join(crude.round(1)).join(asr.round(1))
out = out[["IHD", "IS", "ICH", "SAH", "T2D", "CKD", "total", "crude_rate", "asr"]]
out.to_csv("data/results/joint_burden_2023.csv")

pd.set_option("display.width", 200)
print(
    "\n===== Joint estimate: diet-attributable deaths (count) and death rate (per 100 000, aged 25+)====="
)
print(out.sort_values("crude_rate").to_string())

glb = deaths.loc["GLB"]
print("\n===== Global (millions), 95% UI =====")
print((glb / 1e6).round(3).to_string())
print(f"Six outcomes combined:{glb['val'].sum()/1e6:.3f} M")

cv = ["IHD", "IS", "ICH", "SAH"]
cv_19 = out.drop("GLB")[cv].sum().sum()
cv_glb = out.loc["GLB", cv].sum()
print(
    f"\n19 countries as share of global diet-attributable cardiovascular deaths:{cv_19/cv_glb*100:.1f}%"
)
print("\n-> data/results/joint_burden_2023.csv")

NAME = {v: k for k, v in ISO.items()}
NAME["GLB"] = "Global"
OUTN = {
    "CKD": "Chronic kidney disease",
    "ICH": "Intracerebral haemorrhage",
    "IHD": "Ischaemic heart disease",
    "IS": "Ischaemic stroke",
    "SAH": "Subarachnoid haemorrhage",
    "T2D": "Type 2 diabetes",
}
oc_age = t.groupby(["iso3", "outcome", "age_name"])["val"].sum()
rows = []
for (iso, oc), r in deaths.iterrows():
    a = oc_age.loc[(iso, oc)]
    rows.append(
        {
            "Country": NAME.get(iso, iso),
            "ISO3": iso,
            "Outcome": OUTN[oc],
            "Attributable deaths": round(r.val),
            "UI lower": round(r.lower),
            "UI upper": round(r.upper),
            "Crude rate per 100,000": round(r.val / pop_tot[iso] * 1e5, 1),
            "Age-standardised rate per 100,000": round(
                (a / pop_a.loc[iso] * 1e5)[std_w.index].mul(std_w).sum(), 1
            ),
        }
    )
st2 = pd.DataFrame(rows)
st2["_g"] = (st2.ISO3 != "GLB").astype(int)
st2 = st2.sort_values(["_g", "Country", "Outcome"]).drop(columns="_g")
st2.to_csv("data/results/supp_table2_joint.csv", index=False)
print("\n-> data/results/supp_table2_joint.csv (Supplementary Table 2 format)")
