import os
import numpy as np, pandas as pd

os.chdir(os.environ.get("CRA_ROOT", os.getcwd()))
cm = pd.read_csv("data/params/country_map.csv")
ISO = dict(zip(cm.gbd_location_name, cm.iso3))
a0 = lambda s: pd.to_numeric(
    pd.Series(s).astype(str).str.extract(r"^(\d+)")[0], errors="coerce"
).values
se = pd.read_csv("data/params/select_elig.csv")
pop = pd.read_csv("data/gbd/1-3_population.csv", low_memory=False)
pop = pop[
    (pop.year == 2023)
    & pop.sex_name.isin(["Male", "Female"])
    & pop.location_name.isin(ISO)
].copy()
pop["a0"] = a0(pop.age_name)
pop = pop[pop.a0 >= 25]
pop.loc[pop.a0 >= 95, "a0"] = 95
pop = (
    pop.assign(iso3=pop.location_name.map(ISO), sex=pop.sex_name)
    .groupby(["iso3", "sex", "a0"])
    .val.sum()
    .rename("pop")
)


def cases(fn):
    d = pd.read_csv(f"data/gbd/cause_totals/{fn}", low_memory=False)
    d = d[
        (d.measure_name == "Prevalence")
        & (d.metric_name == "Number")
        & (d.year == 2023)
        & d.sex_name.isin(["Male", "Female"])
        & d.location_name.isin(ISO)
    ].copy()
    d["a0"] = a0(d.age_name)
    d = d[d.a0 >= 25]
    return (
        d.assign(iso3=d.location_name.map(ISO), sex=d.sex_name)
        .groupby(["iso3", "sex", "a0"])
        .val.sum()
    )


nI, nS = cases("queryB_ihd.csv"), cases("queryB_istroke.csv")
x = (
    se.set_index(["iso3", "sex", "a0"])
    .join(pop, how="left")
    .join(nI.rename("nI"))
    .join(nS.rename("nS"))
    .fillna({"nI": 0, "nS": 0})
)
owc = pd.read_csv("data/params/overweight_cvd.csv").set_index("iso3")
rows, sc = [], []
for iso, g in x.groupby(level=0):
    w = g["pop"].fillna(0)
    p25, p27, p30 = [(g[c] * w).sum() / w.sum() for c in ("p25", "p27", "p30")]
    s27_i = (
        (g.share27_ow_ihd * g.nI).sum() / g.nI.sum() if g.nI.sum() > 0 else p27 / p25
    )
    s27_s = (g.share27_ow_is * g.nS).sum() / g.nS.sum() if g.nS.sum() > 0 else p27 / p25
    rows.append(
        dict(
            iso3=iso,
            overweight_prev=p27,
            ow_cvd_ihd=owc.loc[iso, "ow_cvd_ihd"] * s27_i,
            ow_cvd_is=owc.loc[iso, "ow_cvd_is"] * s27_s,
            p25=p25,
            p30=p30,
        )
    )
    sc.append(dict(iso3=iso, scale=p30 / p27))
r = pd.DataFrame(rows)
r[["iso3", "overweight_prev"]].to_csv(
    "data/params/overweight_prevalence_bmi27.csv", index=False
)
r[["iso3", "ow_cvd_ihd", "ow_cvd_is"]].to_csv(
    "data/params/overweight_cvd_bmi27.csv", index=False
)
pd.DataFrame(sc).to_csv("data/params/elig_scale_bmi30_rel27.csv", index=False)
print(
    "Adults >=25: share with BMI >=25, >=27, >=30, and BMI >=27 among people with IHD and IS (%)"
)
print(
    (
        100
        * r.set_index("iso3")[
            ["p25", "overweight_prev", "p30", "ow_cvd_ihd", "ow_cvd_is"]
        ]
    )
    .round(1)
    .rename(columns={"overweight_prev": "p27"})
    .to_string()
)
print(
    "-> data/params/overweight_prevalence_bmi27.csv, data/params/overweight_cvd_bmi27.csv, data/params/elig_scale_bmi30_rel27.csv"
)
