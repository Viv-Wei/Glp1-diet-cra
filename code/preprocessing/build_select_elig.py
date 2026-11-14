import glob, math, os, re, unicodedata
import numpy as np, pandas as pd
from statistics import NormalDist

N01 = NormalDist()
os.chdir(os.environ.get("CRA_ROOT", os.getcwd()))
cm = pd.read_csv("data/params/country_map.csv")
ISO = dict(zip(cm.gbd_location_name, cm.iso3))
a0 = lambda s: pd.to_numeric(
    pd.Series(s).astype(str).str.extract(r"^(\d+)")[0], errors="coerce"
).values
norm = lambda x: re.sub(
    r"[^a-z]",
    "",
    unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower(),
)
RR_POOLED = {"IHD": (1.32, 1.81), "IS": (1.22, 1.64)}
RR_EAST = {"IHD": (1.29, 2.16), "IS": (1.26, 2.03)}
RR_SOUTH = {"IHD": (1.17, 1.33), "IS": (1.00, 1.00)}
rr = lambda iso: (
    RR_EAST
    if iso in {"CHN", "JPN"}
    else RR_SOUTH if iso in {"IND", "BGD"} else RR_POOLED
)

W = pd.read_csv(
    "data/wpp/WPP2024_PopulationByAge5GroupSex_Medium.csv",
    low_memory=False,
    usecols=["Location", "LocTypeName", "Time", "AgeGrp", "PopMale", "PopFemale"],
)
W = W[(W.Time == 2023) & (W.LocTypeName == "Country/Area")].copy()
W["a0"] = a0(W.AgeGrp)
W.loc[W.a0 >= 95, "a0"] = 95
W = W[W.a0 >= 25]
W = W.melt(
    id_vars=["Location", "a0"],
    value_vars=["PopMale", "PopFemale"],
    var_name="sex",
    value_name="pop",
)
W["sex"] = W.sex.map({"PopMale": "Male", "PopFemale": "Female"})
W["key"] = W.Location.map(norm)
WPOP = W.groupby(["key", "sex", "a0"])["pop"].sum()


def with_global(d, val, sexcol, agecol, loccol, idcol=None):

    d = d.copy()
    d["a0"] = a0(d[agecol])
    d = d[d.a0 >= 25]
    d["sex"] = d[sexcol]
    ctry = (
        d[d[loccol].isin(ISO)]
        .assign(iso3=lambda x: x[loccol].map(ISO))
        .groupby(["iso3", "sex", "a0"])[val]
        .mean()
    )
    if "GLB" in set(ctry.index.get_level_values(0)):
        return ctry
    d["key"] = d[loccol].map(norm)
    if idcol:
        d = d[d[idcol] == d.key.map(d.groupby("key")[idcol].min())]
    nat = (
        d[d.key.isin(set(WPOP.index.get_level_values(0)))]
        .groupby(["key", "sex", "a0"])[val]
        .mean()
    )
    w = WPOP.reindex(nat.index)
    g = (nat * w).groupby(level=["sex", "a0"]).sum() / w.groupby(
        level=["sex", "a0"]
    ).sum()
    g.index = pd.MultiIndex.from_tuples(
        [("GLB", s, a) for s, a in g.index], names=["iso3", "sex", "a0"]
    )
    return pd.concat([ctry, g])


def bmi(pattern):
    d = pd.read_csv(glob.glob(f"data/gbd/bmi_exposure/{pattern}")[0], low_memory=False)
    d = d[(d.year_id == 2023) & d.sex.isin(["Male", "Female"])]
    return with_global(
        d, "mean", "sex", "age_group_name", "location_name", "location_id"
    )


p25, p30 = bmi("*PREVALENCE_OF_OBESITY_AND_OVERWEIGHT*.CSV"), bmi(
    "*PREVALENCE_OF_OBESITY_Y*.CSV"
)
mb = bmi("*HIGH_BMI_IN_ADULTS*.CSV")
t = pd.read_csv("data/gbd/cause_totals/queryB_t2d.csv", low_memory=False)
t = t[
    (t.measure_name == "Prevalence")
    & (t.metric_name == "Rate")
    & (t.year == 2023)
    & t.sex_name.isin(["Male", "Female"])
]
if not len(t):
    raise SystemExit("[error] no 2023 Prevalence Rate rows by sex in queryB_t2d.csv")
t = t.assign(val=t.val / 1e5)
pdm = with_global(t, "val", "sex_name", "age_name", "location_name")

rows = []
for iso, sex, ag in p25.index:
    if iso not in set(ISO.values()):
        continue
    a, b = float(p25[(iso, sex, ag)]), float(
        min(p30.get((iso, sex, ag), np.nan), p25[(iso, sex, ag)])
    )
    a, b = min(max(a, 1e-4), 1 - 1e-4), min(max(b, 1e-5), a - 1e-5)
    z25, z30 = N01.inv_cdf(1 - a), N01.inv_cdf(1 - b)
    sd = (math.log(30) - math.log(25)) / (z30 - z25)
    mu = math.log(25) - sd * z25
    p27 = 1 - N01.cdf((math.log(27) - mu) / sd)
    q2527, q2730, q30 = a - p27, p27 - b, b
    out = dict(
        iso3=iso,
        sex=sex,
        a0=int(ag),
        p25=a,
        p27=p27,
        p30=b,
        share27_ow_pop=p27 / a,
        p_dm=float(pdm.get((iso, sex, ag), np.nan)),
        mean_fit=math.exp(mu + sd * sd / 2),
        mean_gbd=float(mb.get((iso, sex, ag), np.nan)),
    )
    for o in ("IHD", "IS"):
        r_ow, r_ob = rr(iso)[o]
        out[f"share27_ow_{o.lower()}"] = (q2730 * r_ow + q30 * r_ob) / (
            q2527 * r_ow + q2730 * r_ow + q30 * r_ob
        )
    rows.append(out)
res = pd.DataFrame(rows).sort_values(["iso3", "sex", "a0"])
res.to_csv("data/params/select_elig.csv", index=False)
res["fit_err"] = res.mean_fit / res.mean_gbd - 1
print(
    "Fit check: mean of fitted log-normal vs GBD mean BMI, relative error by age group (median, max |error|)"
)
print(
    res.groupby("a0")
    .fit_err.agg(["median", lambda x: x.abs().max()])
    .round(3)
    .to_string()
)
PR = 2.5
res["p_dm_cvd"] = res.p_dm * PR / (res.p_dm * PR + 1 - res.p_dm)
print(
    "\nShare with BMI >=27 among adults with BMI >=25 (population, IHD), and diabetes among people with CVD (PR 2.5), ages 45-79, median by country"
)
s = (
    res[(res.a0 >= 45) & (res.a0 < 80)]
    .groupby("iso3")[["share27_ow_pop", "share27_ow_ihd", "p_dm", "p_dm_cvd"]]
    .median()
)
print(s.round(3).to_string())
print(
    "\nPlausibility: US adults aged 35+ with CHD, diagnosed diabetes 30.7% (BRFSS); diabetes is p_dm (GBD includes undiagnosed)"
)
print("-> data/params/select_elig.csv")
