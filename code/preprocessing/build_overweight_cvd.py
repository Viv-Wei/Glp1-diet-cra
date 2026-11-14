import glob, os, re
import numpy as np, pandas as pd

os.chdir(os.environ.get("CRA_ROOT", os.getcwd()))
RR_POOLED = {"IHD": (1.32, 1.81), "IS": (1.22, 1.64)}
RR_EAST_ASIA = {"IHD": (1.29, 2.16), "IS": (1.26, 2.03)}
RR_SOUTH_ASIA = {"IHD": (1.17, 1.33), "IS": (1.00, 1.00)}
EAST_ASIA, SOUTH_ASIA = {"CHN", "JPN"}, {"IND", "BGD"}


def rr_main(iso):
    return (
        RR_EAST_ASIA
        if iso in EAST_ASIA
        else RR_SOUTH_ASIA if iso in SOUTH_ASIA else RR_POOLED
    )


RR_SETS = {
    "main": rr_main,
    "pooled": lambda iso: RR_POOLED,
    "null": lambda iso: {"IHD": (1.0, 1.0), "IS": (1.0, 1.0)},
}
cm = pd.read_csv("data/params/country_map.csv")
ISO = dict(zip(cm.gbd_location_name, cm.iso3))
a0 = lambda s: pd.to_numeric(
    pd.Series(s).astype(str).str.extract(r"^(\d+)")[0], errors="coerce"
).values

import unicodedata

norm = lambda x: re.sub(
    r"[^a-z]",
    "",
    unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower(),
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


def bmi(pattern):
    d = pd.read_csv(glob.glob(f"data/gbd/bmi_exposure/{pattern}")[0], low_memory=False)
    d = d[(d.year_id == 2023) & d.sex.isin(["Male", "Female"])].copy()
    d["a0"] = a0(d.age_group_name)
    d = d[d.a0 >= 25]
    ctry = (
        d[d.location_name.isin(ISO)]
        .assign(iso3=lambda x: x.location_name.map(ISO))
        .groupby(["iso3", "sex", "a0"])["mean"]
        .mean()
    )
    d["key"] = d.location_name.map(norm)
    keep = d.groupby("key").location_id.min()
    nat = d[
        d.key.isin(set(WPOP.index.get_level_values(0)))
        & (d.location_id == d.key.map(keep))
    ]
    nat = nat.groupby(["key", "sex", "a0"])["mean"].mean()
    w = WPOP.reindex(nat.index)
    g = (nat * w).groupby(level=["sex", "a0"]).sum() / w.groupby(
        level=["sex", "a0"]
    ).sum()
    cov = w.sum() / WPOP.sum()
    g.index = pd.MultiIndex.from_tuples(
        [("GLB", sx, ag) for sx, ag in g.index], names=["iso3", "sex", "a0"]
    )
    print(
        f"{pattern}: global aggregate from {nat.index.get_level_values(0).nunique()} countries, "
        f"{100 * cov:.1f}% of the WPP adult population"
    )
    return pd.concat([ctry, g])


p_owob = bmi("*PREVALENCE_OF_OBESITY_AND_OVERWEIGHT*.CSV")
p_ob = bmi("*PREVALENCE_OF_OBESITY_Y*.CSV")
p_ob = p_ob.clip(upper=p_owob.reindex(p_ob.index))
_wg = W.groupby(["sex", "a0"])["pop"].sum()
_pg = p_owob.loc["GLB"]
print(
    f"Global overweight prevalence, adults >=25: {100 * (_pg * _wg.reindex(_pg.index)).sum() / _wg.reindex(_pg.index).sum():.1f}% "
    f"(data/params/overweight_prevalence.csv: {100 * pd.read_csv('data/params/overweight_prevalence.csv').set_index('iso3').overweight_prev.get('GLB', float('nan')):.1f}%)"
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
    d["iso3"] = d.location_name.map(ISO)
    d["sex"] = d.sex_name
    return d.groupby(["iso3", "sex", "a0"]).val.sum()


N = {"IHD": cases("queryB_ihd.csv"), "IS": cases("queryB_istroke.csv")}


def derive(iso, rr):
    out = {}
    pb, po = p_owob.loc[iso], p_ob.loc[iso]
    idx = pb.index.intersection(po.index)
    pb, po = pb.loc[idx], po.loc[idx]
    p_ow, p_norm = (pb - po).clip(lower=0), (1 - pb).clip(lower=0)
    for o in ("IHD", "IS"):
        r_ow, r_ob = rr[o]
        q = (p_ow * r_ow + po * r_ob) / (p_norm + p_ow * r_ow + po * r_ob)
        w = (
            N[o].loc[iso].reindex(idx).fillna(0)
            if iso in N[o].index.get_level_values(0)
            else pd.Series(1.0, index=idx)
        )
        out[o] = float((q * w).sum() / w.sum())
        out["n_" + o] = float(w.sum())
        if o == "IHD":
            out["pop"] = float((pb * w).sum() / w.sum())
    out["CVD"] = (out["IHD"] * out["n_IHD"] + out["IS"] * out["n_IS"]) / (
        out["n_IHD"] + out["n_IS"]
    )
    return out


rows, sens = [], []
for iso in sorted(set(ISO.values())):
    m = derive(iso, rr_main(iso))
    rows.append(
        {
            "iso3": iso,
            "ow_pop": m["pop"],
            "ow_cvd_ihd": m["IHD"],
            "ow_cvd_is": m["IS"],
            "ow_cvd": m["CVD"],
            "rr_source": (
                "Asia Cohort Consortium, east Asia"
                if iso in EAST_ASIA
                else (
                    "Asia Cohort Consortium, south Asia"
                    if iso in SOUTH_ASIA
                    else "Bogers 2007; Strazzullo 2010"
                )
            ),
        }
    )
    sens.append(
        {
            "iso3": iso,
            "ow_pop": m["pop"],
            **{f"ow_cvd_{k}": derive(iso, f(iso))["CVD"] for k, f in RR_SETS.items()},
        }
    )
res = pd.DataFrame(rows)
res.to_csv("data/params/overweight_cvd.csv", index=False)
sv = pd.DataFrame(sens)
sv.to_csv("data/params/overweight_cvd_sensitivity.csv", index=False)
print(
    "Share with overweight (BMI >=25, %) among prevalent IHD and ischaemic stroke cases, by source of relative risks"
)
print((100 * sv.set_index("iso3")).round(1).sort_values("ow_cvd_main").to_string())
print(
    "\nSources:", res.groupby("rr_source").iso3.apply(lambda x: ", ".join(x)).to_dict()
)
