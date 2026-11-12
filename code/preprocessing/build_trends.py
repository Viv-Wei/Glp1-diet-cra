import os
import numpy as np, pandas as pd

os.chdir(os.environ.get("CRA_ROOT", os.getcwd()))
YEARS, T_END = 30, 2050
NAME = {"Global": "GLB", "Australia": "AUS", "Bangladesh": "BGD", "Brazil": "BRA", "Bulgaria": "BGR", "Canada": "CAN", "China": "CHN",
        "Egypt": "EGY", "Germany": "DEU", "India": "IND", "Israel": "ISR", "Japan": "JPN", "Mexico": "MEX", "New Zealand": "NZL",
        "Nigeria": "NGA", "Romania": "ROU", "South Africa": "ZAF", "Turkey": "TUR", "Türkiye": "TUR", "United Kingdom": "GBR",
        "United States of America": "USA"}
# GBD 2021 Diabetes Collaborators, Lancet 2023, table S23: % change in age-standardised prevalence 2021-2050, both sexes
DM_CHANGE = dict(GLB=59.7, AUS=68.1, BGD=61.2, BRA=64.6, BGR=21.3, CAN=19.3, CHN=80.4, EGY=81.1, DEU=30.4, IND=61.5, ISR=25.3, JPN=13.7,
                 MEX=70.0, NZL=41.3, NGA=62.2, ROU=44.6, ZAF=65.2, TUR=83.6, GBR=35.1, USA=20.5)
cal = lambda t: min(2023 + t, T_END)
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); ex = lambda x: 1 / (1 + np.exp(-x))


BANDS = {25: range(15, 50, 5), 50: range(50, 75, 5), 75: range(75, 105, 5)}   # GBD Foresight bands (15-49, 50-74, 75+)


def composition():
    """Change in each broad band's death rate due to its changing age mix alone, holding the 2023 rate of each five-year
    group fixed: C(t) / C(2023), C(t) = sum_a pop_a(t) r_a / sum_a pop_a(t), with population from UN WPP 2024 (medium
    variant, data/input/forecasts/wpp_pop_small.csv). Rates r_a: GBD 2023 for ages 25+ (IHD, IS, and all causes minus
    both), and for ages 15-24 WPP 2024 all-cause death rates for other causes and zero for IHD and IS.
    Output: dict cause -> DataFrame (iso3, sex, a0, year, comp)."""
    import glob, sys
    sys.path.insert(0, "code"); from lifetable.inputs import load_pop
    pop = pd.read_csv("data/input/forecasts/wpp_pop_small.csv"); dw = pd.read_csv("data/input/forecasts/wpp_deaths2024_small.csv")
    p24 = pop[pop.Time == 2024].set_index(["ISO3_code", "AgeGrpStart"])
    gb = {c: pd.read_pickle(glob.glob(f"data/results/v3/cache/{f}.csv.Deaths.*.pkl")[0])
          for c, f in (("IHD", "queryB_ihd"), ("IS", "queryB_istroke"), ("ALL", "queryD_All_causes_deaths"))}
    lab = lambda a: "95+ years" if a >= 95 else f"{a}-{a + 4} years"
    def rate(cause, iso, sx, a):
        if a >= 25:
            k = (iso, sx, lab(min(a, 95))); v = {c: float(gb[c].get(k, np.nan)) for c in gb}
            return v["IHD"] if cause == "IHD" else v["IS"] if cause == "IS" else v["ALL"] - v["IHD"] - v["IS"]
        if cause in ("IHD", "IS"): return 0.0
        col = "DeathMale" if sx == "Male" else "DeathFemale"; pc = "PopMale" if sx == "Male" else "PopFemale"
        r = dw.set_index(["ISO3_code", "AgeGrpStart"]).loc[(iso, a), col] / p24.loc[(iso, a), pc]
        return float(r)
    out = {}
    for cause in ("IHD", "IS", "OTH"):
        rows = []
        for (iso, yr), g in pop.groupby(["ISO3_code", "Time"]):
            g = g.set_index("AgeGrpStart")
            for sx, pc in (("Male", "PopMale"), ("Female", "PopFemale")):
                for a0, ages in BANDS.items():
                    w = np.array([g.loc[a, pc] for a in ages]); r = np.array([rate(cause, iso, sx, a) for a in ages])
                    rows.append(dict(iso3=iso, sex=sx, a0=a0, year=yr, comp=float((w * r).sum() / w.sum())))
        d = pd.DataFrame(rows); base = d[d.year == 2023].set_index(["iso3", "sex", "a0"]).comp
        d["comp"] = d.comp / [base[(i, s_, a)] if base[(i, s_, a)] > 0 else np.nan for i, s_, a in zip(d.iso3, d.sex, d.a0)]
        out[cause] = d.fillna({"comp": 1.0})
    return out


def cvd():
    d = pd.read_csv("data/input/forecasts/cvd_forecast_small.csv")
    d["iso3"] = d.Location.map(NAME); d["a0"] = d.Age.str.extract(r"^(\d+)")[0].astype(int).replace({15: 25})
    d = d.pivot_table(index=["iso3", "Sex", "a0", "Year"], columns="cause_dir", values="Value").reset_index()
    rows = []; C = COMP if "COMP" in globals() else None
    for (iso, sx, a0), g in d.groupby(["iso3", "Sex", "a0"]):
        g = g.set_index("Year"); r23 = g.loc[2023.0]
        for t in range(1, YEARS + 1):
            r = g.loc[float(cal(t))]; y = cal(t)
            ci = float(C["IHD"].set_index(["iso3", "sex", "a0", "year"]).comp.get((iso, sx, a0, y), 1.0)) if C else 1.0
            cs = float(C["IS"].set_index(["iso3", "sex", "a0", "year"]).comp.get((iso, sx, a0, y), 1.0)) if C else 1.0
            rows.append(dict(iso3=iso, sex=sx, a0=a0, t=t, f_ihd=(r.IHD / r23.IHD if r23.IHD > 0 else 1.0) / ci,
                             f_is=(r.IS / r23.IS if r23.IS > 0 else 1.0) / cs, comp_ihd=ci, comp_is=cs))
    return pd.DataFrame(rows)


def oth():
    """Other-cause mortality: all causes minus IHD minus ischaemic stroke, from the same GBD Foresight reference forecasts
    (data/input/forecasts/all_cause_mortality/*.csv, same format as the IHD and IS files). Output: iso3, sex, a0, t, f_oth,
    the other-cause death rate in year t / 2023."""
    import glob
    fs = sorted(glob.glob("data/input/forecasts/all_cause_mortality/*.csv"))
    if not fs: return None
    a = pd.concat(pd.read_csv(f) for f in fs); a = a[a.Year >= 2015].drop_duplicates(["Location", "Sex", "Age", "Year"])
    c = pd.read_csv("data/input/forecasts/cvd_forecast_small.csv")
    c = c.pivot_table(index=["Location", "Sex", "Age", "Year"], columns="cause_dir", values="Value").reset_index()
    d = a.merge(c, on=["Location", "Sex", "Age", "Year"], how="inner"); d = d[d.Age != "0-14 years"]
    d["other"] = d.Value - d.IHD - d.IS; assert (d.other > 0).all(), "all-cause rate below IHD + IS"
    d["iso3"] = d.Location.map(NAME); d["a0"] = d.Age.str.extract(r"^(\d+)")[0].astype(int).replace({15: 25})
    rows = []
    for (iso, sx, a0), g in d.groupby(["iso3", "Sex", "a0"]):
        g = g.set_index("Year"); r23 = g.loc[2023.0, "other"]
        Co = COMP["OTH"].set_index(["iso3", "sex", "a0", "year"]).comp if "COMP" in globals() else None
        for t in range(1, YEARS + 1):
            co = float(Co.get((iso, sx, a0, cal(t)), 1.0)) if Co is not None else 1.0
            rows.append(dict(iso3=iso, sex=sx, a0=a0, t=t, f_oth=g.loc[float(cal(t)), "other"] / r23 / co, comp_oth=co))
    return pd.DataFrame(rows)


def bmi():
    d = pd.read_csv("data/input/forecasts/obesity_forecast_small.csv")
    d["iso3"] = d.location_name.map(NAME); d["sex"] = d.sex_id.map({1: "Male", 2: "Female"})
    d["a0"] = d.age_group_name.str.extract(r"^(\d+)")[0].astype(int)
    d = d.pivot_table(index=["iso3", "sex", "a0", "year_id"], columns="metric", values="mean_prev")
    se = pd.read_csv("data/params/select_elig.csv").set_index(["iso3", "sex", "a0"])
    rows = []
    for (iso, sx, a0), b in se.iterrows():
        if (iso, sx, a0, 2023) not in d.index: continue
        f23 = d.loc[(iso, sx, a0, 2023)]; w = (lg(b.p27) - lg(b.p25)) / (lg(b.p30) - lg(b.p25))   # 2023 position of 27
        for t in range(1, YEARS + 1):
            f = d.loc[(iso, sx, a0, cal(t))]
            p25 = float(np.clip(b.p25 * f.overweight / f23.overweight, 0, 0.999))
            p30 = float(np.clip(b.p30 * f.obesity / f23.obesity, 0, p25))
            p27 = float(ex(lg(p25) + w * (lg(p30) - lg(p25))))
            rows.append(dict(iso3=iso, sex=sx, a0=a0, t=t, p25=p25, p27=min(max(p27, p30), p25), p30=p30))
    return pd.DataFrame(rows)


def dm():
    rows = []
    for iso, pc in DM_CHANGE.items():
        g = (1 + pc / 100) ** (1 / (2050 - 2021))                   # constant annual change, 2021-2050
        rows += [dict(iso3=iso, t=t, r=g ** (cal(t) - 2023)) for t in range(1, YEARS + 1)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("data/params/trends", exist_ok=True)
    if os.path.exists("data/input/forecasts/wpp_pop_small.csv"):           # age-composition adjustment of the band rates
        COMP = composition(); print("age-composition adjustment applied (UN WPP 2024)")
    for name, fn in (("cvd_factor", cvd), ("bmi_target", bmi), ("dm_factor", dm), ("oth_factor", oth)):
        out = fn()
        if out is None: print(f"data/params/trends/{name}.csv: no input data, skipped"); continue
        out.to_csv(f"data/params/trends/{name}.csv", index=False); print(f"data/params/trends/{name}.csv: {out.shape}, {out.iso3.nunique()} locations")
