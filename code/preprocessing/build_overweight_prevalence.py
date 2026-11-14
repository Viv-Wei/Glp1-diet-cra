import glob, os, re, pandas as pd

os.chdir(os.environ.get("CRA_ROOT", os.getcwd()))
d = pd.read_csv(
    glob.glob("data/gbd/bmi_exposure/*PREVALENCE_OF_OBESITY_AND_OVERWEIGHT*.CSV")[0],
    low_memory=False,
)
d = d[(d.year_id == 2023) & (d.sex == "Both")].copy()
d["a0"] = pd.to_numeric(
    d.age_group_name.str.extract(r"^(\d+) (?:to|plus)")[0], errors="coerce"
)
d = d[d.a0 >= 25]
pop = pd.read_csv("data/gbd/1-3_population.csv", low_memory=False)
pop = pop[(pop.year == 2023) & (pop.sex_name == "Both")].copy()
pop["a0"] = pop.age_name.str.extract(r"^(\d+)").astype(float)
cm = pd.read_csv("data/params/country_map.csv")
iso = dict(zip(cm.gbd_location_name, cm.iso3))
rows = {}
for loc, code in iso.items():
    m = d[d.location_name == loc][["a0", "mean"]].merge(
        pop[pop.location_name == loc][["a0", "val"]], on="a0"
    )
    if len(m) == 15:
        rows[code] = (m["mean"] * m.val).sum() / m.val.sum()
w = pd.read_csv(
    "data/wpp/WPP2024_PopulationByAge5GroupSex_Medium.csv", low_memory=False
)
w = w[(w.Time == 2023) & (w.LocTypeName == "Country/Area")].copy()
w["a0"] = w.AgeGrpStart.astype(float).clip(upper=95)
w = w[w.a0 >= 25].groupby(["Location", "a0"]).PopTotal.sum().reset_index()
norm = lambda s: re.sub(r"[^a-z]", "", str(s).lower())
w["k"] = w.Location.map(norm)
d["k"] = d.location_name.map(norm)
m = d.merge(w, on=["k", "a0"])
rows["GLB"] = (m["mean"] * m.PopTotal).sum() / m.PopTotal.sum()
out = pd.Series(rows, name="overweight_prev").rename_axis("iso3").reset_index()
assert len(out) == 20
out.to_csv("data/params/overweight_prevalence.csv", index=False)
print(out.round(3).to_string(index=False))
