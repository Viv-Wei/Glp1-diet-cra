import os
import numpy as np, pandas as pd
from scipy import optimize, stats

os.chdir(os.environ.get("CRA_ROOT", os.getcwd()))
se = pd.read_csv("data/params/select_elig.csv")
cm = pd.read_csv("data/params/country_map.csv")
ISO = dict(zip(cm.gbd_location_name, cm.iso3))
a0 = lambda s: pd.to_numeric(
    pd.Series(s).astype(str).str.extract(r"^(\d+)")[0], errors="coerce"
).values
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


def gamma_p27(p25, p30):

    if not (0 < p30 < p25 < 1):
        return np.nan

    def f(x):
        k, th = np.exp(x)
        return [
            stats.gamma.sf(25, k, scale=th) - p25,
            stats.gamma.sf(30, k, scale=th) - p30,
        ]

    best = None
    for k0 in (20.0, 40.0, 80.0):
        sol = optimize.root(f, np.log([k0, 26.0 / k0]), method="hybr")
        if sol.success and max(abs(np.array(f(sol.x)))) < 1e-8:
            best = sol.x
            break
    if best is None:
        return np.nan
    k, th = np.exp(best)
    return float(stats.gamma.sf(27, k, scale=th))


x = se.set_index(["iso3", "sex", "a0"]).join(pop, how="left")
x["p27_gamma"] = [gamma_p27(r.p25, r.p30) for r in x.itertuples()]
x["p27_gamma"] = x.p27_gamma.fillna(x.p27)
main_ow = (
    pd.read_csv("data/params/overweight_prevalence_bmi27.csv")
    .set_index("iso3")
    .overweight_prev
)
main_cvd = pd.read_csv("data/params/overweight_cvd_bmi27.csv").set_index("iso3")
rows = []
for iso, g in x.groupby(level=0):
    w = g["pop"].fillna(0)
    p27l, p27g = (g.p27 * w).sum() / w.sum(), (g.p27_gamma * w).sum() / w.sum()
    ratio = p27g / p27l if p27l > 0 else 1.0
    rows.append(
        dict(
            iso3=iso,
            p27_lognormal=p27l,
            p27_gamma=p27g,
            ratio=ratio,
            overweight_prev=float(main_ow.get(iso, p27l)) * ratio,
            ow_cvd_ihd=(
                min(float(main_cvd.loc[iso, "ow_cvd_ihd"]) * ratio, 1.0)
                if iso in main_cvd.index
                else np.nan
            ),
            ow_cvd_is=(
                min(float(main_cvd.loc[iso, "ow_cvd_is"]) * ratio, 1.0)
                if iso in main_cvd.index
                else np.nan
            ),
        )
    )
r = pd.DataFrame(rows)
r[["iso3", "overweight_prev"]].to_csv(
    "data/params/overweight_prevalence_bmi27_gamma.csv", index=False
)
r[["iso3", "ow_cvd_ihd", "ow_cvd_is"]].to_csv(
    "data/params/overweight_cvd_bmi27_gamma.csv", index=False
)
r.to_csv("data/params/bmi27_fit_comparison.csv", index=False)
print("Share of adults aged 25+ with BMI >=27 (%), log-normal (main) vs gamma fit")
print(
    (r.set_index("iso3")[["p27_lognormal", "p27_gamma"]] * 100)
    .round(1)
    .assign(ratio=r.set_index("iso3").ratio.round(3))
    .to_string()
)
print(
    "-> data/params/overweight_prevalence_bmi27_gamma.csv, data/params/overweight_cvd_bmi27_gamma.csv, data/params/bmi27_fit_comparison.csv"
)
