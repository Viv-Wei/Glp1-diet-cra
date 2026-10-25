import glob, os
import numpy as np, pandas as pd

MEASURE = os.environ.get("F1_MEASURE", "Deaths")
SFX = "" if MEASURE == "Deaths" else "_dalys"
_cm = pd.read_csv("data/params/country_map.csv")
ISO = dict(zip(_cm.gbd_location_name, _cm.iso3))
AGE15 = [f"{a}-{a+4} years" for a in range(25, 95, 5)] + ["95+ years"]
RISK2FAC = {
    "Diet low in whole grains": "wholegrains",
    "Diet low in fruits": "fruits",
    "Diet low in vegetables": "vegetables",
    "Diet low in fiber": "fiber",
    "Diet low in fibre": "fiber",
    "Diet low in legumes": "legumes",
    "Diet low in nuts and seeds": "nutsseeds",
    "Diet low in seafood omega-3 fatty acids": "seafood_omega3",
    "Diet low in omega-6 polyunsaturated fatty acids": "omega6",
    "Diet high in red meat": "redmeat",
    "Diet high in processed meat": "procmeat",
    "Diet high in sugar-sweetened beverages": "ssb",
    "Diet high in sodium": "sodium",
    "Diet high in trans fatty acids": "transfat",
}


def load_dir_style():
    DIS = {
        "Ischemic heart disease": "IHD",
        "Ischemic stroke": "IschStroke",
        "Diabetes mellitus type 2": "T2D",
    }
    out = []
    for dis, tag in DIS.items():
        for p in sorted(glob.glob(f"data/gbd/{dis}/*/*/*.csv")):
            risk = p.split(os.sep)[3]
            if risk == "Dietary risks":
                continue
            fac = RISK2FAC.get(risk)
            if fac is None:
                continue
            d = pd.read_csv(p, low_memory=False)
            d["outcome"], d["factor"] = tag, fac
            out.append(d)
    return pd.concat(out, ignore_index=True)


def load_merged_style():
    out = []
    for f, tag in [
        ("data/gbd/queryA_ICH_all_merged.csv", "ICH"),
        ("data/gbd/queryA_SAH_all_merged.csv", "SAH"),
        ("data/gbd/queryA_ckd_3years.csv", "CKD"),
    ]:
        d = pd.read_csv(f, low_memory=False)
        d = d[d.rei_name != "Dietary risks"]
        d["outcome"] = tag
        d["factor"] = d.rei_name.map(RISK2FAC)
        out.append(d[d.factor.notna()])
    return pd.concat(out, ignore_index=True)


def main():
    d = pd.concat([load_dir_style(), load_merged_style()], ignore_index=True)
    d = d[
        (d.year == 2023)
        & d.age_name.isin(AGE15)
        & d.sex_name.isin(["Male", "Female"])
        & d.location_name.isin(ISO)
    ]
    d["iso3"] = d.location_name.map(ISO)
    paf = d[(d.metric_name == "Percent") & (d.measure_name == MEASURE)].rename(
        columns={"val": "paf"}
    )[["iso3", "outcome", "factor", "sex_name", "age_name", "paf"]]
    num = d[(d.metric_name == "Number") & (d.measure_name == MEASURE)].rename(
        columns={"val": "deaths_attr", "upper": "deaths_hi", "lower": "deaths_lo"}
    )[
        [
            "iso3",
            "outcome",
            "factor",
            "sex_name",
            "age_name",
            "deaths_attr",
            "deaths_lo",
            "deaths_hi",
        ]
    ]
    t = paf.merge(num, on=["iso3", "outcome", "factor", "sex_name", "age_name"])
    t.to_csv(f"data/results/factor_burden_by_age{SFX}.csv", index=False)
    print(
        f"{len(t)} rows, {t.iso3.nunique()} locations, {t.factor.nunique()} factors -> data/results/factor_burden_by_age{SFX}.csv"
    )


if __name__ == "__main__":
    main()
