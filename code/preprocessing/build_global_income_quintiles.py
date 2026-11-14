import glob, os, re
import numpy as np, pandas as pd

W = "data/worldbank/extracted"
SHARE = {
    "SI.DST.FRST.20": "Q1",
    "SI.DST.02ND.20": "Q2",
    "SI.DST.03RD.20": "Q3",
    "SI.DST.04TH.20": "Q4",
    "SI.DST.05TH.20": "Q5",
}
GNI = "NY.GNP.PCAP.CD"
YEAR_POP = 2023


def bulk(code):
    f = [
        p
        for p in glob.glob(f"{W}/API_{code}*.csv")
        if "Metadata" not in os.path.basename(p)
    ]
    assert len(f) == 1, f"expected one bulk file for {code}, found {f}"
    return pd.read_csv(f[0], skiprows=4)


def latest(row, yrs, lo=None):
    for y in sorted(
        [y for y in yrs if lo is None or int(y) >= lo], key=int, reverse=True
    ):
        if pd.notna(row.get(y)):
            return int(y), float(row[y])
    return None, None


meta = pd.read_csv([p for p in glob.glob(f"{W}/Metadata_Country_API_*.csv")][0])
econ = meta[meta.Region.notna()].set_index("Country Code")
g = bulk(GNI)
yrs = [c for c in g.columns if re.fullmatch(r"\d{4}", str(c))]
rows = {}
for _, r in g[g["Country Code"].isin(econ.index)].iterrows():
    y, v = latest(r, yrs)
    if v is not None:
        rows[r["Country Code"]] = dict(gni_year=y, gni_pc_usd=v)
d = pd.DataFrame.from_dict(rows, orient="index")
for code, q in SHARE.items():
    b = bulk(code)
    yb = [c for c in b.columns if re.fullmatch(r"\d{4}", str(c))]
    for _, r in b[b["Country Code"].isin(d.index)].iterrows():
        y, v = latest(r, yb, lo=2010)
        d.loc[r["Country Code"], q] = v
        if q == "Q1":
            d.loc[r["Country Code"], "share_year"] = y
Q = list(SHARE.values())
d["income_group"] = d.index.map(econ.IncomeGroup)
d["shares_imputed"] = d[Q].isna().any(axis=1)
med = d[~d.shares_imputed].groupby("income_group")[Q].median()
for i in d.index[d.shares_imputed]:
    d.loc[i, Q] = (
        med.loc[d.loc[i, "income_group"]].values
        if d.loc[i, "income_group"] in med.index
        else med.median().values
    )
d[Q] = d[Q].div(d[Q].sum(axis=1), axis=0) * 100

wpp = pd.read_csv(
    "data/wpp/WPP2024_PopulationByAge5GroupSex_Medium.csv",
    low_memory=False,
    usecols=["ISO3_code", "LocTypeName", "Time", "PopTotal"],
)
pop = (
    wpp[(wpp.Time == YEAR_POP) & (wpp.LocTypeName == "Country/Area")]
    .groupby("ISO3_code")
    .PopTotal.sum()
    * 1e3
)
world = float(
    wpp[(wpp.Time == YEAR_POP) & (wpp.LocTypeName == "Country/Area")].PopTotal.sum()
    * 1e3
)
d["pop"] = d.index.map(pop)
d = d[d["pop"].notna()]

grp = pd.DataFrame(
    [
        dict(iso3=i, income=r.gni_pc_usd * r[q] / 100 * 5, pop=r["pop"] / 5)
        for i, r in d.iterrows()
        for q in Q
    ]
)
grp = grp.sort_values("income").reset_index(drop=True)
tot = grp["pop"].sum()
cum0 = grp["pop"].cumsum() - grp["pop"]
out = []
for k, q in enumerate(Q):
    lo, hi = tot * k / 5, tot * (k + 1) / 5
    w = (np.minimum(cum0 + grp["pop"], hi) - np.maximum(cum0, lo)).clip(lower=0)
    out.append(
        dict(
            iso3="GLB",
            quintile=q,
            income_pc_usd=round(float((w * grp.income).sum() / w.sum())),
        )
    )
res = pd.DataFrame(out)
res["economies"] = len(d)
res["economies_shares_imputed"] = int(d.shares_imputed.sum())
res["population_covered_pct"] = round(100 * tot / world, 1)
res["share_year_median"] = int(d.share_year.median())
res["gni_year_mode"] = int(d.gni_year.mode().iloc[0])
os.makedirs("data/params", exist_ok=True)
res.to_csv("data/params/income_quintiles_global.csv", index=False)
print(
    f"{len(d)} economies ({int(d.shares_imputed.sum())} with imputed shares), {100 * tot / world:.1f}% of world population in {YEAR_POP}"
)
print(res[["quintile", "income_pc_usd"]].to_string(index=False))
print(
    f"$140 as % of income: "
    + ", ".join(
        f"{r.quintile} {14000 / r.income_pc_usd:.1f}%" for r in res.itertuples()
    )
)
print("-> data/params/income_quintiles_global.csv")
