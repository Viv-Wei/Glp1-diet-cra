import pandas as pd, glob, os, re

COUNTRIES = [
    "CHN",
    "IND",
    "BRA",
    "CAN",
    "TUR",
    "ZAF",
    "MEX",
    "AUS",
    "NZL",
    "ROU",
    "BGR",
    "ISR",
    "NGA",
    "EGY",
    "BGD",
    "USA",
    "JPN",
    "GBR",
    "DEU",
]

QMAP = {
    "SI.DST.FRST.20": "Q1",
    "SI.DST.02ND.20": "Q2",
    "SI.DST.03RD.20": "Q3",
    "SI.DST.04TH.20": "Q4",
    "SI.DST.05TH.20": "Q5",
}
GNI = "NY.GNP.PCAP.CD"


def read_wb(path):
    df = pd.read_csv(path, skiprows=4)
    df = df[df["Country Code"].isin(COUNTRIES)]
    yrs = [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))]
    return df, yrs


def latest(row, yrs, lo=2010):
    for y in sorted([y for y in yrs if int(y) >= lo], key=int, reverse=True):
        v = row.get(y)
        if pd.notna(v):
            return int(y), float(v)
    return None, None


files = [
    f
    for f in glob.glob("data/worldbank/extracted/API_*.csv")
    if "Metadata" not in os.path.basename(f)
]
print(f"main data files: {len(files)} (expected 6)")

share_rows, gni = [], {}
for f in files:
    df, yrs = read_wb(f)
    code = df["Indicator Code"].iloc[0] if len(df) else "?"
    for _, r in df.iterrows():
        y, v = latest(r, yrs)
        if v is None:
            print(f"  !! {r['Country Code']} {code} has no data after 2010")
            continue
        if code == GNI:
            gni[r["Country Code"]] = (y, v)
        elif code in QMAP:
            share_rows.append(
                dict(
                    iso3=r["Country Code"],
                    quintile=QMAP[code],
                    share_year=y,
                    share_pct=v,
                )
            )

t = pd.DataFrame(share_rows).sort_values(["iso3", "quintile"])
t["gni_year"] = t["iso3"].map(lambda c: gni.get(c, (None,))[0])
t["gni_pc_usd"] = t["iso3"].map(lambda c: gni.get(c, (None, None))[1])
t["income_pc_usd"] = (t["gni_pc_usd"] * t["share_pct"] / 100 * 5).round(0)
t.to_csv("data/params/income_quintiles_wb.csv", index=False)

print(t.to_string(index=False))
print(
    "\nshare survey year by country:",
    {c: t[t.iso3 == c]["share_year"].iloc[0] for c in t.iso3.unique()},
)
chk = t.groupby("iso3")["share_pct"].sum().round(1)
print("share total check (should be ~100):\n", chk.to_string())
