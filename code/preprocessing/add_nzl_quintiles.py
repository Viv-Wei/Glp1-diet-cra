import pandas as pd, glob, re

t = pd.read_csv("data/params/income_quintiles_wb.csv")
if (t.iso3 == "NZL").any():
    print("NZL already present, skipping")
    raise SystemExit
gni_file = [
    f
    for f in glob.glob("data/worldbank/extracted/API_NY.GNP.PCAP.CD*.csv")
    if "Metadata" not in f
][0]
df = pd.read_csv(gni_file, skiprows=4)
row = df[df["Country Code"] == "NZL"].iloc[0]
yrs = sorted(
    [c for c in df.columns if re.fullmatch(r"\d{4}", str(c))], key=int, reverse=True
)
gni_year, gni = next((int(y), float(row[y])) for y in yrs if pd.notna(row[y]))
shares = {"Q1": 7.3, "Q2": 12.3, "Q3": 16.5, "Q4": 22.3, "Q5": 41.6}
rows = [
    dict(
        iso3="NZL",
        quintile=q,
        share_year=2020,
        share_pct=s,
        gni_year=gni_year,
        gni_pc_usd=gni,
        income_pc_usd=round(gni * s / 100 * 5),
    )
    for q, s in shares.items()
]
t = pd.concat([t, pd.DataFrame(rows)]).sort_values(["iso3", "quintile"])
t.to_csv("data/params/income_quintiles_wb.csv", index=False)
print(f"NZL completed: GNI={gni} ({gni_year})")
print(pd.DataFrame(rows).to_string(index=False))
