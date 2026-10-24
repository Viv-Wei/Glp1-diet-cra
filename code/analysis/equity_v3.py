import argparse, os
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--run", default="data/results/v3/equity_cost/main_0.20")
p.add_argument("--paf", default="data/results/v3/equity_paf_by_edu_v2.csv")
p.add_argument("--outdir", default="data/results/v3/equity_cost")
a = p.parse_args()

ISO_W = {
    "AUS": "Australia",
    "BGD": "Bangladesh",
    "BGR": "Bulgaria",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHN": "China",
    "DEU": "Germany",
    "EGY": "Egypt",
    "GBR": "United Kingdom of Great Britain and Northern Ireland",
    "IND": "India",
    "ISR": "Israel",
    "JPN": "Japan",
    "MEX": "Mexico",
    "NGA": "Nigeria",
    "NZL": "New Zealand",
    "ROU": "Romania",
    "TUR": "Turkey",
    "USA": "United States of America",
    "ZAF": "South Africa",
}
ALT = {
    "GBR": ["United Kingdom"],
    "TUR": ["Türkiye", "Turkiye"],
    "USA": ["United States"],
    "EGY": ["Egypt, Arab Rep."],
}
EDU3 = {
    "No Education": 1,
    "Incomplete Primary": 1,
    "Primary": 1,
    "Lower Secondary": 2,
    "Upper Secondary": 2,
    "Post Secondary": 3,
    "Short Post Secondary": 3,
    "Bachelor": 3,
    "Master and higher": 3,
}
LAB = {1: "Low education", 2: "Medium education", 3: "High education"}
PRICES = {
    "Branded": "afford_brand_pct",
    "Generic $140/yr": "afford_generic_hi_pct",
    "Generic $28/yr": "afford_generic_lo_pct",
}
OUT_W = {"IHD": "Ischaemic heart disease", "IschStroke": "Ischaemic stroke"}
THRESH = 10.0
AGE5 = [f"{x}-{x+4} years" for x in range(25, 95, 5)] + ["95+ years"]
a0 = lambda s: pd.to_numeric(
    pd.Series(s).astype(str).str.extract(r"^(\d+)")[0], errors="coerce"
).values

w = pd.read_csv("data/input/wcde_clean.csv")
yr = int(w.Year.unique()[np.argmin(np.abs(w.Year.unique() - 2023))])
w = w[(w.Year == yr) & w.Education.isin(EDU3)].copy()
w["a0"] = a0(w.Age)
w = w[w.a0 >= 25]
w.loc[w.a0 >= 95, "a0"] = 95
w["e"] = w.Education.map(EDU3)
SA, S = {}, {}
for iso, nm in ISO_W.items():
    x = w[w.Area.isin([nm] + ALT.get(iso, []))]
    assert len(x), f"Wittgenstein: no rows for {iso}"
    g = (
        x.groupby(["a0", "e"])
        .Population.sum()
        .unstack("e")
        .reindex(columns=[1, 2, 3])
        .fillna(0)
    )
    SA[iso] = g.div(g.sum(axis=1), axis=0)
    tot = g.sum()
    S[iso] = tot / tot.sum()
S = pd.DataFrame(S).T

paf = pd.read_csv(a.paf)
paf = paf[paf.outcome.isin(OUT_W)]
j = pd.read_csv("data/results/supp_table2_joint.csv")
dcol = next(c for c in j.columns if "death" in c.lower() and "rate" not in c.lower())
ocol = next(c for c in j.columns if c.lower() in ("outcome", "cause"))
icol = next(c for c in j.columns if c.lower() in ("iso3", "iso"))


def R_e(iso):
    jj = j[j[icol] == iso].set_index(ocol)[dcol]
    wo = pd.Series({o: float(jj.get(o, jj.get(OUT_W[o], np.nan))) for o in OUT_W}).clip(
        lower=0
    )
    wo = wo / wo.sum()
    r = pd.Series(0.0, index=[1, 2, 3])
    for o, v in wo.items():
        q = (
            paf[(paf.iso3 == iso) & (paf.outcome == o)]
            .set_index("edu")
            .paf.reindex([1, 2, 3])
        )
        if q.notna().all() and (S.loc[iso] * q).sum() > 0:
            r += v * q / (S.loc[iso] * q).sum()
        else:
            r += v
    return r


aff = pd.read_csv("data/results/affordability_matrix.csv")


def afford_frac(iso, col, lam=1.0):
    ok = (
        aff[aff.iso3 == iso]
        .set_index("quintile")[col]
        .reindex(["Q1", "Q2", "Q3", "Q4", "Q5"])
        <= THRESH
    )
    s = S.loc[iso]
    cum = np.concatenate([[0], np.cumsum(s.values)])
    out = {}
    for k, e in enumerate([1, 2, 3]):
        lo_, hi_ = cum[k], cum[k + 1]
        out[e] = (
            0.0
            if hi_ - lo_ <= 0
            else sum(
                max(0, min(hi_, (q + 1) * 0.2) - max(lo_, q * 0.2)) * ok.iloc[q]
                for q in range(5)
            )
            / (hi_ - lo_)
        )
    return lam * pd.Series(out) + (1 - lam) * ok.mean()


by = pd.read_csv(a.run + "_byage.csv")
by["a0"] = a0(by.age)


def split(iso, arm, weights_fn):
    d = by[(by.iso3 == iso) & (by.arm == arm)].set_index("a0").deaths_averted
    out = pd.Series(0.0, index=[1, 2, 3])
    old = 0.0
    for a_, v in d.items():
        sa = SA[iso].loc[a_] if a_ in SA[iso].index else S.loc[iso]
        wgt = weights_fn(sa)
        out += v * wgt
        if a_ >= 65:
            old += v * wgt
    return out, old


rows = []
for price, col in PRICES.items():
    for iso in sorted(ISO_W):
        r = R_e(iso)
        af = afford_frac(iso, col)
        dd, _ = split(iso, "diet", lambda sa: sa * r / (sa * r).sum())
        gg, g65 = split(iso, "glp1", lambda sa: sa * af)
        gtot, _ = split(iso, "glp1", lambda sa: sa)
        for e in [1, 2, 3]:
            rows.append(
                [
                    price,
                    iso,
                    LAB[e],
                    round(S.loc[iso, e], 3),
                    round(r[e], 2),
                    round(af[e], 3),
                    round(dd[e] / 1e3, 2),
                    round(gtot[e] / 1e3, 2),
                    round(gg[e] / 1e3, 2),
                    round(dd[e] / gg[e], 2) if gg[e] > 0 else None,
                ]
            )
cols = [
    "Price scenario",
    "ISO3",
    "Education group",
    "Population share (25+)",
    "PAF ratio",
    "Affordable fraction",
    "Dietary policy averted (thousands)",
    "GLP-1RA averted if affordable to all (thousands)",
    "GLP-1RA averted (thousands)",
    "Diet-to-treatment ratio",
]
df = pd.DataFrame(rows, columns=cols)
os.makedirs(a.outdir, exist_ok=True)
df.to_csv(os.path.join(a.outdir, "tableS_equity_v3.csv"), index=False)

sens = []
for lam, lab in [
    (1.0, "Education and income perfectly ranked (main analysis)"),
    (0.5, "Partial link (50%)"),
    (0.0, "Education unrelated to income"),
]:
    dl = gl = 0.0
    none = []
    for iso in sorted(ISO_W):
        r = R_e(iso)
        af = afford_frac(iso, PRICES["Generic $140/yr"], lam)
        dl += split(iso, "diet", lambda sa: sa * r / (sa * r).sum())[0][1]
        g = split(iso, "glp1", lambda sa: sa * af)[0][1]
        gl += g
        if g <= 0:
            none.append(iso)
    sens.append(
        {
            "Education-income link": lab,
            "Dietary policy, low education (thousands)": round(dl / 1e3),
            "GLP-1RA, low education (thousands)": round(gl / 1e3),
            "Diet-to-treatment ratio": round(dl / gl, 2) if gl else None,
            "Countries with no treatment benefit": f"{len(none)}/19"
            + (f" ({', '.join(none)})" if none else ""),
        }
    )
pd.DataFrame(sens).to_csv(
    os.path.join(a.outdir, "tableS_equity_mapping_sensitivity_v3.csv"), index=False
)

print(f"Wittgenstein {yr}; run {a.run}")
pool = df.groupby(["Price scenario", "Education group"])[cols[6:9]].sum()
print("\nPooled over 19 countries (thousands):\n" + pool.round(0).to_string())
g140 = df[df["Price scenario"] == "Generic $140/yr"]
pp = g140.groupby("Education group")[
    ["GLP-1RA averted (thousands)", "Dietary policy averted (thousands)"]
].sum()
print(
    "\nTreatment as % of dietary policy deaths averted, generic US$140:\n"
    + (100 * pp.iloc[:, 0] / pp.iloc[:, 1]).round(0).to_string()
)
share_low = {}
for arm in ["diet", "glp1"]:
    t = sum(split(iso, arm, lambda sa: sa)[0] for iso in ISO_W)
    share_low[arm] = float(t[1] / t.sum())
pop_low = float(sum(S.loc[i, 1] for i in ISO_W) / len(ISO_W))
print(
    f"\nShare of deaths averted (before affordability) in the low-education group, pooled: dietary policy "
    f"{100*share_low['diet']:.0f}%, treatment {100*share_low['glp1']:.0f}% (unweighted mean low-education population share {100*pop_low:.0f}%)"
)
print(
    "\nEducation-income link, low education, generic US$140:\n"
    + pd.DataFrame(sens).to_string(index=False)
)
chk = max(
    abs(
        split(i, "diet", lambda sa: sa * R_e(i) / (sa * R_e(i)).sum())[0].sum()
        - by[(by.iso3 == i) & (by.arm == "diet")].deaths_averted.sum()
    )
    for i in ISO_W
)
print(
    f"\nCheck: dietary allocation sums to national total (max abs difference): {chk:.3e}"
)
