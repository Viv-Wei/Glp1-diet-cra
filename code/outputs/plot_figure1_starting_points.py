import argparse, os
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from matplotlib.patches import Patch

p = argparse.ArgumentParser()
p.add_argument("--root", default=".")
p.add_argument("--xlsx", default="data/tables/Supplementary_Tables_final1014.xlsx")
p.add_argument("--mc", default="data/results/v3/mc")
p.add_argument("--programme", default="data/results/v3/decomp/main_check_programme.csv")
p.add_argument("--out", default="data/figures/figure1_starting_points")
a = p.parse_args()
R = lambda f: os.path.join(a.root, f)
X = R(a.xlsx)
tab = lambda n: pd.read_excel(X, f"Supplementary Table {n}", header=1).iloc[:-1]

t2, t6 = tab(2), tab(6)
_cr = pd.read_csv(R(f"{a.mc}/summary_crossover_ratios.csv")).set_index("iso3")
ORDER = ["GLB"] + _cr.drop("GLB").sort_values("xext_median_%").index.tolist()

OG = {
    "Ischaemic heart disease": "IHD",
    "Ischaemic stroke": "Stroke",
    "Intracerebral haemorrhage": "Stroke",
    "Subarachnoid haemorrhage": "Stroke",
    "Type 2 diabetes": "T2D",
    "Chronic kidney disease": "CKD",
}
t2["og"] = t2.Outcome.map(OG)
asr = t2.assign(v=t2["Age-standardised rate per 100,000"].clip(lower=0)).pivot_table(
    index="ISO3", columns="og", values="v", aggfunc="sum"
)[["IHD", "Stroke", "T2D", "CKD"]]

pg = pd.read_csv(R(a.programme))
pg = pg[(pg.arm == "glp1") & (pg.year == 1)].set_index("iso3")
ow = pg.eligible / pg.alive

FG = {
    "Sodium (high)": "Sodium",
    "Whole grains (low)": "Whole grains",
    "SSB (high)": "Sugar-sweetened beverages",
    "Fruits (low)": "Fruits, vegetables, legumes",
    "Vegetables (low)": "Fruits, vegetables, legumes",
    "Legumes (low)": "Fruits, vegetables, legumes",
    "Nuts & seeds (low)": "Nuts, seeds, fibre",
    "Fibre (low)": "Nuts, seeds, fibre",
    "Seafood omega-3 (low)": "Omega-3, omega-6 fats",
    "Omega-6 PUFA (low)": "Omega-3, omega-6 fats",
    "Red meat (high)": "Meat, trans fat",
    "Processed meat (high)": "Meat, trans fat",
    "Trans fat (high)": "Meat, trans fat",
}
GORDER = [
    "Sodium",
    "Whole grains",
    "Sugar-sweetened beverages",
    "Fruits, vegetables, legumes",
    "Nuts, seeds, fibre",
    "Omega-3, omega-6 fats",
    "Meat, trans fat",
]
GCOL = ["#B5651D", "#E1A13A", "#F3D27A", "#00A087", "#7CC9B5", "#4DBBD5", "#B8B8B8"]
t6["g"] = t6["Dietary factor"].map(FG)
fb = (
    t6.assign(v=t6["Diet-attributable CV deaths"].clip(lower=0))
    .pivot_table(index="ISO3", columns="g", values="v", aggfunc="sum")
    .reindex(columns=GORDER)
    .fillna(0)
)
share = fb.div(fb.sum(axis=1), axis=0) * 100
pkg = share[GORDER[:3]].sum(axis=1)

data = pd.concat(
    [
        asr.add_prefix("asr_"),
        ow.rename("overweight"),
        share.add_prefix("share_"),
        pkg.rename("pkg_share"),
    ],
    axis=1,
).reindex(ORDER)
assert data.notna().all().all(), data[data.isna().any(axis=1)]
os.makedirs(os.path.dirname(R(a.out)) or ".", exist_ok=True)
data.round(3).to_csv(R(a.out) + "_data.csv")

GRP = {
    "GLB": "#1a1a1a",
    **{k: "#00A087" for k in "BGR ROU IND MEX ZAF CHN TUR BRA NZL CAN AUS ISR".split()},
    **{k: "#3C5488" for k in "EGY BGD NGA".split()},
    **{k: "#E64B35" for k in "USA DEU GBR JPN".split()},
}
OCOL = {"IHD": "#3C5488", "Stroke": "#8491B4", "T2D": "#F39B7F", "CKD": "#DC9D8E"}
OLAB = {"IHD": "IHD", "Stroke": "Stroke (all types)", "T2D": "T2D", "CKD": "CKD"}
plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "pdf.fonttype": 42,
    }
)
fig = plt.figure(figsize=(7.4, 5.1))
L, B, Hh = 0.075, 0.235, 0.69
axA = fig.add_axes([L, B, 0.30, Hh])
axB = fig.add_axes([L + 0.325, B, 0.14, Hh])
axC = fig.add_axes([L + 0.49, B, 0.33, Hh])
y = np.arange(len(ORDER))[::-1]
h = 0.72
ylab = ["Global" if i == "GLB" else i for i in ORDER]

left = np.zeros(len(ORDER))
for k in ["IHD", "Stroke", "T2D", "CKD"]:
    v = data[f"asr_{k}"].values
    axA.barh(
        y,
        v,
        left=left,
        height=h,
        color=OCOL[k],
        edgecolor="white",
        lw=0.3,
        label=OLAB[k],
    )
    left += v
for yy, tot in zip(y, left):
    axA.text(tot + 5, yy, f"{tot:.0f}", va="center", fontsize=6.2)
axA.set_xlim(0, left.max() * 1.15)
axA.set_xlabel(
    "Diet-attributable deaths per 100 000\n(age-standardised, adults ≥25 years)"
)
axA.legend(
    loc="center right",
    bbox_to_anchor=(1.0, 0.55),
    fontsize=6.3,
    frameon=False,
    handlelength=1.1,
    borderaxespad=0.2,
)

axB.barh(y, data.overweight.values * 100, height=h, color="#9E9AC8")
for yy, v in zip(y, data.overweight.values * 100):
    axB.text(v + 1.5, yy, f"{v:.0f}", va="center", fontsize=6.2)
axB.set_xlim(0, 100)
axB.set_xticks([0, 50, 100])
axB.set_xlabel("Adults eligible\nfor GLP-1RA (%)")

left = np.zeros(len(ORDER))
for gname, col in zip(GORDER, GCOL):
    v = data[f"share_{gname}"].values
    axC.barh(y, v, left=left, height=h, color=col, edgecolor="white", lw=0.3)
    left += v
for yy, v in zip(y, data.pkg_share.values):
    axC.text(
        102,
        yy,
        f"{v:.0f}%",
        va="center",
        fontsize=6.4,
        fontweight="bold",
        color="#8a4a12",
    )
axC.text(
    101,
    len(ORDER) - 0.3,
    "Share\ntargeted*",
    fontsize=6.2,
    color="#8a4a12",
    va="bottom",
    fontweight="bold",
)
axC.set_xlim(0, 100)
axC.set_xticks([0, 50, 100])
axC.set_xlabel(
    "Share of diet-attributable cardiovascular deaths\nby dietary factor (%)"
)

for ax, tag in [(axA, "a"), (axB, "b"), (axC, "c")]:
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_yticks(y)
    ax.tick_params(axis="y", length=0)
    ax.text(
        -0.02 if tag != "a" else -0.21,
        1.03,
        tag,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
    )
    if tag == "a":
        ax.set_yticklabels(ylab)
        for t, iso in zip(ax.get_yticklabels(), ORDER):
            t.set_color(GRP[iso])
            t.set_fontweight("bold" if iso == "GLB" else "normal")
    else:
        ax.set_yticklabels([])
    ax.axhline(len(ORDER) - 1.5, color="grey", lw=0.5, ls=(0, (2, 2)))

fig.legend(
    handles=[
        Patch(color=c, label=g + (" *" if i < 3 else ""))
        for i, (g, c) in enumerate(zip(GORDER, GCOL))
    ],
    loc="lower center",
    bbox_to_anchor=(0.705, 0.0),
    ncol=3,
    fontsize=6.0,
    frameon=False,
    handlelength=1.0,
    columnspacing=0.6,
    handletextpad=0.4,
)
fig.text(
    0.02,
    0.025,
    "Countries ordered as in Figure 3\n(crossover coverage, low to high).\n* Targeted by the feasible\n  dietary package",
    fontsize=6.3,
    color="grey",
)
fig.text(
    0.245,
    0.095,
    "Country code colour\n(semaglutide patent status)",
    fontsize=6.2,
    fontweight="bold",
    va="bottom",
)
for k_, (lab, c) in enumerate(
    (
        ("Patent expiry by 2027", "#00A087"),
        ("No patent filed", "#3C5488"),
        ("Patent protected", "#E64B35"),
    )
):
    fig.text(
        0.245,
        0.07 - k_ * 0.022,
        lab,
        fontsize=6.2,
        color=c,
        fontweight="bold",
        va="bottom",
    )
fig.savefig(R(a.out) + ".pdf")
fig.savefig(R(a.out) + ".png", dpi=400)
print("saved:", R(a.out))
