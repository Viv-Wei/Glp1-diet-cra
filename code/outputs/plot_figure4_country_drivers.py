import argparse, os
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from matplotlib.lines import Line2D

p = argparse.ArgumentParser()
p.add_argument("--root", default=".")
p.add_argument("--xlsx", default="data/tables/Supplementary_Tables_final1014.xlsx")
p.add_argument(
    "--t-factor",
    type=int,
    default=6,
    help="table with diet-attributable CV deaths by dietary factor",
)
p.add_argument(
    "--t-prio",
    type=int,
    default=12,
    help="table with each factor's share of the country CVD benefit",
)
p.add_argument(
    "--prio-csv",
    default="data/tables/table_single_factor_v3.csv",
    help="dietary priorities from code/analysis/run_single_factor_v3.sh (used if present; otherwise --t-prio)",
)
p.add_argument("--programme", default="data/results/v3/decomp/main_check_programme.csv")
p.add_argument("--mc", default="data/results/v3/mc")
p.add_argument("--out", default="data/figures/figure4_country_drivers")
a = p.parse_args()
R = lambda f: os.path.join(a.root, f)
tab = lambda n: pd.read_excel(R(a.xlsx), f"Supplementary Table {n}", header=1).iloc[:-1]

t2, t6 = tab(2), tab(a.t_factor)
if os.path.exists(R(a.prio_csv)):
    t11 = pd.read_csv(R(a.prio_csv)).rename(columns={"Country": "ISO3"})
    print("dietary priorities:", a.prio_csv)
else:
    t11 = tab(a.t_prio)
    print(f"dietary priorities: Supplementary Table {a.t_prio} (earlier engine)")
cr = pd.read_csv(R(f"{a.mc}/summary_crossover_ratios.csv")).set_index("iso3")
key = cr["xext_median_%"]
ORDER = ["GLB"] + key.drop("GLB").sort_values().index.tolist()

PKG = ["Sodium (high)", "Whole grains (low)", "SSB (high)"]
cv = t6.assign(v=t6["Diet-attributable CV deaths"].clip(lower=0))
pkg = (
    cv[cv["Dietary factor"].isin(PKG)].groupby("ISO3").v.sum()
    / cv.groupby("ISO3").v.sum()
    * 100
)

pg = pd.read_csv(R(a.programme))
pg = pg[(pg.arm == "glp1") & (pg.year == 1)].set_index("iso3")
ow = pg.eligible / pg.alive * 100

cross = pd.to_numeric(cr["crossover_median_%"], errors="coerce").where(
    lambda v: v <= 100
)
d = pd.DataFrame(
    {"r20": cr["r20_median"], "cross": cross, "pkg": pkg, "ow": ow}
).reindex(ORDER)
d["idx"] = d.pkg / d.ow
asr = (
    t2.assign(
        v=pd.to_numeric(t2["Age-standardised rate per 100,000"], errors="coerce").clip(
            lower=0
        )
    )
    .groupby("ISO3")
    .v.sum()
)
d["asr"] = d.index.map(asr)
cty = d.drop("GLB", errors="ignore").dropna(subset=["r20", "pkg", "ow", "idx", "asr"])
from scipy.stats import spearmanr

rc = cty.dropna(subset=["cross"])
rho, p_rho = spearmanr(rc.r20, rc.cross) if len(rc) >= 3 else (np.nan, np.nan)

FAC = [
    "Sodium",
    "Whole grains",
    "SSB",
    "Fruits",
    "Vegetables",
    "Legumes",
    "Nuts & seeds",
    "Fibre",
    "Omega-6",
    "Seafood omega-3",
]
sh = t11.pivot_table(
    index="Dietary factor", columns="ISO3", values="Share of country CVD benefit (%)"
).reindex(FAC)[ORDER]
os.makedirs(os.path.dirname(R(a.out)) or ".", exist_ok=True)
d.round(4).to_csv(R(a.out) + "_data.csv")
sh.round(1).to_csv(R(a.out) + "_priorities.csv")

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.2,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    }
)
BINS = [
    (0, 20, "≤20%", "#E64B35"),
    (20, 50, "20–50%", "#F39B7F"),
    (50, 100, "50–100%", "#4DBBD5"),
    (100, np.inf, "Not reached", "#3C5488"),
]
tcol = lambda v: (
    BINS[-1][3]
    if not np.isfinite(v)
    else next(c for lo, hi, _, c in BINS if lo < v <= hi or (lo == 0 and v <= hi))
)
PAT = {
    **{k: "#00A087" for k in "BGR ROU IND MEX ZAF CHN TUR BRA NZL CAN AUS ISR".split()},
    **{k: "#3C5488" for k in "EGY BGD NGA".split()},
    **{k: "#E64B35" for k in "USA DEU GBR JPN".split()},
    "GLB": "black",
}


def rci(x, y):

    r_ = np.corrcoef(np.log10(x), np.log10(y))[0, 1]
    z, se = np.arctanh(r_), 1 / np.sqrt(len(x) - 3)
    return r_, np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


fig = plt.figure(figsize=(7.4, 6.6))
W, G, L0 = 0.205, 0.035, 0.075
axs = [fig.add_axes([L0 + k * (W + G), 0.585, W, 0.33]) for k in range(4)]
axE = fig.add_axes([0.125, 0.115, 0.855, 0.355])
PAN = [
    (
        "asr",
        "Diet-attributable deaths\nper 100 000 (age-standardised)",
        [30, 50, 100, 200, 400],
    ),
    (
        "pkg",
        "Share of diet-attributable CVD\ndeaths targeted by package (%)",
        [5, 10, 20, 50],
    ),
    ("ow", "Adults eligible for\nGLP-1RA treatment (%)", [5, 10, 20, 50]),
    ("idx", "Targeted share /\nshare of adults eligible", [0.1, 0.2, 0.5, 1, 2, 5]),
]
OFF = {
    "asr": {
        "ZAF": (4, -2),
        "NZL": (-17, -7),
        "GBR": (-18, 3),
        "USA": (4, -4),
        "IND": (-15, 4),
        "ROU": (4, -7),
        "GLB": (-28, -3),
        "ISR": (3, 3),
        "BGR": (4, -3),
    },
    "pkg": {
        "GBR": (-18, 3),
        "NZL": (-18, -7),
        "AUS": (-18, 2),
        "USA": (3, 3),
        "IND": (-15, 3),
        "GLB": (-28, 2),
        "ROU": (-20, 2),
        "BGR": (3, -6),
        "CAN": (3, -6),
        "DEU": (3, -7),
    },
    "ow": {
        "IND": (-15, -6),
        "JPN": (-16, 2),
        "CHN": (4, 3),
        "BGD": (4, -5),
        "ROU": (3, 2),
        "GLB": (3, 3),
        "GBR": (-18, 3),
        "AUS": (3, -6),
        "NZL": (3, 3),
        "USA": (3, -6),
        "CAN": (-18, -4),
        "DEU": (-18, 2),
    },
    "idx": {
        "IND": (3, -7),
        "ROU": (-19, 2),
        "BGD": (-20, 3),
        "GLB": (-28, 3),
        "BGR": (3, -6),
        "NZL": (-18, -7),
        "GBR": (-19, 4),
        "USA": (3, -7),
        "AUS": (-19, 1),
        "CAN": (3, -6),
        "DEU": (3, -1),
        "JPN": (3, -7),
        "CHN": (3, 2),
    },
}
for k, (ax, (col, xl, xt)) in enumerate(zip(axs, PAN)):
    r_, lo_, hi_ = rci(cty[col], cty.r20)
    b1_, b0_ = np.polyfit(np.log10(cty[col]), np.log10(cty.r20), 1)
    xs_ = np.logspace(
        np.log10(cty[col].min() * 0.85), np.log10(cty[col].max() * 1.15), 50
    )
    ax.plot(
        xs_,
        10 ** (b0_ + b1_ * np.log10(xs_)),
        color="grey",
        lw=0.8,
        ls=(0, (3, 2)),
        zorder=1,
    )
    ax.axhline(1, color="black", lw=0.6, zorder=1)
    ax.axhspan(1, 1e3, color="#00A087", alpha=0.04, lw=0, zorder=0)
    ax.axhspan(1e-3, 1, color="#E64B35", alpha=0.04, lw=0, zorder=0)
    for iso, rr in d.iterrows():
        glb = iso == "GLB"
        ax.scatter(
            rr[col],
            rr.r20,
            s=55 if glb else 26,
            marker="*" if glb else "o",
            color="black" if glb else tcol(rr.cross),
            edgecolor="white",
            lw=0.4,
            zorder=3,
        )
        ax.annotate(
            "Global" if glb else iso,
            (rr[col], rr.r20),
            xytext=OFF[col].get(iso, (3, 2)),
            textcoords="offset points",
            fontsize=5.4,
            fontweight="bold" if glb else "normal",
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{v:g}" for v in xt])
    if col == "idx":
        ax.set_xlim(0.1, 9)
    YT = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]
    ylo, yhi = d.r20.min() / 1.6, d.r20.max() * 1.6
    ax.set_yticks(YT)
    ax.set_ylim(ylo, yhi)
    (x0_, x1_), (y0_, y1_) = ax.get_xlim(), ax.get_ylim()
    xn = (np.log10(d[col]) - np.log10(x0_)) / (np.log10(x1_) - np.log10(x0_))
    yn = (np.log10(d.r20) - np.log10(y0_)) / (np.log10(y1_) - np.log10(y0_))
    corners = {
        (0.04, 0.04, "left", "bottom"): (xn < 0.45) & (yn < 0.22),
        (0.97, 0.04, "right", "bottom"): (xn > 0.55) & (yn < 0.22),
        (0.04, 0.96, "left", "top"): (xn < 0.45) & (yn > 0.78),
        (0.97, 0.96, "right", "top"): (xn > 0.55) & (yn > 0.78),
    }
    px, py, ha_, va_ = min(corners, key=lambda k: int(corners[k].sum()))
    ax.text(
        px,
        py,
        f"r = {r_:.2f}\n(95% CI {lo_:.2f} to {hi_:.2f})",
        transform=ax.transAxes,
        fontsize=5.5,
        va=va_,
        ha=ha_,
    )
    ax.set_yticklabels([f"{v:g}" for v in YT] if k == 0 else [])
    ax.set_xlabel(xl, fontsize=6.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        -0.08 if k else -0.32,
        1.04,
        "abcd"[k],
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
    )
axs[0].set_ylabel(
    "Deaths averted, dietary policy / GLP-1RA\ntreatment at 20% coverage (log scale)",
    fontsize=6.8,
)
fig.legend(
    handles=[
        Line2D([], [], marker="o", ls="", color=c, ms=4.5, label=l)
        for lo_, hi_, l, c in BINS
    ]
    + [Line2D([], [], marker="*", ls="", color="black", ms=7, label="Global")],
    title="Crossover coverage",
    title_fontsize=6.6,
    fontsize=6.6,
    loc="upper center",
    bbox_to_anchor=(0.53, 0.995),
    ncol=6,
    frameon=False,
    handletextpad=0.2,
    columnspacing=1.2,
)

X = np.arange(len(ORDER))
Y = np.arange(len(FAC))
axE.axhspan(-0.5, 2.5, color="#F3D27A", alpha=0.18, lw=0)
SC = 7.0
for j, iso in enumerate(ORDER):
    col_vals = sh[iso]
    top = col_vals.idxmax()
    for i, f in enumerate(FAC):
        v = col_vals[f]
        if pd.isna(v) or v <= 0:
            continue
        c = "#B5651D" if i < 3 else "#00A087"
        axE.scatter(
            j,
            i,
            s=v * SC,
            color=c,
            alpha=0.85,
            edgecolor="black" if f == top else "white",
            lw=1.0 if f == top else 0.4,
            zorder=3,
        )
        if v >= 15:
            axE.text(
                j,
                i,
                f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=5.4,
                color="white",
                zorder=4,
            )
axE.axvline(0.5, color="grey", lw=0.5)
axE.set_xlim(-0.6, len(ORDER) - 0.4)
axE.set_ylim(len(FAC) - 0.5, -0.5)
axE.set_xticks(X)
axE.set_xticklabels(["Global" if i == "GLB" else i for i in ORDER], fontsize=6.6)
for t, iso in zip(axE.get_xticklabels(), ORDER):
    t.set_color(PAT[iso])
    t.set_fontweight("bold" if iso == "GLB" else "normal")
axE.set_yticks(Y)
axE.set_yticklabels(FAC, fontsize=6.8)
for t in axE.get_yticklabels()[:3]:
    t.set_color("#8a4a12")
    t.set_fontweight("bold")
axE.tick_params(length=0)
axE.grid(color="#eeeeee", lw=0.5, zorder=0)
axE.set_axisbelow(True)
[s_.set_visible(False) for s_ in axE.spines.values()]
axE.set_xlabel(
    "Share of the country's cardiovascular benefit from shifting each factor 20% toward its TMREL",
    fontsize=6.8,
)
leg = [
    Line2D(
        [],
        [],
        marker="o",
        ls="",
        color="grey",
        ms=np.sqrt(v * SC),
        alpha=0.7,
        label=f"{v}%",
    )
    for v in [10, 30, 50]
]
leg += [
    Line2D(
        [],
        [],
        marker="o",
        ls="",
        mfc="#B5651D",
        mec="white",
        ms=6,
        label="In feasible package",
    ),
    Line2D(
        [],
        [],
        marker="o",
        ls="",
        mfc="#00A087",
        mec="white",
        ms=6,
        label="Not in package",
    ),
    Line2D(
        [],
        [],
        marker="o",
        ls="",
        mfc="white",
        mec="black",
        mew=1.0,
        ms=6,
        label="Top priority",
    ),
]
axE.legend(
    handles=leg,
    loc="upper left",
    bbox_to_anchor=(-0.12, -0.12),
    ncol=6,
    fontsize=6.0,
    frameon=False,
    handletextpad=0.4,
    columnspacing=0.8,
    labelspacing=1.2,
)
fig.text(
    0.62,
    0.048,
    "Country code colour (semaglutide patent status)",
    fontsize=6.2,
    fontweight="bold",
)
for x_, (lab, c) in zip(
    (0.62, 0.77, 0.885),
    (
        ("Patent expiry by 2027", "#00A087"),
        ("No patent filed", "#3C5488"),
        ("Patent protected", "#E64B35"),
    ),
):
    fig.text(x_, 0.024, lab, fontsize=5.7, color=c, fontweight="bold")
axE.text(-0.06, 1.04, "e", transform=axE.transAxes, fontsize=11, fontweight="bold")
fig.savefig(R(a.out) + ".pdf")
fig.savefig(R(a.out) + ".png", dpi=400)
rs = {
    c: tuple(round(float(v), 2) for v in rci(cty[c], cty.r20))
    for c in ("asr", "pkg", "ow", "idx")
}
pd.DataFrame(
    [dict(characteristic=c, r=v[0], lo=v[1], hi=v[2]) for c, v in rs.items()]
    + [
        dict(
            characteristic="spearman_ratio20_vs_crossover",
            r=rho,
            lo=np.nan,
            hi=np.nan,
            n=len(rc),
            p=p_rho,
        )
    ]
).to_csv(R(a.out) + "_correlations.csv", index=False)
print("saved:", R(a.out), "| Pearson r (95% CI) with the ratio at 20% coverage:", rs)
print(
    f"Spearman, ratio at 20% vs crossover coverage, {len(rc)} countries reaching crossover: rho = {rho:.2f} (P = {p_rho:.3f})"
)
