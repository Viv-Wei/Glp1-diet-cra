import argparse, logging, os
import numpy as np, pandas as pd, matplotlib

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D

p = argparse.ArgumentParser()
p.add_argument("--root", default=".")
p.add_argument("--out", default="data/figures/suppfig_t2d_ckd")
p.add_argument("--curves", default="data/results/v3/mc_t2dckd/outcome_curves.csv")
p.add_argument(
    "--traj", default="data/results/v3/mc_traj_t2dckd/trajectories_all_outcomes.csv"
)
p.add_argument(
    "--cross", default="data/results/v3/mc_t2dckd/summary_outcome_crossover.csv"
)
p.add_argument("--mc", default="data/results/v3/mc")
a = p.parse_args()
R = lambda f: os.path.join(a.root, f)
tr = pd.read_csv(R(a.traj))
xo = pd.read_csv(R(a.cross))
cu = pd.read_csv(R(a.curves))
NE = {
    oc: set(g.groupby("iso3").diet_median.max().pipe(lambda v: v[v <= 0].index))
    for oc, g in cu.groupby("outcome")
}
cr = pd.read_csv(R(f"{a.mc}/summary_crossover_ratios.csv")).set_index("iso3")
ORDER = ["GLB"] + cr.drop("GLB").sort_values("xext_median_%").index.tolist()
PAT = {
    **{k: "#00A087" for k in "BGR ROU IND MEX ZAF CHN TUR BRA NZL CAN AUS ISR".split()},
    **{k: "#3C5488" for k in "EGY BGD NGA".split()},
    **{k: "#E64B35" for k in "USA DEU GBR JPN".split()},
    "GLB": "#1a1a1a",
}
C_DIET, C_ASP = "#00A087", "#7FCDBB"
C_GLP = {"c0.10": "#F4A582", "c0.20": "#E64B35", "cafford": "#8E1B12"}
LAB_GLP = {
    "c0.10": "GLP-1RA, 10% coverage",
    "c0.20": "GLP-1RA, 20% coverage",
    "cafford": "GLP-1RA, maximum affordable coverage",
}
plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.4,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    }
)


def traj(ax, outcome, ylab):
    g = tr[(tr.outcome == outcome) & (tr.iso3 == "GLB")]

    def series(run, arm):
        s = g[(g.run == run) & (g.arm == arm)].sort_values("year")
        return (
            s.year.values,
            s["median"].values / 1e3,
            s.lo.values / 1e3,
            s.hi.values / 1e3,
        )

    for run, arm, col, ls, lw, lab in [
        ("asp", "diet", C_ASP, (0, (4, 2)), 1.2, "Dietary policy, aspirational"),
        ("c0.20", "diet", C_DIET, "-", 1.8, "Dietary policy, feasible package"),
    ] + [
        (r, "glp1", C_GLP[r], "-", 1.6 if r == "c0.20" else 1.1, LAB_GLP[r])
        for r in ("cafford", "c0.20", "c0.10")
    ]:
        x, m, lo, hi = series(run, arm)
        if not len(x):
            continue
        ax.fill_between(x, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(x, m, color=col, ls=ls, lw=lw, label=lab)
    xd, dm, _, _ = series("c0.20", "diet")
    xg, gm, _, _ = series("c0.20", "glp1")
    k = np.where(dm > gm)[0] if len(dm) and len(gm) else []
    if len(k):
        yv = int(xd[k[0]])
        ax.axvline(yv, color="grey", lw=0.5, ls=":")
        ax.annotate(
            f"Feasible package overtakes\n20% coverage in year {yv}",
            xy=(yv, dm[k[0]]),
            xytext=(yv + 4, dm[k[0]] * 0.45),
            fontsize=6.6,
            arrowprops=dict(arrowstyle="->", lw=0.6),
        )
    else:
        ax.text(
            0.97,
            0.05,
            "Feasible package does not overtake\n20% coverage within 30 years",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.6,
            color="#444444",
        )
    ax.set_xlim(1, 30)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_ylim(0, None)
    ax.set_xlabel("Year of projection")
    ax.set_ylabel(ylab)
    ax.spines[["top", "right"]].set_visible(False)


NR0, NRX, NRT, XR = 104, 108, 113, 138


def cross(ax, outcome):
    d = xo[xo.outcome == outcome].set_index("iso3")
    y = {iso: len(ORDER) - 1 - i for i, iso in enumerate(ORDER)}
    ax.axvspan(0, 20, color="#E64B35", alpha=0.06, lw=0)
    ax.axvspan(101.5, XR, color="#7f7f7f", alpha=0.07, lw=0)
    for v in (10, 20):
        ax.axvline(v, color="grey", lw=0.6, ls=(0, (3, 2)))
    ax.axvline(100, color="#555555", lw=0.8)
    for iso in ORDER:
        if iso not in d.index:
            continue
        if iso in NE.get(outcome, set()):
            ax.text(
                2,
                y[iso],
                "Not estimable",
                fontsize=6.2,
                color="grey",
                va="center",
                style="italic",
            )
            continue
        m, lo, hi = d.loc[iso, ["xext_median_pct", "xext_lo_pct", "xext_hi_pct"]]
        c = PAT[iso]
        yy = y[iso]
        if lo <= 100:
            ax.plot([lo, min(hi, 100)], [yy, yy], color=c, lw=0.9, zorder=3)
            if hi > 100:
                ax.annotate(
                    "",
                    xy=(NR0 - 0.5, yy),
                    xytext=(100, yy),
                    arrowprops=dict(arrowstyle="->", lw=0.6, color=c),
                    annotation_clip=False,
                )
        if m <= 100:
            ax.scatter(
                m,
                yy,
                s=26 if iso == "GLB" else 18,
                color=c,
                edgecolor="white",
                lw=0.5,
                zorder=5,
            )
        else:
            ax.scatter(
                NRX,
                yy,
                s=26 if iso == "GLB" else 18,
                color=c,
                edgecolor="white",
                lw=0.5,
                zorder=5,
            )
            ax.text(
                NRT,
                yy,
                f"{1e4 / m:.0f}%",
                va="center",
                fontsize=6,
                color=c,
                fontweight="bold" if iso == "GLB" else "normal",
            )
    ax.axhline(len(ORDER) - 1.5, color="#cccccc", lw=0.6)
    ax.set_yticks([y[i] for i in ORDER])
    ax.set_yticklabels(["Global" if i == "GLB" else i for i in ORDER])
    for t, iso in zip(ax.get_yticklabels(), ORDER):
        t.set_color(PAT[iso])
        t.set_fontweight("bold" if iso == "GLB" else "normal")
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlim(0, XR)
    ax.set_ylim(-0.7, len(ORDER) + 0.3)
    ax.tick_params(axis="y", length=0)
    ax.text(
        (101.5 + XR) / 2,
        len(ORDER) - 0.35,
        "Not reached\nby 100%",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="grey",
    )
    ax.set_xlabel("Crossover coverage (% of eligible adults)")
    ax.spines[["top", "right"]].set_visible(False)


fig = plt.figure(figsize=(7.4, 8.6))
aT = fig.add_axes([0.085, 0.60, 0.38, 0.34])
aC = fig.add_axes([0.585, 0.60, 0.38, 0.34])
cT = fig.add_axes([0.085, 0.08, 0.38, 0.40])
cC = fig.add_axes([0.585, 0.08, 0.38, 0.40])
traj(aT, "T2D", "Global incident T2D cases averted per year (thousands)")
traj(aC, "CKD", "Global CKD deaths averted per year (thousands)")
cross(cT, "T2D")
cross(cC, "CKD")
aT.set_title(
    "Type 2 diabetes (incident cases)", fontsize=8.2, fontweight="bold", loc="left"
)
aC.set_title(
    "Chronic kidney disease (deaths)", fontsize=8.2, fontweight="bold", loc="left"
)
for ax, tag in [(aT, "a"), (aC, "b"), (cT, "c"), (cC, "d")]:
    ax.text(-0.2, 1.04, tag, transform=ax.transAxes, fontsize=12, fontweight="bold")
h, l = aT.get_legend_handles_labels()
fig.legend(
    h,
    l,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.555),
    ncol=3,
    fontsize=6.6,
    frameon=False,
    handlelength=2.2,
)
fig.text(
    0.085,
    0.032,
    "Not reached by 100%: cases (c) or deaths (d) averted by treatment at 100% coverage as a percentage of those "
    "averted by the feasible package",
    fontsize=6.2,
    color="#444444",
)
fig.text(
    0.085,
    0.012,
    "Country code colour (semaglutide patent status)",
    fontsize=6.4,
    fontweight="bold",
)
for x_, (lab, c) in zip(
    (0.45, 0.61, 0.74),
    (
        ("Patent expiry by 2027", "#00A087"),
        ("No patent filed", "#3C5488"),
        ("Patent protected", "#E64B35"),
    ),
):
    fig.text(x_, 0.012, lab, fontsize=6.2, color=c, fontweight="bold")
os.makedirs(os.path.dirname(R(a.out)) or ".", exist_ok=True)
fig.savefig(R(a.out) + ".pdf")
fig.savefig(R(a.out) + ".png", dpi=400)
print("saved:", R(a.out))
