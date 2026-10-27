import argparse, os
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

p = argparse.ArgumentParser()
p.add_argument("--root", default=".")
p.add_argument("--xlsx", default="data/tables/Supplementary_Tables_v5.xlsx")
p.add_argument("--t-paf", type=int, default=14)
p.add_argument("--t-cost", type=int, default=17)
p.add_argument("--equity", default="data/results/v3/equity_cost_final/cov0.20")
p.add_argument("--mc", default="data/results/v3/mc")
p.add_argument("--budget", default="data/results/v3/expenditure_ui/budget_ui.csv")
p.add_argument("--equity-mc", default="data/results/v3/equity_mc")
p.add_argument("--out", default="data/figures/figure6_equity")
a = p.parse_args()
R = lambda f: os.path.join(a.root, f)
tab = lambda n: pd.read_excel(R(a.xlsx), f"Supplementary Table {n}", header=1).iloc[:-1]

t13, t18 = tab(a.t_paf), tab(a.t_cost)
t16 = pd.read_csv(R(f"{a.equity}/tableS_equity_v3.csv"))
# Monte Carlo medians and 95% UIs (first 300 simulations): pooled over the 19 countries, and by country
_ren = {"diet_median": "diet", "glp1_median": "glp1"}
POOL = pd.read_csv(R(f"{a.equity_mc}/pooled_ui.csv")).rename(columns=_ren)
EQUI = pd.read_csv(R(f"{a.equity_mc}/equity_ui.csv")).rename(columns=_ren)
MAPU = pd.read_csv(R(f"{a.equity_mc}/mapping_ui.csv"))
_cr = pd.read_csv(R(f"{a.mc}/summary_crossover_ratios.csv")).set_index("iso3")
ORDER = _cr.drop("GLB").sort_values("xext_median_%").index.tolist()
num = lambda s: pd.to_numeric(s, errors="coerce")

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
GRP = {
    **{k: "#00A087" for k in "BGR ROU IND MEX ZAF CHN TUR BRA NZL CAN AUS ISR".split()},
    **{k: "#3C5488" for k in "EGY BGD NGA".split()},
    **{k: "#E64B35" for k in "USA DEU GBR JPN".split()},
}
fig = plt.figure(figsize=(7.4, 5.3))
axA = fig.add_axes([0.07, 0.22, 0.175, 0.71])
axB = fig.add_axes([0.265, 0.22, 0.205, 0.71])
axC = fig.add_axes([0.53, 0.22, 0.19, 0.71])         # panel c: deaths averted by education group
axD = fig.add_axes([0.815, 0.22, 0.16, 0.71])        # panel d: cost threshold, with its own country labels
y = np.arange(len(ORDER))[::-1]
Y = dict(zip(ORDER, y))

OUT = [
    ("IHD", "IHD", "o", "#3C5488", 0.22),
    ("IschStroke", "Ischaemic stroke", "s", "#8491B4", 0.0),
    ("T2D", "Type 2 diabetes", "^", "#F39B7F", -0.22),
]
for code, lab, mk, col, dy in OUT:
    x = t13[t13.Outcome == code].set_index("ISO3")
    for iso in ORDER:
        if iso not in x.index:
            continue
        r_ = x.loc[iso]
        m, lo, hi = (
            num(r_["Low/high education PAF ratio"] if "Low/high education PAF ratio" in r_ else r_["Low/high education PAF ratio, median"]),
            num(r_["95% UI lower bound"]),
            num(r_["95% UI upper bound"]),
        )
        if pd.isna(m):
            continue
        if not (pd.isna(lo) or pd.isna(hi)):
            axA.plot(
                [lo, min(hi, 2.6)], [Y[iso] + dy] * 2, color=col, lw=0.7, alpha=0.8
            )
        axA.scatter(
            m,
            Y[iso] + dy,
            marker=mk,
            s=14,
            color=col,
            edgecolor="white",
            lw=0.3,
            zorder=3,
        )
axA.axvline(1, color="black", lw=0.6)
axA.set_xlim(0.7, 2.6)
axA.set_xticks([1, 1.5, 2, 2.5])
axA.set_xlabel("Diet-attributable PAF,\nlow / high education")
axA.text(
    1.03,
    len(ORDER) - 0.15,
    "Higher in low-education\ngroups →",
    fontsize=5.8,
    color="grey",
    va="bottom",
)
axA.legend(
    handles=[
        Line2D([], [], marker=mk, ls="", color=col, ms=4, label=lab)
        for _, lab, mk, col, _ in OUT
    ],
    loc="upper center",
    bbox_to_anchor=(0.4, -0.13),
    ncol=1,
    fontsize=5.8,
    frameon=False,
    handletextpad=0.2,
    columnspacing=0.6,
)

QC = {
    "Q1": "#67000D",
    "Q2": "#CB181D",
    "Q3": "#FB6A4A",
    "Q4": "#FCAE91",
    "Q5": "#FEE5D9",
}
b = t18.set_index(["ISO3", "Income quintile"])
cb, cg = (
    "Branded cost as share of income (%)",
    "Generic upper bound ($140/yr) as share of income (%)",
)
axB.axvspan(10, 2000, color="#E64B35", alpha=0.05, lw=0)
for v, lab, ha in [(10, "10% of\nincome", "right"), (100, "Entire\nincome", "left")]:
    axB.axvline(v, color="grey", lw=0.6, ls=(0, (3, 2)))
    axB.text(
        v * (0.9 if ha == "right" else 1.1),
        len(ORDER) - 0.25,
        lab,
        fontsize=5.8,
        color="grey",
        va="bottom",
        ha=ha,
    )
for iso in ORDER:
    vb = [b.loc[(iso, q), cb] for q in QC]
    vg = [b.loc[(iso, q), cg] for q in QC]
    for vals, dy, filled in [(vb, 0.17, True), (vg, -0.17, False)]:
        axB.plot(
            [min(vals), max(vals)], [Y[iso] + dy] * 2, color="#bbbbbb", lw=0.6, zorder=1
        )
        for q, v in zip(QC, vals):
            axB.scatter(
                v,
                Y[iso] + dy,
                s=13,
                marker="o" if filled else "D",
                facecolor=QC[q] if filled else "white",
                edgecolor=QC[q] if not filled else "#555555",
                lw=0.6 if not filled else 0.25,
                zorder=3,
            )
axB.set_xscale("log")
axB.set_xlim(0.05, 1000)
axB.set_xticks([0.1, 1, 10, 100, 1000])
axB.set_xticklabels(["0.1", "1", "10", "100", "1000"])
axB.set_xlabel("Annual treatment cost\n(% of per-capita income, log scale)")
QLAB = {q: q for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]}
axB.legend(
    handles=[
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc=QC[q],
            mec="#555555",
            mew=0.25,
            ms=4,
            label=QLAB[q],
        )
        for q in QC
    ]
    + [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            mfc="#999999",
            mec="#555555",
            ms=4,
            label="Branded",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            mfc="white",
            mec="#999999",
            ms=3.5,
            label="Generic US$140",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=3,
    fontsize=5.6,
    frameon=False,
    handletextpad=0.1,
    columnspacing=0.7,
    title="Income quintile (Q1 poorest)",
    title_fontsize=5.8,
)

for ax in (axA, axB):
    ax.set_yticks(y)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color="#f2f2f2", lw=0.5)
    ax.set_axisbelow(True)
axA.set_yticklabels(ORDER, fontsize=6.6)
for t_ in axA.get_yticklabels():
    t_.set_color(GRP[t_.get_text()])
axB.set_yticklabels([])

EDU = [
    ("Low education", "Low education", "#3F2D76"),
    ("Medium education", "Medium education", "#7B6BB0"),
    ("High education", "High education", "#B7AED6"),
]
PRICES = [
    ("Branded", "Branded"),
    ("Generic $140/yr", "Generic\nUS$140"),
    ("Generic $28/yr", "Generic\nUS$28"),
]


SHOW_UI = False


def draw_slope(ax, d, scale, small=False, label=False):

    g = d.set_index(["Price scenario", "Education group"])[["diet", "diet_lo", "diet_hi", "glp1", "glp1_lo", "glp1_hi"]] * scale
    X = np.arange(len(PRICES))
    for k_, (edu, lab, col) in enumerate(EDU):
        off_ = (k_ - 1) * (0.09 if not small else 0.12)                     # interval offsets by education group
        tv = [g.loc[(p, edu), "glp1"] for p, _ in PRICES]
        dv = g.loc[("Branded", edu), "diet"]
        # 95% uncertainty intervals: treatment at each price, dietary policy (the same at every price) at the right;
        # not drawn (the intervals overlap too much to read); they are reported in Supplementary Table 18
        if SHOW_UI and not small:
          ax.vlines(X + off_, [g.loc[(p, edu), "glp1_lo"] for p, _ in PRICES], [g.loc[(p, edu), "glp1_hi"] for p, _ in PRICES],
                  color=col, lw=0.6 if small else 0.8, alpha=0.8, zorder=2)
          ax.vlines(X[-1] + 0.25 + 0.1 * k_, g.loc[("Branded", edu), "diet_lo"], g.loc[("Branded", edu), "diet_hi"],
                  color=col, lw=0.6 if small else 0.8, ls=(0, (2, 1)), alpha=0.8, zorder=2)
        ax.plot(
            X,
            [dv] * len(X),
            color=col,
            lw=1.0 if not small else 0.8,
            ls=(0, (4, 2)),
            marker="D",
            ms=4.2 if not small else 2.6,
            mfc=col,
            mec="white",
            mew=0.4,
        )
        ax.plot(
            X,
            tv,
            color=col,
            lw=1.6 if not small else 1.0,
            marker="o",
            ms=4.5 if not small else 2.8,
            mec="white",
            mew=0.4,
            label=lab,
        )
        if label:
            off = 0.0
            ax.text(
                X[-1] + 0.52,
                dv + off,
                f"{lab.split()[0]} {dv:.2f}",
                fontsize=5.4,
                color=col,
                va="center",
            )
            if tv[0] >= 0.005:                                         # a zero label would overprint the next one
              ax.text(
                X[0] - 0.1,
                tv[0],
                f"{tv[0]:.2f}",
                fontsize=5.6,
                color=col,
                ha="right",
                va="center",
            )
    if label:
        ax.text(
            X[-1] + 0.12, max(g["glp1"]) * 1.0, "", fontsize=5
        )
    ax.set_xticks(X)
    ax.set_xticklabels(
        [l for _, l in PRICES] if not small else ["Branded", "US$140", "US$28"],
        fontsize=5.6 if not small else 5.6,
    )
    ax.set_xlim(-0.75 if label else -0.4, len(PRICES) - 1 + (1.5 if label else 0.3))
    ax.set_ylim(0, None)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#f0f0f0", lw=0.5)
    ax.set_axisbelow(True)



# d: cost threshold, the factor by which the cost of the dietary package could rise before its expenditure per death
# averted equalled that of GLP-1RA treatment at 20% coverage (deterministic, equal-budget analysis)
bq = pd.read_csv(R(a.budget))
CL = lambda v: min(max(v, 0.06), 40)
for price, mk, fc, dy in [("Generic $140/yr", "o", "#00A087", 0.15), ("Generic $28/yr", "D", "white", -0.15)]:
    x = bq[bq.price == price].set_index("ISO3")
    for iso in ORDER:
        if iso in x.index and pd.notna(x.loc[iso, "C"]):
            axD.hlines(Y[iso] + dy, CL(x.loc[iso, "C_lo"]), CL(x.loc[iso, "C_hi"]), color="#00A087", lw=0.6, alpha=0.8, zorder=2)
            axD.scatter(CL(x.loc[iso, "C"]), Y[iso] + dy, marker=mk, s=13, color=fc, edgecolor="#00A087", lw=0.6, zorder=3)
axD.axvspan(0.05, 1, color="#FBEFEC", zorder=0)
axD.axvline(1, color="black", lw=0.6)
axD.set_xscale("log"); axD.set_xlim(0.05, 60)
axD.set_xticks([0.1, 1, 10]); axD.set_xticklabels(["0.1", "1", "10"])
axD.set_ylim(axB.get_ylim()); axD.set_yticks(y); axD.set_yticklabels(ORDER, fontsize=6.2)
for lab_ in axD.get_yticklabels(): lab_.set_color(GRP[lab_.get_text()])
for sp in ("top", "right"): axD.spines[sp].set_visible(False)
axD.set_xlabel("Deaths averted per dollar,\ndiet relative to\ntreatment (log scale)")
axD.text(1.15, len(ORDER) - 0.15, "Diet averts\nmore →", fontsize=5.8, color="grey", va="bottom")
axD.legend(handles=[Line2D([], [], marker="o", ls="", color="#00A087", ms=4, label="Generic US$140"),
                    Line2D([], [], marker="D", ls="", mfc="white", mec="#00A087", ms=3.5, label="Generic US$28")],
           loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=5.8, frameon=False, handletextpad=0.2)
draw_slope(axC, POOL, 1e-3, label=True)
axC.set_ylabel("Deaths averted over 30 years, 19 countries combined (millions)")
axC.set_xlabel("GLP-1RA price scenario")
gl = (
    t16.groupby(["Price scenario", "Education group"])[
        "GLP-1RA averted (thousands)"
    ].sum()
    / 1e3
)
r_ = pd.to_numeric(MAPU["ratio_median"], errors="coerce")
axC.text(
    0.03,
    0.72,
    f"Low education, diet-to-\ntreatment ratio at US$140\n{r_.min():.2f}–{r_.max():.2f} across\neducation–income links",
    transform=axC.transAxes,
    ha="left",
    va="top",
    fontsize=5.5,
    color="#444444",
)
LEG_STYLE = [
    Line2D(
        [],
        [],
        color="#777777",
        lw=1.6,
        marker="o",
        ms=4,
        mec="white",
        label="GLP-1RA treatment",
    ),
    Line2D(
        [],
        [],
        color="#777777",
        lw=1.0,
        ls=(0, (4, 2)),
        marker="D",
        ms=3.8,
        mec="white",
        label="Dietary policy (feasible package)",
    ),
]
LEG_C = [
    Line2D([], [], color=c, lw=1.6, marker="o", ms=4, mec="white", label=l)
    for _, l, c in EDU
] + LEG_STYLE
axC.legend(
    handles=LEG_STYLE,
    loc="upper center",
    bbox_to_anchor=(0.45, -0.2),
    ncol=1,
    fontsize=5.8,
    frameon=False,
    handlelength=2.2,
    title="Colour = education group, labelled at right",
    title_fontsize=5.6,
)

for ax, tag, dx in [(axA, "a", -0.36), (axB, "b", -0.08), (axC, "c", -0.28), (axD, "d", -0.3)]:
    ax.text(dx, 1.03, tag, transform=ax.transAxes, fontsize=12, fontweight="bold")
os.makedirs(os.path.dirname(R(a.out)) or ".", exist_ok=True)
fig.savefig(R(a.out) + ".pdf")
fig.savefig(R(a.out) + ".png", dpi=400)
plt.close(fig)

NAME = dict(zip(t18.ISO3, t18.Country))
fig, axs = plt.subplots(
    5, 4, figsize=(7.4, 8.6), gridspec_kw=dict(hspace=0.6, wspace=0.35)
)
for ax, iso in zip(axs.flat, ORDER):
    draw_slope(ax, EQUI[EQUI.ISO3 == iso], 1.0, small=True)
    ax.set_title(NAME.get(iso, iso), fontsize=7, pad=3)
    ax.tick_params(axis="x", labelsize=5.8)
leg_ax = axs.flat[-1]
leg_ax.axis("off")
leg_ax.legend(
    handles=LEG_C,
    loc="center",
    fontsize=5.8,
    frameon=False,
    title="Education groups placed on the\nincome " "percentile scale\nin rank order",
    title_fontsize=6.2,
)
fig.supylabel("Deaths averted over 30 years (thousands)", fontsize=7.2, x=0.02)
fig.supxlabel("GLP-1RA price scenario", fontsize=7.2, y=0.035)
fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.07)
fig.savefig(R(a.out) + "_countries.pdf")
fig.savefig(R(a.out) + "_countries.png", dpi=300)
print("saved:", R(a.out), "+ _countries")
