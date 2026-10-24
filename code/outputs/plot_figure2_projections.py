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
p.add_argument("--xlsx", default="data/tables/Supplementary_Tables_final1013_HR.xlsx")
p.add_argument("--no-supp", action="store_true")
p.add_argument(
    "--outcome",
    default="IHD+IS",
    choices=["IHD+IS", "T2D", "CKD"],
    help="T2D or CKD: country small multiples of annual trajectories only (secondary outcomes)",
)
p.add_argument(
    "--traj-all", default="data/results/v3/mc_traj_t2dckd/trajectories_all_outcomes.csv"
)
p.add_argument(
    "--cum-all", default="data/results/v3/mc_traj_t2dckd/cumulative_by_outcome.csv"
)
p.add_argument(
    "--ovl-all", default="data/results/v3/mc_traj_t2dckd/overlap_by_outcome.csv"
)
p.add_argument("--out", default="data/figures/figure2_projections")
a = p.parse_args()
R = lambda f: os.path.join(a.root, f)

RUN = {
    "c0.10": "10%",
    "c0.20": "20%",
    "cafford": "Maximum affordable",
    "asp": "Aspirational 20%",
}
if a.outcome == "IHD+IS":
    yr = pd.read_csv(R("data/results/v3/mc_traj/trajectories.csv")).rename(
        columns={"median": "med"}
    )
else:
    yr = pd.read_csv(R(a.traj_all))
    yr = (
        yr[yr.outcome == a.outcome]
        .drop(columns="outcome")
        .rename(columns={"median": "med"})
    )
UNIT = {
    "IHD+IS": "Deaths averted",
    "T2D": "Incident T2D cases averted",
    "CKD": "CKD deaths averted",
}[a.outcome]
yr["coverage"] = yr.run.map(RUN)
sd = pd.read_csv(R("data/results/v3/mc/summary_deaths.csv"))
sd["coverage"] = (
    sd["cov"]
    .astype(str)
    .map(
        {
            "0.1": "10%",
            "0.1000": "10%",
            "0.10": "10%",
            "0.2": "20%",
            "0.20": "20%",
            "afford": "Maximum affordable",
        }
    )
)
cr = pd.read_csv(R("data/results/v3/mc/summary_crossover_ratios.csv")).set_index("iso3")
ovl = pd.read_csv(R("data/results/v3/overlap/overlap_by_country.csv"))
asp = pd.read_csv(R("data/results/v3/mc_scen/summary_scenarios.csv"))
asp = asp[asp.scenario == "aspirational"].set_index("iso3")
NAME = dict(
    AUS="Australia",
    BGD="Bangladesh",
    BGR="Bulgaria",
    BRA="Brazil",
    CAN="Canada",
    CHN="China",
    DEU="Germany",
    EGY="Egypt",
    GBR="United Kingdom",
    GLB="Global",
    IND="India",
    ISR="Israel",
    JPN="Japan",
    MEX="Mexico",
    NGA="Nigeria",
    NZL="New Zealand",
    ROU="Romania",
    TUR="Türkiye",
    USA="United States of America",
    ZAF="South Africa",
)
POP = None
if not a.no_supp and os.path.exists(R(a.xlsx)):
    t2 = pd.read_excel(R(a.xlsx), "Supplementary Table 2", header=1).iloc[:-1]
    ihd = t2[t2.Outcome == "Ischaemic heart disease"].set_index("ISO3")
    POP = ihd["Attributable deaths"] / ihd["Crude rate per 100,000"] * 1e5
ORDER = ["GLB"] + cr.drop("GLB").sort_values("xext_median_%").index.tolist()
if a.outcome != "IHD+IS":
    co = pd.read_csv(R(a.cum_all))
    co = co[co.outcome == a.outcome]
    sd = co[co.arm.isin(["diet", "glp1", "combo"])].copy()
    asp = (
        co[co.arm == "asp"]
        .set_index("iso3")
        .rename(
            columns={
                "median": "diet_deaths_median",
                "lo": "diet_deaths_lo",
                "hi": "diet_deaths_hi",
            }
        )
    )
    ovl = pd.read_csv(R(a.ovl_all))
    ovl = ovl[ovl.outcome == a.outcome]

COV = {"10%": "10%", "20%": "20%", "Maximum affordable": "Maximum affordable"}
C_DIET, C_ASP, C_COMBO = "#00A087", "#7CC9B5", "#6A51A3"
C_GLP = {"10%": "#F4A582", "20%": "#E64B35", "Maximum affordable": "#8E1B12"}
plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    }
)


def cum(iso, arm, cov="20%"):

    if arm == "asp":
        x = asp.loc[iso]
        return np.array([x.diet_deaths_median, x.diet_deaths_lo, x.diet_deaths_hi])
    x = sd[(sd.iso3 == iso) & (sd.arm == arm) & (sd.coverage == cov)].iloc[0]
    return np.array([x["median"], x.lo, x.hi])


LINES = [
    (
        "Aspirational 20%",
        "diet",
        C_ASP,
        (0, (4, 2)),
        1.1,
        "Dietary policy, aspirational",
    ),
    ("20%", "diet", C_DIET, "-", 1.6, "Dietary policy, feasible package"),
    (
        "Maximum affordable",
        "glp1",
        C_GLP["Maximum affordable"],
        "-",
        1.1,
        "GLP-1RA, maximum affordable coverage",
    ),
    ("20%", "glp1", C_GLP["20%"], "-", 1.6, "GLP-1RA, 20% coverage"),
    ("10%", "glp1", C_GLP["10%"], "-", 1.1, "GLP-1RA, 10% coverage"),
]


def draw_annual(ax, iso, scale, small=False):
    g = yr[yr.iso3 == iso]
    for cov, arm, col, ls, lw, lab in LINES:
        s = g[(g.coverage == cov) & (g.arm == arm)].sort_values("year")
        ax.fill_between(s.year, s.lo * scale, s.hi * scale, color=col, alpha=0.12, lw=0)
        ax.plot(
            s.year,
            s.med * scale,
            color=col,
            ls=ls,
            lw=lw * (0.75 if small else 1),
            label=lab,
        )
    d = g[(g.coverage == "20%") & (g.arm == "diet")].sort_values("year").med.values
    t = g[(g.coverage == "20%") & (g.arm == "glp1")].sort_values("year").med.values
    hit = np.where(d > t)[0]
    cross = int(hit[0]) + 1 if len(hit) else None
    if cross and small:
        ax.axvline(cross, color="grey", lw=0.5, ls=":")
        late = cross >= 22
        ax.text(
            cross + (-0.6 if late else 0.6),
            0.97,
            f"Year {cross}",
            transform=ax.get_xaxis_transform(),
            fontsize=6.2,
            color="#444444",
            ha="right" if late else "left",
            va="top",
        )
    ax.set_xlim(1, 30)
    ax.set_ylim(0, None)
    ax.spines[["top", "right"]].set_visible(False)
    return cross, d, t


C_ALL = "#3F2D76"


def draw_cumulative(ax, iso, scale, small=False, annotate=False):

    fd, fa = cum(iso, "diet") * scale, cum(iso, "asp") * scale
    ax.axhspan(fd[1], fd[2], color=C_DIET, alpha=0.10, lw=0)
    ax.axhline(fd[0], color=C_DIET, lw=1.2, ls="-", zorder=1)
    ax.axhline(fa[0], color=C_ASP, lw=1.1, ls=(0, (5, 2)), zorder=1)
    if annotate:
        ax.text(
            3.35,
            fd[0],
            f"Feasible package alone ({fd[0]:.2f}M)",
            color="#00795f",
            fontsize=6.8,
            ha="right",
            va="bottom",
        )
        ax.text(
            3.35,
            fa[0],
            f"Aspirational scenario alone ({fa[0]:.2f}M)",
            color="#3f9c86",
            fontsize=6.8,
            ha="right",
            va="bottom",
        )
    hw = 0.09 if not small else 0.13
    cap = 1.5 if small else 2
    for k, lab in enumerate(COV):
        g_ = cum(iso, "glp1", lab) * scale
        o = ovl[(ovl.iso3 == iso) & (ovl.coverage == lab)].set_index("overlap_pct")
        lv = o.combined_median * scale
        c_ = np.array([lv[0], o.combined_lo[0] * scale, o.combined_hi[0] * scale])
        xg, xc = k - 0.14, k + 0.14
        ax.errorbar(
            xg,
            g_[0],
            yerr=[[g_[0] - g_[1]], [g_[2] - g_[0]]],
            fmt="o",
            color=C_GLP[lab],
            ms=3.5 if small else 5,
            elinewidth=0.9,
            capsize=cap,
            zorder=3,
        )
        steps = [lv[0], lv[25], lv[50], lv[75], lv[100]]
        for j in range(4):
            ax.add_patch(
                plt.Rectangle(
                    (xc - hw * 0.8, steps[j + 1]),
                    hw * 1.6,
                    steps[j] - steps[j + 1],
                    facecolor=C_COMBO,
                    alpha=0.62 - 0.12 * j,
                    edgecolor="white",
                    lw=0.4,
                    zorder=2,
                )
            )
        ax.plot(
            [xc - hw, xc + hw],
            [lv[100]] * 2,
            color=C_ALL,
            lw=2.0 if not small else 1.4,
            solid_capstyle="butt",
            zorder=4,
        )
        ax.errorbar(
            xc,
            c_[0],
            yerr=[[c_[0] - c_[1]], [c_[2] - c_[0]]],
            fmt="_",
            color=C_COMBO,
            ms=9 if not small else 6,
            mew=1.4,
            elinewidth=0.8,
            capsize=cap,
            zorder=4,
        )
        if annotate:
            ax.text(
                xg,
                g_[2] * 1.015 + 0.3,
                f"{g_[0]:.2f}",
                ha="center",
                fontsize=6.8,
                color=C_GLP[lab],
            )
            ax.text(
                xc,
                c_[2] * 1.015 + 0.3,
                f"{c_[0]:.2f}",
                ha="center",
                fontsize=6.8,
                color=C_COMBO,
            )
            if lab == "Maximum affordable":
                for v, t, dy, col in [
                    (c_[0], "Independent action", 0, C_COMBO),
                    (lv[100], "Complete overlap", -0.9, C_ALL),
                ]:
                    ax.annotate(
                        t,
                        xy=(xc + hw, v),
                        xytext=(xc + 0.2, v + dy),
                        fontsize=6.8,
                        va="center",
                        color=col,
                        arrowprops=dict(arrowstyle="-", lw=0.4, color=col),
                    )
    ax.set_xticks(range(3))
    ax.set_xticklabels(
        ["10%", "20%", "Maximum\naffordable"] if not small else ["10%", "20%", "Max"]
    )
    ax.set_xlim(-0.5, 3.4 if annotate else 2.5)
    ax.set_ylim(0, ax.get_ylim()[1] * (1.38 if annotate else 1))
    ax.spines[["top", "right"]].set_visible(False)


LEG_LINES = [
    Line2D([], [], color=c, ls=ls, lw=lw, label=l) for _, _, c, ls, lw, l in LINES
]
LEG_BARS = [
    Line2D(
        [],
        [],
        marker="o",
        ls="",
        color=C_GLP["20%"],
        ms=5,
        label="GLP-1RA treatment alone (95% UI)",
    ),
    Patch(
        color=C_COMBO,
        alpha=0.4,
        label="Combination: shading lightens as overlap\nof the two effects rises 0% to 100%",
    ),
    Line2D(
        [],
        [],
        color=C_COMBO,
        marker="_",
        ms=9,
        mew=1.4,
        ls="",
        label="Independent action (95% UI)",
    ),
    Line2D([], [], color=C_ALL, lw=2.0, label="Complete overlap (larger arm alone)"),
    Line2D([], [], color=C_DIET, lw=1.2, label="Feasible package alone"),
    Line2D(
        [], [], color=C_ASP, lw=1.1, ls=(0, (5, 2)), label="Aspirational scenario alone"
    ),
]
os.makedirs(os.path.dirname(R(a.out)) or ".", exist_ok=True)

cross = None
if a.outcome == "IHD+IS":
    fig = plt.figure(figsize=(7.4, 3.9))
    axA = fig.add_axes([0.075, 0.14, 0.40, 0.78])
    axB = fig.add_axes([0.58, 0.14, 0.40, 0.78])
    cross, d, t = draw_annual(axA, "GLB", 1e-3)
    axA.annotate(
        f"Feasible package overtakes\n20% coverage in year {cross}",
        xy=(cross, d[cross - 1] * 1e-3),
        xytext=(cross + 4, d[cross - 1] * 1e-3 * 0.3),
        fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.6),
    )
    axA.set_xticks([1, 5, 10, 15, 20, 25, 30])
    axA.set_xlabel("Year of projection")
    axA.set_ylabel("Global deaths averted per year (thousands)")
    axA.legend(
        handles=LEG_LINES,
        loc="upper left",
        fontsize=6.8,
        frameon=False,
        handlelength=2.2,
        borderaxespad=0.2,
    )
    draw_cumulative(axB, "GLB", 1e-6, annotate=True)
    axB.set_xlabel("GLP-1RA treatment coverage (% of eligible adults)", labelpad=3)
    axB.set_ylabel("Global deaths averted over 30 years (millions)")
    axB.legend(
        handles=LEG_BARS[:4],
        loc="upper left",
        fontsize=6.8,
        frameon=False,
        borderaxespad=0.2,
        handlelength=1.8,
    )
    for ax, tag in [(axA, "a"), (axB, "b")]:
        ax.text(
            -0.15, 1.03, tag, transform=ax.transAxes, fontsize=12, fontweight="bold"
        )
    fig.savefig(R(a.out) + ".pdf")
    fig.savefig(R(a.out) + ".png", dpi=400)
    plt.close(fig)

KINDS = ["annual", "cumulative"] if POP is not None else []
SUF = "" if a.outcome == "IHD+IS" else "_" + a.outcome.lower()
for kind in KINDS:
    fig, axs = plt.subplots(
        5, 4, figsize=(7.4, 9.0), gridspec_kw=dict(hspace=0.55, wspace=0.32)
    )
    for ax, iso in zip(axs.flat, ORDER):
        sc = 1e5 / POP[iso]
        if kind == "annual":
            draw_annual(ax, iso, sc, small=True)
            ax.set_xticks([1, 10, 20, 30])
            gd = yr[(yr.iso3 == iso) & (yr.arm == "diet")]
            if len(gd) and gd.med.abs().max() <= 0:
                ax.text(
                    0.5,
                    0.55,
                    "Dietary effect\nnot estimable",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=6,
                    color="grey",
                    style="italic",
                )
        else:
            draw_cumulative(ax, iso, sc, small=True)
        ax.set_title(
            "Global" if iso == "GLB" else NAME.get(iso, iso),
            fontsize=7,
            fontweight="bold" if iso == "GLB" else "normal",
            pad=3,
        )
        ax.tick_params(labelsize=5.8)
    fig.supylabel(
        f"{UNIT} per 100 000 adults aged ≥25 years"
        + (" per year" if kind == "annual" else " over 30 years"),
        fontsize=7.2,
        x=0.02,
    )
    if kind == "annual":
        fig.supxlabel(
            "Year of projection (dotted line and label: year in which the feasible package overtakes 20% coverage;\nnone shown where it does not within 30 years)",
            fontsize=7,
            y=0.065,
        )
        fig.legend(
            handles=LEG_LINES,
            loc="lower center",
            ncol=3,
            fontsize=6.2,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    else:
        fig.supxlabel(
            "GLP-1RA treatment coverage (% of eligible adults)", fontsize=7, y=0.065
        )
        fig.legend(
            handles=LEG_BARS,
            loc="lower center",
            ncol=3,
            fontsize=6.0,
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.13)
    fig.savefig(R(a.out) + f"_countries_{kind}{SUF}.pdf")
    fig.savefig(R(a.out) + f"_countries_{kind}{SUF}.png", dpi=300)
    plt.close(fig)
print(
    "saved:",
    R(a.out),
    "| outcome:",
    a.outcome,
    "| figures:",
    [f"_countries_{k}{SUF}" for k in KINDS],
    "| global crossover year:",
    cross,
)
