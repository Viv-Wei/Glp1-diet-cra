import argparse, logging, os
import numpy as np, pandas as pd, matplotlib

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

p = argparse.ArgumentParser()
p.add_argument("--root", default=".")
p.add_argument("--out", default="data/figures/figure5_robustness")
a = p.parse_args()
R = lambda f: os.path.join(a.root, f)
cr = pd.read_csv(R("data/results/v3/mc/summary_crossover_ratios.csv")).set_index("iso3")
sc = pd.read_csv(R("data/results/v3/mc_scen/summary_scenarios.csv"))
d3 = pd.read_csv(R("data/results/v3/mc_scen/summary_scenarios_disc3.csv"))
V = sc.pivot(index="scenario", columns="iso3", values="r20_median")
VL = sc.pivot(index="scenario", columns="iso3", values="r20_lo")
VH = sc.pivot(index="scenario", columns="iso3", values="r20_hi")
_d3 = d3[d3.scenario == "main"].set_index("iso3")
V.loc["disc3"], VL.loc["disc3"], VH.loc["disc3"] = (
    _d3.r20_median,
    _d3.r20_lo,
    _d3.r20_hi,
)
ORDER = ["GLB"] + cr.drop("GLB").sort_values("xext_median_%").index.tolist()
PAT = {
    **{k: "#00A087" for k in "BGR ROU IND MEX ZAF CHN TUR BRA NZL CAN AUS ISR".split()},
    **{k: "#3C5488" for k in "EGY BGD NGA".split()},
    **{k: "#E64B35" for k in "USA DEU GBR JPN".split()},
    "GLB": "black",
}

SECTIONS = [
    (
        "Conditions",
        [
            (
                "Treatment persistence",
                "#B5651D",
                [
                    ("persist_full", "Nobody stops treatment"),
                    ("stop_x0.5", "Stopping half as often"),
                    ("stop_x1.5", "Stopping 1.5 times as often"),
                    ("no_restart", "No restarting after stopping"),
                    ("restart_late0", "No restarting after two years off"),
                    ("resid_0", "No benefit retained after stopping"),
                    ("const_retention", "Fixed retention of two thirds of the effect"),
                ],
            ),
            (
                "Effects after disease onset",
                "#8E1B12",
                [
                    ("post_A", "No treatment effect on deaths after onset"),
                    ("post_C", "Nor on fatal first events"),
                    ("diet_post_na", "Dietary sodium effect also after onset"),
                    ("diet_post_na_wg", "Sodium and whole grain effects also after onset"),
                ],
            ),
            (
                "Dietary policy implementation",
                "#00A087",
                [
                    ("maintain_0.8", "Policy effect held at 80%"),
                    ("maintain_0.6", "Policy effect held at 60%"),
                    ("maintain_0.4", "Policy effect held at 40%"),
                    ("phase_12", "Phase-in over 12 years"),
                    ("phase_16", "Phase-in over 16 years"),
                    (
                        "sw_pm0.4_stop0.0_rd16",
                        "40% effect, 16-year phase-in, no stopping",
                    ),
                ],
            ),
            (
                "Scope of dietary policy",
                "#2F8F6F",
                [
                    ("sodium30", "Sodium reduction of 30%"),
                    ("tailored", "Package tailored to each country"),
                    ("aspirational", "Aspirational scenario"),
                ],
            ),
            (
                "Eligible population and programme",
                "#E64B35",
                [
                    ("tier3_bmi27", "All adults with BMI ≥27 eligible"),
                    ("bmi30", "Eligibility restricted to BMI ≥30"),
                    ("bmi25", "Eligibility extended to BMI ≥25"),
                    ("uniform", "Places allocated uniformly"),
                    ("no_dm_tier", "No priority for diabetes"),
                ],
            ),
        ],
    ),
    (
        "Sensitivity analyses",
        [
            (
                "Trial effect estimates",
                "#3C5488",
                [
                    ("class_overall", "Class-level estimate for everyone"),
                    ("select_only", "SELECT only, without diabetes"),
                    ("sattar_dm", "Earlier meta-analysis, with diabetes"),
                    ("tier2_rewind", "REWIND, diabetes without CVD"),
                    ("primary_weak", "40% of the effect without CVD"),
                    ("tier3_none", "No effect without CVD or diabetes"),
                    ("itt_select", "Correction for discontinuation in trials"),
                ],
            ),
            (
                "Model structure",
                "#7F7F7F",
                [
                    ("s0.25_same", "Fatal share of first IHD events 25%"),
                    ("s0.55_same", "Fatal share of first IHD events 55%"),
                    ("s0.4_none", "No effect on fatal first events"),
                    ("dm_no_risk", "Diabetes does not raise cardiovascular risk"),
                    ("dm_pr2", "Diabetes prevalence ratio 2"),
                    ("dm_pr3", "Diabetes prevalence ratio 3"),
                    ("bmi_gamma", "Gamma distribution of BMI"),
                ],
            ),
            (
                "Population and time",
                "#6A5ACD",
                [
                    ("open_pop", "Open population (new adults enter)"),
                    ("disc3", "Discounting 3% a year"),
                ],
            ),
        ],
    ),
]
rows, y = [], 0
rows.append(("main", "Main analysis (median, 95% UI)", "#1a1a1a", "main", y))
y += 1
for sec, groups in SECTIONS:
    y += 0.4
    rows.append(("section", sec, "black", None, y))
    y += 1
    for grp, col, items in groups:
        rows.append(("group", grp, col, None, y))
        y += 1
        for scn, lab in items:
            if scn in V.index:
                rows.append(("row", lab, col, scn, y))
                y += 1
H = y
mr, mlo, mhi = cr.loc["GLB", ["r20_median", "r20_lo", "r20_hi"]]
V.loc["main_mc"] = cr.r20_median

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 6.6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "pdf.fonttype": 42,
    }
)
fig = plt.figure(figsize=(7.4, 8.4))
ax = fig.add_axes([0.265, 0.085, 0.195, 0.855])
hx = fig.add_axes([0.588, 0.085, 0.408, 0.855])
XL = (0.2, 60)
ax.set_xscale("log")
ax.set_xlim(*XL)
ax.set_ylim(H - 0.4, -0.7)
ax.axvspan(XL[0], 1, color="#E64B35", alpha=0.06, lw=0)
ax.axvspan(mlo, mhi, color="grey", alpha=0.12, lw=0)
ax.axvline(1, color="black", lw=0.8)
ax.axvline(mr, color="grey", lw=0.7, ls=(0, (3, 2)))
to_fig = (
    lambda yy: ax.transData.transform((1, yy))[1] / fig.dpi / fig.get_size_inches()[1]
)
for kind, txt, col, scn, yy in rows:
    ty = to_fig(yy)
    if kind == "section":
        fig.text(0.008, ty, txt, fontsize=7.4, fontweight="bold", va="center")
        continue
    if kind == "group":
        fig.text(
            0.015, ty, txt, fontsize=6.7, fontweight="bold", color=col, va="center"
        )
        continue
    if kind == "main":
        ax.plot([mlo, mhi], [yy, yy], color=col, lw=1.2, zorder=3)
        ax.scatter(mr, yy, marker="D", s=34, color=col, zorder=5)
        val = f"{mr:.2f} ({mlo:.2f}–{mhi:.2f})"
    else:
        v, lo_, hi_ = (float(M.loc[scn, "GLB"]) for M in (V, VL, VH))
        ax.plot(
            [max(lo_, XL[0]), min(hi_, XL[1])],
            [yy, yy],
            color=col,
            lw=0.9,
            zorder=2,
            solid_capstyle="round",
        )
        if hi_ > XL[1]:
            ax.annotate(
                "",
                xy=(XL[1], yy),
                xytext=(XL[1] * 0.8, yy),
                arrowprops=dict(arrowstyle="->", lw=0.7, color=col),
            )
        ax.scatter(
            min(max(v, XL[0]), XL[1]),
            yy,
            s=22,
            color=col,
            edgecolor="white",
            lw=0.5,
            zorder=4,
        )
        f_ = lambda z: f"{z:.2f}" if z < 10 else f"{z:.1f}" if z < 100 else f"{z:.0f}"
        val = f"{f_(v)} ({f_(lo_)}–{f_(hi_)})"
    fig.text(
        0.025,
        ty,
        txt,
        fontsize=6.3,
        va="center",
        fontweight="bold" if kind == "main" else "normal",
    )
    fig.text(
        0.468,
        ty,
        val,
        fontsize=5.6,
        va="center",
        fontweight="bold" if kind == "main" else "normal",
    )
ax.text(
    0.97,
    H - 0.7,
    "GLP-1RA\ntreatment\naverts more",
    ha="right",
    va="bottom",
    fontsize=5.9,
    color="#b23a29",
    style="italic",
)
ax.text(
    28,
    H - 0.7,
    "Dietary\npolicy\naverts more",
    ha="right",
    va="bottom",
    fontsize=5.9,
    color="#00795f",
    style="italic",
)
ticks = [0.3, 1, 3, 10, 30]
ax.set_xticks(ticks)
ax.set_xticklabels([f"{t:g}" for t in ticks])
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel(
    "Global ratio of deaths averted,\ndietary policy / GLP-1RA\ntreatment at 20% coverage",
    fontsize=6.4,
)
ax.set_yticks([])
ax.spines[["left", "right", "top"]].set_visible(False)
top = ax.transAxes.transform((0, 1))[1] / fig.dpi / fig.get_size_inches()[1]
fig.text(
    0.468,
    top + 0.004,
    "Global ratio\n(95% UI)",
    fontsize=6.3,
    fontweight="bold",
    va="bottom",
)
fig.text(0.008, top + 0.004, "a", fontsize=11, fontweight="bold", va="bottom")

cmap = LinearSegmentedColormap.from_list(
    "rd", ["#8E0F1E", "#D6604D", "#F6C3B5", "#F7F7F7", "#C7E9DA", "#4DB38F", "#00523F"]
)
norm = TwoSlopeNorm(vmin=np.log10(0.1), vcenter=0, vmax=np.log10(10))
hx.set_xlim(-0.5, len(ORDER) - 0.5)
hx.set_ylim(H - 0.4, -0.7)
for kind, txt, col, scn, yy in rows:
    if kind not in ("row", "main"):
        continue
    key = "main_mc" if kind == "main" else scn
    for j, iso in enumerate(ORDER):
        v = V.loc[key, iso] if iso in V.columns else np.nan
        if pd.isna(v):
            continue
        hx.add_patch(
            plt.Rectangle(
                (j - 0.5, yy - 0.5),
                1,
                1,
                facecolor=cmap(norm(np.log10(np.clip(v, 0.1, 10)))),
                edgecolor="white",
                lw=0.5,
            )
        )
        tc = "white" if abs(np.log10(max(v, 1e-3))) > 0.6 else "#222222"
        hx.text(
            j,
            yy,
            f"{v:.2f}" if v < 10 else f"{v:.0f}",
            ha="center",
            va="center",
            fontsize=4.5,
            color=tc,
            fontweight="bold" if kind == "main" else "normal",
        )
hx.axvline(0.5, color="white", lw=2)
hx.set_xticks(range(len(ORDER)))
hx.set_xticklabels(
    ["Global" if i == "GLB" else i for i in ORDER], rotation=90, fontsize=6.2
)
hx.xaxis.tick_top()
for t, iso in zip(hx.get_xticklabels(), ORDER):
    t.set_color(PAT[iso])
    t.set_fontweight("bold" if iso == "GLB" else "normal")
hx.set_yticks([])
hx.tick_params(length=0)
[s_.set_visible(False) for s_ in hx.spines.values()]
hx.text(
    -0.06,
    1.0 + 0.004 * 8.4 / 8.4,
    "b",
    transform=hx.transAxes,
    fontsize=11,
    fontweight="bold",
    va="bottom",
)
cax = fig.add_axes([0.62, 0.042, 0.32, 0.011])
cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal"
)
tv = [0.1, 0.3, 1, 3, 10]
cb.set_ticks(np.log10(tv))
cb.set_ticklabels([f"{v:g}" for v in tv])
cb.outline.set_visible(False)
cax.tick_params(labelsize=6, length=2)
fig.text(
    0.62, 0.058, "◀ Treatment averts more", fontsize=5.8, color="#b23a29", ha="left"
)
fig.text(
    0.94,
    0.058,
    "Dietary policy averts more ▶",
    fontsize=5.8,
    color="#00795f",
    ha="right",
)
fig.text(
    0.565,
    0.016,
    "Country code colour (semaglutide patent status):",
    fontsize=5.8,
    fontweight="bold",
)
for x_, (lab, c) in zip(
    (0.565, 0.70, 0.80),
    (
        ("Patent expiry by 2027", "#00A087"),
        ("No patent filed", "#3C5488"),
        ("Patent protected", "#E64B35"),
    ),
):
    fig.text(x_, 0.002, lab, fontsize=5.6, color=c, fontweight="bold")
os.makedirs(os.path.dirname(R(a.out)) or ".", exist_ok=True)
out = pd.DataFrame(
    [
        dict(
            section_group=txt if kind != "row" else "",
            label=txt,
            scenario=scn,
            **{iso: V.loc["main_mc" if kind == "main" else scn, iso] for iso in ORDER},
        )
        for kind, txt, col, scn, yy in rows
        if kind in ("row", "main")
    ]
)
out.round(3).to_csv(R(a.out) + "_data.csv", index=False)
fig.savefig(R(a.out) + ".pdf")
fig.savefig(R(a.out) + ".png", dpi=400)
print("saved:", R(a.out), "|", len([r for r in rows if r[0] == "row"]), "scenario rows")
