import argparse, os
import numpy as np, pandas as pd, matplotlib

matplotlib.use("Agg")
import matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator
import logging

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

p = argparse.ArgumentParser()
p.add_argument("--root", default=".")
p.add_argument("--out", default="data/figures/figure3_threshold")
p.add_argument(
    "--panel-a",
    choices=["curves", "heatmap", "montage"],
    default="montage",
    help="curves or heatmap beside the crossover panel; montage: curves on top, heatmap and crossover below",
)
a = p.parse_args()
R = lambda f: os.path.join(a.root, f)

sd = pd.read_csv(R("data/results/v3/mc/summary_deaths.csv"))
sd = sd[sd["cov"] != "afford"].copy()
sd["cov"] = sd["cov"].astype(float)
CUR = sd.pivot_table(
    index=["iso3", "cov"], columns="arm", values="median"
).reset_index()
CUR = CUR[CUR["cov"] > 0]
CUR["ratio"] = CUR.diet / CUR.glp1
cr = pd.read_csv(R("data/results/v3/mc/summary_crossover_ratios.csv")).set_index("iso3")
sc = pd.read_csv(R("data/results/v3/mc_scen/summary_scenarios.csv"))
xv = sc.pivot(index="iso3", columns="scenario", values="xext_median") * 100
ov = pd.read_csv(R("data/results/v3/overlap/overlap_by_country.csv"))
aff = (
    ov[(ov.coverage == "Maximum affordable") & (ov.overlap_pct == 0)]
    .set_index("iso3")
    .coverage_used_pct
)
d = pd.DataFrame(
    {
        "thr": cr["xext_median_%"],
        "lo": cr["xext_lo_%"],
        "hi": cr["xext_hi_%"],
        "thr_asp": xv["aspirational"],
        "thr_full": xv["persist_full"],
        "thr_stop15": xv["stop_x1.5"],
        "maxcov": aff,
        "t100": cr["treat100_pct_of_diet_median"],
    }
)
qf = R("data/results/v3/mc/curve_ratio_quantiles.csv")
Q = pd.read_csv(qf) if os.path.exists(qf) else None

GRP = {
    "GLB": "Global",
    **{
        k: "Patent expiry ≤2027"
        for k in "BGR ROU IND MEX ZAF CHN TUR BRA NZL CAN AUS ISR".split()
    },
    **{k: "Never filed" for k in "EGY BGD NGA".split()},
    **{k: "Protected" for k in "USA DEU GBR JPN".split()},
}
COL = {
    "Global": "#1a1a1a",
    "Patent expiry ≤2027": "#00A087",
    "Never filed": "#3C5488",
    "Protected": "#E64B35",
}
d["grp"] = d.index.map(GRP)
assert d.notna().all().all(), d[d.isna().any(axis=1)]


def curve(iso):

    g = CUR[CUR.iso3 == iso].sort_values("cov")
    x = 100 * g["cov"].values
    y = np.log(g.ratio.values)
    f = PchipInterpolator(x, y)
    return lambda c: np.exp(f(np.clip(c, x.min(), x.max())))


os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
d.round(2).to_csv(a.out + "_data.csv")

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "hatch.linewidth": 0.4,
        "pdf.fonttype": 42,
    }
)
MONT = a.panel_a == "montage"
fig = plt.figure(figsize=(7.4, 9.6) if MONT else (7.4, 5.2))


def tx(v):
    v = np.asarray(v, float)
    return np.where(v <= 100, v, 100 + 50 * np.log10(np.clip(v, 100, None) / 100))


if MONT:
    axC = fig.add_axes([0.09, 0.59, 0.785, 0.37])
    axH = fig.add_axes([0.075, 0.12, 0.46, 0.41])
    axB = fig.add_axes([0.62, 0.12, 0.36, 0.41])
    LET = dict(C="a", H="b", B="c")
    HB = (0.075, 0.46, 0.033)
else:
    axC = axH = fig.add_axes([0.065, 0.25, 0.46, 0.67])
    axB = fig.add_axes([0.62, 0.25, 0.36, 0.67])
    LET = dict(C="a", H="a", B="b")
    HB = (0.075, 0.46, 0.07)
SHOW_C, SHOW_H = a.panel_a in ("curves", "montage"), a.panel_a in ("heatmap", "montage")

order = ["GLB"] + d.drop("GLB").sort_values("thr").index.tolist()
if SHOW_C:
    c = np.linspace(2.5, 100, 400)
    axC.axhspan(1, 40, color="#00A087", alpha=0.05, lw=0)
    axC.axhspan(0.01, 1, color="#E64B35", alpha=0.05, lw=0)
    axC.text(
        99,
        30,
        "Dietary policy averts more",
        ha="right",
        va="top",
        color="#00795f",
        fontsize=7,
        style="italic",
    )
    axC.text(
        99,
        0.0115,
        "GLP-1RA treatment averts more",
        ha="right",
        va="bottom",
        color="#b23a29",
        fontsize=7,
        style="italic",
    )
    for x0, lab in [(10, "10%"), (20, "Near-term\n(20%)")]:
        axC.axvline(x0, color="grey", lw=0.6, ls=(0, (3, 2)), zorder=1)
        axC.text(
            x0 + 1, 22 if x0 == 20 else 30, lab, va="top", fontsize=6.5, color="grey"
        )
    axC.axhline(1, color="black", lw=0.8, zorder=2)

    g = d.loc["GLB"]
    if Q is not None:
        q = Q[(Q.iso3 == "GLB") & (Q["cov"] > 0)].sort_values("cov")
        axC.fill_between(
            100 * q["cov"], q.lo, q.hi, color="grey", alpha=0.18, lw=0, zorder=1
        )
    fg = curve("GLB")
    if g.thr > 100:
        axC.annotate(
            f"Global: not matched even with all\neligible adults treated (treatment\naverts {g.t100:.0f}% as many deaths)",
            xy=(70, fg(70)),
            xytext=(36, 6.5),
            fontsize=6.5,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", lw=0.6, color="black"),
        )

    ends = []
    for iso, r in d.iterrows():
        col, lw, z = (
            COL[r.grp],
            (1.8 if iso == "GLB" else 0.9),
            (5 if iso == "GLB" else 3),
        )
        m = min(r.maxcov, 100)
        f = curve(iso)
        cin, cout = c[c <= m + 1e-9], c[c >= m - 1e-9]
        axC.plot(
            cin, f(cin), color=col, lw=lw, alpha=1 if iso == "GLB" else 0.85, zorder=z
        )
        if len(cout) > 1:
            axC.plot(
                cout,
                f(cout),
                color=col,
                lw=lw * 0.8,
                ls=(0, (1, 1.5)),
                alpha=0.6,
                zorder=z,
            )
        if r.thr <= 100:
            axC.scatter(
                r.thr,
                1,
                s=22 if iso == "GLB" else 12,
                color=col,
                edgecolor="white",
                lw=0.5,
                zorder=6,
            )
        ends.append([iso, float(f(100)), col])

    ends = sorted(ends, key=lambda x: x[1])
    ys = np.log10([e[1] for e in ends])
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + 0.085)
    for (iso, y0, col), yl in zip(ends, ys):
        axC.annotate(
            "Global" if iso == "GLB" else iso,
            xy=(100, y0),
            xytext=(103, 10**yl),
            fontsize=6,
            color=col,
            va="center",
            fontweight="bold" if iso == "GLB" else "normal",
            annotation_clip=False,
            arrowprops=dict(
                arrowstyle="-", lw=0.3, color=col, alpha=0.6, shrinkA=0, shrinkB=0
            ),
        )

    axC.set_yscale("log")
    axC.set_ylim(0.01, 40)
    axC.set_xlim(0, 100)
    yt = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    axC.set_yticks(yt)
    axC.set_yticklabels([f"{v:g}" for v in yt])
    axC.set_xlabel("GLP-1RA treatment coverage (% of eligible adults)")
    axC.set_ylabel(
        "Deaths averted, dietary policy /\nGLP-1RA treatment (log scale)"
        if MONT
        else "Deaths averted, dietary policy / GLP-1RA treatment (log scale)"
    )
    axC.spines[["top", "right"]].set_visible(False)
    axC.text(
        -0.095 if MONT else -0.09,
        1.03,
        LET["C"],
        transform=axC.transAxes,
        fontsize=12,
        fontweight="bold",
    )
    axC.legend(
        handles=[Line2D([], [], color=v, lw=1.4, label=k) for k, v in COL.items()],
        title="Semaglutide patent status",
        title_fontsize=6.5,
        fontsize=6.3,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.07),
        frameon=False,
        handlelength=1.6,
    )

if SHOW_H:
    from matplotlib.colors import LinearSegmentedColormap

    cmapc = LinearSegmentedColormap.from_list(
        "dr",
        [
            (0.0, "#8E0F1E"),
            (0.25, "#D6604D"),
            (0.4999, "#F6C3B5"),
            (0.5001, "#C7E9DA"),
            (0.75, "#4DB38F"),
            (1.0, "#00523F"),
        ],
    )

    def tf(r):
        l = np.clip(np.log10(np.asarray(r, float)), -1, 1)
        return 0.5 + 0.5 * np.sign(l) * np.sqrt(np.abs(l))

    xs = np.arange(2.5, 100.01, 0.25)
    n = len(order)
    for k, iso in enumerate(order):
        yy = n - 1 - k
        axH.imshow(
            tf(curve(iso)(xs))[None, :],
            aspect="auto",
            cmap=cmapc,
            vmin=0,
            vmax=1,
            extent=[2.5, 100, yy - 0.44, yy + 0.44],
            interpolation="bilinear",
            zorder=0,
        )
    for k, iso in enumerate(order):
        yy = n - 1 - k
        r = d.loc[iso]
        x = cr.loc[iso, "crossover_median_%"]
        if np.isfinite(x) and x <= 100:
            axH.plot(max(x, 2.5), yy, "|", ms=10, mew=1.8, color="black", zorder=4)
        if r.maxcov < 100:
            axH.add_patch(
                plt.Rectangle(
                    (r.maxcov, yy - 0.5),
                    100 - r.maxcov,
                    1,
                    facecolor="none",
                    edgecolor="white",
                    hatch="////",
                    lw=0,
                    zorder=3,
                )
            )
    for x0, lab in [(10, "10%"), (20, "Near-term\n(20%)")]:
        axH.axvline(x0, color="#333333", lw=0.6, ls=(0, (3, 2)), zorder=2)
        axH.text(x0 + 1, n - 0.45, lab, va="bottom", fontsize=6.5, color="grey")
    axH.axhline(n - 1.5, color="white", lw=2.5, zorder=2)
    axH.set_xlim(2.5, 100)
    axH.set_ylim(-0.7, n + 0.5)
    axH.set_facecolor("white")
    axH.set_xticks([10, 20, 40, 60, 80, 100])
    axH.set_yticks(range(n)[::-1])
    axH.set_yticklabels(["Global" if i == "GLB" else i for i in order])
    for t, iso in zip(axH.get_yticklabels(), order):
        t.set_color(COL[d.loc[iso, "grp"]])
        t.set_fontweight("bold" if iso == "GLB" else "normal")
    axH.tick_params(axis="y", length=0)
    for sp in axH.spines.values():
        sp.set_visible(False)
    axH.set_xlabel("GLP-1RA treatment coverage (% of eligible adults)")
    axH.text(
        -0.13, 1.04, LET["H"], transform=axH.transAxes, fontsize=12, fontweight="bold"
    )
    cax = fig.add_axes([HB[0], HB[2], HB[1], 0.016 if MONT else 0.022])
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmapc),
        cax=cax,
        orientation="horizontal",
    )
    tv = [0.1, 0.3, 0.5, 0.8, 1, 1.25, 2, 3, 10]
    cb.set_ticks(tf(tv))
    cb.set_ticklabels([f"{v:g}" for v in tv])
    cb.outline.set_visible(False)
    cax.tick_params(length=2, labelsize=6.5)
    cb.set_label(
        "Deaths averted, dietary policy / GLP-1RA treatment", fontsize=7, labelpad=2
    )
    fig.text(
        HB[0],
        HB[2] + (0.024 if MONT else 0.035),
        "◀ Treatment averts more",
        color="#B2182B",
        fontsize=6.8,
    )
    fig.text(
        HB[0] + HB[1],
        HB[2] + (0.024 if MONT else 0.035),
        "Dietary policy averts more ▶",
        color="#00735B",
        fontsize=6.8,
        ha="right",
    )


y = np.arange(len(order))[::-1]
NR0, NRX, NRT, XR = 104, 108, 113, 138
for x0 in [10, 20]:
    axB.axvline(x0, color="grey", lw=0.6, ls=(0, (3, 2)), zorder=1)
axB.axvline(100, color="#555555", lw=0.8, zorder=1)
axB.axvspan(0, 20, color="#E64B35", alpha=0.06, lw=0)
axB.axvspan(101.5, XR, color="#7f7f7f", alpha=0.07, lw=0)
axB.text(
    (101.5 + XR) / 2,
    len(order) - 0.45,
    "Not reached\nby 100%",
    ha="center",
    va="bottom",
    fontsize=6.2,
    color="grey",
)


def seg(lo_, hi_, yy, **kw):
    if lo_ > 100:
        return
    axB.plot([lo_, min(hi_, 100)], [yy, yy], **kw)
    if hi_ > 100:
        axB.annotate(
            "",
            xy=(NR0 - 0.5, yy),
            xytext=(100, yy),
            arrowprops=dict(arrowstyle="->", lw=0.6, color=kw["color"]),
            annotation_clip=False,
        )


for yy, iso in zip(y, order):
    r = d.loc[iso]
    col = COL[r.grp]
    lo_p, hi_p = sorted([r.thr_full, r.thr_stop15])
    seg(lo_p, hi_p, yy, color=col, lw=4.2, alpha=0.18, solid_capstyle="butt", zorder=2)
    seg(r.lo, r.hi, yy, color=col, lw=0.9, zorder=3)
    if r.thr <= 100:
        axB.scatter(
            r.thr,
            yy,
            s=26 if iso == "GLB" else 18,
            color=col,
            zorder=5,
            edgecolor="white",
            lw=0.5,
        )
    else:
        axB.scatter(
            NRX,
            yy,
            s=26 if iso == "GLB" else 18,
            color=col,
            zorder=5,
            edgecolor="white",
            lw=0.5,
        )
        axB.text(
            NRT,
            yy,
            f"{r.t100:.0f}%",
            va="center",
            fontsize=6,
            color=col,
            fontweight="bold" if iso == "GLB" else "normal",
        )
    axB.scatter(
        r.thr_asp if r.thr_asp <= 100 else NR0,
        yy,
        s=14,
        marker="D",
        facecolor="white",
        edgecolor=col,
        lw=0.7,
        zorder=4,
    )
    m = r.maxcov
    if m < 100:
        axB.add_patch(
            plt.Rectangle(
                (m, yy - 0.42),
                100 - m,
                0.84,
                facecolor="none",
                edgecolor="#9a9a9a",
                hatch="////",
                lw=0,
                zorder=1.5,
            )
        )
axB.set_yticks(y)
axB.set_yticklabels([] if SHOW_H else ["Global" if i == "GLB" else i for i in order])
for t, iso in zip(axB.get_yticklabels(), order):
    t.set_color(COL[d.loc[iso, "grp"]])
    t.set_fontweight("bold" if iso == "GLB" else "normal")
axB.set_xlim(0, XR)
axB.set_ylim(-0.7, len(order) + 0.5)
if SHOW_H:
    axB.tick_params(axis="y", length=0)
    axB.axhline(len(order) - 1.5, color="#cccccc", lw=0.6)
axB.set_xticks([0, 20, 40, 60, 80, 100])
axB.set_xlabel("Coverage needed to match dietary policy\n(% of eligible adults)")
axB.spines[["top", "right"]].set_visible(False)
axB.text(
    -0.05 if SHOW_H else -0.2,
    1.04,
    LET["B"],
    transform=axB.transAxes,
    fontsize=12,
    fontweight="bold",
)
axB.legend(
    handles=[
        Line2D(
            [], [], marker="o", ls="", color="grey", ms=4.5, label="Median (95% UI)"
        ),
        Line2D(
            [],
            [],
            color="grey",
            lw=4.2,
            alpha=0.3,
            label="Range across persistence\n(stopping ×1.5 to no stopping)",
        ),
        Line2D(
            [],
            [],
            marker="D",
            ls="",
            mfc="white",
            mec="grey",
            ms=3.6,
            label="Versus aspirational scenario",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="#9a9a9a",
            hatch="////",
            lw=0,
            label="Beyond maximum affordable\ncoverage at generic prices",
        ),
        Line2D(
            [],
            [],
            marker="",
            ls="",
            label="Not reached: treatment at 100%\ncoverage as % of dietary benefit",
        ),
    ]
    + (
        [
            Line2D(
                [],
                [],
                marker="|",
                ls="",
                color="black",
                ms=8,
                mew=1.6,
                label=f"Crossover (panel {LET['H']}, median)",
            )
        ]
        if SHOW_H
        else []
    ),
    loc="upper center",
    bbox_to_anchor=(0.36, -0.115 if MONT else -0.17),
    ncol=2,
    fontsize=6,
    frameon=False,
    handlelength=1.6,
    columnspacing=1.0,
)

fig.savefig(a.out + ".pdf")
fig.savefig(a.out + ".png", dpi=400)
print("saved:", a.out + ".pdf / .png / _data.csv")
