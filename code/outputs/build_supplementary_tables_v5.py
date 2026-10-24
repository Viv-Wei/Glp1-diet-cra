import argparse, copy, glob, os, re
import numpy as np, pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment

ap = argparse.ArgumentParser()
ap.add_argument("--style", default="data/tables/Supplementary_Tables_v4_final.xlsx")
ap.add_argument("--out", default="data/tables/Supplementary_Tables_v5.xlsx")
ap.add_argument("--select-csv", default="data/results/v3/select_main.csv")
A = ap.parse_args()
r1 = lambda v: np.round(v, 1)
r2 = lambda v: np.round(v, 2)
NR = "Not reached"


def style_table(n):

    x = pd.read_excel(A.style, f"Supplementary Table {n}", header=1)
    return x.iloc[:-1].reset_index(drop=True)


t5 = style_table(5)
NAME = dict(zip(t5.ISO3, t5.Country))
NAME["GLB"] = "Global"
CR = pd.read_csv("data/results/v3/mc/summary_crossover_ratios.csv").set_index("iso3")
ORDER = ["GLB"] + CR.drop("GLB").sort_values("xext_median_%").index.tolist()


def order(df, col="ISO3"):
    df = df.copy()
    df["_o"] = df[col].map({k: i for i, k in enumerate(ORDER)})
    return df.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)


N_DRAWS = len(pd.read_csv("data/results/v3/mc/draws.csv"))


def cx(v):

    return NR if (pd.isna(v) or not np.isfinite(v) or v > 100) else round(float(v), 1)


NEW = {}
LEGEND = {}
HEADER = {}

t3 = style_table(3)
cal = pd.read_csv("data/results/v3/decomp/main_check_calibration.csv")
err4 = (
    cal[["err_inc_I", "err_inc_S", "err_d_I", "err_d_S", "err_d_all"]].abs().max().max()
)
full = pd.read_csv("data/results/v3/decomp/main_check_calibration_full.csv")
ev = 100 * pd.concat([full.err_d_I, full.err_d_S]).abs()
CAL = dict(median=float(ev.median()), max=float(ev.max()))
rows = [
    [
        "Calibration, four-state life table",
        "All countries, sexes and five-year age groups",
        "Largest relative difference from GBD 2023 incidence, cause-specific and all-cause death rates",
        f"{err4:.1e}",
        "Numerical tolerance 1e-7",
    ],
    [
        "Calibration, full model with each state divided by diabetes status",
        "All countries, sexes and five-year age groups",
        "Relative difference in first-year IHD and IS deaths from the calibrated rates (%), median (largest)",
        f"{CAL['median']:.2f} ({CAL['max']:.2f})",
        "Largest differences in the oldest age groups",
    ],
]
cc = pd.read_csv("data/results/v3/grid/coverage_check_summary.csv")
cc["alloc"] = cc.run.str.split("_").str[0]
cov_rows = {}
for al, g in cc.groupby("alloc"):
    lab = {"risk": "Risk-prioritised allocation", "unif": "Uniform allocation"}.get(al)
    if lab:
        cov_rows[lab] = [
            round(float(g.max_excess_before_pp.max()), 3),
            f"{int((g.years_overshoot > 0).sum())} of {len(g)} run-country combinations",
        ]
t3 = t3.copy()
for i, r in t3.iterrows():
    if r.Check == "Annual coverage check" and r["Group or country"] in cov_rows:
        t3.loc[i, ["Model value", "Comparison"]] = cov_rows[r["Group or country"]]
NEW[3] = pd.concat([pd.DataFrame(rows, columns=t3.columns), t3], ignore_index=True)
LEGEND[3] = [
    (
        "External validation simulated",
        "Calibration compares modelled first-year rates with GBD 2023 targets in every "
        "country, sex and five-year age group, for the four-state life table and for the full model with each state "
        "divided by diabetes status. External validation simulated",
    ),
    (
        "Calibration error was below 1e-7 and population conservation error below 1e-15 in every country, sex and age (not tabulated).",
        "Population conservation error was below 1e-15 in every country, sex and age (not tabulated).",
    ),
]

t4 = style_table(4)
dr = pd.read_csv("data/results/v3/mc/draws.csv")
KEYS = [
    "hr_mi",
    "hr_stroke",
    "hr_cvd",
    "hr_mi_dm",
    "hr_stroke_dm",
    "hr_cvd_dm",
    "hr_t2d",
    "s_ihd",
    "stop_mult",
    "restart_mult",
    "resid_cv",
    "resid_t2d",
    "dm_hr_ihd",
    "dm_hr_is",
    "sbp_shift",
    "sbp_shift_dm",
]
assert len(t4) == len(KEYS), "template Table 4 rows do not match the parameter list"
rows = [
    [
        t4.Parameter[i],
        t4.Distribution[i],
        r2(dr[k].median()),
        r2(dr[k].quantile(0.025)),
        r2(dr[k].quantile(0.975)),
    ]
    for i, k in enumerate(KEYS)
]
u = pd.read_csv("data/params/diet_paf_uncertainty.csv").set_index("outcome")
for oc, lab in [
    ("IHD", "IHD"),
    ("IschStroke", "ischaemic stroke"),
    ("T2D", "type 2 diabetes"),
    ("CKD", "CKD"),
]:
    k = f"diet_factor_{oc}"
    if k in dr:
        rows.append(
            [
                f"Dietary effect multiplier, {lab}",
                f"Split log-normal, median 1, 95% range {u.lower[oc] / u.val[oc]:.2f} to {u.upper[oc] / u.val[oc]:.2f}",
                r2(dr[k].median()),
                r2(dr[k].quantile(0.025)),
                r2(dr[k].quantile(0.975)),
            ]
        )
NEW[4] = pd.DataFrame(rows, columns=t4.columns)
LEGEND[4] = [
    ("Each of 500 simulations", f"Each of {N_DRAWS} simulations"),
    (
        "Uncertainty in dietary effects was propagated through the potential impact fractions.",
        "Dietary effects were multiplied by an outcome-specific factor, fitted to the GBD 2023 95% uncertainty interval of "
        "the diet-attributable fraction relative to its point estimate, common to all countries within each draw and "
        "drawn independently for each outcome.",
    ),
    ("CV, cardiovascular;", "CKD, chronic kidney disease; CV, cardiovascular;"),
]

t7 = style_table(7)
pdl = pd.read_csv("data/results/v3/supp/prob_diet_leads.csv", dtype={"cov": str}).pivot(
    index="iso3", columns="cov", values="p_diet_leads"
)
ov = pd.read_csv("data/results/v3/mc_traj/overtaking_year.csv").pivot(
    index="iso3", columns="run", values="first_year_diet_ahead"
)
rows = []
for iso, r in CR.iterrows():
    d = [
        NAME.get(iso, iso),
        iso,
        cx(r["xext_median_%"]),
        cx(r["xext_lo_%"]),
        cx(r["xext_hi_%"]),
        round(100 * r.share_not_reached),
        round(100 * r["P(crossover<20%)"]),
        round(r.treat100_pct_of_diet_median),
        r2(r.r20_median),
        r2(r.r20_lo),
        r2(r.r20_hi),
    ]
    d += [round(100 * pdl.loc[iso, c]) for c in ("0.100", "0.200", "0.500", "afford")]
    d += [
        "Not within 30 years" if pd.isna(ov.loc[iso, c]) else int(ov.loc[iso, c])
        for c in ("c0.10", "c0.20", "cafford")
    ]
    rows.append(d)
NEW[7] = order(pd.DataFrame(rows, columns=t7.columns))
HEADER[7] = {
    "Simulations with crossover above 100% (%)": "Simulations in which crossover was not reached (%)"
}
LEGEND[7] = [
    (
        "Where treatment did not match dietary policy even with every eligible adult enrolled, the value above 100% is the "
        "size of programme needed relative to enrolling all eligible adults, calculated as the deaths averted by dietary "
        "policy divided by those averted at full coverage. This is a lower bound, because any additional people treated "
        "would be at lower risk.",
        "Where treatment did not match dietary policy even with every eligible adult enrolled, crossover is reported as not "
        "reached, and treatment at full coverage gives the deaths averted by treatment with every eligible adult enrolled "
        "as a percentage of those averted by dietary policy.",
    ),
    ("share of 500 simulations", f"share of {N_DRAWS} simulations"),
]

t8 = style_table(8)
ob = pd.read_csv("data/results/v3/supp/outcomes_by_arm.csv", dtype={"cov": str})
OUTC = [
    ("deaths", "Deaths from IHD and ischaemic stroke"),
    ("t2d", "Incident type 2 diabetes cases"),
    ("ckd", "Deaths from CKD"),
]
ARM = [
    ("diet", "Dietary policy (feasible package)"),
    ("glp1", "GLP-1RA treatment"),
    ("combo", "Both (independent action)"),
]
COVL = [("0.100", "10%"), ("0.200", "20%"), ("afford", "Maximum affordable")]
obi = ob.set_index(["iso3", "cov", "arm", "outcome"])
rows = []
for iso in ORDER:
    for oc, ol in OUTC:
        for arm, al in ARM:
            for cv, cl in ([("0.200", "Not applicable")] if arm == "diet" else COVL):
                q = obi.loc[(iso, cv, arm, oc)]
                rows.append(
                    [
                        NAME.get(iso, iso),
                        iso,
                        ol,
                        al,
                        cl,
                        r2(q["median"] / 1e3),
                        r2(q.lo / 1e3),
                        r2(q.hi / 1e3),
                    ]
                )
NEW[8] = pd.DataFrame(rows, columns=t8.columns)
LEGEND[8] = [("from 500 simulations", f"from {N_DRAWS} simulations")]

t9 = style_table(9)
o = pd.read_csv("data/results/v3/overlap/overlap_by_country.csv").assign(
    outcome="IHD+IS"
)
o2 = pd.read_csv("data/results/v3/mc_traj_t2dckd/overlap_by_outcome.csv")
o2 = o2[o2.outcome != "IHD+IS"]
OL = {
    "IHD+IS": "Deaths from IHD and ischaemic stroke",
    "T2D": "Incident type 2 diabetes cases",
    "CKD": "Deaths from CKD",
}
rows = []
for oc in ("IHD+IS", "T2D", "CKD"):
    src_ = o if oc == "IHD+IS" else o2[o2.outcome == oc]
    for _, r in src_.iterrows():
        has_ui = "add_diet_to_trt_pct_lo" in r and pd.notna(
            r.get("add_diet_to_trt_pct_lo", np.nan)
        )
        rows.append(
            [
                NAME.get(r.iso3, r.iso3),
                r.iso3,
                OL[oc],
                r.coverage,
                float(r.overlap_pct),
                r2(r.combined_median / 1e3),
                r2(r.combined_lo / 1e3),
                r2(r.combined_hi / 1e3),
                r1(r.add_diet_to_trt_pct_median),
                r2(1 + r.add_diet_to_trt_pct_median / 100),
                r2(1 + r.add_diet_to_trt_pct_lo / 100) if has_ui else "Not estimated",
                r2(1 + r.add_diet_to_trt_pct_hi / 100) if has_ui else "Not estimated",
            ]
        )
x9 = pd.DataFrame(rows, columns=t9.columns)
x9["_c"] = x9[t9.columns[3]].map({"10%": 0, "20%": 1, "Maximum affordable": 2})
x9["_oc"] = x9.Outcome.map({v: i for i, v in enumerate(OL.values())})
NEW[9] = order(x9.sort_values(["_oc", "_c", t9.columns[4]]).drop(columns=["_c", "_oc"]))
LEGEND[9] = [("from 500 simulations", f"from {N_DRAWS} simulations")]

t10 = style_table(10)
s = pd.read_csv("data/results/v3/mc_t2dckd/summary_outcome_crossover.csv")
rows = []
for _, r in s.iterrows():
    ne = r.outcome == "CKD" and r.r20_median <= 0
    rows.append(
        [
            NAME.get(r.iso3, r.iso3),
            r.iso3,
            {"T2D": "Incident type 2 diabetes", "CKD": "Deaths from CKD"}[r.outcome],
        ]
        + (
            ["Not estimable"] * 3
            if ne
            else [cx(r.xext_median_pct), cx(r.xext_lo_pct), cx(r.xext_hi_pct)]
        )
        + (["Not estimable"] if ne else [round(100 * r.share_not_reached)])
        + (
            ["Not estimable"] * 3
            if ne
            else [r2(r.r20_median), r2(r.r20_lo), r2(r.r20_hi)]
        )
    )
x10 = pd.DataFrame(rows, columns=t10.columns)
x10["_oc"] = x10.Outcome.map({"Incident type 2 diabetes": 0, "Deaths from CKD": 1})
NEW[10] = order(x10.sort_values("_oc", kind="stable").drop(columns="_oc"))
HEADER[10] = {
    "Simulations with crossover above 100% (%)": "Simulations in which crossover was not reached (%)"
}
LEGEND[10] = [
    (
        "averted over 30 years.",
        "averted over 30 years. Where treatment did not match dietary policy even with every "
        "eligible adult enrolled, crossover is reported as not reached.",
    )
]

t11 = style_table(11)
f_ = pd.read_csv("data/tables/table_single_factor_v3.csv")
code2name = dict(zip(f_["GDD variable code"], f_["Dietary factor"]))
x11 = f_.drop(columns=["GDD variable code"]).rename(columns={"Country": "ISO3"})
x11.insert(0, "Country", x11.ISO3.map(NAME))
assert len(x11.columns) == len(t11.columns)
x11.columns = t11.columns
fo = {v: i for i, v in enumerate(pd.unique(t11["Dietary factor"]))}
NEW[11] = order(
    x11.assign(_f=x11["Dietary factor"].map(fo))
    .sort_values("_f", kind="stable")
    .drop(columns="_f")
)
t12 = style_table(12)
rows = []
for f in sorted(glob.glob("data/results/v3/single_factor/single_v[0-9][0-9].csv")):
    x = pd.read_csv(f)
    x = x[(x.arm == "diet") & x.outcome.isin(["IHD", "IschStroke"])]
    yr = [c for c in x.columns if c.startswith("yr")]
    g = x.groupby("iso3")[yr].sum()
    fac = code2name.get(re.search(r"single_(v\d+)", f).group(1))
    for iso, v in g.iterrows():
        cum = np.cumsum(v.values)
        rows.append(
            [NAME.get(iso, iso), iso, fac]
            + [round(float(cum[k - 1])) for k in (5, 10, 15, 20, 25, 30)]
        )
x12 = pd.DataFrame(rows, columns=t12.columns)
NEW[12] = order(
    x12.assign(_f=x12["Dietary factor"].map(fo))
    .sort_values("_f", kind="stable")
    .drop(columns="_f")
)

sc = pd.read_csv("data/results/v3/mc_scen/summary_scenarios.csv")
d3 = pd.read_csv("data/results/v3/mc_scen/summary_scenarios_disc3.csv")
sc = pd.concat(
    [sc, d3[d3.scenario == "main"].assign(scenario="discount_3pct")], ignore_index=True
)
FM = {
    "cvd": "as for cardiovascular death",
    "same": "as for non-fatal events",
    "none": "none",
}
G = {
    "Main analysis": {"main": "Main analysis"},
    "Treatment persistence": {
        "const_retention": "Fixed retention of two thirds of the effect",
        "resid_0": "No benefit retained after stopping",
        "no_restart": "No restarting after stopping",
        "restart_late0": "No restarting after two years off",
        "persist_full": "Nobody stops treatment",
        "restart_late0.5": "Restarting halved after two years off",
        "stop_x1.5": "Stopping 1.5 times as often",
        "stop_x0.5": "Stopping half as often",
    },
    "Effect in people with IHD or IS": {
        "post_A": "No effect on deaths after onset",
        "post_C": "Nor on fatal first events",
    },
    "Dietary policy implementation": {
        "phase_12": "Phase-in over 12 years",
        "phase_16": "Phase-in over 16 years",
        "maintain_0.4": "Policy effect held at 40%",
        "maintain_0.6": "Policy effect held at 60%",
        "maintain_0.8": "Policy effect held at 80%",
    },
    "Scope of dietary policy": {
        "aspirational": "Aspirational scenario",
        "tailored": "Package tailored to each country",
        "sodium30": "Sodium reduction of 30%",
        "tailored_cap": "Tailored package, with benefit in people with IHD or IS capped at US level",
    },
    "Eligible population and programme": {
        "tier3_bmi27": "All adults with BMI 27 or more eligible",
        "bmi25": "Eligibility extended to BMI 25 or more",
        "bmi30": "Eligibility restricted to BMI 30 or more",
        "baseline_cohort": "No replacement of enrolled adults who die",
        "no_dm_tier": "No priority for diabetes",
        "select_like": "Only adults resembling SELECT",
        "uniform": "Places allocated uniformly",
        "dynamic": "Every eligible adult with IHD or IS enrolled from year 2",
        "cap": "Absolute benefit in people with IHD or IS capped at US level",
    },
    "Trial effect estimates": {
        "primary_weak": "40% of the effect without CVD",
        "class_overall": "Class-level estimate for everyone",
        "itt_select": "Correction for discontinuation in trials",
        "sattar_dm": "Earlier meta-analysis, with diabetes",
        "tier3_none": "No effect without CVD or diabetes",
        "tier2_rewind": "REWIND, diabetes without CVD",
        "select_only": "SELECT only, without diabetes",
    },
    "Model structure": {
        "is_cf10": "Case fatality of first ischaemic stroke 10%",
        "is_cf20": "Case fatality of first ischaemic stroke 20%",
        "dm_no_risk": "Diabetes does not raise cardiovascular risk",
        "dm_pr2": "Diabetes prevalence ratio 2",
        "dm_pr3": "Diabetes prevalence ratio 3",
        "k0.8": "Relative duration of IHD or IS with BMI ≥27, 0.8",
        "k1.2": "Relative duration of IHD or IS with BMI ≥27, 1.2",
        **{
            f"s{a_}_{m_}": f"IHD deaths as fatal first events {int(round(float(a_) * 100))}%, effect on fatal first events {FM[m_]}"
            for a_ in ("0.25", "0.4", "0.55")
            for m_ in FM
        },
        "bmi_gamma": "Gamma distribution of BMI",
        "or2": "Odds ratio of having both IHD and IS, 2",
        "or4": "Odds ratio of having both IHD and IS, 4",
        "rr1.5": "Risk of the second condition after the first, 1.5",
        "rr2": "Risk of the second condition after the first, 2",
    },
    "Population and time": {
        "discount_3pct": "Discounting 3% a year",
        "open_pop": "Open population (new adults enter)",
    },
}
LBL = {k: (g, v) for g, d_ in G.items() for k, v in d_.items()}


def scen_rows(df):
    out = []
    for _, r in df.iterrows():
        share = (
            [r1(100 * r.t20_median), r1(100 * r.t20_lo), r1(100 * r.t20_hi)]
            if pd.notna(r.get("t20_median"))
            else [np.nan] * 3
        )
        cross = (
            ["Not estimated (20% coverage only)"] * 3
            if r.scenario == "dynamic"
            else [cx(100 * r.xext_median), cx(100 * r.xext_lo), cx(100 * r.xext_hi)]
        )
        ratio = (
            ["See share"] * 3
            if r.scenario == "select_like"
            else [r2(r.r20_median), r2(r.r20_lo), r2(r.r20_hi)]
        )
        row = cross + ratio + [round(100 * r.P_diet_leads_20)] + share
        if pd.notna(r.r20_median) and r.r20_median == 0:
            row = (
                ["No modelled dietary benefit"] * 6
                + [row[6]]
                + ["No modelled dietary benefit"] * 3
            )
        out.append(row)
    return out


t13 = style_table(13)
s13 = sc[~sc.scenario.str.startswith("sw_")].copy()
miss = sorted(set(s13.scenario) - set(LBL))
assert not miss, f"scenarios without a label: {miss}"
x13 = pd.DataFrame(
    [
        [LBL[k][0], LBL[k][1], NAME.get(i, i), i] + v
        for k, i, v in zip(s13.scenario, s13.iso3, scen_rows(s13))
    ],
    columns=t13.columns,
)
GORD = {g: i for i, g in enumerate(G)}
x13 = order(x13)
x13["_g"] = x13["Scenario group"].map(GORD)
NEW[13] = (
    x13.sort_values(["_g", "Scenario"], kind="stable")
    .drop(columns="_g")
    .reset_index(drop=True)
)
LEGEND[13] = [
    (
        "Crossover coverage above 100% is the programme size needed relative to enrolling all eligible adults (lower bound).",
        "Where treatment did not match dietary policy even with every eligible adult enrolled, crossover is reported as not reached.",
    )
]
t14 = style_table(14)
sw = sc[sc.scenario.str.startswith("sw_")].copy()
m = sw.scenario.str.extract(r"sw_pm([\d.]+)_stop([\d.]+)_rd(\d+)")
x14 = pd.DataFrame(
    [
        [int(round(float(a) * 100)), float(b), int(c), NAME.get(i, i), i] + v
        for a, b, c, i, v in zip(m[0], m[1], m[2], sw.iso3, scen_rows(sw))
    ],
    columns=t14.columns,
)
x14 = x14.sort_values(
    list(t14.columns[:3]), ascending=[False, True, True], kind="stable"
)
x14["_o"] = x14.ISO3.map({k: i for i, k in enumerate(ORDER)})
NEW[14] = (
    x14.sort_values(
        list(t14.columns[:3]) + ["_o"],
        ascending=[False, True, True, True],
        kind="stable",
    )
    .drop(columns="_o")
    .reset_index(drop=True)
)
LEGEND[14] = [
    (
        "Medians and 95% uncertainty intervals from 300 simulations per scenario.",
        "Medians and 95% uncertainty intervals from 300 simulations per scenario. Where treatment did not match dietary "
        "policy even with every eligible adult enrolled, crossover is reported as not reached.",
    )
]

E = "data/results/v3/equity_cost_final/cov0.20"
t19 = style_table(19)
e = pd.read_csv(f"{E}/tableS_equity_v3.csv")
e.insert(0, "Country", e.ISO3.map(NAME))
e["Diet-to-treatment ratio"] = (
    e["Diet-to-treatment ratio"]
    .astype(object)
    .where(e["Diet-to-treatment ratio"].notna(), "Treatment not affordable")
)
assert list(e.columns) == list(t19.columns), (list(e.columns), list(t19.columns))
NEW[19] = order(e)
t20 = style_table(20)
x20 = pd.read_csv(f"{E}/tableS_equity_mapping_sensitivity_v3.csv")
assert list(x20.columns) == list(t20.columns)
NEW[20] = x20
t21 = style_table(21)
CV = {"0.10": "10%", "0.20": "20%", "afford": "Maximum affordable"}
c_ = pd.concat(
    pd.read_csv(f"data/results/v3/equity_cost_final/cov{c}/tableS_cost_v3.csv").assign(
        **{"Coverage of eligible adults": l}
    )
    for c, l in CV.items()
)
dk, tk = (
    "Dietary policy expenditure per death averted (USD)",
    "GLP-1RA expenditure per death averted (USD)",
)
ref = c_[c_["Coverage of eligible adults"] == "20%"].set_index(
    ["ISO3", "Price scenario", "Outcomes"]
)[dk]
c_[dk] = [
    ref.get((i, p, o_), v)
    for i, p, o_, v in zip(c_.ISO3, c_["Price scenario"], c_.Outcomes, c_[dk])
]
c_["Ratio, treatment to diet"] = (
    pd.to_numeric(c_[tk], errors="coerce") / pd.to_numeric(c_[dk], errors="coerce")
).round(1)
c_.insert(0, "Country", c_.ISO3.map(NAME))
c_ = c_[["Country", "ISO3"] + [c for c in c_.columns if c not in ("Country", "ISO3")]]
assert list(c_.columns) == list(t21.columns), (list(c_.columns), list(t21.columns))
NEW[21] = order(c_.reset_index(drop=True))
t22 = style_table(22)
q_ = pd.read_csv(
    "data/results/v3/sa_t2d_ckd/summary_sa_t2d_ckd.csv", dtype={"coverage": str}
)
q_ = q_[q_.coverage.isin(["0.20", "0.2", "afford"])].dropna(subset=["t2d_glp1"])
SL = {
    "main": "Main analysis",
    "t2d_bucur026": "GLP-1RA network meta-analysis estimate",
    "t2d_dual010": "Dual GIP/GLP-1 receptor agonist estimate",
    "sbp_select331": "Blood pressure fall from SELECT",
    "sbp_step510": "Blood pressure fall from STEP 1",
    "sbp_meta632": "Larger blood pressure fall",
    "sbp_dm0": "No blood pressure fall with diabetes",
    "sbp_none": "No blood pressure fall",
}
lab22 = t22.drop_duplicates("Analysis").set_index("Analysis")[
    [t22.columns[1], t22.columns[2]]
]
rows = []
for k, l in SL.items():
    for cov, cl in (("0.20", "20%"), ("afford", "Maximum affordable")):
        r = q_[
            (q_.scenario == k)
            & (q_.coverage.isin([cov, "0.2"] if cov == "0.20" else [cov]))
        ]
        if not len(r):
            continue
        r = r.iloc[0]
        rows.append(
            [
                l,
                lab22.loc[l].iloc[0],
                lab22.loc[l].iloc[1],
                cl,
                round(r.t2d_glp1 / 1e3),
                round(r.t2d_diet / 1e3),
                r2(r.t2d_diet / r.t2d_glp1) if r.t2d_glp1 > 0 else "Not estimable",
                r1(r.ckd_glp1 / 1e3),
                r1(r.ckd_diet / 1e3),
                (
                    r2(r.ckd_diet / r.ckd_glp1)
                    if r.ckd_glp1 > 0
                    else "No treatment effect on CKD"
                ),
            ]
        )
NEW[22] = pd.DataFrame(rows, columns=t22.columns)

SORT_KEYS = {
    8: ["Outcome", "Arm", "Coverage of eligible adults"],
    9: ["Outcome", "Coverage of eligible adults", "Overlap of effects (%)"],
    10: ["Outcome"],
    11: ["Dietary factor"],
    12: ["Dietary factor"],
    13: ["Scenario group", "Scenario"],
    14: [
        "Dietary policy effect maintained (%)",
        "Stopping probability multiplier",
        "Dietary policy phase-in (years)",
    ],
    19: ["Price scenario", "Education group"],
    21: ["Price scenario", "Outcomes", "Coverage of eligible adults"],
}


def like_template(n, df):

    t = style_table(n)
    if "ISO3" not in df or "ISO3" not in t:
        return df
    keys = SORT_KEYS.get(n, [])
    df = df.copy()
    df["_c"] = df.ISO3.map({k: i for i, k in enumerate(ORDER)})
    if not keys:
        return (
            df.sort_values("_c", kind="stable")
            .drop(columns="_c")
            .reset_index(drop=True)
        )
    norm = (
        lambda x: x[keys]
        .apply(
            lambda c: c.map(
                lambda v: (
                    f"{float(v):g}"
                    if isinstance(v, (int, float, np.number))
                    else str(v)
                )
            )
        )
        .agg("|".join, axis=1)
    )
    rank = {k: i for i, k in enumerate(pd.unique(norm(t)))}
    df["_k"] = norm(df).map(rank)
    assert df["_k"].notna().all(), (
        n,
        "rows not found in the template",
        list(norm(df)[df["_k"].isna()][:3]),
    )
    country_major = t.ISO3.iloc[0] == t.ISO3.iloc[1]
    return (
        df.sort_values(["_c", "_k"] if country_major else ["_k", "_c"], kind="stable")
        .drop(columns=["_k", "_c"])
        .reset_index(drop=True)
    )


for n in list(NEW):
    if n not in (3, 4, 20, 22):
        NEW[n] = like_template(n, NEW[n])

wb = load_workbook(A.style)
report = []
for n, df in sorted(NEW.items()):
    ws = wb[f"Supplementary Table {n}"]
    tmpl = style_table(n)
    hdr = [ws.cell(2, j).value for j in range(1, ws.max_column + 1)]
    nc = len([h for h in hdr if h is not None])
    assert nc == len(df.columns), (n, nc, len(df.columns))
    assert [str(h) for h in hdr[:nc]] == [str(c) for c in df.columns], (
        n,
        "header mismatch",
    )
    last = ws.max_row
    legend = ws.cell(last, 1).value or ""
    for a_, b_ in LEGEND.get(n, []):
        assert a_ in legend, (n, "legend text not found", a_[:60])
        legend = legend.replace(a_, b_)
    data_style = [copy.copy(ws.cell(3, j)._style) for j in range(1, nc + 1)]
    leg_style, leg_h = (
        copy.copy(ws.cell(last, 1)._style),
        ws.row_dimensions[last].height,
    )
    for rng in list(ws.merged_cells.ranges):
        if rng.min_row > 1:
            ws.unmerge_cells(str(rng))
    ws.delete_rows(3, last - 2)
    for j, h in enumerate(df.columns, 1):
        ws.cell(2, j).value = HEADER.get(n, {}).get(h, h)
    for i, row in enumerate(df.itertuples(index=False), 3):
        for j, v in enumerate(row, 1):
            v = (
                None
                if (isinstance(v, float) and np.isnan(v))
                else (v.item() if hasattr(v, "item") else v)
            )
            c = ws.cell(i, j, v)
            c._style = copy.copy(data_style[j - 1])
    lr = len(df) + 3
    c = ws.cell(lr, 1, legend)
    c._style = leg_style
    ws.merge_cells(start_row=lr, start_column=1, end_row=lr, end_column=nc)
    ws.row_dimensions[lr].height = (
        leg_h if leg_h else 15 * max(2, len(legend) // 150 + 1)
    )
    report.append((n, len(tmpl), len(df)))
DROP = {1}
RENUM = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 6, 7: 7}
RENUM.update({o: o - 1 for o in range(9, 23)})
assert sorted(RENUM.values()) == list(range(1, 22)) and not (set(RENUM) & DROP)
for o in DROP:
    wb.remove(wb[f"Supplementary Table {o}"])
for o in RENUM:
    wb[f"Supplementary Table {o}"].title = f"tmp_{o}"
for o, n in RENUM.items():
    ws = wb[f"tmp_{o}"]
    ws.title = f"Supplementary Table {n}"
    ws.cell(1, 1).value = re.sub(
        r"^Supplementary Table \d+",
        f"Supplementary Table {n}",
        str(ws.cell(1, 1).value),
    )
cs = wb["Contents"]
titles = {}
for r in range(2, cs.max_row + 1):
    v = cs.cell(r, 1).value
    m_ = re.search(r"\d+", str(v)) if v else None
    if m_ and int(m_.group()) in RENUM:
        titles[RENUM[int(m_.group())]] = cs.cell(r, 2).value
last_row = cs.max_row
for i, n in enumerate(sorted(titles), 2):
    cs.cell(i, 1).value = f"Supplementary Table {n}"
    cs.cell(i, 2).value = titles[n]
for r in range(len(titles) + 2, last_row + 1):
    for c in range(1, cs.max_column + 1):
        cs.cell(r, c).value = None
wb._sheets = [cs] + sorted(
    [w for w in wb.worksheets if w.title != "Contents"],
    key=lambda w: int(w.title.split()[-1]),
)
os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
wb.save(A.out)
print(
    "renumbered (old -> new):",
    {o: n for o, n in RENUM.items() if o != n},
    "| dropped:",
    sorted(DROP),
)
print(f"{len(NEW)} tables updated -> {A.out}  ({N_DRAWS} draws)")
for n, a_, b_ in report:
    print(f"  Table {n:2d} (now {RENUM[n]:2d}): rows {a_} -> {b_}")
print(
    f"Full-model calibration: first-year IHD and IS deaths differ from the calibrated rates by a median of "
    f"{CAL['median']:.2f}% (largest {CAL['max']:.2f}%); four-state life table, largest error {err4:.1e}"
)
