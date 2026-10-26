import os
import pandas as pd, numpy as np, argparse

K = pd.read_csv(
    os.environ.get("CRA_K_FILE", "data/params/paf_calibration_k.csv")
).set_index("iso3")


def calib(outc, iso):
    try:
        return float(K.loc[iso, outc])
    except Exception:
        return 1.0


from importlib.machinery import SourceFileLoader

pv2 = SourceFileLoader("pv2", "code/preprocessing/paf_deficiency_v2.py").load_module()

NA2SBP = 3.82 / 2.30

SCENARIOS = {
    "sodium30": {"v37": ("pct", -0.30)},
    "feasible": {
        "v37": ("pct", -0.15),
        "v15": ("pct", -0.20),
        "v08": ("toward_tmrel", 0.15),
    },
    "aspirational": {
        gv: ("toward_tmrel", 0.20)
        for gv in [
            "v01",
            "v02",
            "v05",
            "v06",
            "v08",
            "v09",
            "v10",
            "v15",
            "v29",
            "v30",
            "v34",
            "v37",
        ]
    },
}
SINGLE_FACTORS = [
    "v01",
    "v02",
    "v05",
    "v06",
    "v08",
    "v09",
    "v10",
    "v15",
    "v29",
    "v30",
    "v34",
    "v37",
]
for _gv in SINGLE_FACTORS:
    SCENARIOS[f"single_{_gv}"] = {_gv: ("toward_tmrel", 0.20)}
# Parts of the feasible package used only for the effect on deaths after onset (SENS_DIET_POST, scenario analyses):
# sodium, whose effect through blood pressure is the same with and without previous cardiovascular disease in trials
# (BPLTTC, Lancet 2021), and sodium with whole grains (cohort evidence in myocardial infarction survivors, Li, BMJ 2014)
SCENARIOS["post_na"] = {"v37": SCENARIOS["feasible"]["v37"]}
SCENARIOS["post_na_wg"] = {"v37": SCENARIOS["feasible"]["v37"], "v08": SCENARIOS["feasible"]["v08"]}
TAILORED = {}
_tf = os.environ.get("CRA_TAILORED_FILE", "data/params/tailored_top3.csv")
if os.path.exists(_tf):
    _t = pd.read_csv(_tf).sort_values(["iso3", "rank"])
    _x = float(os.environ.get("SENS_TAILORED_SHIFT", 0.15))
    for _iso, _g in _t.groupby("iso3"):
        TAILORED[_iso] = {
            gv: (("pct", -_x) if gv == "v37" else ("toward_tmrel", _x))
            for gv in _g.gdd_var.head(3)
        }
    SCENARIOS["tailored"] = {}


def shifted_exposure(expo, curves_df, scenario):
    e = expo.copy()
    tm = curves_df.groupby("gdd_var").tmrel.first()
    if scenario == "tailored":
        items = [(iso, gv, mx) for iso, d in TAILORED.items() for gv, mx in d.items()]
    else:
        items = [(None, gv, mx) for gv, mx in SCENARIOS[scenario].items()]
    # Direction of each factor: protective factors (GBD "low_" curves) are only raised towards the TMREL, harmful
    # factors only lowered. A stratum already on the favourable side of the TMREL is left unchanged, because a policy
    # promoting a protective food would not lower intake where it is already high (and vice versa).
    protective = curves_df.groupby("gdd_var").curve_id.first().str.startswith("low_")
    for iso_t, gv, (mode, x) in items:
        m = (e.gdd_var == gv) & ((e.iso3 == iso_t) if iso_t is not None else True)
        if mode == "pct":
            e.loc[m, "intake_median"] *= (1 + x)
        else:
            t = 3.0 if gv == "v37" else tm.get(gv, np.nan)   # sodium TMREL (3.0 g/day) is not in the curve table
            if pd.notna(t):
                step = x * (t - e.loc[m, "intake_median"])
                step = step.clip(lower=0) if bool(protective.get(gv, False)) else step.clip(upper=0)
                e.loc[m, "intake_median"] += step
    return e


def sodium_paf_via_sbp(
    expo_df, curves_df, sbp_file, na_tmrel=3.0, curve_id="high_sbp_ihd"
):
    cur = pv2.Curve(curves_df[curves_df.curve_id == curve_id])
    sbp = pd.read_csv(sbp_file).set_index(["iso3", "age"]).sbp_mean
    na = expo_df[(expo_df.gdd_var == "v37") & (expo_df.year == 2023)]
    out = {}
    for iso, g in na.groupby("iso3"):
        vals = []
        for _, r in g.iterrows():
            s0 = sbp.get((iso, r.age), np.nan)
            if pd.isna(s0):
                continue
            excess = max(float(r.intake_median) - na_tmrel, 0.0)
            rr0 = cur.rr(s0, strict=False)
            rr1 = cur.rr(s0 - excess * NA2SBP, strict=False)
            vals.append(1 - rr1 / rr0)
        out[iso] = float(np.mean(vals)) if vals else 0.0
    return out


def combined_paf(expo_df, curves_df, cvmap, outcomes=("IHD", "IschStroke", "T2D")):
    out = {}
    for cid, sub in curves_df.groupby("curve_id"):
        gv = sub.gdd_var.iloc[0]
        if pd.isna(gv):
            continue
        fac, outc, _ = pv2.parse_curve_id(cid)
        if outc not in outcomes:
            continue
        cur = pv2.Curve(sub)
        e = expo_df[(expo_df.gdd_var == gv) & (expo_df.year == 2023)]
        for iso, g in e.groupby("iso3"):
            pafs = [pv2.paf_dist(cur, v, cvmap.get(gv, 0.6)) for v in g.intake_median]
            out.setdefault((outc, iso), {})[fac] = float(np.mean(pafs))
    return {k: 1 - np.prod([1 - p for p in d.values()]) for k, d in out.items()}


def main(a):
    curves = pd.read_csv(a.curves)
    cvt = pd.read_csv(a.cv_file)
    cvmap = dict(zip(cvt.gdd_var, cvt.cv_pop))
    rows = []
    for sex in ["male", "female"]:
        expo = pd.read_csv(f"data/input/exposure_gbd_{sex}.csv")
        sbpf = f"data/input/sbp_gbd2023_{sex}.csv"
        base = combined_paf(expo, curves, cvmap)
        for _iso in set(k[1] for k in base):
            base[("CKD", _iso)] = 0.0
        na_b = {
            "T2D": {iso: 0.0 for iso in set(k[1] for k in base)},
            "IHD": sodium_paf_via_sbp(expo, curves, sbpf),
            "IschStroke": sodium_paf_via_sbp(expo, curves, sbpf, curve_id="high_sbp_s"),
            "CKD": sodium_paf_via_sbp(expo, curves, sbpf, curve_id="high_sbp_ckd"),
        }
        for name in list(SCENARIOS.keys()):
            sh = shifted_exposure(expo, curves, name)
            scen = combined_paf(sh, curves, cvmap)
            na_s = {
                "T2D": {iso: 0.0 for iso in set(k[1] for k in base)},
                "IHD": sodium_paf_via_sbp(sh, curves, sbpf),
                "IschStroke": sodium_paf_via_sbp(
                    sh, curves, sbpf, curve_id="high_sbp_s"
                ),
                "CKD": sodium_paf_via_sbp(sh, curves, sbpf, curve_id="high_sbp_ckd"),
            }
            for outc, iso in base:
                b = 1 - (1 - base[(outc, iso)]) * (1 - na_b[outc][iso])
                sc = 1 - (1 - scen.get((outc, iso), 0.0)) * (1 - na_s[outc][iso])
                rows.append(
                    dict(
                        sex=sex,
                        scenario=name,
                        outcome=outc,
                        iso3=iso,
                        paf_base=b * calib(outc, iso),
                        paf_scen=sc * calib(outc, iso),
                        averted_frac_alldeaths=(b - sc) * calib(outc, iso),
                    )
                )
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    g = t.groupby(["scenario", "outcome", "iso3"])[
        ["paf_base", "paf_scen", "averted_frac_alldeaths"]
    ].mean()
    print((g * 100).round(2).to_string())
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--curves", default="data/params/dose_response_conv_fixed.csv")
    p.add_argument("--cv-file", default="data/params/cv_by_factor.csv")
    p.add_argument("--out", default="data/results/diet_policy_arm.csv")
    p.add_argument("--wholegrain-disc", type=float, default=1.0)
    _a = p.parse_args()
    if _a.wholegrain_disc != 1.0:
        _k, _v = SCENARIOS["feasible"]["v08"]
        SCENARIOS["feasible"]["v08"] = (_k, _v * _a.wholegrain_disc)
        print(
            f"[wholegrain discount] feasible v08: {_v} -> {_v * _a.wholegrain_disc:.4f}"
        )
    main(_a)
