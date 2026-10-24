import pandas as pd, numpy as np, glob
from importlib.machinery import SourceFileLoader

dpa = SourceFileLoader("dpa", "code/preprocessing/diet_policy_arm.py").load_module()

_g = pd.read_csv("data/exposure_19country.csv", low_memory=False)
_g = _g[_g.age >= 22.5]
ratios = {}
for gv, d in _g.groupby("gdd_var"):
    nat = d[d.stratum_type == "age_marginal"].groupby("iso3").intake_median.mean()
    cr = d[d.stratum_type == "cross"]
    for e in [1, 2, 3]:
        st = cr[cr.edu == e].groupby("iso3").intake_median.mean()
        for iso in nat.index:
            if nat[iso] > 0 and np.isfinite(st.get(iso, np.nan)):
                ratios[(gv, iso, e)] = float(st[iso] / nat[iso])
print(f"gradient ratios: {len(ratios)}")

curves = pd.read_csv("data/params/dose_response_conv_fixed.csv")
cvt = pd.read_csv("data/params/cv_by_factor.csv")
cvmap = dict(zip(cvt.gdd_var, cvt.cv_between * 1.8))
rows = []
for sex in ["male", "female"]:
    e23 = pd.read_csv(f"data/input/exposure_gbd_{sex}.csv")
    e23 = e23[e23.year == 2023]
    for tier in [1, 2, 3]:
        et = e23.copy()
        et["intake_median"] = et.apply(
            lambda r: r.intake_median * ratios.get((r.gdd_var, r.iso3, tier), 1.0),
            axis=1,
        )
        paf = dpa.combined_paf(et, curves, cvmap, outcomes=("IHD", "IschStroke", "T2D"))
        nb = {
            "IHD": dpa.sodium_paf_via_sbp(
                et, curves, f"data/input/sbp_gbd2023_{sex}.csv"
            ),
            "IschStroke": dpa.sodium_paf_via_sbp(
                et, curves, f"data/input/sbp_gbd2023_{sex}.csv", curve_id="high_sbp_s"
            ),
            "CKD": dpa.sodium_paf_via_sbp(
                et, curves, f"data/input/sbp_gbd2023_{sex}.csv", curve_id="high_sbp_ckd"
            ),
            "T2D": {iso: 0.0 for iso in set(k[1] for k in paf)},
        }
        for (outc, iso), v in paf.items():
            rows.append(
                dict(
                    sex=sex,
                    edu=tier,
                    outcome=outc,
                    iso3=iso,
                    paf=1 - (1 - v) * (1 - nb[outc].get(iso, 0.0)),
                )
            )
        for iso, ckd_paf in nb["CKD"].items():
            rows.append(dict(sex=sex, edu=tier, outcome="CKD", iso3=iso, paf=ckd_paf))
t = pd.DataFrame(rows)
out = t.groupby(["outcome", "iso3", "edu"]).paf.mean().mul(100).round(2)
out.to_csv("data/results/v3/equity_paf_by_edu_v2.csv")
print(out.unstack("edu").to_string())
print(
    "\nLow/high education PAF ratio (gradient direction: >1 means burden concentrated in low education):"
)
w = out.unstack("edu")
print((w[1] / w[3]).round(2).unstack(0).to_string())
