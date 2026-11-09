# -*- coding: utf-8 -*-
"""Low-to-high education ratio of the diet-attributable fraction, with uncertainty (Supplementary Table 14, Figure 6a).

Uses the same method as code/preprocessing/equity_paf_by_edu_v2.py, which gives the education-specific PAFs of Supplementary Table 15:
education gradients in intake from the Global Dietary Database (stratum mean intake / national mean intake, adults aged
25 or older) are applied to the GBD 2023 national exposures of each sex, the adjusted exposures are passed through the
corrected GBD 2023 dose-response curves, and PAFs are averaged over sexes. The central estimate (no sampling) therefore
equals the ratio of the Table 15 PAFs. Uncertainty: in each draw, national and education-specific GDD intakes are
sampled from normal distributions matching their 95% uncertainty intervals, and the gradient recomputed; dose-response
curves are held fixed, so these intervals reflect uncertainty in the education gradient only.
Output: data/results/v3/equity_paf_ratio_ui.csv (outcome, iso3, central, med, lo, hi)
Usage (repository root): python code/preprocessing/equity_paf_by_edu_mc.py [--draws 200] [--jobs 32]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.machinery import SourceFileLoader
import numpy as np, pandas as pd

dpa = SourceFileLoader("dpa", "code/preprocessing/diet_policy_arm.py").load_module()
CURVES = "data/params/dose_response_conv_fixed.csv"
G = None


def setup():
    global G
    g = pd.read_csv("data/exposure_19country.csv", low_memory=False)
    g = g[g.age >= 22.5]
    agg = dict(m=("intake_median", "mean"), lo=("intake_lo95", "mean"), hi=("intake_hi95", "mean"))
    nat = g[g.stratum_type == "age_marginal"].groupby(["gdd_var", "iso3"]).agg(**agg)
    cr = g[g.stratum_type == "cross"]
    st = {e: cr[cr.edu == e].groupby(["gdd_var", "iso3"]).agg(**agg) for e in (1, 3)}
    curves = pd.read_csv(CURVES)
    cvt = pd.read_csv("data/params/cv_by_factor.csv"); cvmap = dict(zip(cvt.gdd_var, cvt.cv_between * 1.8))
    e23 = {s: pd.read_csv(f"data/input/exposure_gbd_{s}.csv").query("year == 2023") for s in ("male", "female")}
    G = dict(nat=nat, st=st, curves=curves, cvmap=cvmap, e23=e23)


def pafs(draw, seed):
    """Education-specific PAFs (low and high), averaged over sexes; draw < 0 gives the central estimate."""
    if G is None: setup()
    rng = np.random.default_rng([seed, max(draw, 0)])
    samp = lambda m, lo, hi: m if draw < 0 else max(rng.normal(m, max((hi - lo) / (2 * 1.96), 1e-9)), 1e-9)
    ratio = {}
    for (gv, iso), r in G["nat"].iterrows():
        n_d = samp(r.m, r.lo, r.hi)
        if n_d <= 0: continue
        for e in (1, 3):
            if (gv, iso) in G["st"][e].index:
                s = G["st"][e].loc[(gv, iso)]; ratio[(gv, iso, e)] = samp(s.m, s.lo, s.hi) / n_d
    out = {}
    for sex, e23 in G["e23"].items():
        sbp = f"data/input/sbp_gbd2023_{sex}.csv"
        for e in (1, 3):
            et = e23.copy()
            et["intake_median"] = [v * ratio.get((gv, iso, e), 1.0) for v, gv, iso in zip(et.intake_median, et.gdd_var, et.iso3)]
            p = dpa.combined_paf(et, G["curves"], G["cvmap"], outcomes=("IHD", "IschStroke", "T2D"))
            nb = {"IHD": dpa.sodium_paf_via_sbp(et, G["curves"], sbp),
                  "IschStroke": dpa.sodium_paf_via_sbp(et, G["curves"], sbp, curve_id="high_sbp_s"),
                  "CKD": dpa.sodium_paf_via_sbp(et, G["curves"], sbp, curve_id="high_sbp_ckd")}
            for (oc, iso), v in p.items():
                out.setdefault((oc, iso, e), []).append(1 - (1 - v) * (1 - nb.get(oc, {}).get(iso, 0.0)))
            for iso, v in nb["CKD"].items():
                out.setdefault(("CKD", iso, e), []).append(v)
    return draw, {k: float(np.mean(v)) for k, v in out.items()}


def main(a):
    with ProcessPoolExecutor(a.jobs, initializer=setup) as ex:
        res = dict(ex.map(pafs, range(-1, a.draws), [a.seed] * (a.draws + 1)))
    rows = [dict(draw=d, outcome=oc, iso3=iso, edu=e, paf=v) for d, r in res.items() for (oc, iso, e), v in r.items()]
    w = pd.DataFrame(rows).pivot_table(index=["draw", "outcome", "iso3"], columns="edu", values="paf")
    w["ratio"] = np.where(w[3] > 0, w[1] / w[3], np.nan)
    c = w.xs(-1, level="draw").ratio.rename("central")
    q = w.drop(index=-1, level="draw").groupby(["outcome", "iso3"]).ratio.quantile([.5, .025, .975]).unstack()
    q.columns = ["med", "lo", "hi"]
    t = pd.concat([c, q], axis=1).round(3)
    t.to_csv("data/results/v3/equity_paf_ratio_ui.csv")
    # the central estimate must reproduce the point estimates of code/preprocessing/equity_paf_by_edu_v2.py
    v2 = pd.read_csv("data/results/v3/equity_paf_by_edu_v2.csv").pivot_table(index=["outcome", "iso3"], columns="edu", values="paf")
    chk = (100 * w.xs(-1, level="draw")[[1, 3]]).round(2)
    d = (chk - v2[[1, 3]].reindex(chk.index)).abs().max().max()
    print(f"central PAFs vs data/results/v3/equity_paf_by_edu_v2.csv: largest difference {d:.3f} percentage points")
    print(t.to_string()); print("-> data/results/v3/equity_paf_ratio_ui.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--draws", type=int, default=200); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--jobs", type=int, default=32); main(p.parse_args())
