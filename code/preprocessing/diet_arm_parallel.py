import argparse, os, sys
from multiprocessing import Pool
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader

M = SourceFileLoader(
    "diet_policy_arm",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "diet_policy_arm.py"),
).load_module()

G = {}


def sodium_all(expo, sbpf, isos):
    curves = G["curves"]
    return {
        "T2D": {iso: 0.0 for iso in isos},
        "IHD": M.sodium_paf_via_sbp(expo, curves, sbpf),
        "IschStroke": M.sodium_paf_via_sbp(expo, curves, sbpf, curve_id="high_sbp_s"),
        "CKD": M.sodium_paf_via_sbp(expo, curves, sbpf, curve_id="high_sbp_ckd"),
    }


def work(task):
    sex, name = task
    curves, cvmap = G["curves"], G["cvmap"]
    expo, sbpf, base, na_b = (
        G["expo"][sex],
        f"data/input/sbp_gbd2023_{sex}.csv",
        G["base"][sex],
        G["na_b"][sex],
    )
    sh = M.shifted_exposure(expo, curves, name)
    scen = M.combined_paf(sh, curves, cvmap)
    na_s = sodium_all(sh, sbpf, set(k[1] for k in base))
    rows = []
    for outc, iso in base:
        b = 1 - (1 - base[(outc, iso)]) * (1 - na_b[outc][iso])
        sc = 1 - (1 - scen.get((outc, iso), 0.0)) * (1 - na_s[outc][iso])
        k = M.calib(outc, iso)
        rows.append(
            dict(
                sex=sex,
                scenario=name,
                outcome=outc,
                iso3=iso,
                paf_base=b * k,
                paf_scen=sc * k,
                averted_frac_alldeaths=(b - sc) * k,
            )
        )
    return rows


def main(a):
    if a.wholegrain_disc != 1.0:
        kind, v = M.SCENARIOS["feasible"]["v08"]
        M.SCENARIOS["feasible"]["v08"] = (kind, v * a.wholegrain_disc)
        print(
            f"[whole-grain discount] feasible v08: {v} -> {v * a.wholegrain_disc:.4f}"
        )
    G["curves"] = pd.read_csv(a.curves)
    cvt = pd.read_csv(a.cv_file)
    G["cvmap"] = dict(zip(cvt.gdd_var, cvt.cv_pop))
    G["expo"], G["base"], G["na_b"] = {}, {}, {}
    for sex in ["male", "female"]:
        expo = pd.read_csv(f"data/input/exposure_gbd_{sex}.csv")
        base = M.combined_paf(expo, G["curves"], G["cvmap"])
        for iso in set(k[1] for k in base):
            base[("CKD", iso)] = 0.0
        G["expo"][sex], G["base"][sex] = expo, base
        G["na_b"][sex] = sodium_all(
            expo, f"data/input/sbp_gbd2023_{sex}.csv", set(k[1] for k in base)
        )
    tasks = [(sex, name) for sex in ["male", "female"] for name in M.SCENARIOS]
    print(f"{len(tasks)} scenario x sex tasks on {min(a.jobs, len(tasks))} processes")
    with Pool(min(a.jobs, len(tasks))) as pool:
        out = pool.map(work, tasks)
    t = pd.DataFrame([r for rows in out for r in rows])
    t.to_csv(a.out, index=False)
    print(f"-> {a.out} ({len(t)} rows, {t.scenario.nunique()} scenarios)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--curves", default="data/params/dose_response_conv_fixed.csv")
    p.add_argument("--cv-file", default="data/params/cv_by_factor.csv")
    p.add_argument("--out", default="data/results/diet_policy_arm.csv")
    p.add_argument("--wholegrain-disc", type=float, default=1.0)
    p.add_argument("--jobs", type=int, default=32)
    main(p.parse_args())
