import glob, os, sys
import numpy as np, pandas as pd

D = sys.argv[1] if len(sys.argv) > 1 else "data/results/v3/grid"
fs = sorted(f for f in glob.glob(f"{D}/**/*_check_programme.csv", recursive=True))
if not fs:
    sys.exit(f"no *_check_programme.csv under {D}")
rows, summ = [], []
for f in fs:
    run = os.path.relpath(f, D).replace("_check_programme.csv", "")
    p = pd.read_csv(f)
    p = p[p.arm == "glp1"].copy()
    if p.empty:
        continue
    p["target_cov"] = p.target / p.eligible
    p["dev_after_pp"] = 100 * (p.coverage_after - p.target_cov)
    p["excess_before_pp"] = 100 * (p.coverage_before - p.target_cov)
    p.insert(0, "run", run)
    rows.append(p)
    for iso, g in p.groupby("iso3"):
        summ.append(
            dict(
                run=run,
                iso3=iso,
                target_cov_pct=100 * g.target_cov.median(),
                max_abs_dev_after_pp=g.dev_after_pp.abs().max(),
                max_excess_before_pp=g.excess_before_pp.max(),
                years_overshoot=int((g.excess_before_pp > 1e-6).sum()),
                years_target_not_met=int((g.dev_after_pp < -1e-6).sum()),
                mean_new_enrolment_pct_of_enrolled=100
                * (g.new_enrolment / g.enrolled_after).mean(),
            )
        )
by_year = pd.concat(rows)
by_year.to_csv(f"{D}/coverage_check_by_year.csv", index=False)
s = pd.DataFrame(summ)
s.to_csv(f"{D}/coverage_check_summary.csv", index=False)
print(f"{len(fs)} runs, {len(s)} run-country combinations")
print(
    f"  largest |actual - target| after top-up      : {s.max_abs_dev_after_pp.max():.2e} percentage points"
)
print(
    f"  largest carry-over above target before top-up: {s.max_excess_before_pp.max():.2e} percentage points"
)
print(
    f"  run-country combinations with any overshoot  : {(s.years_overshoot > 0).sum()}"
)
print(
    f"  run-country combinations with target unmet   : {(s.years_target_not_met > 0).sum()} (coverage of 100% cannot exceed the pool)"
)
if (s.years_overshoot > 0).any():
    print("\nOvershoot (enrolment carried over above target):")
    print(
        s[s.years_overshoot > 0]
        .sort_values("max_excess_before_pp", ascending=False)
        .head(20)
        .round(3)
        .to_string(index=False)
    )
if (s.years_target_not_met > 0).any():
    print("\nTarget not met after top-up (eligible pool exhausted):")
    print(
        s[s.years_target_not_met > 0]
        .sort_values("max_abs_dev_after_pp", ascending=False)
        .head(20)
        .round(3)
        .to_string(index=False)
    )
print(f"-> {D}/coverage_check_by_year.csv, {D}/coverage_check_summary.csv")
