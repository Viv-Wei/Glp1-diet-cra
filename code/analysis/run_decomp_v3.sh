set -e
source code/pipeline/env_main.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LT_DP_OVERRIDE=data/results/v3/diet_policy_arm_v2_tailored.csv
O=data/results/v3/decomp
[ -d $O ] && [ ! -d data/results/v3_before_dietunc/decomp ] && mkdir -p data/results/v3_before_dietunc && mv $O data/results/v3_before_dietunc/decomp
mkdir -p $O
for v in "main:X=1" "postA:SENS_POST_NONE=1" "postC:SENS_POST_NONE=1 SENS_FATAL_MODE=none"; do
  t=${v%%:*}; e=${v
  env $e SENS_TREAT=maintain SENS_ALLOC=risk SENS_T2D=0 SENS_CKD=0 python code/lifetable_v3.py --years 30 --coverage 0.2 \
      --diet-scenario feasible --out $O/$t.csv > $O/$t.log 2>&1 &
done; wait
python - << 'EOF'
import pandas as pd
O = "data/results/v3/decomp"; r = {}
for t in ("main", "postA", "postC"):
    x = pd.read_csv(f"{O}/{t}.csv"); yr = [c for c in x.columns if c.startswith("yr")]
    r[t] = x[x.outcome.isin(["IHD", "IschStroke"])].groupby(["iso3", "arm"])[yr].sum().sum(axis=1).unstack("arm")
m = r["main"]
out = pd.DataFrame({"diet_M": m.diet / 1e6, "glp1_main_M": m.glp1 / 1e6, "glp1_postA_M": r["postA"].glp1 / 1e6,
                    "glp1_postC_M": r["postC"].glp1 / 1e6})
out["share_post_onset_pct"] = 100 * (1 - out.glp1_postA_M / out.glp1_main_M)
out.round(3).to_csv(f"{O}/decomp_summary.csv")
p = pd.read_csv(f"{O}/main_check_programme.csv"); p = p[(p.arm == "glp1") & (p.year == 1)].set_index("iso3")
out["eligible_share_pct"] = 100 * p.eligible / p.alive
pd.set_option("display.width", 200); print(out.round(2).to_string())
print(f"-> {O}/decomp_summary.csv; Figures 1 and 4 read {O}/main_check_programme.csv")
EOF
