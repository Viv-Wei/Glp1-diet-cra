set -u
J=${1:-8}
OUT=data/results/v3/sa_t2d_ckd; mkdir -p $OUT/params
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SENS_TREAT=maintain SENS_ALLOC=risk SENS_T2D=1 SENS_CKD=1
source code/pipeline/env_main.sh 2>/dev/null || true
python - << 'EOF'
import pandas as pd
tp = pd.read_csv("data/params/glp1_trial_params.csv")
for tag, v in [("bucur026", 0.26), ("dual010", 0.10)]:
    x = tp.copy(); x.loc[x.param == "t2d_incidence_hr", "value"] = v; x.to_csv(f"data/results/v3/sa_t2d_ckd/params/trial_{tag}.csv", index=False)
EOF
run() {
  tag=$1; shift
  for cov in 0.20 afford; do
    echo "env $* python code/lifetable_v3.py --years 30 --coverage $cov --diet-scenario feasible --out $OUT/${tag}_c${cov}.csv > $OUT/${tag}_c${cov}.log 2>&1"
  done
}
{
  run main X=1
  run t2d_bucur026 T2D_TRIAL_PARAMS=$OUT/params/trial_bucur026.csv
  run t2d_dual010 T2D_TRIAL_PARAMS=$OUT/params/trial_dual010.csv
  run sbp_select331 SENS_SBP_SHIFT=3.31 SENS_SBP_SHIFT_DM=2.13
  run sbp_step510 SENS_SBP_SHIFT=5.10 SENS_SBP_SHIFT_DM=2.13
  run sbp_meta632 SENS_SBP_SHIFT=6.32 SENS_SBP_SHIFT_DM=2.13
  run sbp_dm0 SENS_SBP_SHIFT=3.80 SENS_SBP_SHIFT_DM=0
  run sbp_none SENS_SBP_SHIFT=0 SENS_SBP_SHIFT_DM=0
} > $OUT/jobs.txt
xargs -P $J -I{} bash -c "{}" < $OUT/jobs.txt
python - << 'EOF'
import glob, os, pandas as pd
O = "data/results/v3/sa_t2d_ckd"; rows = []
for f in sorted(glob.glob(f"{O}/*_c*.csv")):
    if f.endswith(("_t2d.csv", "_ckd.csv", "_byage.csv")) or "_check_" in f: continue
    tag, cov = os.path.basename(f)[:-4].rsplit("_c", 1)
    r = dict(scenario=tag, coverage=cov)
    t2 = f[:-4] + "_t2d.csv"
    if os.path.exists(t2):
        x = pd.read_csv(t2); x = x[x.iso3 == "GLB"].set_index("arm").cases_averted_total
        r.update(t2d_glp1=x.get("glp1"), t2d_diet=x.get("diet"))
    ck = f[:-4] + "_ckd.csv"
    if os.path.exists(ck):
        y = pd.read_csv(ck); yr = [c for c in y.columns if c.startswith("yr")]
        y = y[y.iso3 == "GLB"].groupby("arm")[yr].sum().sum(axis=1)
        r.update(ckd_glp1=y.get("glp1"), ckd_diet=y.get("diet"))
    rows.append(r)
s = pd.DataFrame(rows); s.to_csv(f"{O}/summary_sa_t2d_ckd.csv", index=False)
print(s.to_string(index=False))
EOF
