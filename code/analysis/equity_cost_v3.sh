set -e
source code/pipeline/env_main.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export LT_DP_OVERRIDE=data/results/v3/diet_policy_arm_v2_tailored.csv
OUT=${OUT:-data/results/v3/equity_cost_final}; mkdir -p $OUT
[ -f data/results/v3/equity_paf_by_edu_v2.csv ] || { echo "run code/preprocessing/equity_paf_by_edu_v2.py first"; exit 1; }
for c in 0.10 0.20 afford; do
  SENS_TREAT=maintain SENS_ALLOC=risk python code/lifetable_v3.py --years 30 --coverage $c --diet-scenario feasible \
    --out $OUT/main_$c.csv > $OUT/main_$c.log 2>&1 &
done; wait
for c in 0.10 0.20 afford; do
  echo "################ coverage $c: equity";      python code/analysis/equity_v3.py --run $OUT/main_$c --outdir $OUT/cov$c
  echo "################ coverage $c: expenditure"; python code/analysis/cost_v3.py   --run $OUT/main_$c --outdir $OUT/cov$c
done
