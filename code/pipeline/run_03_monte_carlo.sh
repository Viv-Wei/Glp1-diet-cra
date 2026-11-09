#!/usr/bin/env bash
set -e; J=${1:-120}; t() { echo "### $(date '+%F %T') $*"; }
mkdir -p logs
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

t "0. checks"
free=$(df -BG --output=avail . | tail -1 | tr -dc 0-9); echo "free space: ${free} GB"
[ "$free" -ge 60 ] || { echo "STOP: less than 60 GB free"; exit 1; }
grep -E "SENS_ELIG_STRUCT|SENS_DIET_POST|SENS_ELIG_RULE" code/pipeline/env_main.sh && { echo "STOP: env_main.sh overrides the new defaults"; exit 1; } || true
python - << 'PY'
import os, sys
sys.path.insert(0, "code")
os.environ.setdefault("CRA_ROOT", os.getcwd())
from lifetable.settings import read_settings, ELIG_STRUCT
s = read_settings("0.20")
print(f"defaults: structure {ELIG_STRUCT}, rule {s.elig_rule}, dietary effect after onset {s.diet_post}, thresholds {s.thr_cvd}/{s.thr_ob}")
assert ELIG_STRUCT == "bmi" and s.elig_rule == "new" and s.diet_post == "post_na", "STOP: defaults are not the new main analysis"
PY
python code/lifetable/test_bmi.py | tail -1

t "2. main Monte Carlo: 1000 draws x 27 coverage levels"
bash code/analysis/mc_v3.sh 1000 $J > logs/bmi_mc.log 2>&1; tail -3 logs/bmi_mc.log
python code/analysis/mc_curve_quantiles.py --jobs 32 > logs/bmi_curve_quantiles.log 2>&1
python code/analysis/overlap_v3.py --jobs 32 > logs/bmi_overlap.log 2>&1

t "3. trajectories"
export MC_TRAJ_DIR=data/results/v3/mc_traj; mkdir -p $MC_TRAJ_DIR
python code/analysis/mc_traj_v3.py prepare
cat $MC_TRAJ_DIR/jobs.txt | xargs -P $J -n 2 sh -c 'python code/analysis/mc_traj_v3.py worker --draw $0 --run $1 2>> '$MC_TRAJ_DIR'/errors.log'
python code/analysis/mc_traj_v3.py aggregate > logs/bmi_traj.log 2>&1; tail -2 logs/bmi_traj.log; unset MC_TRAJ_DIR

t "4. education groups: 1000 draws at 20% coverage"
B=data/results/v3/equity_mc; mkdir -p $B/out $B/byage $B/alloc
seq 0 999 | xargs -P $J -I{} sh -c 'MC_OUT='$B'/out MC_KEEP_BYAGE='$B'/byage python code/analysis/mc_v3.py worker --draw {} --cov 0.200 2>> '$B'/errors.log'
python - << 'PY'
import pandas as pd, sys
diff = max((pd.read_csv(f"data/results/v3/equity_mc/out/d{k}_c0.200.csv").set_index(["iso3", "arm"]).deaths
            - pd.read_csv(f"data/results/v3/mc/out/d{k}_c0.200.csv").set_index(["iso3", "arm"]).deaths).abs().max() for k in range(1000))
print(f"  largest difference from the main Monte Carlo, same draws: {diff:.2e}")
if diff > 1e-6: sys.exit("STOP: the education runs do not reproduce the main Monte Carlo")
PY
seq 0 999 | xargs -P 32 -I{} sh -c 'python code/analysis/equity_v3.py --run '$B'/byage/d{}_c0.200 --outdir '$B'/alloc/d{} > /dev/null'
python code/analysis/equity_mc_aggregate.py > logs/bmi_equity.log 2>&1; tail -3 logs/bmi_equity.log
rm -rf $B/byage                                                     # large; the allocations are kept

t "5. T2D and CKD"
bash code/analysis/run_t2d_ckd.sh $J > logs/bmi_t2d_ckd.log 2>&1; tail -2 logs/bmi_t2d_ckd.log
python code/analysis/outcome_cumulative_v3.py --jobs 32 > logs/bmi_outcome_cumulative.log 2>&1

t "6. scenario Monte Carlo: 134 scenarios x 1000 draws (several days on 120 cores)"
bash code/analysis/mc_scen_v3.sh 1000 $J > logs/bmi_scen.log 2>&1; tail -3 logs/bmi_scen.log

t "6b. expenditure, and single-factor, secondary-outcome and discounted summaries from the simulations"
python code/analysis/expenditure_ui_v3.py > logs/bmi_expenditure.log 2>&1; tail -2 logs/bmi_expenditure.log
python code/analysis/scen_extra.py > logs/scen_extra.log 2>&1; tail -1 logs/scen_extra.log

t "7. convergence (first vs second 500 draws)"
python code/analysis/convergence_check.py --jobs 32 > logs/bmi_convergence.log 2>&1; tail -5 logs/bmi_convergence.log

t "errors (lines): main $(cat data/results/v3/mc/errors.log 2>/dev/null | wc -l), trajectories $(cat data/results/v3/mc_traj/errors.log 2>/dev/null | wc -l), education $(cat data/results/v3/equity_mc/errors.log 2>/dev/null | wc -l), T2D/CKD $(cat data/results/v3/mc_t2dckd/errors.log 2>/dev/null | wc -l), scenarios $(cat data/results/v3/mc_scen/errors.log 2>/dev/null | wc -l)"
t "done"
