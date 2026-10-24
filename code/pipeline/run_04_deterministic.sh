set -e; J=${1:-120}; source code/pipeline/env_main.sh; t() { echo "$(date '+%F %T') $*"; }; mkdir -p logs
t "4.1 single-factor dietary ranking";    bash code/analysis/run_single_factor_v3.sh 12 > logs/s4_single_factor.log 2>&1
python code/analysis/rank_tailored_factors.py --check || { echo "the top three factors differ from data/params/tailored_top3.csv:"
  echo "run python code/analysis/rank_tailored_factors.py, then stage 1 (dietary arm) and stages 3-5 again"; exit 1; }
t "4.2 coverage grid and annual coverage check"; bash code/analysis/grid_v3.sh $J 41 > logs/s4_grid.log 2>&1; python code/analysis/coverage_check.py > logs/s4_coverage_check.log 2>&1
t "4.3 deterministic runs for Figures 1 and 4"; bash code/analysis/run_decomp_v3.sh > logs/s4_decomp.log 2>&1
t "4.4 external validation (SELECT-like cohort)"; bash code/analysis/test_select_v3.sh > logs/s4_select.log 2>&1
t "4.5 T2D and CKD sensitivity analyses";  bash code/analysis/sa_t2d_ckd.sh 16 > logs/s4_sa_t2d_ckd.log 2>&1
t "4.6 equity and expenditure";            bash code/analysis/equity_cost_v3.sh > logs/s4_equity_cost.log 2>&1
t "stage 4 done"
