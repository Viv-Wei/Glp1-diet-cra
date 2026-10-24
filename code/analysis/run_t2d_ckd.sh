set -e
J=${1:-120}
source code/pipeline/env_main.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LT_DP_OVERRIDE=data/results/v3/diet_policy_arm_v2_tailored.csv
export MC_DIR=data/results/v3/mc
mkdir -p data/results/v3/mc_t2dckd data/results/v3/mc_traj_t2dckd
echo "$(date '+%F %T') 1. coverage grid with T2D and CKD"
MC_OUT=data/results/v3/mc_t2dckd/out MC_ALL_KEY=1 sh -c "cat $MC_DIR/jobs.txt | xargs -P $J -n 2 sh -c 'python code/analysis/mc_v3.py worker --draw \$0 --cov \$1 2>> data/results/v3/mc_t2dckd/errors.log'"
echo "$(date '+%F %T')    errors: $(wc -l < data/results/v3/mc_t2dckd/errors.log 2>/dev/null || echo 0) lines"
python code/analysis/outcome_crossover_v3.py --dir data/results/v3/mc_t2dckd --jobs 32
echo "$(date '+%F %T') 2. trajectories with T2D and CKD"
export MC_TRAJ_DIR=data/results/v3/mc_traj_t2dckd MC_TRAJ_OUTCOMES=all
[ -f $MC_TRAJ_DIR/jobs.txt ] || python code/analysis/mc_traj_v3.py prepare
cat $MC_TRAJ_DIR/jobs.txt | xargs -P $J -n 2 sh -c 'python code/analysis/mc_traj_v3.py worker --draw $0 --run $1 2>> '$MC_TRAJ_DIR'/errors.log'
echo "$(date '+%F %T')    errors: $(wc -l < $MC_TRAJ_DIR/errors.log 2>/dev/null || echo 0) lines"
python code/analysis/mc_traj_v3.py aggregate
echo "$(date '+%F %T') all done"
