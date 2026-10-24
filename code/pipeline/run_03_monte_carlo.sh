set -e; J=${1:-120}; source code/pipeline/env_main.sh; t() { echo "$(date '+%F %T') $*"; }; mkdir -p logs
export MC_DIR=data/results/v3/mc
t "3.1 draws and per-draw dietary files";  python code/analysis/mc_v3.py prepare --draws 1000 > logs/s3_prepare.log
t "3.2 main Monte Carlo";                  bash code/analysis/mc_v3.sh 1000 $J > logs/s3_mc.log 2>&1; tail -2 logs/s3_mc.log
python code/analysis/mc_curve_quantiles.py --jobs 32 > logs/s3_curve_quantiles.log 2>&1
t "3.3 combination with effect overlap";   python code/analysis/overlap_v3.py --jobs 32 > logs/s3_overlap.log 2>&1
t "3.4 trajectories of the primary outcome"
export MC_TRAJ_DIR=data/results/v3/mc_traj; mkdir -p $MC_TRAJ_DIR
[ -f $MC_TRAJ_DIR/jobs.txt ] || python code/analysis/mc_traj_v3.py prepare
cat $MC_TRAJ_DIR/jobs.txt | xargs -P $J -n 2 sh -c 'python code/analysis/mc_traj_v3.py worker --draw $0 --run $1 2>> '$MC_TRAJ_DIR'/errors.log'
python code/analysis/mc_traj_v3.py aggregate > logs/s3_traj.log 2>&1; unset MC_TRAJ_DIR
t "3.5 T2D and CKD";                       bash code/analysis/run_t2d_ckd.sh $J > logs/s3_t2d_ckd.log 2>&1
python code/analysis/outcome_cumulative_v3.py --jobs 32 > logs/s3_outcome_cumulative.log 2>&1
t "3.6 scenario Monte Carlo (300 draws)";  bash code/analysis/mc_scen_v3.sh 300 $J > logs/s3_scen.log 2>&1; tail -2 logs/s3_scen.log
t "3.7 convergence (first vs second 500 draws)"; python code/analysis/convergence_check.py --jobs 32 > logs/s3_convergence.log 2>&1
t "errors (lines): main $(wc -l < $MC_DIR/errors.log 2>/dev/null || echo 0), trajectories $(wc -l < data/results/v3/mc_traj/errors.log 2>/dev/null || echo 0), T2D/CKD $(wc -l < data/results/v3/mc_t2dckd/errors.log 2>/dev/null || echo 0), scenarios $(wc -l < data/results/v3/mc_scen/errors.log 2>/dev/null || echo 0)"
t "stage 3 done"
