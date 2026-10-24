set -e
source code/pipeline/env_main.sh
export LT_DP_OVERRIDE=data/results/v3/diet_policy_arm_v2_tailored.csv
export MC_DIR=${MC_DIR:-data/results/v3/mc}
N=${1:-500}; J=${2:-120}
mkdir -p $MC_DIR
[ -f $MC_DIR/draws.csv ] || python code/analysis/mc_v3.py prepare --draws $N
start=$(date +%s)
cat $MC_DIR/jobs.txt | xargs -P $J -n 2 sh -c 'python code/analysis/mc_v3.py worker --draw $0 --cov $1 2>> '$MC_DIR'/errors.log'
echo "jobs finished in $(( ($(date +%s) - start) / 60 )) minutes; errors: $(wc -l < $MC_DIR/errors.log 2>/dev/null || echo 0) lines"
python code/analysis/mc_v3.py aggregate
