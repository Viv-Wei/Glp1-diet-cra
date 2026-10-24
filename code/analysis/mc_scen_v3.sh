set -e
source code/pipeline/env_main.sh
export MC_DIR=data/results/v3/mc MC_SCEN_DIR=${MC_SCEN_DIR:-data/results/v3/mc_scen}
N=${1:-300}; J=${2:-120}
[ -f $MC_DIR/draws.csv ] || { echo "main Monte Carlo draws not found"; exit 1; }
mkdir -p $MC_SCEN_DIR
[ -f $MC_SCEN_DIR/jobs.txt ] || python code/analysis/mc_scen_v3.py prepare --draws $N
start=$(date +%s)
cat $MC_SCEN_DIR/jobs.txt | xargs -P $J -n 3 sh -c 'python code/analysis/mc_scen_v3.py worker --scen $0 --draw $1 --cov $2 2>> '$MC_SCEN_DIR'/errors.log'
echo "jobs finished in $(( ($(date +%s) - start) / 60 )) minutes; errors: $(wc -l < $MC_SCEN_DIR/errors.log 2>/dev/null || echo 0) lines"
python code/analysis/mc_scen_v3.py aggregate
