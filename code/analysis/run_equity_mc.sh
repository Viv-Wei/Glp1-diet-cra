#!/usr/bin/env bash
# Deaths averted by education group with uncertainty: the first 300 main Monte Carlo draws at 20% coverage, rerun to keep
# deaths averted by age, then the education allocation (equity_v3.py) applied to each draw. The new runs must reproduce
# the main Monte Carlo's deaths for the same draws. Usage (repository root): nohup bash code/analysis/run_equity_mc.sh [jobs] > run_equity_mc.log 2>&1 &
set -e; J=${1:-120}; t() { echo "$(date '+%F %T') $*"; }; B=data/results/v3/equity_mc
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p $B/out $B/byage $B/alloc
t "1. 300 draws at 20% coverage, keeping deaths averted by age"
seq 0 299 | xargs -P $J -I{} sh -c 'MC_OUT='$B'/out MC_KEEP_BYAGE='$B'/byage python code/analysis/mc_v3.py worker --draw {} --cov 0.200 2>> '$B'/errors.log'
echo "   by-age files: $(ls $B/byage | wc -l) of 300; errors: $(cat $B/errors.log 2>/dev/null | wc -l)"
python - << 'PY'
import pandas as pd, numpy as np, sys
d = [(pd.read_csv(f"data/results/v3/equity_mc/out/d{k}_c0.200.csv"), pd.read_csv(f"data/results/v3/mc/out/d{k}_c0.200.csv")) for k in range(300)]
diff = max((a.set_index(["iso3", "arm"]).deaths - b.set_index(["iso3", "arm"]).deaths).abs().max() / b.deaths.abs().max() for a, b in d)
print(f"   largest relative difference from the main Monte Carlo, same draws: {diff:.2e}")
if diff > 1e-9: sys.exit("   STOP: the new runs do not reproduce the main Monte Carlo")
PY
t "2. education allocation in each draw"
seq 0 299 | xargs -P 32 -I{} sh -c 'python code/analysis/equity_v3.py --run '$B'/byage/d{}_c0.200 --outdir '$B'/alloc/d{} > /dev/null'
echo "   allocations: $(ls $B/alloc | wc -l) of 300"
t "3. summary"; python code/analysis/equity_mc_aggregate.py
t "done"
