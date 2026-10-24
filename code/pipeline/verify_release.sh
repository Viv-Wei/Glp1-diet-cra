set -u; P=$(readlink -f "${1:?give the directory that holds data/gbd, data/wpp, data/worldbank and data/exposure_19country.csv}")
J=${2:-32}; R=$(pwd); W=${VERIFY_DIR:-/tmp/glp1_verify_$(id -un)}; t() { echo "$(date '+%F %T') $*"; }; fails=0
rm -rf $W; mkdir -p $W; for x in $(ls -A $R | grep -v -E "^(verify_work|logs|verify\.log)$"); do cp -r $R/$x $W/; done
ln -s $R $W/.linktest 2>/dev/null && rm $W/.linktest || { echo "$W does not allow symbolic links: set VERIFY_DIR to a directory that does"; exit 1; }
t "working copy: $W"
cd $W; mkdir -p logs data
for d in data/gbd data/wpp data/worldbank data/exposure_19country.csv; do
  [ -e $P/$d ] || { echo "   not found: $P/$d"; exit 1; }; ln -sfn $P/$d $d || { echo "   could not link $d"; exit 1; }; done
cmp_() { python code/pipeline/verify_compare.py "$@" || fails=$((fails+1)); }
pub() { for f in "$@"; do echo "$R/$f $W/$f"; done; }

t "A. stages 1 and 2 from the raw downloads"
bash code/pipeline/run_01_dietary_inputs.sh $J > logs/verify_stage1.log 2>&1 || { echo "   stage 1 FAILED"; tail -5 logs/verify_stage1.log; fails=$((fails+1)); }
bash code/pipeline/run_02_treatment_inputs.sh > logs/verify_stage2.log 2>&1 || { echo "   stage 2 FAILED"; tail -5 logs/verify_stage2.log; fails=$((fails+1)); }
cmp_ $(pub data/params/dose_response_conv_fixed.csv data/params/diet_paf_uncertainty.csv data/results/joint_burden_2023.csv \
    data/results/factor_burden_by_age.csv data/results/v3/diet_policy_arm_v2_tailored.csv data/results/v3/equity_paf_by_edu_v2.csv \
    data/params/overweight_prevalence.csv data/params/select_elig.csv data/params/overweight_cvd.csv data/params/overweight_prevalence_bmi27.csv \
    data/params/overweight_cvd_bmi27.csv data/params/elig_scale_bmi30_rel27.csv data/params/overweight_prevalence_bmi27_gamma.csv \
    data/params/overweight_cvd_bmi27_gamma.csv data/params/income_quintiles_wb.csv data/results/affordability_matrix.csv \
    data/params/income_quintiles_global.csv data/params/coverage_afford_cap.csv)
python code/outputs/check_burden_tables.py || fails=$((fails+1))

t "B. unit tests and deterministic runs"
python -m pytest code/tests -q 2>&1 | tail -1
bash code/analysis/run_decomp_v3.sh > logs/verify_decomp.log 2>&1 || { echo "   decomp FAILED"; fails=$((fails+1)); }
cmp_ $(cd $R && for f in data/results/v3/decomp/*.csv; do echo "$R/$f $W/$f"; done)

if [ -d $P/data/results/v3/mc/out ]; then
  t "C. first 20 Monte Carlo draws"
  source code/pipeline/env_main.sh; export MC_DIR=data/results/v3/mc
  python code/analysis/mc_v3.py prepare --draws 1000 > logs/verify_prepare.log 2>&1; cmp_ $(pub data/results/v3/mc/draws.csv)
  for d in $(seq 0 19); do for c in 0.100 0.200 afford; do echo "$d $c"; done; done | \
    xargs -P $J -n 2 sh -c 'python code/analysis/mc_v3.py worker --draw $0 --cov $1 2>> logs/verify_mc_errors.log'
  cmp_ $(for d in $(seq 0 19); do for c in 0.100 0.200 afford; do echo "$P/data/results/v3/mc/out/d${d}_c${c}.csv $W/data/results/v3/mc/out/d${d}_c${c}.csv"; done; done)
else
  t "C. skipped: no per-draw Monte Carlo files in $P/data/results/v3/mc/out"
fi

t "D. supplementary tables and figures from the result summaries"
rm -rf data/results/v3/mc/out
bash code/pipeline/run_05_tables_figures.sh > logs/verify_stage5.log 2>&1 || { echo "   stage 5 had failures:"; grep FAILED logs/verify_stage5.log; fails=$((fails+1)); }
cmp_ $(pub data/tables/Supplementary_Tables_v5.xlsx)
t "$([ $fails -eq 0 ] && echo 'RELEASE REPRODUCES THE PUBLISHED RESULTS' || echo "$fails check(s) failed: see above")"
