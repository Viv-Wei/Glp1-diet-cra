set -u; source code/pipeline/env_main.sh; t() { echo "$(date '+%F %T') $*"; }; mkdir -p logs data/figures; fail=0
C=code/outputs
n=0; run() { n=$((n+1)); t "$1"; shift; "$@" > logs/s5_$n.log 2>&1 || { echo "   FAILED: $* (logs/s5_$n.log)"; tail -6 logs/s5_$n.log | sed "s/^/      /"; fail=$((fail+1)); }; }
[ -d data/results/v3/mc/out ] && run "5.1 Monte Carlo summaries for the tables" python code/outputs/supp_preaggregate.py --jobs 32
run "5.2 supplementary tables"                python code/outputs/build_supplementary_tables_v5.py
[ -f data/results/factor_burden_by_age.csv ] && run "5.3 check of Supplementary Table 5" python code/outputs/check_burden_tables.py
run "Figure 1"                                python $C/plot_figure1_starting_points.py
run "Figure 2, Supplementary Figures 2 and 3" python $C/plot_figure2_projections.py
run "Supplementary Figures 5 and 6 (T2D)"     python $C/plot_figure2_projections.py --outcome T2D
run "Supplementary Figures 7 and 8 (CKD)"     python $C/plot_figure2_projections.py --outcome CKD
run "Figure 3"                                python $C/plot_figure3_threshold.py
run "Figure 4"                                python $C/plot_figure4_country_drivers.py
run "Figure 5"                                python $C/plot_figure5_robustness.py
run "Figure 6, Supplementary Figure 9"        python $C/plot_figure6_equity.py
run "Supplementary Figure 4"                  python $C/plot_suppfig_t2d_ckd.py
run "Supplementary Figure 10"                 Rscript $C/plot_suppfig_expenditure.R
t "stage 5 done, failed steps: $fail"; exit $fail
