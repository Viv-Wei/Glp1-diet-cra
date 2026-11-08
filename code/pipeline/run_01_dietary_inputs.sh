set -e; J=${1:-32}; source code/pipeline/env_main.sh; t() { echo "$(date '+%F %T') $*"; }; mkdir -p data/results/v3 logs
K=data/params/paf_calibration_k.csv; CURVES=data/params/dose_response_conv_fixed.csv
t "1.1 dose-response curves";          python code/preprocessing/build_curves.py > logs/s1_curves.log
t "1.2 dietary uncertainty";           python code/preprocessing/build_diet_uncertainty.py > logs/s1_diet_uncertainty.log
t "1.3 joint and single-factor burden"; python code/preprocessing/build_joint_burden.py > logs/s1_joint_burden.log
python code/preprocessing/build_factor_burden.py > logs/s1_factor_burden.log
t "1.4 dietary policy arm (all scenarios, including the tailored package of data/params/tailored_top3.csv)"
CRA_K_FILE=$K python code/preprocessing/diet_arm_parallel.py --jobs $J --curves $CURVES --out data/results/v3/diet_policy_arm_v2_tailored.csv > logs/s1_diet_arm.log 2>&1
t "1.5 education-specific PAFs";       CRA_K_FILE=$K python code/preprocessing/equity_paf_by_edu_v2.py > logs/s1_equity_paf.log 2>&1
t "1.6 education PAF ratios with uncertainty (200 draws)"; python code/preprocessing/equity_paf_by_edu_mc.py --draws 200 --jobs 32 > logs/s1_equity_paf_mc.log 2>&1
t "stage 1 done"
