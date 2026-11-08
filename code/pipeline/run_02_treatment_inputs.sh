set -e; source code/pipeline/env_main.sh; t() { echo "$(date '+%F %T') $*"; }; mkdir -p logs
t "2.1 BMI >=25 prevalence";             python code/preprocessing/build_overweight_prevalence.py > logs/s2_overweight.log
t "2.2 BMI distribution and diabetes";   python code/preprocessing/build_select_elig.py > logs/s2_select_elig.log
t "2.3 overweight among people with IHD or IS"; python code/preprocessing/build_overweight_cvd.py > logs/s2_overweight_cvd.log
t "2.4 eligibility at BMI >=27";         python code/preprocessing/build_bmi27_params.py > logs/s2_bmi27.log
t "2.5 gamma BMI distribution (sensitivity)"; python code/preprocessing/build_bmi_gamma.py > logs/s2_bmi_gamma.log
t "2.6 income quintiles and affordability"
python code/preprocessing/parse_income_quintiles.py > logs/s2_income.log; python code/preprocessing/add_nzl_quintiles.py >> logs/s2_income.log
python code/preprocessing/affordability_model.py > logs/s2_affordability.log
python code/preprocessing/build_global_income_quintiles.py > logs/s2_global_income.log; python code/preprocessing/build_coverage_afford_cap.py > logs/s2_afford_cap.log
t "stage 2 done"
