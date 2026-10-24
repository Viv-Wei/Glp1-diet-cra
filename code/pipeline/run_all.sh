set -e; J=${1:-120}; mkdir -p logs
bash code/pipeline/run_01_dietary_inputs.sh 32
bash code/pipeline/run_02_treatment_inputs.sh
python -m pytest code/tests -q
bash code/pipeline/run_04_deterministic.sh $J
bash code/pipeline/run_03_monte_carlo.sh $J
bash code/pipeline/run_05_tables_figures.sh
echo "$(date '+%F %T') all done"
