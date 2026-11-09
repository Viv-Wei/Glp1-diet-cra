set -e
source code/pipeline/env_main.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p data/results/v3
ISO="USA GLB DEU GBR CAN JPN AUS CHN"
echo "######## main (s_IHD 0.40, fatal first-event effect same as non-fatal)"
python code/preprocessing/select_check_v3.py --iso $ISO --out data/results/v3/select_main.csv
echo; echo "######## variants, USA and global only"
for v in "SENS_IHD_FIRST=0.25" "SENS_IHD_FIRST=0.55" "SENS_FATAL_MODE=cvd" "SENS_FATAL_MODE=none"; do
  echo "== $v"; env $v python code/preprocessing/select_check_v3.py --iso USA GLB --out data/results/v3/select_${v//=/_}.csv | sed -n '2,5p'
done
echo "== half of the >=2-condition group has both MI and stroke"
python code/preprocessing/select_check_v3.py --iso USA GLB --both-share 0.5 --out data/results/v3/select_both05.csv | sed -n '2,5p'
