set -e
source code/pipeline/env_main.sh
J=${1:-8}; NG=${2:-41}
mkdir -p data/results/v3/grid
COVS=$(python3 -c "print(' '.join(f'{i/($NG-1):.3f}' for i in range($NG)))")
jobs=()
for c in $COVS afford; do
  jobs+=("SENS_TREAT=maintain SENS_ALLOC=risk|$c|risk_$c")
  jobs+=("SENS_TREAT=maintain SENS_ALLOC=uniform|$c|unif_$c")
done
printf '%s\n' "${jobs[@]}" | xargs -P $J -I{} bash -c '
  IFS="|" read -r envs cov tag <<< "{}"
  env $envs python code/lifetable_v3.py --years 30 --coverage $cov --diet-scenario feasible \
      --out data/results/v3/grid/$tag.csv > data/results/v3/grid/$tag.log 2>&1 || echo "FAILED $tag"'
NG=$NG python3 << 'PYEOF'
import glob, os, numpy as np, pandas as pd
def tot(tag):
    t = pd.read_csv(f'data/results/v3/grid/{tag}.csv'); yr = [c for c in t.columns if c.startswith('yr')]
    return t.groupby(['iso3', 'arm'])[yr].sum().sum(axis=1).unstack('arm')
NG = int(os.environ["NG"]); covs = [round(i / (NG - 1), 3) for i in range(NG)]
res = {}
for al in ['risk', 'unif']:
    g = {c: tot(f'{al}_{c:.3f}') for c in covs}
    diet = g[covs[1]].diet
    glp = pd.DataFrame({c: g[c].glp1 for c in covs})
    cross = {}
    for iso in glp.index:
        y = glp.loc[iso].values; d = diet[iso]; k = np.where(y >= d)[0]
        if len(k) == 0: cross[iso] = np.nan
        elif k[0] == 0: cross[iso] = 0.0
        else:
            i = k[0]; x0, x1, y0, y1 = covs[i - 1], covs[i], y[i - 1], y[i]
            cross[iso] = x0 + (d - y0) / (y1 - y0) * (x1 - x0)
    af = tot(f'{al}_afford')
    at = lambda c: glp.apply(lambda r: np.interp(c, covs, r.values), axis=1)
    res[al] = pd.DataFrame({'crossover_%': pd.Series(cross) * 100, 'ratio_10': diet / at(0.10), 'ratio_20': diet / at(0.20),
                            'ratio_50': diet / at(0.50), 'ratio_afford': diet / af.glp1,
                            'monotone': glp.diff(axis=1).iloc[:, 1:].ge(-1e-6).all(axis=1)})
    if al == 'risk':
        for c in (0.1, 0.2):
            fn = f'data/results/v3/grid/risk_{c:.3f}_check_allocation.csv'
            if os.path.exists(fn): res[al][f'prior_cvd_{int(c*100)}'] = pd.read_csv(fn).set_index('iso3').share_treated_with_prior_cvd
    print(f"\n===== allocation: {al} (target-coverage maintenance); crossover NaN = not reached even at 100% of eligible adults")
    print(res[al].round(2).sort_values('crossover_%').to_string())
    print(f"  diet leads (ratio > 1): 10% {int((res[al].ratio_10.drop('GLB') > 1).sum())}/19, 20% {int((res[al].ratio_20.drop('GLB') > 1).sum())}/19, "
          f"50% {int((res[al].ratio_50.drop('GLB') > 1).sum())}/19, max affordable {int((res[al].ratio_afford.drop('GLB') > 1).sum())}/19")
pd.concat(res, axis=1).to_csv('data/results/v3/grid/summary_crossover.csv')
print('\n-> data/results/v3/grid/summary_crossover.csv')
PYEOF
