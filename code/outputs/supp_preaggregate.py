import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", type=int, default=32)
A = ap.parse_args()
O = "data/results/v3/supp"
os.makedirs(O, exist_ok=True)


def norm(c):
    return "afford" if str(c) == "afford" else f"{float(c):.3f}"


def load(pattern_dir, covs, cols):
    fs = [f for c in covs for f in glob.glob(f"{pattern_dir}/d*_c{c}.csv")]
    with ProcessPoolExecutor(A.jobs) as ex:
        x = pd.concat(ex.map(_read, [(f, tuple(cols)) for f in fs], chunksize=64))
    x["cov"] = x["cov"].map(norm)
    return x, len(fs)


def _read(a):
    return pd.read_csv(a[0], usecols=list(a[1]))


x, n = load(
    "data/results/v3/mc/out",
    ("0.100", "0.200", "0.500", "afford"),
    ["draw", "cov", "iso3", "arm", "deaths"],
)
w = (
    x[x.arm.isin(["diet", "glp1"])]
    .pivot_table(index=["cov", "iso3", "draw"], columns="arm", values="deaths")
    .reset_index()
)
p = (
    w.assign(diet_leads=w.diet > w.glp1)
    .groupby(["iso3", "cov"])
    .agg(p_diet_leads=("diet_leads", "mean"), draws=("draw", "nunique"))
    .reset_index()
)
p.to_csv(f"{O}/prob_diet_leads.csv", index=False)
print(
    f"{n} files -> {O}/prob_diet_leads.csv; global at 20%: dietary policy averts more in "
    f"{100 * p[(p.iso3 == 'GLB') & (p['cov'] == '0.200')].p_diet_leads.iloc[0]:.1f}% of draws"
)

x, n = load(
    "data/results/v3/mc_t2dckd/out",
    ("0.100", "0.200", "afford"),
    ["draw", "cov", "iso3", "arm", "deaths", "t2d", "ckd"],
)
rows = []
for (iso, cv, arm), g in x.groupby(["iso3", "cov", "arm"]):
    for col in ("deaths", "t2d", "ckd"):
        v = g[col].values
        rows.append(
            dict(
                iso3=iso,
                cov=cv,
                arm=arm,
                outcome=col,
                median=np.median(v),
                lo=np.quantile(v, 0.025),
                hi=np.quantile(v, 0.975),
                draws=g.draw.nunique(),
            )
        )
pd.DataFrame(rows).to_csv(f"{O}/outcomes_by_arm.csv", index=False)
print(f"{n} files -> {O}/outcomes_by_arm.csv ({len(rows)} rows)")
