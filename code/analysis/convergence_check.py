import argparse, glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--mc", default="data/results/v3/mc")
ap.add_argument("--jobs", type=int, default=32)
A = ap.parse_args()


def read(f):
    x = pd.read_csv(f, usecols=["draw", "cov", "iso3", "arm", "deaths"])
    return x[x.arm.isin(["diet", "glp1"])]


fs = [
    f
    for c in ("0.100", "0.200", "afford")
    for f in glob.glob(f"{A.mc}/out/d*_c{c}.csv")
]
with ProcessPoolExecutor(A.jobs) as ex:
    x = pd.concat(ex.map(read, fs, chunksize=64))
x["cov"] = (
    x["cov"].astype(str).map(lambda c: "afford" if c == "afford" else f"{float(c):.3f}")
)
w = x.pivot_table(index=["draw", "iso3"], columns=["cov", "arm"], values="deaths")
q = pd.DataFrame(
    {
        "diet": w[("0.200", "diet")],
        "glp1_10": w[("0.100", "glp1")],
        "glp1_20": w[("0.200", "glp1")],
        "glp1_max": w[("afford", "glp1")],
    }
)
q["ratio_20"] = q.diet / q.glp1_20
q = q.reset_index()
n = q.draw.nunique()
half = n // 2
assert n >= 2 and sorted(q.draw.unique())[-1] == n - 1, "draws are not numbered 0..n-1"
stat = {
    "median": lambda v: np.median(v),
    "lo": lambda v: np.quantile(v, 0.025),
    "hi": lambda v: np.quantile(v, 0.975),
}
rows = []
for iso, g in q.groupby("iso3"):
    a, b = g[g.draw < half], g[g.draw >= half]
    for col in ["diet", "glp1_10", "glp1_20", "glp1_max", "ratio_20"]:
        for s, f in stat.items():
            full, h1, h2 = f(g[col].values), f(a[col].values), f(b[col].values)
            rows.append(
                dict(
                    iso3=iso,
                    quantity=col,
                    stat=s,
                    full=full,
                    half1=h1,
                    half2=h2,
                    rel_diff_pct=100 * abs(h1 - h2) / abs(full) if full else np.nan,
                )
            )
r = pd.DataFrame(rows)
r.to_csv(f"{A.mc}/convergence_halves.csv", index=False)
print(f"{n} draws, halves of {half}: draws 0-{half-1} vs {half}-{n-1}\n")
r["kind"] = np.where(r.stat == "median", "median", "UI bound")
for scope, d in [("global (GLB)", r[r.iso3 == "GLB"]), ("all 20 locations", r)]:
    t = d.groupby("kind").rel_diff_pct.agg(["median", "max"]).round(2)
    print(f"{scope}: relative difference between halves, %\n{t.to_string()}\n")
m = r[r.kind == "UI bound"].nlargest(5, "rel_diff_pct")[
    ["iso3", "quantity", "stat", "half1", "half2", "rel_diff_pct"]
]
print("largest differences in UI bounds:\n" + m.round(3).to_string(index=False))
print(f"\n-> {A.mc}/convergence_halves.csv")
