set -e
J=${1:-12}
source code/pipeline/env_main.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SENS_TREAT=maintain SENS_ALLOC=risk SENS_T2D=1 SENS_CKD=1
export LT_DP_OVERRIDE=${LT_DP_OVERRIDE:-data/results/v3/diet_policy_arm_v2_tailored.csv}
OUT=data/results/v3/single_factor; mkdir -p $OUT data/tables
CODES="v01 v02 v05 v06 v08 v09 v10 v15 v29 v30 v34 v37"
for gv in $CODES; do
  python -c "import pandas as pd,sys; s=set(pd.read_csv('$LT_DP_OVERRIDE').scenario); sys.exit(0 if 'single_$gv' in s else 1)" \
    || { echo "scenario single_$gv not found in $LT_DP_OVERRIDE"; exit 1; }
done
for gv in $CODES; do
  echo "python code/lifetable_v3.py --years 30 --coverage 0.20 --diet-scenario single_$gv --out $OUT/single_$gv.csv > $OUT/single_$gv.log 2>&1"
done | xargs -P $J -I{} sh -c "{}"
python - << 'PY'
import glob, os, pandas as pd
LBL = {"v01": "Fruits", "v02": "Vegetables", "v05": "Legumes", "v06": "Nuts & seeds", "v08": "Whole grains",
       "v09": "Processed meat", "v10": "Red meat", "v15": "SSB", "v29": "Omega-6", "v30": "Seafood omega-3",
       "v34": "Fibre", "v37": "Sodium"}
rows = []
for f in sorted(glob.glob("data/results/v3/single_factor/single_v*.csv")):
    if any(s in f for s in ("_t2d", "_ckd", "_byage", "_check_")): continue
    gv = os.path.basename(f)[len("single_"):-4]
    x = pd.read_csv(f); x = x[x.arm == "diet"]; yr = [c for c in x.columns if c.startswith("yr")]
    g = x.groupby(["iso3", "outcome"])[yr].sum().sum(axis=1).unstack("outcome")
    t2 = pd.read_csv(f[:-4] + "_t2d.csv"); t2 = t2[t2.arm == "diet"].set_index("iso3").cases_averted_total
    ck = pd.read_csv(f[:-4] + "_ckd.csv") if os.path.exists(f[:-4] + "_ckd.csv") else None
    ckv = ck[ck.arm == "diet"].set_index("iso3").deaths_averted_total if ck is not None else None
    for iso in g.index:
        rows.append({"Country": iso, "GDD variable code": gv, "Dietary factor": LBL[gv],
                     "IHD deaths averted (30-year)": g.loc[iso].get("IHD", 0.0),
                     "Ischaemic stroke deaths averted (30-year)": g.loc[iso].get("IschStroke", 0.0),
                     "CKD deaths averted (30-year)": float(ckv.get(iso, float("nan"))) if ckv is not None else float("nan"),
                     "T2D incident cases averted (30-year)": float(t2.get(iso, float("nan")))})
w = pd.DataFrame(rows)
w["Total CVD deaths averted (30-year)"] = w["IHD deaths averted (30-year)"] + w["Ischaemic stroke deaths averted (30-year)"]
tot = w.groupby("Country")["Total CVD deaths averted (30-year)"].transform(lambda v: v.clip(lower=0).sum())
w["Share of country CVD benefit (%)"] = (w["Total CVD deaths averted (30-year)"] / tot * 100).round(1)
t2tot = w.groupby("Country")["T2D incident cases averted (30-year)"].transform(lambda v: v.clip(lower=0).sum())
w["Share of country T2D benefit (%)"] = (w["T2D incident cases averted (30-year)"] / t2tot * 100).round(1)
num = [c for c in w.columns if "averted" in c]; w[num] = w[num].round(0)
w.to_csv("data/tables/table_single_factor_v3.csv", index=False)
top = w[w.Country != "GLB"].sort_values("Share of country CVD benefit (%)", ascending=False).groupby("Country").head(1)
print("Top factor by country:"); print(top[["Country", "Dietary factor", "Share of country CVD benefit (%)"]].to_string(index=False))
print("\nCount of countries by top factor:"); print(top["Dietary factor"].value_counts().to_string())
print("-> data/tables/table_single_factor_v3.csv")
PY
