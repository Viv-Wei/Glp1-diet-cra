from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(".")
CUR = BASE / "data/gbd/curves"
tm = pd.read_csv(BASE / "data/params/tmrel_gbd2017_converted.csv")

MAP = {
    "low_fruits_ihd": ("v01", "Diet low in fruits", 1.0),
    "low_fruits_Is": ("v01", "Diet low in fruits", 1.0),
    "low_fruits_t2d": ("v01", "Diet low in fruits", 1.0),
    "low_vegetables_ihd": ("v02", "Diet low in vegetables", 1.0),
    "low_vegetables_Is": ("v02", "Diet low in vegetables", 1.0),
    "low_vegetables_t2d": ("v02", "Diet low in vegetables", 1.0),
    "low_legumes_ihd": ("v05", "Diet low in legumes", 1.0),
    "low_nutsseeds_ihd": ("v06", "Diet low in nuts and seeds", 1.0),
    "low_wholegrains_ihd": ("v08", "Diet low in whole grains", 1.0),
    "low_wholegrains_Is": ("v08", "Diet low in whole grains", 1.0),
    "low_wholegrains_t2d": ("v08", "Diet low in whole grains", 1.0),
    "high_procmeat_t2d": ("v09", "Diet high in processed meat", 1.0),
    "high_high_redmeat_ihd": ("v10", "Diet high in red meat", 1.0),
    "high_redmeat_t2d": ("v10", "Diet high in red meat", 1.0),
    "high_ssb_ihd": ("v15", "Diet high in sugar-sweetened beverages", 1.0),
    "high_ssb_t2d": ("v15", "Diet high in sugar-sweetened beverages", 1.0),
    "low_fiber_ihd": ("v34", "Diet low in fibre", 1.0),
    "low_fiber_Is": ("v34", "Diet low in fibre", 1.0),
    "low_fiber_t2d": ("v34", "Diet low in fibre", 1.0),
    "low_seafood_omega3_ihd": ("v30", "Diet low in seafood omega-3 fatty acids", 1.0),
    "low_omega6_ihd": ("v29", "Diet low in polyunsaturated fatty acids", 1.0),
    "high_glucose_ihd": (None, "High fasting plasma glucose", 1.0),
    "high_glucose_istroke": (None, "High fasting plasma glucose", 1.0),
    "high_sbp_ihd": (None, "High systolic blood pressure", 1.0),
    "high_sbp_ckd": (None, "High systolic blood pressure", 1.0),
    "high_glucose_ckd": (None, "High fasting plasma glucose", 1.0),
    "high_sbp_s": (None, "High systolic blood pressure", 1.0),
}

MAP = {k.lower(): v for k, v in MAP.items()}

DEFICIENCY = {"v01", "v02", "v05", "v06", "v08", "v29", "v30", "v34"}


def main():
    recs = []
    for p in sorted(CUR.glob("*.csv")):
        key = p.stem.lower()
        if key not in MAP:
            print(f"  skipped unmapped file: {key}")
            continue
        gdd_var, tm_name, conv = MAP[key]
        df = pd.read_csv(p)
        df.columns = [c.strip() for c in df.columns]
        if "Log relative risk of outcome" not in df.columns:
            print(f"  {key} is not a curve file, skipped")
            continue
        df = df.rename(columns={"Log relative risk of outcome": "logrr", "Risk": "exp"})
        df = df[["exp", "logrr"]].dropna().sort_values("exp")

        row = tm[tm.risk_factor == tm_name]
        if not len(row):
            print(f"  !! {tm_name} not found in the TMREL table")
            continue
        lo, hi = float(row.tmrel_low.iloc[0]), float(row.tmrel_high.iloc[0])
        tmrel = (lo + hi) / 2.0
        _cert = pd.read_csv(BASE / "data/params/tmrel_certified.csv")
        _hit = _cert[_cert.k == key.lower()]
        if len(_hit):
            tmrel = float(_hit.tmrel.iloc[0])

        lr_t = np.interp(tmrel, df.exp, df.logrr)
        df["rr"] = np.exp(df.logrr - lr_t).clip(lower=1.0)
        df["curve_id"] = key
        df["gdd_var"] = gdd_var if gdd_var else ""
        df["risk_factor"] = tm_name
        df["tmrel"] = tmrel
        df["conv_gdd_to_curve"] = conv
        df["burden_type"] = "deficiency" if gdd_var in DEFICIENCY else "excess"
        recs.append(df)

        print(
            f"  {key:<24} TMREL={tmrel:<8.1f} curve range {df.exp.min():.0f}-{df.exp.max():.0f}"
            f"  RR@0={np.exp(np.interp(0,df.exp,df.logrr)-lr_t):.3f}"
        )

    out = pd.concat(recs, ignore_index=True)
    out.to_csv(BASE / "data/params/dose_response_conv_fixed.csv", index=False)
    print(f"\n-> {BASE/'data/params/dose_response_conv_fixed.csv'}")
    print(f"  {out.curve_id.nunique()} curves, {len(out):,} points")


if __name__ == "__main__":
    main()
