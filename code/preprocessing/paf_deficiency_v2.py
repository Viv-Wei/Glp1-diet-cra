import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTCOME_MAP = {
    "ihd": "IHD",
    "t2d": "T2D",
    "s": "IschStroke",
    "Is": "IschStroke",
    "istroke": "IschStroke",
    "is": "IschStroke",
}
MEDIATORS = {"sbp", "glucose", "bmi", "ldl"}

RISK2FACTOR = {
    "Diet low in fruits": "fruits",
    "Diet low in vegetables": "vegetables",
    "Diet low in whole grains": "wholegrains",
    "Diet low in nuts and seeds": "nutsseeds",
    "Diet low in legumes": "legumes",
    "Diet low in fibre": "fiber",
    "Diet low in milk": "milk",
    "Diet low in seafood omega-3 fatty acids": "seafood_omega3",
    "Diet low in polyunsaturated fatty acids": "omega6",
    "Diet high in red meat": "redmeat",
    "Diet high in processed meat": "procmeat",
    "Diet high in sugar-sweetened beverages": "ssb",
    "Diet high in trans fatty acids": "transfat",
    "Diet high in sodium": "sodium",
}
CAUSE2OUTCOME = {
    "Ischemic heart disease": "IHD",
    "Ischaemic heart disease": "IHD",
    "Ischaemic stroke": "IschStroke",
    "Intracerebral haemorrhage": "ICH",
    "Subarachnoid haemorrhage": "SAH",
    "Ischemic stroke": "IschStroke",
    "Diabetes mellitus type 2": "T2D",
}


def parse_curve_id(cid):
    c = re.sub(r"^(high|low)_(high|low)_", r"\1_", cid)
    m = re.match(r"^(high|low)_(.+?)_(ihd|t2d|istroke|is|Is|s)$", c)
    return (
        (m.group(2), OUTCOME_MAP[m.group(3)], m.group(1))
        if m
        else (cid, "UNPARSED", "?")
    )


class Curve:

    def __init__(self, sub):
        s = sub.sort_values("exp")
        self.x = s.exp.to_numpy(float)
        self.tmrel = float(s.tmrel.iloc[0])
        c = s.conv_gdd_to_curve.iloc[0]
        self.conv = float(c) if pd.notna(c) else 1.0
        self.j_interval = (0.0, 200.0) if "redmeat" in str(s.curve_id.iloc[0]) else None
        self.lo, self.hi = self.x.min(), self.x.max()
        rr = s.rr.to_numpy(float)
        if not (self.lo - 1e-9 <= self.tmrel <= self.hi + 1e-9):
            print(
                f"[warning] {s.curve_id.iloc[0]}: TMREL={self.tmrel:.4g} lies outside the curve range "
                f"[{self.lo:.4g},{self.hi:.4g}]; the normalisation anchor is clamped and PAF may be biased."
            )
        self.y = rr / float(
            np.interp(np.clip(self.tmrel, self.lo, self.hi), self.x, rr)
        )
        if self.j_interval is not None:
            j_lo, j_hi = self.j_interval
            in_j = (self.x >= j_lo) & (self.x <= j_hi)
            if in_j.any():
                self.y = self.y / self.y[in_j].min()
                self.y[in_j] = 1.0

    def rr(self, v, strict=True):
        v = np.asarray(v, float)
        if strict and (np.nanmin(v) < self.lo - 1e-9 or np.nanmax(v) > self.hi + 1e-9):
            raise ValueError(
                f"Exposure outside the curve range [{self.lo:.4g},{self.hi:.4g}], "
                f"observed [{np.nanmin(v):.4g},{np.nanmax(v):.4g}]."
                f"likely conv_gdd_to_curve was not applied or the coefficients were not re-checked after switching tables."
            )
        out = np.interp(np.clip(v, self.lo, self.hi), self.x, self.y)
        return np.maximum(out, 1.0)


def paf_mean(curve, intake_gdd):
    return 1.0 - 1.0 / float(curve.rr(intake_gdd * curve.conv))


def paf_dist(curve, intake_gdd, cv, n=200_000, seed=0):
    m = intake_gdd * curve.conv
    if m <= 0 or cv <= 0:
        return paf_mean(curve, intake_gdd)
    sigma = np.sqrt(np.log1p(cv**2))
    mu = np.log(m) - sigma**2 / 2
    x = np.random.default_rng(seed).lognormal(mu, sigma, n)
    return 1.0 - 1.0 / float(curve.rr(x, strict=False).mean())


def load_mediation(path):
    md = pd.read_csv(path)
    fixed = 0
    for i, r in md.iterrows():
        c = str(r["cause"])
        if c.strip().startswith("in adults"):
            md.at[i, "mediator"] = str(r["mediator"]).strip() + " in adults"
            md.at[i, "cause"] = re.sub(r"^\s*in adults\s+", "", c).strip()
            fixed += 1
    if fixed:
        print(
            f'[mediation matrix] fixed {fixed} rows with label contamination ("in adults" leaked into the cause column)'
        )
        print(
            "           This is only a patch; the SSB-via-BMI rows for IHD/stroke in Table S6 are most likely still missing and need to be re-extracted"
        )
    md["factor"] = md.risk_factor.map(RISK2FACTOR)
    md["outcome"] = md.cause.map(CAUSE2OUTCOME)
    return md.dropna(subset=["factor", "outcome"])


def substitutability(md):
    rec = {}
    for (f, o), g in md.groupby(["factor", "outcome"]):
        chans = {}
        for _, r in g.iterrows():
            med = str(r.mediator)
            k = (
                "BMI"
                if "body-mass" in med
                else (
                    "FPG"
                    if "fasting plasma glucose" in med
                    else (
                        "SBP"
                        if "systolic blood pressure" in med
                        else "LDL" if "LDL" in med else "other"
                    )
                )
            )
            chans[k] = max(chans.get(k, 0.0), float(r.mf))
        if "BMI" in chans:
            lab = "BMI channel - substitutable"
        elif chans.get("FPG", 0) >= 0.99:
            lab = "FPG channel - fully substitutable"
        elif (
            max([v for k, v in chans.items() if k in ("FPG", "SBP", "LDL")] or [0])
            >= 0.5
        ):
            lab = "Metabolic channel - partly substitutable"
        else:
            lab = "Non-metabolic mediator - not substitutable"
        rec[(f, o)] = dict(
            glp1_channel=lab,
            mf_BMI=chans.get("BMI"),
            mf_FPG=chans.get("FPG"),
            mf_SBP=chans.get("SBP"),
            mf_LDL=chans.get("LDL"),
        )
    return rec


def load_channels(path):
    ch = pd.read_csv(path)
    need = {"factor", "outcome", "glp1_channel", "convergence"}
    miss = need - set(ch.columns)
    if miss:
        sys.exit(
            f"Convergence table is missing columns {miss}; generate it with channels_gbd2021.py"
        )
    out = {}
    for _, r in ch.iterrows():
        out[(r.factor, r.outcome)] = dict(
            glp1_channel=r.glp1_channel,
            convergence=float(r.convergence),
            via_mediator=(r.via_mediator if pd.notna(r.get("via_mediator")) else None),
            mf_FPG=r.get("mf_FPG"),
            mf_SBP=r.get("mf_SBP"),
            mf_LDL=r.get("mf_LDL"),
        )
    return out


def _star_num(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int,)) or (isinstance(v, float) and not pd.isna(v)):
        return int(v)
    m = {"1 star": 1, "2 stars": 2, "3 stars": 3, "4 stars": 4, "5 stars": 5}
    t = str(v).strip()
    return m.get(t, int(t) if t.isdigit() else None)


def load_evidence(path):
    ev = pd.read_csv(path)
    out = {}
    for _, r in ev.iterrows():
        out[r.curve_id] = dict(
            stars=_star_num(r.get("stars", r.get("stars_txt"))),
            ros=(float(r.ros) if pd.notna(r.ros) else None),
            outcome_actual=r.get("outcome_actual"),
            outcome_match=bool(r.get("match", True)),
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--exposure", required=True)
    ap.add_argument(
        "--mediation",
        default=None,
        help="Fallback for the legacy definition. Prefer --channels",
    )
    ap.add_argument(
        "--channels",
        default=None,
        help="Convergence table produced by channels_gbd2021.py (recommended)",
    )
    ap.add_argument("--year", type=int, default=2018)
    ap.add_argument("--cv", type=float, default=0.6)
    ap.add_argument(
        "--cv-file",
        default=None,
        help="Per-factor population CV table (columns: gdd_var, cv_pop). Matched factors use the table value; unmatched fall back to --cv",
    )
    ap.add_argument("--exclude", default="omega6")
    ap.add_argument("--paf-method", choices=["mean", "dist"], default="mean")
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--min-stars", type=int, default=None)
    ap.add_argument("--max-ui-ratio", type=float, default=None)
    ap.add_argument("--out", default="data/results/paf_denominator.csv")
    a = ap.parse_args()

    CVMAP = {}
    if a.cv_file:
        _cvt = pd.read_csv(a.cv_file)
        CVMAP = dict(zip(_cvt.gdd_var, _cvt.cv_pop))
        print(
            f"[CV] per-factor CV table {a.cv_file}: {len(CVMAP)} factors, unmatched fall back to {a.cv}"
        )

    exclude = {x.strip() for x in a.exclude.split(",") if x.strip()}
    if exclude:
        print(f"[excluded factors] {sorted(exclude)}")

    dr = pd.read_csv(a.curves)
    if "conv_gdd_to_curve" not in dr.columns:
        dr["conv_gdd_to_curve"] = 1.0
    curves = {cid: Curve(sub) for cid, sub in dr.groupby("curve_id")}
    cmeta = {cid: parse_curve_id(cid) for cid in curves}
    gddvar = dr.groupby("curve_id").gdd_var.first().to_dict()

    expo = pd.read_csv(a.exposure)
    need = {"gdd_var", "iso3", "year", "age", "stratum_type", "intake_median"}
    miss = need - set(expo.columns)
    if miss:
        sys.exit(
            f"Exposure table is missing columns {miss}. Use the new table produced by gdd_extract.py"
        )

    yrs = sorted(expo.year.unique())
    if a.year not in yrs:
        sys.exit(f"--year {a.year} is not in the data. Available: {yrs}")
    print(f"[year] locked to {a.year} (table contains {yrs})")
    n_before = expo.gdd_var.nunique()
    expo = expo[(expo.year == a.year) & (expo.stratum_type == "age_marginal")]
    n_after = expo.gdd_var.nunique()
    if n_after < n_before:
        print(
            f"  !! after locking to this year, {n_before-n_after} factors have no data and were dropped"
        )

    if "intake_hi95" in expo.columns and "intake_lo95" in expo.columns:
        expo = expo.assign(
            ui_ratio=expo.intake_hi95 / expo.intake_lo95.replace(0, np.nan)
        )
    else:
        expo = expo.assign(ui_ratio=np.nan)

    if a.max_ui_ratio is not None:
        n0 = len(expo)
        expo = expo[expo.ui_ratio.isna() | (expo.ui_ratio <= a.max_ui_ratio)]
        print(
            f"[data quality] removed {n0-len(expo)} cells with ui_ratio>{a.max_ui_ratio}"
        )

    EV = load_evidence(a.evidence) if a.evidence else {}
    if EV:
        mism = [c for c, v in EV.items() if not v["outcome_match"]]
        if mism:
            print(
                f"[outcome check] !! {len(mism)} curves have mismatched outcome definitions: {mism}"
            )
            print(
                "            These curves will have systematic PAF bias and must be flagged in the text"
            )
        n_ns = sum(1 for v in EV.values() if v["stars"] is None)
        print(f"[evidence strength] {a.evidence}  loaded {len(EV)-n_ns} star ratings")

    idx = expo.set_index(["iso3", "gdd_var", "age"])
    if a.channels:
        sub = load_channels(a.channels)
        print(f"[channel classification] convergence table {a.channels}")
        CH_DEFAULT = "Direct effect only - structurally unreachable"
    elif a.mediation:
        sub = substitutability(load_mediation(a.mediation))
        print(f"[channel classification] legacy definition {a.mediation}")
        print(
            "  !! In GBD 2021 no dietary factor is mediated by BMI; the legacy method would systematically classify all as unreachable"
        )
        print("     Use --channels instead")
        CH_DEFAULT = (
            "No mediation record (treated as direct effect; possibly an extraction gap)"
        )
    else:
        sub, CH_DEFAULT = {}, "Unclassified"

    rows, skipped = [], []
    for cid, cur in curves.items():
        fac, out, direction = cmeta[cid]
        if fac in MEDIATORS or fac in exclude:
            continue
        ev = EV.get(cid, {})
        if a.min_stars is not None and (ev.get("stars") or 0) < a.min_stars:
            skipped.append(
                (cid, f"star rating {ev.get('stars')} < {a.min_stars}; excluded")
            )
            continue
        gv = gddvar.get(cid)
        if pd.isna(gv):
            skipped.append((cid, "no gdd_var mapping"))
            continue
        for iso in sorted(expo.iso3.unique()):
            for age in sorted(expo.age.unique()):
                k = (iso, gv, age)
                if k not in idx.index:
                    continue
                row = idx.loc[k]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                intake = float(row.intake_median)
                try:
                    pm = paf_mean(cur, intake)
                except ValueError as e:
                    skipped.append((cid, f"{iso} age{age}: {e}"))
                    continue
                at_tr = (
                    bool(row.at_truncation) if "at_truncation" in row.index else False
                )
                pdst = np.nan if at_tr else paf_dist(cur, intake, CVMAP.get(gv, a.cv))
                tag = sub.get((fac, out), {})
                rows.append(
                    dict(
                        iso3=iso,
                        age=age,
                        year=a.year,
                        factor=fac,
                        outcome=out,
                        direction=direction,
                        intake_median=intake,
                        intake_curve=intake * cur.conv,
                        tmrel=cur.tmrel,
                        PAF_mean=round(pm, 5),
                        PAF_dist=(round(pdst, 5) if pd.notna(pdst) else np.nan),
                        at_truncation=at_tr,
                        ui_ratio=(
                            round(float(row.ui_ratio), 2)
                            if pd.notna(row.get("ui_ratio", np.nan))
                            else np.nan
                        ),
                        glp1_channel=tag.get("glp1_channel", CH_DEFAULT),
                        convergence=tag.get("convergence"),
                        via_mediator=tag.get("via_mediator"),
                        mf_BMI=tag.get("mf_BMI"),
                        mf_FPG=tag.get("mf_FPG"),
                        mf_SBP=tag.get("mf_SBP"),
                        mf_LDL=tag.get("mf_LDL"),
                        stars=ev.get("stars"),
                        ros=ev.get("ros"),
                        outcome_match=ev.get("outcome_match", True),
                        curve_id=cid,
                    )
                )

    if not rows:
        sys.exit("No PAF was computed. Check the gdd_var mapping and the year")

    res = pd.DataFrame(rows)
    for k in ("PAF_mean", "PAF_dist"):
        res[k.replace("PAF", "reachable")] = res[k] * res.convergence.fillna(0)
    key = "PAF_mean" if a.paf_method == "mean" else "PAF_dist"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.sort_values(["iso3", "outcome", "factor", "age"]).to_csv(a.out, index=False)
    print(f"\nwrote {len(res):,} rows (by age) -> {a.out}")
    if skipped:
        print(f"\nskipped {len(skipped)} items:")
        for c, why in skipped[:10]:
            print(f"   {c}: {why}")

    if a.paf_method == "dist":
        n_tr = int(res.at_truncation.sum())
        if n_tr:
            print(
                f"\n!! under the distribution method, {n_tr} truncated cells have empty PAF_dist (right tail flattened)"
            )

    print(
        "\n!! The summary below averages age groups without weighting and is for a quick look only."
    )
    print(
        "   Final results must weight the age-specific PAF in the CSV by the age-specific GBD burden"
    )
    show = res.groupby(
        ["iso3", "factor", "outcome", "glp1_channel"], as_index=False
    ).agg(
        {
            key: "mean",
            "intake_curve": "mean",
            "tmrel": "first",
            "at_truncation": "any",
            "ui_ratio": "mean",
            "convergence": "first",
            "stars": "first",
            "outcome_match": "all",
        }
    )

    print(
        f"\n===== IHD attributable fractions by country (definition={key}, descending)====="
    )
    for iso, g in (
        show[show.outcome == "IHD"].sort_values(key, ascending=False).groupby("iso3")
    ):
        print(f"\n{iso}")
        for _, r in g.iterrows():
            flag = " [truncated]" if r.at_truncation else ""
            flag += (
                f" [UI×{r.ui_ratio:.1f}]"
                if pd.notna(r.ui_ratio) and r.ui_ratio > 3
                else ""
            )
            cv = (
                f" convergence={r.convergence:.3f}"
                if pd.notna(r.get("convergence"))
                else ""
            )
            if pd.notna(r.get("stars")):
                cv += f" {int(r.stars)} stars"
            if r.get("outcome_match") is False:
                flag = " [outcome definition mismatch]" + flag
            print(
                f"   {r.factor:16s} intake={r.intake_curve:8.2f} TMREL={r.tmrel:7.2f} "
                f"PAF={100*r[key]:6.2f}%{cv}  [{r.glp1_channel}]{flag}"
            )

    print(f"\n===== Combined PAF (multiplicative 1-prod(1-p), definition={key})=====")
    print(
        "(Assumes factors are independent; the final version must use the mediation matrix. Channel shares sum to >100%, which is inherent to multiplicative combination)"
    )
    for (iso, out), g in show.groupby(["iso3", "outcome"]):
        v = g[key].dropna().to_numpy()
        tot = 1 - np.prod(1 - v)
        line = f"  {iso} {out:11s} combined PAF={100*tot:6.2f}%"
        for ch, gg in g.groupby("glp1_channel"):
            vv = gg[key].dropna().to_numpy()
            s = 1 - np.prod(1 - vv)
            line += f" | {ch.split('(')[0]}={100*s/max(tot,1e-12):.0f}%"
        print(line)

    print(
        "\n===== Age gradient (IHD, PAF difference oldest minus youngest group, percentage points)====="
    )
    ag = (
        res[res.outcome == "IHD"]
        .sort_values("age")
        .groupby(["iso3", "factor"])
        .agg(young=(key, "first"), old=(key, "last"))
    )
    ag["diff_pp"] = (ag.old - ag.young) * 100
    top = ag.reindex(ag["diff_pp"].abs().sort_values(ascending=False).index).head(10)
    out = pd.DataFrame(
        {
            "youngest group %": (top.young * 100).round(2),
            "oldest group %": (top.old * 100).round(2),
            "difference (pp)": top.diff_pp.round(2),
        }
    )
    print(out.to_string())
    print(
        "  -> A large difference means pooling across ages would be badly distorted, confirming the need to keep the age dimension"
    )


if __name__ == "__main__":
    main()
