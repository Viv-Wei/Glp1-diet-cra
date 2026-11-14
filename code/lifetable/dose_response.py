import re

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
