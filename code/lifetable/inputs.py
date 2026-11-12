import os
import pandas as pd
from .settings import AGE5, SEXES

cm = pd.read_csv("data/params/country_map.csv")
ISO = dict(zip(cm.gbd_location_name, cm.iso3))              # GBD location name -> ISO3 (GLB = global aggregate)


def gbd(fname, measure):
    """GBD 2023 rate per person (val / 1e5) for one measure, indexed by (iso3, sex, five-year age group).
    Input: a file in data/gbd/cause_totals/ and a measure name (Incidence, Prevalence, Deaths). Cached as a pickle."""
    src = f"data/gbd/cause_totals/{fname}"
    cache = f"{os.environ.get('LT_CACHE_DIR', 'data/results/v3/cache')}/{fname}.{measure}.{int(os.path.getmtime(src))}.pkl"
    if os.path.exists(cache):
        try: return pd.read_pickle(cache)
        except Exception: pass
    out = _gbd_read(src, measure)
    try:
        os.makedirs(os.path.dirname(cache), exist_ok=True); tmp = f"{cache}.{os.getpid()}"; out.to_pickle(tmp); os.replace(tmp, cache)
    except Exception: pass
    return out


def _gbd_read(src, measure):
    d = pd.read_csv(src, low_memory=False)
    d = d[(d.measure_name == measure) & (d.metric_name == "Rate") & (d.year == 2023) & d.age_name.isin(AGE5)
          & d.sex_name.isin(SEXES) & d.location_name.isin(ISO)]
    return (d.assign(iso3=d.location_name.map(ISO)).set_index(["iso3", "sex_name", "age_name"]).val / 1e5)


def load_pop():
    """GBD 2023 population by (iso3, sex, five-year age group)."""
    d = pd.read_csv("data/gbd/1-3_population.csv", low_memory=False)
    d = d[(d.year == 2023) & d.age_name.isin(AGE5) & d.sex_name.isin(SEXES) & d.location_name.isin(ISO)]
    return d.assign(iso3=d.location_name.map(ISO)).set_index(["iso3", "sex_name", "age_name"]).val


def a0_(s):
    """Lower bound of an age label ("45 to 49", "45-49 years") as a number."""
    return pd.to_numeric(pd.Series(s).astype(str).str.extract(r"^(\d+)")[0], errors="coerce").values


def _load_diet_module():
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader("diet_policy_arm", "code/preprocessing/diet_policy_arm.py").load_module()


def sodium_drop(expo_sx, iso, scenario, dmod=None):
    """Drop in excess sodium (g/day above 3.0) by five-year age group under the dietary scenario (feasible: -15%)."""
    e = expo_sx[expo_sx.iso3 == iso]
    out = {}
    for _, r in e.iterrows():
        x0 = float(r.intake_median)
        if dmod is not None:
            mode_x = dmod.TAILORED.get(iso, {}).get("v37") if scenario == "tailored" else dmod.SCENARIOS[scenario].get("v37")
        else:
            mode_x = ("pct", -0.15)
        if mode_x is None: x1 = x0
        elif mode_x[0] == "pct": x1 = x0 * (1 + mode_x[1])
        else: x1 = x0 + mode_x[1] * (3.0 - x0)
        out[int(r.a0)] = max(x0 - 3.0, 0) - max(x1 - 3.0, 0)
    return out


def dietary_pif(dp_file, scenario):
    """Share of deaths averted by the dietary scenario, by (outcome, iso3), from the dietary policy file."""
    dp = pd.read_csv(dp_file)
    if scenario not in set(dp.scenario):
        raise SystemExit(f"[error] dietary scenario '{scenario}' not found in {dp_file}")
    dp = dp[dp.scenario == scenario].groupby(["outcome", "iso3"]).averted_frac_alldeaths.mean()
    if dp.max() > 1: dp = dp / 100
    return dp
