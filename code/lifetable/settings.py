# -*- coding: utf-8 -*-
"""Constants and the settings of one run.

Settings are read from SENS_* environment variables, so that Monte Carlo draws and scenarios can be run by setting
variables rather than editing code. read_settings() collects them in one place; every value has the default of the main
analysis. CRA_ROOT sets the project root (default: the current directory).
"""
import os
from types import SimpleNamespace
import numpy as np

E = os.environ.get
os.chdir(E("CRA_ROOT", os.getcwd()))

AGE5 = [f"{a}-{a+4} years" for a in range(25, 95, 5)] + ["95+ years"]   # GBD five-year age groups
G0 = list(range(25, 95, 5)) + [95]                       # lower bound of each five-year group
AGES = np.arange(25, 96)                                 # single-year ages, 95 = 95+
GIDX = np.minimum((AGES - 25) // 5, len(AGE5) - 1)       # five-year group index of each single-year age
SEXES = ["Male", "Female"]
ST = ["N", "I", "S", "B"]                                # neither, IHD only, IS only, both
ARMS = ["baseline", "glp1", "diet", "combo"]             # the baseline arm runs first (it sets the diabetes split)
# compartments within each sex and state: the original structure (T enrolled, E untreated eligible, O ineligible), or
# BMI classes L (<27), M (27 to <30) and H (>=30), each enrolled (T) or not (U), with eligibility evaluated each year
ELIG_STRUCT = os.environ.get("SENS_ELIG_STRUCT", "bmi")   # main analysis: BMI structure
KS = ["T", "E", "O"] if ELIG_STRUCT == "legacy" else ["T0", "T1", "T2", "T3", "U0", "U1", "U2", "U3"]


def read_settings(coverage):
    """All settings of one run, with the defaults of the main analysis.

    Input: coverage, the --coverage argument (a share of eligible adults, "afford" or "cstar").
    Output: a namespace; names follow the environment variables (SENS_IHD_FIRST -> ihd_first, and so on)."""
    s = SimpleNamespace()
    # life-table structure
    s.ihd_first = float(E("SENS_IHD_FIRST", 0.40))       # share of IHD deaths occurring as fatal first events
    s.is_cf28 = float(E("SENS_IS_CF28", 0.15))           # 28-day case fatality of first ischaemic stroke
    s.or_comorb = float(E("SENS_OR_COMORB", 1.0))        # odds ratio of having both IHD and IS at baseline
    s.rr_cond = float(E("SENS_RR_COND", 1.0))            # risk of the second condition after the first
    s.k_dur = float(E("SENS_K_DUR", 1.0))                # relative duration of IHD or IS with BMI >=27
    s.open_pop = E("SENS_OPEN_POP") == "1"               # people turning 25 enter each year
    # treatment effects
    s.fatal_mode = E("SENS_FATAL_MODE", "same")          # effect on fatal first events: same | cvd | none
    s.post_none = E("SENS_POST_NONE") == "1"             # no effect on deaths after onset
    s.hr_env = {k: E(k) for k in ("SENS_HR_MI", "SENS_HR_STROKE", "SENS_HR_CVD", "SENS_HR_MI_DM", "SENS_HR_STROKE_DM", "SENS_HR_CVD_DM")}
    s.primary_scale = float(E("SENS_PRIMARY_SCALE", 1.0))   # first-event effects in people without CVD
    s.tier3_scale = float(E("SENS_TIER3_SCALE", 1.0))       # first-event effects in tier 3
    s.tier2_mi, s.tier2_stroke = E("SENS_TIER2_MI"), E("SENS_TIER2_STROKE")   # e.g. REWIND 0.96, 0.76 for tier 2
    s.retain = float(E("SENS_RETAIN", 0.76))             # constant retention, used only with SENS_ONOFF=0
    s.cap_ref = E("SENS_CAP_REF")                        # e.g. USA: absolute benefit capped at that country's risk
    # programme and eligibility
    s.treat = E("SENS_TREAT", "maintain")                # maintain | baseline | dynamic
    s.alloc = E("SENS_ALLOC", "risk")                    # risk | uniform
    s.bmi_min = E("SENS_BMI_MIN", "27")
    s.elig_struct = ELIG_STRUCT                          # legacy | bmi
    s.elig_rule = E("SENS_ELIG_RULE", "new")             # new | legacy (bmi structure only)
    s.bmi_rebal = E("SENS_BMI_REBAL", "1") == "1"        # keep BMI class shares by age at 2023 values (bmi only)
    s.dm_byclass = E("SENS_DM_BYCLASS", "1") == "1"      # diabetes by BMI class (bmi only)
    s.freeze = E("SENS_TEST_FREEZE", "0") == "1"         # tests only: no new IHD, IS or T2D
    # projected trends (GBD forecasts): cvd, bmi, dm; main analysis all (BMI structure); none = rates held at 2023 values
    tr = set(filter(None, E("SENS_TREND", "all" if ELIG_STRUCT == "bmi" else "none").split(",")))
    s.trend = set() if "none" in tr else {"cvd", "bmi", "dm", "oth"} if "all" in tr else tr     # oth: other-cause mortality
    s.trend_cvd_mode = E("SENS_TREND_CVD", "both")           # both (incidence and deaths after onset) | mortality
    s.ow_file, s.ow_cvd_file = E("SENS_OW_FILE"), E("SENS_OW_CVD_FILE")
    s.elig_file = E("SENS_ELIG_FILE")
    s.elig_col = E("SENS_ELIG_COL", "p30" if "bmi30" in E("SENS_ELIG_FILE", "") else ("p27" if s.bmi_min == "27" else "p25"))
    s.tier3_bmi30 = E("SENS_TIER3_BMI", "30") == "30"    # tier 3 needs BMI >=30 ("27": all BMI >=27 eligible)
    s.select_like = E("SENS_SELECT_LIKE") == "1"
    s.select_file = E("SENS_SELECT_FILE", "data/params/select_elig.csv")
    s.coverage = coverage
    # dietary policy
    s.policy_maintain = float(E("SENS_POLICY_MAINTAIN", 1.0))
    s.diet_post = E("SENS_DIET_POST", "post_na")         # dietary effect on deaths after onset: sodium (main) | post_na_wg | none
    if s.diet_post in ("none", "0", ""): s.diet_post = None
    s.dp_file = E("LT_DP_OVERRIDE", "data/results/diet_policy_arm.csv")
    # diabetes and T2D
    s.t2d_out = E("SENS_T2D", "1") == "1"                # write the T2D file (diabetes is modelled either way)
    s.t2d_params = E("T2D_TRIAL_PARAMS", "data/params/glp1_trial_params.csv")
    s.retain_t2d = float(E("SENS_RETAIN_T2D", 0.72))     # used only with SENS_ONOFF=0
    s.dm_pr = float(E("SENS_DM_PR", 2.5))                # diabetes prevalence ratio, people with vs without IHD or IS
    s.t2d_rr = E("SENS_T2D_RR", "age")                   # relative risks of T2D by BMI: age | region | abdullah
    s.dm_cvd = E("SENS_DM_CVD", "1") == "1"
    s.dm_hr_ihd, s.dm_hr_is = float(E("SENS_DM_HR_IHD", 2.00)), float(E("SENS_DM_HR_IS", 2.27))
    s.dm_tier = E("SENS_DM_TIER", "1") == "1"
    # persistence (Rodriguez et al, JAMA Netw Open 2025)
    s.onoff = E("SENS_ONOFF", "1") == "1"
    s.stop_mult, s.restart_mult = float(E("SENS_STOP_MULT", 1.0)), float(E("SENS_RESTART_MULT", 1.0))
    s.onoff_env = {k: E(k) for k in ("SENS_STOP1_NODM", "SENS_STOP1_DM", "SENS_STOP2_NODM", "SENS_STOP2_DM",
                                     "SENS_RESTART1_NODM", "SENS_RESTART1_DM", "SENS_RESTART2_NODM", "SENS_RESTART2_DM")}
    s.resid_cv, s.resid_t2d = float(E("SENS_RESID_CV", 1 / 3)), float(E("SENS_RESID_T2D", 0.25))
    s.itt_cv, s.itt_t2d = float(E("SENS_ITT_CV", 1.0)), float(E("SENS_ITT_T2D", 1.0))
    s.restart_late = float(E("SENS_RESTART_LATE", 1.0))
    # CKD
    s.ckd = E("SENS_CKD", "1") == "1" and not s.select_like
    s.ckd_curves = E("CKD_CURVES", "v2_conv_fixed/params/dose_response_conv_fixed.csv")
    s.sbp_shift, s.sbp_shift_dm = float(E("SENS_SBP_SHIFT", 3.80)), float(E("SENS_SBP_SHIFT_DM", 2.13))
    s.f_bp = float(E("SENS_F_BP", 0.5))                  # share of time on drug, used only with SENS_ONOFF=0
    # BMI thresholds of the BMI structure: IHD or IS (and diabetes under the legacy rule), and tier 3; the scenarios of
    # the original structure (SENS_BMI_MIN=25, the BMI >=30 scaling file, SENS_TIER3_BMI=27) map onto them
    s.thr_cvd = float(E("SENS_BMI_CVD", "25" if s.bmi_min == "25" else "30" if "bmi30" in (s.elig_file or "") else "27"))
    s.thr_ob = float(E("SENS_TIER3_BMI", "30"))
    if s.elig_struct == "bmi":                           # baseline shares always at BMI >=27; scenarios act via thresholds
        s.bmi_min, s.elig_file, s.elig_col = "27", None, "p27"
    return s
