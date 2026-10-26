import os
from types import SimpleNamespace
import numpy as np

E = os.environ.get
os.chdir(E("CRA_ROOT", os.getcwd()))

AGE5 = [f"{a}-{a+4} years" for a in range(25, 95, 5)] + ["95+ years"]
G0 = list(range(25, 95, 5)) + [95]
AGES = np.arange(25, 96)
GIDX = np.minimum((AGES - 25) // 5, len(AGE5) - 1)
SEXES = ["Male", "Female"]
ST = ["N", "I", "S", "B"]
ARMS = ["baseline", "glp1", "diet", "combo"]
KS = ["T", "E", "O"]


def read_settings(coverage):

    s = SimpleNamespace()
    s.ihd_first = float(E("SENS_IHD_FIRST", 0.40))
    s.is_cf28 = float(E("SENS_IS_CF28", 0.15))
    s.or_comorb = float(E("SENS_OR_COMORB", 1.0))
    s.rr_cond = float(E("SENS_RR_COND", 1.0))
    s.k_dur = float(E("SENS_K_DUR", 1.0))
    s.open_pop = E("SENS_OPEN_POP") == "1"
    s.fatal_mode = E("SENS_FATAL_MODE", "same")
    s.post_none = E("SENS_POST_NONE") == "1"
    s.hr_env = {
        k: E(k)
        for k in (
            "SENS_HR_MI",
            "SENS_HR_STROKE",
            "SENS_HR_CVD",
            "SENS_HR_MI_DM",
            "SENS_HR_STROKE_DM",
            "SENS_HR_CVD_DM",
        )
    }
    s.primary_scale = float(E("SENS_PRIMARY_SCALE", 1.0))
    s.tier3_scale = float(E("SENS_TIER3_SCALE", 1.0))
    s.tier2_mi, s.tier2_stroke = E("SENS_TIER2_MI"), E("SENS_TIER2_STROKE")
    s.retain = float(E("SENS_RETAIN", 0.76))
    s.cap_ref = E("SENS_CAP_REF")
    s.treat = E("SENS_TREAT", "maintain")
    s.alloc = E("SENS_ALLOC", "risk")
    s.bmi_min = E("SENS_BMI_MIN", "27")
    s.ow_file, s.ow_cvd_file = E("SENS_OW_FILE"), E("SENS_OW_CVD_FILE")
    s.elig_file = E("SENS_ELIG_FILE")
    s.elig_col = E(
        "SENS_ELIG_COL",
        (
            "p30"
            if "bmi30" in E("SENS_ELIG_FILE", "")
            else ("p27" if s.bmi_min == "27" else "p25")
        ),
    )
    s.tier3_bmi30 = E("SENS_TIER3_BMI", "30") == "30"
    s.select_like = E("SENS_SELECT_LIKE") == "1"
    s.select_file = E("SENS_SELECT_FILE", "data/params/select_elig.csv")
    s.coverage = coverage
    s.policy_maintain = float(E("SENS_POLICY_MAINTAIN", 1.0))
    s.diet_post = E("SENS_DIET_POST")  # dietary scenario whose effect also applies to deaths after onset
    s.dp_file = E("LT_DP_OVERRIDE", "data/results/v3/diet_policy_arm_v2_tailored.csv")
    s.t2d_out = E("SENS_T2D", "1") == "1"
    s.t2d_params = E("T2D_TRIAL_PARAMS", "data/params/glp1_trial_params.csv")
    s.retain_t2d = float(E("SENS_RETAIN_T2D", 0.72))
    s.dm_pr = float(E("SENS_DM_PR", 2.5))
    s.t2d_rr = E("SENS_T2D_RR", "age")
    s.dm_cvd = E("SENS_DM_CVD", "1") == "1"
    s.dm_hr_ihd, s.dm_hr_is = float(E("SENS_DM_HR_IHD", 2.00)), float(
        E("SENS_DM_HR_IS", 2.27)
    )
    s.dm_tier = E("SENS_DM_TIER", "1") == "1"
    s.onoff = E("SENS_ONOFF", "1") == "1"
    s.stop_mult, s.restart_mult = float(E("SENS_STOP_MULT", 1.0)), float(
        E("SENS_RESTART_MULT", 1.0)
    )
    s.onoff_env = {
        k: E(k)
        for k in (
            "SENS_STOP1_NODM",
            "SENS_STOP1_DM",
            "SENS_STOP2_NODM",
            "SENS_STOP2_DM",
            "SENS_RESTART1_NODM",
            "SENS_RESTART1_DM",
            "SENS_RESTART2_NODM",
            "SENS_RESTART2_DM",
        )
    }
    s.resid_cv, s.resid_t2d = float(E("SENS_RESID_CV", 1 / 3)), float(
        E("SENS_RESID_T2D", 0.25)
    )
    s.itt_cv, s.itt_t2d = float(E("SENS_ITT_CV", 1.0)), float(E("SENS_ITT_T2D", 1.0))
    s.restart_late = float(E("SENS_RESTART_LATE", 1.0))
    s.ckd = E("SENS_CKD", "1") == "1" and not s.select_like
    s.ckd_curves = E("CKD_CURVES", "data/params/dose_response_conv_fixed.csv")
    s.sbp_shift, s.sbp_shift_dm = float(E("SENS_SBP_SHIFT", 3.80)), float(
        E("SENS_SBP_SHIFT_DM", 2.13)
    )
    s.f_bp = float(E("SENS_F_BP", 0.5))
    return s
