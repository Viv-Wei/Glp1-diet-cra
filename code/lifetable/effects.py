from .transitions import ramp


def hr_sets(s):

    def hr_set(sfx, mi, st, cvd):
        v = lambda name, d: float(
            s.hr_env[name + sfx] if s.hr_env.get(name + sfx) is not None else d
        )
        h = dict(
            mi=v("SENS_HR_MI", mi),
            st=v("SENS_HR_STROKE", st),
            cvd=v("SENS_HR_CVD", cvd),
        )
        h["fI"] = {"same": h["mi"], "cvd": h["cvd"], "none": 1.0}[s.fatal_mode]
        h["fS"] = {"same": h["st"], "cvd": h["cvd"], "none": 1.0}[s.fatal_mode]
        if s.post_none:
            h["cvd"] = 1.0
        return h

    return {False: hr_set("", 0.77, 0.93, 0.87), True: hr_set("_DM", 0.87, 0.93, 0.86)}


def make_multipliers(s, HRS, dI, dS, ramp_diet, ramp_glp1, dI_post=0.0, dS_post=0.0):
    # with SENS_DIET_POST, dietary policy also multiplies deaths after IHD and IS onset by (1 - dI_post x phase-in x
    # maintenance) and (1 - dS_post x ...), combined multiplicatively with the treatment effect on those deaths

    def mult(arm, t, k, Rt=None, dmf=False):
        m = {}
        if arm in ("diet", "combo"):
            fI = 1 - dI * s.policy_maintain * ramp(t, ramp_diet)
            fS = 1 - dS * s.policy_maintain * ramp(t, ramp_diet)
            m.update(I_fatal=fI, I_nonfatal=fI, S_fatal=fS, S_nonfatal=fS)
            if dI_post or dS_post:  # deaths after onset (SENS_DIET_POST)
                m.update(
                    post_I=1 - dI_post * s.policy_maintain * ramp(t, ramp_diet),
                    post_S=1 - dS_post * s.policy_maintain * ramp(t, ramp_diet),
                )
        mN = {k_: v_ for k_, v_ in m.items() if k_ not in ("post_I", "post_S")}  # people without IHD or IS
        if arm in ("glp1", "combo") and k == "T":
            rt = (Rt["cv"] if Rt else s.retain) * ramp(t, ramp_glp1)
            h = HRS[bool(dmf)]
            for ch, hr in [
                ("I_fatal", h["fI"]),
                ("I_nonfatal", h["mi"]),
                ("S_fatal", h["fS"]),
                ("S_nonfatal", h["st"]),
                ("post", h["cvd"]),
            ]:
                m[ch] = m.get(ch, 1.0) * (1 - rt * (1 - hr))
            sN = s.primary_scale * (1.0 if dmf else s.tier3_scale)
            hN = dict(
                I_fatal=h["fI"], I_nonfatal=h["mi"], S_fatal=h["fS"], S_nonfatal=h["st"]
            )
            if dmf and s.tier2_mi:
                hN.update(
                    I_nonfatal=float(s.tier2_mi),
                    I_fatal=(
                        float(s.tier2_mi) if s.fatal_mode == "same" else hN["I_fatal"]
                    ),
                )
            if dmf and s.tier2_stroke:
                hN.update(
                    S_nonfatal=float(s.tier2_stroke),
                    S_fatal=(
                        float(s.tier2_stroke)
                        if s.fatal_mode == "same"
                        else hN["S_fatal"]
                    ),
                )
            for ch, hr in hN.items():
                mN[ch] = mN.get(ch, 1.0) * (1 - rt * sN * (1 - hr))
        if "post_I" in m and "post" in m:  # both arms act on deaths after onset
            m["post_I"] *= m["post"]
            m["post_S"] *= m["post"]
        m["_N"] = mN
        return m

    return mult


def add_t2d_multiplier(mult, s, dT2D, hr_t2d, ramp_diet, ramp_glp1):

    def mult2(arm, t, k, Rt=None, dmf=False):
        m = dict(mult(arm, t, k, Rt, dmf))
        f_ = 1.0
        if arm in ("diet", "combo"):
            f_ *= 1 - dT2D * s.policy_maintain * ramp(t, ramp_diet)
        if arm in ("glp1", "combo") and k == "T":
            f_ *= 1 - (Rt["t2d"] if Rt else s.retain_t2d) * ramp(t, ramp_glp1) * (
                1 - hr_t2d
            )
        m["T2D"] = f_
        return m

    return mult2
