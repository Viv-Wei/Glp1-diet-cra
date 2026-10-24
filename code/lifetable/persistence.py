import numpy as np

PUBLISHED = dict(nd=(0.648, 0.844, 0.363, 0.464), d=(0.465, 0.641, 0.473, 0.573))


def onoff_params(s):

    if not s.onoff:
        return None
    oo = {}
    for k_, env_, nd, d, mul in [
        ("s1", "SENS_STOP1", 0.648, 0.465, s.stop_mult),
        ("s2", "SENS_STOP2", 0.557, 0.329, s.stop_mult),
        ("r1", "SENS_RESTART1", 0.363, 0.473, s.restart_mult),
        ("r2", "SENS_RESTART2", 0.159, 0.190, s.restart_mult),
    ]:
        v = lambda name, dflt: float(
            s.onoff_env[name] if s.onoff_env.get(name) is not None else dflt
        )
        oo[k_ + "_nd"] = min(v(env_ + "_NODM", nd) * mul, 1.0)
        oo[k_ + "_d"] = min(v(env_ + "_DM", d) * mul, 1.0)
    oo.update(
        res_cv=s.resid_cv,
        res_t2d=s.resid_t2d,
        itt_cv=s.itt_cv,
        itt_t2d=s.itt_t2d,
        w_dm=0.0,
        late=s.restart_late,
    )
    return oo


def reconstruction_check(oo):

    rec = []
    for g_ in ("nd", "d"):
        s1, s2, r1, r2 = oo["s1_" + g_], oo["s2_" + g_], oo["r1_" + g_], oo["r2_" + g_]
        mdl = (s1, 1 - (1 - s1) * (1 - s2), r1, 1 - (1 - r1) * (1 - r2))
        for nm_, pv, mv in zip(
            ("discontinued_1y", "discontinued_2y", "reinitiated_1y", "reinitiated_2y"),
            PUBLISHED[g_],
            mdl,
        ):
            rec.append(
                dict(
                    group="without T2D" if g_ == "nd" else "with T2D",
                    quantity=nm_,
                    published=pv,
                    model=mv,
                    diff=mv - pv,
                )
            )
    return rec


def retained(q, onoff, g_):

    o = {k_: onoff[k_ + "_" + g_] for k_ in ("s1", "s2", "r1", "r2")}
    o["r3"] = min(o["r2"] * onoff.get("late", 1.0), 1.0)
    o.update({k_: onoff[k_] for k_ in ("res_cv", "res_t2d", "itt_cv", "itt_t2d")})
    stop_ = q[0] * o["s1"] + q[1] * o["s2"]
    rest_ = q[2] * o["r1"] + q[3] * o["r2"] + q[4] * o["r3"]
    on = q[0] + q[1] - 0.5 * stop_ + 0.5 * rest_
    off1 = q[2] * (1 - 0.5 * o["r1"]) + 0.5 * stop_
    return dict(
        cv=(on + off1 * o["res_cv"]) / o["itt_cv"],
        t2d=(on + off1 * o["res_t2d"]) / o["itt_t2d"],
        on=on,
        o=o,
    )


def next_states(q, o):

    return np.array(
        [
            q[2] * o["r1"] + q[3] * o["r2"] + q[4] * o["r3"],
            q[0] * (1 - o["s1"]) + q[1] * (1 - o["s2"]),
            q[0] * o["s1"] + q[1] * o["s2"],
            q[2] * (1 - o["r1"]),
            q[3] * (1 - o["r2"]) + q[4] * (1 - o["r3"]),
        ]
    )


def carry_converters(q_d, q_nd, enrolled_dm, converters, enrolled_dm_start):

    cs_ = (
        enrolled_dm * converters / (enrolled_dm_start + converters)
        if enrolled_dm_start + converters > 0
        else 0.0
    )
    return (
        (q_d * (enrolled_dm - cs_) + q_nd * cs_) / enrolled_dm
        if enrolled_dm > 0
        else q_d
    )


def mix_new_enrolment(q, before, after):

    return (
        (q * before + np.array([1.0, 0, 0, 0, 0]) * max(after - before, 0.0))
        / max(after, before)
        if after > 0
        else q
    )
