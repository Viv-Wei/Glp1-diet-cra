import math
import numpy as np
from .settings import ST

DEST = {
    ("N", "I_non"): "I",
    ("N", "S_non"): "S",
    ("I", "S_non"): "B",
    ("S", "I_non"): "B",
}


def ramp(t, yrs):

    k = 4.0 / yrs
    return 1.0 / (1.0 + math.exp(-k * (t - yrs / 1.6)))


def hazards(lam_I, lam_S, mu_I, mu_S, mu_0, q_i, q_s, rr_cond, m=None):

    m = m or {}
    g = lambda k: m.get(k, 1.0)
    mN = m.get("_N", m)
    gN = lambda k: mN.get(k, 1.0)
    li_f, li_n = lam_I * q_i * g("I_fatal"), lam_I * (1 - q_i) * g("I_nonfatal")
    ls_f, ls_n = lam_S * q_s * g("S_fatal"), lam_S * (1 - q_s) * g("S_nonfatal")
    mi, ms = mu_I * m.get("post_I", g("post")), mu_S * m.get("post_S", g("post"))
    return {
        "N": dict(
            I_fatal=lam_I * q_i * gN("I_fatal"),
            I_non=lam_I * (1 - q_i) * gN("I_nonfatal"),
            S_fatal=lam_S * q_s * gN("S_fatal"),
            S_non=lam_S * (1 - q_s) * gN("S_nonfatal"),
            bg=mu_0,
        ),
        "I": dict(S_fatal=ls_f * rr_cond, S_non=ls_n * rr_cond, post_I=mi, bg=mu_0),
        "S": dict(I_fatal=li_f * rr_cond, I_non=li_n * rr_cond, post_S=ms, bg=mu_0),
        "B": dict(post_I=mi, post_S=ms, bg=mu_0),
    }


def step(x, hz):

    new = {st: 0.0 for st in ST}
    flows = {st: 0.0 for st in ST}
    ev = dict(
        fat_I=0.0, fat_S=0.0, post_I=0.0, post_S=0.0, bg=0.0, inc_I=0.0, inc_S=0.0
    )
    py = 0.0
    for st in ST:
        d = hz[st]
        H = sum(d.values())
        xs = x[st]
        f = (1 - math.exp(-H)) / H if H > 0 else 0.0
        py = py + xs * (f if H > 0 else 1.0)
        new[st] = new[st] + xs * math.exp(-H)
        ev["fat_I"] = ev["fat_I"] + xs * d.get("I_fatal", 0) * f
        ev["fat_S"] = ev["fat_S"] + xs * d.get("S_fatal", 0) * f
        ev["post_I"] = ev["post_I"] + xs * d.get("post_I", 0) * f
        ev["post_S"] = ev["post_S"] + xs * d.get("post_S", 0) * f
        ev["bg"] = ev["bg"] + xs * d.get("bg", 0) * f
        ev["inc_I"] = ev["inc_I"] + xs * (d.get("I_fatal", 0) + d.get("I_non", 0)) * f
        ev["inc_S"] = ev["inc_S"] + xs * (d.get("S_fatal", 0) + d.get("S_non", 0)) * f
        for ch in ("I_non", "S_non"):
            if ch not in d:
                continue
            flow = xs * d[ch] * f
            dst = DEST[(st, ch)]
            dd = hz[dst]
            h2 = {k: v for k, v in dd.items() if k in ("post_I", "post_S", "bg")}
            H2 = sum(h2.values())
            pdie = 1 - math.exp(-H2 / 2)
            for k, v in h2.items():
                ev[k] = ev[k] + flow * pdie * v / H2 if H2 > 0 else ev[k]
            py = py + flow * (1 - math.exp(-H2 / 2)) / H2 if H2 > 0 else py + flow * 0.5
            new[dst] = new[dst] + flow * (1 - pdie)
            flows[dst] = flows[dst] + flow * (1 - pdie)
    ev["flows"] = flows
    return new, ev, py


def step_vec(x, hz):

    new = {st: 0.0 for st in ST}
    ev = {k: 0.0 for k in ("fat_I", "fat_S", "post_I", "post_S", "bg")}
    for st in ST:
        d = hz[st]
        xs = x[st]
        H = sum(d.values())
        Hs = np.where(H > 0, H, 1.0)
        E1 = -np.expm1(-Hs)
        xf = xs * np.where(H > 0, E1 / Hs, 0.0)
        new[st] = new[st] + xs * np.exp(-H)
        if "I_fatal" in d:
            ev["fat_I"] = ev["fat_I"] + xf * d["I_fatal"]
        if "S_fatal" in d:
            ev["fat_S"] = ev["fat_S"] + xf * d["S_fatal"]
        for k_ in ("post_I", "post_S", "bg"):
            if k_ in d:
                ev[k_] = ev[k_] + xf * d[k_]
        for ch in ("I_non", "S_non"):
            if ch not in d:
                continue
            flow = xf * d[ch]
            dst = DEST[(st, ch)]
            dd = hz[dst]
            H2 = sum(v for k_, v in dd.items() if k_ in ("post_I", "post_S", "bg"))
            H2s = np.where(H2 > 0, H2, 1.0)
            pdie = -np.expm1(-H2 / 2)
            fd = np.where(H2 > 0, flow * pdie / H2s, 0.0)
            for k_ in ("post_I", "post_S", "bg"):
                if k_ in dd:
                    ev[k_] = ev[k_] + fd * dd[k_]
            new[dst] = new[dst] + flow * (1 - pdie)
    n = len(next(v for v in x.values() if np.ndim(v)))
    return (
        {st: np.broadcast_to(v, (n,)).astype(float) for st, v in new.items()},
        {k: np.broadcast_to(v, (n,)).astype(float) for k, v in ev.items()},
    )


def age_one_year(v):

    a_ = np.zeros_like(v)
    a_[1:] = v[:-1]
    a_[-1] += v[-1]
    return a_
