import numpy as np
from .settings import AGES, AGE5, G0, GIDX, SEXES, ST

BC = ("0", "1", "2", "3")                                               # BMI <25, 25 to <27, 27 to <30, >=30
LOW = {"0": 0.0, "1": 25.0, "2": 27.0, "3": 30.0}                       # lower bound of each class
KT = tuple("T" + b for b in BC); KU = tuple("U" + b for b in BC)
CVD = ("I", "S", "B")
RR_POOLED = {"IHD": (1.32, 1.81), "IS": (1.22, 1.64)}                 # Bogers 2007; Strazzullo 2010
RR_EAST_ASIA = {"IHD": (1.29, 2.16), "IS": (1.26, 2.03)}              # Chen 2013, east Asians
RR_SOUTH_ASIA = {"IHD": (1.17, 1.33), "IS": (1.00, 1.00)}             # Chen 2013, south Asians


def at_least(thr):
    """Classes at or above a BMI threshold (25, 27 or 30)."""
    return [b for b in BC if LOW[b] >= thr]


def rr_cvd(iso):
    return RR_EAST_ASIA if iso in ("CHN", "JPN") else RR_SOUTH_ASIA if iso in ("IND", "BGD") else RR_POOLED


def _shares(iso, eg_st, st, p25, p27, p30, legacy_dm):
    """BMI class shares of one state by age: the share >=27 (eg_st), split at 30 and, below 27, at 25 with the population
    BMI distribution (no IHD or IS, or legacy_dm) or with Bayes' rule and the relative risks of IHD and IS (rr_cvd)."""
    e27 = np.clip(eg_st, 0, 1)
    h_pop = np.minimum(p30 / np.maximum(p27, 1e-9), 1.0); j_pop = np.clip((p25 - p27) / np.maximum(1 - p27, 1e-9), 0, 1)
    if st == "N" or legacy_dm:
        h, j = h_pop, j_pop
    else:
        rc = rr_cvd(iso)
        def hj(o):
            ro, rb = rc[o]
            return (np.minimum(p30 * rb / np.maximum((p27 - p30) * ro + p30 * rb, 1e-12), 1.0),
                    np.clip((p25 - p27) * ro / np.maximum((1 - p25) + (p25 - p27) * ro, 1e-12), 0, 1))
        (hI, jI), (hS, jS) = hj("IHD"), hj("IS")
        h, j = dict(I=(hI, jI), S=(hS, jS), B=((hI + hS) / 2, (jI + jS) / 2))[st]
    return {"0": (1 - e27) * (1 - j), "1": (1 - e27) * j, "2": e27 * (1 - h), "3": e27 * h}


def _bmi_age(iso, sx, bmit, ow):
    """p25, p27, p30 by single-year age (2023 BMI distribution of data/params/select_elig.csv)."""
    n = len(AGES); p25, p27, p30 = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        a5 = G0[GIDX[i]]; b = bmit.loc[(iso, sx, a5)] if (iso, sx, a5) in bmit.index else None
        p25[i] = float(b.p25) if b is not None else ow; p27[i] = float(b.p27) if b is not None else ow
        p30[i] = float(b.p30) if b is not None else ow / 2
    return p25, p27, p30


def initial_states_bmi(iso, base, eg, ow, t2_inc, t2_prv, bmit, s, rr_t2d, legacy_dm=False):
    """Initial counts by BMI class, the T2D overlay and the BMI class targets (2023 shares by age).
    Inputs: base[sex][state] (counts by single-year age); eg[sex][state] (share with BMI >=27, as in the original model);
    ow (national share with BMI >=27, a fallback); GBD T2D incidence and prevalence; bmit (BMI distribution by country,
    sex and age); s (settings); rr_t2d (diabetes.rr_t2d); legacy_dm, diabetes shared within BMI <27 and within >=27
    as in the original model (for checks).
    Output: init[sex][(state, k)], t2d (lam, e[sex][class], free0[(sex, state, k)]), targets[(sex, state)][class]."""
    init = {sx: {} for sx in SEXES}; t2d = dict(lam={}, e={}, free0={}); targets = {}
    for sx in SEXES:
        n = len(AGES); p25, p27, p30 = _bmi_age(iso, sx, bmit, ow)
        r1, r2, pdm, lam = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
        for i in range(n):
            a5 = G0[GIDX[i]]; key = (iso, sx, AGE5[GIDX[i]])
            r1[i], r2[i] = rr_t2d(iso, a5, s.t2d_rr)
            pv = float(t2_prv.get(key, 0.0)); ic = float(t2_inc.get(key, 0.0)); pdm[i] = pv; lam[i] = ic / max(1 - pv, 1e-9)
        den = (1 - p25) + (p25 - p30) * r1 + p30 * r2                    # mean relative risk of T2D in the population
        RR = {"0": np.ones(n), "1": r1, "2": r1, "3": r2}                  # Yu et al: overweight r1, obesity r2
        t2d["lam"][sx] = lam; t2d["e"][sx] = {b: RR[b] / den for b in BC}
        S_ow = ((p27 - p30) * r1 + p30 * r2) / den; e_ow = S_ow / np.maximum(p27, 1e-9)   # as diabetes.t2d_overlay
        tot = sum(base[sx][st] for st in ST); tot = np.where(tot > 0, tot, 1.0)
        p_cvd = pdm * s.dm_pr / (pdm * s.dm_pr + 1 - pdm)
        sh_cvd = (base[sx]["I"] + base[sx]["S"] + base[sx]["B"]) / tot; sh_N = base[sx]["N"] / tot
        p_N = np.clip((pdm - p_cvd * sh_cvd) / np.where(sh_N > 0, sh_N, 1.0), 0, 1)
        for st in ST:
            sh = _shares(iso, eg[sx][st], st, p25, p27, p30, legacy_dm); targets[(sx, st)] = sh
            p_st = p_N if st == "N" else p_cvd; e27 = np.clip(eg[sx][st], 0, 1)
            if legacy_dm:                                                   # as the original model
                q_ow = np.clip(p_st * e_ow, 0, 0.999); q_no = np.clip((p_st - e27 * q_ow) / np.maximum(1 - e27, 1e-9), 0, 0.999)
                q = {"0": q_no, "1": q_no, "2": q_ow, "3": q_ow}
            else:                                                           # by class, Yu et al relative risks
                d_st = sum(sh[b] * RR[b] for b in BC)
                q = {b: np.clip(p_st * RR[b] / np.maximum(d_st, 1e-12), 0, 0.999) for b in BC}
            for b in BC:
                init[sx][(st, "U" + b)] = base[sx][st] * sh[b]; init[sx][(st, "T" + b)] = base[sx][st] * 0.0
                t2d["free0"][(sx, st, "U" + b)] = 1 - q[b]; t2d["free0"][(sx, st, "T" + b)] = 1 - q[b]
    return init, t2d, targets


def year_targets(iso, eg, bmit, bt, ow, legacy_dm=False, years=30):
    """BMI class targets by year for the trend scenario. Each state's share with BMI >=27 moves with the population share
    on the logit scale; the splits at 25 and 30 follow the year's BMI distribution as at baseline (_shares).
    Inputs: bt, data/params/trends/bmi_target.csv for this country, indexed by sex, a0, t (p25, p27, p30).
    Output: a list over t of {(sex, state): {class: share by age}}."""
    lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9))); ex = lambda x: 1 / (1 + np.exp(-x))
    a5 = np.array([G0[i] for i in GIDX]); out = []
    base_p = {sx: _bmi_age(iso, sx, bmit, ow) for sx in SEXES}
    b_ = bt.reset_index(); arr = {}                                    # arrays (year x single-year age), built once
    for sx in SEXES:
        for c in ("p25", "p27", "p30"):
            pv = b_[b_.sex == sx].pivot(index="t", columns="a0", values=c)
            arr[(sx, c)] = pv.values[:, [pv.columns.get_loc(int(a)) for a in a5]]
    for t in range(1, years + 1):
        tg = {}
        for sx in SEXES:
            p25, p27, p30 = (arr[(sx, c)][t - 1] for c in ("p25", "p27", "p30"))
            shift = lg(p27) - lg(base_p[sx][1])
            for st in ST:
                tg[(sx, st)] = _shares(iso, ex(lg(np.clip(eg[sx][st], 0, 1)) + shift), st, p25, p27, p30, legacy_dm)
        out.append(tg)
    return out


def entrants_bmi(ent_old, targets):
    """Open population: the composition of entrants by BMI class (the age-25 targets of each state)."""
    if ent_old is None: return None
    ent = {}
    for sx, e_ in ent_old.items():
        share = {}
        for st in ST:
            sst = e_["share"][(st, "E")] + e_["share"][(st, "O")]
            for b in BC:
                share[(st, "U" + b)] = sst * float(targets[(sx, st)][b][0]); share[(st, "T" + b)] = 0.0
        ent[sx] = dict(n=e_["n"], share=share)
    return ent


def pools(X, F, rule, thr_cvd=27.0, thr_ob=30.0):
    """Eligible people not enrolled, by tier: lists of (key, part) with part all | dm | free."""
    up, ob = at_least(thr_cvd), at_least(thr_ob)
    t1, t2, t3 = [], [], []
    for sx in SEXES:
        for st in CVD:
            t1 += [((sx, st, "U" + b), "all") for b in up]
            if rule == "new": t1 += [((sx, st, "U" + b), "dm") for b in BC if b not in up]
        t2 += [((sx, "N", "U" + b), "dm") for b in (BC if rule == "new" else up)]
        t3 += [((sx, "N", "U" + b), "free") for b in ob]
    return t1, t2, t3


def _size(X, F, key, part):
    if part == "all": return float(X[key].sum())
    if part == "dm": return float(np.maximum(X[key] - F[key], 0.0).sum())
    return float(F[key].sum())


def _move(X, F, G, items, frac):
    """Move a fraction of each pool part from U to the T compartment of the same BMI class."""
    for key, part in items:
        kT = (key[0], key[1], "T" + key[2][1])
        if part == "all": mv = X[key] * frac; mvf = F[key] * frac
        elif part == "dm": mv = np.maximum(X[key] - F[key], 0.0) * frac; mvf = None
        else: mv = F[key] * frac; mvf = mv
        if G is not None:
            if part == "all": mvg = G[key] * frac
            elif part == "dm": mvg = np.maximum(G[key] - F[key], 0.0) * frac
            else: mvg = np.minimum(F[key], G[key]) * frac
            G[key] = G[key] - mvg; G[kT] = G[kT] + mvg
        X[key] = X[key] - mv; X[kT] = X[kT] + mv
        if mvf is not None: F[key] = F[key] - mvf; F[kT] = F[kT] + mvf


def allocate_bmi(X, F, G, Tm, first, C, mode, alloc, tier2, rule, thr_cvd=27.0, thr_ob=30.0):
    """Fill places up to C x (enrolled + eligible not enrolled), in tier order (risk) or uniformly.
    Output: (eligible, target, new enrolment)."""
    t1, t2, t3 = pools(X, F, rule, thr_cvd, thr_ob)
    P = [sum(_size(X, F, k, p) for k, p in t) for t in (t1, t2, t3)]
    Et = Tm + sum(P); K = C * Et
    if mode == "baseline" and not first: return Et, K, 0.0
    if mode == "dynamic" and not first:
        _move(X, F, G, t1, 1.0); return Et, K, P[0]
    A = max(K - Tm, 0.0); done = 0.0
    if alloc == "uniform":
        tot = sum(P); fr = min(A / tot, 1.0) if tot > 0 else 0.0
        _move(X, F, G, t1 + t2 + t3, fr); return Et, K, min(A, tot)
    order = [(t1, P[0]), (t2, P[1]), (t3, P[2])] if tier2 else [(t1, P[0]), (t2 + t3, P[1] + P[2])]
    for items, size in order:
        if A <= 0 or size <= 0: continue
        m = min(A, size); _move(X, F, G, items, m / size); A -= m; done += m
    return Et, K, done


_TG_CACHE = {}


def rebalance(X, F, G, targets):
    """Keep the BMI class shares at each age at their targets within each sex and state, enrolled and not enrolled
    together. The shifts between classes are computed from the combined counts and applied in the same proportions to
    the enrolled and to the not enrolled, so neither group changes in size and selective enrolment is not undone.
    People moving between classes keep their diabetes status (F, G move in proportion)."""
    tid = id(targets)
    if tid not in _TG_CACHE:                                          # stacked targets, built once per target set
        if len(_TG_CACHE) > 64: _TG_CACHE.clear()
        _TG_CACHE[tid] = (targets, {key: np.stack([tg[b] for b in BC]) for key, tg in targets.items()})
    TG = _TG_CACHE[tid][1]
    for (sx, st), tg in TG.items():
        kT = [(sx, st, "T" + b) for b in BC]; kU = [(sx, st, "U" + b) for b in BC]
        XT = np.stack([X[k] for k in kT]); XU = np.stack([X[k] for k in kU]); Xs = XT + XU; tot = Xs.sum(axis=0)
        if not tot.any(): continue
        want = tot * tg
        don = np.divide(np.maximum(Xs - want, 0.0), Xs, out=np.zeros_like(Xs), where=Xs > 0)     # share leaving each class
        deficit = np.maximum(want - Xs, 0.0); dsum = deficit.sum(axis=0)
        rsh = np.divide(deficit, dsum, out=np.zeros_like(deficit), where=dsum > 0)                # share of arrivals by class
        keep = 1 - don
        for keys in (kT, kU):                                            # enrolled and not enrolled, each conserved
            Xg = np.stack([X[k] for k in keys])
            if not Xg.any(): continue
            Fg = np.stack([F[k] for k in keys])
            Xn = Xg * keep + (Xg * don).sum(axis=0) * rsh; Fn = Fg * keep + (Fg * don).sum(axis=0) * rsh
            if G is not None:
                Gg = np.stack([G[k] for k in keys]); Gn = Gg * keep + (Gg * don).sum(axis=0) * rsh
            for i_, k in enumerate(keys):
                X[k] = Xn[i_]; F[k] = Fn[i_]
                if G is not None: G[k] = Gn[i_]
