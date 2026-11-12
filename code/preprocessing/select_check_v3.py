import argparse, math, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lifetable_v3 as L


def rates_for(iso, s_first, q_s, rrc, orc):
    pop = L.load_pop()
    inc_I, prv_I, dth_I = (
        L.gbd("queryB_ihd.csv", "Incidence"),
        L.gbd("queryB_ihd.csv", "Prevalence"),
        L.gbd("queryB_ihd.csv", "Deaths"),
    )
    inc_S, prv_S, dth_S = (
        L.gbd("queryB_istroke.csv", "Incidence"),
        L.gbd("queryB_istroke.csv", "Prevalence"),
        L.gbd("queryB_istroke.csv", "Deaths"),
    )
    allc = L.gbd("queryD_All_causes_deaths.csv", "Deaths")
    gi_ = [
        (
            float(inc_S.get(("GLB", sx, ag), 0)) * float(pop.get(("GLB", sx, ag), 0)),
            float(dth_S.get(("GLB", sx, ag), 0)) * float(pop.get(("GLB", sx, ag), 0)),
        )
        for sx in L.SEXES
        for ag in L.AGE5
    ]
    s_S = q_s * sum(i for i, _ in gi_) / sum(d for _, d in gi_)
    R = {}
    for sx in L.SEXES:
        R[sx] = []
        for ag in L.AGE5:
            k = (iso, sx, ag)
            pi, ps = float(prv_I.get(k, 0)), float(prv_S.get(k, 0))
            pb = L.or_joint(pi, ps, orc)
            sh = [1 - pi - ps + pb, pi - pb, ps - pb, pb]
            tgt = dict(
                inc_I=float(inc_I.get(k, 0)),
                inc_S=float(inc_S.get(k, 0)),
                d_I=float(dth_I.get(k, 0)),
                d_S=float(dth_S.get(k, 0)),
                d_all=float(allc.get(k, 0)),
            )
            qi = (
                min(s_first * tgt["d_I"] / tgt["inc_I"], 1.0)
                if tgt["inc_I"] > 0
                else 0.0
            )
            qs = min(s_S * tgt["d_S"] / tgt["inc_S"], 1.0) if tgt["inc_S"] > 0 else 0.0
            r, _ = L.calibrate(sh, tgt, qi, qs, rrc)
            r.update(q_i=qi, q_s=qs, rr_cond=rrc)
            R[sx].append(r)
    return R


def cohort(both_share):
    w = np.array([0.676, 0.179, 0.082 * both_share])
    w = w / w.sum()
    a = L.AGES.astype(float)
    dens = np.exp(-0.5 * ((a - 61.6) / 8.9) ** 2)
    dens[a < 45] = 0
    dens /= dens.sum()
    X = {}
    for sx, ps in [("Male", 0.723), ("Female", 0.277)]:
        for st, ws in zip(L.ST, [0.0, *w]):
            X[(sx, st)] = 1000.0 * ps * ws * dens
    return X


def simulate(R, X0, mult, years=4):
    X = {k: v.copy() for k, v in X0.items()}
    cum = dict(death=[0.0], mi_in_S=[0.0], st_in_I=[0.0], atrisk_S=[], atrisk_I=[])
    d = n_mi = n_st = 0.0
    for t in range(1, years + 1):
        new = {k: np.zeros_like(v) for k, v in X.items()}
        cum["atrisk_S"].append(sum(X[(sx, "S")].sum() for sx in L.SEXES))
        cum["atrisk_I"].append(sum(X[(sx, "I")].sum() for sx in L.SEXES))
        for sx in L.SEXES:
            for gi in range(len(L.AGE5)):
                sel = L.GIDX == gi
                r = R[sx][gi]
                hz = L.hazards(
                    r["lam_I"],
                    r["lam_S"],
                    r["mu_I"],
                    r["mu_S"],
                    r["mu_0"],
                    r["q_i"],
                    r["q_s"],
                    r["rr_cond"],
                    mult,
                )
                xs = {st: X[(sx, st)][sel] for st in L.ST}
                nx, ev, _ = L.step(xs, hz)
                for st in L.ST:
                    new[(sx, st)][sel] += nx[st]
                d += float(
                    np.sum(ev["fat_I"] + ev["post_I"] + ev["fat_S"] + ev["post_S"])
                )
                for state, ch, key in [("S", "I_non", "mi"), ("I", "S_non", "st")]:
                    h = hz[state]
                    H = sum(h.values())
                    f = (1 - math.exp(-H)) / H if H > 0 else 0.0
                    val = float(np.sum(xs[state] * h[ch] * f))
                    if key == "mi":
                        n_mi += val
                    else:
                        n_st += val
        for k in new:
            v = new[k]
            a_ = np.zeros_like(v)
            a_[1:] = v[:-1]
            a_[-1] += v[-1]
            new[k] = a_
        X = new
        cum["death"].append(d)
        cum["mi_in_S"].append(n_mi)
        cum["st_in_I"].append(n_st)
    at40 = lambda s: s[3] + (s[4] - s[3]) / 3.0
    return dict(
        death=at40(cum["death"]),
        mi=at40(cum["mi_in_S"]),
        st=at40(cum["st_in_I"]),
        death_yr=cum["death"][1:],
    )


def main(a):
    E = os.environ.get
    s_first = float(E("SENS_IHD_FIRST", 0.40))
    q_s = float(E("SENS_IS_CF28", 0.15))
    HR_MI, HR_ST, HR_CVD = (
        float(E("SENS_HR_MI", 0.72)),
        float(E("SENS_HR_STROKE", 0.93)),
        float(E("SENS_HR_CVD", 0.85)),
    )
    mode = E("SENS_FATAL_MODE", "same")
    HRF_I = {"same": HR_MI, "cvd": HR_CVD, "none": 1.0}[mode]
    HRF_S = {"same": HR_ST, "cvd": HR_CVD, "none": 1.0}[mode]
    trt = dict(
        I_fatal=HRF_I, I_nonfatal=HR_MI, S_fatal=HRF_S, S_nonfatal=HR_ST, post=HR_CVD
    )
    rows = []
    for iso in a.iso:
        R = rates_for(
            iso,
            s_first,
            q_s,
            float(E("SENS_RR_COND", 1.0)),
            float(E("SENS_OR_COMORB", 1.0)),
        )
        X0 = cohort(a.both_share)
        p, t = simulate(R, X0, None), simulate(R, X0, trt)
        rows.append(
            dict(
                iso3=iso,
                placebo_death_pct=p["death"] / 10,
                treated_death_pct=t["death"] / 10,
                death_ratio=t["death"] / p["death"],
                averted_per_1000_per_year=(p["death"] - t["death"]) / (40 / 12),
                placebo_death_rate_per_100py=p["death_yr"][0] / 10,
                mi_ratio_in_stroke_only=t["mi"] / p["mi"] if p["mi"] else np.nan,
                stroke_ratio_in_mi_only=t["st"] / p["st"] if p["st"] else np.nan,
            )
        )
    df = pd.DataFrame(rows).set_index("iso3")
    df.to_csv(a.out)
    print(
        f"SELECT-like cohort, 40 months (s_IHD = {s_first}, fatal mode = {mode}, share with both MI and stroke x {a.both_share})"
    )
    print(df.round(3).to_string())
    print(
        "\nReference (SELECT): placebo cardiovascular death 2.98% (definite cardiovascular 1.95%); death HR 0.85 (0.71-1.01);"
        " about 1.4 deaths averted per 1000 per year; non-fatal MI 0.72, non-fatal stroke 0.93."
        "\nHennessy et al model (US): cardiovascular death RR 0.85, MI 0.72, stroke 0.94."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--iso", nargs="+", default=["USA", "GLB", "DEU", "GBR", "CAN", "JPN"]
    )
    p.add_argument("--both-share", type=float, default=1.0)
    p.add_argument("--out", default="data/results/v3/select_check.csv")
    main(p.parse_args())
