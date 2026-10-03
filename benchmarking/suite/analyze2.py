#!/usr/bin/env python3
"""Metrics for the second-design grid (benchmarking_results/suite2).

For each model and condition, every description is compared with SSA ensemble A:
  w1      Wasserstein-1 distance between pooled samples (one per regulator
          lifetime), divided by the SSA standard deviation
  mean, cv (not for the ODE), amplitude errors
  period (oscillators): mean of per-trajectory ACF periods, and the fraction of
          trajectories with no detectable period
  tac (stationary): lag at which the ACF falls below 1/e
  switching (toggle): basin changes, with 95% Poisson interval
Each metric carries a 95% bootstrap interval over trajectories (200 resamples).
SSA ensemble B gives the null value. Architecture comparisons (SSA of H against
SSA of S, ODE of H against ODE of S) are written separately. For stationary
models the LNA prediction of the promoter share of the variance is added.
"""
import os, glob, re
import numpy as np, pandas as pd
from scipy import stats
from systems2 import MODELS, BASE_OMEGA, STATIONARY, OSCILLATOR
from engine import eta
from observables import acf_period, switching

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
DIR = os.path.join(RES, "suite2")
NB = 200
rng = np.random.default_rng(0)


def load(model, method, s, om, ens):
    p = os.path.join(DIR, "%s_%s_s%g_om%g_%s.npz" % (model, method, s, om, ens))
    if not os.path.exists(p):
        return None, None
    d = np.load(p)
    return d["traj"].astype(float), d["wall"]


def acf_time(y, dt):
    y = y - y.mean()
    if np.allclose(y, 0):
        return np.nan
    n = len(y); f = np.fft.rfft(y, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]; ac /= ac[0]
    i = np.where(ac < np.exp(-1))[0]
    return i[0] * dt if len(i) else np.nan


def w1n(a, b, sd):
    return stats.wasserstein_distance(a, b) / sd if sd > 0 else np.nan


def per_traj(T, M, burn, system):
    Y = T[:, burn:, 0]
    d = {"level": Y.mean(axis=1), "var": Y.var(axis=1), "amp": Y.std(axis=1)}
    if system in OSCILLATOR:
        d["period"] = np.array([acf_period(y, M.record_dt) for y in Y])
    if system in STATIONARY:
        d["tac"] = np.array([acf_time(y, M.record_dt) for y in Y])
    if system == "Toggle":
        r, f = switching(T[:, :, 0], T[:, :, 1], M.record_dt, burn)
        d["nswitch"] = r * (T.shape[1] - burn) * M.record_dt
    return d


def summary(TA, TM, M, burn, system, method, idxA=None, idxM=None):
    """Point metrics for (possibly resampled) trajectory index sets."""
    iA = np.arange(TA.shape[0]) if idxA is None else idxA
    iM = np.arange(TM.shape[0]) if idxM is None else idxM
    thin = max(1, int(round(M.tau_prot / M.record_dt)))
    YA, YM = TA[iA, burn:, 0], TM[iM, burn:, 0]
    out = {"mean_err": YM.mean() / YA.mean() - 1}
    if method != "ode":
        out["cv_err"] = (YM.std() / YM.mean()) / (YA.std() / YA.mean()) - 1
        out["var_ratio"] = YM.var() / YA.var()
        out["w1"] = w1n(YM[:, ::thin].ravel(), YA[:, ::thin].ravel(), YA.std())
    return out


def compare(TA, TM, M, burn, system, method):
    point = summary(TA, TM, M, burn, system, method)
    boots = [summary(TA, TM, M, burn, system, method,
                     rng.integers(0, TA.shape[0], TA.shape[0]),
                     rng.integers(0, TM.shape[0], TM.shape[0])) for _ in range(NB)]
    out = {}
    for k, v in point.items():
        b = np.array([x[k] for x in boots])
        out[k] = v; out[k + "_lo"], out[k + "_hi"] = np.percentile(b, [2.5, 97.5])
    pa, pm = per_traj(TA, M, burn, system), per_traj(TM, M, burn, system)
    if system in OSCILLATOR and method != "ode":
        a, b = pa["period"], pm["period"]
        out["period_A"] = np.nanmean(a); out["period_M"] = np.nanmean(b)
        out["nonosc_A"] = np.mean(~np.isfinite(a)); out["nonosc_M"] = np.mean(~np.isfinite(b))
        aa, bb = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(aa) > 2 and len(bb) > 2:
            bt = [np.mean(rng.choice(bb, len(bb))) / np.mean(rng.choice(aa, len(aa))) - 1 for _ in range(NB)]
            out["period_err"] = bb.mean() / aa.mean() - 1
            out["period_err_lo"], out["period_err_hi"] = np.percentile(bt, [2.5, 97.5])
    if system in OSCILLATOR and method == "ode":
        from observables import acf_period as ap
        out["period_A"] = np.nanmean(pa["period"])
        out["period_M"] = ap(TM[0, burn:, 0], M.record_dt)
        out["period_err"] = out["period_M"] / out["period_A"] - 1
    if system in STATIONARY and method != "ode":
        a, b = pa["tac"], pm["tac"]
        out["tac_A"], out["tac_M"] = np.nanmean(a), np.nanmean(b)
        out["tac_err"] = np.nanmean(b) / np.nanmean(a) - 1
        bt = [np.nanmean(rng.choice(b, len(b))) / np.nanmean(rng.choice(a, len(a))) - 1 for _ in range(NB)]
        out["tac_err_lo"], out["tac_err_hi"] = np.percentile(bt, [2.5, 97.5])
    if system == "Toggle":
        nA, nM = pa["nswitch"].sum(), pm["nswitch"].sum()
        out["switch_A"], out["switch_M"] = nA, nM
        out["switch_M_lo"], out["switch_M_hi"] = stats.poisson.interval(0.95, nM) if nM > 0 else (0, 3.69)
    return out


def s1_baseline(M, s, om, TA, burn):
    """Closed-form baseline of Eq. (S1): promoter share of the variance of a
    product transcribed by a two-state promoter, CV^2 = 1/<n> + (pi_off/pi_on)
    eta/(1+eta), evaluated at the operating point (time-averaged regulator,
    lambda2 there, regulator lifetime tau)."""
    from engine import _stationary
    X = np.zeros(len(M.species))
    for m, j in enumerate(M.observe):
        X[j] = TA[:, burn:, m].mean()
    A = M.arrays(1.0, s)
    x = X.copy()
    _stationary(x, A["k"], A["fi"], A["fe"], A["isD"], A["isprom"], A["src"], A["tgt"], A["blk"], A["bmem"], x)
    pon = 0.0
    seen = set()
    for r in range(len(A["k"])):
        if A["isprom"][r]:
            continue
        for m_ in range(A["fi"].shape[1]):
            j = A["fi"][r, m_]
            if j >= 0 and A["isD"][j] and j not in seen:
                seen.add(j); pon += x[j]
    pon = min(max(pon, 1e-9), 1 - 1e-9)
    et = eta(M, s, [X], M.tau_prot)[0]
    N = TA[:, burn:, 0].mean() * om
    prom = (1 - pon) / pon * et / (1 + et)
    return prom / (prom + 1.0 / N)


def eta_of(M, s, TA, burn):
    X = np.zeros((TA.shape[0] * (TA.shape[1] - burn), len(M.species)))
    for m, j in enumerate(M.observe):
        X[:, j] = TA[:, burn:, m].ravel()
    op = eta(M, s, [X.mean(axis=0)], M.tau_prot)[0]
    idx = np.linspace(0, len(X) - 1, 3000).astype(int)
    rows = np.vstack([X[idx], np.zeros(len(M.species))])
    ev = eta(M, s, rows, M.tau_prot)
    return op, float(ev.max())


def _condition_rows(cond):
    """Metrics of every method for one (model, s, Omega) condition."""
    model, s, om = cond
    rows = []
    s, om = float(s), float(om); system = model.split("-")[0]
    M = MODELS[model](); burn = int(M.t_burn / M.record_dt)
    TA, wA = load(model, "ssa", s, om, "A")
    op, wc = eta_of(M, s, TA, burn)
    N = TA[:, burn:, 0].mean() * om
    base = dict(model=model, system=system, arch=model[-1], s=s, omega=om, eta_op=op, eta_wc=wc, N=N,
                tau_reg=M.tau_prot, wall_ref=wA.mean())
    if system in STATIONARY:
        from lna import lna
        try:
            r = lna(M, om, s)
            base.update(lna_share=r["share_prom"], lna_var=r["var"], ssa_var=(TA[:, burn:, 0] * om).var())
            base.update(s1_share=s1_baseline(M, s, om, TA, burn))
            if system == "p53":
                r2 = lna(M, om, s, obs_i=1)
                base.update(lna_share_mdm2=r2["share_prom"])
        except Exception as e:
            print("LNA failed", model, s, om, e)
    for method, ens in (("ssa", "B"), ("pdmp", "A"), ("sqssa", "A"), ("ode", "A"), ("pdmp", "h4"), ("pdmpx", "A")):
        if ens == "h4" and om != BASE_OMEGA[system]:
            continue
        TM, wM = load(model, method, s, om, ens)
        if TM is None:
            continue
        r = dict(base); r["method"] = {"B": "ssa_null", "h4": "pdmp_h4"}.get(ens, method)
        r["wall"] = wM.mean()
        r.update(compare(TA, TM, M, burn, system, method))
        rows.append(r)
    print("done", model, s, om, flush=True)
    return rows


def main(only=None):
    rows, arch = [], []
    files = glob.glob(os.path.join(DIR, "*_ssa_*_A.npz"))
    conds = sorted({re.match(r"([\w]+-[HSEMP])_ssa_s([\d.e+-]+)_om([\d.e+-]+)_A", os.path.basename(f)).groups()
                    for f in files})
    todo = [(m, s_, o) for m, s_, o in conds if not (only and m not in only)]
    import concurrent.futures as fut
    workers = int(os.environ.get("ANALYZE_WORKERS", "12"))
    with fut.ProcessPoolExecutor(max_workers=workers) as ex:
        for part in ex.map(_condition_rows, todo, chunksize=1):
            rows += part
    df = pd.DataFrame(rows)
    path = os.path.join(RES, "suite2_metrics.csv")
    if only and os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old[~old.model.isin(only)], df], ignore_index=True)
    df.to_csv(path, index=False)
    # architecture comparison: SSA and ODE of S against those of H, same condition
    systems = {m.split("-")[0] for m in (only or MODELS)}
    for system in systems:
      for ref in ("H", "E", "M"):
        if system + "-S" not in MODELS or system + "-" + ref not in MODELS:
            continue
        for (s, om) in {(c[1], c[2]) for c in conds if c[0].startswith(system)}:
            s, om = float(s), float(om)
            MH, MS = MODELS[system + "-" + ref](), MODELS[system + "-S"]()
            burn = int(MH.t_burn / MH.record_dt)
            H, _ = load(system + "-" + ref, "ssa", s, om, "A"); S_, _ = load(system + "-S", "ssa", s, om, "A")
            if H is None or S_ is None:
                continue
            r = dict(system=system, ref=ref, s=s, omega=om)
            r.update({"ssa_" + k: v for k, v in compare(H, S_, MH, burn, system, "ssa").items()})
            OH, _ = load(system + "-" + ref, "ode", s, om, "A"); OS, _ = load(system + "-S", "ode", s, om, "A")
            r["ode_mean_err"] = OS[0, burn:, 0].mean() / OH[0, burn:, 0].mean() - 1
            if system in OSCILLATOR:
                r["ode_period_H"] = acf_period(OH[0, burn:, 0], MH.record_dt)
                r["ode_period_S"] = acf_period(OS[0, burn:, 0], MH.record_dt)
                r["ode_amp_H"] = OH[0, burn:, 0].std(); r["ode_amp_S"] = OS[0, burn:, 0].std()
            arch.append(r)
    arch = pd.DataFrame(arch)
    apath = os.path.join(RES, "suite2_architecture.csv")
    if only and os.path.exists(apath):
        old = pd.read_csv(apath)
        arch = pd.concat([old[~old.system.isin(systems)], arch], ignore_index=True)
    arch.to_csv(apath, index=False)
    print("wrote", len(rows), "rows and", len(arch), "architecture rows")


if __name__ == "__main__":
    import sys
    main(sys.argv[1].split(",") if len(sys.argv) > 1 else None)
