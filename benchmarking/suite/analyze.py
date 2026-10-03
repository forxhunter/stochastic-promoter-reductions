#!/usr/bin/env python3
"""Metrics for every cached condition of the benchmark grid.

For each (system, s, Omega), each method is compared against SSA ensemble A.
SSA ensemble B, run independently with the same settings, gives the null value
of every metric at the same sample size.

Writes benchmarking_results/suite_metrics.csv.
"""
import os, glob, re
import numpy as np
import pandas as pd
from scipy import stats

from systems import SYSTEMS, BASE_OMEGA
from engine import eta
from observables import acf_period, switching

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
DIR = os.path.join(RES, "suite")
OSC = {"Repressilator", "Goodwin"}


def load(system, method, s, om, ens):
    p = os.path.join(DIR, "%s_%s_s%g_om%g_%s.npz" % (system, method, s, om, ens))
    if not os.path.exists(p):
        return None, None
    d = np.load(p)
    return d["traj"].astype(float), d["wall"]


def acf_time(y, dt):
    """Lag at which the autocorrelation first falls below 1/e."""
    y = y - y.mean()
    if np.allclose(y, 0):
        return np.nan
    n = len(y)
    f = np.fft.rfft(y, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    i = np.where(ac < np.exp(-1))[0]
    return i[0] * dt if len(i) else np.nan


def per_traj(T, M, burn, system):
    """Per-trajectory observables of the primary species (column 0)."""
    Y = T[:, burn:, 0]
    d = {"level": Y.mean(axis=1), "amp": Y.std(axis=1)}
    if system in OSC:
        d["period"] = np.array([acf_period(y, M.record_dt) for y in Y])
    else:
        d["tac"] = np.array([acf_time(y, M.record_dt) for y in Y])
    if system == "Toggle":
        r, f = switching(T[:, :, 0], T[:, :, 1], M.record_dt, burn)
        d["switch"] = r; d["frac_high"] = f
    return d


def rel(a, b):
    return (a - b) / b if b != 0 and np.isfinite(b) else np.nan


def compare(TA, TM, M, burn, system):
    thin = max(1, int(round(M.tau_prot / M.record_dt)))
    YA, YM = TA[:, burn:, 0], TM[:, burn:, 0]
    out = {}
    out["mean_err"] = rel(YM.mean(), YA.mean())
    out["cv_err"] = rel(YM.std() / YM.mean(), YA.std() / YA.mean()) if YM.mean() > 0 else np.nan
    out["ks"] = stats.ks_2samp(YM[:, ::thin].ravel(), YA[:, ::thin].ravel()).statistic
    pa, pm = per_traj(TA, M, burn, system), per_traj(TM, M, burn, system)
    out["amp_err"] = rel(np.nanmean(pm["amp"]), np.nanmean(pa["amp"]))
    key = "period" if system in OSC else "tac"
    a, b = pa[key], pm[key]
    out["time_obs"] = key
    out["time_A"] = np.nanmean(a); out["time_M"] = np.nanmean(b)
    out["time_err"] = rel(np.nanmean(b), np.nanmean(a))
    out["time_nan_frac_M"] = float(np.mean(~np.isfinite(b)))
    aa, bb = a[np.isfinite(a)], b[np.isfinite(b)]
    out["time_p"] = stats.ttest_ind(bb, aa, equal_var=False).pvalue if len(aa) > 2 and len(bb) > 2 else np.nan
    if system == "Toggle":
        out["switch_A"] = pa["switch"].mean(); out["switch_M"] = pm["switch"].mean()
        out["frac_high_A"] = pa["frac_high"].mean(); out["frac_high_M"] = pm["frac_high"].mean()
    return out


def eta_of(M, s, TA, burn):
    X = np.zeros((TA.shape[0] * (TA.shape[1] - burn), len(M.species)))
    for m, j in enumerate(M.observe):
        X[:, j] = TA[:, burn:, m].ravel()
    op = eta(M, s, [X.mean(axis=0)], M.tau_prot)[0]
    idx = np.linspace(0, len(X) - 1, 4000).astype(int)
    ev = eta(M, s, X[idx], M.tau_prot)
    return op, float(ev.max()), float(np.median(ev))


def main():
    rows = []
    files = glob.glob(os.path.join(DIR, "*_ssa_*_A.npz"))
    conds = sorted({re.match(r"(\w+?)_ssa_s([\d.e+-]+)_om([\d.e+-]+)_A", os.path.basename(f)).groups()
                    for f in files})
    for system, s, om in conds:
        s, om = float(s), float(om)
        M = SYSTEMS[system](); burn = int(M.t_burn / M.record_dt)
        TA, wA = load(system, "ssa", s, om, "A")
        op, worst, med = eta_of(M, s, TA, burn)
        for method, ens in (("ssa", "B"), ("pdmp", "A"), ("sqssa", "A"), ("ode", "A"), ("pdmp", "h4")):
            if ens == "h4" and not (s == 1 and om == BASE_OMEGA[system]):
                continue
            TM, wM = load(system, method, s, om, ens)
            if TM is None:
                continue
            r = dict(system=system, s=s, omega=om, method=method if ens != "B" else "ssa_null",
                     ens=ens, eta_op=op, eta_worst=worst, eta_median=med,
                     n_ref=TA.shape[0], n=TM.shape[0],
                     wall_ref=wA.mean(), wall=wM.mean(), speedup=wA.mean() / wM.mean())
            if method == "pdmp" and ens == "h4":
                r["method"] = "pdmp_h4"
            r.update(compare(TA, TM, M, burn, system))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "suite_metrics.csv"), index=False)
    print(df.to_string(max_rows=400, float_format=lambda v: "%.3g" % v))


if __name__ == "__main__":
    main()
