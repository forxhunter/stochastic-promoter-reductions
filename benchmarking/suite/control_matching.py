#!/usr/bin/env python3
"""Exact mean and variance of the exact-response two-state control as a function
of its switching scale lambda, for the two autoregulated genes, by finite state
projection, and the lambda given by four matching rules:

  E   worst-case relaxation (slowest lambda_2 over the regulator range)
  M   zero-frequency noise strength S0 at the deterministic fixed point of S
  Mop S0 at the exact stationary mean regulator level of the S circuit
  D   S0 averaged over the exact stationary distribution of the regulator in S
  R   relaxation rate lambda_2 at the exact stationary mean of S

All matching uses the S circuit at s = 1 and Omega = 1. The lambda that
reproduces the exact S mean is found by bisection. Writes
benchmarking_results/control_matching.csv (one row per circuit, rule) and
control_lambda_scan.csv (mean against lambda).
"""
import os
import numpy as np, pandas as pd
from scipy.optimize import brentq
from systems2 import (MODELS, OMEGA_COOP, match_site, site_Q, noise_strength, X_OP,
                      _operating_points, lam2_site)
from math import comb
import fsp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
PARS = {"NAR": dict(n=2, K=30.0, kb=1.0, repress=True, xmax=150.0),
        "PFB": dict(n=2, K=30.0, kb=1.0, repress=False, xmax=150.0)}


def exact_moments(M, s, om, scale=1.0):
    """FSP mean (concentration) and variance (copy number) of the observed
    species, with promoter rates multiplied by `scale`."""
    rx0 = M.reactions
    M.reactions = [dict(r, k=r["k"] * scale) if r["w"] == 0 else r for r in rx0]
    try:
        Q, X, box = fsp.build(M, om, s, False)
        p = fsp.stationary(Q)
        mean, var, _ = fsp.moments(M, p, X, box)
    finally:
        M.reactions = rx0
    return mean / om, var, p, X


def s0_two_state(coef, act, x):
    """S0 * lambda of the two-state control at x: 2 pon (1 - pon) / sum w."""
    w = np.array([c * x ** k for k, c in enumerate(coef)])
    pon = np.dot(act, w) / w.sum(axis=0)
    return 2 * pon * (1 - pon) / w.sum(axis=0)


def main():
    rows, scan = [], []
    for sysname, P in PARS.items():
        n = P["n"]
        KS, kbs = match_site(n, P["K"], P["kb"], P["repress"], P["xmax"])
        coef = [comb(n, k) * OMEGA_COOP ** (k * (k - 1) / 2) * KS ** (-k) for k in range(n + 1)]
        act = np.array([1.0] * n + [0.0] if P["repress"] else [0.0] * n + [1.0])
        MS = MODELS[sysname + "-S"]()
        mS, vS, p, X = exact_moments(MS, 1.0, 1.0)
        j = MS.idx[MS.reg_species[0]]
        xs = X[:, j].astype(float)                         # Omega = 1: copy number = concentration
        vals = np.unique(xs)
        px = np.array([p[xs == v].sum() for v in vals])
        S0S = np.array([noise_strength(site_Q(v, n, KS, kbs, OMEGA_COOP), act)[0] for v in vals])
        g = s0_two_state(coef, act, vals)
        xm = np.dot(px, vals)
        xfp = _operating_points()[sysname]
        lam = {}
        lam["M"] = s0_two_state(coef, act, xfp) / noise_strength(site_Q(xfp, n, KS, kbs, OMEGA_COOP), act)[0]
        lam["Mop"] = s0_two_state(coef, act, xm) / noise_strength(site_Q(xm, n, KS, kbs, OMEGA_COOP), act)[0]
        lam["D"] = np.dot(px, g) / np.dot(px, S0S)
        wsum = lambda x: sum(c * x ** k for k, c in enumerate(coef))
        lam["R"] = lam2_site([xm], n, KS, kbs, OMEGA_COOP)[0] / wsum(xm)
        xs_e = np.concatenate([[0.0], np.logspace(np.log10(P["K"]) - 4, np.log10(P["xmax"]), 200)])
        lam["E"] = lam2_site(xs_e, n, KS, kbs, OMEGA_COOP).min()
        # the M model as built uses lam["M"]; other rules rescale its promoter rates
        X_OP["S"] = xfp
        MM = MODELS[sysname + "-M"]()
        lam_built = lam["M"]
        def mean_at(l, s=1.0, om=1.0):
            return exact_moments(MM, s, om, scale=l / lam_built)[:2]
        for l in np.logspace(np.log10(lam["M"]) - 1, np.log10(lam["E"]) + 1, 25):
            m, v = mean_at(l)
            scan.append(dict(system=sysname, lam=l, mean=m, var=v))
        lo, hi = lam["M"] / 100, lam["E"] * 100
        f = lambda ll: mean_at(np.exp(ll))[0] - mS
        try:
            lam["fit"] = np.exp(brentq(f, np.log(lo), np.log(hi), xtol=1e-4))
        except ValueError:
            lam["fit"] = np.nan
        for rule, l in lam.items():
            r = dict(system=sysname, rule=rule, lam=l, x_mean_S=xm, x_fp=xfp, mean_S=mS, var_S=vS)
            if np.isfinite(l):
                for s in (0.3, 1.0, 3.0):
                    m, v = mean_at(l, s=s)
                    mS_s = exact_moments(MS, s, 1.0)[0] if s != 1.0 else mS
                    r["mean_s%g" % s] = m; r["S_minus_ctrl_s%g" % s] = mS_s / m - 1
                # relaxation of the control at the S operating level, relative to S
                r["lam2_ratio_at_xm"] = l * wsum(xm) / lam2_site([xm], n, KS, kbs, OMEGA_COOP)[0]
            rows.append(r)
            print(r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "control_matching.csv"), index=False)
    pd.DataFrame(scan).to_csv(os.path.join(RES, "control_lambda_scan.csv"), index=False)


if __name__ == "__main__":
    main()


def lambda_D(sysname, s, om):
    """lambda of the control D matched at condition (s, Omega): S0 averaged over the
    exact stationary distribution of the regulator in the S circuit at that
    condition equals the same average for the two-state control. Both S0 scale
    as 1/s, so the ratio is taken with the promoter rates at s = 1."""
    P = PARS[sysname]; n = P["n"]
    KS, kbs = match_site(n, P["K"], P["kb"], P["repress"], P["xmax"])
    coef = [comb(n, k) * OMEGA_COOP ** (k * (k - 1) / 2) * KS ** (-k) for k in range(n + 1)]
    act = np.array([1.0] * n + [0.0] if P["repress"] else [0.0] * n + [1.0])
    MS = MODELS[sysname + "-S"]()
    _, _, p, X = exact_moments(MS, s, om)
    j = MS.idx[MS.reg_species[0]]
    xs = X[:, j].astype(float) / om
    vals = np.unique(xs)
    px = np.array([p[xs == v].sum() for v in vals])
    S0S = np.array([noise_strength(site_Q(v, n, KS, kbs, OMEGA_COOP), act)[0] for v in vals])
    g = s0_two_state(coef, act, vals)
    return np.dot(px, g) / np.dot(px, S0S)
