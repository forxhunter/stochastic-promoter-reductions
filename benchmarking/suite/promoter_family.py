#!/usr/bin/env python3
"""Architecture test beyond nearly two-state promoters, for the two autoregulated
genes (one promoter, one species). For each promoter P, the controls are two-state
promoters with exactly the response of P:

  F   relaxation rate lambda_F(x) = 2 pi_on pi_off / S0_P(x): same S0(x) at every x
  D   relaxation lambda * Z(x) (Z the binding polynomial of the site-resolved
      promoter, the family of E, M, D in the main text), lambda matched in S0
      averaged over the exact stationary distribution of P
  C   constant relaxation rate lambda, matched in the same way

Promoters P: site-resolved with n = 2 and pairwise factor omega in {1, 3, 10, 100}
(half-maximum and worst-case relaxation matched to the concerted promoter as in
the main text), and a non-equilibrium refractory cycle whose off-time is the sum of
two exponential stages. Exact stationary moments by finite-state projection on
(promoter state, copy number); strict first-order coefficient c1 of P against each
control from the local expansion of the quasi-steady-state diffusion
approximation; spectral ratio lambda_3/lambda_2 of P at the fixed point.
Writes benchmarking_results/promoter_family.csv.
"""
import os
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.optimize import brentq
from math import comb
from systems2 import match_site
from qss_diffusion import fundamental, CIRC
import fsp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
N_SITES, K_HALF = 2, 30.0
REPRESS = {"NAR": True, "PFB": False}


# ---------------------------------------------------------------- promoter chains
def site_promoter(sysname, omega):
    KS, kb = match_site(N_SITES, K_HALF, 1.0, REPRESS[sysname], 150.0, omega=omega)
    n = N_SITES
    act = np.array([1.0] * n + [0.0] if REPRESS[sysname] else [0.0] * n + [1.0])

    def Q(x):
        q = np.zeros((n + 1, n + 1))
        for k in range(n):
            q[k, k + 1] = (n - k) * kb * x / KS
            q[k + 1, k] = (k + 1) * kb * omega ** (-k)
        return q - np.diag(q.sum(axis=1))
    coef = [comb(n, k) * omega ** (k * (k - 1) / 2) * KS ** (-k) for k in range(n + 1)]
    Z = lambda x: sum(c * x ** k for k, c in enumerate(coef))
    return Q, act, Z


def cycle_promoter(sysname, c=2.0, kr=1.0):
    """Three-state non-equilibrium cycle. Repressor (NAR): A -> R1 at kr (x/K)^2,
    R1 -> R2 -> A at rate c each. Activator (PFB): I1 -> I2 at c, I2 -> A at
    kr (x/K)^2, A -> I1 at c/2, I1 -> I2 at 5c; the off-time before activation
    includes a refractory stage. The regulator dependence is concerted (order 2)."""
    if REPRESS[sysname]:
        act = np.array([1.0, 0.0, 0.0])                  # A, R1, R2
        def Q(x):
            q = np.zeros((3, 3)); q[0, 1] = kr * (x / K_HALF) ** 2; q[1, 2] = c; q[2, 0] = c
            return q - np.diag(q.sum(axis=1))
    else:
        act = np.array([0.0, 0.0, 1.0])                  # I1, I2, A
        def Q(x):
            q = np.zeros((3, 3)); q[0, 1] = 5 * c; q[1, 2] = kr * (x / K_HALF) ** 2; q[2, 0] = c / 2
            return q - np.diag(q.sum(axis=1))
    return Q, act, None


# ---------------------------------------------------------------- helpers
def pon_S0(Q, act, x):
    """Activity and zero-frequency noise strength (s = 1) at each x."""
    p, s0 = [], []
    for v in np.atleast_1d(x):
        pi, Zm = fundamental(Q(v))
        a = act - np.dot(pi, act)
        p.append(np.dot(pi, act)); s0.append(2 * np.dot(pi * a, Zm @ a))
    return np.array(p), np.array(s0)


def two_state(pon_fun, lam_fun):
    def Q(x):
        p = pon_fun(x); l = lam_fun(x)
        return np.array([[-l * p, l * p], [l * (1 - p), -l * (1 - p)]])  # state 0 off, 1 on
    return Q, np.array([0.0, 1.0])


def fsp_moments(sysname, Q, act, s, om):
    c = CIRC[sysname]
    nmax = int(om * (c["beta"] + c["alpha"]) / c["gamma"] + 12 * np.sqrt(om * (c["beta"] + c["alpha"])) + 30)
    nn = np.arange(nmax + 1); x = np.maximum(nn / om, 1e-2)
    m = len(act); N = nmax + 1
    idx = lambda st, k: st * N + k
    rows, cols, vals = [], [], []
    qs = np.array([Q(v) for v in x]) * s                 # (N, m, m)
    for a in range(m):
        birth = om * (c["beta"] + c["alpha"] * act[a]) * np.ones(N)
        k = np.arange(N - 1); rows.append(idx(a, k)); cols.append(idx(a, k + 1)); vals.append(birth[:-1])
        k = np.arange(1, N); rows.append(idx(a, k)); cols.append(idx(a, k - 1)); vals.append(c["gamma"] * nn[1:])
        for b in range(m):
            if a != b:
                rows.append(idx(a, nn)); cols.append(idx(b, nn)); vals.append(qs[:, a, b])
    r = np.concatenate(rows); cc = np.concatenate(cols); v = np.concatenate(vals)
    ok = v > 0
    G = sp.coo_matrix((v[ok], (r[ok], cc[ok])), shape=(m * N, m * N)).tocsr()
    G = G - sp.diags(np.asarray(G.sum(axis=1)).ravel())
    p = fsp.stationary(G)
    px = p.reshape(m, N).sum(axis=0)
    mean = np.dot(px, nn); var = np.dot(px, (nn - mean) ** 2)
    return mean / om, var, px


def local_shift(sysname, Q, act, xs, h=1e-3):
    """Promoter part of the first-order mean shift at s = 1 (local expansion)."""
    c = CIRC[sysname]
    x = xs + h * np.arange(-3, 4)
    Fb, Dp, b = [], [], []
    phis, pis, Fs = [], [], []
    for v in x:
        pi, Zm = fundamental(Q(v))
        F = c["beta"] + c["alpha"] * act - c["gamma"] * v
        fb = np.dot(pi, F); phi = Zm @ (F - fb)
        Fb.append(fb); Dp.append(np.dot(pi * (F - fb), phi)); phis.append(phi); pis.append(pi); Fs.append(F)
    Fb = np.array(Fb); phis = np.array(phis); pis = np.array(pis); Fs = np.array(Fs)
    dphi = np.gradient(phis, x, axis=0)
    b = np.sum(pis * Fs * dphi, axis=1)
    F1 = np.gradient(Fb, x)[3]; F2 = np.gradient(np.gradient(Fb, x), x)[3]
    return -(0.5 * F2 * Dp[3] / abs(F1) + b[3]) / F1


def fixed_point(sysname, Q, act):
    c = CIRC[sysname]
    f = lambda v: c["beta"] + c["alpha"] * pon_S0(Q, act, v)[0][0] - c["gamma"] * v
    grid = np.linspace(1e-3, 200.0, 2001)
    vals = np.array([f(v) for v in grid])
    roots = [brentq(f, grid[i], grid[i + 1]) for i in range(len(grid) - 1) if vals[i] > 0 >= vals[i + 1]]
    return max(roots)                                   # the stable fixed point with the highest level


def spectral_ratio(Q, x):
    ev = np.sort(np.abs(np.real(np.linalg.eigvals(Q(x)))))
    return ev[2] / ev[1] if len(ev) > 2 else np.inf


def run(sysname, label, Q, act, Z, s_list=(0.1, 0.3, 1.0, 3.0, 10.0), om=1.0):
    xs = fixed_point(sysname, Q, act)
    pon = lambda x: pon_S0(Q, act, x)[0]
    S0 = lambda x: pon_S0(Q, act, x)[1]
    lamF = lambda x: 2 * pon(x) * (1 - pon(x)) / S0(x)
    QF, aF = two_state(lambda x: pon(x)[0] if np.ndim(pon(x)) else pon(x), lambda x: lamF(x)[0])
    dP = local_shift(sysname, Q, act, xs)
    dF = local_shift(sysname, QF, aF, xs)
    rows = []
    for s in s_list:
        mP, vP, px = fsp_moments(sysname, Q, act, s, om)
        xv = np.arange(len(px)) / om
        xv_e = np.maximum(xv, 1e-2)
        p_x, s0_x = pon_S0(Q, act, xv_e)
        g2 = 2 * p_x * (1 - p_x)                        # S0 * lambda for a two-state promoter
        mF, vF, _ = fsp_moments(sysname, QF, aF, s, om)
        r = dict(system=sysname, promoter=label, s=s, omega=om, x_star=xs,
                 lam3_lam2=spectral_ratio(Q, xs), mean_P=mP, cv_P=np.sqrt(vP) / (mP * om),
                 S_minus_F=mP / mF - 1, cv_S_minus_F=(np.sqrt(vP) / mP) / (np.sqrt(vF) / mF) - 1,
                 c1_F=100 * (dP - dF) / xs)
        # constant-lambda family C, matched in S0 averaged over the distribution of P
        lamC = np.dot(px, g2) / np.dot(px, s0_x)
        QC, aC = two_state(lambda x: pon(x)[0], lambda x: lamC)
        mC, vC, _ = fsp_moments(sysname, QC, aC, s, om)
        r["S_minus_C"] = mP / mC - 1; r["c1_C"] = 100 * (dP - local_shift(sysname, QC, aC, xs)) / xs
        if Z is not None:                                # lambda * Z(x) family (D)
            lamD = np.dot(px, g2 / Z(xv_e)) / np.dot(px, s0_x)
            QD, aD = two_state(lambda x: pon(x)[0], lambda x: lamD * Z(x))
            mD, vD, _ = fsp_moments(sysname, QD, aD, s, om)
            r["S_minus_D"] = mP / mD - 1; r["c1_D"] = 100 * (dP - local_shift(sysname, QD, aD, xs)) / xs
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return rows


if __name__ == "__main__":
    rows = []
    for sysname in ("NAR", "PFB"):
        for omega in (1.0, 3.0, 10.0, 100.0):
            Q, act, Z = site_promoter(sysname, omega)
            rows += run(sysname, "site omega=%g" % omega, Q, act, Z)
        Q, act, Z = cycle_promoter(sysname)
        rows += run(sysname, "refractory cycle", Q, act, Z)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "promoter_family.csv"), index=False)
