#!/usr/bin/env python3
"""Extended architecture test for the two autoregulated genes.

Promoters P: site-resolved with n = 2 (omega = 1, 3, 10, 100) and n = 3
(omega = 1, 10), refractory promoters with two and three stages, all at Omega = 1
for s from 0.03 to 10. For each P:

  F     two-state control with the response and S0(x) of P at every x
  W     two-state control with the response of P and the same stationary flux J(x)
        between the active and inactive groups (mean dwell times matched):
        k_on = J / (1 - r), k_off = J / r
  D, C  the lambda Z(x) and constant-rate families matched on average (as before)

Reported: exact differences in mean and CV, the Wasserstein-1 distance divided by
the SD of P and the total-variation distance between the stationary
distributions, and the decomposition of the first-order drift
b = b_2(r, S0) - R, where b_2 is the two-state expression and
R = r F_on sum_on rho_m' phi_m + (1 - r) F_off sum_off rho_m' phi_m with rho the
conditional distributions within the active and inactive groups. R vanishes when
those conditional distributions do not depend on x; c1 against F equals
100 R / (F' x*). Writes benchmarking_results/promoter_family2.csv.
"""
import os
import numpy as np, pandas as pd
from math import comb
from systems2 import match_site
from qss_diffusion import fundamental, CIRC
from promoter_family import (REPRESS, K_HALF, pon_S0, two_state, fsp_moments, local_shift,
                             fixed_point, spectral_ratio)

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def site(sysname, n, omega):
    KS, kb = match_site(n, K_HALF, 1.0, REPRESS[sysname], 150.0, omega=omega)
    act = np.array([1.0] * n + [0.0] if REPRESS[sysname] else [0.0] * n + [1.0])

    def Q(x):
        q = np.zeros((n + 1, n + 1))
        for k in range(n):
            q[k, k + 1] = (n - k) * kb * x / KS
            q[k + 1, k] = (k + 1) * kb * omega ** (-k)
        return q - np.diag(q.sum(axis=1))
    coef = [comb(n, k) * omega ** (k * (k - 1) / 2) * KS ** (-k) for k in range(n + 1)]
    return Q, act, (lambda x: sum(c * x ** k for k, c in enumerate(coef)))


def refractory(sysname, stages, c=2.0, kr=1.0):
    """Repressor: A -> R1 at kr (x/K)^2, then R1 -> ... -> R_stages -> A at rate c*stages/2
    each (mean refractory time 2/c, independent of the number of stages). Activator:
    A -> I1 at c/2, I1 -> ... -> I_stages at rate 5c*stages each... the last stage I_k ->
    A at kr (x/K)^2."""
    m = stages + 1
    if REPRESS[sysname]:
        act = np.zeros(m); act[0] = 1.0
        def Q(x):
            q = np.zeros((m, m)); q[0, 1] = kr * (x / K_HALF) ** 2
            for i in range(1, m):
                q[i, (i + 1) % m] = c * stages / 2
            return q - np.diag(q.sum(axis=1))
    else:
        act = np.zeros(m); act[-1] = 1.0
        def Q(x):
            q = np.zeros((m, m)); q[m - 1, 0] = c / 2
            for i in range(0, m - 2):
                q[i, i + 1] = 5 * c * (stages - 1)
            q[m - 2, m - 1] = kr * (x / K_HALF) ** 2
            return q - np.diag(q.sum(axis=1))
    return Q, act, None


def flux(Q, act, x):
    pi, _ = fundamental(Q(x))
    q = Q(x); on = act > 0.5
    return sum(pi[i] * q[i, j] for i in np.where(on)[0] for j in np.where(~on)[0])


def residual_R(sysname, Q, act, xs, h=1e-3):
    c = CIRC[sysname]
    x = xs + h * np.arange(-3, 4)
    on = act > 0.5
    pis, phis = [], []
    for v in x:
        pi, Zm = fundamental(Q(v))
        F = c["beta"] + c["alpha"] * act - c["gamma"] * v
        pis.append(pi); phis.append(Zm @ (F - np.dot(pi, F)))
    pis = np.array(pis); phis = np.array(phis)
    r = pis[:, on].sum(axis=1)
    rho_on = pis[:, on] / r[:, None]; rho_off = pis[:, ~on] / (1 - r)[:, None]
    drho_on = np.gradient(rho_on, x, axis=0)[3]; drho_off = np.gradient(rho_off, x, axis=0)[3]
    Fon = c["beta"] + c["alpha"] - c["gamma"] * xs; Foff = c["beta"] - c["gamma"] * xs
    R = r[3] * Fon * np.dot(drho_on, phis[3, on]) + (1 - r[3]) * Foff * np.dot(drho_off, phis[3, ~on])
    Fb = c["beta"] + c["alpha"] * r - c["gamma"] * x
    F1 = np.gradient(Fb, x)[3]
    return R, 100 * R / (F1 * xs)


def distances(px, py):
    n = max(len(px), len(py))
    px = np.pad(px, (0, n - len(px))); py = np.pad(py, (0, n - len(py)))
    k = np.arange(n); sd = np.sqrt(np.dot(px, k ** 2) - np.dot(px, k) ** 2)
    w1 = np.abs(np.cumsum(px) - np.cumsum(py)).sum() / sd
    tv = 0.5 * np.abs(px - py).sum()
    return w1, tv


def run(sysname, label, Q, act, Z, s_list=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0), om=1.0):
    xs = fixed_point(sysname, Q, act)
    pon = lambda x: pon_S0(Q, act, x)[0][0]
    S0 = lambda x: pon_S0(Q, act, x)[1][0]
    QF, aF = two_state(pon, lambda x: 2 * pon(x) * (1 - pon(x)) / S0(x))
    QW, aW = two_state(pon, lambda x: flux(Q, act, x) / (pon(x) * (1 - pon(x))))
    dP = local_shift(sysname, Q, act, xs)
    R, c1R = residual_R(sysname, Q, act, xs)
    c1F = 100 * (dP - local_shift(sysname, QF, aF, xs)) / xs
    c1W = 100 * (dP - local_shift(sysname, QW, aW, xs)) / xs
    rows = []
    for s in s_list:
        mP, vP, pP = fsp_moments(sysname, Q, act, s, om)
        r = dict(system=sysname, promoter=label, s=s, x_star=xs, lam3_lam2=spectral_ratio(Q, xs),
                 mean_P=mP, cv_P=np.sqrt(vP) / (mP * om), c1_F=c1F, c1_R=c1R, R=R, c1_W=c1W)
        for tag, (Qc, ac) in (("F", (QF, aF)), ("W", (QW, aW))):
            mc, vc, pc = fsp_moments(sysname, Qc, ac, s, om)
            w1, tv = distances(pP, pc)
            r["S_minus_" + tag] = mP / mc - 1
            r["cv_S_minus_" + tag] = (np.sqrt(vP) / mP) / (np.sqrt(vc) / mc) - 1
            r["W1_" + tag] = w1; r["TV_" + tag] = tv
            if sysname == "PFB":
                lo = int(round(0.25 * xs * om))          # weight below a quarter of the fixed point
                r["low_P"] = pP[:lo].sum(); r["low_" + tag] = pc[:lo].sum()
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return rows


if __name__ == "__main__":
    rows = []
    for sysname in ("NAR", "PFB"):
        for n, omega in ((2, 1.0), (2, 3.0), (2, 10.0), (2, 100.0), (3, 1.0), (3, 10.0)):
            Q, act, Z = site(sysname, n, omega)
            rows += run(sysname, "site n=%d omega=%g" % (n, omega), Q, act, Z)
        for stages in (2, 3):
            Q, act, Z = refractory(sysname, stages)
            rows += run(sysname, "refractory %d stages" % stages, Q, act, Z)
        pd.DataFrame(rows).to_csv(os.path.join(RES, "promoter_family2.csv"), index=False)
