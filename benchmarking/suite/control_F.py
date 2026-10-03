#!/usr/bin/env python3
"""Control F for the two autoregulated genes: a two-state promoter with exactly
the site-resolved response pi_on(x) and the same zero-frequency noise strength
S0(x) as the site-resolved promoter at every regulator level x. Its relaxation
rate is lambda_F(x) = 2 pi_on pi_off / S0^S(x) (both at s = 1, scaled by s), so
that the only remaining difference from S is the correlation structure of the
promoter beyond S0. Exact stationary moments by finite-state projection on
(promoter state, copy number), with rates evaluated at x = n / Omega as for the
other exact-response controls; strict first-order coefficient c1 from the local
expansion. Writes benchmarking_results/control_F.csv."""
import os
import numpy as np, pandas as pd
import scipy.sparse as sp
from systems2 import match_site, site_Q, noise_strength, site_activity, OMEGA_COOP, _operating_points
from control_matching import PARS
from qss_diffusion import CIRC
import fsp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def functions(sysname):
    P = PARS[sysname]; n = P["n"]
    KS, kbs = match_site(n, P["K"], P["kb"], P["repress"], P["xmax"])
    act = np.array([1.0] * n + [0.0] if P["repress"] else [0.0] * n + [1.0])
    def pon(x):
        return site_activity(np.asarray(x, float), n, KS, OMEGA_COOP, P["repress"])
    def S0S(x):
        return np.array([noise_strength(site_Q(v, n, KS, kbs, OMEGA_COOP), act)[0] for v in np.atleast_1d(x)])
    def lamF(x):
        p = pon(x)
        return 2 * p * (1 - p) / S0S(x)
    return pon, lamF


def exact(sysname, s, om, nmax=None):
    c = CIRC[sysname]; pon, lamF = functions(sysname)
    nmax = nmax or int(om * (c["beta"] + c["alpha"]) / c["gamma"] + 12 * np.sqrt(om * (c["beta"] + c["alpha"])) + 30)
    nn = np.arange(nmax + 1); x = nn / om
    xe = np.maximum(x, 1e-2)                              # lambda_F is 0/0 at x = 0; use its limit
    p = pon(x); lam = lamF(xe) * s
    kon, koff = lam * p, lam * (1 - p)
    N = nmax + 1; idx = lambda m, k: m * N + k          # m = 0 off, 1 on
    rows, cols, vals = [], [], []
    def add(i, j, v):
        rows.append(i); cols.append(j); vals.append(v)
    for m in (0, 1):
        birth = om * (c["beta"] + c["alpha"] * m) * np.ones(N)
        death = c["gamma"] * nn
        k = np.arange(N - 1)
        add(idx(m, k), idx(m, k + 1), birth[:-1])
        k = np.arange(1, N)
        add(idx(m, k), idx(m, k - 1), death[1:])
    add(idx(0, nn), idx(1, nn), kon); add(idx(1, nn), idx(0, nn), koff)
    r = np.concatenate(rows); cc = np.concatenate(cols); v = np.concatenate(vals)
    Q = sp.coo_matrix((v, (r, cc)), shape=(2 * N, 2 * N)).tocsr()
    Q = Q - sp.diags(np.asarray(Q.sum(axis=1)).ravel())
    pr = fsp.stationary(Q)
    px = pr[:N] + pr[N:]
    mean = np.dot(px, nn); var = np.dot(px, (nn - mean) ** 2)
    return mean / om, var, px[-1]


def c1_F(sysname):
    """Strict first-order coefficient of S relative to F."""
    from first_order_coefficient import delta_p
    from systems2 import MODELS
    c = CIRC[sysname]; pon, lamF = functions(sysname)
    xs = _operating_points()[sysname]
    dS, bS, DS = delta_p(MODELS[sysname + "-S"](), xs)
    h = 1e-3; x = xs + h * np.arange(-3, 4)
    p = pon(x); lam = lamF(x)
    F = c["beta"] + c["alpha"] * p - c["gamma"] * x
    Dp = c["alpha"] ** 2 * p * (1 - p) / lam                # = alpha^2 S0 / 2
    b = c["alpha"] ** 2 * p * (1 - p) * np.gradient(1 / lam, x)
    F1 = np.gradient(F, x)[3]; F2 = np.gradient(np.gradient(F, x), x)[3]
    dF = -(0.5 * F2 * Dp[3] / abs(F1) + b[3]) / F1
    return 100 * (dS - dF) / xs, bS, b[3]


if __name__ == "__main__":
    ex = pd.read_csv(os.path.join(RES, "architecture_exact.csv"))
    rows = []
    for sysname in ("NAR", "PFB"):
        c1, bS, bF = c1_F(sysname)
        conds = [(s, 1.0) for s in fsp.S_GRID] + [(1.0, om) for om in fsp.OM_GRID if om != 1.0]
        for s, om in conds:
            m, v, edge = exact(sysname, s, om)
            S = ex[(ex.system == sysname) & (ex.arch == "S") & (ex.s == s) & (ex.omega == om)].iloc[0]
            rows.append(dict(system=sysname, s=s, omega=om, mean_F=m, var_F=v, cv_F=np.sqrt(v) / (m * om), edge=edge,
                             S_minus_F=S["mean"] / m - 1, cv_S_minus_F=S.cv / (np.sqrt(v) / (m * om)) - 1,
                             c1_F=c1, b_S=bS, b_F=bF))
            r = rows[-1]
            print(sysname, s, om, "S-F %.4f  cv %.4f  c1 %.2f  bS %.2f  bF %.2f  edge %.1e" % (
                r["S_minus_F"], r["cv_S_minus_F"], c1, bS, bF, edge), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "control_F.csv"), index=False)
