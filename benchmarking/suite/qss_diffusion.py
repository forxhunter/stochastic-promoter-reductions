#!/usr/bin/env python3
"""One-dimensional diffusion approximation of an autoregulated gene with a fast
multi-state promoter, to first order in 1/s (Chapman-Enskog expansion of the
PDMP forward equation, as in the quasi-steady-state diffusion approximation of
Newby and Bressloff), with the molecular noise of the CME added as a
Kramers-Moyal diffusion.

With promoter states m, flows F_m(x) = beta + alpha a_m - gamma x, promoter
generator s Q(x) with stationary distribution pi(x) and fundamental matrix
Z(x) / s, the density P(x) of the regulator concentration obeys

    dP/dt = -d/dx [ (Fbar + b) P - d/dx (D P) ],
    Fbar = sum_m pi_m F_m,
    phi  = Z (F - Fbar) / s,
    D_p  = sum_m pi_m (F_m - Fbar) phi_m        (= alpha^2 S0 / 2),
    b    = sum_m pi_m F_m d(phi_m)/dx            (noise-induced drift),
    D    = D_p + (beta + alpha abar + gamma x) / (2 Omega).

The stationary density is P ∝ exp(∫ (Fbar + b) / D dx) / D. Its mean satisfies
E[Fbar + b] = 0, so two promoters with the same response abar(x) and the same
S0 at the operating point differ in mean, to first order in 1/s, only through
b. Writes benchmarking_results/qss_diffusion.csv with the mean and variance of
this approximation next to the exact (FSP) values.
"""
import os
import numpy as np, pandas as pd
from math import comb
from systems2 import OMEGA_COOP, match_site, site_Q, _operating_points
from control_matching import PARS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
CIRC = {"NAR": dict(alpha=100.0, beta=0.0, gamma=1.0),
        "PFB": dict(alpha=80.0, beta=4.0, gamma=1.0)}


def fundamental(Q):
    n = Q.shape[0]
    A = Q.T.copy(); A[-1, :] = 1; rhs = np.zeros(n); rhs[-1] = 1
    pi = np.linalg.solve(A, rhs)
    Pi = np.outer(np.ones(n), pi)
    return pi, np.linalg.inv(Pi - Q) - Pi


_SITE = {}


def generator(arch, x, sysname, lam=None):
    P = PARS[sysname]; n = P["n"]
    if sysname not in _SITE:
        _SITE[sysname] = match_site(n, P["K"], P["kb"], P["repress"], P["xmax"])
    KS, kbs = _SITE[sysname]
    if arch == "S":
        Q = site_Q(x, n, KS, kbs, OMEGA_COOP)
        act = np.array([1.0] * n + [0.0] if P["repress"] else [0.0] * n + [1.0])
        return Q, act
    coef = [comb(n, k) * OMEGA_COOP ** (k * (k - 1) / 2) * KS ** (-k) for k in range(n + 1)]
    actw = np.array([1.0] * n + [0.0] if P["repress"] else [0.0] * n + [1.0])
    w = np.array([c * x ** k for k, c in enumerate(coef)])
    on, off = lam * np.dot(actw, w), lam * np.dot(1 - actw, w)
    return np.array([[-on, on], [off, -off]]), np.array([0.0, 1.0])   # state 0 off, 1 on


def density(arch, sysname, s, om, lam=None, xgrid=None):
    c = CIRC[sysname]
    x = xgrid if xgrid is not None else np.linspace(1e-3, 160.0, 6001)
    nx = len(x)
    Fb = np.zeros(nx); Dp = np.zeros(nx); abar = np.zeros(nx)
    Fm_all, pi_all, phi_all = [], [], []
    for i, xv in enumerate(x):
        Q, act = generator(arch, xv, sysname, lam)
        pi, Z = fundamental(Q)
        F = c["beta"] + c["alpha"] * act - c["gamma"] * xv
        fb = np.dot(pi, F)
        phi = Z @ (F - fb) / s
        Fb[i] = fb; Dp[i] = np.dot(pi * (F - fb), phi); abar[i] = np.dot(pi, act)
        Fm_all.append(F); pi_all.append(pi); phi_all.append(phi)
    Fm = np.array(Fm_all); PI = np.array(pi_all); PHI = np.array(phi_all)
    dphi = np.gradient(PHI, x, axis=0)
    b = np.sum(PI * Fm * dphi, axis=1)
    Dm = (c["beta"] + c["alpha"] * abar + c["gamma"] * x) / (2 * om)
    D = Dp + Dm
    g = (Fb + b) / D
    G = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))])
    logp = G - np.log(D)
    p = np.exp(logp - logp.max()); p /= np.trapezoid(p, x)
    mean = np.trapezoid(x * p, x); var = np.trapezoid((x - mean) ** 2 * p, x)
    # the same without the noise-induced drift
    g0 = Fb / D
    G0 = np.concatenate([[0.0], np.cumsum(0.5 * (g0[1:] + g0[:-1]) * np.diff(x))])
    p0 = np.exp(G0 - np.log(D) - (G0 - np.log(D)).max()); p0 /= np.trapezoid(p0, x)
    mean0 = np.trapezoid(x * p0, x)
    # modes of the density
    modes = x[1:-1][(p[1:-1] > p[:-2]) & (p[1:-1] > p[2:])]
    i0 = np.argmin(np.abs(Fb[200:] )) + 200
    return dict(mean=mean, var=var * om ** 2, mean_no_b=mean0, n_modes=len(modes),
                modes=";".join("%.2f" % m for m in modes), b_at_mean=np.interp(mean, x, b),
                x=x, p=p, b=b, Fb=Fb, D=D)


def main():
    from control_matching import exact_moments
    from systems2 import MODELS, X_OP
    cm = pd.read_csv(os.path.join(RES, "control_matching.csv"))
    rows = []
    for sysname in ("NAR", "PFB"):
        lams = cm[cm.system == sysname].set_index("rule").lam
        X_OP["S"] = _operating_points()[sysname]
        MS, MM = MODELS[sysname + "-S"](), MODELS[sysname + "-M"]()
        for s in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
            for om in (1.0, 30.0):
                from control_matching import lambda_D
                for arch, lam in (("S", None), ("M", lams["M"]), ("D", lambda_D(sysname, s, om)), ("E", lams["E"])):
                    r = density("S" if arch == "S" else "2", sysname, s, om, lam)
                    if arch == "S":
                        ex = exact_moments(MS, s, om)
                    else:
                        ex = exact_moments(MM, s, om, scale=lam / lams["M"])
                    rows.append(dict(system=sysname, arch=arch, s=s, omega=om,
                                     qss_mean=r["mean"], qss_mean_no_b=r["mean_no_b"], qss_var=r["var"],
                                     n_modes=r["n_modes"], modes=r["modes"],
                                     exact_mean=ex[0], exact_var=ex[1]))
                    print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "qss_diffusion.csv"), index=False)


if __name__ == "__main__":
    main()


def density_model(M, s, om, scale=1.0, xmax=170.0, n=6001):
    """The same approximation built from the model's own reactions (any
    architecture of a circuit with one promoter and one molecular species P)."""
    from pdmp_stationary import rate
    block = M.blocks[0]; names = [M.species[j] for j in block]; nb = len(block)
    x = np.linspace(1e-3, xmax, n)
    Q = np.zeros((n, nb, nb)); F = np.zeros((n, nb)); Fm0 = np.zeros((n, nb)); Dmol = np.zeros((n, nb))
    for rx in M.reactions:
        if rx["w"] == 0:
            src = [k for k, d in rx["change"].items() if d == -1][0]
            dst = [k for k, d in rx["change"].items() if d == +1][0]
            a = rate(rx, x, s, 1.0) * scale
            Q[:, names.index(src), names.index(dst)] += a
            Q[:, names.index(src), names.index(src)] -= a
        else:
            d = rx["change"].get("P", 0)
            prom = [k for k in rx["reac"] if k in names]
            for mi, nm in enumerate(names):
                on = 1.0 if (not prom or prom[0] == nm) else 0.0
                a = rate(rx, x, s, on)
                F[:, mi] += d * a; Dmol[:, mi] += d * d * a
    pi = np.zeros((n, nb)); phi = np.zeros((n, nb))
    for i in range(n):
        p_, Z = fundamental(Q[i])
        pi[i] = p_
        fb = np.dot(p_, F[i])
        phi[i] = Z @ (F[i] - fb)            # Q already carries the factor s
    Fb = np.sum(pi * F, axis=1)
    Dp = np.sum(pi * (F - Fb[:, None]) * phi, axis=1)
    b = np.sum(pi * F * np.gradient(phi, x, axis=0), axis=1)
    D = Dp + np.sum(pi * Dmol, axis=1) / (2 * om)
    g = (Fb + b) / D
    G = np.concatenate([[0.0], np.cumsum(0.5 * (g[1:] + g[:-1]) * np.diff(x))])
    lp = G - np.log(D); p = np.exp(lp - lp.max()); p /= np.trapezoid(p, x)
    mean = np.trapezoid(x * p, x); var = np.trapezoid((x - mean) ** 2 * p, x)
    mins = x[1:-1][(p[1:-1] < p[:-2]) & (p[1:-1] < p[2:])]
    xa = mins[0] if len(mins) else 0.0
    low = np.trapezoid(p[x < xa], x[x < xa]) if xa > 0 else 0.0
    return dict(mean=mean, var=var * om ** 2, low_mass=low, antimode=xa, x=x, p=p)
