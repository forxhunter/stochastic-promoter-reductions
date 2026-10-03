#!/usr/bin/env python3
"""Strict first-order coefficient of the mean difference between promoter
architectures in the autoregulated genes. From the local expansion of the
first-order quasi-steady-state diffusion approximation about the fixed point x*
(Supplementary Section S1.3), the promoter contribution to the mean shift is
delta_p = -[F''(x*) Var_p / 2 + b(x*)] / F'(x*) with Var_p = D_p(x*) / |F'(x*)|,
evaluated at s = 1; it scales as 1/s, and the molecular contribution is the same
for promoters with the same response. The relative mean difference between S and
a control C is therefore c1 / s with c1 = 100 (delta_p^S - delta_p^C) / x*.
Writes benchmarking_results/first_order_coefficient.csv with c1 and the exact
s * Delta from architecture_exact.csv."""
import os
import numpy as np, pandas as pd
from systems2 import MODELS, X_OP, _operating_points
from qss_diffusion import fundamental
from pdmp_stationary import rate
from control_matching import lambda_D

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def local_terms(M, x, scale=1.0):
    """Fbar, D_p and b of the first-order approximation at s = 1 on the grid x."""
    block = M.blocks[0]; names = [M.species[j] for j in block]; nb = len(block)
    n = len(x)
    Q = np.zeros((n, nb, nb)); F = np.zeros((n, nb))
    for rx in M.reactions:
        if rx["w"] == 0:
            src = [k for k, d in rx["change"].items() if d == -1][0]
            dst = [k for k, d in rx["change"].items() if d == +1][0]
            a = rate(rx, x, 1.0, 1.0) * scale
            Q[:, names.index(src), names.index(dst)] += a
            Q[:, names.index(src), names.index(src)] -= a
        else:
            d = rx["change"].get("P", 0)
            prom = [k for k in rx["reac"] if k in names]
            for mi, nm in enumerate(names):
                on = 1.0 if (not prom or prom[0] == nm) else 0.0
                F[:, mi] += d * rate(rx, x, 1.0, on)
    pi = np.zeros((n, nb)); phi = np.zeros((n, nb))
    for i in range(n):
        p_, Z = fundamental(Q[i])
        pi[i] = p_; phi[i] = Z @ (F[i] - np.dot(p_, F[i]))
    Fb = np.sum(pi * F, axis=1)
    Dp = np.sum(pi * (F - Fb[:, None]) * phi, axis=1)
    b = np.sum(pi * F * np.gradient(phi, x, axis=0), axis=1)
    return Fb, Dp, b


def delta_p(M, xstar, scale=1.0, h=1e-3):
    x = xstar + h * np.arange(-3, 4)
    Fb, Dp, b = local_terms(M, x, scale)
    F1 = np.gradient(Fb, x)[3]; F2 = np.gradient(np.gradient(Fb, x), x)[3]
    var_p = Dp[3] / abs(F1)
    return -(0.5 * F2 * var_p + b[3]) / F1, b[3], Dp[3]


if __name__ == "__main__":
    ex = pd.read_csv(os.path.join(RES, "architecture_exact.csv"))
    cm = pd.read_csv(os.path.join(RES, "control_matching.csv"))
    rows = []
    for sysname in ("NAR", "PFB"):
        xstar = _operating_points()[sysname]
        lam = cm[cm.system == sysname].set_index("rule").lam
        X_OP["S"] = xstar
        MS, MM, ME = MODELS[sysname + "-S"](), MODELS[sysname + "-M"](), MODELS[sysname + "-E"]()
        dS, bS, DS = delta_p(MS, xstar)
        lamD = lambda_D(sysname, 100.0, 1.0)               # D matched in the fast-switching limit
        for ref, Mc, sc in (("M", MM, 1.0), ("D", MM, lamD / lam["M"]), ("E", ME, 1.0)):
            dC, bC, DC = delta_p(Mc, xstar, sc)
            c1 = 100 * (dS - dC) / xstar
            r = dict(system=sysname, ref=ref, x_star=xstar, c1=c1, b_S=bS, b_ctrl=bC, Dp_S=DS, Dp_ctrl=DC)
            x = ex[(ex.system == sysname) & (ex.omega == 1.0)].pivot_table(index="s", columns="arch", values="mean")
            for s in (1.0, 3.0, 10.0, 30.0):
                r["exact_sDelta_s%g" % s] = s * 100 * (x.loc[s, "S"] / x.loc[s, ref] - 1)
            rows.append(r); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "first_order_coefficient.csv"), index=False)
