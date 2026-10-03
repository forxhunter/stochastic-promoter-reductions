#!/usr/bin/env python3
"""Closed-form stationary density of the hybrid (PDMP) for one species driven by a
two-state promoter, used to validate the finite-volume solver and to state the
boundary-mode criterion.

With an off state of flow F0(x) = -gamma (x - a) and an on state of flow
F1(x) = gamma (b - x) on the support (a, b), and switching rates s k_on(x) (off to
on) and s k_off(x), zero total flux F0 p0 + F1 p1 = 0 gives
    q = F0 p0,  dq/dx = -q [s k_on / F0 + s k_off / F1],
so p(x) = |q| (1/|F0| + 1/|F1|) with |q| = exp(-int (s k_on/F0 + s k_off/F1) dx)
(Kepler and Elston 2001; Zeiser et al. 2010). Near x = a the density behaves as
(x - a)^(s k_on(a)/gamma - 1), so the hybrid has a mode at the boundary if and only
if s k_on(a) < gamma, and likewise at x = b with k_off(b). The integrals are done
on a tanh-sinh grid that resolves the integrable boundary singularities.

Writes benchmarking_results/pdmp_closed_form.csv (closed form against the
finite-volume results for every two-state architecture of the two autoregulated
genes) and the boundary-mode criterion for every model.
"""
import os
import numpy as np, pandas as pd
from pdmp_stationary import rate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def density(kon, koff, a, b, gamma, s, T=2.75, n=40001):
    """Mean and variance of the closed-form density. The tanh-sinh grid reaches
    within about 1e-10 (b - a) of each boundary; the remaining boundary layers,
    where the density is a power law (x - a)^(e - 1) with e = s k_on(a)/gamma (and
    likewise at b), are integrated analytically. For e close to 0 they carry most
    of the probability."""
    t = np.linspace(-T, T, n)
    u = np.pi * np.sinh(t)
    sig = 1.0 / (1.0 + np.exp(-u)); oms = 1.0 / (1.0 + np.exp(u))   # sigma and 1 - sigma, stable
    L = b - a
    da, db = L * sig, L * oms                           # distances to the two boundaries
    x = a + da
    dxdt = L * sig * oms * np.pi * np.cosh(t)
    F0, F1 = -gamma * da, gamma * db
    g = s * kon(x) / F0 + s * koff(x) / F1               # d ln|q| / dx = -g
    gi = -g * dxdt
    lnq = np.concatenate([[0.0], np.cumsum(0.5 * (gi[1:] + gi[:-1]) * np.diff(t))])
    ref = lnq.max()
    px = np.exp(lnq - ref) * (1 / np.abs(F0) + 1 / np.abs(F1))      # density in x, unnormalised
    w = px * dxdt
    core = np.trapezoid(w, t); core_m = np.trapezoid(x * w, t); core_m2 = np.trapezoid(x * x * w, t)
    ea = s * kon(np.array([a]))[0] / gamma; eb = s * koff(np.array([b]))[0] / gamma
    tail_a = px[0] * da[0] / ea                          # int_0^da C d^(ea-1) = p(da) da / ea
    tail_b = px[-1] * db[-1] / eb
    Z = core + tail_a + tail_b
    m1 = (core_m + tail_a * a + tail_b * b) / Z
    m2 = (core_m2 + tail_a * a * a + tail_b * b * b) / Z
    return m1, m2 - m1 * m1


def two_state_rates(M):
    """k_on(x), k_off(x) at s = 1 and the support of a two-state model."""
    block = M.blocks[0]; names = [M.species[j] for j in block]
    on = [k for rx in M.reactions if rx["w"] == 1 and rx["change"].get("P", 0) > 0
          for k in rx["reac"] if k in names][0]
    off = [nm for nm in names if nm != on][0]
    up = [rx for rx in M.reactions if rx["w"] == 0 and rx["change"].get(on, 0) == 1]
    dn = [rx for rx in M.reactions if rx["w"] == 0 and rx["change"].get(off, 0) == 1]
    kon = lambda x: sum(rate(rx, x, 1.0, 1.0) for rx in up)
    koff = lambda x: sum(rate(rx, x, 1.0, 1.0) for rx in dn)
    basal = sum(rx["k"] for rx in M.reactions if rx["w"] == 1 and not rx["reac"] and rx["change"].get("P", 0) > 0)
    alpha = sum(rx["k"] for rx in M.reactions if rx["w"] == 1 and on in rx["reac"] and rx["change"].get("P", 0) > 0)
    gamma = sum(rx["k"] for rx in M.reactions if rx["w"] == 1 and rx["reac"] == {"P": 1} and rx["change"].get("P", 0) < 0)
    return kon, koff, basal / gamma, (basal + alpha) / gamma, gamma


if __name__ == "__main__":
    from systems2 import MODELS, X_OP, _operating_points
    fv = pd.read_csv(os.path.join(RES, "pdmp_stationary.csv")).set_index(["model", "s"])
    cm = pd.read_csv(os.path.join(RES, "control_matching.csv"))
    rows = []
    for sysname in ("NAR", "PFB"):
        lam = cm[cm.system == sysname].set_index("rule").lam
        for arch in ("H", "E", "M", "D"):
            X_OP["S"] = _operating_points()[sysname]
            M = MODELS["%s-%s" % (sysname, "M" if arch == "D" else arch)]()
            kon, koff, a, b, gamma = two_state_rates(M)
            sc = lam["D"] / lam["M"] if arch == "D" else 1.0
            k1 = (lambda f, c: (lambda x: c * f(x)))(kon, sc); k0 = (lambda f, c: (lambda x: c * f(x)))(koff, sc)
            for s in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
                m, v = density(k1, k0, a, b, gamma, s)
                key = ("%s-%s" % (sysname, arch), s)
                r = dict(model=key[0], s=s, mean_closed=m, var_closed=v,
                         mean_fv=fv.loc[key, "mean"], var_fv=fv.loc[key, "var"],
                         boundary_mode_low=bool(s * k1(np.array([a + 1e-12]))[0] < gamma),
                         boundary_mode_high=bool(s * k0(np.array([b - 1e-12]))[0] < gamma))
                r["mean_rel_diff"] = r["mean_fv"] / m - 1; r["var_rel_diff"] = r["var_fv"] / v - 1
                rows.append(r)
                print({k: (round(val, 5) if isinstance(val, float) else val) for k, val in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "pdmp_closed_form.csv"), index=False)
