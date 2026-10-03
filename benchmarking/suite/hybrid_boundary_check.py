#!/usr/bin/env python3
"""Boundary-mode criterion for the hybrid, as a model-only bimodality check.
For one species with flows F_m(x) = beta + alpha r_m - gamma x, the hybrid lives on
[a, b] with a = min_m (beta + alpha r_m)/gamma and b = max_m (...). Near a boundary
the density behaves as (x - x_b)^(s k_exit / gamma - 1), where k_exit is the rate at
which the promoter leaves the set of states whose flow vanishes at x_b (the
smallest eigenvalue of minus the generator restricted to that set, at x_b, s = 1).
The hybrid density diverges at the boundary, so that its probability is not
concentrated near the fixed point, if and only if s k_exit < gamma. The check
flags a condition when this holds at either boundary. It is computed from the
model alone and does not depend on Omega. Writes
benchmarking_results/hybrid_boundary_check.csv and the performance table."""
import os
import numpy as np, pandas as pd
from pdmp_stationary import rate

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def second_mode_mass(M, s, n_cells=32000):
    """Probability of the exact hybrid density on the far side of the antimode that
    separates its second mode (boundary or interior) from the largest mode."""
    from pdmp_stationary import stationary
    _, _, xc, px = stationary(M, s, n_cells=n_cells)
    dens = px
    peaks = [i for i in range(len(dens)) if (i == 0 or dens[i] >= dens[i - 1]) and (i == len(dens) - 1 or dens[i] >= dens[i + 1]) and dens[i] > 0]
    if len(peaks) < 2:
        return 0.0
    main = max(peaks, key=lambda i: dens[i])
    others = [i for i in peaks if i != main]
    best = 0.0
    for o in others:
        lo, hi = sorted((o, main))
        anti = lo + int(np.argmin(dens[lo:hi + 1]))
        m = dens[:anti].sum() if o < main else dens[anti + 1:].sum()
        best = max(best, m)
    return best


def exit_rates(M):
    block = M.blocks[0]; names = [M.species[j] for j in block]; nb = len(block)
    alpha = np.zeros(nb); beta = 0.0; gamma = 0.0
    for rx in M.reactions:
        if rx["w"] != 1:
            continue
        d = rx["change"].get("P", 0)
        prom = [k for k in rx["reac"] if k in names]
        if d > 0 and prom:
            alpha[names.index(prom[0])] += rx["k"]
        elif d > 0:
            beta += rx["k"]
        elif rx["reac"] == {"P": 1}:
            gamma += rx["k"]
    lev = (beta + alpha) / gamma
    out = {}
    for side, xb in (("low", lev.min()), ("high", lev.max())):
        S = [i for i in range(nb) if np.isclose(lev[i], xb)]
        Q = np.zeros((nb, nb))
        for rx in M.reactions:
            if rx["w"] != 0:
                continue
            src = [k for k, d in rx["change"].items() if d == -1][0]
            dst = [k for k, d in rx["change"].items() if d == +1][0]
            v = rate(rx, np.array([max(xb, 1e-9)]), 1.0, 1.0)[0]
            Q[names.index(src), names.index(dst)] += v
        Q -= np.diag(Q.sum(axis=1))
        sub = -Q[np.ix_(S, S)]
        out[side] = (xb, np.min(np.real(np.linalg.eigvals(sub))))
    return out, gamma


if __name__ == "__main__":
    from systems2 import MODELS
    from check_circuits import CIRCUITS
    dev = pd.read_csv(os.path.join(RES, "qss_criterion.csv"))
    dev["dev"] = np.maximum(dev.lna_dev_hybrid.abs(), dev.lna_dev_sqssa.abs())
    test = pd.read_csv(os.path.join(RES, "check_circuits.csv"))
    test["lna_var_err"] = test.lna_var / test.ssa_var - 1
    rows = []
    for label, d, get in (("development (NAR, PFB)", dev, lambda m: MODELS[m]()),
                          ("test (SA, PFBC)", test, lambda m: CIRCUITS[m]())):
        cache = {}
        for i, r in d.iterrows():
            if r.model not in cache:
                cache[r.model] = exit_rates(get(r.model))
            ex, gamma = cache[r.model]
            d.loc[i, "k_exit_low"] = ex["low"][1]; d.loc[i, "k_exit_high"] = ex["high"][1]
            d.loc[i, "flag_boundary"] = bool(r.s * ex["low"][1] < gamma or r.s * ex["high"][1] < gamma)
            key = (r.model, r.s)
            if key not in cache:
                cache[key] = second_mode_mass(get(r.model), r.s)
            d.loc[i, "hybrid_second_mass"] = cache[key]
        d["flag_qss"] = d.qss_low_mass > 0.01
        d["flag_hybrid"] = d.hybrid_second_mass > 0.01
        for crit, fail in (("fraction > 0.1", d.dev > 0.1),
                           ("fraction > 0.1 or LNA variance error > 25%", (d.dev > 0.1) | (d.lna_var_err.abs() > 0.25))):
            for check in ("flag_qss", "flag_boundary", "flag_hybrid"):
                f = d[check].astype(bool)
                rows.append(dict(set=label, criterion=crit, check={"flag_qss": "first-order density", "flag_boundary": "hybrid boundary divergence", "flag_hybrid": "exact hybrid density"}[check],
                                 failures=int(fail.sum()), detected=int((fail & f).sum()), missed=int((fail & ~f).sum()),
                                 false_alarms=int((~fail & f).sum()), correct_passes=int((~fail & ~f).sum())))
        d.to_csv(os.path.join(RES, "hybrid_boundary_check_%s.csv" % ("dev" if label.startswith("dev") else "test")), index=False)
    r = pd.DataFrame(rows); r.to_csv(os.path.join(RES, "hybrid_boundary_check.csv"), index=False)
    print(r.to_string())
