#!/usr/bin/env python3
"""Out-of-sample test of the model-computable bimodality check (first-order QSS
diffusion density) on two circuits not used to develop it:

  SA   a non-cooperative self-activating gene (order 1), in which slow promoter
       switching produces bimodality without bistability (To and Maheshri 2010);
  PFBC a cooperative self-activating gene (order 2) with a larger basal rate, so
       that its rate balance stays far from zero below the fixed point: a
       strongly nonlinear, unimodal negative control.

For every condition of the s and Omega grids the script computes, from the
model alone, the LNA promoter share and variance and the probability mass of the
first-order density in a second mode, and, as references, the exact stationary
moments of the full model and the sQSSA (finite-state projection) and of the
hybrid (discretised forward equation). The failure criterion and the threshold
of the check are fixed in advance: the LNA partition is taken to fail when a
variance fraction deviates from its prediction by more than 0.1, and the check
flags a condition when the second mode holds more than 1% of the probability.
Writes benchmarking_results/check_circuits.csv.
"""
import os
import numpy as np, pandas as pd
from engine import Model
from systems2 import promoter, R, finish
import fsp
from lna import lna, fixed_point
from pdmp_stationary import stationary as pdmp_stationary
from qss_diffusion import density_model

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
TOL, THR = 0.1, 0.01
S_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
OM_GRID = [0.3, 3.0, 10.0, 30.0]


def selfact(name, arch, basal, alpha, K, n, xmax=150.0):
    st, rx, _ = promoter(arch, "S", "P", n, K, 1.0, False, alpha, "P", xmax=xmax)
    rx += [R(basal, 1, {}, {"P": +1}), R(1.0, 1, {"P": 1}, {"P": -1})]
    M = Model("%s-%s" % (name, arch), st + ["P"], [st], rx, {st[0]: 1, "P": 30.0}, observe=["P"],
              record_dt=0.02, t_burn=20.0, t_total=420.0)
    return finish(M, 0.001, 1.0, "dimensionless", ["P"], [(n, K)])


CIRCUITS = {
    "SA-H": lambda: selfact("SA", "H", 2.0, 60.0, 30.0, 1),
    "PFBC-H": lambda: selfact("PFBC", "H", 20.0, 80.0, 30.0, 2),
    "PFBC-S": lambda: selfact("PFBC", "S", 20.0, 80.0, 30.0, 2),
}


def evaluate(name, s, om):
    M = CIRCUITS[name]()
    out = dict(model=name, s=s, omega=om)
    for label, avg in (("ssa", False), ("sqssa", True)):
        Q, X, box = fsp.build(M, om, s, avg)
        p = fsp.stationary(Q)
        m, v, edge = fsp.moments(M, p, X, box)
        out[label + "_mean"] = m / om; out[label + "_var"] = v; out[label + "_edge"] = edge
    m32, v32, _, _ = pdmp_stationary(M, s, n_cells=32000, xmax=200.0)
    m128, v128, _, _ = pdmp_stationary(M, s, n_cells=128000, xmax=200.0)
    out["pdmp_mean"] = m128 + (m128 - m32) / 3
    out["pdmp_var"] = (v128 + (v128 - v32) / 3) * om ** 2
    r = lna(M, om, s)
    out["lna_share"] = r["share_prom"]; out["lna_var"] = r["var"]
    out["x_star"] = fixed_point(M, om, s)[M.observe[0]] / om
    q = density_model(M, s, om, xmax=200.0)
    out["qss_low_mass"] = q["low_mass"]; out["qss_mean"] = q["mean"]
    out["hyb_frac"] = out["pdmp_var"] / out["ssa_var"]
    out["sq_frac"] = out["sqssa_var"] / out["ssa_var"]
    out["dev"] = max(abs(out["hyb_frac"] - out["lna_share"]), abs(out["sq_frac"] - (1 - out["lna_share"])))
    return out


if __name__ == "__main__":
    rows = []
    for name in CIRCUITS:
        for s, om in [(s, 1.0) for s in S_GRID] + [(1.0, om) for om in OM_GRID]:
            rows.append(evaluate(name, s, om))
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()
                   if k in ("model", "s", "omega", "ssa_mean", "x_star", "lna_share", "hyb_frac", "sq_frac",
                            "dev", "qss_low_mass")}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "check_circuits.csv"), index=False)
