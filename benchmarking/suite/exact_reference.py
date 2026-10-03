#!/usr/bin/env python3
"""Exact stationary references for the two autoregulated genes: mean, CV and
variance of the full model and the sQSSA (finite-state projection,
fsp_stationary.csv), of the hybrid (converged discretisation of the PDMP
forward equation, pdmp_stationary.csv) and the averaged ODE fixed point, with
the errors of each reduction and the LNA prediction. Writes
benchmarking_results/exact_reference.csv."""
import os
import numpy as np, pandas as pd
from systems2 import MODELS
from lna import fixed_point

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")

if __name__ == "__main__":
    f = pd.read_csv(os.path.join(RES, "fsp_stationary.csv"))
    f = f[f.system.isin(["NAR", "PFB"])]
    p = pd.read_csv(os.path.join(RES, "pdmp_stationary.csv"))[["model", "s", "mean", "var"]].rename(
        columns={"mean": "pdmp_mean", "var": "pdmp_varx"})
    d = pd.read_csv(os.path.join(RES, "suite2_metrics.csv"))
    l = d[d.method == "pdmp"][["model", "s", "omega", "lna_share", "lna_var", "s1_share", "eta_op", "eta_wc"]]
    x = f.merge(p, on=["model", "s"]).merge(l, on=["model", "s", "omega"])
    ode = []
    for r in x.itertuples():
        M = MODELS[r.model]()
        X = fixed_point(M, r.omega, r.s)
        ode.append(X[M.observe[0]] / r.omega)
    x["ode_mean"] = ode
    om = x.omega
    x["ssa_cv"] = np.sqrt(x.ssa_var) / (x.ssa_mean * om)
    x["sqssa_cv"] = np.sqrt(x.sqssa_var) / (x.sqssa_mean * om)
    x["pdmp_var"] = x.pdmp_varx * om ** 2
    x["pdmp_cv"] = np.sqrt(x.pdmp_varx) / x.pdmp_mean
    for m in ("pdmp", "sqssa", "ode"):
        x[m + "_mean_err"] = x[m + "_mean"] / x.ssa_mean - 1
    for m in ("pdmp", "sqssa"):
        x[m + "_cv_err"] = x[m + "_cv"] / x.ssa_cv - 1
        x[m + "_var_ratio"] = x[m + "_var"] / x.ssa_var
    x["lna_dev_hybrid"] = x.pdmp_var_ratio - x.lna_share
    x["lna_dev_sqssa"] = x.sqssa_var_ratio - (1 - x.lna_share)
    x["s1_dev_hybrid"] = x.pdmp_var_ratio - x.s1_share
    x["s1_dev_sqssa"] = x.sqssa_var_ratio - (1 - x.s1_share)
    x["lna_var_err"] = x.lna_var / x.ssa_var - 1
    x.to_csv(os.path.join(RES, "exact_reference.csv"), index=False)
    rm = lambda v: np.sqrt(np.mean(v ** 2))
    for sy in ("NAR", "PFB"):
        g = x[x.system == sy]
        print(sy, "n", len(g),
              "LNA dev hybrid rms %.3f max %.3f" % (rm(g.lna_dev_hybrid), g.lna_dev_hybrid.abs().max()),
              "sQSSA rms %.3f max %.3f" % (rm(g.lna_dev_sqssa), g.lna_dev_sqssa.abs().max()),
              "| S1 hybrid rms %.3f sQSSA %.3f" % (rm(g.s1_dev_hybrid), rm(g.s1_dev_sqssa)))
