#!/usr/bin/env python3
"""Model-computable check for the LNA partition in the two autoregulated genes:
the probability mass that the first-order QSS diffusion density places in a
second mode away from the deterministic fixed point, against the exact
deviations of the variance fractions from the LNA prediction. Writes
benchmarking_results/qss_criterion.csv."""
import os
import numpy as np, pandas as pd
from systems2 import MODELS
from qss_diffusion import density_model

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")

if __name__ == "__main__":
    x = pd.read_csv(os.path.join(RES, "exact_reference.csv"))
    rows = []
    for r in x.itertuples():
        q = density_model(MODELS[r.model](), r.s, r.omega)
        rows.append(dict(model=r.model, s=r.s, omega=r.omega, qss_low_mass=q["low_mass"],
                         qss_mean=q["mean"], qss_var=q["var"]))
    y = x.merge(pd.DataFrame(rows), on=["model", "s", "omega"])
    y["qss_var_err"] = y.qss_var / y.ssa_var - 1
    y["qss_mean_err"] = y.qss_mean / y.ssa_mean - 1
    y.to_csv(os.path.join(RES, "qss_criterion.csv"), index=False)
    y["dev"] = np.maximum(y.lna_dev_hybrid.abs(), y.lna_dev_sqssa.abs())
    for thr in (0.005, 0.01, 0.02, 0.05):
        flag = y.qss_low_mass > thr
        print("threshold %.3f: flagged %d; max dev unflagged %.3f; min dev flagged %.3f; |LNA var err| max unflagged %.2f"
              % (thr, flag.sum(), y.dev[~flag].max(), y.dev[flag].min(), y.lna_var_err[~flag].abs().max()))
    print(y[["model", "s", "omega", "qss_low_mass", "dev", "lna_var_err"]].sort_values("qss_low_mass").tail(40).round(3).to_string())
