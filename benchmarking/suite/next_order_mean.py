#!/usr/bin/env python3
"""Next-order correction to the mean in the moment-linearised promoter
description (lna.emre), against the exact stationary mean, for every condition of
the two autoregulated genes. Writes benchmarking_results/next_order_mean.csv."""
import os
import pandas as pd
from systems2 import MODELS
from lna import emre

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")

if __name__ == "__main__":
    x = pd.read_csv(os.path.join(RES, "exact_reference.csv"))
    rows = []
    for r in x.itertuples():
        e = emre(MODELS[r.model](), r.omega, r.s)
        rows.append(dict(model=r.model, s=r.s, omega=r.omega, x_star=e["x_star"] / r.omega,
                         predicted_shift=e["rel"], exact_shift=r.ssa_mean / (e["x_star"] / r.omega) - 1))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(RES, "next_order_mean.csv"), index=False)
    print(d[d.model.str.endswith("-S")].round(3).to_string())
