#!/usr/bin/env python3
"""Exact Wasserstein-1 distances between the stationary distributions of the
observed species under the full model and under each reduction, for the two
autoregulated genes, divided by the SSA standard deviation. SSA and sQSSA
distributions come from the finite-state projection, the hybrid distribution
from the discretised forward equation (in copy-number units y = Omega x). For
distributions on the line W1 is the integral of |F1 - F2|. Writes
benchmarking_results/exact_w1.csv."""
import os
import numpy as np, pandas as pd
from systems2 import MODELS
import fsp
from pdmp_stationary import stationary

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def marginal(M, om, s, averaged):
    Q, X, box = fsp.build(M, om, s, averaged)
    p = fsp.stationary(Q)
    j = M.observe[0]
    k = X[:, j]
    n = int(k.max()) + 1
    return np.bincount(k, weights=p, minlength=n)


if __name__ == "__main__":
    rows = []
    for name in [m for m in MODELS if m.split("-")[0] in ("NAR", "PFB")]:
        M = MODELS[name]()
        conds = [(s, 1.0) for s in fsp.S_GRID] + [(1.0, om) for om in fsp.OM_GRID if om != 1.0]
        pd_cache = {}
        for s, om in conds:
            pa = marginal(M, om, s, False)
            pq = marginal(M, om, s, True)
            k = np.arange(len(pa))
            sd = np.sqrt(np.dot(pa, k ** 2) - np.dot(pa, k) ** 2)
            n = max(len(pa), len(pq))
            Fa = np.cumsum(np.pad(pa, (0, n - len(pa)))); Fq = np.cumsum(np.pad(pq, (0, n - len(pq))))
            w1q = np.abs(Fa - Fq).sum()                        # unit spacing between integers
            if s not in pd_cache:
                _, _, xc, px, P, names, edges = stationary(M, s, n_cells=128000, joint=True)
                pd_cache[s] = (px, edges)
            px, edges = pd_cache[s]
            # hybrid CDF at copy number y = om * x, evaluated on a fine grid
            ymax = max(n, om * edges[-1])
            y = np.linspace(0.0, ymax, 400001)
            Fh = np.interp(y / om, edges, np.concatenate([[0.0], np.cumsum(px)]))
            Fs = np.concatenate([np.cumsum(pa), [1.0]])[np.minimum(np.floor(y).astype(int), len(pa))]
            w1h = np.trapezoid(np.abs(Fs - Fh), y)
            rows.append(dict(model=name, s=s, omega=om, sd=sd, w1_pdmp=w1h / sd, w1_sqssa=w1q / sd))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "exact_w1.csv"), index=False)
