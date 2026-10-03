#!/usr/bin/env python3
"""Exact (FSP) stationary mean and CV of the observed species for every
architecture of the two autoregulated genes, including the control D matched
in S0 averaged over the stationary distribution of S at each condition (s,
Omega), on the full s and Omega grid. Writes benchmarking_results/architecture_exact.csv."""
import os
import numpy as np, pandas as pd
from systems2 import MODELS, X_OP, _operating_points
from control_matching import exact_moments, lambda_D
import fsp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")

if __name__ == "__main__":
    cm = pd.read_csv(os.path.join(RES, "control_matching.csv"))
    rows = []
    for sysname in ("NAR", "PFB"):
        lam = cm[cm.system == sysname].set_index("rule").lam
        X_OP["S"] = _operating_points()[sysname]
        conds = [(s, 1.0) for s in fsp.S_GRID] + [(1.0, om) for om in fsp.OM_GRID if om != 1.0]
        for arch in ("H", "S", "E", "M", "D"):
            M = MODELS["%s-%s" % (sysname, "M" if arch == "D" else arch)]()
            for s, om in conds:
                lamD = lambda_D(sysname, s, om) if arch == "D" else np.nan
                scale = lamD / lam["M"] if arch == "D" else 1.0
                m, v, _, _ = exact_moments(M, s, om, scale=scale)
                r = dict(system=sysname, arch=arch, s=s, omega=om, mean=m, var=v,
                         cv=np.sqrt(v) / (m * om), lam=lamD)
                if arch == "D":
                    # elasticity of the mean with respect to lambda, from +-10%
                    mp = exact_moments(M, s, om, scale=scale * 1.1)[0]
                    mm = exact_moments(M, s, om, scale=scale / 1.1)[0]
                    r["dlnmean_dlnlam"] = (np.log(mp) - np.log(mm)) / (2 * np.log(1.1))
                rows.append(r)
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "architecture_exact.csv"), index=False)
