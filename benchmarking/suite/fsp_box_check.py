#!/usr/bin/env python3
"""Truncation check for the finite-state projection: stationary mean and variance
with the default copy-number box and with a box enlarged by half, for the
site-resolved promoters of the two autoregulated genes and the cascade.
Writes benchmarking_results/fsp_box_check.csv."""
import os
import numpy as np, pandas as pd
import fsp
from systems2 import MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
_orig = fsp.molecular_box


def enlarged(M, om, f=1.5):
    box = _orig(M, om)
    return {j: (lo, int(np.ceil(hi * f))) for j, (lo, hi) in box.items()}


if __name__ == "__main__":
    rows = []
    for name, conds in (("NAR-S", [(0.03, 1.0), (1.0, 1.0), (30.0, 1.0), (1.0, 30.0)]),
                        ("PFB-S", [(0.03, 1.0), (1.0, 1.0), (30.0, 1.0), (1.0, 30.0)]),
                        ("Cascade-S", [(0.03, 1.0), (1.0, 3.0)])):
        M = MODELS[name]()
        for s, om in conds:
            out = {}
            for label, boxf in (("default", _orig), ("enlarged", enlarged)):
                fsp.molecular_box = boxf
                for avg in (False, True):
                    Q, X, box = fsp.build(M, om, s, avg)
                    p = fsp.stationary(Q)
                    m, v, edge = fsp.moments(M, p, X, box)
                    out[(label, avg)] = (m, v, edge, Q.shape[0])
            fsp.molecular_box = _orig
            for avg in (False, True):
                (m0, v0, e0, n0), (m1, v1, e1, n1) = out[("default", avg)], out[("enlarged", avg)]
                rows.append(dict(model=name, s=s, omega=om, description="sQSSA" if avg else "full",
                                 n_default=n0, n_enlarged=n1, edge_default=e0,
                                 mean_rel_change=m1 / m0 - 1, var_rel_change=v1 / v0 - 1))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "fsp_box_check.csv"), index=False)
