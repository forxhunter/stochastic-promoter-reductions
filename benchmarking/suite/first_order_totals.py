#!/usr/bin/env python3
"""Total first-order shift of the mean from the fixed point for each promoter of the
two autoregulated genes, against the exact shift. From the local expansion,
delta = delta_p / s + delta_m with delta_p the promoter part at s = 1 and
delta_m = -F'' Var_m / (2 F') the molecular part, Var_m = D_m / |F'| and
D_m = (birth + death) / (2 Omega) at the fixed point. Writes
benchmarking_results/first_order_totals.csv."""
import os
import numpy as np, pandas as pd
from systems2 import MODELS, X_OP, _operating_points
from first_order_coefficient import local_terms, delta_p
from control_matching import lambda_D
from qss_diffusion import CIRC

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")

if __name__ == "__main__":
    ex = pd.read_csv(os.path.join(RES, "architecture_exact.csv"))
    cm = pd.read_csv(os.path.join(RES, "control_matching.csv"))
    rows = []
    for sysname in ("NAR", "PFB"):
        c = CIRC[sysname]
        xs = _operating_points()[sysname]; X_OP["S"] = xs
        lam = cm[cm.system == sysname].set_index("rule").lam
        for arch in ("S", "M", "D", "E"):
            M = MODELS["%s-%s" % (sysname, "M" if arch == "D" else arch)]()
            sc = lambda_D(sysname, 100.0, 1.0) / lam["M"] if arch == "D" else 1.0
            dp, _, _ = delta_p(M, xs, sc)
            h = 1e-3; x = xs + h * np.arange(-3, 4)
            Fb, Dp, b = local_terms(M, x, sc)
            F1 = np.gradient(Fb, x)[3]; F2 = np.gradient(np.gradient(Fb, x), x)[3]
            abar = (Fb[3] + c["gamma"] * xs - c["beta"]) / c["alpha"]
            for om in (1.0,):
                Dm = (c["beta"] + c["alpha"] * abar + c["gamma"] * xs) / (2 * om)
                dm = -(0.5 * F2 * Dm / abs(F1)) / F1
                for s in (1.0, 3.0, 10.0, 30.0):
                    e = ex[(ex.system == sysname) & (ex.arch == arch) & (ex.s == s) & (ex.omega == om)]
                    rows.append(dict(system=sysname, arch=arch, s=s, omega=om,
                                     predicted=100 * (dp / s + dm) / xs, promoter_part=100 * dp / s / xs,
                                     molecular_part=100 * dm / xs, exact=100 * (e["mean"].iloc[0] / xs - 1)))
    d = pd.DataFrame(rows); d.to_csv(os.path.join(RES, "first_order_totals.csv"), index=False)
    print(d.round(2).to_string())
