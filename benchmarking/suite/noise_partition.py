#!/usr/bin/env python3
"""Noise partition across the grid. For each condition, the share of the SSA
variance of the primary species that each reduction retains:
    share_prom = Var(PDMP)/Var(SSA)   (PDMP keeps promoter noise only)
    share_mol  = Var(sQSSA)/Var(SSA)  (sQSSA keeps molecular noise only)
and the combination R = N eta/(1 + eta), N the mean copy number, which in the
linear two-state case is proportional to the ratio of promoter to molecular
noise (Supplementary Section S1). Fits share_prom = R/(R + c) per system.
Writes benchmarking_results/noise_partition.csv."""
import os
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from analyze import load, RES
from systems import SYSTEMS

d = pd.read_csv(os.path.join(RES, "suite_metrics.csv"))
rows = []
for (sysn, s, om), g in d[d.method == "pdmp"].groupby(["system", "s", "omega"]):
    M = SYSTEMS[sysn](); burn = int(M.t_burn / M.record_dt)
    v = {m: load(sysn, m, s, om, "A")[0][:, burn:, 0].var() for m in ("ssa", "pdmp", "sqssa")}
    N = load(sysn, "ssa", s, om, "A")[0][:, burn:, 0].mean() * om
    for conv in ("eta_op", "eta_worst"):
        pass
    rows.append(dict(system=sysn, s=s, omega=om, eta_op=g.eta_op.iloc[0], eta_worst=g.eta_worst.iloc[0], N=N,
                     share_prom=v["pdmp"] / v["ssa"], share_mol=v["sqssa"] / v["ssa"]))
r = pd.DataFrame(rows)
for conv in ("eta_op", "eta_worst"):
    r["R_" + conv] = r[conv] / (1 + r[conv]) * r.N
r.to_csv(os.path.join(RES, "noise_partition.csv"), index=False)
f = lambda R, c: R / (R + c)
for sysn, g in r.groupby("system"):
    for conv in ("R_eta_op", "R_eta_worst"):
        try:
            (c,), _ = curve_fit(f, g[conv], g.share_prom, p0=[10.0], bounds=(1e-9, 1e9))
            res = g.share_prom - f(g[conv], c)
            print("%-13s %-12s c = %8.3g   rms resid %.3f   max |resid| %.3f   share sum range %.2f-%.2f"
                  % (sysn, conv, c, np.sqrt(np.mean(res ** 2)), np.abs(res).max(),
                     (g.share_prom + g.share_mol).min(), (g.share_prom + g.share_mol).max()))
        except Exception as e:
            print(sysn, conv, "fit failed", e)
