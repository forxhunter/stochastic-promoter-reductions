#!/usr/bin/env python3
"""Range of the timescale ratio eta(x) over the regulator levels visited by each
stationary circuit at its base parameters (reviewer R2, minor comment 2).

eta(x) = (1 / lambda_2(Q(x))) / tau_reg is evaluated at the 5th, 25th, 50th, 75th
and 95th percentiles of the stationary regulator distribution of the SSA reference
ensemble (stored trajectories, benchmarking_results/suite2), and at the
time-averaged level (eta_op). The worst case over 3000 sampled states and zero
regulator (eta_wc) is that of the main text.
Writes benchmarking_results/revision/eta_range.csv.
"""
import os
import numpy as np, pandas as pd
from systems2 import MODELS, BASE_OMEGA
from engine import eta
from analyze2 import load

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "benchmarking_results", "revision")
Q = [5, 25, 50, 75, 95]


def main():
    rows = []
    for model in ("p53-S", "p53-H", "NAR-S", "NAR-H", "Cascade-S", "Cascade-H", "PFB-S", "PFB-H"):
        system = model.split("-")[0]; om = BASE_OMEGA[system]
        M = MODELS[model]()
        TA, _ = load(model, "ssa", 1.0, om, "A")
        burn = int(M.t_burn / M.record_dt)
        reg = M.reg_species[0]
        j = M.species.index(reg); j_obs = M.observe.index(j)
        x = TA[:, burn:, j_obs].ravel()
        qs = np.percentile(x, Q)
        X = np.zeros((len(Q) + 2, len(M.species)))
        X[:len(Q), j] = qs; X[len(Q), j] = x.mean(); X[len(Q) + 1, j] = 0.0
        e = eta(M, 1.0, X, M.tau_prot)
        r = dict(model=model, omega=om, regulator=reg, x_mean=x.mean(), eta_op=e[len(Q)], eta_x0=e[-1])
        for q, v, ev in zip(Q, qs, e[:len(Q)]):
            r["x_p%d" % q] = v; r["eta_p%d" % q] = ev
        r["eta_ratio_p95_p5"] = max(e[0], e[len(Q) - 1]) / min(e[0], e[len(Q) - 1])
        rows.append(r)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "eta_range.csv"), index=False)


if __name__ == "__main__":
    main()
