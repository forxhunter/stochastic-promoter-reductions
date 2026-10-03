#!/usr/bin/env python3
"""Which model-only quantity orders the variance lost by the sQSSA? For every
condition of the four stationary circuits: the LNA promoter share; the closed form
of Supplementary Eq. S1, a function of the group eps = N (pi_off/pi_on) eta/(1+eta)
that combines the timescale ratio with the copy number; the timescale ratio from
the spectral gap, eta = (1/lambda_2)/tau; and eta_a = tau_a/tau from the integrated
correlation time of the promoter's transcriptional output, tau_a = S0/(2 Var a),
which for a two-state promoter equals 1/lambda_2. Promoter quantities are evaluated
at the deterministic fixed point. Spearman coefficients pooled and within the s
sweep only. Writes benchmarking_results/ranking_criteria.csv."""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from systems2 import MODELS, BASE_OMEGA, STATIONARY
from lna import fixed_point
from qss_diffusion import fundamental

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def promoter_times(M, om, s):
    X = fixed_point(M, om, s)
    block = M.blocks[0]; names = [M.species[j] for j in block]; nb = len(block)
    idx = M.idx; isD = M.isD
    Q = np.zeros((nb, nb))
    for rx in M.reactions:
        if rx["w"] != 0:
            continue
        src = [k for k, d in rx["change"].items() if d == -1 and isD[idx[k]]][0]
        dst = [k for k, d in rx["change"].items() if d == +1 and isD[idx[k]]][0]
        v = rx["k"] * s
        for k, e in rx["reac"].items():
            if not isD[idx[k]]:
                v *= (X[idx[k]] / om) ** e
        Q[names.index(src), names.index(dst)] += v
    Q -= np.diag(Q.sum(axis=1))
    act = np.zeros(nb)
    for rx in M.reactions:
        if rx["w"] == 1:
            prom = [k for k in rx["reac"] if k in names]
            if prom and any(d > 0 for d in rx["change"].values()):
                act[names.index(prom[0])] += rx["k"]
    act /= act.max()
    pi, Z = fundamental(Q)
    a = act - np.dot(pi, act)
    S0 = 2 * np.dot(pi * a, Z @ a); var = np.dot(pi, a * a)
    lam2 = np.sort(np.abs(np.real(np.linalg.eigvals(Q))))[1]
    return (1 / lam2) / M.tau_prot, (S0 / (2 * var)) / M.tau_prot


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(RES, "suite2_metrics.csv"))
    ex = pd.read_csv(os.path.join(RES, "exact_reference.csv")).set_index(["model", "s", "omega"])
    q = d[(d.method == "sqssa") & d.system.isin(STATIONARY)].copy()
    rows = []
    for r in q.itertuples():
        k = (r.model, r.s, r.omega)
        loss = 1 - (ex.loc[k, "sqssa_var_ratio"] if k in ex.index else r.var_ratio)
        eta_fp, eta_a = promoter_times(MODELS[r.model](), r.omega, r.s)
        rows.append(dict(model=r.model, system=r.system, s=r.s, omega=r.omega, loss=loss,
                         lna=r.lna_share, s1=r.s1_share, eta_op=r.eta_op, eta_wc=r.eta_wc,
                         eta_fp=eta_fp, eta_a=eta_a,
                         s_sweep=(r.omega == BASE_OMEGA[r.system])))
    x = pd.DataFrame(rows); x.to_csv(os.path.join(RES, "ranking_inputs.csv"), index=False)
    out = []
    for label, sub in (("pooled, four circuits", x), ("pooled, first three", x[x.system != "PFB"]),
                       ("s sweep only, first three", x[(x.system != "PFB") & x.s_sweep])):
        r = dict(set=label, n=len(sub))
        for c in ("lna", "s1", "eta_op", "eta_wc", "eta_fp", "eta_a"):
            r[c] = spearmanr(sub.loss, sub[c]).correlation
        out.append(r)
    for sysn, sub in x.groupby("system"):
        r = dict(set=sysn, n=len(sub))
        for c in ("lna", "s1", "eta_op", "eta_wc", "eta_fp", "eta_a"):
            r[c] = spearmanr(sub.loss, sub[c]).correlation
        out.append(r)
    o = pd.DataFrame(out); o.to_csv(os.path.join(RES, "ranking_criteria.csv"), index=False)
    print(o.round(3).to_string())
