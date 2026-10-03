#!/usr/bin/env python3
"""Variance kept by each description for Mdm2, the product of the regulated
promoter of the p53-Mdm2 loop (second observed species), with 95% trajectory
bootstrap intervals. Writes benchmarking_results/p53_mdm2.csv."""
import os, glob, re
import numpy as np, pandas as pd
from systems2 import MODELS
from analyze2 import load, DIR, RES

rng = np.random.default_rng(1)
rows = []
for f in sorted(glob.glob(os.path.join(DIR, "p53-*_ssa_*_A.npz"))):
    model, s, om = re.match(r"(p53-[HSEM])_ssa_s([\d.e+-]+)_om([\d.e+-]+)_A", os.path.basename(f)).groups()
    s, om = float(s), float(om)
    M = MODELS[model](); burn = int(M.t_burn / M.record_dt)
    TA, _ = load(model, "ssa", s, om, "A")
    YA = TA[:, burn:, 1]
    for method, ens in (("ssa", "B"), ("pdmp", "A"), ("sqssa", "A")):
        TM, _ = load(model, method, s, om, ens)
        if TM is None:
            continue
        YM = TM[:, burn:, 1]
        b = []
        for _ in range(200):
            i = rng.integers(0, len(YA), len(YA)); j = rng.integers(0, len(YM), len(YM))
            b.append(YM[j].var() / YA[i].var())
        rows.append(dict(model=model, arch=model[-1], s=s, omega=om,
                         method={"B": "ssa_null"}.get(ens, method),
                         var_ratio=YM.var() / YA.var(), lo=np.percentile(b, 2.5), hi=np.percentile(b, 97.5),
                         mean_err=YM.mean() / YA.mean() - 1))
    print("done", model, s, om, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(RES, "p53_mdm2.csv"), index=False)
