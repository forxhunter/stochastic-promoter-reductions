#!/usr/bin/env python3
"""Effect of the label-sampling interval on simulated toggle escape rates: the same
SSA trajectories are recorded every 0.05 time units and analysed with labels
checked every 0.05 and every 0.5 units, for comparison with the exact
continuous-time rates. Omega = 1, s = 1. Writes
benchmarking_results/toggle_dt_check.csv."""
import os, zlib
import concurrent.futures as fut
import numpy as np, pandas as pd
from toggle_switching import model, basin_stats, rates

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
DT = 0.05


def one(job):
    arch, start, rep = job
    from engine import simulate
    M = model(arch, start); M.record_dt = DT
    seed = (zlib.crc32(("dt_%s_%s" % (arch, start)).encode()) % 1000000) * 1000 + rep
    out, _ = simulate(M, "ssa", 1.0, 1.0, seed % (2 ** 31 - 1), h=M.h)
    return job, out[:, :2].astype(np.float32)


if __name__ == "__main__":
    archs = ["H", "P", "S", "E", "M"]
    jobs = [(a, st, r) for a in archs for st in "UV" for r in range(64)]
    res = {}
    with fut.ProcessPoolExecutor(max_workers=16) as ex:
        for job, out in ex.map(one, jobs, chunksize=1):
            res[job] = out
    rows = []
    for a in archs:
        T = np.stack([res[(a, st, r)] for st in "UV" for r in range(64)]).astype(float)
        for dt, step in ((DT, 1), (0.5, 10)):
            X = T[:, ::step, :]
            st = basin_stats(X[:, :, 0], X[:, :, 1], int(50 / dt))
            r = dict(arch=a, dt=dt); r.update(rates(st, dt))
            rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "toggle_dt_check.csv"), index=False)
