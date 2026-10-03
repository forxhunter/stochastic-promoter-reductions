#!/usr/bin/env python3
"""Toggle switch: switching rates from basin-balanced ensembles.

For each architecture (H concerted, S site-resolved, E two-state control with the
exact site-resolved response) and method, 64 trajectories start in the U-high
basin and 64 in the V-high basin and run for T = 10000 after a burn-in of 50.
Basins are assigned with hysteresis (U > 2V enters U-high, U < V/2 leaves it).
The escape rate from each basin is the number of exits divided by the total time
spent in it (maximum-likelihood estimate for exponential residence), with a 95%
interval from 1000 bootstrap resamples of trajectories. The hybrid is run with
step h, h/4 and h/16.

Usage: python3 toggle_switching.py --workers 22
"""
import argparse, os, time
import concurrent.futures as fut
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "benchmarking_results", "toggle_switching")
T_RUN = 10050.0
CONDS = [(1, 1), (0.3, 1), (3, 1), (1, 0.3), (1, 3), (1, 10), (1, 30)]
ARCH = ["H", "S", "E", "P", "M", "Q"]
METHODS = [("ssa", 1.0), ("pdmp", 1.0), ("pdmp", 0.25), ("pdmp", 1 / 16), ("pdmpx", 1.0), ("sqssa", 1.0)]


def model(arch, start):
    from systems2 import MODELS
    M = MODELS["Toggle-" + arch]()
    M.t_total = T_RUN; M.t_burn = 50.0; M.record_dt = 0.5
    if start == "V":
        x0 = {"U": 0.0, "V": 30.0, "Sv_on": 1}
        stU = [s for s in M.species if s.startswith("Su")]
        x0[stU[-1] if arch in ("S", "Q") else "Su_off"] = 1
        M.x0 = x0
    return M


def one(job):
    arch, meth, hfac, s, om, start, rep = job
    from engine import simulate
    M = model(arch, start)
    seed = (hash((arch, meth, hfac, s, om, start)) & 0xFFFFF) * 1000 + rep
    import zlib
    seed = (zlib.crc32(("%s_%s_%g_%g_%g_%s" % (arch, meth, hfac, s, om, start)).encode()) % 1000000) * 1000 + rep
    t0 = time.perf_counter()
    out, _ = simulate(M, meth, om, s, seed % (2 ** 31 - 1), h=M.h * hfac)
    return job, out.astype(np.float32), time.perf_counter() - t0


def basin_stats(U, V, burn):
    """Per trajectory: time in U-high, exits from U-high, time in V-high,
    exits from V-high (record interval units)."""
    res = []
    for u, v in zip(U[:, burn:], V[:, burn:]):
        state = 1 if u[0] > v[0] else 0
        tu = tv = eu = ev = 0
        for a, b in zip(u, v):
            if state == 1 and a < 0.5 * b:
                state = 0; eu += 1
            elif state == 0 and a > 2 * b:
                state = 1; ev += 1
            if state:
                tu += 1
            else:
                tv += 1
        res.append((tu, eu, tv, ev))
    return np.array(res, float)


def rates(st, dt, nb=1000, seed=0):
    rng = np.random.default_rng(seed)
    def est(x):
        ku = x[:, 1].sum() / max(x[:, 0].sum() * dt, 1e-12)
        kv = x[:, 3].sum() / max(x[:, 2].sum() * dt, 1e-12)
        return ku, kv
    ku, kv = est(st)
    b = np.array([est(st[rng.integers(0, len(st), len(st))]) for _ in range(nb)])
    return dict(k_U=ku, k_U_lo=np.percentile(b[:, 0], 2.5), k_U_hi=np.percentile(b[:, 0], 97.5),
                k_V=kv, k_V_lo=np.percentile(b[:, 1], 2.5), k_V_hi=np.percentile(b[:, 1], 97.5),
                exits_U=int(st[:, 1].sum()), exits_V=int(st[:, 3].sum()),
                frac_U=st[:, 0].sum() / (st[:, 0].sum() + st[:, 2].sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--n", type=int, default=64); args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    groups, jobs = {}, []
    for arch in ARCH:
        for (s, om) in CONDS:
            for meth, hfac in METHODS:
                if meth == "sqssa" and arch == "Q":
                    continue
                if meth == "pdmp" and hfac < 1 and not (om == 1 and (s == 1 or (s == 3 and arch == "S"))):
                    continue
                for start in ("U", "V"):
                    key = (arch, meth, hfac, s, om, start)
                    path = os.path.join(OUT, "%s_%s_h%g_s%g_om%g_%s.npz" % key)
                    if os.path.exists(path):
                        continue
                    groups[key] = [None] * args.n
                    jobs += [key + (r,) for r in range(args.n)]
    jobs.sort(key=lambda j: (-(j[4] if j[1] != "pdmp" else 0), j[2]))
    print("%d groups, %d trajectories" % (len(groups), len(jobs)), flush=True)
    with fut.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for job, out, el in ex.map(one, jobs, chunksize=1):
            key = job[:6]
            groups[key][job[6]] = out
            if all(x is not None for x in groups[key]):
                np.savez_compressed(os.path.join(OUT, "%s_%s_h%g_s%g_om%g_%s.npz" % key), traj=np.stack(groups[key]))
                print("saved", key, "%.1fs" % el, flush=True)
                del groups[key]
    # analysis
    rows = []
    for arch in ARCH:
        for (s, om) in CONDS:
            for meth, hfac in METHODS:
                f = [os.path.join(OUT, "%s_%s_h%g_s%g_om%g_%s.npz" % (arch, meth, hfac, s, om, st)) for st in "UV"]
                if not all(os.path.exists(x) for x in f):
                    continue
                T = np.concatenate([np.load(x)["traj"].astype(float) for x in f])
                st = basin_stats(T[:, :, 0], T[:, :, 1], int(50 / 0.5))
                r = dict(arch=arch, method=meth, hfac=hfac, s=s, omega=om); r.update(rates(st, 0.5))
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "..", "..", "benchmarking_results", "toggle_switching.csv"), index=False)
    print(df.to_string(float_format=lambda v: "%.3g" % v))


if __name__ == "__main__":
    main()
