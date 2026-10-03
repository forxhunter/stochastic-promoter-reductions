#!/usr/bin/env python3
"""Run the benchmark grid: for each system, an eta sweep (promoter rates scaled
by s) at two system sizes and an Omega sweep at s = 1, with every method.
Each (system, method, s, Omega, ensemble) is cached as one npz file.

SSA is run as two independent ensembles, A (reference) and B (null baseline).

Usage: python3 run_grid.py --workers 22 [--systems Goodwin,Toggle] [--n 64]
"""
import argparse, os, time, itertools
import concurrent.futures as fut
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "benchmarking_results", "suite")

S_GRID = {"Repressilator": [0.1, 0.3, 1, 3, 10, 30, 100],
          "Toggle":        [0.3, 1, 3, 10, 30, 100, 300],
          "Goodwin":       [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.1, 1],
          "p53":           [0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
OMEGA_GRID = {"Repressilator": [0.1, 0.3, 1, 3, 10],
              "Toggle":        [0.3, 1, 3, 10, 30],
              "Goodwin":       [1, 3, 10, 30, 100, 300, 1000],
              "p53":           [5, 15, 50, 150, 500]}
OMEGA_LARGE = {"Repressilator": 10, "Toggle": 30, "Goodwin": 300, "p53": 500}


def conditions(system):
    from systems import BASE_OMEGA
    om0 = BASE_OMEGA[system]
    cond = set()
    for s in S_GRID[system]:
        cond.add((s, om0)); cond.add((s, OMEGA_LARGE[system]))
    for om in OMEGA_GRID[system]:
        cond.add((1, om))
    return sorted(cond)


def tag(system, method, s, om, ens):
    return "%s_%s_s%g_om%g_%s" % (system, method, s, om, ens)


def one(job):
    system, method, s, om, ens, rep, hfac = job
    from systems import SYSTEMS
    from engine import simulate
    M = SYSTEMS[system]()
    import zlib
    seed = (zlib.crc32(tag(system, method, s, om, ens).encode()) % 1000000) * 1000 + rep
    seed = seed % (2 ** 31 - 1)
    t0 = time.perf_counter()
    out, n = simulate(M, method, om, s, seed, h=M.h * hfac)
    return job, out.astype(np.float32), time.perf_counter() - t0, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--systems", default="Goodwin,Toggle,p53,Repressilator")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    groups = {}
    for system in args.systems.split(","):
        for (s, om) in conditions(system):
            for method, ens in (("ssa", "A"), ("ssa", "B"), ("pdmp", "A"), ("sqssa", "A")):
                groups[(system, method, s, om, ens, 1.0)] = args.n
            groups[(system, "ode", s, om, "A", 1.0)] = 1
        # PDMP step-size convergence at the base condition
        from systems import BASE_OMEGA
        groups[(system, "pdmp", 1, BASE_OMEGA[system], "h4", 0.25)] = args.n

    jobs, pending = [], {}
    for key, n in groups.items():
        system, method, s, om, ens, hfac = key
        path = os.path.join(OUT, tag(system, method, s, om, ens) + ".npz")
        if os.path.exists(path):
            continue
        pending[key] = [None] * n
        for rep in range(n):
            jobs.append((system, method, s, om, ens, rep, hfac))
    # longest groups first: SSA at large Omega
    jobs.sort(key=lambda j: (j[1] != "ssa", -j[3] * max(j[2], 1)))
    print("%d groups pending, %d trajectories" % (len(pending), len(jobs)), flush=True)

    times = {k: [] for k in pending}
    done = 0; t_start = time.time()
    with fut.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for job, out, el, n in ex.map(one, jobs, chunksize=1):
            key = job[:5] + (job[6],)
            pending[key][job[5]] = out
            times[key].append(el)
            done += 1
            if all(x is not None for x in pending[key]):
                system, method, s, om, ens, hfac = key
                np.savez_compressed(os.path.join(OUT, tag(system, method, s, om, ens) + ".npz"),
                                    traj=np.stack(pending[key]), wall=np.array(times[key]))
                del pending[key]
                print("[%6.0fs] %5d/%d  saved %s  (%.2fs/traj)"
                      % (time.time() - t_start, done, len(jobs), tag(system, method, s, om, ens),
                         np.mean(times[key])), flush=True)


if __name__ == "__main__":
    main()
