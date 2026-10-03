#!/usr/bin/env python3
"""Second-design benchmark grid (systems2: H and S promoter architectures).
Run the benchmark grid: for each system, an eta sweep (promoter rates scaled
by s) at two system sizes and an Omega sweep at s = 1, with every method.
Each (system, method, s, Omega, ensemble) is cached as one npz file.

SSA is run as two independent ensembles, A (reference) and B (null baseline).

Usage: python3 run_grid.py --workers 22 [--systems Goodwin,Toggle] [--n 64]
"""
import argparse, os, time, itertools
import concurrent.futures as fut
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "benchmarking_results", "suite2")

S_GRID = {"Repressilator": [0.1, 0.3, 1, 3, 10, 30, 100],
          "Goodwin":       [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1],
          "Toggle":        [0.03, 0.1, 0.3, 1, 3, 10, 30],
          "p53":           [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
          "NAR":           [0.03, 0.1, 0.3, 1, 3, 10, 30],
          "Cascade":       [0.03, 0.1, 0.3, 1, 3, 10, 30],
          "PFB":           [0.03, 0.1, 0.3, 1, 3, 10, 30]}
OMEGA_GRID = {"Repressilator": [0.1, 0.3, 1, 3, 10],
              "Goodwin":       [10, 30, 100, 300, 1000],
              "Toggle":        [0.3, 1, 3, 10, 30],
              "p53":           [5, 15, 50, 150, 500],
              "NAR":           [0.3, 1, 3, 10, 30],
              "Cascade":       [0.3, 1, 3, 10, 30],
              "PFB":           [0.3, 1, 3, 10, 30]}


def conditions(model):
    from systems2 import BASE_OMEGA
    system = model.split("-")[0]
    om0 = BASE_OMEGA[system]
    cond = {(sv, om0) for sv in S_GRID[system]} | {(1, om) for om in OMEGA_GRID[system]}
    return sorted(cond)


def tag(system, method, s, om, ens):
    return "%s_%s_s%g_om%g_%s" % (system, method, s, om, ens)


_FSP_CACHE = {}


def stationary_start(system, s, om, rep, ens, method):
    """Initial state drawn from the exact stationary distribution of the
    description being simulated, at (s, Omega): the full model for SSA and ODE
    (finite-state projection), the sQSSA master equation for the sQSSA, and the
    discretised forward equation for the hybrid. Used for the positively
    autoregulated gene, whose slow switching between its two modes is not
    equilibrated within the burn-in from a fixed initial state."""
    import zlib, fsp
    from systems2 import MODELS
    kind = {"sqssa": "sqssa", "pdmp": "pdmp", "pdmpx": "pdmp"}.get(method, "ssa")
    key = (system, s, om, kind)
    if key not in _FSP_CACHE:
        M = MODELS[system]()
        if kind == "pdmp":
            from pdmp_stationary import stationary
            _, _, xc, _, P, names, edges = stationary(M, s, n_cells=32000, joint=True)
            _FSP_CACHE[key] = (M, (P.ravel(), names, edges))
        else:
            Q, X, _ = fsp.build(M, om, s, kind == "sqssa")
            _FSP_CACHE[key] = (M, (X, fsp.stationary(Q)))
    M, data = _FSP_CACHE[key]
    rng = np.random.default_rng(zlib.crc32(("x0_%s_%s_%g_%g_%s" % (system, method, s, om, ens)).encode()) + rep)
    if kind == "pdmp":
        p, names, edges = data
        nb = len(names)
        k = rng.choice(len(p), p=p / p.sum())
        cell, m = divmod(k, nb)
        x = rng.uniform(edges[cell], edges[cell + 1])
        x0 = {sp: 0.0 for sp in M.species}
        x0[names[m]] = 1.0; x0["P"] = x
        return x0
    X, p = data
    i = rng.choice(len(p), p=p)
    x0 = {sp: (float(X[i, j]) if M.isD[j] else X[i, j] / om) for j, sp in enumerate(M.species)}
    if kind == "sqssa":                      # promoter state is not simulated; keep a valid indicator
        blk = M.blocks[0]
        for j in blk:
            x0[M.species[j]] = 0.0
        x0[M.species[blk[0]]] = 1.0
    return x0


def one(job):
    system, method, s, om, ens, rep, hfac = job
    from systems2 import MODELS
    from engine import simulate
    M = MODELS[system]()
    if system.startswith("PFB-"):
        M.x0 = stationary_start(system, s, om, rep, ens, method)
    import zlib
    seed = (zlib.crc32(tag(system, method, s, om, ens).encode()) % 1000000) * 1000 + rep
    seed = seed % (2 ** 31 - 1)
    t0 = time.perf_counter()
    out, n = simulate(M, method, om, s, seed, h=M.h * hfac)
    return job, out.astype(np.float32), time.perf_counter() - t0, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--systems", default="Goodwin-H,NAR-H,NAR-S,Cascade-H,Cascade-S,Toggle-H,Toggle-S,p53-S,p53-H,Repressilator-H,Repressilator-S")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--exact-only", dest="exact_only", action="store_true",
                    help="run only the exact-flow PDMP (pdmpx) for every condition")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    groups = {}
    for system in args.systems.split(","):
        for (s, om) in conditions(system):
            meths = (("ssa", "A"), ("ssa", "B"), ("pdmp", "A"), ("sqssa", "A"))
            if args.exact_only:
                meths = (("pdmpx", "A"),)
            for method, ens in meths:
                groups[(system, method, s, om, ens, 1.0)] = args.n
            if not args.exact_only:
                groups[(system, "ode", s, om, "A", 1.0)] = 1
        # PDMP step-size convergence at the base condition
        from systems2 import BASE_OMEGA
        if not args.exact_only:
            groups[(system, "pdmp", 1, BASE_OMEGA[system.split("-")[0]], "h4", 0.25)] = args.n
            smax = max(S_GRID[system.split("-")[0]])
            groups[(system, "pdmp", smax, BASE_OMEGA[system.split("-")[0]], "h4", 0.25)] = args.n

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
