#!/usr/bin/env python3
"""Exact escape rates of the toggle switch from its stationary distribution, for the
full model (CME) and for the sQSSA, at small Omega and any promoter speed.

The basin label of the simulations (U-high entered when U > 2V, left when U < V/2)
is added to the state as a bit, so that the chain on (promoter states, U, V, label)
reproduces the hysteretic estimator: the escape rate from the U-high basin is the
stationary rate of 1 -> 0 label flips divided by the stationary probability of
label 1, and likewise for the V-high basin. The state space is truncated to a box
of copy numbers wide enough that the probability at its edges is negligible
(reported); births leaving the box are dropped. The labels change at every
crossing in continuous time, while the simulations check them every 0.5 time
units, so short excursions across a threshold count here and may be missed in
simulation. Writes benchmarking_results/toggle_exact.csv.
"""
import os, sys, itertools, time
import numpy as np, pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from systems2 import MODELS
import fsp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def box_hi(M, name, om, k=7):
    birth = sum(r["k"] * om ** r["w"] for r in M.reactions if r["w"] == 1 and r["change"].get(name, 0) > 0)
    return int(np.ceil(birth + k * np.sqrt(birth) + 12))


def conditional_pi(M, X, om, s, block):
    """Conditional stationary distribution of one promoter block at each molecular
    state, from that block's own reactions with the SSA's propensities."""
    idx = M.idx
    pos = {j: i for i, j in enumerate(block)}; nb = len(block)
    Qs = np.zeros((X.shape[0], nb, nb))
    for rx in M.reactions:
        if rx["w"] != 0:
            continue
        srcj = [idx[n] for n, d in rx["change"].items() if d == -1 and M.isD[idx[n]]][0]
        dstj = [idx[n] for n, d in rx["change"].items() if d == +1 and M.isD[idx[n]]][0]
        if srcj not in pos:
            continue
        Xs = X.copy()
        for j in block:
            Xs[:, j] = 0
        Xs[:, srcj] = 1
        a = fsp.propensity(rx, Xs, om, s, M.isD, idx=idx)
        Qs[:, pos[srcj], pos[dstj]] += a
        Qs[:, pos[srcj], pos[srcj]] -= a
    A = np.transpose(Qs, (0, 2, 1)).copy()
    A[:, -1, :] = 1.0
    b = np.zeros((X.shape[0], nb)); b[:, -1] = 1.0
    p = np.linalg.solve(A, b[..., None])[..., 0]
    return {j: np.maximum(p[:, i], 0.0) for j, i in pos.items()}


def build(M, om, s, averaged):
    idx = M.idx
    iU, iV = idx["U"], idx["V"]
    hu, hv = box_hi(M, "U", om), box_hi(M, "V", om)
    U, V = np.meshgrid(np.arange(hu + 1), np.arange(hv + 1), indexing="ij")
    U = U.ravel(); V = V.ravel(); nm = len(U)
    blocks = M.blocks
    configs = [None] if averaged else list(itertools.product(*blocks))
    nc = len(configs)
    ns = nm * nc
    X = np.zeros((ns, len(M.species)), dtype=np.int64)
    for ci, cfg in enumerate(configs):
        sl = slice(ci * nm, (ci + 1) * nm)
        X[sl, iU] = U; X[sl, iV] = V
        if cfg is not None:
            for j in cfg:
                X[sl, j] = 1
    cfg_index = {cfg: ci for ci, cfg in enumerate(configs)}

    def index(Y):
        ok = (Y[:, iU] >= 0) & (Y[:, iU] <= hu) & (Y[:, iV] >= 0) & (Y[:, iV] <= hv)
        mi = np.clip(Y[:, iU], 0, hu) * (hv + 1) + np.clip(Y[:, iV], 0, hv)
        if averaged:
            return mi, ok
        ci = np.zeros(len(Y), dtype=np.int64)
        for cfg, c in cfg_index.items():
            sel = np.all(Y[:, list(cfg)] == 1, axis=1)
            ci[sel] = c
        return ci * nm + mi, ok

    pi = None
    if averaged:
        pi = {}
        for b in blocks:
            pi.update(conditional_pi(M, X, om, s, b))
    rows, cols, vals = [], [], []
    src, _ = index(X)
    for rx in M.reactions:
        if averaged and rx["w"] == 0:
            continue
        a = fsp.propensity(rx, X, om, s, M.isD, pi=pi, idx=idx)
        if rx.get("noscale"):
            a = a / s                                    # degradation of bound regulator: not a promoter rate
        Y = X.copy()
        for name, d in rx["change"].items():
            Y[:, idx[name]] += d
        dst, ok = index(Y)
        ok &= a > 0
        rows.append(src[ok]); cols.append(dst[ok]); vals.append(a[ok])
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    return rows, cols, vals, X[:, iU], X[:, iV], ns, (hu, hv)


def stationary(G, anchor):
    """Stationary vector of a sparse generator: fix the probability of one state,
    drop its balance equation and solve the rest by sparse LU (no dense row, which
    would destroy sparsity); re-anchor on the most probable state."""
    A = G.T.tocsc(); n = A.shape[0]

    def solve(k):
        keep = np.r_[0:k, k + 1:n]
        x = spla.spsolve(A[keep][:, keep], -np.asarray(A[keep][:, [k]].todense()).ravel(),
                         permc_spec="COLAMD")
        p = np.maximum(np.insert(x, k, 1.0), 0.0)
        return p / p.sum()
    p = solve(anchor)
    p = solve(int(np.argmax(p)))
    res = np.abs(A @ p).sum() / np.abs(A.diagonal() * p).sum()
    return p, res


def escape_rates(M, om, s, averaged, enter=2.0, leave=0.5, dt_sample=None):
    """Escape rates for the hysteretic labelling (enter U-high when U > enter*V, leave
    when U < leave*V). With dt_sample, also the rates of the labelling as the
    simulations apply it, from samples every dt_sample time units: the flux of label-1
    states into the leaving set at the next sample, pi_1 exp(dt Q) 1_leave / (dt P_1),
    which assumes that basin switching is slow compared with mixing within a basin."""
    t0 = time.time()
    rows, cols, vals, U, V, ns, (hu, hv) = build(M, om, s, averaged)
    # label augmentation: state (i, l), index i + l * ns
    lab_src = np.r_[np.zeros(len(rows), int), np.ones(len(rows), int)]
    r2 = np.r_[rows, rows]; c2 = np.r_[cols, cols]; v2 = np.r_[vals, vals]
    Ud, Vd = U[c2], V[c2]
    lab_dst = lab_src.copy()
    lab_dst[(lab_src == 1) & (Ud < leave * Vd)] = 0
    lab_dst[(lab_src == 0) & (Ud > enter * Vd)] = 1
    R = r2 + lab_src * ns; C = c2 + lab_dst * ns
    G = sp.coo_matrix((v2, (R, C)), shape=(2 * ns, 2 * ns)).tocsr()
    G = G - sp.diags(np.asarray(G.sum(axis=1)).ravel())
    # labels that cannot occur (label 1 with U < V/2, label 0 with U > 2V) are transient
    anchor = int(np.argmin(np.abs(U - 300 * om) + 10 * V)) + ns          # near the U-high state, label 1
    p, res = stationary(G, anchor)
    p1 = p[ns:].sum(); p0 = p[:ns].sum()
    flip10 = (lab_src == 1) & (lab_dst == 0); flip01 = (lab_src == 0) & (lab_dst == 1)
    k_U = np.sum(p[R[flip10]] * v2[flip10]) / p1
    k_V = np.sum(p[R[flip01]] * v2[flip01]) / p0
    pm = p[:ns] + p[ns:]
    edge = pm[(U == hu) | (V == hv)].sum()
    out = dict(k_U=k_U, k_V=k_V, frac_U=p1, edge=edge, residual=res, n_states=2 * ns, sec=time.time() - t0)
    if dt_sample:
        Qb = sp.coo_matrix((vals, (rows, cols)), shape=(ns, ns)).tocsr()
        Qb = Qb - sp.diags(np.asarray(Qb.sum(axis=1)).ravel())
        leave_set = (U < leave * V).astype(float); enter_set = (U > enter * V).astype(float)
        w1 = spla.expm_multiply(Qb.T * dt_sample, p[ns:])
        w0 = spla.expm_multiply(Qb.T * dt_sample, p[:ns])
        out["k_U_sampled"] = np.dot(w1, leave_set) / (dt_sample * p1)
        out["k_V_sampled"] = np.dot(w0, enter_set) / (dt_sample * p0)
    return out


def toggle_Qd():
    """Q with bound regulators degraded at the free rate: each bound V molecule is
    lost at rate 1, which moves the promoter down one occupancy level without
    returning V. Separates sequestration from protection against degradation."""
    from systems2 import MODELS
    M = MODELS["Toggle-Q"]()
    for k in (1, 2, 3):
        M.reactions.append(dict(k=float(k), w=0, reac={"Su_b%d" % k: 1},
                                change={"Su_b%d" % k: -1, "Su_b%d" % (k - 1): 1}, noscale=True))
    M.name = "Toggle-Qd"
    return M


if __name__ == "__main__":
    archs = sys.argv[1].split(",") if len(sys.argv) > 1 else ["H", "P", "S", "E", "M", "Q"]
    oms = [float(v) for v in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.3, 1.0]
    s_list = [0.3, 1.0, 3.0, 10.0, 30.0]
    path = os.path.join(RES, "toggle_exact_%s.csv" % "_".join(archs))
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["arch"], r["method"], r["s"], r["omega"]) for r in rows}
    for arch in archs:
        M = toggle_Qd() if arch == "Qd" else MODELS["Toggle-" + arch]()
        for om in oms:
            for s in s_list:
                for method, avg in (("ssa", False), ("sqssa", True)):
                    if avg and arch in ("Q", "Qd"):
                        continue
                    if (arch, method, s, om) in done:
                        continue
                    r = dict(arch=arch, method=method, s=s, omega=om)
                    r.update(escape_rates(M, om, s, avg))
                    rows.append(r)
                    print({k: (("%.3g" % v) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
                    pd.DataFrame(rows).to_csv(path, index=False)
