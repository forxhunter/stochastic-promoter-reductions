#!/usr/bin/env python3
"""Stationary distribution of the hybrid (PDMP) for circuits with one promoter
and one molecular species (the two autoregulated genes), by an upwind
finite-volume discretisation of its forward equation.

With promoter states m, flows F_m(x) and promoter generator s Q(x), the
upwind scheme on cells of width h is itself a continuous-time Markov chain on
(m, cell): from cell i the chain moves to i+1 at rate max(F_m(x_{i+1/2}), 0)/h
and to i-1 at rate max(-F_m(x_{i-1/2}), 0)/h, and switches promoter state at the
rates of s Q(x_i). Its stationary vector converges to the PDMP stationary
distribution as h -> 0 (first-order numerical diffusion |F| h / 2); results are
reported for two grid sizes and extrapolated (Richardson, first order). The PDMP does not depend on
Omega. Writes benchmarking_results/pdmp_stationary.csv with the mean and variance
of the regulator concentration.
"""
import os
import numpy as np, pandas as pd
import scipy.sparse as sp
from systems2 import MODELS
import fsp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def null_vector(Q):
    """Stationary vector of a sparse conservative generator: fix one state with
    positive mass, drop its balance equation, solve the sparse remainder."""
    import scipy.sparse.linalg as spla
    A0 = Q.T.tocsc()
    n = A0.shape[0]
    dg = np.abs(A0.diagonal()); dg[dg == 0] = 1.0
    A = (A0 @ sp.diags(1.0 / dg)).tocsc()            # solve for q = diag * p, better conditioned

    def solve(k):
        keep = np.r_[0:k, k + 1:n]
        x = spla.spsolve(A[keep][:, keep], -np.asarray(A[keep][:, [k]].todense()).ravel())
        p = np.maximum(np.insert(x, k, 1.0), 0.0)
        return p / p.sum()
    q = solve(n // 2)
    q = solve(int(np.argmax(q)))                     # re-anchor on the most probable state
    p = q / dg; p /= p.sum()
    res = np.abs(A0 @ p).sum() / np.abs(A0.diagonal() * p).sum()
    if res > 1e-8:
        raise RuntimeError("stationary solve residual %.2e" % res)
    return p


def rate(rx, x, s, prom_on):
    """Propensity in concentration units at regulator concentration x (array),
    with the promoter indicator of rx's reactant set to prom_on."""
    a = np.full_like(x, rx["k"] * (s if rx["w"] == 0 else 1.0))
    for name, e in rx["reac"].items():
        if name == "P" or name == "A":
            a = a * x ** e
        else:
            a = a * prom_on
    return a


def stationary(M, s, n_cells=8000, xmax=170.0, scale=1.0, joint=False):
    """Stationary distribution of the upwind chain by linear level reduction:
    levels are cells, each with nb promoter states; levels above i are censored
    out one at a time, with diagonals formed as negative row sums of positive
    rates (Grassmann-Taksar-Heyman), which keeps the elimination stable."""
    block = M.blocks[0]
    names = [M.species[j] for j in block]
    nb = len(block)
    edges = np.linspace(0.0, xmax, n_cells + 1)
    xc = 0.5 * (edges[1:] + edges[:-1]); h = edges[1] - edges[0]
    F = np.zeros((nb, n_cells + 1))
    for rx in M.reactions:
        if rx["w"] == 0:
            continue
        d = rx["change"].get("P", 0)
        prom = [n for n in rx["reac"] if n in names]
        for mi, nm in enumerate(names):
            on = 1.0 if (not prom or prom[0] == nm) else 0.0
            F[mi] += d * rate(rx, edges, s, on)
    up = np.maximum(F[:, 1:-1], 0.0).T / h            # (n_cells-1, nb): level i -> i+1, same promoter state
    dn = np.maximum(-F[:, 1:-1], 0.0).T / h           # level i+1 -> i
    W = np.zeros((n_cells, nb, nb))                   # promoter switching within a level
    for rx in M.reactions:
        if rx["w"] != 0:
            continue
        src = [n for n, d in rx["change"].items() if d == -1][0]
        dst = [n for n, d in rx["change"].items() if d == +1][0]
        W[:, names.index(src), names.index(dst)] += rate(rx, xc, s, 1.0) * scale
    down_out = np.zeros((n_cells, nb)); down_out[1:] = dn
    R = [None] * n_cells
    Ret = np.zeros((nb, nb))                          # R_i D_{i+1}: return from above, censored
    N_last = None
    for i in range(n_cells - 1, -1, -1):
        off = W[i] + Ret
        np.fill_diagonal(off, 0.0)
        N = off.copy()
        np.fill_diagonal(N, -(off.sum(axis=1) + down_out[i]))
        if i == 0 or down_out[i].sum() == 0.0:
            # no flow leaves this level downwards: it is the bottom of the support
            N_last, bottom = N, i
            break
        # R_{i-1} = U_{i-1} (-N_i)^{-1}, U diagonal
        R[i - 1] = np.diag(up[i - 1]) @ np.linalg.inv(-N)
        Ret = R[i - 1] @ np.diag(dn[i - 1])
    # level 0: null vector of the censored generator
    A = N_last.T.copy(); A[-1, :] = 1.0
    b = np.zeros(nb); b[-1] = 1.0
    p0 = np.linalg.solve(A, b)
    P = np.zeros((n_cells, nb)); P[bottom] = np.maximum(p0, 0)
    for i in range(bottom + 1, n_cells):
        P[i] = P[i - 1] @ R[i - 1]
    P /= P.sum()
    px = P.sum(axis=1)
    mean = np.dot(px, xc); var = np.dot(px, (xc - mean) ** 2) + h ** 2 / 12
    if joint:
        return mean, var, xc, px, P, names, edges
    return mean, var, xc, px


def main():
    from systems2 import X_OP, _operating_points
    cm = pd.read_csv(os.path.join(RES, "control_matching.csv"))
    rows = []
    for sysname in ("NAR", "PFB"):
        lam = cm[cm.system == sysname].set_index("rule").lam
        for arch in ("H", "S", "E", "M", "D"):
            X_OP["S"] = _operating_points()[sysname]
            M = MODELS["%s-%s" % (sysname, "M" if arch == "D" else arch)]()
            scale = lam["D"] / lam["M"] if arch == "D" else 1.0
            for s in fsp.S_GRID:
                r = dict(system=sysname, arch=arch, model="%s-%s" % (sysname, arch), s=s)
                for nc in (32000, 128000):
                    m, v, _, _ = stationary(M, s, n_cells=nc, scale=scale)
                    r["mean_%d" % nc] = m; r["var_%d" % nc] = v
                # first-order scheme: Richardson extrapolation from h and h/4
                r["mean"] = r["mean_128000"] + (r["mean_128000"] - r["mean_32000"]) / 3
                r["var"] = r["var_128000"] + (r["var_128000"] - r["var_32000"]) / 3
                rows.append(r); print(r, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "pdmp_stationary.csv"), index=False)


if __name__ == "__main__":
    main()
