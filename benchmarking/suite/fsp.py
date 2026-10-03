#!/usr/bin/env python3
"""Exact stationary distributions of the stationary circuits by the finite state
projection (FSP), for the full model (the CME simulated by SSA) and for the
sQSSA (the CME with the promoter replaced by its conditional stationary
distribution at the current copy numbers).

The state space is the product of the promoter configurations and a box of
molecular copy numbers. The box is wide enough that birth reactions leaving it
carry negligible probability; they are dropped, so the truncated generator is
conservative and its null vector is the stationary distribution on the box. The
propensities are those of the engine: falling factorials for integer orders of
H promoters in the full model, plain powers otherwise, and for the sQSSA the
conditional promoter distribution computed from copy numbers with the same
propensities (engine._stationary_counts).

Writes benchmarking_results/fsp_stationary.csv with, for every model and
condition of the grid, the exact mean and variance of the observed species
under the full model and under the sQSSA, and the probability mass at the upper
edge of the box as a truncation check.
"""
import os, sys, time, itertools
import numpy as np, pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from systems2 import MODELS, BASE_OMEGA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
S_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
OM_GRID = [0.3, 1.0, 3.0, 10.0, 30.0]


def molecular_box(M, om):
    """Copy-number range for each molecular species: [lo, hi] from the largest
    possible birth rate and the linear death rate, with a wide Poisson margin."""
    box = {}
    for j, name in enumerate(M.species):
        if M.isD[j]:
            continue
        birth = sum(r["k"] * om ** r["w"] for r in M.reactions
                    if r["change"].get(name, 0) > 0)
        death = sum(r["k"] for r in M.reactions
                    if r["change"].get(name, 0) < 0 and list(r["reac"]) == [name])
        m = birth / death
        hi = int(np.ceil(m + 12 * np.sqrt(m) + 30))
        lo = 0
        # a species with constant birth and linear death is Poisson: window it
        const = all(len(r["reac"]) == 0 for r in M.reactions if r["change"].get(name, 0) > 0)
        if const:
            lo = max(0, int(np.floor(m - 10 * np.sqrt(m) - 10)))
            hi = int(np.ceil(m + 10 * np.sqrt(m) + 10))
        box[j] = (lo, hi)
    return box


def propensity(rx, X, om, s, isD, pi=None, idx=None):
    """Vectorised engine propensity of reaction rx over states X (n_states x n_species).
    If pi is given (sQSSA), promoter indicators are replaced by pi[:, state]."""
    a = np.full(X.shape[0], rx["k"] * om ** rx["w"] * (s if rx["w"] == 0 else 1.0))
    for name, e in rx["reac"].items():
        j = idx[name]
        if pi is not None and j in pi:
            a = a * pi[j] ** e
            continue
        x = X[:, j].astype(float)
        if isD[j]:
            a = a * x ** e                       # promoter indicator
        elif rx.get("ff", True) and e >= 2 and e == int(e):
            v = np.ones_like(x)
            for q in range(int(e)):
                v = v * np.maximum(x - q, 0.0) / om
            a = a * v
        else:
            a = a * (x / om) ** e
    return a


def stationary(Q):
    """Null vector of a conservative generator (rows sum to zero)."""
    n = Q.shape[0]
    A = Q.T.tocsr().tolil()
    A[n - 1, :] = np.ones(n)
    b = np.zeros(n); b[-1] = 1.0
    p = spla.spsolve(A.tocsc(), b)
    p = np.maximum(p, 0.0)
    return p / p.sum()


def build(M, om, s, averaged):
    idx = M.idx
    mol = [j for j in range(len(M.species)) if not M.isD[j]]
    box = molecular_box(M, om)
    block = M.blocks[0]                            # one promoter per stationary circuit
    ranges = [range(box[j][0], box[j][1] + 1) for j in mol]
    mol_states = np.array(list(itertools.product(*ranges)), dtype=np.int64)
    nm = len(mol_states)
    prom = [None] if averaged else list(block)
    ns = nm * len(prom)
    X = np.zeros((ns, len(M.species)), dtype=np.int64)
    for p_i, pj in enumerate(prom):
        X[p_i * nm:(p_i + 1) * nm][:, mol] = mol_states
        if pj is not None:
            X[p_i * nm:(p_i + 1) * nm, pj] = 1
    # index of a state: promoter position * nm + mixed-radix molecular index
    widths = [box[j][1] - box[j][0] + 1 for j in mol]
    strides = np.cumprod([1] + widths[::-1][:-1])[::-1]

    def index(Xs):
        mi = np.zeros(len(Xs), dtype=np.int64)
        ok = np.ones(len(Xs), dtype=bool)
        for c, j in enumerate(mol):
            v = Xs[:, j] - box[j][0]
            ok &= (v >= 0) & (v < widths[c])
            mi += v * strides[c]
        if averaged:
            return mi, ok
        pp = np.zeros(len(Xs), dtype=np.int64)
        for p_i, pj in enumerate(prom):
            pp[Xs[:, pj] == 1] = p_i
        return pp * nm + mi, ok

    pi = None
    if averaged:
        pi = conditional_pi(M, X, om, s, block)
    rows, cols, vals = [], [], []
    src, _ = index(X)
    for rx in M.reactions:
        if averaged and rx["w"] == 0:
            continue
        a = propensity(rx, X, om, s, M.isD, pi=pi, idx=idx)
        Y = X.copy()
        for name, d in rx["change"].items():
            Y[:, idx[name]] += d
        # a promoter reaction only fires from its source state
        dst, ok = index(Y)
        ok &= a > 0
        rows.append(src[ok]); cols.append(dst[ok]); vals.append(a[ok])
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    Q = sp.coo_matrix((vals, (rows, cols)), shape=(ns, ns)).tocsr()
    Q = Q - sp.diags(np.asarray(Q.sum(axis=1)).ravel())
    return Q, X, box


def conditional_pi(M, X, om, s, block):
    """Conditional stationary promoter distribution at each molecular state,
    from the promoter reactions with the SSA's own propensities."""
    idx = M.idx
    nb = len(block)
    pos = {j: i for i, j in enumerate(block)}
    Qs = np.zeros((X.shape[0], nb, nb))
    for rx in M.reactions:
        if rx["w"] != 0:
            continue
        srcj = [idx[n] for n, d in rx["change"].items() if d == -1][0]
        dstj = [idx[n] for n, d in rx["change"].items() if d == +1][0]
        Xs = X.copy()
        for j in block:
            Xs[:, j] = 0
        Xs[:, srcj] = 1
        a = propensity(rx, Xs, om, s, M.isD, idx=idx)
        Qs[:, pos[srcj], pos[dstj]] += a
        Qs[:, pos[srcj], pos[srcj]] -= a
    A = np.transpose(Qs, (0, 2, 1)).copy()
    A[:, -1, :] = 1.0
    b = np.zeros((X.shape[0], nb)); b[:, -1] = 1.0
    p = np.linalg.solve(A, b[..., None])[..., 0]
    return {j: np.maximum(p[:, i], 0.0) for j, i in pos.items()}


def moments(M, p, X, box):
    j = M.observe[0]
    x = X[:, j].astype(float)
    mean = np.dot(p, x); var = np.dot(p, (x - mean) ** 2)
    edge = 0.0
    for jj, (lo, hi) in box.items():
        edge = max(edge, p[X[:, jj] == hi].sum())
        if lo > 0:
            edge = max(edge, p[X[:, jj] == lo].sum())
    return mean, var, edge


def main(models):
    rows = []
    for name in models:
        M = MODELS[name]()
        om0 = BASE_OMEGA[name.split("-")[0]]
        conds = [(s, om0) for s in S_GRID] + [(1.0, om) for om in OM_GRID if om != om0]
        if name.startswith("Cascade"):
            # two molecular species: the sparse LU of the box exceeds memory above Omega = 3
            conds = [c for c in conds if c[1] <= 3.0]
        for s, om in conds:
            r = dict(model=name, system=name.split("-")[0], arch=name[-1], s=s, omega=om)
            for label, avg in (("ssa", False), ("sqssa", True)):
                t0 = time.time()
                Q, X, box = build(M, om, s, avg)
                if Q.shape[0] > 3_000_000:
                    r[label + "_n"] = Q.shape[0]
                    continue
                p = stationary(Q)
                mean, var, edge = moments(M, p, X, box)
                r.update({label + "_mean": mean / om, label + "_var": var, label + "_edge": edge,
                          label + "_n": Q.shape[0], label + "_sec": time.time() - t0})
            print(name, s, om, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                                if k.endswith(("mean", "var", "edge", "n"))}, flush=True)
            rows.append(r)
            pd.DataFrame(rows).to_csv(os.path.join(RES, "fsp_stationary.partial.csv"), index=False)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    models = sys.argv[1].split(",") if len(sys.argv) > 1 else \
        [m for m in MODELS if m.split("-")[0] in ("NAR", "Cascade", "PFB")]
    df = main(models)
    path = os.path.join(RES, "fsp_stationary.csv")
    if len(sys.argv) > 1 and os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old[~old.model.isin(models)], df], ignore_index=True)
    df.to_csv(path, index=False)
    print("wrote", path, len(df))
