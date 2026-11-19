#!/usr/bin/env python3
"""Check of the boundary power law of the stationary hybrid (PDMP) density for
multistate promoters (reviewer R1, minor comment 5).

Claim (Supplementary Section S1.4 of the revision). Let x_b be a boundary of the
support at which the flows of a group g of promoter states vanish linearly,
F_m(x) = -gamma (x - x_b) for m in g, while the other states have flows pointing
away from x_b. If Q is Lipschitz near x_b and the restriction Q_gg(x_b) is
irreducible, the stationary density behaves as |x - x_b|^theta with
theta = s k_exit / gamma - 1, where k_exit is the Perron eigenvalue of
-Q_gg(x_b) (the decay rate of the probability of remaining in g). The density
diverges at x_b iff s k_exit < gamma.

Numerical check: the stationary density of the PDMP is computed with an upwind
finite-volume scheme on a grid refined geometrically towards both boundaries
(cells per decade 50, 100, 200, 400), and theta is fitted to the density summed
over the states of g on 1e-7 < |x - x_b| < 1e-4.
Writes benchmarking_results/revision/boundary_exponent.csv.
"""
import os
import numpy as np, pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qss_diffusion import CIRC
from promoter_family2 import site as site_all
from revision_reduction import site_any

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "benchmarking_results", "revision")


def grid(xa, xb, per_decade, ymin=1e-9, ymax=1.0, n_mid=1500):
    g = np.logspace(np.log10(ymin), np.log10(ymax), int(per_decade * np.log10(ymax / ymin)) + 1)
    left = xa + np.r_[0.0, g]
    right = xb - np.r_[0.0, g][::-1]
    mid = np.linspace(left[-1], right[0], n_mid)[1:-1]
    return np.r_[left, mid, right]


def stationary_density(Q, act, c, s, edges):
    """Upwind finite volumes: a CTMC on (promoter state, cell)."""
    h = np.diff(edges); xc = 0.5 * (edges[1:] + edges[:-1]); nc = len(h); m = len(act)
    idx = lambda st, i: st * nc + i
    rows, cols, vals = [], [], []
    for a in range(m):
        Fe = c["beta"] + c["alpha"] * act[a] - c["gamma"] * edges      # flow at the interfaces
        i = np.arange(nc - 1)
        up = np.maximum(Fe[1:-1], 0.0) / h[:-1]                         # cell i -> i+1
        dn = np.maximum(-Fe[1:-1], 0.0) / h[1:]                         # cell i+1 -> i
        rows += [idx(a, i), idx(a, i + 1)]; cols += [idx(a, i + 1), idx(a, i)]; vals += [up, dn]
    Qc = np.array([Q(v) for v in xc]) * s
    for a in range(m):
        for b in range(m):
            if a != b:
                rows.append(idx(a, np.arange(nc))); cols.append(idx(b, np.arange(nc))); vals.append(Qc[:, a, b])
    r = np.concatenate(rows); cc = np.concatenate(cols); v = np.concatenate(vals)
    ok = v > 0
    G = sp.coo_matrix((v[ok], (r[ok], cc[ok])), shape=(m * nc, m * nc)).tocsr()
    G = G - sp.diags(np.asarray(G.sum(axis=1)).ravel())
    A = G.T.tocsc(); n = A.shape[0]
    k = int(np.argmin(np.abs(xc - 0.5 * (edges[0] + edges[-1]))))      # anchor in the interior, state 0
    keep = np.r_[0:k, k + 1:n]
    x = spla.spsolve(A[keep][:, keep], -np.asarray(A[keep][:, [k]].todense()).ravel())
    p = np.maximum(np.insert(x, k, 1.0), 0.0); p /= p.sum()
    return xc, (p.reshape(m, nc) / h[None, :])


def k_exit(Q, act, c, xb):
    F = c["beta"] + c["alpha"] * act - c["gamma"] * xb
    g = np.abs(F) < 1e-12
    q = Q(xb)[np.ix_(g, g)]
    ev = np.linalg.eigvals(-q)
    return float(np.min(ev.real)), g


def main():
    cases = []
    for name, sysname in (("NAR-S", "NAR"), ("PFB-S", "PFB")):
        Q, act, _ = site_all(sysname, 2, 100.0)
        cases.append((name, sysname, Q, act))
    Q, act, _, _ = site_any(3, 3.0)
    cases.append(("NAR-X", "NAR", Q, act))
    rows = []
    for name, sysname, Q, act in cases:
        c = CIRC[sysname]
        xa, xb = c["beta"] / c["gamma"], (c["beta"] + c["alpha"]) / c["gamma"]
        for side, x0 in (("lower", xa), ("upper", xb)):
            ke, g = k_exit(Q, act, c, x0)
            for s in (0.03, 0.1, 0.3, 1.0, 3.0):
                theta_pred = s * ke / c["gamma"] - 1
                for pdec in (50, 100, 200, 400):
                    edges = grid(xa, xb, pdec)
                    xc, P = stationary_density(Q, act, c, s, edges)
                    y = np.abs(xc - x0); dens = P[g].sum(axis=0)
                    sel = (y > 1e-7) & (y < 1e-4) & (dens > 0)
                    theta = np.polyfit(np.log(y[sel]), np.log(dens[sel]), 1)[0] if sel.sum() > 5 else np.nan
                    rows.append(dict(promoter=name, boundary=side, x_b=x0, n_group=int(g.sum()), k_exit=ke, s=s,
                                     cells_per_decade=pdec, theta_pred=theta_pred, theta_fit=theta))
                    print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "boundary_exponent.csv"), index=False)


if __name__ == "__main__":
    main()
