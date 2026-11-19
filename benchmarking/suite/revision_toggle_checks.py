#!/usr/bin/env python3
"""Revision checks of the exact toggle-switch escape rates (reviewer comments
R1.6 and R1.7 of JTB-D-26-01677).

1. Truncation. The exact rates of toggle_exact.py are recomputed with the copy-number
   box reaching k = 5, 7 (production value), 9 and 11 standard deviations above the
   largest production rate, for the conditions behind the main-text rare-event
   statements. Each stationary vector is computed by sparse LU anchored at the most
   probable state and improved by one step of iterative refinement with the same
   factorisation; the change of the rates under refinement measures the accuracy of
   the linear solve for small probabilities.
2. Metastable rates. For the chain without the basin label, the slowest non-zero
   eigenvalue lambda_1 of the generator (shift-invert Arnoldi) and the next one,
   lambda_2. For a metastable two-basin chain lambda_1 = k_U + k_V, where k are the
   rates of the coarse two-state process, and the basin probabilities give
   k_U = |lambda_1| P_V and k_V = |lambda_1| P_U. P_U is computed from the right
   eigenvector psi_1 (soft basin weight (psi - psi_V)/(psi_U - psi_V), clipped to
   [0, 1]). This rate does not depend on the thresholds of the label and is not
   inflated by recrossings; the ratio of the label-flip rate to it measures the
   recrossing effect.
Writes benchmarking_results/revision/toggle_truncation.csv and toggle_spectral.csv.
Usage: revision_toggle_checks.py truncation|spectral ARCHS OMEGAS [S] [METHOD] [K_LIST]
"""
import os, sys, time
import numpy as np, pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import toggle_exact as te
from systems2 import MODELS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "benchmarking_results", "revision")
_box = te.box_hi


def set_box(k):
    te.box_hi = lambda M, name, om, kk=k: _box(M, name, om, kk)


def model(arch):
    return te.toggle_Qd() if arch == "Qd" else MODELS["Toggle-" + arch]()


def label_generator(M, om, s, averaged, enter=2.0, leave=0.5):
    rows, cols, vals, U, V, ns, (hu, hv) = te.build(M, om, s, averaged)
    lab_src = np.r_[np.zeros(len(rows), int), np.ones(len(rows), int)]
    r2 = np.r_[rows, rows]; c2 = np.r_[cols, cols]; v2 = np.r_[vals, vals]
    Ud, Vd = U[c2], V[c2]
    lab_dst = lab_src.copy()
    lab_dst[(lab_src == 1) & (Ud < leave * Vd)] = 0
    lab_dst[(lab_src == 0) & (Ud > enter * Vd)] = 1
    R = r2 + lab_src * ns; C = c2 + lab_dst * ns
    G = sp.coo_matrix((v2, (R, C)), shape=(2 * ns, 2 * ns)).tocsr()
    G = G - sp.diags(np.asarray(G.sum(axis=1)).ravel())
    return dict(G=G, R=R, v2=v2, lab_src=lab_src, lab_dst=lab_dst, U=U, V=V, ns=ns, hu=hu, hv=hv,
                rows=rows, cols=cols, vals=vals)


def stationary_refined(G, anchor):
    """Two anchored sparse-LU solves and one refinement step with the second factorisation."""
    A = G.T.tocsc(); n = A.shape[0]

    def factor(k):
        keep = np.r_[0:k, k + 1:n]
        lu = spla.splu(A[keep][:, keep], permc_spec="COLAMD")
        return keep, lu
    keep, lu = factor(anchor)
    x = lu.solve(-A[keep][:, [anchor]].toarray().ravel())
    p = np.maximum(np.insert(x, anchor, 1.0), 0.0); p /= p.sum()
    k = int(np.argmax(p)); del lu
    keep, lu = factor(k)
    b = -A[keep][:, [k]].toarray().ravel()
    x = lu.solve(b)
    p1 = np.insert(x, k, 1.0)
    r = A @ p1
    dx = lu.solve(-r[keep])
    p2 = p1 + np.insert(dx, k, 0.0)
    out = []
    for q in (p1, p2):
        neg = float(-q[q < 0].sum() / np.abs(q).sum()) if (q < 0).any() else 0.0
        q = np.maximum(q, 0.0); q = q / q.sum()
        res = np.abs(A @ q).sum() / np.abs(A.diagonal() * q).sum()
        out.append((q, res, neg))
    return out


def rates_from(L, p):
    ns = L["ns"]; R, v2, ls, ld = L["R"], L["v2"], L["lab_src"], L["lab_dst"]
    p1, p0 = p[ns:].sum(), p[:ns].sum()
    f10 = (ls == 1) & (ld == 0); f01 = (ls == 0) & (ld == 1)
    kU = np.sum(p[R[f10]] * v2[f10]) / p1
    kV = np.sum(p[R[f01]] * v2[f01]) / p0
    pm = p[:ns] + p[ns:]
    edge = pm[(L["U"] == L["hu"]) | (L["V"] == L["hv"])].sum()
    return kU, kV, p1, edge


def anchor_of(L, om):
    return int(np.argmin(np.abs(L["U"] - 300 * om) + 10 * L["V"])) + L["ns"]


def task_truncation(archs, oms, s_list, method, ks):
    path = os.path.join(OUT, "toggle_truncation_%s_%s.csv" % ("_".join(archs), method))
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["arch"], r["s"], r["omega"], r["k"]) for r in rows}
    for arch in archs:
        M = model(arch)
        for om in oms:
            for s in s_list:
                for k in ks:
                    if (arch, s, om, k) in done:
                        continue
                    t0 = time.time(); set_box(k)
                    L = label_generator(M, om, s, method == "sqssa")
                    (pA, resA, negA), (pB, resB, negB) = stationary_refined(L["G"], anchor_of(L, om))
                    kU, kV, fU, edge = rates_from(L, pA)
                    kU2, kV2, _, _ = rates_from(L, pB)
                    rows.append(dict(arch=arch, method=method, s=s, omega=om, k=k, hu=L["hu"], hv=L["hv"],
                                     n_states=L["G"].shape[0], k_U=kU, k_V=kV, frac_U=fU, edge=edge,
                                     k_U_refined=kU2, k_V_refined=kV2, residual=resA, residual_refined=resB,
                                     neg_mass=negA, sec=time.time() - t0))
                    print({a: (("%.4g" % b) if isinstance(b, float) else b) for a, b in rows[-1].items()}, flush=True)
                    pd.DataFrame(rows).to_csv(path, index=False)
    set_box(7)


def task_spectral(archs, oms, s_list, method, k=7):
    path = os.path.join(OUT, "toggle_spectral_%s_%s.csv" % ("_".join(archs), method))
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["arch"], r["s"], r["omega"]) for r in rows}
    set_box(k)
    for arch in archs:
        M = model(arch)
        for om in oms:
            for s in s_list:
                if (arch, s, om) in done:
                    continue
                t0 = time.time()
                L = label_generator(M, om, s, method == "sqssa")
                (p, res, neg), _ = stationary_refined(L["G"], anchor_of(L, om))
                kU, kV, fU, edge = rates_from(L, p)
                ns = L["ns"]; pm = p[:ns] + p[ns:]
                Qb = sp.coo_matrix((L["vals"], (L["rows"], L["cols"])), shape=(ns, ns)).tocsr()
                Qb = (Qb - sp.diags(np.asarray(Qb.sum(axis=1)).ravel())).tocsc()
                sigma = 1e-9
                ev, W = spla.eigs(Qb, k=4, sigma=sigma, which="LM", tol=1e-12, ncv=24)
                o = np.argsort(np.abs(ev)); ev = ev[o]; W = W[:, o]
                lam1, lam2 = ev[1], ev[2]
                psi = np.real(W[:, 1])
                inU = L["U"] > 2 * L["V"]; inV = L["U"] < 0.5 * L["V"]
                psiU = np.dot(pm[inU], psi[inU]) / pm[inU].sum(); psiV = np.dot(pm[inV], psi[inV]) / pm[inV].sum()
                w = np.clip((psi - psiV) / (psiU - psiV), 0.0, 1.0)
                PU = float(np.dot(pm, w))
                a1 = abs(lam1.real)
                rows.append(dict(arch=arch, method=method, s=s, omega=om, k=k, lam1_re=lam1.real, lam1_im=lam1.imag,
                                 lam2_re=lam2.real, lam2_im=lam2.imag, separation=abs(lam2.real) / a1,
                                 P_U_soft=PU, P_U_label=fU, k_U_label=kU, k_V_label=kV,
                                 k_U_met=a1 * (1 - PU), k_V_met=a1 * PU,
                                 ratio_U=kU / (a1 * (1 - PU)), ratio_V=kV / (a1 * PU),
                                 lam1_vs_sum=a1 / (kU + kV), sec=time.time() - t0))
                print({a: (("%.4g" % b) if isinstance(b, float) else b) for a, b in rows[-1].items()}, flush=True)
                pd.DataFrame(rows).to_csv(path, index=False)
    set_box(7)


if __name__ == "__main__":
    task = sys.argv[1]
    archs = sys.argv[2].split(",")
    oms = [float(v) for v in sys.argv[3].split(",")]
    s_list = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [1.0]
    method = sys.argv[5] if len(sys.argv) > 5 else "ssa"
    ks = [int(v) for v in sys.argv[6].split(",")] if len(sys.argv) > 6 else [5, 7, 9, 11]
    if task == "truncation":
        task_truncation(archs, oms, s_list, method, ks)
    else:
        task_spectral(archs, oms, s_list, method)
