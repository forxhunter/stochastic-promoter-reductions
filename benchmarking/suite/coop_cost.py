#!/usr/bin/env python3
"""Kinetic cost of cooperativity. For n sites with pairwise factor omega placed
in the unbinding rates (unbinding from k+1 bound at (k+1) k_b omega^-k), find the
omega that gives a half-maximum Hill-plot slope of 0.9 n, then the binding-rate
scale k_b needed for the slowest promoter relaxation (over regulator levels from
zero to five times the half-maximum) to equal that of a concerted two-state
promoter with on-rate 1 and the same half-maximum. Writes a LaTeX table."""
import sys
import numpy as np
from math import lgamma, log
from scipy.optimize import brentq


def logw(lr, n, lw):
    k = np.arange(n + 1)
    lc = np.array([lgamma(n + 1) - lgamma(kk + 1) - lgamma(n - kk + 1) for kk in k])
    return lc + k * lr + k * (k - 1) / 2 * lw


def act(lr, n, lw):          # repressed: active unless all bound
    l = logw(lr, n, lw); m = l.max()
    p = np.exp(l - m); p /= p.sum()
    return 1 - p[-1]


def half(n, lw):
    return brentq(lambda lr: act(lr, n, lw) - 0.5, -50 - n * lw, 50)


def slope(n, lw):
    lr = half(n, lw); h = 1e-5
    g = lambda z: np.log(act(z, n, lw) / (1 - act(z, n, lw)))
    return abs(g(lr + h) - g(lr - h)) / (2 * h)


def lam2(n, lw, kb, x):      # x in units of K_S
    Q = np.zeros((n + 1, n + 1))
    for k in range(n):
        Q[k, k + 1] = (n - k) * kb * x
        Q[k + 1, k] = (k + 1) * kb * np.exp(-k * lw)
    Q -= np.diag(Q.sum(axis=1))
    return np.sort(np.abs(np.real(np.linalg.eigvals(Q))))[1]


def row(n, target=0.9):
    lw = brentq(lambda z: slope(n, z) - target * n, 1e-6, 60)
    xh = np.exp(half(n, lw))                       # half-maximum in units of K_S
    xs = np.concatenate([[0.0], np.linspace(xh / 1e3, 5 * xh, 400)])
    lam_s = min(lam2(n, lw, 1.0, x) for x in xs)
    # concerted promoter: lambda2 = 1 + (x/K)^n >= 1, minimum 1 at x = 0
    return n, np.exp(lw), 1.0 / lam_s


if __name__ == "__main__":
    rows = [row(n) for n in range(2, 11)]
    for n, w, f in rows:
        print("n=%2d omega=%.3g  kb factor=%.3g" % (n, w, f))
    if len(sys.argv) > 1:
        L = [r"\begin{table}[!htbp]", r"\centering",
             r"\caption{Kinetic cost of cooperativity. For $n$ sites with pairwise factor $\omega$ acting on unbinding, $\omega$ is chosen so that the half-maximum Hill-plot slope is $0.9n$; the binding and unbinding rates are then scaled so that the slowest promoter relaxation equals that of a concerted two-state promoter with the same half-maximum and unit on-rate. The last column is that scale factor.}",
             r"\label{tab:si_coopcost}", r"\small", r"\begin{tabular}{rrr}", r"\toprule",
             r"$n$ & $\omega$ & rate scale \\", r"\midrule"]
        for n, w, f in rows:
            fmt = lambda v: ("%.3g" % v) if v < 100 else (r"%d" % round(v) if v < 1e4 else r"%.2g\times10^{%d}" % (v / 10 ** int(np.log10(v)), int(np.log10(v))))
            L.append(r"%d & $%s$ & $%s$ \\" % (n, fmt(w), fmt(f)))
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(sys.argv[1], "w").write("\n".join(L) + "\n")
