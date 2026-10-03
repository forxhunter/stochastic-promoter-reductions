#!/usr/bin/env python3
"""SI table: first-order QSS diffusion approximation against exact means
(qss_diffusion.csv)."""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")

if __name__ == "__main__":
    d = pd.read_csv(os.path.join(RES, "qss_diffusion.csv"))
    L = [r"{\footnotesize", r"\begin{longtable}{llrrrrrrl}",
         r"\caption{First-order quasi-steady-state diffusion approximation (Eq.~\eqref{eq:si_qss_density}) against the exact stationary mean (finite-state projection) of the regulator concentration in the two autoregulated genes. QSS: mean of the approximation; no $b$: the same without the noise-induced drift; exact: finite-state projection; error: QSS relative to exact. Modes: local maxima of the approximate density. S: site-resolved; M: two-state control with the S response matched in $S_0$ at the deterministic fixed point; D: the same control matched at each condition in $S_0$ averaged over the stationary distribution of S; E: matched in worst-case $\eta$.}\label{tab:si_qss}\\",
         r"\toprule circuit & arch. & $s$ & $\Omega$ & QSS & no $b$ & exact & error & modes \\ \midrule\endfirsthead",
         r"\toprule circuit & arch. & $s$ & $\Omega$ & QSS & no $b$ & exact & error & modes \\ \midrule\endhead"]
    for _, r in d.iterrows():
        L.append("%s & %s & %g & %g & %.2f & %.2f & %.2f & %+.1f\\%% & %s \\\\" % (
            r.system, r.arch, r.s, r.omega, r.qss_mean, r.qss_mean_no_b, r.exact_mean,
            100 * (r.qss_mean / r.exact_mean - 1), str(r.modes).replace(";", ", ")))
    L += [r"\bottomrule", r"\end{longtable}}"]
    open(sys.argv[1], "w").write("\n".join(L) + "\n")
