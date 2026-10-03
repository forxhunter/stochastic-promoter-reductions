#!/usr/bin/env python3
"""SI tables for the out-of-sample test of the bimodality check
(check_circuits.csv, bimodality_check_performance.csv)."""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")

if __name__ == "__main__":
    d = pd.read_csv(os.path.join(RES, "check_circuits.csv"))
    d["lna_var_err"] = d.lna_var / d.ssa_var - 1
    L = [r"{\footnotesize", r"\begin{longtable}{lrrrrrrrrrl}",
         r"\caption{Test circuits for the bimodality check, exact references. SA: non-cooperative self-activating gene (order 1; basal rate 2, maximal rate 60, half-maximum 30); PFBC: cooperative self-activating gene (order 2; basal rate 20, maximal rate 80, half-maximum 30), concerted (H) and site-resolved (S). $N$: exact mean copy number; $x^*$: fixed point (concentration); LNA: promoter share; hyb., sQ.: exact fractions of the SSA variance kept by the hybrid and the sQSSA; dev.: larger deviation of the two fractions from the LNA; var.: LNA variance relative to the exact variance minus one; mass: probability of the second mode of the first-order density. Failure: dev.\ $>0.1$; flag: mass $>0.01$.}\label{tab:si_check}\\",
         r"\toprule model & $s$ & $\Omega$ & $N$ & $x^*$ & LNA & hyb. & sQ. & dev. & var. & mass \\ \midrule\endfirsthead",
         r"\toprule model & $s$ & $\Omega$ & $N$ & $x^*$ & LNA & hyb. & sQ. & dev. & var. & mass \\ \midrule\endhead"]
    for _, r in d.iterrows():
        L.append("%s & %g & %g & %.1f & %.1f & %.2f & %.2f & %.2f & %.3f & %+.2f & %.3f \\\\" % (
            r.model, r.s, r.omega, r.ssa_mean * r.omega, r.x_star, r.lna_share, r.hyb_frac, r.sq_frac, r.dev,
            r.lna_var_err, r.qss_low_mass))
    L += [r"\bottomrule", r"\end{longtable}}"]
    p = pd.read_csv(os.path.join(RES, "bimodality_check_performance.csv"))
    L += [r"\begin{table}[!htbp]", r"\centering",
          r"\caption{Performance of the bimodality check on the development set (the two autoregulated genes of the main benchmark, 88 conditions) and the test set (Table~\ref{tab:si_check}, 33 conditions), for three thresholds on the probability of the second mode and two failure criteria: a variance fraction deviating from the LNA by more than 0.1 (fixed before the test circuits were evaluated), and the same or an LNA variance error above 25\% (added after the evaluation, because the check detects bimodality, which breaks the LNA variance as well as the fractions). The threshold of 1\% was fixed before the test circuits were evaluated.}",
          r"\label{tab:si_check_perf}", r"\small", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{llrrrrrr}", r"\toprule",
          r"failure criterion & set & threshold & failures & detected & missed & false alarms & correct passes \\", r"\midrule"]
    for _, r in p.iterrows():
        L.append("%s & %s & %g\\%% & %d & %d & %d & %d & %d \\\\" % (r.criterion.replace("%", "\\%").replace(">", "$>$"), r.set, 100 * r.threshold, r.failures, r.detected,
                                                               r.missed, r.false_pos, r.true_neg))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(sys.argv[1], "w").write("\n".join(L) + "\n")
