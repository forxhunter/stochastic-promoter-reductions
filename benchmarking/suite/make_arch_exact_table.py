#!/usr/bin/env python3
"""SI table: exact (FSP) differences between the site-resolved promoter and
two-state promoters in the autoregulated genes (architecture_exact.csv)."""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")

if __name__ == "__main__":
    d = pd.read_csv(os.path.join(RES, "architecture_exact.csv"))
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Exact stationary differences (finite-state projection) between the site-resolved promoter (S) and promoters with the same half-maximum (H) or exactly the same response (E, M, D) in the two autoregulated genes: mean(S)/mean(ref)$-1$ and CV(S)/CV(ref)$-1$, in per cent. E: matched in worst-case $\eta$; M: matched in $S_0$ at the deterministic fixed point; D: matched at each condition in $S_0$ averaged over the exact stationary distribution of the regulator in S; $\lambda_D$ its switching scale and elast.\ the elasticity $d\ln\langle x\rangle_D/d\ln\lambda_D$.}",
         r"\label{tab:si_arch_exact}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule",
         r" & & & \multicolumn{4}{c}{mean} & \multicolumn{4}{c}{CV} & & \\",
         r"Circuit & $s$ & $\Omega$ & H & E & M & D & H & E & M & D & $\lambda_D$ & elast. \\", r"\midrule"]
    names = {"NAR": "neg.\\ autoreg.", "PFB": "pos.\\ autoreg."}
    for sy in ("NAR", "PFB"):
        x = d[d.system == sy]
        mean = x.pivot_table(index=["s", "omega"], columns="arch", values="mean")
        cv = x.pivot_table(index=["s", "omega"], columns="arch", values="cv")
        for (s, om), _ in mean.iterrows():
            cells = ["%+.1f" % (100 * (mean.loc[(s, om), "S"] / mean.loc[(s, om), r] - 1)) for r in "HEMD"]
            cells += ["%+.1f" % (100 * (cv.loc[(s, om), "S"] / cv.loc[(s, om), r] - 1)) for r in "HEMD"]
            dd = x[(x.arch == "D") & (x.s == s) & (x.omega == om)].iloc[0]
            cells += ["%.3f" % dd.lam, "%.2f" % dd.dlnmean_dlnlam]
            L.append("%s & %g & %g & " % (names[sy], s, om) + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(sys.argv[1], "w").write("\n".join(L) + "\n")
