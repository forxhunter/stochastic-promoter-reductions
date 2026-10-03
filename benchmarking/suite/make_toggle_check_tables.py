#!/usr/bin/env python3
"""SI tables for the two checks of the toggle escape rates: the label-sampling
interval (toggle_dt_check.csv against the exact rates) and the label thresholds
(toggle_robustness.csv). Writes
Supplementary_Material_v2_next_revision/generated/toggle_check_tables.tex."""
import os, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
OUT = os.path.join(HERE, "..", "..", "Supplementary_Material_v2_next_revision", "generated",
                   "toggle_check_tables.tex")


def sci(v):
    m, e = ("%.2e" % v).split("e")
    return "$%s\\times10^{%d}$" % (m, int(e))


def main():
    dt = pd.read_csv(os.path.join(RES, "toggle_dt_check.csv"))
    ex = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RES, "toggle_exact_*.csv"))])
    ex = ex.drop_duplicates(["arch", "method", "s", "omega"])
    e = ex[(ex.method == "ssa") & (ex.s == 1) & (ex.omega == 1)].set_index("arch")
    L = ["\\begin{table}[!htbp]\\centering\\small",
         "\\caption{Toggle switch at $\\Omega=1$, $s=1$: escape rates from new SSA ensembles "
         "(64 trajectories started in each basin, $T=10^4$) with the basin label checked every "
         "$\\Delta t$ time units, bootstrap 95\\% intervals and numbers of exits, against the exact "
         "continuous-time rates, all in units of $10^{-4}$ per unit time (\\url{benchmarking_results/toggle_dt_check.csv}).}"
         "\\label{tab:si_toggle_dt}",
         "\\begin{tabular}{llllll}\\toprule",
         "arch. & $\\Delta t$ & $k_U$ [interval] (exits) & $k_U$ exact & $k_V$ [interval] (exits) & $k_V$ exact \\\\ \\midrule"]
    for a in "HPSEM":
        for _, r in dt[dt.arch == a].iterrows():
            f = lambda v: "%.3g" % (1e4 * v)
            L.append("%s & %g & %s [%s, %s] (%d) & %s & %s [%s, %s] (%d) & %s \\\\" % (
                a, r["dt"], f(r.k_U), f(r.k_U_lo), f(r.k_U_hi), r.exits_U, f(e.loc[a, "k_U"]),
                f(r.k_V), f(r.k_V_lo), f(r.k_V_hi), r.exits_V, f(e.loc[a, "k_V"])))
    L += ["\\bottomrule\\end{tabular}",
          "\\end{table}", ""]
    rb = pd.read_csv(os.path.join(RES, "toggle_robustness.csv"))
    L += ["\\begin{table}[!htbp]\\centering\\small",
          "\\caption{Exact escape rates of the toggle switch for two basin labels: entering the "
          "$U$-high basin at $U>2V$ and leaving it at $U<V/2$ (as in the simulations), and at $U>3V$ "
          "and $U<V/3$ (\\url{benchmarking_results/toggle_robustness.csv}).}\\label{tab:si_toggle_thr}",
          "\\begin{tabular}{lllllll}\\toprule",
          "arch. & $\\Omega$ & $s$ & $k_U$ $(2, \\frac12)$ & $k_U$ $(3, \\frac13)$ & $k_V$ $(2, \\frac12)$ & $k_V$ $(3, \\frac13)$ \\\\ \\midrule"]
    for _, r in rb.sort_values(["arch", "omega", "s"]).iterrows():
        L.append("%s & %g & %g & %s & %s & %s & %s \\\\" % (r.arch, r.omega, r.s, sci(r.k_U), sci(r.k_U_3),
                                                           sci(r.k_V), sci(r.k_V_3)))
    L += ["\\bottomrule\\end{tabular}", "\\end{table}", ""]
    open(OUT, "w").write("\n".join(L))


if __name__ == "__main__":
    main()
