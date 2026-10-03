#!/usr/bin/env python3
"""SI table of exact toggle escape rates (toggle_exact_*.csv)."""
import os, sys, glob
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def f(v):
    if v == 0 or v != v:
        return "---"
    import math
    e = int(math.floor(math.log10(v))); return r"$%.2f\times10^{%d}$" % (v / 10 ** e, e)


if __name__ == "__main__":
    d = pd.concat([pd.read_csv(x) for x in glob.glob(os.path.join(RES, "toggle_exact_*.csv"))])
    d = d.drop_duplicates(["arch", "method", "s", "omega"])
    p = d.pivot_table(index=["arch", "omega", "s"], columns="method", values=["k_U", "k_V"]).reset_index()
    order = {a: i for i, a in enumerate(["H", "P", "Q", "Qd", "S", "E", "M"])}
    p = p.sort_values(by=[("arch", ""), ("omega", ""), ("s", "")], key=lambda c: c.map(order) if c.name == ("arch", "") else c)
    L = [r"{\footnotesize", r"\begin{longtable}{lrrllll}",
         r"\caption{Exact escape rates of the toggle switch from the stationary distribution of the chain augmented by the basin label (Supplementary Section~\ref{si:toggle_exact}), for the full model (SSA) and the sQSSA: $k_U$ from the $U$-high and $k_V$ from the $V$-high basin, per unit time. Qd: Q with bound regulators degraded at the free rate. No sQSSA is given for Q and Qd.}\label{tab:si_toggle_exact}\\",
         r"\toprule arch. & $\Omega$ & $s$ & $k_U$ (SSA) & $k_U$ (sQSSA) & $k_V$ (SSA) & $k_V$ (sQSSA) \\ \midrule\endfirsthead",
         r"\toprule arch. & $\Omega$ & $s$ & $k_U$ (SSA) & $k_U$ (sQSSA) & $k_V$ (SSA) & $k_V$ (sQSSA) \\ \midrule\endhead"]
    for _, r in p.iterrows():
        g = lambda k, m: r[(k, m)] if (k, m) in r.index else float("nan")
        L.append("%s & %g & %g & %s & %s & %s & %s \\\\" % (r[("arch", "")], r[("omega", "")], r[("s", "")],
                 f(g("k_U", "ssa")), f(g("k_U", "sqssa")), f(g("k_V", "ssa")), f(g("k_V", "sqssa"))))
    L += [r"\bottomrule", r"\end{longtable}}"]
    open(sys.argv[1], "w").write("\n".join(L) + "\n")
