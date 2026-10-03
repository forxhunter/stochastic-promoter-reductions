#!/usr/bin/env python3
"""LaTeX tables of the full benchmark grid for the Supplement, generated from
benchmarking_results/suite_metrics.csv."""
import os, sys
import numpy as np
import pandas as pd
from systems import BASE_OMEGA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
NAME = {"ssa_null": "SSA$_B$", "pdmp": "PDMP", "sqssa": "sQSSA", "ode": "ODE", "pdmp_h4": "PDMP, $h/4$"}


def pct(v):
    return "---" if not np.isfinite(v) else ("%+.0f" % (100 * v) if abs(v) >= 0.1 else "%+.1f" % (100 * v))


def g(v, fmt="%.3g"):
    return "---" if not np.isfinite(v) else fmt % v


def grid_table(df, sysn):
    d = df[(df.system == sysn) & (df.method != "pdmp_h4")].sort_values(["omega", "s", "method"])
    osc = sysn in ("Repressilator", "Goodwin")
    tname = "period" if osc else "corr.\\ time"
    head = (r"$s$ & $\Omega$ & $\eta_{\mathrm{op}}$ & $\eta_{\mathrm{wc}}$ & method & mean & CV & amp. & KS & %s & $p$ \\" % tname)
    L = [r"{\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\begin{longtable}{rrrrlrrrrrr}",
         r"\caption{%s: every condition of the benchmark grid. Errors are relative to SSA ensemble A, in per cent; KS is the distance to ensemble A; $p$ is Welch's test on the temporal observable. SSA$_B$ is the second, independent SSA ensemble. $\eta_{\mathrm{op}}$ and $\eta_{\mathrm{wc}}$ are the operating-point and worst-case timescale ratios.}\label{tab:grid_%s}\\" % (sysn if sysn != "p53" else "p53--Mdm2", sysn.lower()),
         r"\toprule", head, r"\midrule\endfirsthead", r"\toprule", head, r"\midrule\endhead"]
    last = None
    for _, r in d.iterrows():
        key = (r.s, r.omega)
        first = key != last
        if first and last is not None:
            L.append(r"\addlinespace[2pt]")
        last = key
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            g(r.s, "%g") if first else "", g(r.omega, "%g") if first else "",
            g(r.eta_op, "%.2g") if first else "", g(r.eta_worst, "%.2g") if first else "",
            NAME[r.method], pct(r.mean_err), pct(r.cv_err), pct(r.amp_err), g(r.ks, "%.3f"),
            pct(r.time_err), g(r.time_p, "%.2g")))
    L += [r"\bottomrule", r"\end{longtable}}"]
    return "\n".join(L)


def check_table(df):
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{PDMP step-size check at the base condition of each system: metrics with the standard step $h$ and with $h/4$, both against SSA ensemble A.}",
         r"\label{tab:si_hcheck}", r"\small", r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"System & step & mean & CV & amp. & KS & temporal \\", r"\midrule"]
    for sysn in ("Repressilator", "Goodwin", "Toggle", "p53"):
        for m in ("pdmp", "pdmp_h4"):
            r = df[(df.system == sysn) & (df.method == m) & (df.s == 1) & (df.omega == BASE_OMEGA[sysn])]
            if r.empty:
                continue
            r = r.iloc[0]
            L.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
                sysn if m == "pdmp" else "", "$h$" if m == "pdmp" else "$h/4$",
                pct(r.mean_err), pct(r.cv_err), pct(r.amp_err), g(r.ks, "%.3f"), pct(r.time_err)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def cost_table(df):
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{Wall-clock time per trajectory at the base condition (single core, same compiled engine for every method), and the SSA/method ratio.}",
         r"\label{tab:si_cost}", r"\small", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"System & SSA (s) & PDMP (s) & sQSSA (s) & ODE (s) & SSA/PDMP & SSA/sQSSA \\", r"\midrule"]
    for sysn in ("Repressilator", "Goodwin", "Toggle", "p53"):
        d = df[(df.system == sysn) & (df.s == 1) & (df.omega == BASE_OMEGA[sysn])]
        w = {m: d[d.method == m].wall.values[0] for m in ("pdmp", "sqssa", "ode") if (d.method == m).any()}
        ref = d.wall_ref.values[0]
        L.append("%s & %.3g & %.3g & %.3g & %.3g & %.2g & %.2g \\\\" % (
            sysn, ref, w["pdmp"], w["sqssa"], w["ode"], ref / w["pdmp"], ref / w["sqssa"]))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    out = sys.argv[1]
    df = pd.read_csv(os.path.join(RES, "suite_metrics.csv"))
    with open(os.path.join(out, "grid_tables.tex"), "w") as f:
        for sysn in ("Repressilator", "Goodwin", "Toggle", "p53"):
            f.write(grid_table(df, sysn) + "\n\n")
    with open(os.path.join(out, "hcheck_table.tex"), "w") as f:
        f.write(check_table(df) + "\n")
    with open(os.path.join(out, "cost_table.tex"), "w") as f:
        f.write(cost_table(df) + "\n")
    print("wrote tables to", out)
