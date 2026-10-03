#!/usr/bin/env python3
"""Main-text Tables 2 and 3 from benchmarking_results/suite2_metrics.csv, with one
rounding rule (half away from zero) so that the text, the tables and the
Supplement agree."""
import os, sys
from decimal import Decimal, ROUND_HALF_UP
import numpy as np, pandas as pd
from systems2 import BASE_OMEGA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
ORDER = ["Repressilator-H", "Repressilator-S", "Goodwin-H", "Toggle-H", "Toggle-S",
         "p53-S", "p53-H", "NAR-H", "NAR-S", "Cascade-H", "Cascade-S", "PFB-H", "PFB-S"]
NAME = {"Repressilator": "Repressilator", "Goodwin": "Goodwin", "Toggle": "Toggle",
        "p53": "p53--Mdm2", "NAR": r"Neg.\ autoregulation", "Cascade": "Cascade", "PFB": r"Pos.\ autoregulation"}


def rnd(v, nd=0):
    if not np.isfinite(v):
        return "---"
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP))


def pct(v, nd=0):
    if not np.isfinite(v):
        return "---"
    s = rnd(100 * v, nd)
    if float(s) == 0:
        return "0"
    return "+" + s if not s.startswith("-") else s


def g(v, sig=2):
    if not np.isfinite(v):
        return "---"
    if v < 1e-3:
        e = int(np.floor(np.log10(v)))
        return r"$%.0f\times10^{%d}$" % (v / 10 ** e, e)
    return "%.*g" % (sig, v)


def exact():
    """Exact stationary references for the two autoregulated genes, keyed by model
    at the base condition."""
    x = pd.read_csv(os.path.join(RES, "exact_reference.csv"))
    return x[(x.s == 1) & (x.omega == 1)].set_index("model")


def base(df):
    return df[[r.s == 1 and r.omega == BASE_OMEGA[r.system] for r in df.itertuples()]]


def table2(df):
    b = base(df)
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Base parameters. $\eta_{\mathrm{op}}$ and $\eta_{\mathrm{wc}}$ are the operating-point and worst-case timescale ratios, $N$ the mean copy number of the observed species (exact for the two autoregulated genes). W1 is the Wasserstein-1 distance to SSA divided by the SSA standard deviation (SSA$_B$: second SSA ensemble). The toggle switch, whose ensembles are bimodal and not stationary on the simulated time, is treated separately (Table~\ref{tab:toggle}). The variance columns give the fraction of the SSA variance kept by each reduction (not given for oscillators, where it mainly reflects amplitude); for stationary circuits the LNA prediction for the hybrid is given alongside (the sQSSA prediction is one minus it). For the two autoregulated genes the W1 distances of the reductions and the variance fractions are exact: finite-state projection for SSA and sQSSA, and a converged discretisation of the forward equation for the hybrid; the SSA$_B$ column is from simulation for every circuit.}",
         r"\label{tab:benchmark_synthesis}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
         r" & & & & & \multicolumn{3}{c}{W1/SD} & \multicolumn{3}{c}{variance kept} \\",
         r"\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
         r"Circuit & Arch. & $\eta_{\mathrm{op}}$ & $\eta_{\mathrm{wc}}$ & $N$ & SSA$_B$ & hybrid & sQSSA & hybrid & sQSSA & LNA \\",
         r"\midrule"]
    for m in ORDER:
        x = b[b.model == m].set_index("method")
        sysn = m.split("-")[0]
        if x.empty or sysn == "Toggle":
            continue
        osc = sysn in ("Repressilator", "Goodwin")
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            NAME[sysn] if ORDER.index(m) == min(i for i, o in enumerate(ORDER) if o.startswith(sysn + "-")) else "", m[-1],
            g(x.eta_op.iloc[0]), g(x.eta_wc.iloc[0]), g(EX.loc[m, "ssa_mean"] * EX.loc[m, "omega"] if m in EX.index else x.N.iloc[0], 3),
            rnd(x.loc["ssa_null", "w1"], 3),
            rnd(EW.loc[(m, 1.0, 1.0), "w1_pdmp"] if (m, 1.0, 1.0) in EW.index else x.loc["pdmp", "w1"], 2),
            rnd(EW.loc[(m, 1.0, 1.0), "w1_sqssa"] if (m, 1.0, 1.0) in EW.index else x.loc["sqssa", "w1"], 2),
            "---" if osc else rnd(EX.loc[m, "pdmp_var_ratio"] if m in EX.index else x.loc["pdmp", "var_ratio"], 2),
            "---" if osc else rnd(EX.loc[m, "sqssa_var_ratio"] if m in EX.index else x.loc["sqssa", "var_ratio"], 2),
            rnd(x.lna_share.iloc[0], 2) if np.isfinite(x.lna_share.iloc[0]) else "---"))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L)


def table3(df):
    b = base(df)
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Errors at the base parameters, relative to SSA, in per cent, with 95\% bootstrap intervals for the stochastic descriptions. The temporal observable is the period for oscillators and the autocorrelation time for stationary circuits. The toggle switch is treated separately (Table~\ref{tab:toggle}). The averaged ODE is compared on mean and period only. $^{*}$For the two autoregulated genes mean and CV errors are relative to the exact stationary distribution (finite-state projection; for the hybrid, a converged discretisation of its forward equation) and carry no interval; the last column gives the error of the Monte Carlo SSA reference against the exact distribution, and the correlation time is compared with Monte Carlo as for the other circuits.}",
         r"\label{tab:observable_errors}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{lllllll}", r"\toprule",
         r"Model & quantity & SSA$_B$ & hybrid & sQSSA & ODE & SSA (MC) \\", r"\midrule"]
    ci = lambda r, k: "%s [%s, %s]" % (pct(r[k]), pct(r[k + "_lo"]), pct(r[k + "_hi"])) if np.isfinite(r[k]) else "---"
    for m in ORDER:
        sysn = m.split("-")[0]
        if sysn == "Toggle":
            continue
        x = b[b.model == m].set_index("method")
        if x.empty:
            continue
        name = "%s-%s" % (NAME[sysn], m[-1])
        if m in EX.index:
            e = EX.loc[m]
            mc_mean = x.N.iloc[0] / e.omega / e.ssa_mean - 1
            mc_cv = np.sqrt(x.ssa_var.iloc[0]) / x.N.iloc[0] / e.ssa_cv - 1
            L.append("%s$^{*}$ & mean & %s & %s & %s & %s & %s \\\\" % (name, ci(x.loc["ssa_null"], "mean_err"), pct(e.pdmp_mean_err), pct(e.sqssa_mean_err), pct(e.ode_mean_err), pct(mc_mean)))
            L.append(" & CV & %s & %s & %s & --- & %s \\\\" % (ci(x.loc["ssa_null"], "cv_err"), pct(e.pdmp_cv_err), pct(e.sqssa_cv_err), pct(mc_cv)))
        else:
            L.append("%s & mean & %s & %s & %s & %s & \\\\" % (name, ci(x.loc["ssa_null"], "mean_err"), ci(x.loc["pdmp"], "mean_err"),
                                                      ci(x.loc["sqssa"], "mean_err"), pct(x.loc["ode", "mean_err"])))
            L.append(" & CV & %s & %s & %s & --- & \\\\" % (ci(x.loc["ssa_null"], "cv_err"), ci(x.loc["pdmp"], "cv_err"), ci(x.loc["sqssa"], "cv_err")))
        if sysn in ("Repressilator", "Goodwin"):
            L.append(" & period & %s & %s & %s & %s & \\\\" % (ci(x.loc["ssa_null"], "period_err"), ci(x.loc["pdmp"], "period_err"),
                                                    ci(x.loc["sqssa"], "period_err"), pct(x.loc["ode", "period_err"])))
        else:
            L.append(" & corr.\\ time & %s & %s & %s & --- & \\\\" % (ci(x.loc["ssa_null"], "tac_err"), ci(x.loc["pdmp"], "tac_err"), ci(x.loc["sqssa"], "tac_err")))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    out = sys.argv[1]
    df = pd.read_csv(os.path.join(RES, "suite2_metrics.csv"))
    EX = exact()
    EW = pd.read_csv(os.path.join(RES, "exact_w1.csv")).set_index(["model", "s", "omega"])
    open(os.path.join(out, "table_base.tex"), "w").write(table2(df) + "\n")
    open(os.path.join(out, "table_errors.tex"), "w").write(table3(df) + "\n")
    print(table2(df)); print(table3(df))
