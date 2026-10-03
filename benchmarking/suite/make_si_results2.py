#!/usr/bin/env python3
"""Supplement tables for the second-design grid (suite2_metrics.csv,
suite2_architecture.csv)."""
import os, sys
import numpy as np, pandas as pd
from systems2 import BASE_OMEGA, MODELS
from make_main_tables import rnd, pct

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
NAME = {"ssa_null": "SSA$_B$", "pdmp": "hybrid", "sqssa": "sQSSA", "ode": "ODE", "pdmp_h4": "hybrid, $h/4$"}


def g(v, sig=2):
    if not np.isfinite(v):
        return "---"
    if abs(v) >= 100:
        return "%d" % round(v)
    return "%.*g" % (sig, v)


def model_table(df, model):
    d = df[(df.model == model) & ~df.method.isin(["pdmp_h4", "pdmpx"])].sort_values(["omega", "s", "method"])
    if d.empty:
        return ""
    sysn = model.split("-")[0]
    tcol = {"Repressilator": "period", "Goodwin": "period", "Toggle": "switches"}.get(sysn, "corr.\\ time")
    head = r"$s$ & $\Omega$ & $\eta_{\mathrm{op}}$ & $\eta_{\mathrm{wc}}$ & $N$ & LNA & S1 & method & mean & CV & W1/SD & var.\ kept & %s \\" % tcol
    L = [r"{\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{rrrrrrrlrrrrr}",
         r"\caption{%s: every condition. Errors relative to SSA ensemble A in per cent; W1/SD is the normalised Wasserstein-1 distance; var.\ kept is the fraction of the SSA variance; %s. LNA: predicted fraction of the variance kept by the hybrid (stationary circuits); S1: the same fraction from the closed form of Eq.~\ref{eq:si_cv_decomposition} at the operating point.}\label{tab:grid2_%s}\\" % (
             model.replace("p53", "p53--Mdm2"),
             "switches are basin changes in 64 trajectories (SSA A in the SSA$_B$ row, after the semicolon)" if sysn == "Toggle" else "the last column is the relative error of the temporal observable",
             model.lower()),
         r"\toprule", head, r"\midrule\endfirsthead", r"\toprule", head, r"\midrule\endhead"]
    last = None
    for _, r in d.iterrows():
        key = (r.s, r.omega); first = key != last
        if first and last is not None:
            L.append(r"\addlinespace[2pt]")
        last = key
        if sysn == "Toggle":
            t = ("%d; %d" % (r.switch_A, r.switch_M)) if r.method == "ssa_null" else ("%d" % r.switch_M if np.isfinite(r.switch_M) else "---")
        elif sysn in ("Repressilator", "Goodwin"):
            t = pct(r.period_err, 1) if np.isfinite(r.get("period_err", np.nan)) else "---"
        else:
            t = pct(r.tac_err, 0) if np.isfinite(r.get("tac_err", np.nan)) else "---"
        lna = rnd(r.lna_share, 3) if (first and np.isfinite(r.get("lna_share", np.nan))) else ("---" if first else "")
        s1 = rnd(r.s1_share, 3) if (first and np.isfinite(r.get("s1_share", np.nan))) else ("---" if first else "")
        L.append("%s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s & %s \\\\" % (
            g(r.s, 3) if first else "", g(r.omega, 3) if first else "", g(r.eta_op) if first else "", g(r.eta_wc) if first else "",
            g(r.N, 3) if first else "", lna, s1, NAME[r.method], pct(r.mean_err, 1),
            pct(r.cv_err, 1) if np.isfinite(r.get("cv_err", np.nan)) else "---",
            rnd(r.w1, 3) if np.isfinite(r.get("w1", np.nan)) else "---",
            rnd(r.var_ratio, 3) if np.isfinite(r.get("var_ratio", np.nan)) else "---", t))
    L += [r"\bottomrule", r"\end{longtable}}"]
    return "\n".join(L)


def hcheck(df):
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Hybrid step-size check at the base condition ($s=1$) and at the largest promoter-speed factor of each model: metrics with the standard Runge--Kutta step $h$ and with $h/4$, both against SSA ensemble A.}",
         r"\label{tab:si_hcheck}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Model & step & mean & CV & W1/SD & var.\ kept & temporal \\", r"\midrule"]
    for m, s_ in [(m, s_) for m in MODELS for s_ in sorted(df[(df.model == m) & (df.method == "pdmp_h4")].s.unique())]:
        sysn = m.split("-")[0]
        for meth in ("pdmp", "pdmp_h4"):
            r = df[(df.model == m) & (df.method == meth) & (df.s == s_) & (df.omega == BASE_OMEGA[sysn])]
            if r.empty or sysn == "Toggle":
                continue
            r = r.iloc[0]
            if sysn == "Toggle":
                t = "%d switches" % r.switch_M
            elif sysn in ("Repressilator", "Goodwin"):
                t = pct(r.period_err, 1)
            else:
                t = pct(r.tac_err, 0)
            L.append("%s & %s & %s & %s & %s & %s & %s \\\\" % ((m.replace("p53", "p53--Mdm2") + ", $s=%g$" % s_) if meth == "pdmp" else "",
                     "$h$" if meth == "pdmp" else "$h/4$", pct(r.mean_err, 1), pct(r.cv_err, 1), rnd(r.w1, 3), rnd(r.var_ratio, 3), t))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L)


def arch_table(ar):
    L = [r"{\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{lrrllr}",
         r"\caption{Site-resolved architecture against the concerted one (H) and the exact-response controls matched in worst-case $\eta$ (E) and in zero-frequency noise strength at the deterministic fixed point (M), SSA ensembles A, at every condition: relative difference in mean and CV (S relative to the reference, per cent, with 95\% trajectory-bootstrap intervals) and in the temporal observable (period for the repressilator, autocorrelation time for the stationary circuits). For the toggle switch the last column gives the number of basin changes in the reference and in the S ensemble started in one basin; escape rates are in Table~\ref{tab:si_toggle}.}\label{tab:si_arch}\\",
         r"\toprule System & $s$ & $\Omega$ & mean & CV & temporal \\ \midrule\endfirsthead",
         r"\toprule System & $s$ & $\Omega$ & mean & CV & temporal \\ \midrule\endhead"]
    ci = lambda r, k: "%s [%s, %s]" % (pct(r[k], 1), pct(r[k + "_lo"], 1), pct(r[k + "_hi"], 1))
    for _, r in ar.sort_values(["system", "ref", "omega", "s"]).iterrows():
        if r.system == "Toggle":
            t = "%d; %d" % (r.ssa_switch_A, r.ssa_switch_M)
        elif r.system == "Repressilator":
            t = pct(r.ssa_period_err, 1)
        else:
            t = pct(r.ssa_tac_err, 0)
        L.append("%s & %s & %s & %s & %s & %s \\\\" % (r.system.replace("p53", "p53--Mdm2") + " (S vs %s)" % r.ref, g(r.s, 3), g(r.omega, 3),
                 ci(r, "ssa_mean_err"), ci(r, "ssa_cv_err"), t))
    L += [r"\bottomrule", r"\end{longtable}}"]
    return "\n".join(L)


def arch_summary(ar, df):
    """S against each reference in the stationary circuits, split by the
    operating-point eta of the S promoter: largest |difference| in mean and CV,
    and the number of conditions whose 95% interval excludes zero."""
    e = df[df.method == "pdmp"][["model", "s", "omega", "eta_op"]]
    e = e[e.model.str.endswith("-S")].assign(system=lambda x: x.model.str.split("-").str[0]).drop(columns="model")
    x = ar[ar.system.isin(["p53", "NAR", "Cascade", "PFB"])].merge(e, on=["system", "s", "omega"])
    names = {"p53": "p53--Mdm2", "NAR": "neg.\\ autoregulation", "Cascade": "cascade", "PFB": "pos.\\ autoregulation"}
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Site-resolved promoter (S) against the concerted promoter (H) and the exact-response controls E and M in the stationary circuits, separately for conditions with operating-point $\eta$ of the S promoter below and above 1: number of conditions, largest absolute relative difference in mean and in CV (per cent), and in brackets the number of conditions whose 95\% interval excludes zero. Condition-by-condition values are in Table~\ref{tab:si_arch}.}",
         r"\label{tab:si_arch_summary}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{llrllrll}", r"\toprule",
         r" & & \multicolumn{3}{c}{$\eta_{\mathrm{op}}<1$} & \multicolumn{3}{c}{$\eta_{\mathrm{op}}>1$} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Circuit & ref. & $n$ & mean & CV & $n$ & mean & CV \\", r"\midrule"]
    def cell(g, k):
        if g.empty:
            return "---"
        sig = ((g[k + "_lo"] > 0) | (g[k + "_hi"] < 0)).sum()
        return "%s (%d)" % (rnd(100 * g[k].abs().max(), 0), sig)
    for sysn in ("NAR", "Cascade", "p53", "PFB"):
        for ref in ("M", "E", "H"):
            g = x[(x.system == sysn) & (x.ref == ref)]
            lo, hi = g[g.eta_op < 1], g[g.eta_op >= 1]
            L.append("%s & %s & %d & %s & %s & %d & %s & %s \\\\" % (names[sysn] if ref == "M" else "", ref,
                     len(lo), cell(lo, "ssa_mean_err"), cell(lo, "ssa_cv_err"),
                     len(hi), cell(hi, "ssa_mean_err"), cell(hi, "ssa_cv_err")))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L)


def lna_table(df):
    """RMS and largest deviation of the variance fraction kept by each reduction
    from the LNA prediction and from the Eq. S1 baseline, by circuit."""
    st = df[df.lna_share.notna()]
    p = st[st.method == "pdmp"].set_index(["model", "s", "omega"])
    q = st[st.method == "sqssa"].set_index(["model", "s", "omega"])
    nl = st[st.method == "ssa_null"].set_index(["model", "s", "omega"])
    e = pd.DataFrame(dict(system=p.system, share=p.lna_share, floor=nl.var_ratio - 1,
                          lh=p.var_ratio - p.lna_share, lq=q.var_ratio - (1 - q.lna_share),
                          bh=p.var_ratio - p.s1_share, bq=q.var_ratio - (1 - q.s1_share)))
    rm = lambda v: np.sqrt(np.mean(v ** 2))
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Monte Carlo references: deviation of the fraction of the SSA variance kept by the hybrid and the sQSSA from the LNA prediction and from the closed form of Eq.~\ref{eq:si_cv_decomposition} evaluated at the operating point (S1), by circuit, over all conditions and architectures (H, S, E, M) and over conditions with LNA promoter share between 0.2 and 0.8: root mean square (largest absolute value). SSA$_B$: the same statistic for the variance ratio of the two SSA ensembles minus one, the Monte Carlo noise floor of the reference.}",
         r"\label{tab:si_lna}", r"\small", r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{lrlllll}", r"\toprule",
         r"Circuit & $n$ & LNA, hybrid & LNA, sQSSA & S1, hybrid & S1, sQSSA & SSA$_B$ \\", r"\midrule"]
    cell = lambda v: "%s (%s)" % (rnd(rm(v), 3), rnd(v.abs().max(), 2))
    names = {"p53": "p53--Mdm2", "NAR": "negative autoregulation", "Cascade": "cascade", "PFB": "positive autoregulation"}
    for label, sub in (("all", e), ("share 0.2--0.8", e[(e.share > 0.2) & (e.share < 0.8)])):
        L.append(r"\multicolumn{7}{l}{\emph{%s}} \\" % label)
        for sysn in ("NAR", "Cascade", "p53", "PFB"):
            x = sub[sub.system == sysn]
            if x.empty:
                continue
            L.append("%s & %d & %s & %s & %s & %s & %s \\\\" % (names[sysn], len(x), cell(x.lh), cell(x.lq), cell(x.bh), cell(x.bq), cell(x.floor)))
        x = sub[sub.system != "PFB"]
        L.append("first three & %d & %s & %s & %s & %s & %s \\\\" % (len(x), cell(x.lh), cell(x.lq), cell(x.bh), cell(x.bq), cell(x.floor)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    return "\n".join(L)


def fsp_table(df):
    f = pd.read_csv(os.path.join(RES, "fsp_stationary.csv"))
    m = df[df.method.isin(["pdmp", "sqssa"])].pivot_table(index=["model", "s", "omega"], columns="method",
                                                        values="var_ratio").reset_index()
    b = df[df.method == "pdmp"][["model", "s", "omega", "system", "lna_share", "ssa_var", "N"]].rename(
        columns={"ssa_var": "mc_var", "N": "mc_mean"})
    x = f.merge(b, on=["model", "s", "omega", "system"]).merge(m, on=["model", "s", "omega"])
    x["kq_exact"] = x.sqssa_var / x.ssa_var
    x["kh"] = x.pdmp * x.mc_var / x.ssa_var               # hybrid variance (MC) over exact SSA variance
    ex = pd.read_csv(os.path.join(RES, "exact_reference.csv"))[["model", "s", "omega", "pdmp_var_ratio"]]
    x = x.merge(ex, on=["model", "s", "omega"], how="left")
    x["kh"] = x.pdmp_var_ratio.fillna(x.kh)               # exact hybrid where available (autoregulated genes)
    x["lq"] = x.kq_exact - (1 - x.lna_share)
    x["lh"] = x.kh - x.lna_share
    x["mcq"] = x.sqssa - x.kq_exact                        # Monte Carlo error of the sQSSA fraction
    x["mcv"] = x.mc_var / x.ssa_var - 1                    # Monte Carlo error of the SSA variance
    x["mcm"] = x.mc_mean / (x.ssa_mean * x.omega) - 1
    lv = df[df.method == "pdmp"][["model", "s", "omega", "lna_var"]]
    x = x.merge(lv, on=["model", "s", "omega"])
    x["lv"] = x.lna_var / x.ssa_var - 1                    # LNA variance relative to exact variance
    rm = lambda v: np.sqrt(np.mean(v ** 2))
    cell = lambda v: "%s (%s)" % (rnd(rm(v), 3), rnd(v.abs().max(), 2))
    names = {"NAR": "negative autoregulation", "Cascade": "cascade", "PFB": "positive autoregulation"}
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Exact stationary references from the finite state projection (FSP), by circuit, over the architectures H, S, E and M and the conditions computed (all eleven for the autoregulated genes, the nine with $\Omega\le3$ for the cascade), and over those with LNA promoter share between 0.2 and 0.8. Columns: deviation of the exact fraction of the variance kept by the sQSSA from the LNA prediction; deviation of the hybrid fraction from the LNA prediction (exact hybrid variance from the discretised forward equation for the autoregulated genes, Monte Carlo hybrid variance for the cascade, both over the exact SSA variance); Monte Carlo error of the sQSSA fraction; Monte Carlo error of the SSA variance and of the SSA mean (relative); LNA variance relative to the exact variance. Root mean square (largest absolute value).}",
         r"\label{tab:si_fsp}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{lrllllll}", r"\toprule",
         r"Circuit & $n$ & sQSSA$-$LNA & hybrid$-$LNA & MC, sQSSA & MC, SSA var. & MC, SSA mean & LNA var. \\", r"\midrule"]
    for label, sub in (("all", x), ("share 0.2--0.8", x[(x.lna_share > 0.2) & (x.lna_share < 0.8)])):
        L.append(r"\multicolumn{8}{l}{\emph{%s}} \\" % label)
        for sysn in ("NAR", "Cascade", "PFB"):
            g = sub[sub.system == sysn]
            if g.empty:
                continue
            L.append("%s & %d & %s & %s & %s & %s & %s & %s \\\\" % (names[sysn], len(g), cell(g.lq), cell(g.lh),
                     cell(g.mcq), cell(g.mcv), cell(g.mcm), cell(g.lv)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    x.to_csv(os.path.join(RES, "fsp_vs_mc.csv"), index=False)
    return "\n".join(L)


def mdm2_table():
    """Variance of Mdm2 (product of the regulated promoter) kept by each
    description of the p53-Mdm2 loop, against the LNA prediction for Mdm2."""
    m = pd.read_csv(os.path.join(RES, "p53_mdm2.csv"))
    d = pd.read_csv(os.path.join(RES, "suite2_metrics.csv"))
    L_ = d[(d.system == "p53") & (d.method == "pdmp")][["model", "s", "omega", "lna_share", "lna_share_mdm2", "eta_op"]]
    L = [r"{\scriptsize\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{lrrrrrr}",
         r"\caption{p53--Mdm2 loop: fraction of the SSA variance of Mdm2, the product of the regulated promoter, kept by the hybrid and the sQSSA, with 95\% trajectory-bootstrap intervals, against the LNA promoter share for Mdm2.}\label{tab:si_mdm2}\\",
         r"\toprule Arch. & $s$ & $\Omega$ & $\eta_{\mathrm{op}}$ & LNA & hybrid & sQSSA \\ \midrule\endfirsthead",
         r"\toprule Arch. & $s$ & $\Omega$ & $\eta_{\mathrm{op}}$ & LNA & hybrid & sQSSA \\ \midrule\endhead"]
    for _, r in L_.sort_values(["model", "omega", "s"]).iterrows():
        x = m[(m.model == r.model) & (m.s == r.s) & (m.omega == r.omega)].set_index("method")
        c = lambda k: "%s [%s, %s]" % (rnd(x.loc[k, "var_ratio"], 3), rnd(x.loc[k, "lo"], 3), rnd(x.loc[k, "hi"], 3))
        L.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (r.model[-1], g(r.s, 3), g(r.omega, 3), g(r.eta_op),
                 rnd(r.lna_share_mdm2, 3), c("pdmp"), c("sqssa")))
    L += [r"\bottomrule", r"\end{longtable}}"]
    return "\n".join(L)


def cost(df):
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Single-core wall-clock time per trajectory at the base condition. The hybrid and ODE use a fixed step chosen for accuracy, so these times describe this implementation only.}",
         r"\label{tab:si_cost}", r"\small", r"\begin{tabular}{lrrr}", r"\toprule",
         r"Model & SSA (s) & hybrid (s) & sQSSA (s) \\", r"\midrule"]
    for m in MODELS:
        sysn = m.split("-")[0]
        d = df[(df.model == m) & (df.s == 1) & (df.omega == BASE_OMEGA[sysn])]
        if d.empty:
            continue
        w = {x: d[d.method == x].wall.values[0] for x in ("pdmp", "sqssa")}
        L.append("%s & %s & %s & %s \\\\" % (m.replace("p53", "p53--Mdm2"), g(d.wall_ref.values[0], 3), g(w["pdmp"], 3), g(w["sqssa"], 3)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


if __name__ == "__main__":
    out = sys.argv[1]
    df = pd.read_csv(os.path.join(RES, "suite2_metrics.csv"))
    ar = pd.read_csv(os.path.join(RES, "suite2_architecture.csv"))
    with open(os.path.join(out, "grid_tables.tex"), "w") as f:
        for m in MODELS:
            f.write(model_table(df, m) + "\n\n")
    open(os.path.join(out, "hcheck_table.tex"), "w").write(hcheck(df) + "\n")
    open(os.path.join(out, "arch_table.tex"), "w").write(arch_table(ar) + "\n")
    open(os.path.join(out, "arch_summary.tex"), "w").write(arch_summary(ar, df) + "\n")
    open(os.path.join(out, "cost_table.tex"), "w").write(cost(df) + "\n")
    open(os.path.join(out, "lna_table.tex"), "w").write(lna_table(df) + "\n")
    open(os.path.join(out, "mdm2_table.tex"), "w").write(mdm2_table() + "\n")
    if os.path.exists(os.path.join(RES, "fsp_stationary.csv")):
        open(os.path.join(out, "fsp_table.tex"), "w").write(fsp_table(df) + "\n")
    for f in ("switch_table.tex",):
        p = os.path.join(out, f)
        if os.path.exists(p):
            os.remove(p)
    print("ok")
