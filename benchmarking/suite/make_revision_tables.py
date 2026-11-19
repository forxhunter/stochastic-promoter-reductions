#!/usr/bin/env python3
"""LaTeX tables of the revision analyses for the Supplementary Material, from
benchmarking_results/revision/*.csv. Writes into OUTDIR (the SI generated/ folder).
The caption of every table is highlighted in the marked version (\\rev)."""
import os, sys, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results", "revision")
PN = {"NAR-S": "NAR, \\arch{S}", "PFB-S": "PFB, \\arch{S}", "NAR-X": "NAR, \\arch{X}"}
CN = {"F": "\\arch{F}", "L2": "\\arch{L2}", "L1": "\\arch{L1}", "M": "\\arch{M}", "C": "\\arch{L0}"}


def f(v, d=2):
    if pd.isna(v):
        return "---"
    return ("%+." + str(d) + "f") % v


def highlight_caption(body):
    """Wrap the caption text in \\rev. Highlighting the whole table breaks at the
    page breaks of a longtable, so a new table is marked through its caption."""
    i = body.index("\\caption{") + len("\\caption{")
    depth, j = 1, i
    while depth:
        depth += {"{": 1, "}": -1}.get(body[j], 0)
        j += 1
    text = body[i:j - 1]
    return body[:i - 1] + "[" + text.replace("[", "{[}").replace("]", "{]}") + "]{\\rev{" + text + "}}" + body[j:]


def write(out, name, body):
    with open(os.path.join(out, name), "w") as fh:
        fh.write(highlight_caption(body))


def boundary(out):
    b = pd.read_csv(os.path.join(RES, "boundary_exponent.csv"))
    p = b.pivot_table(index=["promoter", "boundary", "n_group", "k_exit", "s", "theta_pred"],
                      columns="cells_per_decade", values="theta_fit").reset_index()
    p["rich"] = 2 * p[400] - p[200]
    lines = []
    for _, r in p.iterrows():
        lines.append("%s & %s & %d & %.4g & %g & %.4f & %.4f & %.4f & %.4f \\\\" % (
            PN[r.promoter], r.boundary, r.n_group, r.k_exit, r.s, r.theta_pred, r[100], r[400], r.rich))
    body = r"""\begin{table}[!htbp]
\centering
\caption{Boundary exponent of the stationary hybrid density (Section~\ref{si:boundary}). $\theta$: predicted exponent $sk_{\mathrm{exit}}/\gamma-1$, with $k_{\mathrm{exit}}$ the Perron eigenvalue of $-\mathbf Q_{gg}$ at the boundary and $|g|$ the number of states whose flow vanishes there. Fitted exponents from upwind finite volumes with 100 and 400 cells per decade, and the Richardson extrapolation $2\theta_{400}-\theta_{200}$ from 200 and 400 cells per decade ($\theta_{200}$ not shown). At the lower boundary of \arch{X}, $\mathbf Q_{gg}(0)$ is triangular (reducible), and the predicted value is that of its slowest class, which the fit range does not reach.}
\label{tab:si_boundary_exp}
\footnotesize
\begin{tabular}{llrrrrrrr}
\toprule
promoter & boundary & $|g|$ & $k_{\mathrm{exit}}$ & $s$ & $\theta$ & fit, 100 & fit, 400 & Richardson \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "rev_boundary_table.tex", body)
    irr = p[~((p.promoter == "NAR-X") & (p.boundary == "lower"))]
    rel = ((irr.rich - irr.theta_pred).abs() / irr.theta_pred.abs().clip(lower=0.05))
    print("boundary: irreducible cases, max |Richardson - theta| / max(|theta|,0.05) = %.4f; max abs %.4f" % (
        rel.max(), (irr.rich - irr.theta_pred).abs().max()))


def identity(out):
    d = pd.read_csv(os.path.join(RES, "identity_check.csv"))
    prof = pd.read_csv(os.path.join(RES, "residual_profiles.csv"))
    lines = []
    for name, g in prof.groupby("promoter"):
        g = g.iloc[20:-20]
        lines.append("$b=b_2-R$, %s & --- & 2 & %.1e & %.1e \\\\" % (
            PN[name], np.max(np.abs(g.b - (g.b2 - g.R))), np.max(np.abs(g.b))))
        lines.append("$b$ of \\arch{F} $=b_2$, %s & --- & 2 & %.1e & %.1e \\\\" % (
            PN[name], np.max(np.abs(g.b_F - g.b2)), np.max(np.abs(g.b))))
    ml = d[d.test == "multilevel identity"]
    lines.append("Eq.~\\eqref{eq:si_multilevel}, 20 random chains & %d--%d & %d--%d & %.1e & %.1f--%.1f \\\\" % (
        ml.m.min(), ml.m.max(), ml.levels.min(), ml.levels.max(), ml.max_abs_err.max(), ml.max_abs_b.min(), ml.max_abs_b.max()))
    lu = d[d.test != "multilevel identity"].iloc[0]
    lines.append("$R$ for a strongly lumpable promoter ($\\rho$ varies by %.2f) & 3 & 2 & %.1e & %.1f \\\\" % (
        lu.rho_range, lu.max_abs_err, lu.max_abs_b))
    body = r"""\begin{table}[!htbp]
\centering
\caption{Numerical checks of the drift decomposition (Proposition~1, Section~\ref{si:qss}) and of its multilevel form (Eq.~\eqref{eq:si_multilevel}). Largest absolute discrepancy on a grid of $x$ (derivatives by central differences) and largest $|b|$ on the same grid ($s=1$). NAR: negatively autoregulated gene, $\alpha=100$, $\beta=0$; random chains: $\alpha=100$, $\beta=5$, $x\in[5,40]$.}
\label{tab:si_identity}
\footnotesize
\begin{tabular}{lrrrr}
\toprule
check & states & levels & max discrepancy & max $|b|$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "rev_identity_table.tex", body)


def lamF0(out):
    d = pd.read_csv(os.path.join(RES, "lamF_origin.csv"))
    lines = []
    for (name, s), g in d.groupby(["promoter", "s"], sort=False):
        g = g.set_index("x0")
        row = "%s & %g & %.3f & " % (PN[name], s, g.p0_P.iloc[0])
        row += " & ".join("%+.3f" % g.loc[k, "dmean"] for k in ("limit", "1e-3", "1e-2", "1e-1", "0.5"))
        row += " & " + "%+.3f" % g.loc["limit", "dcv"] + " & %.3f" % (g.dcv.max() - g.dcv.min())
        lines.append(row + " \\\\")
    lim = d.groupby("promoter").lamF_limit.first()
    body = r"""\begin{table}[!htbp]
\centering
\caption{Sensitivity of the comparison between a promoter and its control \arch{F} to the relaxation rate of \arch{F} at $X=0$. Mean difference (per cent) with $\lambda_F(0)$ set to its limit (%s) or to $\lambda_F(x_0)$; $P(X=0)$ of the promoter; CV difference with the limit and its range over the five choices. Exact stationary distributions, $\Omega=1$.}
\label{tab:si_lamF0}
\footnotesize
\begin{tabular}{lrrrrrrrrr}
\toprule
 & & & \multicolumn{5}{c}{mean(test)/mean(F)$-1$ (\%%), $x_0$:} & \multicolumn{2}{c}{CV (\%%)} \\
\cmidrule(lr){4-8}\cmidrule(lr){9-10}
promoter & $s$ & $P(X=0)$ & limit & $10^{-3}$ & $10^{-2}$ & $10^{-1}$ & 0.5 & limit & range \\
\midrule
""" % "; ".join("%s: %.4f" % (PN[k], v) for k, v in lim.items()) + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "rev_lamF0_table.tex", body)


def controls(out):
    d = pd.read_csv(os.path.join(RES, "reduction_exact.csv"))
    d1 = d[d.omega == 1.0]
    order = ["F", "L2", "L1", "M", "C"]
    lines = []
    for name in ("NAR-S", "PFB-S", "NAR-X"):
        g = d1[d1.promoter == name]
        if g.empty:
            continue
        c1 = g.groupby("control").c1.first()
        lines.append("%s & $c_1$ & " % PN[name] + " & ".join(f(c1.get(c, np.nan), 2) for c in order) + " & & & & \\\\")
        for s, h in g.groupby("s"):
            h = h.set_index("control")
            lines.append(" & %g & " % s + " & ".join(f(h.dmean.get(c, np.nan), 2) for c in order) + " & " +
                         " & ".join("%.3f" % h.eps_S0.get(c, np.nan) for c in order[1:]) + " \\\\")
        lines.append("\\midrule")
    body = r"""{\footnotesize
\begin{longtable}{llrrrrrrrrr}
\caption{How much of $S_0(x)$ must be matched. Exact differences in mean, mean(test)/mean(control)$-1$ (per cent), between a test promoter (\arch{S} or \arch{X}) and two-state controls with exactly its response: \arch{F} matches $S_0(x)$ at every $x$; \arch{L2} and \arch{L1} match $S_0$ and its first two or first derivatives at $x^*$; \arch{M} matches $S_0(x^*)$ with $\lambda\propto B(x)$; \arch{L0} matches $S_0(x^*)$ with constant $\lambda$. First row of each block: strict first-order coefficient $c_1$ (the mean difference is $c_1/s$ to first order). Right: $S_0$ mismatch over the stationary distribution of the test promoter, $\langle|\ln(S_0^{\mathrm C}/S_0^{\mathrm{test}})|\rangle_{\mathrm{test}}$ (zero for \arch{F}). $\Omega=1$.}\label{tab:si_controls}\\
\toprule
 & & \multicolumn{5}{c}{mean difference (\%)} & \multicolumn{4}{c}{$S_0$ mismatch} \\
promoter & $s$ & \arch{F} & \arch{L2} & \arch{L1} & \arch{M} & \arch{L0} & \arch{L2} & \arch{L1} & \arch{M} & \arch{L0} \\
\midrule\endfirsthead
\toprule promoter & $s$ & \arch{F} & \arch{L2} & \arch{L1} & \arch{M} & \arch{L0} & \arch{L2} & \arch{L1} & \arch{M} & \arch{L0} \\ \midrule\endhead
""" + "\n".join(lines[:-1]) + r"""
\bottomrule
\end{longtable}}
"""
    write(out, "rev_controls_table.tex", body)


def p_vs_F(out):
    d = pd.read_csv(os.path.join(RES, "reduction_exact.csv"))
    d = d[(d.omega == 1.0) & (d.control == "F")]
    lines = []
    for name in ("NAR-S", "PFB-S", "NAR-X"):
        g = d[d.promoter == name].sort_values("s")
        for _, r in g.iterrows():
            lines.append("%s & %g & %s & %s & %s & %.3f & %.3f & %s & %s \\\\" % (
                PN[name], r.s, f(r.dmean), f(r.s * r.dmean), f(r.dcv), r.W1, r.TV, f(r.dtau_int), f(r.dtau_e)))
        if not g.empty:
            lines.append("\\midrule")
    body = r"""{\footnotesize
\begin{longtable}{llrrrrrrr}
\caption{Test promoter against its control \arch{F} (same $\bar r(x)$ and $S_0(x)$): differences at the three levels of agreement. Mean and CV differences (per cent), $s$ times the mean difference (to be compared with $c_1$ of Table~\ref{tab:si_controls}), Wasserstein-1 distance divided by the SD of the test promoter and total-variation distance between the stationary distributions, and differences in the integrated and $1/e$ autocorrelation times of the copy number, test/F$-1$ (per cent), from the generator. Exact, $\Omega=1$.}\label{tab:si_PF}\\
\toprule
promoter & $s$ & mean & $s\times$mean & CV & W1/SD & TV & $\tau_{\mathrm{int}}$ & $\tau_{1/e}$ \\
\midrule\endfirsthead
\toprule promoter & $s$ & mean & $s\times$mean & CV & W1/SD & TV & $\tau_{\mathrm{int}}$ & $\tau_{1/e}$ \\ \midrule\endhead
""" + "\n".join(lines[:-1]) + r"""
\bottomrule
\end{longtable}}
"""
    write(out, "rev_PF_table.tex", body)


def omega(out):
    d = pd.read_csv(os.path.join(RES, "reduction_exact.csv"))
    d = d[d.s.isin([1.0, 10.0])]
    order = ["F", "L2", "L1", "M", "C"]
    lines = []
    for name in ("NAR-S", "PFB-S", "NAR-X"):
        g = d[d.promoter == name]
        for (s, om), h in g.groupby(["s", "omega"]):
            h = h.set_index("control")
            lines.append("%s & %g & %g & " % (PN[name], s, om) + " & ".join(f(h.dmean.get(c, np.nan)) for c in order) + " \\\\")
        if not g.empty:
            lines.append("\\midrule")
    body = r"""{\footnotesize
\begin{longtable}{lllrrrrr}
\caption{Dependence of the architecture differences on system size: mean(test)/mean(control)$-1$ (per cent) at $s=1$ and $s=10$ for $\Omega$ from 0.3 to 30 (exact). The first-order theory predicts that the differences do not depend on $\Omega$ where it holds (main text Section~2.4).}\label{tab:si_omega}\\
\toprule
promoter & $s$ & $\Omega$ & \arch{F} & \arch{L2} & \arch{L1} & \arch{M} & \arch{L0} \\
\midrule\endfirsthead
\toprule promoter & $s$ & $\Omega$ & \arch{F} & \arch{L2} & \arch{L1} & \arch{M} & \arch{L0} \\ \midrule\endhead
""" + "\n".join(lines[:-1]) + r"""
\bottomrule
\end{longtable}}
"""
    write(out, "rev_omega_table.tex", body)


def eta_table(out):
    e = pd.read_csv(os.path.join(RES, "eta_range.csv"))
    name = {"p53": "p53--Mdm2", "NAR": "NAR", "Cascade": "cascade", "PFB": "PFB"}
    lines = []
    for _, r in e.iterrows():
        sysn, a = r.model.split("-")
        lines.append("%s & \\arch{%s} & %s & %.3g & %.3g & %.3g & %.3g & %.3g & %.3g & %.3g & %.3g & %.2f \\\\" % (
            name[sysn], a, r.regulator.replace("p53", "p53"), r.x_p5, r.x_p50, r.x_p95, r.eta_p5, r.eta_p50,
            r.eta_p95, r.eta_op, r.eta_x0, r.eta_ratio_p95_p5))
    body = r"""\begin{table}[!htbp]
\centering
\caption{Timescale ratio over the visited regulator range at the base parameters ($s=1$, base $\Omega$). Percentiles of the stationary regulator distribution (SSA reference ensemble, concentration units), $\eta$ evaluated there, at the time-averaged level ($\eta_{\mathrm{op}}$) and at zero regulator, and the ratio of $\eta$ at the 5th and 95th percentiles.}
\label{tab:si_eta_range}
\footnotesize
\begin{tabular}{lllrrrrrrrrr}
\toprule
circuit & arch. & regulator & $x_{5\%}$ & $x_{50\%}$ & $x_{95\%}$ & $\eta(x_{5\%})$ & $\eta(x_{50\%})$ & $\eta(x_{95\%})$ & $\eta_{\mathrm{op}}$ & $\eta(0)$ & ratio \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "rev_eta_table.tex", body)


def toggle_trunc(out):
    t = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RES, "toggle_truncation_*.csv"))])
    lines = []
    for (a, m, s, om), g in t.groupby(["arch", "method", "s", "omega"]):
        g = g.sort_values("k")
        ref = g.iloc[-1]
        chU = (g.k_U / ref.k_U - 1).abs().max(); chV = (g.k_V / ref.k_V - 1).abs().max()
        refi = max((g.k_U_refined / g.k_U - 1).abs().max(), (g.k_V_refined / g.k_V - 1).abs().max())
        lines.append("\\arch{%s} & %s & %g & %g & %s & %.3g & %.3g & %.0e & %.0e & %.0e & %.0e \\\\" % (
            a, "SSA" if m == "ssa" else "sQSSA", s, om, ",".join(str(int(k)) for k in g.k), ref.k_U, ref.k_V,
            max(chU, chV), refi, g.edge.iloc[0], g.edge.iloc[-1]))
    body = r"""\begin{table}[!htbp]
\centering
\caption{Truncation and solver checks of the exact toggle escape rates (label-flip rates). The upper bound of the copy-number box is $b+k\sqrt b+12$, with $b$ the sum of the production rates of the species over all promoter states ($k=7$ in the submitted results). $k_U$, $k_V$: rates at the largest $k$; largest relative change of either rate over the values of $k$; largest relative change under one step of iterative refinement of the stationary vector; probability at the edge of the box for the smallest and largest $k$.}
\label{tab:si_toggle_trunc}
\footnotesize
\begin{tabular}{llrrlrrrrrr}
\toprule
arch. & method & $s$ & $\Omega$ & $k$ & $k_U$ & $k_V$ & change over $k$ & refinement & edge, min $k$ & edge, max $k$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    write(out, "rev_toggle_trunc_table.tex", body)
    print("toggle truncation: max change over k %.1e, max refinement change %.1e, cases %d" % (
        max((g.k_U / g.sort_values('k').k_U.iloc[-1] - 1).abs().max() for _, g in t.groupby(["arch", "method", "s", "omega"])),
        ((t.k_U_refined / t.k_U - 1).abs().max()), t.groupby(["arch", "method", "s", "omega"]).ngroups))


def toggle_spec(out):
    d = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(RES, "toggle_spectral_*.csv"))])
    d = d.sort_values(["omega", "s", "arch"])
    lines = []
    for _, r in d.iterrows():
        lines.append("\\arch{%s} & %g & %g & %.3g & %.0f & %.3f & %.3g & %.3g & %.2f & %.3g & %.3g & %.2f \\\\" % (
            r.arch, r.s, r.omega, -r.lam1_re, r.separation, r.P_U_soft, r.k_U_label, r.k_U_met, r.ratio_U,
            r.k_V_label, r.k_V_met, r.ratio_V))
    body = r"""{\footnotesize
\begin{longtable}{lrrrrrrrrrrr}
\caption{Label-flip rates and metastable escape rates of the toggle switch (full model, exact). $|\lambda_1|$: slowest non-zero eigenvalue of the generator; separation: $|\mathrm{Re}\,\lambda_2|/|\lambda_1|$; $P_U$: probability of the $U$-high basin from the eigenvector $\psi_1$. Metastable rates $k_U^{\mathrm{met}}=|\lambda_1|(1-P_U)$, $k_V^{\mathrm{met}}=|\lambda_1|P_U$; label-flip rates as in Table~\ref{tab:si_toggle_exact}; ratio: label/metastable.}\label{tab:si_toggle_spec}\\
\toprule
arch. & $s$ & $\Omega$ & $|\lambda_1|$ & sep. & $P_U$ & $k_U$ label & $k_U^{\mathrm{met}}$ & ratio & $k_V$ label & $k_V^{\mathrm{met}}$ & ratio \\
\midrule\endfirsthead
\toprule arch. & $s$ & $\Omega$ & $|\lambda_1|$ & sep. & $P_U$ & $k_U$ label & $k_U^{\mathrm{met}}$ & ratio & $k_V$ label & $k_V^{\mathrm{met}}$ & ratio \\ \midrule\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}}
"""
    write(out, "rev_toggle_spec_table.tex", body)


def c1_scan(out):
    d = pd.read_csv(os.path.join(RES, "c1_scan.csv"))
    lines = []
    for _, r in d.iterrows():
        arch = ("sites, %s" % ("all" if r.rule == "all" else "any")) if r.family == "site" else "refractory"
        om = "---" if pd.isna(r.omega) else "%g" % r.omega
        lines.append("%s & %s & %d & %s & %.1f & %.2f & %.1f & %.3f & %+.2f & %+.2f \\\\" % (
            r.circuit, arch, r.n, om, r.x_star, r.p_intermediate, r.lam_ratio if np.isfinite(r.lam_ratio) else float("nan"),
            r.R_over_b, r.R, r.c1_F))
    body = r"""{\footnotesize
\begin{longtable}{llrrrrrrrr}
\caption{First-order coefficient $c_1=100R(x^*)/(\bar F'(x^*)x^*)$ of a promoter against its control \arch{F} across architectures (local expansion, no simulation; Section~\ref{si:c1scan}). Sites: $n$ identical sites with pairwise cooperativity $\omega$; ``all'': the output changes only when all sites are occupied (repression in NAR, activation in PFB; the benchmark promoter \arch{S} is NAR or PFB, all, $n=2$, $\omega=100$); ``any'': it changes as soon as one site is occupied (\arch{X} is NAR, any, $n=3$, $\omega=3$). Refractory: the promoters of Table~\ref{tab:si_family2}, $n$ stages. Every site promoter has half-maximum 30 and slowest relaxation rate 1 over $0\le x\le150$. $x^*$: fixed point; $p_{\mathrm{int}}$: probability of the partially occupied states at $x^*$; $\lambda_3/\lambda_2$ at $x^*$; $R/|b|$: residual relative to the noise-induced drift at $x^*$; $R$ at $s=1$.}\label{tab:si_c1scan}\\
\toprule
circuit & promoter & $n$ & $\omega$ & $x^*$ & $p_{\mathrm{int}}$ & $\lambda_3/\lambda_2$ & $R/|b|$ & $R$ & $c_1$ \\
\midrule\endfirsthead
\toprule circuit & promoter & $n$ & $\omega$ & $x^*$ & $p_{\mathrm{int}}$ & $\lambda_3/\lambda_2$ & $R/|b|$ & $R$ & $c_1$ \\ \midrule\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}}
"""
    write(out, "rev_c1scan_table.tex", body)


def spectral_check(out):
    d = pd.read_csv(os.path.join(RES, "toggle_spectral_check.csv"))
    lines = []
    for (a, om), g in d.groupby(["arch", "omega"], sort=False):
        g = g.sort_values("k"); ref = g.iloc[-1]
        for _, r in g.iterrows():
            dref = max(abs(r.lam1_refined / r.lam1 - 1), abs(r.P_U_refined / r.P_U - 1),
                       abs(r.k_U_met_refined / r.k_U_met - 1), abs(r.k_V_met_refined / r.k_V_met - 1))
            dk = max(abs(r.k_U_met / ref.k_U_met - 1), abs(r.k_V_met / ref.k_V_met - 1), abs(r.lam1 / ref.lam1 - 1))
            lines.append("\\arch{%s} & %g & %d & %d & %.6g & %.0e & %.0e & %.0e & %.0e & %.4g & %.4g & %.0e \\\\" % (
                a, om, r.k, r.n_states, -r.lam1, abs(r.lam1_left / r.lam1 - 1), r.backward_err, r.res_right_refined,
                dref, 1e4 * r.k_U_met, 1e4 * r.k_V_met, dk))
    body = r"""{\footnotesize
\begin{longtable}{lrrrrrrrrrrr}
\caption{Convergence checks of the metastable escape rates of the toggle switch ($s=1$, full model; Section~\ref{si:toggle_checks}). For each box size $k$ (upper bound $b+k\sqrt b+12$): number of states; $|\lambda_1|$; relative difference between the eigenvalues of the right and the left shift-invert solves; backward error $\|\mathbf Q\psi-\lambda_1\psi\|_1/(\|\mathbf Q\|_1\|\psi\|_1)$ of the right eigenvector; its relative residual $\|\mathbf Q\psi-\lambda_1\psi\|/(|\lambda_1|\|\psi\|)$ after two steps of inverse iteration; the largest relative change of $\lambda_1$, $P_U$ and the rates under these two steps; the metastable rates (units of $10^{-4}$ per unit time); and their largest relative change against the largest box.}\label{tab:si_spec_check}\\
\toprule
arch. & $\Omega$ & $k$ & states & $|\lambda_1|$ & left/right & backward & residual & refinement & $k_U^{\mathrm{met}}$ & $k_V^{\mathrm{met}}$ & change \\
\midrule\endfirsthead
\toprule arch. & $\Omega$ & $k$ & states & $|\lambda_1|$ & left/right & backward & residual & refinement & $k_U^{\mathrm{met}}$ & $k_V^{\mathrm{met}}$ & change \\ \midrule\endhead
""" + "\n".join(lines) + r"""
\bottomrule
\end{longtable}}
"""
    write(out, "rev_spec_check_table.tex", body)

if __name__ == "__main__":
    out = sys.argv[1]
    for fn in (boundary, identity, lamF0, controls, p_vs_F, omega, eta_table, toggle_trunc, toggle_spec,
               c1_scan, spectral_check):
        try:
            fn(out)
        except Exception as e:
            print("FAILED", fn.__name__, repr(e))
