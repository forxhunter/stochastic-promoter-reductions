#!/usr/bin/env python3
"""Main-text table of toggle-switch escape rates (toggle_switching.csv)."""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")


def f(v, lo=None, hi=None):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(v)))
    m = v / 10 ** e
    s = r"$%.1f\times10^{%d}$" % (m, e)
    if lo is not None:
        s += r" [%.1f, %.1f]" % (lo / 10 ** e, hi / 10 ** e)
    return s


def pooled_hybrid(arch, s_=1.0, oms=(0.3, 1, 3, 10, 30)):
    """Exact-flow hybrid rates pooled over the Omega runs. Without sequestration
    the hybrid does not depend on Omega (a fixed seed gives the same trajectory
    at every Omega), so the five runs are five independent samples."""
    from toggle_switching import OUT, basin_stats, rates
    T = []
    for om in oms:
        for st in "UV":
            f = os.path.join(OUT, "%s_pdmpx_h1_s%g_om%g_%s.npz" % (arch, s_, om, st))
            T.append(np.load(f)["traj"].astype(float))
    r = rates(basin_stats(np.concatenate(T)[:, :, 0], np.concatenate(T)[:, :, 1], int(50 / 0.5)), 0.5)
    # interval from the variance between the five independent runs (each run uses its own seeds)
    k = []
    for i in range(0, len(T), 2):
        Ti = np.concatenate(T[i:i + 2])
        k.append(rates(basin_stats(Ti[:, :, 0], Ti[:, :, 1], int(50 / 0.5)), 0.5, nb=1)["k_U"])
    se = np.std(k, ddof=1) / np.sqrt(len(k))
    r["k_U_run_lo"], r["k_U_run_hi"] = r["k_U"] - 2.776 * se, r["k_U"] + 2.776 * se
    return r


def cell(k, lo, hi, n):
    """Rate in units of 1e-4 with the number of exits; with fewer than three exits
    the exact Poisson (Garwood) 95% interval over the time spent in the basin is
    given instead of the rate."""
    from scipy.stats import chi2
    v = lambda x: ("%.0f" % (x * 1e4)) if x * 1e4 >= 100 else (("%.2g" % (x * 1e4)) if x * 1e4 >= 0.01 else ("%.1g" % (x * 1e4)))
    if n < 3 and n > 0:
        T = n / k
        glo, ghi = chi2.ppf(0.025, 2 * n) / (2 * T), chi2.ppf(0.975, 2 * n + 2) / (2 * T)
        return r"[%s, %s] (%d)" % (v(glo), v(ghi), n)
    return "%s (%d)" % (v(k), n)


if __name__ == "__main__":
    d = pd.read_csv(os.path.join(RES, "toggle_switching.csv"))
    cn = pd.read_csv(os.path.join(RES, "toggle_basin_copynumbers.csv"))
    rows = [(1, 0.3), (1, 1), (1, 3), (1, 10), (1, 30), (0.3, 1), (3, 1)]
    A = "HPQSEM"
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Toggle switch: rate of escape from the $U$-high basin per unit time, in units of $10^{-4}$, with the number of escapes observed in parentheses (basin-balanced ensembles, 64 trajectories started in each basin, $T=10^4$). Where fewer than three escapes were observed, the exact Poisson 95\% interval is given instead of the rate; intervals for every entry are in Supplementary Section S6. H: concerted promoter with falling-factorial counting; P: the same promoter with plain-power counting; Q: site-resolved with sequestration (binding removes a free $V$ molecule); S: site-resolved with catalytic binding; E: two-state control with exactly the site-resolved response, matched in worst-case $\eta$; M: the same control matched to S in zero-frequency promoter noise strength at the deterministic fixed point. $\langle V\rangle$: mean $V$ copy number in the $U$-high basin (SSA, S). Hybrid: exact-flow PDMP. Without sequestration the hybrid does not depend on $\Omega$ (with a common seed it gives the same trajectory at every $\Omega$); the production runs at the five values of $\Omega$ used different seeds and are five independent samples, pooled here at $s=1$ (640 trajectories); the rows at $s=0.3$ and 3 are single runs at $\Omega=1$. The Q hybrid at $\Omega\le1$ is omitted because a single binding event removes more than the free regulator concentration (Supplementary Section S6).}",
         r"\label{tab:toggle}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{llllllllr}", r"\toprule",
         r"$s$ & $\Omega$ & H & P & Q & S & E & M & $\langle V\rangle$ \\", r"\midrule",
         r"\multicolumn{9}{l}{SSA} \\"]
    get = lambda a_, m, s_, om: d[(d.arch == a_) & (d.method == m) & (d.s == s_) & (d.omega == om) & (d.hfac == 1)].iloc[0]
    for s_, om in rows:
        cells = [cell(r.k_U, r.k_U_lo, r.k_U_hi, r.exits_U) for r in (get(a_, "ssa", s_, om) for a_ in A)]
        vv = cn[(cn.arch == "S") & (cn.Omega == om)] if s_ == 1 else cn.iloc[0:0]
        cells.append("%.2g" % vv.V_mean_Uhigh.iloc[0] if len(vv) else "---")
        L.append("%g & %g & " % (s_, om) + " & ".join(cells) + r" \\")
    L += [r"\midrule", r"\multicolumn{9}{l}{hybrid (exact flow)} \\"]
    pooled = {a_: pooled_hybrid(a_) for a_ in A if a_ != "Q"}
    cells = [cell(pooled[a_]["k_U"], pooled[a_]["k_U_lo"], pooled[a_]["k_U_hi"], pooled[a_]["exits_U"]) if a_ != "Q" else "---" for a_ in A]
    L.append("1 & pooled & " + " & ".join(cells) + r" & \\")
    for om in (3, 10, 30):
        r = get("Q", "pdmpx", 1, om)
        cells = ["" if a_ != "Q" else cell(r.k_U, r.k_U_lo, r.k_U_hi, r.exits_U) for a_ in A]
        L.append("1 & %g & " % om + " & ".join(cells) + r" & \\")
    for s_ in (0.3, 3):
        cells = ["---" if a_ == "Q" else cell(r.k_U, r.k_U_lo, r.k_U_hi, r.exits_U)
                 for a_, r in ((a_, get(a_, "pdmpx", s_, 1)) for a_ in A)]
        L.append("%g & 1 & " % s_ + " & ".join(cells) + r" & \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(sys.argv[1], "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    P = pd.DataFrame({a_: {k: v for k, v in pooled[a_].items()} for a_ in pooled}).T
    P.to_csv(os.path.join(RES, "toggle_hybrid_pooled.csv"))
    print(P)


def si_table(d):
    L = [r"{\footnotesize\setlength{\tabcolsep}{3pt}", r"\begin{longtable}{llrrrrrr}",
         r"\caption{Toggle switch, every condition and method: escape rates from the $U$-high ($k_U$) and $V$-high ($k_V$) basins per unit time with 95\% trajectory-bootstrap intervals, number of exits from each basin, and fraction of time in the $U$-high basin. 64 trajectories started in each basin, $T=10^4$. $h$: hybrid step relative to the standard step; hybrid (exact): exact-flow sampler. No sQSSA of Q was computed (its promoter reactions change the free regulator; a slow-scale SSA with total $V$ as the slow variable exists but was not constructed). Rates estimated from fewer than three exits (H at $s=1$, $\Omega=0.3$ and at $s=3$, $\Omega=1$) are effectively upper bounds given by the upper end of their intervals.}\label{tab:si_toggle}\\",
         r"\toprule Arch. & method & $s$ & $\Omega$ & $k_U$ & $k_V$ & exits & $U$-high \\ \midrule\endfirsthead",
         r"\toprule Arch. & method & $s$ & $\Omega$ & $k_U$ & $k_V$ & exits & $U$-high \\ \midrule\endhead"]
    for _, r in d.sort_values(["arch", "omega", "s", "method", "hfac"]).iterrows():
        meth = {"pdmpx": "hybrid (exact)"}.get(r.method, r.method) if r.method != "pdmp" else ("hybrid" + ("" if r.hfac == 1 else ", $h/%d$" % round(1 / r.hfac)))
        L.append("%s & %s & %g & %g & %s & %s & %d, %d & %.2f \\\\" % (
            r.arch, meth, r.s, r.omega, f(r.k_U, r.k_U_lo, r.k_U_hi), f(r.k_V, r.k_V_lo, r.k_V_hi),
            r.exits_U, r.exits_V, r.frac_U))
    L += [r"\bottomrule", r"\end{longtable}}"]
    return "\n".join(L)


if __name__ == "__main__" and len(sys.argv) > 2:
    d = pd.read_csv(os.path.join(RES, "toggle_switching.csv"))
    open(sys.argv[2], "w").write(si_table(d) + "\n")
