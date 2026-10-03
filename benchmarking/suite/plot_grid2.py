#!/usr/bin/env python3
"""Figures for the second-design grid (benchmarking_results/suite2_metrics.csv,
suite2_architecture.csv)."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
from systems2 import BASE_OMEGA

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
COL = {"ssa_null": "0.6", "pdmp": "#2656a8", "sqssa": "#3a8a3a"}
LAB = {"ssa_null": "SSA vs SSA", "pdmp": "hybrid (PDMP)", "sqssa": "sQSSA"}
ORDER = ["Repressilator", "Goodwin", "p53", "NAR", "Cascade", "PFB"]
NAME = {"p53": "p53-Mdm2", "NAR": "neg. autoregulation", "Cascade": "cascade",
        "PFB": "pos. autoregulation"}
STAT = ["p53", "NAR", "Cascade", "PFB"]
SCOL = {"p53": "k", "NAR": "#2656a8", "Cascade": "#3a8a3a", "PFB": "#b2325e"}
ACOL = {"H": "#d0671f", "P": "#8c6d1f", "S": "k", "E": "#2656a8", "M": "#b2325e"}


def band(ax, d, x, y, col, ls, label):
    d = d.sort_values(x)
    ax.plot(d[x], d[y], color=col, ls=ls, marker="o", ms=3, lw=1.2, label=label)
    if y + "_lo" in d and y != "w1":
        ax.fill_between(d[x], d[y + "_lo"], d[y + "_hi"], color=col, alpha=0.15, lw=0)


def grid_figure(df, y, ylabel, fname, logy=True):
    fig, axs = plt.subplots(len(ORDER), 2, figsize=(8.5, 1.9 * len(ORDER)), sharey="row")
    for i, sysn in enumerate(ORDER):
        d = df[df.system == sysn]
        om0 = BASE_OMEGA[sysn]
        for arch, ls in (("H", "-"), ("S", "--")):
            for m in ("ssa_null", "pdmp", "sqssa"):
                dd = d[(d.arch == arch) & (d.method == m)]
                if dd.empty:
                    continue
                lab = "%s, %s" % (LAB[m], arch)
                band(axs[i, 0], dd[dd.omega == om0], "eta_wc", y, COL[m], ls, lab)
                band(axs[i, 1], dd[dd.s == 1], "omega", y, COL[m], ls, lab)
        axs[i, 0].set_ylabel("%s\n%s" % (NAME.get(sysn, sysn), ylabel), fontsize=10)
        for a in axs[i]:
            a.set_xscale("log")
            if logy:
                a.set_yscale("log")
            a.tick_params(labelsize=9)
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    axs[0, 0].set_title(r"promoter speed ($s$ sweep)", fontsize=9)
    axs[0, 1].set_title(r"system size ($\Omega$ sweep)", fontsize=9)
    axs[-1, 0].set_xlabel(r"$\eta$ (worst case)", fontsize=9)
    axs[-1, 1].set_xlabel(r"$\Omega$", fontsize=9)
    h, l = axs[2, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, fname))


def lna_figure(df, fname):
    """Variance fractions kept by the hybrid (top) and the sQSSA (bottom) against
    the LNA prediction, one column per circuit. Exact references where they exist
    (filled symbols: exact; open: simulation)."""
    d = df[df.method.isin(["pdmp", "sqssa"]) & df.lna_share.notna()].copy()
    d["exact"] = False
    ex = pd.read_csv(os.path.join(RES, "exact_reference.csv")).set_index(["model", "s", "omega"])
    fs = pd.read_csv(os.path.join(RES, "fsp_stationary.csv")).set_index(["model", "s", "omega"])
    for i, r in d.iterrows():
        k = (r.model, r.s, r.omega)
        if k in ex.index:
            d.loc[i, "var_ratio"] = ex.loc[k, "pdmp_var_ratio" if r.method == "pdmp" else "sqssa_var_ratio"]
            d.loc[i, "exact"] = True
        elif r.method == "sqssa" and k in fs.index:
            d.loc[i, "var_ratio"] = fs.loc[k, "sqssa_var"] / fs.loc[k, "ssa_var"]
            d.loc[i, "exact"] = True
    fig, axs = plt.subplots(2, 4, figsize=(11, 5.6), sharex=True, sharey=True)
    for c, sysn in enumerate(STAT):
        g = d[d.system == sysn]
        for arch in "HSEM":
            for row, meth in enumerate(("pdmp", "sqssa")):
                p = g[(g.method == meth) & (g.arch == arch)]
                xv = p.lna_share if meth == "pdmp" else 1 - p.lna_share
                for exact in (True, False):
                    sel = p.exact == exact
                    axs[row, c].scatter(xv[sel], p.var_ratio[sel], marker="o", s=18,
                                        facecolors=ACOL[arch] if exact else "none", edgecolors=ACOL[arch], linewidths=0.9)
        axs[0, c].set_title(NAME.get(sysn, sysn), fontsize=10)
        for row in range(2):
            a = axs[row, c]
            a.axvspan(0.2, 0.8, color="0.92", lw=0, zorder=0)
            a.plot([0, 1], [0, 1], "k--", lw=0.8)
            a.set_xlim(0, 1.05); a.set_ylim(0, 1.5)
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
        axs[1, c].set_xlabel("LNA share")
    axs[0, 0].set_ylabel("hybrid: Var(PDMP)/Var(SSA)")
    axs[1, 0].set_ylabel("sQSSA: Var(sQSSA)/Var(SSA)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc=ACOL[a], mec=ACOL[a], label=a) for a in "HSEM"]
    h += [Line2D([], [], ls="", marker="o", mfc="0.3", mec="0.3", label="exact"),
          Line2D([], [], ls="", marker="o", mfc="none", mec="0.3", label="simulation")]
    axs[0, 0].legend(handles=h, fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))


def arch_figure(ar, df, fname):
    """A: exact mean differences between S and two-state controls with the same
    response in the autoregulated genes, with the first-order QSS diffusion
    prediction; B: CV differences; C: repressilator period; D: toggle escape rates."""
    ex = pd.read_csv(os.path.join(RES, "architecture_exact.csv"))
    qs = pd.read_csv(os.path.join(RES, "qss_diffusion.csv"))
    fig, axs = plt.subplots(2, 2, figsize=(10, 7.6))
    eta = df[df.method == "pdmp"][["model", "s", "omega", "eta_op"]]
    ax = axs[0, 0]
    for sysn in ("NAR", "PFB"):
        e = eta[eta.model == sysn + "-S"][["s", "omega", "eta_op"]]
        x = ex[(ex.system == sysn) & (ex.omega == 1.0)].pivot_table(index="s", columns="arch", values="mean")
        q = qs[(qs.system == sysn) & (qs.omega == 1.0)].pivot_table(index="s", columns="arch", values="qss_mean")
        g = e[e.omega == 1.0].set_index("s").eta_op
        for ref, ls in (("D", "-"), ("M", "--")):
            y = 100 * (x["S"] / x[ref] - 1)
            ax.plot(g.reindex(y.index), y, color=SCOL[sysn], ls=ls, marker="o", ms=4, lw=1.4,
                    label="%s, S vs %s" % (NAME[sysn], ref))
        cf = pd.read_csv(os.path.join(RES, "control_F.csv"))
        f = cf[(cf.system == sysn) & (cf.omega == 1.0)].set_index("s")
        ax.plot(g.reindex(f.index), 100 * f.S_minus_F, color=SCOL[sysn], ls=":", marker="s", ms=3.5, lw=1.4,
                label="%s, S vs F" % NAME[sysn])
        c1 = pd.read_csv(os.path.join(RES, "first_order_coefficient.csv"))
        c1d = c1[(c1.system == sysn) & (c1.ref == "D")].c1.iloc[0]
        ss = np.array([1.0, 3.0, 10.0, 30.0])
        ax.plot(g.reindex(ss), c1d / ss, color=SCOL[sysn], ls="none", marker="x", ms=6)
    ax.plot([], [], color="0.3", ls="none", marker="x", label="first order, $c_1/s$ (vs D)")
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xscale("log")
    ax.set_xlabel(r"$\eta_{\mathrm{op}}$ of the S promoter"); ax.set_ylabel("mean(S)/mean(control) - 1  (%)")
    ax.set_title("autoregulated genes, exact ($\\Omega=1$)", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    ax = axs[0, 1]
    for sysn in ("p53", "Cascade"):
        om0 = BASE_OMEGA[sysn]
        e = eta[eta.model == sysn + "-S"][["s", "omega", "eta_op"]]
        g = ar[(ar.system == sysn) & (ar.ref == "M") & (ar.omega == om0)].merge(e, on=["s", "omega"]).sort_values("eta_op")
        ax.plot(g.eta_op, 100 * g.ssa_cv_err, marker="o", ms=4, lw=1.4, color=SCOL[sysn], label="%s, S vs M (SSA)" % NAME[sysn])
        ax.fill_between(g.eta_op, 100 * g.ssa_cv_err_lo, 100 * g.ssa_cv_err_hi, color=SCOL[sysn], alpha=0.15, lw=0)
    for sysn in ("NAR", "PFB"):
        e = eta[eta.model == sysn + "-S"][["s", "omega", "eta_op"]]
        x = ex[(ex.system == sysn) & (ex.omega == 1.0)].pivot_table(index="s", columns="arch", values="cv")
        g = e[e.omega == 1.0].set_index("s").eta_op
        y = 100 * (x["S"] / x["D"] - 1)
        ax.plot(g.reindex(y.index), y, color=SCOL[sysn], marker="o", ms=4, lw=1.4, label="%s, S vs D (exact)" % NAME[sysn])
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xscale("log")
    ax.set_xlabel(r"$\eta_{\mathrm{op}}$ of the S promoter"); ax.set_ylabel("CV(S)/CV(control) - 1  (%)")
    ax.set_title("noise, same response and matched $S_0$", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False)
    g = ar[(ar.system == "Repressilator") & (ar.ref == "H") & (ar.omega == BASE_OMEGA["Repressilator"])]
    e = df[(df.model == "Repressilator-S") & (df.method == "pdmp")][["s", "omega", "eta_wc"]]
    g = g.merge(e, on=["s", "omega"]).sort_values("eta_wc")
    ax = axs[1, 0]
    ax.plot(g.eta_wc, 100 * g.ssa_period_err, marker="o", ms=4, color="k")
    ax.fill_between(g.eta_wc, 100 * g.ssa_period_err_lo, 100 * g.ssa_period_err_hi, color="k", alpha=0.15, lw=0)
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xscale("log")
    ax.set_xlabel(r"$\eta_{\mathrm{wc}}$"); ax.set_ylabel("period(S)/period(H) - 1  (%)")
    ax.set_title("repressilator", fontsize=10)
    t = pd.read_csv(os.path.join(RES, "toggle_switching.csv"))
    pooled = pd.read_csv(os.path.join(RES, "toggle_hybrid_pooled.csv"), index_col=0)
    ax = axs[1, 1]
    ACOL2 = dict(ACOL, Q="#6a3d9a")
    for a in "HPQSEM":
        x = t[(t.arch == a) & (t.method == "ssa") & (t.s == 1)].sort_values("omega")
        ok = x.exits_U >= 3
        ax.errorbar(x.omega[ok], x.k_U[ok], yerr=[(x.k_U - x.k_U_lo)[ok], (x.k_U_hi - x.k_U)[ok]], color=ACOL2[a],
                    marker="o", ms=4, capsize=2, lw=1.4, label="SSA, %s" % a)
        if (~ok).any():
            ax.plot(x.omega[~ok], x.k_U_hi[~ok], color=ACOL2[a], marker="v", ls="none", ms=6)
    lo, hi = pooled.k_U.min(), pooled.k_U.max()
    ax.axhspan(lo, hi, color="0.7", alpha=0.4, lw=0, label="hybrid, H P S E M")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\Omega$"); ax.set_ylabel(r"escape rate from $U$-high basin")
    ax.set_title(r"toggle switch, $s=1$", fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    for a in axs.flat:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    for a, l in zip(axs.flat, "ABCD"):
        a.text(-0.14, 1.04, l, transform=a.transAxes, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))


def qss_figure(fname):
    """First-order diffusion approximation against exact for the mean difference
    between S and each two-state control, all conditions of both loops."""
    d = pd.read_csv(os.path.join(RES, "qss_diffusion.csv"))
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    mk = {"M": "s", "D": "o", "E": "^"}
    for sysn in ("NAR", "PFB"):
        for om, filled in ((1.0, True), (30.0, False)):
            x = d[(d.system == sysn) & (d.omega == om)].pivot_table(index="s", columns="arch",
                                                                    values=["qss_mean", "exact_mean"])
            for ref in "MDE":
                ex = 100 * (x["exact_mean"]["S"] / x["exact_mean"][ref] - 1)
                qs = 100 * (x["qss_mean"]["S"] / x["qss_mean"][ref] - 1)
                big = x.index >= 10
                for sel, alpha in ((big, 1.0), (~big, 0.45)):
                    ax.scatter(ex[sel], qs[sel], marker=mk[ref], s=26, alpha=alpha,
                               facecolors=SCOL[sysn] if filled else "none", edgecolors=SCOL[sysn], linewidths=0.9)
    lim = [-20, 50]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlim(lim); ax.set_ylim(-20, 75)
    ax.set_xlabel("exact: mean(S)/mean(control) - 1  (%)")
    ax.set_ylabel("quadrature of first-order density  (%)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", mfc=SCOL[k], mec=SCOL[k], label=NAME[k]) for k in ("NAR", "PFB")]
    h += [Line2D([], [], ls="", marker=mk[r], mfc="none", mec="k", label="vs " + r) for r in "MDE"]
    ax.legend(handles=h, fontsize=7.5, frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))


if __name__ == "__main__":
    OUT = sys.argv[1] if len(sys.argv) > 1 else RES
    df = pd.read_csv(os.path.join(RES, "suite2_metrics.csv"))
    df = df[df.method != "pdmp_h4"]
    # exact W1 for the two autoregulated genes (exact_w1.py)
    ew = pd.read_csv(os.path.join(RES, "exact_w1.csv")).set_index(["model", "s", "omega"])
    for i, r in df[df.method.isin(["pdmp", "sqssa"])].iterrows():
        k = (r.model, r.s, r.omega)
        if k in ew.index:
            df.loc[i, "w1"] = ew.loc[k, "w1_" + r.method]
    grid_figure(df, "w1", "W1/SD", "grid2_w1.pdf")
    grid_figure(df.assign(cv_abs=df.cv_err.abs(), cv_abs_lo=np.nan, cv_abs_hi=np.nan), "cv_abs", "|CV error|", "grid2_cv.pdf")
    lna_figure(df, "lna_partition.pdf")
    arch_figure(pd.read_csv(os.path.join(RES, "suite2_architecture.csv")), df, "architecture.pdf")
    qss_figure("qss_first_order.pdf")
