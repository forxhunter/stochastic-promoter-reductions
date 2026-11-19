#!/usr/bin/env python3
"""Figure 5 of the revised manuscript (residual drift, local matching, temporal
statistics) and Supplementary Figure S-rev1 (error against S0 mismatch), from
benchmarking_results/revision/*.csv (revision_reduction.py).
Usage: revision_figures.py OUTDIR"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8})

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results", "revision")
PCOL = {"NAR-S": "#2656a8", "PFB-S": "#b2325e", "NAR-X": "#d0671f"}
PNAME = {"NAR-S": "neg. autoreg., S", "PFB-S": "pos. autoreg., S", "NAR-X": "neg. autoreg., X"}
CSTY = {"F": ("k", "o", "-"), "L2": ("#3a8a3a", "^", "-"), "L1": ("#2656a8", "s", "-"),
        "M": ("#b2325e", "D", "-"), "C": ("0.55", "v", "-")}
CNAME = {"F": "F: $S_0(x)$ at every $x$", "L2": "L2: $S_0, S_0', S_0''$ at $x^*$",
         "L1": "L1: $S_0, S_0'$ at $x^*$", "M": "M: $S_0(x^*)$, $\\lambda\\propto B(x)$",
         "C": "L0: $S_0(x^*)$, constant $\\lambda$"}


def clean(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


def main(out):
    prof = pd.read_csv(os.path.join(RES, "residual_profiles.csv"))
    ex = pd.read_csv(os.path.join(RES, "reduction_exact.csv"))
    acf = pd.read_csv(os.path.join(RES, "promoter_acf.csv"))
    e1 = ex[ex.omega == 1.0]
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.3))

    # A: drift decomposition of the example promoter X
    ax = axs[0, 0]
    p = prof[prof.promoter == "NAR-X"]
    ax.plot(p.x, p.b, color="k", lw=1.6, label=r"$b$ (multistate X)")
    ax.plot(p.x, p.b2, color="#2656a8", lw=1.4, ls="--", label=r"$b_2(\bar r, S_0)$ (= $b$ of control F)")
    ax.plot(p.x, p.R, color="#d0671f", lw=1.6, label=r"$R = b_2 - b$")
    q = prof[prof.promoter == "NAR-S"]
    ax.plot(q.x, q.R, color="0.6", lw=1.2, ls=":", label=r"$R$ of benchmark S")
    xs = e1[e1.promoter == "NAR-X"].x_star.iloc[0]
    ax.axvline(xs, color="0.7", lw=0.8)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlim(0, 100); ax.set_ylim(-60, 40)
    ax.set_xlabel(r"regulator concentration $x$"); ax.set_ylabel(r"drift at $s=1$")
    ax.set_title("residual drift, negative autoregulation", fontsize=10)
    ax.legend(frameon=False, loc="lower left")

    # B: predicted vs exact effect of R: s * Delta against s, P vs F
    ax = axs[0, 1]
    for name in ("NAR-X", "NAR-S", "PFB-S"):
        d = e1[(e1.promoter == name) & (e1.control == "F")].sort_values("s")
        ax.plot(d.s, d.s * d.dmean, color=PCOL[name], marker="o", ms=4, lw=1.4, label="%s vs F, exact" % PNAME[name])
        ax.axhline(d.c1.iloc[0], color=PCOL[name], ls="--", lw=1.0)
    ax.plot([], [], color="0.4", ls="--", label=r"first order, $c_1 = 100R(x^*)/(\bar F' x^*)$")
    ax.set_xscale("log"); ax.set_xlabel(r"promoter speed factor $s$")
    ax.set_ylabel(r"$s\times$[mean(P)/mean(F) $-$ 1]  (%)")
    ax.set_title(r"effect of $R$ on the mean ($\Omega=1$)", fontsize=10)
    ax.legend(frameon=False, loc="lower left")

    # C: how much of S0(x) must be matched
    ax = axs[1, 0]
    for name, ls in (("NAR-S", "-"), ("PFB-S", "--")):
        for c in ("C", "M", "L1", "L2", "F"):
            d = e1[(e1.promoter == name) & (e1.control == c)].sort_values("s")
            col, mk, _ = CSTY[c]
            ax.plot(d.s, d.dmean, color=col, marker=mk, ms=3.5, lw=1.3, ls=ls,
                    label=CNAME[c] if name == "NAR-S" else None)
    ax.plot([], [], color="0.3", ls="-", label="neg. autoreg. (S)")
    ax.plot([], [], color="0.3", ls="--", label="pos. autoreg. (S)")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel(r"promoter speed factor $s$"); ax.set_ylabel("mean(S)/mean(control) $-$ 1  (%)")
    ax.set_title(r"controls matched to $S_0$ locally or globally", fontsize=10)
    hC, lC = ax.get_legend_handles_labels()

    # D: temporal statistics, P vs F
    ax = axs[1, 1]
    for name in ("NAR-X", "NAR-S", "PFB-S"):
        d = e1[(e1.promoter == name) & (e1.control == "F")].sort_values("s")
        ax.plot(d.s, d.dtau_int, color=PCOL[name], marker="o", ms=4, lw=1.4, label="%s vs F" % PNAME[name])
    ax.axhline(0, color="0.5", lw=0.8); ax.set_xscale("log")
    ax.set_xlabel(r"promoter speed factor $s$"); ax.set_ylabel(r"integrated correlation time, P/F $-$ 1  (%)")
    ax.set_title(r"temporal statistics with matched $\bar{r}(x)$ and $S_0(x)$", fontsize=10)
    ax.set_ylim(-15, 60)
    ax.legend(frameon=False, loc="upper left")
    ins = ax.inset_axes([0.6, 0.5, 0.37, 0.4])
    a = acf[acf.promoter == "NAR-X"]
    lam = a.lam_F.iloc[0]
    ins.plot(a.t * lam, a.acf_P, color=PCOL["NAR-X"], lw=1.3, label="X")
    ins.plot(a.t * lam, a.acf_F, color="k", lw=1.0, ls="--", label="F")
    ins.set_yscale("log"); ins.set_ylim(1e-3, 1.1); ins.set_xlim(0, 6)
    ins.set_xlabel(r"$\lambda_F(x^*)\,t$", fontsize=7); ins.set_ylabel("promoter ACF", fontsize=7)
    ins.tick_params(labelsize=6); ins.legend(frameon=False, fontsize=6)
    for a_ in axs.flat:
        clean(a_)
    for a_, l in zip(axs.flat, "ABCD"):
        a_.text(-0.14, 1.04, l, transform=a_.transAxes, fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(hC, lC, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, "residual.pdf"))

    # Supplementary: error against S0 mismatch over the visited distribution
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    d = ex[(ex.omega == 1.0) & (ex.control != "F")]
    for name in d.promoter.unique():
        for c in ("C", "M", "L1", "L2"):
            g = d[(d.promoter == name) & (d.control == c)]
            col, mk, _ = CSTY[c]
            ax.scatter(g.eps_S0, g.dmean.abs(), marker=mk, s=22, facecolors=PCOL[name], edgecolors="none", alpha=0.8)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", color=PCOL[n], label=PNAME[n]) for n in d.promoter.unique()]
    h += [Line2D([], [], ls="", marker=CSTY[c][1], mfc="none", mec="k", label=CNAME[c].split(":")[0]) for c in ("C", "M", "L1", "L2")]
    ax.legend(handles=h, frameon=False, fontsize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$S_0$ mismatch $\langle|\ln(S_0^{C}/S_0^{P})|\rangle_P$")
    ax.set_ylabel(r"$|$mean(P)/mean(C) $-$ 1$|$  (%)")
    clean(ax); fig.tight_layout()
    fig.savefig(os.path.join(out, "s0_mismatch.pdf"))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else RES)
