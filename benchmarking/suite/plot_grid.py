#!/usr/bin/env python3
"""Figures of the benchmark grid from benchmarking_results/suite_metrics.csv.

  grid_ks.pdf     KS distance against eta (base and large Omega) and against Omega
  grid_time.pdf   relative error of the temporal observable, same layout
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from systems import BASE_OMEGA
from run_grid import OMEGA_LARGE

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")
STYLE = {"ssa_null": dict(color="0.6", ls="-", marker=".", label="SSA vs SSA (null)"),
         "pdmp": dict(color="#2656a8", ls="-", marker="o", label="PDMP (hybrid)"),
         "sqssa": dict(color="#3a8a3a", ls="-", marker="s", label="sQSSA"),
         "ode": dict(color="#d0671f", ls="--", marker="^", label="averaged ODE")}
SYSTEMS = ["Repressilator", "Goodwin", "Toggle", "p53"]
TLABEL = {"Repressilator": "period", "Goodwin": "period", "Toggle": "corr. time", "p53": "corr. time"}


def panel(ax, d, xcol, ycol, absval):
    for m, st in STYLE.items():
        dd = d[d.method == m].sort_values(xcol)
        if dd.empty:
            continue
        y = dd[ycol].abs() if absval else dd[ycol]
        ax.plot(dd[xcol], y, ms=4, lw=1.2, **st)
    ax.set_xscale("log")


def figure(df, ycol, ylabel, fname, absval=True, logy=True, systems=SYSTEMS):
    fig, axs = plt.subplots(len(systems), 3, figsize=(11, 2.75 * len(systems)), sharey="row")
    for i, sysn in enumerate(systems):
        d = df[df.system == sysn]
        om0, omL = BASE_OMEGA[sysn], OMEGA_LARGE[sysn]
        panel(axs[i, 0], d[(d.omega == om0)], "eta_worst", ycol, absval)
        panel(axs[i, 1], d[(d.omega == omL)], "eta_worst", ycol, absval)
        panel(axs[i, 2], d[(d.s == 1)], "omega", ycol, absval)
        axs[i, 0].set_ylabel("%s\n%s" % (sysn if sysn != "p53" else "p53-Mdm2",
                                          ylabel if ycol != "time_err" else "|rel. error, %s|" % TLABEL[sysn]))
        axs[i, 0].set_title(r"$\eta$ sweep, $\Omega=%g$" % om0, fontsize=9)
        axs[i, 1].set_title(r"$\eta$ sweep, $\Omega=%g$" % omL, fontsize=9)
        axs[i, 2].set_title(r"$\Omega$ sweep, $s=1$", fontsize=9)
        for a in axs[i]:
            if logy:
                a.set_yscale("log")
            a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    for a in axs[-1, :2]:
        a.set_xlabel(r"$\eta$ (worst case)")
    axs[-1, 2].set_xlabel(r"system size $\Omega$")
    axs[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out = sys.argv[1] if len(sys.argv) > 1 else RES
    fig.savefig(os.path.join(out, fname))


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(RES, "suite_metrics.csv"))
    df = df[df.method != "pdmp_h4"]
    figure(df, "ks", "KS distance", "grid_ks.pdf")
    figure(df, "time_err", "", "grid_time.pdf", systems=["Repressilator", "Goodwin"])
    figure(df, "cv_err", "|rel. error, CV|", "grid_cv.pdf")


def partition_figure(out):
    r = pd.read_csv(os.path.join(RES, "noise_partition.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    mk = {"Repressilator": "o", "Goodwin": "^", "Toggle": "s", "p53": "D"}
    col = {"Repressilator": "#2656a8", "Goodwin": "#d0671f", "Toggle": "#3a8a3a", "p53": "k"}
    for sysn, g in r.groupby("system"):
        axs[0].scatter(g.R_eta_worst, g.share_prom, marker=mk[sysn], color=col[sysn], s=18,
                       label=sysn if sysn != "p53" else "p53-Mdm2")
        axs[1].scatter(g.R_eta_worst, g.share_prom + g.share_mol, marker=mk[sysn], color=col[sysn], s=18)
    g = r[r.system == "p53"]
    from scipy.optimize import curve_fit
    (c,), _ = curve_fit(lambda R, c: R / (R + c), g.R_eta_worst, g.share_prom, p0=[10.0])
    RR = np.logspace(-1, 4, 200)
    axs[0].plot(RR, RR / (RR + c), "k--", lw=1, label=r"$R/(R+%.0f)$" % c)
    axs[0].set_xscale("log"); axs[1].set_xscale("log"); axs[1].set_yscale("log")
    axs[0].set_xlim(1e-1, 1e4); axs[1].set_xlim(1e-1, 1e4)
    axs[1].axhline(1, color="0.5", lw=0.8)
    axs[0].set_xlabel(r"$R=N\,\eta/(1+\eta)$"); axs[1].set_xlabel(r"$R=N\,\eta/(1+\eta)$")
    axs[0].set_ylabel("Var(PDMP) / Var(SSA)")
    axs[1].set_ylabel("[Var(PDMP) + Var(sQSSA)] / Var(SSA)")
    axs[0].legend(fontsize=7, frameon=False)
    for a in axs:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "noise_partition.pdf"))


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else RES
    partition_figure(out)
