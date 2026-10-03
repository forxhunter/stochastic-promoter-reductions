#!/usr/bin/env python3
"""How far a promoter occupancy curve departs from a Hill function, and what a
fitted Hill coefficient reports instead of the site count.

Two sequential binding schemes are compared, both with n sites of equal affinity
K and no cooperative interaction between them:

  ladder    every step forward at k_on*x and back at k_off, with no combinatorial
            factors. Detailed balance gives pi_k proportional to (x/K)^k, so the
            fully bound state is
                pi_n(x) = x^n / (K^n + K^(n-1) x + ... + x^n).
            This is an ordered, single-file scheme: each state has exactly one
            forward and one reverse route.

  Adair     n identical independent sites. The forward rate from state k carries
            the number of free sites and the reverse carries the number bound,
            (n-k) k_on x and k k_off, so pi_k = C(n,k) (x/K)^k and
                pi_n(x) = (x / (x + K))^n.
            This is the scheme normally meant by "n identical binding sites",
            and the one implied when a fitted Hill coefficient is read as a site
            count.

For each, a Hill function alpha_max x^h / (Kh^h + x^h) is fitted by least squares
on log-spaced x, and the fitted h is compared against n.

Usage:
    python3 hill_vs_rational.py --out ../ElsiverFSPromoter_v2_next_revision/figures
"""

import argparse
import os
from math import comb, log

import numpy as np
from scipy.optimize import brentq, curve_fit


def occupancy_ladder(x, n, K=1.0):
    r = np.asarray(x, dtype=float) / K
    num = r ** n
    den = sum(r ** j for j in range(n + 1))
    return num / den


def occupancy_adair(x, n, K=1.0):
    r = np.asarray(x, dtype=float) / K
    return (r / (1.0 + r)) ** n


def occupancy_coop(x, n, K=1.0, w=1.0):
    """n identical sites with an all-to-all pairwise interaction factor w
    (w = 1 recovers independent sites; w -> infinity the concerted limit)."""
    r = np.asarray(x, dtype=float) / K
    terms = [comb(n, k) * r ** k * w ** (k * (k - 1) / 2) for k in range(n + 1)]
    return terms[-1] / sum(terms)


def hill(x, h, Kh):
    return x ** h / (np.abs(Kh) ** h + x ** h)


def local_hill_slope(f, n, K=1.0):
    """Hill slope of the true curve at its own half-maximum: the quantity an
    experimenter reads off a Hill plot."""
    g = lambda lx: f(np.exp(lx), n, K)
    lx_half = brentq(lambda lx: g(lx) - 0.5, log(1e-8), log(1e8))
    h = 1e-5
    dg = (g(lx_half + h) - g(lx_half - h)) / (2 * h)
    return dg / (0.5 * 0.5)          # d ln(f/(1-f)) / d ln x at f = 1/2


def fit_hill(f, n, K=1.0, npts=400):
    """Least-squares Hill fit over four decades around the half-maximum."""
    g = lambda lx: f(np.exp(lx), n, K)
    lx_half = brentq(lambda lx: g(lx) - 0.5, log(1e-8), log(1e8))
    x = np.exp(np.linspace(lx_half - 2 * log(10), lx_half + 2 * log(10), npts))
    y = f(x, n, K)
    p, _ = curve_fit(hill, x, y, p0=[float(n), float(np.exp(lx_half))], maxfev=20000)
    resid = y - hill(x, *p)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return p[0], p[1], rmse, float(np.max(np.abs(resid)))


def fit_hill_log(f, n, K=1.0, fmin=0.01, fmax=0.99, npts=400, **kw):
    """Hill fit under multiplicative error: least squares on ln f, over the
    input range where the true output lies in [fmin, fmax] (two decades of
    output above background at fmin = 0.01)."""
    g = lambda lx: f(np.exp(lx), n, K, **kw)
    lo = brentq(lambda lx: g(lx) - fmin, log(1e-12), log(1e8))
    hi = brentq(lambda lx: g(lx) - fmax, log(1e-12), log(1e8))
    x = np.exp(np.linspace(lo, hi, npts))
    y = f(x, n, K, **kw)
    lh = lambda x, h, Kh: np.log(hill(x, h, Kh))
    p, _ = curve_fit(lh, x, np.log(y), p0=[1.5, float(np.exp(0.5 * (lo + hi)))], maxfev=20000)
    return p[0], float(np.max(np.abs(np.log(y) - lh(x, *p))))


def slope_at_level(f, n, level, K=1.0, **kw):
    """Hill-plot slope d ln[f/(1-f)]/d ln x where the true output equals level."""
    g = lambda lx: f(np.exp(lx), n, K, **kw)
    lx = brentq(lambda lx: g(lx) - level, log(1e-12), log(1e8))
    lo = lambda l: np.log(g(l) / (1 - g(l)))
    h = 1e-5
    return (lo(lx + h) - lo(lx - h)) / (2 * h)


def write_si_tables(out):
    """Cooperativity and log-fit tables for the Supplement."""
    os.makedirs(out, exist_ok=True)
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Fitted Hill coefficient for $n$ identical sites with all-to-all pairwise cooperativity factor $\omega$: linear-scale least-squares exponent over four decades of input, with the half-maximum Hill-plot slope in parentheses. $\omega=1$ is Table~\ref{tab:si_hill_fit}.}",
         r"\label{tab:si_coop}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{c*{6}{c}}", r"\toprule",
         r" & \multicolumn{6}{c}{$\omega$} \\ \cmidrule(lr){2-7}",
         r"$n$ & 1 & 3 & 10 & 30 & 100 & 1000 \\", r"\midrule"]
    for n in (2, 3, 4, 5, 6):
        cells = []
        for w in (1, 3, 10, 30, 100, 1000):
            h = fit_hill(lambda x, n, K=1.0: occupancy_coop(x, n, K, w), n)[0]
            cells.append("%.2f (%.2f)" % (h, slope_at_level(occupancy_coop, n, 0.5, w=w)))
        L.append("%d & " % n + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(os.path.join(out, "coop_table.tex"), "w").write("\n".join(L) + "\n")
    L = [r"\begin{table}[!htbp]", r"\centering",
         r"\caption{Hill fits under multiplicative error: exponent from least squares on $\ln f$ over the input range where $0.01\le f\le0.99$, the largest deviation of that fit in $\ln f$, and the Hill-plot slope where the true output equals 0.1 and 0.01.}",
         r"\label{tab:si_logfit}", r"\small", r"\begin{adjustbox}{max width=\linewidth}",
         r"\begin{tabular}{ccccc|cccc}", r"\toprule",
         r" & \multicolumn{4}{c|}{Independent sites} & \multicolumn{4}{c}{Ordered ladder} \\",
         r"$n$ & log $h$ & max $|\Delta\ln f|$ & slope$_{0.1}$ & slope$_{0.01}$ & log $h$ & max $|\Delta\ln f|$ & slope$_{0.1}$ & slope$_{0.01}$ \\",
         r"\midrule"]
    for n in range(1, 7):
        ha, da = fit_hill_log(occupancy_adair, n); hl, dl = fit_hill_log(occupancy_ladder, n)
        L.append("%d & %.2f & %.3f & %.2f & %.2f & %.2f & %.3f & %.2f & %.2f \\\\" % (
            n, ha, da, slope_at_level(occupancy_adair, n, 0.1), slope_at_level(occupancy_adair, n, 0.01),
            hl, dl, slope_at_level(occupancy_ladder, n, 0.1), slope_at_level(occupancy_ladder, n, 0.01)))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}"]
    open(os.path.join(out, "logfit_table.tex"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="directory for the figure")
    ap.add_argument("--nmax", type=int, default=6)
    ap.add_argument("--si-tables", dest="si_tables", default=None,
                    help="directory for the Supplement's LaTeX tables")
    args = ap.parse_args()
    if args.si_tables:
        write_si_tables(args.si_tables)

    print("  Hill coefficient recovered from a promoter with n equal-affinity sites")
    print("  %-3s | %-28s | %-28s" % ("n", "ordered ladder", "independent sites (Adair)"))
    print("  %-3s | %8s %8s %9s | %8s %8s %9s"
          % ("", "slope", "fit h", "max|res|", "slope", "fit h", "max|res|"))
    rows = []
    for n in range(1, args.nmax + 1):
        sl_l = local_hill_slope(occupancy_ladder, n)
        h_l, _, _, mx_l = fit_hill(occupancy_ladder, n)
        sl_a = local_hill_slope(occupancy_adair, n)
        h_a, _, _, mx_a = fit_hill(occupancy_adair, n)
        print("  %-3d | %8.2f %8.2f %9.4f | %8.2f %8.2f %9.4f"
              % (n, sl_l, h_l, mx_l, sl_a, h_a, mx_a))
        rows.append((n, sl_l, h_l, mx_l, sl_a, h_a, mx_a))

    print("\n  'slope'    = d ln(f/(1-f)) / d ln x at half-maximum, what a Hill plot reports")
    print("  'fit h'    = least-squares Hill exponent over four decades")
    print("  'max|res|' = largest absolute deviation of the best Hill fit from the true curve")

    print("\n  Closed form, independent sites: half-maximum slope = 2n(1 - 2^(-1/n)),")
    print("  limit 2 ln 2 = %.3f; ordered ladder limit 2." % (2 * log(2)))
    for n in range(1, args.nmax + 1):
        print("    n=%d  %.3f" % (n, 2 * n * (1 - 2 ** (-1.0 / n))))

    print("\n  Multiplicative error: Hill fit on ln f over outputs in [0.01, 0.99]")
    print("  %-3s | %-34s | %-34s" % ("n", "independent sites", "ordered ladder"))
    print("  %-3s | %8s %8s %8s %8s | %8s %8s %8s %8s"
          % ("", "log h", "max|dln|", "sl@0.1", "sl@0.01", "log h", "max|dln|", "sl@0.1", "sl@0.01"))
    for n in range(1, args.nmax + 1):
        ha, da = fit_hill_log(occupancy_adair, n)
        hl, dl = fit_hill_log(occupancy_ladder, n)
        print("  %-3d | %8.2f %8.3f %8.2f %8.2f | %8.2f %8.3f %8.2f %8.2f"
              % (n, ha, da, slope_at_level(occupancy_adair, n, 0.1),
                 slope_at_level(occupancy_adair, n, 0.01),
                 hl, dl, slope_at_level(occupancy_ladder, n, 0.1),
                 slope_at_level(occupancy_ladder, n, 0.01)))

    print("\n  Cooperative sites (pairwise factor w): linear-scale fit h, half-max slope")
    for n in (2, 3, 4):
        for w in (1, 3, 10, 30, 100):
            h_c = fit_hill(lambda x, n, K=1.0: occupancy_coop(x, n, K, w), n)[0]
            s_c = slope_at_level(occupancy_coop, n, 0.5, w=w)
            print("    n=%d  w=%-4g  fit h %.2f  slope %.2f" % (n, w, h_c, s_c))

    if not args.out:
        return rows

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11, "legend.fontsize": 9})
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    ax = axs.ravel()
    xs = np.logspace(-2, 2, 400)

    for n, style in ((2, "-"), (4, "--")):
        ax[0].semilogx(xs, occupancy_adair(xs, n), style, color="k",
                       label="independent, $n=%d$" % n)
        ax[0].semilogx(xs, occupancy_ladder(xs, n), style, color="0.55",
                       label="ordered, $n=%d$" % n)
    ax[0].set_xlabel("regulator $x/K$"); ax[0].set_ylabel("fraction fully bound")
    ax[0].legend(frameon=False); ax[0].set_title("(A) occupancy")

    n = 4
    y = occupancy_adair(xs, n)
    h, Kh, _, _ = fit_hill(occupancy_adair, n)
    ax[1].loglog(xs, y, "k-", label="independent sites, $n=4$")
    ax[1].loglog(xs, hill(xs, h, Kh), "r--", label="linear-scale Hill fit, $h=%.2f$" % h)
    ax[1].set_ylim(1e-3, 1.5)
    ax[1].set_xlabel("regulator $x/K$"); ax[1].set_ylabel("fraction fully bound (log)")
    ax[1].legend(frameon=False, loc="lower right")
    ax[1].set_title("(B) the same fit on a log scale")

    ns = [r[0] for r in rows]
    ax[2].plot(ns, ns, "k:", label="$h=n$")
    ax[2].plot(ns, [r[5] for r in rows], "ks-", label="independent, linear fit")
    ax[2].plot(ns, [fit_hill_log(occupancy_adair, m)[0] for m in ns], "ks--",
               mfc="white", label="independent, log fit")
    ax[2].plot(ns, [r[2] for r in rows], "o-", color="0.55", label="ordered, linear fit")
    ax[2].plot(ns, [fit_hill_log(occupancy_ladder, m)[0] for m in ns], "o--",
               color="0.55", mfc="white", label="ordered, log fit")
    ax[2].set_xlabel("number of sites $n$"); ax[2].set_ylabel("fitted Hill coefficient $h$")
    ax[2].legend(frameon=False)
    ax[2].set_title("(C) fitted $h$ depends on the error model")

    ws = np.logspace(0, 3, 40)
    for m, col in ((2, "0.7"), (3, "0.45"), (4, "k")):
        ax[3].semilogx(ws, [fit_hill(lambda x, n, K=1.0: occupancy_coop(x, n, K, w), m)[0]
                            for w in ws], "-", color=col, label="$n=%d$" % m)
        ax[3].axhline(m, color=col, ls=":", lw=0.8)
    ax[3].set_xlabel("pairwise cooperativity factor $\\omega$")
    ax[3].set_ylabel("fitted Hill coefficient $h$")
    ax[3].legend(frameon=False)
    ax[3].set_title("(D) cooperative sites")

    for a in ax:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, "hill_vs_rational.%s" % ext), dpi=200)
    print("\n  wrote %s/hill_vs_rational.{pdf,png}" % args.out)
    return rows


if __name__ == "__main__":
    main()
