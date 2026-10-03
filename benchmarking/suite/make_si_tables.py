#!/usr/bin/env python3
"""Generate LaTeX reaction tables for the Supplement directly from systems.py,
so the Supplement cannot drift from the simulated models."""
import sys
import numpy as np
from systems2 import MODELS, BASE_OMEGA

TEX = {"S1_off": r"S_1^{0}", "S1_on": r"S_1^{1}", "S2_off": r"S_2^{0}", "S2_on": r"S_2^{1}",
       "S3_off": r"S_3^{0}", "S3_on": r"S_3^{1}", "S_off": r"S^{0}", "S_on": r"S^{1}",
       "Su_off": r"S_U^{0}", "Su_on": r"S_U^{1}", "Sv_off": r"S_V^{0}", "Sv_on": r"S_V^{1}",
       "Mdm2": r"\mathrm{Mdm2}", "p53": r"\mathrm{p53}", "ATM": r"\mathrm{ATM}"}


def t(s):
    import re
    m = re.match(r"^(S\d?|Su|Sv|S|P)_(off|on)$", s)
    if m:
        base = {"S": "S", "Su": "S_U", "Sv": "S_V", "P": "P"}.get(m.group(1), "S_%s" % m.group(1)[1:])
        return "%s^{%s}" % (base, "0" if m.group(2) == "off" else "1")
    m = re.match(r"^(S\d?|Su|S)_b(\d+)$", s)
    if m:
        base = {"S": "B", "Su": "B_U"}.get(m.group(1), "B_%s" % m.group(1)[1:])
        return "%s^{(%s)}" % (base, m.group(2))
    if s in TEX:
        return TEX[s]
    if s[0] in "mup" and s[1:].isdigit():
        return "%s_%s" % (s[0], s[1:])
    if s[0] == "P" and s[1:].isdigit():
        return "P_%s" % s[1:]
    return s


def num(v):
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    if -2 <= e <= 3:
        return ("%.4g" % v)
    m = v / 10 ** e
    return r"%.3g\times10^{%d}" % (m, e)


def propensity(rx, isD):
    k = rx["k"]; parts = [num(k)]
    if rx["w"] == 1:
        parts.append(r"\Omega")
    for sp, e in rx["reac"].items():
        if sp in isD:
            parts.append("[%s]" % t(sp))
        else:
            parts.append(("(%s/\\Omega)" % t(sp)) + ("^{%g}" % e if e != 1 else ""))
    return r"\,".join(parts)


def reaction(rx):
    cat = [t(sp) for sp in rx["reac"] if rx["change"].get(sp, 0) == 0]
    lhs = [t(sp) for sp, d in rx["change"].items() if d < 0] + cat
    rhs = cat + [t(sp) for sp, d in rx["change"].items() if d > 0]
    return (" + ".join(lhs) or r"\emptyset") + r" \to " + (" + ".join(rhs) or r"\emptyset")


def table(name):
    M = MODELS[name]()
    isD = {M.species[j] for j in range(len(M.species)) if M.isD[j]}
    lines = [r"\begin{longtable}{p{0.40\linewidth}p{0.52\linewidth}}",
             r"\caption{Reactions of the %s model, as simulated. $[S]$ is the 0/1 indicator of promoter state $S$; molecular species enter propensities as concentrations $X/\Omega$, and in the SSA an integer power $(X/\Omega)^n$ with $n\ge2$ is the falling factorial $X(X-1)\cdots(X-n+1)/\Omega^n$, except in the promoters of the P, E and M architectures, which use plain powers. Promoter transitions (no factor $\Omega$) are multiplied by $s$ in the $\eta$ sweep.}\label{tab:rx_%s}\\" % (name.replace("p53", "p53--Mdm2"), name.lower()),
             r"\toprule \textbf{Reaction} & \textbf{Propensity} \\ \midrule \endfirsthead",
             r"\toprule \textbf{Reaction} & \textbf{Propensity} \\ \midrule \endhead"]
    for rx in M.reactions:
        lines.append("$%s$ & $%s$ \\\\" % (reaction(rx), propensity(rx, isD)))
    lines += [r"\bottomrule", r"\end{longtable}"]
    info = (r"Base system size $\Omega=%g$; time unit %s; recorded every %g; burn-in %g; total %g; "
            r"regulator lifetime $\tau=%.3g$; PDMP step $h=%g$." %
            (BASE_OMEGA[name.split("-")[0]], M.time_unit, M.record_dt, M.t_burn, M.t_total, M.tau_prot, M.h))
    return "\n".join(lines) + "\n\n\\noindent{\\small " + info + "}\n"


if __name__ == "__main__":
    out = sys.argv[1]
    with open(out, "w") as f:
        for name in MODELS:
            f.write("\\subsection{%s}\n\\label{si:model_%s}\n" % (name.replace("p53", "p53--Mdm2"), name.lower()))
            f.write(table(name) + "\n")
    print("wrote", out)
