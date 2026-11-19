#!/usr/bin/env python3
"""Scan of the first-order coefficient c1 of a multistate promoter against its control F
over promoter architectures (second-round revision of JTB-D-26-01677).

c1(F) = 100 (delta_P - delta_F) / x* = 100 R(x*) / (Fbar'(x*) x*) is computed from the
local first-order expansion only (no simulation), with the same functions as
revision_reduction.py. It measures how large the residual drift R of Proposition 1 is
in relative terms.

Architectures, for the negatively (NAR) and positively (PFB) autoregulated genes:
  site, "all"   n identical sites, pairwise cooperativity omega; the output changes only
                when all n sites are occupied (repression in NAR, activation in PFB);
                as the benchmark promoter S and the extended family of Table S40.
  site, "any"   the output changes as soon as one site is occupied (in NAR this is
                promoter X for n = 3, omega = 3).
  refractory    the refractory promoters of Table S40 (2 and 3 stages).
Every site promoter has its half-maximum at x = 30 and slowest relaxation rate 1 over
[0, 150], as in the main text. Also reported: R(x*) / |b(x*)|, the ratio
lambda_3 / lambda_2 at x*, and the probability of the intermediate states at x*.

Writes benchmarking_results/revision/c1_scan.csv.
"""
import os
import numpy as np, pandas as pd
from math import comb
from scipy.optimize import brentq
from qss_diffusion import fundamental
from promoter_family import fixed_point
from promoter_family2 import site as site_all, refractory
from revision_reduction import (OUT, K_HALF, lam2, activity, controls, drift_split,
                                local_shift_Q, local_shift_two_state)


def site_rule(sysname, n, omega, rule):
    """Site promoter; rule 'all' uses promoter_family2.site (identical to Table S40);
    rule 'any': the output changes when at least one site is occupied."""
    if rule == "all":
        return site_all(sysname, n, omega)
    act = np.zeros(n + 1)
    if sysname == "NAR":
        act[0] = 1.0                 # active only when all sites are free
    else:
        act[1:] = 1.0                # active when at least one site is occupied

    def make(KS, kb):
        def Q(x):
            q = np.zeros((n + 1, n + 1))
            for k in range(n):
                q[k, k + 1] = (n - k) * kb * x / KS
                q[k + 1, k] = (k + 1) * kb * omega ** (-k)
            return q - np.diag(q.sum(axis=1))
        return Q
    KS = np.exp(brentq(lambda lk: activity(make(np.exp(lk), 1.0), act, K_HALF) - 0.5,
                       np.log(K_HALF) - 12, np.log(K_HALF) + 12))
    xs = np.concatenate([[0.0], np.logspace(-3, np.log10(150.0), 300)])
    kb = 1.0 / min(lam2(make(KS, 1.0), x) for x in xs)
    coef = [comb(n, k) * omega ** (k * (k - 1) / 2) * KS ** (-k) for k in range(n + 1)]
    B = lambda x: sum(c * np.asarray(x, float) ** k for k, c in enumerate(coef))
    return make(KS, kb), act, B


def c1_F(sysname, Q, act, B):
    xs = fixed_point(sysname, Q, act)
    pon_fun = lambda v: activity(Q, act, v)
    C, _ = controls(dict(Q=Q, act=act, B=B), xs)
    c1 = 100 * (local_shift_Q(sysname, Q, act, xs) - local_shift_two_state(sysname, pon_fun, C["F"], xs)) / xs
    h = 1e-3 * xs
    d = drift_split(sysname, Q, act, xs + h * np.arange(-3, 4))
    pi, _ = fundamental(Q(xs))
    ev = np.sort(np.abs(np.real(np.linalg.eigvals(Q(xs)))))
    return dict(x_star=xs, c1_F=c1, R=d["R"][3], b=d["b"][3], R_over_b=d["R"][3] / abs(d["b"][3]),
                lam_ratio=ev[2] / ev[1] if len(ev) > 2 else np.inf,
                p_intermediate=float(pi[1:-1].sum()) if len(pi) > 2 else 0.0)


def main():
    rows = []
    for sysname in ("NAR", "PFB"):
        for rule in ("all", "any"):
            for n in (2, 3, 4):
                for omega in (1.0, 3.0, 10.0, 100.0):
                    Q, act, B = site_rule(sysname, n, omega, rule)
                    r = dict(circuit=sysname, family="site", rule=rule, n=n, omega=omega)
                    r.update(c1_F(sysname, Q, act, B))
                    rows.append(r)
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
        for stages in (2, 3):
            Q, act, _ = refractory(sysname, stages)
            r = dict(circuit=sysname, family="refractory", rule="-", n=stages, omega=np.nan)
            r.update(c1_F(sysname, Q, act, None))
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "c1_scan.csv"), index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
