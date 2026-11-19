#!/usr/bin/env python3
"""Revision computations for the first-order promoter reduction.

Addresses reviewer comments R1.3, R1 minor 4, R2.2, R2.4 and R2 minor 4 of
JTB-D-26-01677. One regulated species x, catalytic binding, binary output.

Promoters P
  NAR-S, PFB-S  benchmark site-resolved promoters of the main text (n = 2,
                omega = 100; transcription needs all sites free (NAR) or all
                sites occupied (PFB)).
  NAR-X         example with a large residual drift R: three sites, omega = 3,
                and any occupied site represses transcription. Half-maximum at
                30 and slowest relaxation over [0, 150] equal to 1, as for the
                benchmark promoters.

Two-state controls with exactly the response pi_on(x) of P, which differ only in
the relaxation rate lambda(x) (S0 = 2 pi_on pi_off / lambda for two states):
  F   lambda_F(x) = 2 pi_on pi_off / S0_P(x): same S0 at every x
  L2  log lambda quadratic in x - x*: same S0 and first two derivatives at x*
  L1  log lambda linear in x - x*: same S0 and dS0/dx at x*
  M   lambda = kappa B(x), B the binding polynomial of P, S0 matched at x*
  C   constant lambda, S0 matched at x*

lambda_F at x = 0 (0/0) is set to its limit 1 / m, with m the mean exit time
from the promoter group whose probability vanishes at x = 0, started from the
limiting conditional distribution within that group (Supplementary Section S2.4
of the revision); minor comment 4 checks the sensitivity to this choice.

For each P, control, promoter speed s and system size Omega: exact stationary
distributions by finite-state projection on (promoter state, copy number) with
rates evaluated at x = n / Omega; mean, CV, Wasserstein-1 distance / SD of P,
total-variation distance; the autocorrelation function of the copy number (1/e
time and integrated time) from the generator; the S0 mismatch over the
stationary distribution of P, E_P|ln(S0_C / S0_P)|; the strict first-order
coefficient c1 from the local expansion. Also profiles of R(x), b(x), b_2(x)
and the promoter-output autocorrelation at x*.

Outputs in benchmarking_results/revision/:
  residual_profiles.csv, reduction_exact.csv, promoter_acf.csv,
  lamF_origin.csv, identity_check.csv
Usage: revision_reduction.py [profiles|exact|origin|identity|all]
"""
import os, sys, time
import numpy as np, pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm
from scipy.optimize import brentq
from math import comb
from qss_diffusion import fundamental, CIRC
from promoter_family import fixed_point
from promoter_family2 import site as site_all

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "benchmarking_results", "revision")
os.makedirs(OUT, exist_ok=True)
K_HALF = 30.0
S_GRID = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
OM_GRID = [0.3, 3.0, 10.0, 30.0]           # at s = 1 and s = 10, in addition to Omega = 1


# ---------------------------------------------------------------- promoters
def lam2(Q, x):
    return np.sort(np.abs(np.real(np.linalg.eigvals(Q(x)))))[1]


def activity(Q, act, x):
    pi, _ = fundamental(Q(x))
    return float(np.dot(pi, act))


def site_any(n, omega):
    """Repressor with n sites and pairwise cooperativity omega; any occupied site
    represses. K_S sets the half-maximum to K_HALF and k_b the slowest relaxation
    over [0, 150] to 1."""
    act = np.zeros(n + 1); act[0] = 1.0

    def make(KS, kb):
        def Q(x):
            q = np.zeros((n + 1, n + 1))
            for k in range(n):
                q[k, k + 1] = (n - k) * kb * x / KS
                q[k + 1, k] = (k + 1) * kb * omega ** (-k)
            return q - np.diag(q.sum(axis=1))
        return Q
    KS = np.exp(brentq(lambda lk: activity(make(np.exp(lk), 1.0), act, K_HALF) - 0.5,
                       np.log(K_HALF) - 10, np.log(K_HALF) + 10))
    xs = np.concatenate([[0.0], np.logspace(-3, np.log10(150.0), 300)])
    kb = 1.0 / min(lam2(make(KS, 1.0), x) for x in xs)
    coef = [comb(n, k) * omega ** (k * (k - 1) / 2) * KS ** (-k) for k in range(n + 1)]
    B = lambda x: sum(c * np.asarray(x, float) ** k for k, c in enumerate(coef))
    return make(KS, kb), act, B, dict(KS=KS, kb=kb)


def promoters():
    P = {}
    for sysname in ("NAR", "PFB"):
        Q, act, B = site_all(sysname, 2, 100.0)
        P[sysname + "-S"] = dict(sys=sysname, Q=Q, act=act, B=B)
    Q, act, B, info = site_any(3, 3.0)
    P["NAR-X"] = dict(sys="NAR", Q=Q, act=act, B=B, info=info)
    return P


# ---------------------------------------------------------------- first-order quantities
def local_quantities(sysname, Q, act, x):
    """pi_on, S0 (s = 1), Fbar, D_p, phi and pi on a grid x."""
    c = CIRC[sysname]
    pon, S0, Fb, Dp, PI, PHI, FM = [], [], [], [], [], [], []
    for v in np.atleast_1d(x):
        pi, Z = fundamental(Q(v))
        F = c["beta"] + c["alpha"] * act - c["gamma"] * v
        fb = np.dot(pi, F); phi = Z @ (F - fb)
        a = act - np.dot(pi, act)
        pon.append(np.dot(pi, act)); S0.append(2 * np.dot(pi * a, Z @ a))
        Fb.append(fb); Dp.append(np.dot(pi * (F - fb), phi)); PI.append(pi); PHI.append(phi); FM.append(F)
    return (np.array(pon), np.array(S0), np.array(Fb), np.array(Dp),
            np.array(PI), np.array(PHI), np.array(FM))


def drift_split(sysname, Q, act, x):
    """b (direct), b_2(pi_on, S0) and R of Proposition 1 on a uniform grid x."""
    c = CIRC[sysname]
    pon, S0, Fb, Dp, PI, PHI, FM = local_quantities(sysname, Q, act, x)
    b = np.sum(PI * FM * np.gradient(PHI, x, axis=0), axis=1)
    on = act > 0.5
    dr = np.gradient(pon, x)
    b2 = np.gradient(Dp, x) - dr * Dp * (Fb + c["alpha"] * (1 - 2 * pon)) / (c["alpha"] * pon * (1 - pon))
    rho_on = PI[:, on] / pon[:, None]; rho_off = PI[:, ~on] / (1 - pon)[:, None]
    d_on = np.gradient(rho_on, x, axis=0); d_off = np.gradient(rho_off, x, axis=0)
    Fon = c["beta"] + c["alpha"] - c["gamma"] * x; Foff = c["beta"] - c["gamma"] * x
    R = pon * Fon * np.sum(d_on * PHI[:, on], axis=1) + (1 - pon) * Foff * np.sum(d_off * PHI[:, ~on], axis=1)
    return dict(pon=pon, S0=S0, Fb=Fb, Dp=Dp, b=b, b2=b2, R=R)


def lamF_limit0(Q, act):
    """Limit of lambda_F = 2 pi_on pi_off / S0 at x -> 0: 1 / (rho^T (-Q_gg)^-1 1), with g the
    group whose probability vanishes and rho its limiting conditional distribution."""
    pi0, _ = fundamental(Q(1e-6))
    on = act > 0.5
    g = on if pi0[on].sum() < pi0[~on].sum() else ~on
    rho = pi0[g] / pi0[g].sum()
    q = Q(0.0)[np.ix_(g, g)]
    return 1.0 / float(rho @ np.linalg.solve(-q, np.ones(g.sum()))), rho


def lamF_fun(Q, act, x0=None):
    """lambda_F on an array of x; at x = 0 use the limit (x0 None) or lambda_F(x0)."""
    lim, _ = lamF_limit0(Q, act)

    def f(x):
        x = np.atleast_1d(np.asarray(x, float))
        out = np.empty_like(x)
        z = x <= 0
        if (~z).any():
            pon, S0 = local_quantities("NAR", Q, act, x[~z])[:2]
            out[~z] = 2 * pon * (1 - pon) / S0
        if z.any():
            if x0 is None:
                out[z] = lim
            else:
                pon, S0 = local_quantities("NAR", Q, act, [x0])[:2]
                out[z] = 2 * pon[0] * (1 - pon[0]) / S0[0]
        return out
    return f


def controls(pr, xs):
    """Relaxation-rate functions lambda_C(x) of the two-state controls."""
    Q, act, B = pr["Q"], pr["act"], pr["B"]
    lF = lamF_fun(Q, act)
    h = 1e-2 * xs
    L = np.log(lF(xs + h * np.arange(-2, 3)))
    a1 = (L[0] - 8 * L[1] + 8 * L[3] - L[4]) / (12 * h)
    a2 = (-L[0] + 16 * L[1] - 30 * L[2] + 16 * L[3] - L[4]) / (12 * h ** 2)
    l0 = float(lF(xs)[0])
    C = {"F": lF,
         "L2": lambda x: l0 * np.exp(a1 * (np.asarray(x, float) - xs) + 0.5 * a2 * (np.asarray(x, float) - xs) ** 2),
         "L1": lambda x: l0 * np.exp(a1 * (np.asarray(x, float) - xs)),
         "C": lambda x: l0 * np.ones_like(np.asarray(x, float))}
    if B is not None:
        C["M"] = lambda x: l0 * B(x) / B(xs)
    return C, dict(lam_xs=l0, dlnlam=a1, d2lnlam=a2)


def two_state_Q(pon_fun, lam_fun):
    def Q(x):
        p = pon_fun(x); l = float(lam_fun(np.array([x]))[0])
        return np.array([[-l * p, l * p], [l * (1 - p), -l * (1 - p)]])     # state 0 off, 1 on
    return Q, np.array([0.0, 1.0])


# ---------------------------------------------------------------- exact FSP
def generator(sysname, Qx, act, s, om):
    """CME generator on (promoter state, copy number); Qx: (N, m, m) promoter
    generators at x = n / Omega (s = 1)."""
    c = CIRC[sysname]
    N = Qx.shape[0]; m = len(act); nn = np.arange(N)
    idx = lambda st, k: st * N + k
    rows, cols, vals = [], [], []
    for a in range(m):
        birth = om * (c["beta"] + c["alpha"] * act[a]) * np.ones(N)
        k = np.arange(N - 1); rows.append(idx(a, k)); cols.append(idx(a, k + 1)); vals.append(birth[:-1])
        k = np.arange(1, N); rows.append(idx(a, k)); cols.append(idx(a, k - 1)); vals.append(c["gamma"] * nn[1:])
        for b in range(m):
            if a != b:
                rows.append(idx(a, nn)); cols.append(idx(b, nn)); vals.append(s * Qx[:, a, b])
    r = np.concatenate(rows); cc = np.concatenate(cols); v = np.concatenate(vals)
    ok = v > 0
    G = sp.coo_matrix((v[ok], (r[ok], cc[ok])), shape=(m * N, m * N)).tocsr()
    return G - sp.diags(np.asarray(G.sum(axis=1)).ravel())


def nbox(sysname, om):
    c = CIRC[sysname]
    return int(om * (c["beta"] + c["alpha"]) / c["gamma"] + 12 * np.sqrt(om * (c["beta"] + c["alpha"])) + 30)


def stationary(G):
    """Null vector of a conservative generator. First pass: normalisation row in
    place of the last balance equation (as fsp.stationary); second pass: fix the
    most probable state and solve the remaining balance equations, which keeps
    small probabilities accurate. The relative residual is checked."""
    A = G.T.tocsr().tolil(); n = A.shape[0]
    A[n - 1, :] = np.ones(n)
    rhs = np.zeros(n); rhs[-1] = 1.0
    p = np.maximum(spla.spsolve(A.tocsc(), rhs), 0.0); p /= p.sum()
    At = G.T.tocsc()
    k = int(np.argmax(p)); keep = np.r_[0:k, k + 1:n]
    x = spla.spsolve(At[keep][:, keep], -np.asarray(At[keep][:, [k]].todense()).ravel())
    p = np.maximum(np.insert(x, k, 1.0), 0.0); p /= p.sum()
    res = np.abs(At @ p).sum() / np.abs(At.diagonal() * p).sum()
    if res > 1e-10:
        raise RuntimeError("stationary residual %.2e" % res)
    return p


def acf(G, p, f, var):
    """Integrated autocorrelation time and 1/e time of f under generator G."""
    n = G.shape[0]; k = int(np.argmax(p)); keep = np.r_[0:k, k + 1:n]
    Gr = G.tocsc()[keep][:, keep]
    u = np.zeros(n); u[keep] = spla.spsolve(Gr, -f[keep])
    tint = float(np.dot(p * f, u) / var)
    T = 8 * tint; K = 1601
    W = spla.expm_multiply(G, f, start=0.0, stop=T, num=K, endpoint=True)
    C = (W @ (p * f)) / var
    t = np.linspace(0, T, K)
    j = np.argmax(C < np.exp(-1))
    te = np.interp(np.exp(-1), [C[j], C[j - 1]], [t[j], t[j - 1]]) if j > 0 else np.nan
    return tint, float(te)


def exact(sysname, Q, act, s, om, lam=None, pon_fun=None, need_acf=True):
    """Stationary distribution and statistics of P (lam None) or of the two-state
    control with relaxation lam(x) and activity pon_fun(x)."""
    N = nbox(sysname, om) + 1; nn = np.arange(N); x = nn / om
    if lam is None:
        Qx = np.array([Q(v) for v in x]); a = act
    else:
        pon = np.array([pon_fun(v) for v in x]); l = lam(x)
        Qx = np.zeros((N, 2, 2))
        Qx[:, 0, 1] = l * pon; Qx[:, 1, 0] = l * (1 - pon)
        Qx[:, 0, 0] = -Qx[:, 0, 1]; Qx[:, 1, 1] = -Qx[:, 1, 0]
        a = np.array([0.0, 1.0])
    G = generator(sysname, Qx, a, s, om)
    p = stationary(G)
    m = len(a)
    px = p.reshape(m, N).sum(axis=0)
    mean = np.dot(px, nn); var = np.dot(px, (nn - mean) ** 2)
    out = dict(mean=mean / om, var=var, cv=np.sqrt(var) / mean, edge=px[-1], px=px)
    if need_acf:
        f = np.tile(nn - mean, m).astype(float)
        out["tau_int"], out["tau_e"] = acf(G, p, f, var)
    return out


def distances(px, py):
    n = max(len(px), len(py))
    px = np.pad(px, (0, n - len(px))); py = np.pad(py, (0, n - len(py)))
    k = np.arange(n); sd = np.sqrt(np.dot(px, k ** 2) - np.dot(px, k) ** 2)
    return np.abs(np.cumsum(px) - np.cumsum(py)).sum() / sd, 0.5 * np.abs(px - py).sum()


def local_shift_two_state(sysname, pon_fun, lam_fun, xs, h=1e-3):
    Qc, ac = two_state_Q(pon_fun, lam_fun)
    return local_shift_Q(sysname, Qc, ac, xs, h)


def local_shift_Q(sysname, Q, act, xs, h=1e-3):
    x = xs + h * np.arange(-3, 4)
    pon, S0, Fb, Dp, PI, PHI, FM = local_quantities(sysname, Q, act, x)
    b = np.sum(PI * FM * np.gradient(PHI, x, axis=0), axis=1)
    F1 = np.gradient(Fb, x)[3]; F2 = np.gradient(np.gradient(Fb, x), x)[3]
    return -(0.5 * F2 * Dp[3] / abs(F1) + b[3]) / F1


# ---------------------------------------------------------------- tasks
def task_profiles():
    rows, acfs = [], []
    for name, pr in promoters().items():
        sysname = pr["sys"]; Q, act = pr["Q"], pr["act"]
        xs = fixed_point(sysname, Q, act)
        x = np.arange(0.5, 120.0, 0.05)
        d = drift_split(sysname, Q, act, x)
        lF = 2 * d["pon"] * (1 - d["pon"]) / d["S0"]
        # b of the F control from its own generator: must equal b_2
        pon_fun = lambda v: activity(Q, act, v)
        QF, aF = two_state_Q(pon_fun, lamF_fun(Q, act))
        dF = drift_split(sysname, QF, aF, x)
        for i in range(len(x)):
            rows.append(dict(promoter=name, x=x[i], pon=d["pon"][i], S0=d["S0"][i], lamF=lF[i], Fbar=d["Fb"][i],
                             Dp=d["Dp"][i], b=d["b"][i], b2=d["b2"][i], R=d["R"][i], b_F=dF["b"][i]))
        j = slice(20, -20)
        print("%s x*=%.2f  max|b-(b2-R)|=%.2e  max|b_F-b2|=%.2e  max|b|=%.2f  R(x*)=%.3f" % (
            name, xs, np.max(np.abs(d["b"] - (d["b2"] - d["R"]))[j]), np.max(np.abs(dF["b"] - d["b2"])[j]),
            np.max(np.abs(d["b"][j])), np.interp(xs, x, d["R"])), flush=True)
        # promoter-output autocorrelation at x* (s = 1)
        q = Q(xs); pi, _ = fundamental(q); a = act - np.dot(pi, act)
        lam_xs = float(lamF_fun(Q, act)(xs)[0])
        for t in np.linspace(0, 6.0 / lam_xs, 241):
            CP = float(np.dot(pi * a, expm(q * t) @ a)) / float(np.dot(pi * a, a))
            acfs.append(dict(promoter=name, t=t, acf_P=CP, acf_F=np.exp(-lam_xs * t), lam_F=lam_xs))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "residual_profiles.csv"), index=False)
    pd.DataFrame(acfs).to_csv(os.path.join(OUT, "promoter_acf.csv"), index=False)


def task_exact():
    """Resumes from an existing reduction_exact.csv. Autocorrelation times are
    computed for P and for control F only (the locally matched controls have very
    large relaxation rates far from x*, which makes the matrix exponential slow,
    and only their stationary statistics are used)."""
    path = os.path.join(OUT, "reduction_exact.csv")
    rows = pd.read_csv(path).to_dict("records") if os.path.exists(path) else []
    done = {(r["promoter"], r["control"], r["s"], r["omega"]) for r in rows}
    for name, pr in promoters().items():
        sysname = pr["sys"]; Q, act = pr["Q"], pr["act"]
        xs = fixed_point(sysname, Q, act)
        pon_fun = lambda v, Q=Q, act=act: activity(Q, act, v)
        C, info = controls(pr, xs)
        lF = C["F"]
        dP = local_shift_Q(sysname, Q, act, xs)
        c1 = {k: 100 * (dP - local_shift_two_state(sysname, pon_fun, f, xs)) / xs for k, f in C.items()}
        conds = [(s, 1.0) for s in S_GRID] + [(s, om) for s in (1.0, 10.0) for om in OM_GRID]
        for s, om in conds:
            if all((name, ck, s, om) in done for ck in C):
                continue
            t0 = time.time()
            ex_P = exact(sysname, Q, act, s, om)
            N = len(ex_P["px"]); xg = np.arange(N) / om
            lFg = lF(xg)
            for ck, f in C.items():
                if (name, ck, s, om) in done:
                    continue
                ex_C = exact(sysname, None, None, s, om, lam=f, pon_fun=pon_fun, need_acf=(ck == "F"))
                ex_C.setdefault("tau_int", np.nan); ex_C.setdefault("tau_e", np.nan)
                w1, tv = distances(ex_P["px"], ex_C["px"])
                eps = float(np.dot(ex_P["px"], np.abs(np.log(f(xg) / lFg))))
                rows.append(dict(promoter=name, control=ck, s=s, omega=om, x_star=xs,
                                 mean_P=ex_P["mean"], mean_C=ex_C["mean"], cv_P=ex_P["cv"], cv_C=ex_C["cv"],
                                 dmean=100 * (ex_P["mean"] / ex_C["mean"] - 1), dcv=100 * (ex_P["cv"] / ex_C["cv"] - 1),
                                 W1=w1, TV=tv, tau_int_P=ex_P["tau_int"], tau_int_C=ex_C["tau_int"],
                                 tau_e_P=ex_P["tau_e"], tau_e_C=ex_C["tau_e"],
                                 dtau_int=100 * (ex_P["tau_int"] / ex_C["tau_int"] - 1),
                                 dtau_e=100 * (ex_P["tau_e"] / ex_C["tau_e"] - 1),
                                 eps_S0=eps, c1=c1[ck], edge_P=ex_P["edge"], edge_C=ex_C["edge"],
                                 p0_P=ex_P["px"][0], **info))
                r = rows[-1]
                print("%s %-2s s=%-5g om=%-4g dmean=%7.2f s*dmean=%7.2f c1=%7.2f dcv=%6.2f W1=%.3f dtau_int=%6.2f "
                      "dtau_e=%6.2f eps=%.3f" % (name, ck, s, om, r["dmean"], s * r["dmean"], r["c1"], r["dcv"],
                                                  w1, r["dtau_int"], r["dtau_e"], eps), flush=True)
            print("   (%.1f s)" % (time.time() - t0), flush=True)
            pd.DataFrame(rows).to_csv(path, index=False)


def task_origin():
    """Minor comment 4: sensitivity of control F to lambda_F at x = 0."""
    rows = []
    for name, pr in promoters().items():
        sysname = pr["sys"]; Q, act = pr["Q"], pr["act"]
        lim, rho = lamF_limit0(Q, act)
        pon_fun = lambda v, Q=Q, act=act: activity(Q, act, v)
        approach = {xv: float(lamF_fun(Q, act)(xv)[0]) for xv in (1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0)}
        print(name, "limit %.6g" % lim, {k: "%.6g" % v for k, v in approach.items()}, flush=True)
        for s in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
            ex_P = exact(sysname, Q, act, s, 1.0, need_acf=False)
            for tag, x0 in (("limit", None), ("1e-3", 1e-3), ("1e-2", 1e-2), ("1e-1", 1e-1), ("0.5", 0.5)):
                f = lamF_fun(Q, act, x0)
                ex_C = exact(sysname, None, None, s, 1.0, lam=f, pon_fun=pon_fun, need_acf=False)
                rows.append(dict(promoter=name, x0=tag, lamF0=float(f(np.array([0.0]))[0]), lamF_limit=lim, s=s,
                                 p0_P=ex_P["px"][0], dmean=100 * (ex_P["mean"] / ex_C["mean"] - 1),
                                 dcv=100 * (ex_P["cv"] / ex_C["cv"] - 1)))
                r = rows[-1]
                print("  %s s=%-5g x0=%-5s lamF(0)=%.4g dmean=%7.3f dcv=%7.3f  P(X=0)=%.3f" % (
                    name, s, tag, r["lamF0"], r["dmean"], r["dcv"], r["p0_P"]), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "lamF_origin.csv"), index=False)


def task_identity(seed=1):
    """Checks of the algebra of Proposition 1 and of its multilevel form on random
    promoter chains (and of the two sufficient conditions for R = 0)."""
    rng = np.random.default_rng(seed)
    c = dict(alpha=100.0, beta=5.0, gamma=1.0)
    rows = []

    def random_chain(m, levels):
        A = rng.uniform(0.2, 2.0, (m, m)); Bx = rng.uniform(0.0, 0.05, (m, m)) * (rng.random((m, m)) < 0.5)
        np.fill_diagonal(A, 0); np.fill_diagonal(Bx, 0)
        def Q(x):
            q = A + Bx * x
            return q - np.diag(q.sum(axis=1))
        return Q, levels

    def b_parts(Q, r, x):
        pis, phis = [], []
        for v in x:
            pi, Z = fundamental(Q(v))
            F = c["beta"] + c["alpha"] * r - c["gamma"] * v
            pis.append(pi); phis.append(Z @ (F - np.dot(pi, F)))
        PI = np.array(pis); PHI = np.array(phis)
        FM = c["beta"] + c["alpha"] * r[None, :] - c["gamma"] * x[:, None]
        b = np.sum(PI * FM * np.gradient(PHI, x, axis=0), axis=1)
        Dp = np.sum(PI * (FM - np.sum(PI * FM, axis=1)[:, None]) * PHI, axis=1)
        lev = np.unique(r)
        grp = np.zeros_like(b); Rm = np.zeros_like(b)
        for g in lev:
            sel = r == g
            Pg = PI[:, sel].sum(axis=1); rho = PI[:, sel] / Pg[:, None]
            Phig = np.sum(rho * PHI[:, sel], axis=1)
            Fg = c["beta"] + c["alpha"] * g - c["gamma"] * x
            grp += Fg * np.gradient(Pg, x) * Phig
            Rm += Fg * Pg * np.sum(np.gradient(rho, x, axis=0) * PHI[:, sel], axis=1)
        return b, np.gradient(Dp, x) - grp - Rm, Rm
    x = np.linspace(5.0, 40.0, 701)
    for trial in range(20):
        m = int(rng.integers(3, 7)); K = int(rng.integers(2, min(m, 4) + 1))
        levels = np.sort(rng.choice(np.linspace(0, 1, K), size=m)); levels[:K] = np.linspace(0, 1, K)
        Q, r = random_chain(m, levels)
        b, b_formula, Rm = b_parts(Q, r, x)
        j = slice(10, -10)
        rows.append(dict(test="multilevel identity", trial=trial, m=m, levels=K,
                         max_abs_err=float(np.max(np.abs(b - b_formula)[j])), max_abs_b=float(np.max(np.abs(b[j])))))
    # sufficient condition (ii): within-group correctors equal, rho regulator-dependent
    u = 1.5
    def Qii(x):
        q = np.zeros((3, 3)); q[0, 1] = 0.05 * x; q[0, 2] = 0.002 * x ** 2; q[1, 0] = u; q[2, 0] = u
        q[1, 2] = 0.3 * x; q[2, 1] = 2.0
        return q - np.diag(q.sum(axis=1))
    r = np.array([1.0, 0.0, 0.0])
    b, bf, Rm = b_parts(Qii, r, x)
    PI = np.array([fundamental(Qii(v))[0] for v in x])
    rho = PI[:, 1] / PI[:, 1:].sum(axis=1)
    rows.append(dict(test="equal exit rates (lumpable), rho varies", trial=0, m=3, levels=2,
                     max_abs_err=float(np.max(np.abs(Rm[10:-10]))), max_abs_b=float(np.max(np.abs(b[10:-10]))),
                     rho_range=float(rho.max() - rho.min())))
    df = pd.DataFrame(rows)
    print(df.to_string(), flush=True)
    df.to_csv(os.path.join(OUT, "identity_check.csv"), index=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("profiles", "all"):
        task_profiles()
    if what in ("identity", "all"):
        task_identity()
    if what in ("origin", "all"):
        task_origin()
    if what in ("exact", "all"):
        task_exact()
