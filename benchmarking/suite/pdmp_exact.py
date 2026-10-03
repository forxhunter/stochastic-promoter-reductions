"""PDMP sampled with exact flows, for models whose molecular dynamics between
promoter transitions are linear: every molecular reaction is either production
(no molecular reactant; rate constant, possibly times a promoter indicator) or
first-order degradation of one species. Between jumps each species then follows
x_j(t) = c_j + (x_j(0) - c_j) exp(-g_j t) with c_j = b_j / g_j, b_j the production
rate in the current promoter state. The cumulative promoter hazard is integrated
with 8-point Gauss-Legendre quadrature on chunks of length at most 0.05/max(g),
and the jump time is found by Newton iteration on the exact cumulative hazard,
so jump times are located to quadrature precision and do not depend on an
integration step. Transitions that also change a molecular species (a
sequestering site) update it by the stoichiometry divided by Omega.
"""
import numpy as np
from numba import njit

GL_X, GL_W = np.polynomial.legendre.leggauss(8)


def linear_structure(M, omega, s):
    """Extract (degradation rates, production terms, promoter reactions) and check
    that the molecular dynamics are linear."""
    A = M.arrays(omega, s)
    nS = len(M.species)
    g = np.zeros(nS)
    prod = []                              # (target species, rate, indicator or -1)
    for r in range(len(A["k"])):
        if A["isprom"][r]:
            continue
        reac = [(A["fi"][r, m], A["fe"][r, m]) for m in range(A["fi"].shape[1]) if A["fi"][r, m] >= 0]
        ch = [(A["ci"][r, m], A["cv"][r, m]) for m in range(A["ci"].shape[1]) if A["ci"][r, m] >= 0]
        mol = [(j, e) for j, e in reac if not A["isD"][j]]
        ind = [j for j, e in reac if A["isD"][j]]
        if len(mol) == 1 and mol[0][1] == 1 and len(ch) == 1 and ch[0] == (mol[0][0], -1):
            g[mol[0][0]] += A["k"][r]
        elif len(mol) == 0 and len(ch) == 1 and ch[0][1] == 1 and len(ind) <= 1:
            prod.append((ch[0][0], A["k"][r], ind[0] if ind else -1))
        else:
            raise ValueError("reaction %d is not linear production or degradation" % r)
    P = np.array(prod, dtype=float)
    return A, g, P


@njit(cache=True)
def _xt(x0, c, g, isD, t, out):
    for j in range(x0.shape[0]):
        if isD[j] or g[j] == 0.0:
            out[j] = x0[j] + (c[j] * t if (not isD[j] and g[j] == 0.0) else 0.0)
        else:
            out[j] = c[j] + (x0[j] - c[j]) * np.exp(-g[j] * t)


@njit(cache=True)
def _haz(x, k, fi, fe, isD, isprom):
    h = 0.0
    for r in range(k.shape[0]):
        if not isprom[r]:
            continue
        a = k[r]
        for m in range(fi.shape[1]):
            j = fi[r, m]
            if j < 0:
                break
            a *= x[j] ** fe[r, m]
        h += a
    return h


@njit(cache=True)
def _inputs(x, P, isD, nS, c, g):
    b = np.zeros(nS)
    for q in range(P.shape[0]):
        j = int(P[q, 0]); ind = int(P[q, 2])
        b[j] += P[q, 1] * (x[ind] if ind >= 0 else 1.0)
    for j in range(nS):
        if not isD[j]:
            c[j] = b[j] / g[j] if g[j] > 0 else b[j]


@njit(cache=True)
def _cum(x0, c, g, isD, k, fi, fe, isprom, t0, t1, glx, glw, tmp):
    """Integral of the hazard from t0 to t1 (times measured from the last jump)."""
    half = 0.5 * (t1 - t0); mid = 0.5 * (t1 + t0); acc = 0.0
    for i in range(glx.shape[0]):
        _xt(x0, c, g, isD, mid + half * glx[i], tmp)
        acc += glw[i] * _haz(tmp, k, fi, fe, isD, isprom)
    return acc * half


@njit(cache=True)
def run(X0c, k, fi, fe, ci, cv, isD, isprom, src, tgt, P, g, omega, obs, dt_rec, n_rec, seed, glx, glw):
    np.random.seed(seed)
    nS = X0c.shape[0]
    x0 = X0c.copy()                       # state at the last jump
    c = np.zeros(nS); tmp = np.zeros(nS); xx = np.zeros(nS)
    _inputs(x0, P, isD, nS, c, g)
    out = np.zeros((n_rec, obs.shape[0]))
    gmax = 0.0
    for j in range(nS):
        if g[j] > gmax:
            gmax = g[j]
    chunk = 0.05 / gmax
    T = 0.0                               # absolute time of the last jump
    tau = 0.0                             # time since the last jump
    E = -np.log(np.random.random()); L = 0.0
    irec = 0; njump = 0
    Tend = (n_rec - 1) * dt_rec
    while irec < n_rec:
        # next boundary: chunk end or the next record time
        t_next_rec = irec * dt_rec - T
        t1 = min(tau + chunk, t_next_rec) if t_next_rec > tau else tau + chunk
        if t_next_rec <= tau + 1e-15:
            _xt(x0, c, g, isD, tau, xx)
            for m in range(obs.shape[0]):
                out[irec, m] = xx[obs[m]]
            irec += 1
            continue
        dL = _cum(x0, c, g, isD, k, fi, fe, isprom, tau, t1, glx, glw, tmp)
        if L + dL < E:
            L += dL; tau = t1
            continue
        # jump inside (tau, t1]: Newton on F(t) = L + int_tau^t H - E
        lo = tau; hi = t1; t = tau + (E - L) / max(dL / (t1 - tau), 1e-300)
        if t <= lo or t >= hi:
            t = 0.5 * (lo + hi)
        for it in range(60):
            F = L + _cum(x0, c, g, isD, k, fi, fe, isprom, tau, t, glx, glw, tmp) - E
            if abs(F) < 1e-12:
                break
            if F > 0:
                hi = t
            else:
                lo = t
            _xt(x0, c, g, isD, t, tmp)
            dF = _haz(tmp, k, fi, fe, isD, isprom)
            tn = t - F / dF if dF > 0 else 0.5 * (lo + hi)
            if tn <= lo or tn >= hi:
                tn = 0.5 * (lo + hi)
            t = tn
        # state at the jump
        _xt(x0, c, g, isD, t, xx)
        tot = _haz(xx, k, fi, fe, isD, isprom)
        u = np.random.random() * tot; acc = 0.0
        for r in range(k.shape[0]):
            if not isprom[r]:
                continue
            a = k[r]
            for m in range(fi.shape[1]):
                j = fi[r, m]
                if j < 0:
                    break
                a *= xx[j] ** fe[r, m]
            acc += a
            if u < acc:
                xx[src[r]] = 0.0; xx[tgt[r]] = 1.0
                for m in range(ci.shape[1]):
                    j = ci[r, m]
                    if j < 0:
                        break
                    if not isD[j]:
                        xx[j] = max(xx[j] + cv[r, m] / omega, 0.0)
                break
        T += t
        # records falling before the jump time were handled by the chunk boundary logic
        for j in range(nS):
            x0[j] = xx[j]
        _inputs(x0, P, isD, nS, c, g)
        tau = 0.0; L = 0.0; E = -np.log(np.random.random()); njump += 1
    return out, njump


def simulate_exact(M, omega, s, seed, n_rec=None):
    A, g, P = linear_structure(M, omega, s)
    if n_rec is None:
        n_rec = int(round(M.t_total / M.record_dt)) + 1
    X0c = np.array([A["X0"][j] if M.isD[j] else A["X0"][j] / omega for j in range(len(M.species))])
    return run(X0c, A["k"], A["fi"], A["fe"], A["ci"], A["cv"], A["isD"], A["isprom"], A["src"], A["tgt"],
               P, g, float(omega), A["obs"], M.record_dt, n_rec, seed, GL_X, GL_W)
