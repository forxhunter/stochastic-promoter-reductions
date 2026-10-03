"""Simulation engine for the promoter-reduction benchmark suite.

A model is one microscopic reaction network with explicit promoter states. Four
descriptions are derived from the same definition, so that every comparison is
between approximations of one model:

  ssa    Gillespie direct method on the full network (reference).
  pdmp   promoter stochastic, downstream species deterministic. Promoter
         transition times are drawn by integrating the total promoter hazard
         along the deterministic flow (RK4 with step h, crossing located by
         linear interpolation of the integrated hazard).
  sqssa  promoter replaced by its stationary distribution given the current copy
         numbers, computed with the SSA's own propensities; downstream reactions
         remain stochastic with averaged propensities.
  ode    promoter replaced by pi(x); downstream deterministic.

Propensity convention (all methods):
    a_r = k_r * Omega**w_r * prod_{j in C} (X_j/Omega)**e_rj * prod_{j in D} X_j**e_rj
In the SSA an integer exponent e >= 2 on a molecular species is taken as the
falling factorial X(X-1)...(X-e+1)/Omega**e, the mass-action propensity of an
e-molecular step; non-integer exponents (phenomenological Hill orders) keep the
power law. The PDMP, sQSSA and ODE use concentrations and hence x**e.
where C are molecular species and D are promoter-state indicator species (one
promoter per gene, independent of Omega). Downstream reactions have w_r = 1 and
change only C; promoter transitions have w_r = 0 and move one indicator from a
source state to a target state within one promoter block.
"""

import numpy as np
from numba import njit


class Model:
    def __init__(self, name, species, promoter_blocks, reactions, x0, observe,
                 record_dt, t_burn, t_total):
        self.name = name
        self.species = list(species)
        self.idx = {s: i for i, s in enumerate(self.species)}
        self.blocks = [[self.idx[s] for s in b] for b in promoter_blocks]
        self.isD = np.zeros(len(species), dtype=np.bool_)
        for b in self.blocks:
            for j in b:
                self.isD[j] = True
        self.reactions = reactions          # list of dicts: k, w, reac{sp:exp}, change{sp:int}
        self.x0 = x0                        # dict: species -> concentration (C) or 0/1 (D)
        self.observe = [self.idx[s] for s in observe]
        self.record_dt = record_dt
        self.t_burn = t_burn
        self.t_total = t_total

    def arrays(self, omega, s_prom=1.0):
        """Flatten to numba-friendly arrays. s_prom multiplies every promoter
        transition rate (eta scales as 1/s_prom)."""
        nR, nS = len(self.reactions), len(self.species)
        maxf = max(len(r["reac"]) for r in self.reactions)
        k = np.zeros(nR); w = np.zeros(nR)
        fi = -np.ones((nR, maxf), dtype=np.int64); fe = np.zeros((nR, maxf))
        ci = -np.ones((nR, 3), dtype=np.int64); cv = np.zeros((nR, 3), dtype=np.int64)
        isprom = np.zeros(nR, dtype=np.bool_)
        ff = np.ones(nR, dtype=np.bool_)
        for r, rx in enumerate(self.reactions):
            k[r] = rx["k"]; w[r] = rx["w"]
            ff[r] = rx.get("ff", True)
            if rx["w"] == 0:
                isprom[r] = True
                k[r] *= s_prom
            for m, (sp, e) in enumerate(rx["reac"].items()):
                fi[r, m] = self.idx[sp]; fe[r, m] = e
            for m, (sp, d) in enumerate(rx["change"].items()):
                ci[r, m] = self.idx[sp]; cv[r, m] = d
        # promoter structure: for each promoter reaction, its source and target state
        src = -np.ones(nR, dtype=np.int64); tgt = -np.ones(nR, dtype=np.int64)
        for r, rx in enumerate(self.reactions):
            if rx["w"] == 0:
                for sp, d in rx["change"].items():
                    if not self.isD[self.idx[sp]]:
                        continue            # e.g. a regulator sequestered on binding
                    if d == -1: src[r] = self.idx[sp]
                    if d == +1: tgt[r] = self.idx[sp]
        blk = -np.ones(nS, dtype=np.int64)
        for b, members in enumerate(self.blocks):
            for j in members:
                blk[j] = b
        nb = len(self.blocks); bmax = max(len(b) for b in self.blocks)
        bmem = -np.ones((nb, bmax), dtype=np.int64)
        for b, members in enumerate(self.blocks):
            bmem[b, :len(members)] = members
        X0 = np.zeros(nS)
        for sp, v in self.x0.items():
            X0[self.idx[sp]] = v if self.isD[self.idx[sp]] else float(np.round(v * omega))
        return dict(k=k, w=w, fi=fi, fe=fe, ci=ci, cv=cv, isprom=isprom, isD=self.isD, ff=ff,
                    src=src, tgt=tgt, blk=blk, bmem=bmem, X0=X0,
                    obs=np.array(self.observe, dtype=np.int64), omega=float(omega))


# ---------------------------------------------------------------- propensities

@njit(cache=True)
def _prop(r, X, k, w, fi, fe, isD, omega, ff):
    a = k[r] * omega ** w[r]
    for m in range(fi.shape[1]):
        j = fi[r, m]
        if j < 0:
            break
        if isD[j]:
            if fe[r, m] != 0.0:
                a *= X[j] ** fe[r, m]
        else:
            e = fe[r, m]
            if ff[r] and e >= 2.0 and e == np.floor(e):
                # integer order > 1: falling factorial, the mass-action
                # propensity of an e-molecular step
                v = 1.0
                for q in range(int(e)):
                    v *= max(X[j] - q, 0.0) / omega
                a *= v
            else:
                a *= (X[j] / omega) ** e
    return a


@njit(cache=True)
def _prop_conc(r, x, k, fi, fe, isD):
    """a_r / Omega**w_r with x holding concentrations for C and indicators or
    stationary probabilities for D."""
    a = k[r]
    for m in range(fi.shape[1]):
        j = fi[r, m]
        if j < 0:
            break
        a *= x[j] ** fe[r, m]
    return a


@njit(cache=True)
def _stationary(x, k, fi, fe, isD, isprom, src, tgt, blk, bmem, out):
    """Fill out[j] for promoter indicator species j with the stationary
    probability of that state given the concentrations in x."""
    nb = bmem.shape[0]
    for b in range(nb):
        n = 0
        for m in range(bmem.shape[1]):
            if bmem[b, m] >= 0:
                n += 1
        Q = np.zeros((n, n))
        for r in range(k.shape[0]):
            if not isprom[r] or blk[src[r]] != b:
                continue
            # rate with the source indicator set to 1
            a = k[r]
            for m in range(fi.shape[1]):
                j = fi[r, m]
                if j < 0:
                    break
                if not isD[j]:
                    a *= x[j] ** fe[r, m]
            i1 = -1; i2 = -1
            for m in range(n):
                if bmem[b, m] == src[r]: i1 = m
                if bmem[b, m] == tgt[r]: i2 = m
            Q[i1, i2] += a
            Q[i1, i1] -= a
        # solve pi Q = 0, sum pi = 1  ->  A pi = rhs with A = Q^T, last row ones
        A = Q.T.copy()
        rhs = np.zeros(n)
        for m in range(n):
            A[n - 1, m] = 1.0
        rhs[n - 1] = 1.0
        # Gaussian elimination with partial pivoting
        for c in range(n):
            p = c
            for rr in range(c + 1, n):
                if abs(A[rr, c]) > abs(A[p, c]):
                    p = rr
            if p != c:
                for cc in range(n):
                    t = A[c, cc]; A[c, cc] = A[p, cc]; A[p, cc] = t
                t = rhs[c]; rhs[c] = rhs[p]; rhs[p] = t
            for rr in range(c + 1, n):
                f = A[rr, c] / A[c, c]
                for cc in range(c, n):
                    A[rr, cc] -= f * A[c, cc]
                rhs[rr] -= f * rhs[c]
        pi = np.zeros(n)
        for c in range(n - 1, -1, -1):
            s = rhs[c]
            for cc in range(c + 1, n):
                s -= A[c, cc] * pi[cc]
            pi[c] = s / A[c, c]
        for m in range(n):
            out[bmem[b, m]] = max(pi[m], 0.0)


@njit(cache=True)
def _stationary_counts(X, k, fi, fe, isD, isprom, src, tgt, blk, bmem, omega, ff, out):
    """As _stationary, but promoter rates are evaluated from copy numbers with the
    SSA's own propensities (falling factorials for integer orders), so that the
    sQSSA is the fast-switching limit of the SSA itself."""
    nb = bmem.shape[0]
    for b in range(nb):
        n = 0
        for m in range(bmem.shape[1]):
            if bmem[b, m] >= 0:
                n += 1
        Q = np.zeros((n, n))
        for r in range(k.shape[0]):
            if not isprom[r] or blk[src[r]] != b:
                continue
            a = k[r]
            for m in range(fi.shape[1]):
                j = fi[r, m]
                if j < 0:
                    break
                if isD[j]:
                    continue
                e = fe[r, m]
                if ff[r] and e >= 2.0 and e == np.floor(e):
                    v = 1.0
                    for q in range(int(e)):
                        v *= max(X[j] - q, 0.0) / omega
                    a *= v
                else:
                    a *= (X[j] / omega) ** e
            i1 = -1; i2 = -1
            for m in range(n):
                if bmem[b, m] == src[r]: i1 = m
                if bmem[b, m] == tgt[r]: i2 = m
            Q[i1, i2] += a
            Q[i1, i1] -= a
        A = Q.T.copy()
        rhs = np.zeros(n)
        for m in range(n):
            A[n - 1, m] = 1.0
        rhs[n - 1] = 1.0
        for c in range(n):
            p = c
            for rr in range(c + 1, n):
                if abs(A[rr, c]) > abs(A[p, c]):
                    p = rr
            if p != c:
                for cc in range(n):
                    t = A[c, cc]; A[c, cc] = A[p, cc]; A[p, cc] = t
                t = rhs[c]; rhs[c] = rhs[p]; rhs[p] = t
            for rr in range(c + 1, n):
                f = A[rr, c] / A[c, c]
                for cc in range(c, n):
                    A[rr, cc] -= f * A[c, cc]
                rhs[rr] -= f * rhs[c]
        pi = np.zeros(n)
        for c in range(n - 1, -1, -1):
            sm = rhs[c]
            for cc in range(c + 1, n):
                sm -= A[c, cc] * pi[cc]
            pi[c] = sm / A[c, c]
        for m in range(n):
            out[bmem[b, m]] = max(pi[m], 0.0)


# ------------------------------------------------------------------------- SSA

@njit(cache=True)
def ssa(X0, k, w, fi, fe, ci, cv, isD, omega, obs, dt_rec, n_rec, seed, averaged,
        isprom, src, tgt, blk, bmem, ff):
    """Direct method. If averaged, promoter reactions are skipped and each
    downstream propensity uses the stationary promoter distribution (sQSSA)."""
    np.random.seed(seed)
    X = X0.copy()
    nR = k.shape[0]
    out = np.zeros((n_rec, obs.shape[0]))
    a = np.zeros(nR)
    x = np.zeros(X.shape[0])
    t = 0.0
    irec = 0
    nev = 0
    while irec < n_rec:
        if averaged:
            for j in range(X.shape[0]):
                x[j] = X[j] / omega if not isD[j] else 0.0
            _stationary_counts(X, k, fi, fe, isD, isprom, src, tgt, blk, bmem, omega, ff, x)
            for r in range(nR):
                if isprom[r]:
                    a[r] = 0.0
                else:
                    a[r] = omega * _prop_conc(r, x, k, fi, fe, isD)
        else:
            for r in range(nR):
                a[r] = _prop(r, X, k, w, fi, fe, isD, omega, ff)
        a0 = 0.0
        for r in range(nR):
            a0 += a[r]
        tau = -np.log(np.random.random()) / a0 if a0 > 0 else 1e300
        tn = t + tau
        while irec < n_rec and irec * dt_rec <= tn:
            for m in range(obs.shape[0]):
                out[irec, m] = X[obs[m]] / omega
            irec += 1
        if irec >= n_rec:
            break
        u = np.random.random() * a0
        c = 0.0
        rsel = nR - 1
        for r in range(nR):
            c += a[r]
            if u < c:
                rsel = r
                break
        for m in range(ci.shape[1]):
            j = ci[rsel, m]
            if j < 0:
                break
            X[j] += cv[rsel, m]
        t = tn
        nev += 1
    return out, nev


# ------------------------------------------------------------- PDMP and ODE

@njit(cache=True)
def _drift(x, k, fi, fe, ci, cv, isD, isprom, dx):
    for j in range(x.shape[0]):
        dx[j] = 0.0
    for r in range(k.shape[0]):
        if isprom[r]:
            continue
        a = _prop_conc(r, x, k, fi, fe, isD)
        for m in range(ci.shape[1]):
            j = ci[r, m]
            if j < 0:
                break
            dx[j] += cv[r, m] * a


@njit(cache=True)
def _hazard(x, k, fi, fe, isD, isprom):
    h = 0.0
    for r in range(k.shape[0]):
        if isprom[r]:
            h += _prop_conc(r, x, k, fi, fe, isD)
    return h


@njit(cache=True)
def _rk4(x, h, k, fi, fe, ci, cv, isD, isprom, averaged, src, tgt, blk, bmem, xn):
    """One RK4 step of the downstream flow. Returns the integrated promoter
    hazard over the step (Simpson rule on the RK4 stages)."""
    n = x.shape[0]
    k1 = np.zeros(n); k2 = np.zeros(n); k3 = np.zeros(n); k4 = np.zeros(n)
    y = x.copy()
    if averaged:
        _stationary(y, k, fi, fe, isD, isprom, src, tgt, blk, bmem, y)
    h0 = 0.0 if averaged else _hazard(y, k, fi, fe, isD, isprom)
    _drift(y, k, fi, fe, ci, cv, isD, isprom, k1)
    for j in range(n):
        y[j] = x[j] + 0.5 * h * k1[j] if not isD[j] else x[j]
    if averaged:
        _stationary(y, k, fi, fe, isD, isprom, src, tgt, blk, bmem, y)
    hm = 0.0 if averaged else _hazard(y, k, fi, fe, isD, isprom)
    _drift(y, k, fi, fe, ci, cv, isD, isprom, k2)
    for j in range(n):
        y[j] = x[j] + 0.5 * h * k2[j] if not isD[j] else x[j]
    if averaged:
        _stationary(y, k, fi, fe, isD, isprom, src, tgt, blk, bmem, y)
    _drift(y, k, fi, fe, ci, cv, isD, isprom, k3)
    for j in range(n):
        y[j] = x[j] + h * k3[j] if not isD[j] else x[j]
    if averaged:
        _stationary(y, k, fi, fe, isD, isprom, src, tgt, blk, bmem, y)
    h1 = 0.0 if averaged else _hazard(y, k, fi, fe, isD, isprom)
    _drift(y, k, fi, fe, ci, cv, isD, isprom, k4)
    for j in range(n):
        if isD[j]:
            xn[j] = x[j]
        else:
            xn[j] = max(x[j] + h / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]), 0.0)
    return h / 6.0 * (h0 + 4 * hm + h1)


@njit(cache=True)
def pdmp(X0, k, fi, fe, ci, cv, isD, isprom, src, tgt, blk, bmem, omega, obs,
         dt_rec, n_rec, h, seed, averaged):
    """Piecewise-deterministic simulation (averaged=False) or the averaged ODE
    (averaged=True). X0 holds counts; state is converted to concentrations."""
    np.random.seed(seed)
    n = X0.shape[0]
    x = np.zeros(n)
    for j in range(n):
        x[j] = X0[j] if isD[j] else X0[j] / omega
    xn = np.zeros(n)
    out = np.zeros((n_rec, obs.shape[0]))
    nsub = max(1, int(round(dt_rec / h)))
    hh = dt_rec / nsub
    E = -np.log(np.random.random())
    L = 0.0
    njump = 0
    xa = np.zeros(n)
    for irec in range(n_rec):
        if averaged:
            for j in range(n):
                xa[j] = x[j]
            _stationary(xa, k, fi, fe, isD, isprom, src, tgt, blk, bmem, xa)
        for m in range(obs.shape[0]):
            out[irec, m] = xa[obs[m]] if averaged else x[obs[m]]
        rem = hh * nsub
        while rem > 1e-12:
            step = min(hh, rem)
            dL = _rk4(x, step, k, fi, fe, ci, cv, isD, isprom, averaged, src, tgt, blk, bmem, xn)
            if averaged or L + dL < E:
                L += dL
                for j in range(n):
                    x[j] = xn[j]
                rem -= step
                continue
            # a promoter transition occurs inside this step
            theta = (E - L) / dL
            part = theta * step
            _rk4(x, part, k, fi, fe, ci, cv, isD, isprom, averaged, src, tgt, blk, bmem, xn)
            for j in range(n):
                x[j] = xn[j]
            rem -= part
            # choose which promoter reaction fired
            tot = 0.0
            for r in range(k.shape[0]):
                if isprom[r]:
                    tot += _prop_conc(r, x, k, fi, fe, isD)
            u = np.random.random() * tot
            c = 0.0
            for r in range(k.shape[0]):
                if isprom[r]:
                    c += _prop_conc(r, x, k, fi, fe, isD)
                    if u < c:
                        x[src[r]] = 0.0
                        x[tgt[r]] = 1.0
                        # molecular species changed by the transition (sequestration)
                        for m in range(ci.shape[1]):
                            j = ci[r, m]
                            if j < 0:
                                break
                            if not isD[j]:
                                x[j] = max(x[j] + cv[r, m] / omega, 0.0)
                        break
            njump += 1
            L = 0.0
            E = -np.log(np.random.random())
    return out, njump


def simulate(model, method, omega, s_prom, seed, n_rec=None, h=None):
    A = model.arrays(omega, s_prom)
    if n_rec is None:
        n_rec = int(round(model.t_total / model.record_dt)) + 1
    if method in ("ssa", "sqssa"):
        out, nev = ssa(A["X0"], A["k"], A["w"], A["fi"], A["fe"], A["ci"], A["cv"], A["isD"],
                       A["omega"], A["obs"], model.record_dt, n_rec, seed, method == "sqssa",
                       A["isprom"], A["src"], A["tgt"], A["blk"], A["bmem"], A["ff"])
        return out, nev
    if method in ("pdmp", "ode"):
        hstep = h if h is not None else model.h
        out, nj = pdmp(A["X0"], A["k"], A["fi"], A["fe"], A["ci"], A["cv"], A["isD"], A["isprom"],
                       A["src"], A["tgt"], A["blk"], A["bmem"], A["omega"], A["obs"],
                       model.record_dt, n_rec, hstep, seed, method == "ode")
        return out, nj
    if method == "pdmpx":
        from pdmp_exact import simulate_exact
        return simulate_exact(model, omega, s_prom, seed, n_rec=n_rec)
    raise ValueError(method)


def eta(model, s_prom, xs, tau_prot):
    """Spectral gap of each promoter block evaluated at concentration vectors xs
    (rows); returns eta = (1/lambda2)/tau_prot for the slowest block at each row."""
    A = model.arrays(1.0, s_prom)
    out = []
    for x in xs:
        lam_min = np.inf
        for b, members in enumerate(model.blocks):
            n = len(members)
            Q = np.zeros((n, n))
            for r in range(len(A["k"])):
                if not A["isprom"][r] or A["blk"][A["src"][r]] != b:
                    continue
                a = A["k"][r]
                for m in range(A["fi"].shape[1]):
                    j = A["fi"][r, m]
                    if j < 0:
                        break
                    if not A["isD"][j]:
                        a *= x[j] ** A["fe"][r, m]
                i1 = members.index(A["src"][r]); i2 = members.index(A["tgt"][r])
                Q[i1, i2] += a; Q[i1, i1] -= a
            ev = np.sort(np.abs(np.real(np.linalg.eigvals(Q))))
            lam_min = min(lam_min, ev[1])
        out.append((1.0 / lam_min) / tau_prot)
    return np.array(out)
