"""Linear noise approximation for circuits with a stable fixed point.

State in copy-number units: promoter-state probabilities (one promoter per gene)
and molecule counts. Drift f(X) = sum_r nu_r a_r(X) with the power-law
propensities of the deterministic descriptions. At the fixed point the
covariance solves J S + S J^T + D = 0 with D = sum_r nu_r nu_r^T a_r. The
diffusion matrix splits into a promoter part (promoter transitions) and a
molecular part (all other reactions); because the equation is linear in D, the
covariance splits into S_prom + S_mol. S_prom is the variance the PDMP keeps and
S_mol the variance the sQSSA keeps, to linear order. One promoter state per block
is eliminated through the normalisation of its probabilities.
"""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from engine import simulate


def _arrays(M, omega, s):
    A = M.arrays(omega, s)
    return A


def drift_and_diffusion(M, omega, s, X):
    A = _arrays(M, omega, s)
    nS = len(M.species); nR = len(A["k"])
    a = np.zeros(nR); nu = np.zeros((nR, nS))
    for r in range(nR):
        v = A["k"][r] * omega ** A["w"][r]
        for m in range(A["fi"].shape[1]):
            j = A["fi"][r, m]
            if j < 0:
                break
            v *= (X[j] if A["isD"][j] else X[j] / omega) ** A["fe"][r, m]
        a[r] = v
        for m in range(A["ci"].shape[1]):
            j = A["ci"][r, m]
            if j < 0:
                break
            nu[r, j] += A["cv"][r, m]
    return a, nu, A["isprom"]


def fixed_point(M, omega, s):
    """Molecular concentrations from a long averaged-ODE run, promoter
    probabilities from the conditional stationary distribution."""
    from engine import _stationary
    A = M.arrays(omega, s)
    x = _full_ode_state(M, omega, s)
    _stationary(x, A["k"], A["fi"], A["fe"], A["isD"], A["isprom"], A["src"], A["tgt"], A["blk"], A["bmem"], x)
    X = np.array([x[j] if M.isD[j] else x[j] * omega for j in range(len(x))])
    return X


def _full_ode_state(M, omega, s, T=None):
    from engine import _rk4
    A = M.arrays(omega, s)
    x = np.array([A["X0"][j] if M.isD[j] else A["X0"][j] / omega for j in range(len(M.species))], float)
    xn = np.zeros_like(x)
    T = T if T is not None else 4 * M.t_total
    h = M.h * 10
    for _ in range(int(T / h)):
        _rk4(x, h, A["k"], A["fi"], A["fe"], A["ci"], A["cv"], A["isD"], A["isprom"], True,
             A["src"], A["tgt"], A["blk"], A["bmem"], xn)
        x[:] = xn
    return x


def lna(M, omega, s, obs_i=0):
    X = fixed_point(M, omega, s)
    nS = len(X)
    # reduced coordinates: drop the first state of every promoter block
    drop = [b[0] for b in M.blocks]
    keep = [j for j in range(nS) if j not in drop]

    def full(Xr):
        Xf = np.zeros(nS); Xf[keep] = Xr
        for b in M.blocks:
            Xf[b[0]] = 1.0 - sum(Xf[j] for j in b[1:])
        return Xf

    def f(Xr):
        a, nu, _ = drift_and_diffusion(M, omega, s, full(Xr))
        return (nu.T @ a)[keep]

    Xr = X[keep]
    J = np.zeros((len(keep), len(keep)))
    for i in range(len(keep)):
        dh = 1e-6 * max(abs(Xr[i]), 1e-3)
        e = np.zeros(len(keep)); e[i] = dh
        J[:, i] = (f(Xr + e) - f(Xr - e)) / (2 * dh)
    a, nu, isprom = drift_and_diffusion(M, omega, s, X)
    nuk = nu[:, keep]
    Dp = (nuk[isprom].T * a[isprom]) @ nuk[isprom]
    Dm = (nuk[~isprom].T * a[~isprom]) @ nuk[~isprom]
    if np.max(np.real(np.linalg.eigvals(J))) >= 0:
        return None
    Sp = solve_continuous_lyapunov(J, -Dp)
    Sm = solve_continuous_lyapunov(J, -Dm)
    obs = keep.index(M.observe[obs_i])
    vp, vm = Sp[obs, obs], Sm[obs, obs]
    return dict(mean=X[M.observe[obs_i]], var_prom=vp, var_mol=vm, var=vp + vm, share_prom=vp / (vp + vm))


if __name__ == "__main__":
    from systems2 import MODELS, BASE_OMEGA
    import time
    for name in ("NAR-H", "NAR-S", "Cascade-H", "Cascade-S", "p53-S", "p53-H"):
        M = MODELS[name](); om = BASE_OMEGA[name.split("-")[0]]
        t0 = time.time(); r = lna(M, om, 1.0)
        print("%-10s mean %.4g  var %.4g  share_prom %.3f  (%.1fs)" % (name, r["mean"], r["var"], r["share_prom"], time.time() - t0))


def lna_sqssa(M, omega, s, obs_i=0):
    """LNA of the sQSSA itself: molecular species only, promoter indicators
    replaced by the conditional stationary probabilities pi(x). Returns the
    variance of the observed species in copy-number units."""
    from engine import _stationary
    A = M.arrays(omega, s)
    X = fixed_point(M, omega, s)
    mol = [j for j in range(len(M.species)) if not M.isD[j]]

    def full_state(Xm):
        Xf = np.zeros(len(M.species)); Xf[mol] = Xm
        x = np.array([Xf[j] / omega if not M.isD[j] else 0.0 for j in range(len(Xf))])
        _stationary(x, A["k"], A["fi"], A["fe"], A["isD"], A["isprom"], A["src"], A["tgt"], A["blk"], A["bmem"], x)
        for j in range(len(Xf)):
            if M.isD[j]:
                Xf[j] = x[j]
        return Xf

    def flux(Xm):
        a, nu, isprom = drift_and_diffusion(M, omega, s, full_state(Xm))
        a = np.where(isprom, 0.0, a)
        return a, nu[:, mol]

    Xm = X[mol]
    f = lambda Xm: (flux(Xm)[1].T @ flux(Xm)[0])
    J = np.zeros((len(mol), len(mol)))
    for i in range(len(mol)):
        dh = 1e-6 * max(abs(Xm[i]), 1e-3)
        e = np.zeros(len(mol)); e[i] = dh
        J[:, i] = (f(Xm + e) - f(Xm - e)) / (2 * dh)
    a, nu = flux(Xm)
    D = (nu.T * a) @ nu
    S = solve_continuous_lyapunov(J, -D)
    return S[mol.index(M.observe[obs_i]), mol.index(M.observe[obs_i])]


def _ssa_flux(M, omega, s, X):
    """Drift sum_r nu_r a_r(X) with the SSA's own propensities (falling factorials
    where the reaction uses them), in copy-number units."""
    A = _arrays(M, omega, s)
    nS = len(M.species); nR = len(A["k"])
    f = np.zeros(nS)
    for r in range(nR):
        v = A["k"][r] * omega ** A["w"][r]
        for m in range(A["fi"].shape[1]):
            j = A["fi"][r, m]
            if j < 0:
                break
            e = A["fe"][r, m]
            if A["isD"][j]:
                v *= X[j] ** e
            elif A["ff"][r] and e >= 2 and e == int(e):
                for q in range(int(e)):
                    v *= (X[j] - q) / omega
            else:
                v *= (X[j] / omega) ** e
        for m in range(A["ci"].shape[1]):
            j = A["ci"][r, m]
            if j < 0:
                break
            f[j] += A["cv"][r, m] * v
    return f


def emre(M, omega, s, obs_i=0):
    """Next-order correction to the mean (effective mesoscopic rate equation,
    Grima 2010) in the moment-linearised promoter description:
    0 = f(X*) + J d + (1/2) H : C, with f the SSA drift (falling factorials
    included), J and H its first and second derivatives at the deterministic
    fixed point X*, and C the LNA covariance. Returns d for the observed species
    relative to X*, a quantity computed from the model alone."""
    X = fixed_point(M, omega, s)
    nS = len(X)
    drop = [b[0] for b in M.blocks]
    keep = [j for j in range(nS) if j not in drop]

    def full(Xr):
        Xf = np.zeros(nS); Xf[keep] = Xr
        for b in M.blocks:
            Xf[b[0]] = 1.0 - sum(Xf[j] for j in b[1:])
        return Xf

    f = lambda Xr: _ssa_flux(M, omega, s, full(Xr))[keep]
    Xr = X[keep]; n = len(keep)
    hs = np.array([1e-4 * max(abs(v), 1.0) if keep[i] not in sum(M.blocks, []) else 1e-4
                   for i, v in enumerate(Xr)])
    J = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n); e[i] = hs[i]
        J[:, i] = (f(Xr + e) - f(Xr - e)) / (2 * hs[i])
    Hc = np.zeros(n)
    r = lna_cov(M, omega, s)
    C = r
    f0 = f(Xr)
    for i in range(n):
        for k in range(n):
            if C[i, k] == 0:
                continue
            ei = np.zeros(n); ei[i] = hs[i]; ek = np.zeros(n); ek[k] = hs[k]
            d2 = (f(Xr + ei + ek) - f(Xr + ei - ek) - f(Xr - ei + ek) + f(Xr - ei - ek)) / (4 * hs[i] * hs[k])
            Hc += 0.5 * d2 * C[i, k]
    d = np.linalg.solve(J, -(f0 + Hc))
    o = keep.index(M.observe[obs_i])
    return dict(x_star=Xr[o], delta=d[o], rel=d[o] / Xr[o])


def lna_cov(M, omega, s):
    """Full LNA covariance in the reduced coordinates used by emre()."""
    X = fixed_point(M, omega, s)
    nS = len(X)
    drop = [b[0] for b in M.blocks]
    keep = [j for j in range(nS) if j not in drop]

    def full(Xr):
        Xf = np.zeros(nS); Xf[keep] = Xr
        for b in M.blocks:
            Xf[b[0]] = 1.0 - sum(Xf[j] for j in b[1:])
        return Xf

    def f(Xr):
        a, nu, _ = drift_and_diffusion(M, omega, s, full(Xr))
        return (nu.T @ a)[keep]
    Xr = X[keep]
    J = np.zeros((len(keep), len(keep)))
    for i in range(len(keep)):
        dh = 1e-6 * max(abs(Xr[i]), 1e-3)
        e = np.zeros(len(keep)); e[i] = dh
        J[:, i] = (f(Xr + e) - f(Xr - e)) / (2 * dh)
    a, nu, _ = drift_and_diffusion(M, omega, s, X)
    nuk = nu[:, keep]
    D = (nuk.T * a) @ nuk
    return solve_continuous_lyapunov(J, -D)
