"""Benchmark circuits, second design: each regulated promoter exists in two
architectures with approximately the same stationary response.

  H  concerted two-state promoter whose regulator-dependent rate carries the
     Hill order n (the promoter a Hill-type ODE implicitly assumes). The step is
     n-molecular; the SSA uses the falling factorial for integer n.
  S  n elementary binding sites. Binding k -> k+1 at (n-k) k_b x/K_S, unbinding
     k+1 -> k at (k+1) k_b omega^(-k): bound regulators stabilise each other with
     pairwise factor omega, and pi_k is proportional to C(n,k) r^k omega^(k(k-1)/2).
     A repressed promoter transcribes unless all n sites are bound; an activated
     one transcribes when all n are bound.

For S, omega = 100, K_S is set so that the half-maximum of the response equals
that of H, and k_b is set so that the slowest promoter relaxation over the
relevant regulator range equals that of H (matched worst-case eta). The p53
circuit is the exception: its original promoter is already site-resolved (three
sites, occupancy-proportional readout), and its H version is a two-state
promoter whose Hill function is fitted to that response.
"""
import numpy as np
from math import comb
from scipy.optimize import brentq, curve_fit
from engine import Model

OMEGA_COOP = 100.0


def R(k, w, reac, change):
    return dict(k=float(k), w=w, reac=reac, change=change)


# ------------------------------------------------------------ promoter blocks

def hill_block(tag, reg, n, K, kb, repress, alpha, product, extra_rx=None, plain=False):
    off, on = tag + "_off", tag + "_on"
    rx = []
    if repress:          # on -> off at kb (x/K)^n ; off -> on at kb
        rx.append(R(kb, 0, {off: 1}, {off: -1, on: +1}))
        rx.append(dict(R(kb * K ** -n, 0, {on: 1, reg: n}, {on: -1, off: +1}), ff=not plain))
    else:                # off -> on at kb (x/K)^n ; on -> off at kb
        rx.append(dict(R(kb * K ** -n, 0, {off: 1, reg: n}, {off: -1, on: +1}), ff=not plain))
        rx.append(R(kb, 0, {on: 1}, {on: -1, off: +1}))
    rx.append(R(alpha, 1, {on: 1}, {product: +1}))
    return [off, on], rx, {on: 1}


def site_block(tag, reg, n, KS, kb, omega, repress, alpha, product, sequester=False):
    """sequester=True: binding consumes one free regulator molecule and release
    returns it, so filling n sites needs n distinct molecules."""
    B = ["%s_b%d" % (tag, k) for k in range(n + 1)]
    rx = []
    for k in range(n):
        ch_on = {B[k]: -1, B[k + 1]: +1}
        ch_off = {B[k + 1]: -1, B[k]: +1}
        if sequester:
            ch_on[reg] = -1; ch_off[reg] = +1
        rx.append(R((n - k) * kb / KS, 0, {B[k]: 1, reg: 1}, ch_on))
        rx.append(R((k + 1) * kb * omega ** (-k), 0, {B[k + 1]: 1}, ch_off))
    active = B[:n] if repress else [B[n]]
    for b in active:
        rx.append(R(alpha, 1, {b: 1}, {product: +1}))
    return B, rx, {b: 1 for b in active}


def site_activity(x, n, KS, omega, repress):
    r = np.asarray(x, float) / KS
    w = np.array([comb(n, k) * r ** k * omega ** (k * (k - 1) / 2) for k in range(n + 1)])
    p = w / w.sum(axis=0)
    return 1 - p[n] if repress else p[n]


def hill_activity(x, n, K, repress):
    f = (np.asarray(x, float) / K) ** n
    return 1 / (1 + f) if repress else f / (1 + f)


def lam2_hill(x, n, K, kb):
    return kb * (1 + (np.asarray(x, float) / K) ** n)


def lam2_site(x, n, KS, kb, omega):
    out = []
    for xx in np.atleast_1d(x):
        Q = np.zeros((n + 1, n + 1))
        for k in range(n):
            Q[k, k + 1] = (n - k) * kb * xx / KS
            Q[k + 1, k] = (k + 1) * kb * omega ** (-k)
        Q -= np.diag(Q.sum(axis=1))
        ev = np.sort(np.abs(np.real(np.linalg.eigvals(Q))))
        out.append(ev[1])
    return np.array(out)


def match_site(n, K, kb_hill, repress, xmax, omega=OMEGA_COOP):
    """K_S with the same half-maximum as the Hill promoter, and k_b with the
    same minimum relaxation rate over x in [0, xmax]."""
    lx = np.log(K)
    f = lambda KS: brentq(lambda z: site_activity(np.exp(z), n, KS, omega, repress) - 0.5,
                          lx - 15, lx + 15) - lx
    KS = np.exp(brentq(lambda lk: f(np.exp(lk)), lx - 8, lx + 8))
    xs = np.concatenate([[0.0], np.logspace(np.log10(K) - 4, np.log10(xmax), 200)])
    target = lam2_hill(xs, n, K, kb_hill).min()
    kb = target / lam2_site(xs, n, KS, 1.0, omega).min()
    return KS, kb


def exact_block(tag, reg, coef, active_w, lam, alpha, product):
    """Two-state control whose stationary activity equals that of a
    site-resolved promoter exactly. With stationary weights sum_k coef[k] x^k and
    readout weights active_w[k] in [0, 1], the off->on rate is
    lam * sum_k active_w[k] coef[k] x^k and the on->off rate
    lam * sum_k (1 - active_w[k]) coef[k] x^k, so pi_on is the site-resolved
    activity at every x and lambda_2 = lam * sum_k coef[k] x^k >= lam. Each
    monomial is one reaction; the SSA uses plain powers of X/Omega (as the
    catalytic binding of the site-resolved promoter implies), not falling
    factorials."""
    off, on = tag + "_off", tag + "_on"
    rx = []
    for k, c in enumerate(coef):
        for frac, src, dst in ((active_w[k], off, on), (1 - active_w[k], on, off)):
            if frac <= 0:
                continue
            reac = {src: 1}
            if k > 0:
                reac[reg] = k
            rx.append(dict(k=float(lam * frac * c), w=0, reac=reac, change={src: -1, dst: +1}, ff=False))
    rx.append(R(alpha, 1, {on: 1}, {product: +1}))
    return [off, on], rx, {on: 1}


def noise_strength(Q, a):
    """Zero-frequency spectral density 2 sum_ij a_i pi_i Z_ij a_j of the output a(S)
    of a Markov chain with generator Q, Z the fundamental matrix."""
    n = Q.shape[0]
    A = Q.T.copy(); A[-1, :] = 1; b = np.zeros(n); b[-1] = 1
    pi = np.linalg.solve(A, b)
    Pi = np.outer(np.ones(n), pi)
    Z = np.linalg.inv(Pi - Q) - Pi
    return 2 * np.sum(a[:, None] * pi[:, None] * Z * a[None, :]), pi


def site_Q(x, n, KS, kb, omega):
    Q = np.zeros((n + 1, n + 1))
    for k in range(n):
        Q[k, k + 1] = (n - k) * kb * x / KS
        Q[k + 1, k] = (k + 1) * kb * omega ** (-k)
    Q -= np.diag(Q.sum(axis=1))
    return Q


def matched_lambda(coef, active_w, S0_target, x):
    """lambda for the exact-response two-state control so that its zero-frequency
    noise strength at x equals S0_target (unit transcription rate)."""
    w = np.array([c * x ** k for k, c in enumerate(coef)])
    pon = np.dot(active_w, w) / w.sum()
    # two-state: S0 = 2 pon (1 - pon) / lambda2, lambda2 = lam * sum w
    return 2 * pon * (1 - pon) / (S0_target * w.sum())


X_OP = {}   # regulator concentration at the operating point, set per circuit


def promoter(arch, tag, reg, n, K, kb, repress, alpha, product, xmax):
    if arch == "H":
        return hill_block(tag, reg, n, K, kb, repress, alpha, product)
    if arch == "P":
        return hill_block(tag, reg, n, K, kb, repress, alpha, product, plain=True)
    if arch == "Q":
        KS, kbs = match_site(n, K, kb, repress, xmax)
        return site_block(tag, reg, n, KS, kbs, OMEGA_COOP, repress, alpha, product, sequester=True)
    KS, kbs = match_site(n, K, kb, repress, xmax)
    if arch in ("E", "M"):
        coef = [comb(n, k) * OMEGA_COOP ** (k * (k - 1) / 2) * KS ** (-k) for k in range(n + 1)]
        act = [1.0] * n + [0.0] if repress else [0.0] * n + [1.0]
        if arch == "E":
            xs = np.concatenate([[0.0], np.logspace(np.log10(K) - 4, np.log10(xmax), 200)])
            lam = lam2_site(xs, n, KS, kbs, OMEGA_COOP).min()
        else:
            x0 = X_OP[tag]
            S0, _ = noise_strength(site_Q(x0, n, KS, kbs, OMEGA_COOP), np.array(act, float))
            lam = matched_lambda(coef, np.array(act, float), S0, x0)
        return exact_block(tag, reg, coef, act, lam, alpha, product)
    return site_block(tag, reg, n, KS, kbs, OMEGA_COOP, repress, alpha, product)


def finish(M, h, tau_reg, unit, reg_species, promoters):
    M.h = h; M.tau_prot = tau_reg; M.time_unit = unit
    M.reg_species = reg_species        # regulator for each promoter block
    M.promoters = promoters            # (n, K or None) info for reporting
    return M


# ------------------------------------------------------------------ circuits

def repressilator(arch="H"):
    p = dict(k_trans=5.0, k_leak=5e-4, k_deg_m=np.log(2) / 2, k_transl=45.0,
             k_fold=1.0 / 60, k_deg_p=np.log(2) / 6.0, n=2, KM=100.0, k_burst=0.05)
    rep = {1: "p3", 2: "p1", 3: "p2"}
    sp, blocks, rx = [], [], []
    for i in (1, 2, 3):
        st, r, _ = promoter(arch, "S%d" % i, rep[i], p["n"], p["KM"], p["k_burst"], True,
                            p["k_trans"], "m%d" % i, xmax=3000.0)
        sp += st; blocks.append(st); rx += r
    sp += ["m1", "m2", "m3", "u1", "u2", "u3", "p1", "p2", "p3"]
    for i in (1, 2, 3):
        m, u, pp = "m%d" % i, "u%d" % i, "p%d" % i
        rx += [R(p["k_leak"], 1, {}, {m: +1}), R(p["k_deg_m"], 1, {m: 1}, {m: -1}),
               R(p["k_transl"], 1, {m: 1}, {u: +1}), R(p["k_fold"], 1, {u: 1}, {u: -1, pp: +1}),
               R(p["k_deg_p"], 1, {u: 1}, {u: -1}), R(p["k_deg_p"], 1, {pp: 1}, {pp: -1})]
    x0 = {b[0]: 1 for b in blocks}
    x0.update({"m1": 5.0, "u1": 48.0, "p1": 1000.0})
    M = Model("Repressilator-" + arch, sp, blocks, rx, x0, observe=["p1", "p2", "p3"],
              record_dt=1.0, t_burn=500.0, t_total=5500.0)
    return finish(M, 0.02, 1 / p["k_deg_p"], "min", ["p3", "p1", "p2"], [(2, 100.0)])


def goodwin(arch="H"):
    assert arch in ("H", "P")
    s = 0.06
    p = dict(k1=s, k2=s, k3=s, b=0.1 * s, n=10, KM=1.0, k_burst=2.0 * s)
    st, rx, _ = hill_block("S", "Z", p["n"], p["KM"], p["k_burst"], True, p["k1"], "X", plain=(arch == "P"))
    rx += [R(p["b"], 1, {"X": 1}, {"X": -1}), R(p["k2"], 1, {"X": 1}, {"Y": +1}),
           R(p["b"], 1, {"Y": 1}, {"Y": -1}), R(p["k3"], 1, {"Y": 1}, {"Z": +1}),
           R(p["b"], 1, {"Z": 1}, {"Z": -1})]
    M = Model("Goodwin-" + arch, st + ["X", "Y", "Z"], [st], rx, {"S_on": 1, "X": 0.1, "Y": 0.1, "Z": 0.1},
              observe=["X", "Y", "Z"], record_dt=5.0, t_burn=5000.0, t_total=65000.0)
    return finish(M, 0.01, 1 / p["b"], "min", ["Z"], [(10, 1.0)])


def toggle(arch="H"):
    p = dict(alpha1=300.0, alpha2=30.0, beta=3, gamma=1, k_burst=1.0)
    stU, rxU, _ = promoter(arch, "Su", "V", p["beta"], 1.0, p["k_burst"], True, p["alpha1"], "U", xmax=60.0)
    stV, rxV, _ = hill_block("Sv", "U", p["gamma"], 1.0, p["k_burst"], True, p["alpha2"], "V")
    rx = rxU + rxV + [R(1.0, 1, {"U": 1}, {"U": -1}), R(1.0, 1, {"V": 1}, {"V": -1})]
    x0 = {stU[0]: 1, "Sv_off": 1, "U": 150.0, "V": 0.0}
    M = Model("Toggle-" + arch, stU + stV + ["U", "V"], [stU, stV], rx, x0, observe=["U", "V"],
              record_dt=0.5, t_burn=50.0, t_total=2000.0)
    return finish(M, 0.001, 1.0, "dimensionless", ["V", "U"], [(3, 1.0), (1, 1.0)])


def nar(arch="H"):
    p = dict(alpha=100.0, gamma=1.0, K=30.0, n=2, k_burst=1.0)
    st, rx, _ = promoter(arch, "S", "P", p["n"], p["K"], p["k_burst"], True, p["alpha"], "P", xmax=150.0)
    rx += [R(p["gamma"], 1, {"P": 1}, {"P": -1})]
    M = Model("NAR-" + arch, st + ["P"], [st], rx, {st[0]: 1, "P": 30.0}, observe=["P"],
              record_dt=0.02, t_burn=20.0, t_total=420.0)
    return finish(M, 0.001, 1 / p["gamma"], "dimensionless", ["P"], [(2, 30.0)])


def cascade(arch="H"):
    p = dict(beta_A=40.0, gamma_A=1.0, alpha=100.0, gamma=1.0, K=40.0, n=2, k_burst=1.0)
    st, rx, _ = promoter(arch, "S", "A", p["n"], p["K"], p["k_burst"], False, p["alpha"], "P", xmax=120.0)
    rx += [R(p["beta_A"], 1, {}, {"A": +1}), R(p["gamma_A"], 1, {"A": 1}, {"A": -1}),
           R(p["gamma"], 1, {"P": 1}, {"P": -1})]
    M = Model("Cascade-" + arch, st + ["A", "P"], [st], rx, {st[0]: 1, "A": 40.0, "P": 50.0},
              observe=["P", "A"], record_dt=0.02, t_burn=20.0, t_total=420.0)
    return finish(M, 0.001, 1 / p["gamma_A"], "dimensionless", ["A"], [(2, 40.0)])


def pfb(arch="H"):
    """Positive autoregulation: P activates its own promoter (order 2). With these
    values the deterministic system has a single stable fixed point (P* ~ 72)
    and a near-tangency of the rate balance near P ~ 6 (a saddle-node ghost),
    so the stationary distribution is strongly non-Gaussian at small Omega."""
    p = dict(basal=4.0, alpha=80.0, gamma=1.0, K=30.0, n=2, k_burst=1.0)
    st, rx, _ = promoter(arch, "S", "P", p["n"], p["K"], p["k_burst"], False, p["alpha"], "P", xmax=150.0)
    rx += [R(p["basal"], 1, {}, {"P": +1}), R(p["gamma"], 1, {"P": 1}, {"P": -1})]
    # trajectories start from the exact stationary distribution (run_grid2); the
    # longer burn-in lets the reductions relax from it
    M = Model("PFB-" + arch, st + ["P"], [st], rx, {st[0]: 1, "P": 70.0}, observe=["P"],
              record_dt=0.02, t_burn=200.0, t_total=600.0)
    return finish(M, 0.001, 1 / p["gamma"], "dimensionless", ["P"], [(2, 30.0)])


def _p53_site_activity(x, kon, koff, coop, n=3):
    """Stationary mean occupancy fraction of the original three-site promoter."""
    x = np.atleast_1d(np.asarray(x, float)); out = []
    for xx in x:
        Q = np.zeros((n + 1, n + 1))
        for k in range(n):
            Q[k, k + 1] = kon * (n - k) * (coop if k > 0 else 1.0) * xx
            Q[k + 1, k] = koff * (k + 1)
        Q -= np.diag(Q.sum(axis=1))
        A = Q.T.copy(); A[-1, :] = 1; b = np.zeros(n + 1); b[-1] = 1
        pi = np.linalg.solve(A, b)
        out.append(np.dot(pi, np.arange(n + 1)) / n)
    return np.array(out)


def p53(arch="S", signal=10.0):
    c = 1.8
    p = dict(k_syn_p53=0.1 * c, k_act_ATM=5.0 * c, k_deg_p53=0.1 * c, k_ub=5.0 * c,
             k_trans_Mdm2=5.0 * c, k_deg_Mdm2=1.0 * c, k_deact_ATM=0.5 * c,
             n=3, k_on=10.0 * c, k_off=10.0 * c, coop=2.0)
    n = p["n"]
    rx = [R(signal, 1, {}, {"ATM": +1}), R(p["k_deact_ATM"], 1, {"ATM": 1}, {"ATM": -1}),
          R(p["k_syn_p53"], 1, {}, {"p53": +1}), R(p["k_act_ATM"], 1, {"ATM": 1}, {"p53": +1}),
          R(p["k_deg_p53"], 1, {"p53": 1}, {"p53": -1}),
          R(p["k_ub"], 1, {"Mdm2": 1, "p53": 1}, {"p53": -1}),
          R(p["k_deg_Mdm2"], 1, {"Mdm2": 1}, {"Mdm2": -1})]
    if arch == "S":
        P = ["P%d" % i for i in range(n + 1)]
        for kk in range(n):
            cf = p["coop"] if kk > 0 else 1.0
            rx.append(R(p["k_on"] * (n - kk) * cf, 0, {P[kk]: 1, "p53": 1}, {P[kk]: -1, P[kk + 1]: +1}))
            rx.append(R(p["k_off"] * (kk + 1), 0, {P[kk + 1]: 1}, {P[kk + 1]: -1, P[kk]: +1}))
        for kk in range(1, n + 1):
            rx.append(R(p["k_trans_Mdm2"] * kk / n, 1, {P[kk]: 1}, {"Mdm2": +1}))
        st = P
    elif arch in ("E", "M"):
        coef = [1.0]
        for kk in range(n):
            cf = p["coop"] if kk > 0 else 1.0
            coef.append(coef[-1] * p["k_on"] * (n - kk) * cf / (p["k_off"] * (kk + 1)))
        if arch == "E":
            lam_s = min(_lam_p53(xx, p) for xx in np.concatenate([[0.0], np.logspace(-3, 1, 100)]))
        else:
            x0 = X_OP["p53"]
            S0, _ = noise_strength(_Q_p53(x0, p), np.arange(n + 1) / n)
            lam_s = matched_lambda(coef, np.arange(n + 1) / n, S0, x0)
        st, r2, _ = exact_block("P", "p53", coef, [kk / n for kk in range(n + 1)], lam_s,
                                p["k_trans_Mdm2"], "Mdm2")
        rx += r2
    else:
        xs = np.logspace(-3, 2, 300)
        y = _p53_site_activity(xs, p["k_on"], p["k_off"], p["coop"])
        (h, K), _ = curve_fit(lambda x, h, K: x ** h / (K ** h + x ** h), xs, y, p0=[1.5, 0.5])
        # two-state promoter with activity x^h/(K^h+x^h) and the same slowest
        # relaxation as the site-resolved promoter over the p53 range
        lam_s = min(_lam_p53(xx, p) for xx in np.concatenate([[0.0], np.logspace(-3, 1, 100)]))
        kb = lam_s
        st, r2, _ = hill_block("P", "p53", h, K, kb, False, p["k_trans_Mdm2"], "Mdm2")
        rx += r2
        p["h_fit"], p["K_fit"] = h, K
    M = Model("p53-" + arch, st + ["ATM", "p53", "Mdm2"], [st], rx, {st[0]: 1}, observe=["p53", "Mdm2"],
              record_dt=0.02, t_burn=20.0, t_total=420.0)
    # regulator (p53) effective lifetime at the deterministic fixed point:
    # 1/(k_deg_p53 + k_ub Mdm2*)
    atm = signal / p["k_deact_ATM"]
    prod = p["k_syn_p53"] + p["k_act_ATM"] * atm
    if arch in ("S", "E", "M"):
        act = lambda x: _p53_site_activity(x, p["k_on"], p["k_off"], p["coop"])[0]
    else:
        act = lambda x: x ** p["h_fit"] / (p["K_fit"] ** p["h_fit"] + x ** p["h_fit"])
    g = lambda x: prod - (p["k_deg_p53"] + p["k_ub"] * p["k_trans_Mdm2"] * act(x) / p["k_deg_Mdm2"]) * x
    xstar = brentq(g, 1e-9, 1e4)
    mdm2 = p["k_trans_Mdm2"] * act(xstar) / p["k_deg_Mdm2"]
    tau = 1.0 / (p["k_deg_p53"] + p["k_ub"] * mdm2)
    M = finish(M, 0.0005, tau, "model", ["p53"], [(n, None)])
    M.params = p; M.fixed_point = dict(p53=xstar, Mdm2=mdm2, ATM=atm)
    return M


def _Q_p53(x, p, n=3):
    Q = np.zeros((n + 1, n + 1))
    for k in range(n):
        Q[k, k + 1] = p["k_on"] * (n - k) * (p["coop"] if k > 0 else 1.0) * x
        Q[k + 1, k] = p["k_off"] * (k + 1)
    return Q - np.diag(Q.sum(axis=1))


def _lam_p53(x, p, n=3):
    Q = np.zeros((n + 1, n + 1))
    for k in range(n):
        Q[k, k + 1] = p["k_on"] * (n - k) * (p["coop"] if k > 0 else 1.0) * x
        Q[k + 1, k] = p["k_off"] * (k + 1)
    Q -= np.diag(Q.sum(axis=1))
    return np.sort(np.abs(np.real(np.linalg.eigvals(Q))))[1]


def _operating_points():
    """Deterministic regulator levels used to match the M promoters: fixed points
    of the site-resolved circuits."""
    from scipy.optimize import brentq as _b
    KS, _ = match_site(2, 30.0, 1.0, True, 150.0)
    X_OP["S"] = None
    nar_x = _b(lambda P: 100.0 * site_activity(P, 2, KS, OMEGA_COOP, True) - P, 1e-6, 200)
    KSc, _ = match_site(2, 40.0, 1.0, False, 120.0)
    KSp, _ = match_site(2, 30.0, 1.0, False, 150.0)
    pfb_x = _b(lambda P: 4.0 + 80.0 * site_activity(P, 2, KSp, OMEGA_COOP, False) - P, 40, 200)
    return dict(NAR=nar_x, Cascade=40.0, PFB=pfb_x)


def _with_xop(fn, key, tag="S"):
    def build():
        X_OP[tag] = _operating_points()[key]
        return fn()
    return build


def _toggle_M():
    KSu, _ = match_site(3, 1.0, 1.0, True, 60.0)
    U = 300.0
    for _ in range(200):                      # U-high fixed point of the S toggle
        V = 30.0 / (1 + U)
        U = 300.0 * site_activity(V, 3, KSu, OMEGA_COOP, True)
    X_OP["Su"] = V
    return toggle("M")


def _p53_M():
    X_OP["p53"] = p53("S").fixed_point["p53"]
    return p53("M")


MODELS = {
    "Repressilator-H": lambda: repressilator("H"), "Repressilator-S": lambda: repressilator("S"),
    "Goodwin-H": lambda: goodwin("H"),
    "Toggle-H": lambda: toggle("H"), "Toggle-S": lambda: toggle("S"),
    "p53-S": lambda: p53("S"), "p53-H": lambda: p53("H"),
    "NAR-H": lambda: nar("H"), "NAR-S": lambda: nar("S"),
    "Cascade-H": lambda: cascade("H"), "Cascade-S": lambda: cascade("S"),
    "Toggle-E": lambda: toggle("E"), "p53-E": lambda: p53("E"),
    "NAR-E": lambda: nar("E"), "Cascade-E": lambda: cascade("E"),
    "Toggle-P": lambda: toggle("P"), "Toggle-Q": lambda: toggle("Q"), "Goodwin-P": lambda: goodwin("P"), "Toggle-M": _toggle_M, "p53-M": _p53_M,
    "NAR-M": _with_xop(lambda: nar("M"), "NAR"), "Cascade-M": _with_xop(lambda: cascade("M"), "Cascade"),
    "PFB-H": lambda: pfb("H"), "PFB-S": lambda: pfb("S"), "PFB-E": lambda: pfb("E"),
    "PFB-M": _with_xop(lambda: pfb("M"), "PFB"),
}
BASE_OMEGA = {"Repressilator": 1.0, "Goodwin": 100.0, "Toggle": 1.0, "p53": 50.0, "NAR": 1.0, "Cascade": 1.0,
              "PFB": 1.0}
STATIONARY = {"p53", "NAR", "Cascade", "PFB"}
OSCILLATOR = {"Repressilator", "Goodwin"}
