"""Observables computed identically for every method, and distances between
two ensembles."""

import numpy as np
from scipy import stats


def acf_period(y, dt):
    """Period from the first local maximum of the autocorrelation function
    after its first zero crossing. NaN if the ACF has no such maximum."""
    y = np.asarray(y, float) - np.mean(y)
    if np.allclose(y, 0):
        return np.nan
    n = len(y)
    f = np.fft.rfft(y, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    neg = np.where(ac < 0)[0]
    if len(neg) == 0:
        return np.nan
    i0 = neg[0]
    j = None
    for i in range(i0 + 1, n // 2 - 1):
        if ac[i] > 0 and ac[i] >= ac[i - 1] and ac[i] > ac[i + 1]:
            j = i
            break
    if j is None:
        return np.nan
    a, b, c = ac[j - 1], ac[j], ac[j + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0.0
    i0, j = 0, j
    return (i0 + j + d) * dt


def ensemble_stats(traj, dt, burn, oscillator):
    """traj: array (n_traj, n_time) of one observed species after recording."""
    Y = traj[:, burn:]
    out = {}
    out["mean"] = float(Y.mean())
    out["cv"] = float(Y.std() / Y.mean()) if Y.mean() > 0 else np.nan
    out["amp"] = Y.std(axis=1)                       # per-trajectory amplitude (s.d. over time)
    if oscillator:
        out["period"] = np.array([acf_period(y, dt) for y in Y])
    return out


def ks(a, b, thin):
    """KS distance between pooled stationary samples, thinned in time."""
    return float(stats.ks_2samp(a[:, ::thin].ravel(), b[:, ::thin].ravel()).statistic)


def switching(u, v, dt, burn):
    """Toggle switch: fraction of time in the U-high basin and number of basin
    changes per unit time, with hysteresis (U > 2V enters high, U < V/2 enters low)."""
    rates, frac = [], []
    for uu, vv in zip(u[:, burn:], v[:, burn:]):
        state = 1 if uu[0] > vv[0] else 0
        n = 0; hi = 0
        for a, b in zip(uu, vv):
            if state == 1 and a < 0.5 * b:
                state = 0; n += 1
            elif state == 0 and a > 2 * b:
                state = 1; n += 1
            hi += state
        rates.append(n / (len(uu) * dt)); frac.append(hi / len(uu))
    return np.array(rates), np.array(frac)
