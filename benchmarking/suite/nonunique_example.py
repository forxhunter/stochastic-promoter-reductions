#!/usr/bin/env python3
"""Two promoters with the same mean dose response and different noise.

Two-state promoter, q01 = lam * r^2/(1 + r^2), q10 = lam /(1 + r^2) with
r = x/K, so pi_on = r^2/(1 + r^2) for every lam, and the mean response
alpha * pi_on is the same Hill function for every switching rate lam.
Protein: produced at alpha while on, degraded at gamma (discrete copies).
Exact stationary moments (Peccoud & Ycart 1995):
    mean = alpha pi_on / gamma
    Fano = 1 + alpha pi_off / (lam + gamma)
The formula is checked here against SSA.
"""
import numpy as np
from engine import Model, simulate

ALPHA, GAMMA = 20.0, 1.0


def model(lam, r):
    kon = lam * r ** 2 / (1 + r ** 2); koff = lam / (1 + r ** 2)
    rx = [dict(k=kon, w=0, reac={"S_off": 1}, change={"S_off": -1, "S_on": 1}),
          dict(k=koff, w=0, reac={"S_on": 1}, change={"S_on": -1, "S_off": 1}),
          dict(k=ALPHA, w=1, reac={"S_on": 1}, change={"P": 1}),
          dict(k=GAMMA, w=1, reac={"P": 1}, change={"P": -1})]
    M = Model("telegraph", ["S_off", "S_on", "P"], [["S_off", "S_on"]], rx, {"S_on": 1},
              observe=["P"], record_dt=0.5, t_burn=20.0, t_total=4020.0)
    M.h = 0.001
    return M


def exact(lam, r):
    pon = r ** 2 / (1 + r ** 2)
    mean = ALPHA * pon / GAMMA
    fano = 1 + ALPHA * (1 - pon) / (lam + GAMMA)
    return mean, fano, np.sqrt(fano / mean)


if __name__ == "__main__":
    print(" lam   eta   r |  mean(ex)  Fano(ex)  CV(ex) |  mean(SSA) Fano(SSA) CV(SSA)")
    for lam in (100.0, 1.0):
        for r in (0.5, 1.0, 2.0):
            m, f, cv = exact(lam, r)
            M = model(lam, r); b = int(M.t_burn / M.record_dt)
            y = np.concatenate([simulate(M, "ssa", 1.0, 1.0, seed=s)[0][b:, 0] for s in range(8)])
            print("%5g %5.2f %4g | %8.2f %8.3f %7.3f | %8.2f %8.3f %7.3f"
                  % (lam, GAMMA / (lam), r, m, f, cv, y.mean(), y.var() / y.mean(), y.std() / y.mean()))
