"""The four benchmark circuits, each defined once as a microscopic network with
explicit promoter states. Parameter values are those of the original benchmark
drivers (benchmarking/benchmark_parallel.py, benchmarking/run_toggle_final.py).
"""

import numpy as np
from engine import Model


def R(k, w, reac, change):
    return dict(k=float(k), w=w, reac=reac, change=change)


def repressilator():
    p = dict(k_trans=5.0, k_leak=5e-4, k_deg_m=np.log(2) / 2, k_transl=45.0,
             k_fold=1.0 / 60, k_deg_p=np.log(2) / 6.0, n=2.0, KM=100.0, k_burst=0.05)
    rep = {1: "p3", 2: "p1", 3: "p2"}
    sp, blocks, rx = [], [], []
    for i in (1, 2, 3):
        sp += ["S%d_off" % i, "S%d_on" % i]
        blocks.append(["S%d_off" % i, "S%d_on" % i])
    sp += ["m1", "m2", "m3", "u1", "u2", "u3", "p1", "p2", "p3"]
    for i in (1, 2, 3):
        off, on, m, u, pp = "S%d_off" % i, "S%d_on" % i, "m%d" % i, "u%d" % i, "p%d" % i
        rx.append(R(p["k_burst"], 0, {off: 1}, {off: -1, on: +1}))
        rx.append(R(p["k_burst"] * p["KM"] ** -p["n"], 0, {on: 1, rep[i]: p["n"]}, {on: -1, off: +1}))
        rx.append(R(p["k_trans"], 1, {on: 1}, {m: +1}))
        rx.append(R(p["k_leak"], 1, {}, {m: +1}))
        rx.append(R(p["k_deg_m"], 1, {m: 1}, {m: -1}))
        rx.append(R(p["k_transl"], 1, {m: 1}, {u: +1}))
        rx.append(R(p["k_fold"], 1, {u: 1}, {u: -1, pp: +1}))
        rx.append(R(p["k_deg_p"], 1, {u: 1}, {u: -1}))
        rx.append(R(p["k_deg_p"], 1, {pp: 1}, {pp: -1}))
    x0 = {"S1_on": 1, "S2_on": 1, "S3_on": 1, "m1": 5.0, "u1": 48.0, "p1": 1000.0}
    M = Model("Repressilator", sp, blocks, rx, x0, observe=["p1", "p2", "p3"],
              record_dt=1.0, t_burn=500.0, t_total=5500.0)
    M.h = 0.02
    M.tau_prot = 1.0 / p["k_deg_p"]
    M.time_unit = "min"
    return M


def goodwin():
    s = 0.06
    p = dict(k1=s, k2=s, k3=s, b1=0.1 * s, b2=0.1 * s, b3=0.1 * s, n=10.0, KM=1.0, k_burst=2.0 * s)
    sp = ["S_off", "S_on", "X", "Y", "Z"]
    rx = [R(p["k_burst"], 0, {"S_off": 1}, {"S_off": -1, "S_on": +1}),
          R(p["k_burst"] * p["KM"] ** -p["n"], 0, {"S_on": 1, "Z": p["n"]}, {"S_on": -1, "S_off": +1}),
          R(p["k1"], 1, {"S_on": 1}, {"X": +1}),
          R(p["b1"], 1, {"X": 1}, {"X": -1}),
          R(p["k2"], 1, {"X": 1}, {"Y": +1}),
          R(p["b2"], 1, {"Y": 1}, {"Y": -1}),
          R(p["k3"], 1, {"Y": 1}, {"Z": +1}),
          R(p["b3"], 1, {"Z": 1}, {"Z": -1})]
    x0 = {"S_on": 1, "X": 0.1, "Y": 0.1, "Z": 0.1}
    M = Model("Goodwin", sp, [["S_off", "S_on"]], rx, x0, observe=["X", "Y", "Z"],
              record_dt=5.0, t_burn=5000.0, t_total=65000.0)
    M.h = 0.01
    M.tau_prot = 1.0 / p["b1"]
    M.time_unit = "min"
    return M


def toggle():
    p = dict(alpha1=300.0, alpha2=30.0, beta=2.5, gamma=1.0, k_burst=1.0)
    sp = ["Su_off", "Su_on", "Sv_off", "Sv_on", "U", "V"]
    rx = [R(p["k_burst"], 0, {"Su_off": 1}, {"Su_off": -1, "Su_on": +1}),
          R(p["k_burst"], 0, {"Su_on": 1, "V": p["beta"]}, {"Su_on": -1, "Su_off": +1}),
          R(p["k_burst"], 0, {"Sv_off": 1}, {"Sv_off": -1, "Sv_on": +1}),
          R(p["k_burst"], 0, {"Sv_on": 1, "U": p["gamma"]}, {"Sv_on": -1, "Sv_off": +1}),
          R(p["alpha1"], 1, {"Su_on": 1}, {"U": +1}),
          R(p["alpha2"], 1, {"Sv_on": 1}, {"V": +1}),
          R(1.0, 1, {"U": 1}, {"U": -1}),
          R(1.0, 1, {"V": 1}, {"V": -1})]
    x0 = {"Su_on": 1, "Sv_off": 1, "U": 150.0, "V": 0.0}
    M = Model("Toggle", sp, [["Su_off", "Su_on"], ["Sv_off", "Sv_on"]], rx, x0, observe=["U", "V"],
              record_dt=0.5, t_burn=50.0, t_total=2000.0)
    M.h = 0.001
    M.tau_prot = 1.0
    M.time_unit = "dimensionless"
    return M


def p53(signal=10.0):
    c = 1.8
    p = dict(k_syn_p53=0.1 * c, k_act_ATM=5.0 * c, k_deg_p53=0.1 * c, k_ub=5.0 * c,
             k_trans_Mdm2=5.0 * c, k_deg_Mdm2=1.0 * c, k_deact_ATM=0.5 * c,
             n=3, k_on=10.0 * c, k_off=10.0 * c, coop=2.0)
    n = p["n"]
    P = ["P%d" % i for i in range(n + 1)]
    sp = P + ["ATM", "p53", "Mdm2"]
    rx = [R(signal, 1, {}, {"ATM": +1}),
          R(p["k_deact_ATM"], 1, {"ATM": 1}, {"ATM": -1}),
          R(p["k_syn_p53"], 1, {}, {"p53": +1}),
          R(p["k_act_ATM"], 1, {"ATM": 1}, {"p53": +1}),
          R(p["k_deg_p53"], 1, {"p53": 1}, {"p53": -1}),
          R(p["k_ub"], 1, {"Mdm2": 1, "p53": 1}, {"p53": -1}),
          R(p["k_deg_Mdm2"], 1, {"Mdm2": 1}, {"Mdm2": -1})]
    for kk in range(n):
        cf = p["coop"] if kk > 0 else 1.0
        rx.append(R(p["k_on"] * (n - kk) * cf, 0, {P[kk]: 1, "p53": 1}, {P[kk]: -1, P[kk + 1]: +1}))
        rx.append(R(p["k_off"] * (kk + 1), 0, {P[kk + 1]: 1}, {P[kk + 1]: -1, P[kk]: +1}))
    for kk in range(1, n + 1):
        rx.append(R(p["k_trans_Mdm2"] * kk / n, 1, {P[kk]: 1}, {"Mdm2": +1}))
    x0 = {"P0": 1}
    M = Model("p53", sp, [P], rx, x0, observe=["p53", "Mdm2"],
              record_dt=0.1, t_burn=50.0, t_total=1050.0)
    M.h = 0.0005
    M.tau_prot = 1.0 / p["k_deg_Mdm2"]
    M.time_unit = "model"
    return M


SYSTEMS = {"Repressilator": repressilator, "Goodwin": goodwin, "Toggle": toggle, "p53": p53}
BASE_OMEGA = {"Repressilator": 1.0, "Goodwin": 1.0, "Toggle": 1.0, "p53": 50.0}
