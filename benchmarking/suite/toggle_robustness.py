#!/usr/bin/env python3
"""Robustness of the exact toggle escape rates to the basin thresholds: the default
pair (enter U > 2V, leave U < V/2) against a second pair (enter U > 3V, leave
U < V/3). Writes
benchmarking_results/toggle_robustness.csv."""
import os
import pandas as pd
from systems2 import MODELS
from toggle_exact import escape_rates

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "benchmarking_results")

if __name__ == "__main__":
    rows = []
    for arch in ("H", "P", "E", "M", "S"):
        M = MODELS["Toggle-" + arch]()
        for om in (0.3, 1.0):
            for s in (1.0, 3.0):
                r = dict(arch=arch, omega=om, s=s)
                a = escape_rates(M, om, s, False)
                r.update(k_U=a["k_U"], k_V=a["k_V"])
                b = escape_rates(M, om, s, False, enter=3.0, leave=1 / 3)
                r.update(k_U_3=b["k_U"], k_V_3=b["k_V"])
                rows.append(r); print(r, flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(RES, "toggle_robustness.csv"), index=False)
