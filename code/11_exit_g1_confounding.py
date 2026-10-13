"""Is the arrest-exit rate identifiable separately from the G1 duration?

After reoxygenation, FUCCI-green reappears in the core only once a cell has
left the arrested state AND crossed G1 (exit time + G1 transit). The radial
fits put k1max on its bound (G1 = 30 h) and give a slow exit. Here the G1
bound is widened to 120 h and the exit rate is fixed on a grid; every other
parameter is re-optimised (fine integrator, three starts with different G1).
If fast exit + slow G1 fits as well as slow exit + G1 = 30 h, only their
combination is identifiable from radius data.

Usage: python 11_exit_g1_confounding.py LINE [TAG exit_h,exit_h,... g1_h,g1_h,...]
Output: results/exit_g1_confounding_<LINE>[_TAG].csv
"""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
EXIT_ARG = sys.argv[3] if len(sys.argv) > 3 else None
G1_ARG = sys.argv[4] if len(sys.argv) > 4 else None
VARIANT = "finQ"
HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [sys.argv[0], LINE, "1", VARIANT]
spec = importlib.util.spec_from_file_location("f7", HERE / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f7)
sys.argv = _argv
f7.LOG_BOUNDS["k1max"] = (np.log10(1 / 120), np.log10(1 / 5))  # G1 5-120 h
f7.LOG_BOUNDS["bmax"] = (-4.0, 0.0)  # exit times up to 1e4 h

import os
_src = f"radial_fit_{LINE}_alldata.json" if os.environ.get("FIT_ALL") else f"radial_fit_{LINE}.json"
fit = json.loads((f7.ROOT / "results" / _src).read_text())[VARIANT]
names, lo, hi = f7.bounds(VARIANT)
x_best = np.clip(np.log10([fit["params"][n] for n in names]), lo, hi)
jb, jk = names.index("bmax"), names.index("k1max")
others = [i for i in range(len(names)) if i != jb]

EXIT_H = [float(v) for v in EXIT_ARG.split(",")] if EXIT_ARG else [24.0, 48.0, 96.0, 144.0, 500.0]
G1_START_H = [float(v) for v in G1_ARG.split(",")] if G1_ARG else [30.0, 60.0, 100.0]
SUFFIX = f"_{TAG}" if TAG else ""


def job(args):
    exit_h, g1_h = args
    x0 = x_best.copy()
    x0[jb] = np.log10(1 / exit_h)
    x0[jk] = np.log10(1 / g1_h)

    def obj(z):
        x = x0.copy()
        x[others] = z
        return f7.objective(x, VARIANT, True)
    r1 = minimize(obj, x0[others], method="Powell", options=dict(maxfev=1400, xtol=1e-3, ftol=1e-6))
    r2 = minimize(obj, r1.x, method="Nelder-Mead", options=dict(maxfev=500, xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    x = x0.copy()
    x[others] = r.x
    return dict(exit_time_h=exit_h, g1_start_h=g1_h, nll=float(r.fun),
                g1_fit_h=float(1 / 10 ** x[jk]), Kq=float(10 ** x[names.index("Kq")]),
                mu0=float(10 ** x[names.index("mu0")]))


if __name__ == "__main__":
    jobs = [(e, g) for e in EXIT_H for g in G1_START_H]
    with Pool() as pool:
        out = pool.map(job, jobs)
    df = pd.DataFrame(out)
    best = df.loc[df.groupby("exit_time_h").nll.idxmin()].sort_values("exit_time_h")
    ref = fit["nll_train"]
    best["delta_vs_best_fit"] = best.nll - min(ref, df.nll.min())
    df.to_csv(f7.ROOT / "results" / f"exit_g1_confounding_{LINE}{SUFFIX}_all.csv", index=False,
              float_format="%.5g")
    best.to_csv(f7.ROOT / "results" / f"exit_g1_confounding_{LINE}{SUFFIX}.csv", index=False,
                float_format="%.5g")
    print("reference best-fit NLL (G1 <= 30 h):", round(ref, 2))
    print(best.round(3).to_string(index=False))
