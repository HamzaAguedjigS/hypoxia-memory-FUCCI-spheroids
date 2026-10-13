"""Polish the all-data finQ fit from the best profile point (exit ~125 h).

The refined all-data exit profile (11_exit_g1_confounding.py, grid 110-260 h)
found a lower NLL at exit = 125 h (G1 ~36 h) than the stored all-data fit.
This script re-optimises every finQ parameter from that region (fine
integrator) and, if better, replaces the finQ entry of
results/radial_fit_<LINE>_alldata.json and its predictions.

Usage: FIT_ALL=1 python 15_polish_finQ_alldata.py LINE
"""
import importlib.util
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

assert os.environ.get("FIT_ALL"), "run with FIT_ALL=1 (all protocols in the likelihood)"
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [sys.argv[0], LINE, "1", "finQ"]
spec = importlib.util.spec_from_file_location("f7", HERE / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f7)
sys.argv = _argv
rm = f7.rm
FIT_FILE = f7.ROOT / "results" / f"radial_fit_{LINE}_alldata.json"
fits = json.loads(FIT_FILE.read_text())
names, lo, hi = f7.bounds("finQ")
x0 = np.clip(np.log10([fits["finQ"]["params"][n] for n in names]), lo, hi)


def start(exit_h, g1_h):
    x = x0.copy()
    x[names.index("bmax")] = np.log10(1 / exit_h)
    x[names.index("k1max")] = np.log10(1 / g1_h)
    return x


def polish(x):
    r1 = minimize(f7.objective, x, args=("finQ", True), method="Powell",
                  options=dict(maxfev=1800, xtol=1e-3, ftol=1e-6))
    r2 = minimize(f7.objective, r1.x, args=("finQ", True), method="Nelder-Mead",
                  options=dict(maxfev=800, xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    return r.x.tolist(), float(r.fun)


if __name__ == "__main__":
    xs = [start(125, 35.6), start(125, 31.0), start(135, 35.6), start(115, 35.6), x0]
    with Pool() as pool:
        out = pool.map(polish, xs)
    out.sort(key=lambda o: o[1])
    print("polished NLLs:", [round(o[1], 2) for o in out], " stored:", round(fits["finQ"]["nll_train"], 2))
    if out[0][1] < fits["finQ"]["nll_train"]:
        x = np.clip(np.array(out[0][0]), lo, hi)
        pred = f7.predict(x, "finQ", fine=True)
        tr, sds = f7.nll(pred, f7.IDX_TRAIN)
        k = len(names) + 4
        n_obs = int(np.isfinite(f7.IDX_TRAIN[2]).sum())
        th = f7.to_theta(x, "finQ")
        at_bound = [n for n, xi, l, h in zip(names, x, lo, hi) if xi <= l + 0.02 or xi >= h - 0.02]
        fits["finQ"].update(params=th, at_bound=at_bound, nll_train=tr,
                            sd=dict(zip(f7.OBS, sds.tolist())), AIC=2 * tr + 2 * k,
                            BIC=2 * tr + k * np.log(n_obs), polish_nlls=[round(o[1], 2) for o in out])
        FIT_FILE.write_text(json.dumps(fits, indent=2))
        print(f"updated finQ: NLL={tr:.2f} AIC={2 * tr + 2 * k:.1f} exit={1 / th['bmax']:.0f} h "
              f"G1={1 / th['k1max']:.1f} h bounds={at_bound}")
        pred_file = f7.ROOT / "results" / f"radial_pred_{LINE}_alldata.csv"
        allp = pd.read_csv(pred_file)
        allp = allp[allp.variant != "finQ"]
        rows = []
        t_dense = np.arange(0, 7 * 24 + 1, 6.0)
        rm.DT = 0.5
        mdl = rm.Model("finQ", M=24, protocols=list(rm.PROTOCOLS))
        o = mdl.run(th, t_dense)
        for i, p in enumerate(mdl.protocols):
            for j, t in enumerate(t_dense):
                rows.append(dict(variant="finQ", protocol=p, day=rm.T0_DAY + t / 24, R_outer=o[i, j, 0],
                                 R_inhib=o[i, j, 1], R_nec=o[i, j, 2], R_pim=o[i, j, 3], f_Q=o[i, j, 4]))
        pd.concat([allp, pd.DataFrame(rows)]).to_csv(pred_file, index=False, float_format="%.5g")
    for v, x in fits.items():
        print(f"  {v:7s} AIC={x['AIC']:.1f}")
