"""Final refinement of the radial fits for one line (fine integrator, v5 bounds).

For every variant, start from the current fit (radial_fit_<LINE>.json) and,
for the finite-kinetics variants, also from the same point with slow exit
times (the exit-time profile, 11_exit_g1_confounding.py, found the optimum
near 500 h). Each start is polished with the fine integrator (DT = 0.5 h,
24 shells) and the best result per variant replaces the fit file; dense
predictions are regenerated.

Usage: python 14_refine_final.py LINE
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
HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [sys.argv[0], LINE, "1"]
spec = importlib.util.spec_from_file_location("f7", HERE / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f7)
sys.argv = _argv
rm = f7.rm
import os
SRC_FILE = f7.ROOT / "results" / f"radial_fit_{LINE}.json"
current = json.loads(SRC_FILE.read_text())
SUFFIX = "_alldata" if os.environ.get("FIT_ALL") else ""
FIT_FILE = f7.ROOT / "results" / f"radial_fit_{LINE}{SUFFIX}.json"


def starts(variant):
    names, lo, hi = f7.bounds(variant)
    th = dict(f7.GUESS)
    th.update(current[variant]["params"])
    x = np.clip(np.log10([th[n] for n in names]), lo, hi)
    out = [x]
    v4 = f7.ROOT / "results" / f"radial_fit_{LINE}_v4.json"
    if v4.exists():  # the v4 optimum predicted the held-out protocol well
        th4 = dict(f7.GUESS)
        th4.update(json.loads(v4.read_text())[variant]["params"])
        out.append(np.clip(np.log10([th4[n] for n in names]), lo, hi))
    if variant in ("finQ", "delayQ"):
        for exit_h in (144.0, 200.0, 300.0, 500.0):
            y = x.copy()
            y[names.index("bmax")] = np.log10(1 / exit_h)
            out.append(y)
    return [(v, variant) for v in out]


def polish(args):
    x0, variant = args
    r1 = minimize(f7.objective, x0, args=(variant, True), method="Powell",
                  options=dict(maxfev=1500, xtol=1e-3, ftol=1e-6))
    r2 = minimize(f7.objective, r1.x, args=(variant, True), method="Nelder-Mead",
                  options=dict(maxfev=700, xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    return variant, r.x.tolist(), float(r.fun)


if __name__ == "__main__":
    variants = list(current)
    jobs = [j for v in variants for j in starts(v)]
    with Pool() as pool:
        out = pool.map(polish, jobs)
    results = {}
    for v in variants:
        runs = sorted([o for o in out if o[0] == v], key=lambda o: o[2])
        names, lo, hi = f7.bounds(v)
        x = np.clip(np.array(runs[0][1]), lo, hi)
        pred = f7.predict(x, v, fine=True)
        tr, sds = f7.nll(pred, f7.IDX_TRAIN)
        te, _ = f7.nll(pred, f7.IDX_TEST, sds) if len(f7.IDX_TEST[2]) else (np.nan, None)
        k = len(names) + 4
        n_obs = int(np.isfinite(f7.IDX_TRAIN[2]).sum())
        th = f7.to_theta(x, v)
        at_bound = [n for n, xi, l, h in zip(names, x, lo, hi) if xi <= l + 0.02 or xi >= h - 0.02]
        results[v] = dict(params=th, free=names, at_bound=at_bound, nll_train=tr,
                          sd=dict(zip(f7.OBS, sds.tolist())), n_params=k, n_obs=n_obs,
                          AIC=2 * tr + 2 * k, BIC=2 * tr + k * np.log(n_obs), nll_test=te,
                          refine_nlls=[round(o[2], 2) for o in runs],
                          previous_nll=current[v]["nll_train"])
        print(f"{LINE} {v:7s} NLL {current[v]['nll_train']:.2f} -> {tr:.2f}  AIC={2 * tr + 2 * k:.1f} "
              f"test={te:.2f} exit_h={1 / th['bmax'] if v in ('finQ', 'delayQ') else float('nan'):.0f} "
              f"G1_h={1 / th['k1max']:.1f} bounds={at_bound}")
    FIT_FILE.write_text(json.dumps(results, indent=2))

    rows = []
    t_dense = np.arange(0, 7 * 24 + 1, 6.0)
    for v in variants:
        rm.DT = 0.5
        mdl = rm.Model(v, M=24, protocols=list(rm.PROTOCOLS))
        o = mdl.run(results[v]["params"], t_dense)
        for i, p in enumerate(mdl.protocols):
            for j, t in enumerate(t_dense):
                rows.append(dict(variant=v, protocol=p, day=rm.T0_DAY + t / 24, R_outer=o[i, j, 0],
                                 R_inhib=o[i, j, 1], R_nec=o[i, j, 2], R_pim=o[i, j, 3], f_Q=o[i, j, 4]))
    pd.DataFrame(rows).to_csv(f7.ROOT / "results" / f"radial_pred_{LINE}{SUFFIX}.csv", index=False,
                              float_format="%.5g")
