"""Out-of-sample test of the exit time: training-only profile with held-out scores.

The finQ model is fitted without the day-4 reoxygenation (training set) with
the exit time 1/bmax fixed on a grid; every other parameter is re-optimised
(fine integrator, two starting points: the training-only and the all-data
optima). For each grid point the held-out protocol is then predicted with
the training noise SDs. The result shows which exit times the training data
alone can exclude, and which ones the held-out protocol excludes.

Usage: python 21_validation_profile.py [LINE] [N_WORKERS]
Outputs: results/validation_profile_<LINE>.csv, results/log_21_validation_<LINE>.txt (stdout)
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

os.environ.pop("FIT_ALL", None)  # hold out reox_d4 (also in spawned workers)
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [sys.argv[0], LINE, "1"]
spec = importlib.util.spec_from_file_location("f7", HERE / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f7)
sys.argv = _argv
assert f7.HELD_OUT == ["reox_d4"]

EXIT_GRID = [96.0, 125.0, 144.0, 200.0, 300.0, 500.0, 800.0, 1500.0]
SOURCES = {"training": f"radial_fit_{LINE}.json", "alldata": f"radial_fit_{LINE}_alldata.json"}
VARIANT = "finQ"
names, lo, hi = f7.bounds(VARIANT)
jb = names.index("bmax")
others = [i for i in range(len(names)) if i != jb]


def x_from(source, variant=VARIANT):
    n, l, h = f7.bounds(variant)
    th = dict(f7.GUESS)
    th.update(json.loads((f7.ROOT / "results" / SOURCES[source]).read_text())[variant]["params"])
    return np.clip(np.log10([th[k] for k in n]), l, h)


def scores(x, variant):
    """Training NLL, held-out NLL (training SDs) and held-out R_inh predictions."""
    pred = f7.predict(x, variant, fine=True)
    tr, sds = f7.nll(pred, f7.IDX_TRAIN)
    te, _ = f7.nll(pred, f7.IDX_TEST, sds)
    pi = f7.PROTS.index("reox_d4")
    ti6 = int(np.argmin(np.abs(f7.T_OBS - (6 - f7.rm.T0_DAY) * 24)))
    ti8 = int(np.argmin(np.abs(f7.T_OBS - (8 - f7.rm.T0_DAY) * 24)))
    return dict(nll_train=tr, nll_test=te, Rinh_d6_pred=float(pred[pi, ti6, 1]),
                Rinh_d8_pred=float(pred[pi, ti8, 1]), sd_inh_train=float(sds[1]))


def job(args):
    exit_h, source = args
    x0 = x_from(source)
    x0[jb] = np.log10(1 / exit_h)

    def obj(z):
        x = x0.copy()
        x[others] = z
        return f7.objective(x, VARIANT, True)
    r1 = minimize(obj, x0[others], method="Powell", options=dict(maxfev=1400, xtol=1e-3, ftol=1e-6))
    r2 = minimize(obj, r1.x, method="Nelder-Mead", options=dict(maxfev=500, xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    x = x0.copy()
    x[others] = r.x
    x = np.clip(x, lo, hi)
    out = dict(exit_time_h=exit_h, start=source, G1_h=float(1 / 10 ** x[names.index("k1max")]))
    out.update(scores(x, VARIANT))
    return out


if __name__ == "__main__":
    jobs = [(e, s) for e in EXIT_GRID for s in SOURCES]
    with Pool(N_WORKERS) as pool:
        out = []
        for i, o in enumerate(pool.imap_unordered(job, jobs), 1):
            out.append(o)
            print(f"[{i}/{len(jobs)}] exit={o['exit_time_h']:.0f} h start={o['start']} "
                  f"train={o['nll_train']:.2f} test={o['nll_test']:.2f} "
                  f"Rinh_d6={o['Rinh_d6_pred']:.0f} um", flush=True)
    df = pd.DataFrame(out)
    best = df.loc[df.groupby("exit_time_h").nll_train.idxmin()].sort_values("exit_time_h").copy()
    best["delta_train"] = best.nll_train - best.nll_train.min()
    best["delta_test"] = best.nll_test - best.nll_test.min()
    obs = f7.d[(f7.d.protocol == "reox_d4")].groupby("day").R_inhib.mean()
    best["Rinh_d6_obs"] = float(obs.get(6, np.nan))
    best["Rinh_d8_obs"] = float(obs.get(8, np.nan))
    # reference: held-out scores of the stored training-only instQ and noQ fits
    refs = {}
    for v in ("instQ", "noQ"):
        refs[v] = scores(x_from("training", v), v)
    best["nll_test_instQ"] = refs["instQ"]["nll_test"]
    best["nll_test_noQ"] = refs["noQ"]["nll_test"]
    df.to_csv(f7.ROOT / "results" / f"validation_profile_{LINE}_all.csv", index=False, float_format="%.6g")
    best.to_csv(f7.ROOT / "results" / f"validation_profile_{LINE}.csv", index=False, float_format="%.6g")
    print("\nTraining-only finQ exit profile with held-out (day-4 reoxygenation) scores")
    print(best[["exit_time_h", "G1_h", "nll_train", "delta_train", "nll_test", "delta_test",
                "Rinh_d6_pred", "Rinh_d6_obs"]].round(2).to_string(index=False))
    print({v: round(r["nll_test"], 2) for v, r in refs.items()}, "(held-out NLL, instQ / noQ)")
