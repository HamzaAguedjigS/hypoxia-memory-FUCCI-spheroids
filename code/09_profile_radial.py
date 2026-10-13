"""Profile likelihood of the arrest entry/exit rates in the radial model.

For a fitted variant, fix log10(param) on a grid around its estimate and
re-optimise all other parameters (warm start from the best fit).
Mean times: exit ~ 1/bmax (well-oxygenated cells), entry ~ 1/amax (anoxia).

Usage: python 09_profile_radial.py LINE VARIANT PARAM [n_grid] [half_width_log10]
Output: results/radial_profile_<LINE>_<VARIANT>_<PARAM>.csv
"""
import importlib.util
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

LINE, VARIANT, PARAM = sys.argv[1], sys.argv[2], sys.argv[3]
N_GRID = int(sys.argv[4]) if len(sys.argv) > 4 else 12
HALF = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

HERE = Path(__file__).resolve().parent
_saved_argv = sys.argv
sys.argv = [sys.argv[0], LINE, "1", VARIANT]  # 07 reads its own argv at import
spec = importlib.util.spec_from_file_location("f7", HERE / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f7)
sys.argv = _saved_argv  # spawned workers must see the original arguments
ROOT = f7.ROOT

fit = json.loads((ROOT / "results" / f"radial_fit_{LINE}.json").read_text())[VARIANT]
names, lo, hi = f7.bounds(VARIANT)
x_best = np.clip(np.log10([fit["params"][n] for n in names]), lo, hi)
j = names.index(PARAM)
others = [i for i in range(len(names)) if i != j]


def profile_point(val):
    def obj(z):
        x = x_best.copy()
        x[others] = z
        x[j] = val
        return f7.objective(x, VARIANT, True)  # fine integrator (unbiased)
    r1 = minimize(obj, x_best[others], method="Powell", options=dict(maxfev=1200, xtol=1e-3, ftol=1e-6))
    r2 = minimize(obj, r1.x, method="Nelder-Mead", options=dict(maxfev=600, xatol=1e-4, fatol=1e-6))
    return val, float(min(r1.fun, r2.fun))


if __name__ == "__main__":
    grid = np.clip(x_best[j] + np.linspace(-HALF, HALF, N_GRID), lo[j], hi[j])
    grid = np.unique(np.r_[grid, x_best[j]])
    with Pool() as pool:
        out = pool.map(profile_point, grid)
    df = pd.DataFrame(out, columns=["log10_" + PARAM, "nll"])
    best = df.nll.min()
    df["delta_nll"] = df.nll - best
    df[PARAM] = 10 ** df["log10_" + PARAM]
    df["mean_time_h"] = 1 / df[PARAM]
    df = df.sort_values(PARAM)
    df.to_csv(ROOT / "results" / f"radial_profile_{LINE}_{VARIANT}_{PARAM}.csv", index=False,
              float_format="%.5g")
    inside = df[df.delta_nll <= 1.92]
    print(df.round(3).to_string(index=False))
    print(f"95% profile interval for 1/{PARAM} (h): {inside.mean_time_h.min():.1f} - "
          f"{inside.mean_time_h.max():.1f} (grid-limited: edges {df.mean_time_h.min():.1f}, "
          f"{df.mean_time_h.max():.1f})")
