"""Sensitivity of the final all-data fits to EPS_Q and K2.

Each scenario changes one fixed modelling choice at a time and re-fits finQ,
instQ and noQ to all WM793b observations. The stored all-data estimates are
used only as starting points; every free parameter is re-optimised.

Outputs
-------
results/sensitivity_fixed_choices.json
results/sensitivity_fixed_choices.csv
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

os.environ["FIT_ALL"] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_argv = sys.argv
sys.argv = [sys.argv[0], "WM793b", "1", "finQ", "instQ", "noQ"]
spec = importlib.util.spec_from_file_location("f7_sensitivity", HERE / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f7)
sys.argv = _argv
rm = f7.rm

VARIANTS = ("finQ", "instQ", "noQ")
SCENARIOS = {
    "epsQ_0.25": (0.25, 0.10),
    "epsQ_1.00": (1.00, 0.10),
    "k2_0.07": (0.50, 0.07),
    "k2_0.14": (0.50, 0.14),
}
BASE = json.loads((ROOT / "results" / "radial_fit_WM793b_alldata.json").read_text())


def configure(eps_q, k2):
    rm.EPS_Q = float(eps_q)
    rm.K2 = float(k2)
    f7.MODELS.clear()


def baseline_x(variant):
    names, lo, hi = f7.bounds(variant)
    th = BASE[variant]["params"]
    return names, lo, hi, np.clip(np.log10([th[n] for n in names]), lo, hi)


def starts(variant):
    names, lo, hi, x = baseline_x(variant)
    out = [x]
    if variant == "finQ":
        for days in (5.0, 8.0, 11.0):
            y = x.copy()
            y[names.index("bmax")] = np.log10(1.0 / (24.0 * days))
            out.append(np.clip(y, lo, hi))
    return out


def fit_fast(job):
    scenario, variant, x0 = job
    eps_q, k2 = SCENARIOS[scenario]
    configure(eps_q, k2)
    r1 = minimize(f7.objective, x0, args=(variant, False), method="Powell",
                  options=dict(maxfev=1600, xtol=1e-3, ftol=1e-6))
    r2 = minimize(f7.objective, r1.x, args=(variant, False), method="Nelder-Mead",
                  options=dict(maxfev=600, xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    return scenario, variant, r.x.tolist(), float(r.fun)


def fit_fine(job):
    scenario, variant, x0 = job
    eps_q, k2 = SCENARIOS[scenario]
    configure(eps_q, k2)
    r1 = minimize(f7.objective, x0, args=(variant, True), method="Powell",
                  options=dict(maxfev=450, xtol=8e-4, ftol=1e-6))
    r2 = minimize(f7.objective, r1.x, args=(variant, True), method="Nelder-Mead",
                  options=dict(maxfev=180, xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    return scenario, variant, r.x.tolist(), float(r.fun)


def summarise(scenario, variant, x):
    eps_q, k2 = SCENARIOS[scenario]
    configure(eps_q, k2)
    names, lo, hi = f7.bounds(variant)
    x = np.clip(np.asarray(x), lo, hi)
    pred = f7.predict(x, variant, fine=True)
    nll, sds = f7.nll(pred, f7.IDX_TRAIN)
    n_params = len(names) + 4
    n_obs = int(np.isfinite(f7.IDX_TRAIN[2]).sum())
    th = f7.to_theta(x, variant)
    at_bound = [n for n, xi, l, h in zip(names, x, lo, hi)
                if xi <= l + 0.02 or xi >= h - 0.02]
    return {
        "scenario": scenario, "eps_Q": eps_q, "k2_h-1": k2, "variant": variant,
        "nll": float(nll), "AIC": float(2 * nll + 2 * n_params),
        "n_params": n_params, "n_obs": n_obs,
        "exit_days": float(1 / th["bmax"] / 24) if variant == "finQ" else None,
        "sd_R_out": float(sds[0]), "sd_R_inh": float(sds[1]),
        "sd_R_nec": float(sds[2]), "sd_R_pim": float(sds[3]),
        "at_bound": at_bound, "params": th,
    }


if __name__ == "__main__":
    jobs = [(s, v, x) for s in SCENARIOS for v in VARIANTS for x in starts(v)]
    with Pool(processes=4) as pool:
        fine = []
        for completed, result in enumerate(pool.imap_unordered(fit_fine, jobs), 1):
            fine.append(result)
            print(f"completed {completed}/{len(jobs)}: {result[0]} {result[1]} "
                  f"NLL={result[3]:.2f}", flush=True)

    rows = []
    details = {}
    for s in SCENARIOS:
        details[s] = {}
        for v in VARIANTS:
            candidates = sorted([z for z in fine if z[0] == s and z[1] == v], key=lambda z: z[3])
            result = summarise(s, v, candidates[0][2])
            details[s][v] = result
            rows.append({k: val for k, val in result.items() if k != "params"})
            print(f"{s:10s} {v:5s} NLL={result['nll']:.2f} AIC={result['AIC']:.2f} "
                  f"exit={result['exit_days']} bounds={result['at_bound']}", flush=True)

    df = pd.DataFrame(rows)
    best = df.groupby("scenario").AIC.transform("min")
    df["delta_AIC"] = df.AIC - best
    df.to_csv(ROOT / "results" / "sensitivity_fixed_choices.csv", index=False,
              float_format="%.6g")
    (ROOT / "results" / "sensitivity_fixed_choices.json").write_text(
        json.dumps(details, indent=2))
    print("\n", df[["scenario", "variant", "nll", "AIC", "delta_AIC", "exit_days"]]
          .round(2).to_string(index=False), flush=True)
