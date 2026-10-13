"""Fit the radial model (radial_model.py) to every spheroid of one cell line.

Data: the four measured radii (outer, inhibited, necrotic, pimonidazole) of
every retained spheroid, days 2-8, all oxygen protocols. The reoxygenation at
day 4 (reox_d4) is held out and predicted.
Likelihood: independent Gaussian errors on each radius type with its own SD
(maximised out analytically: NLL_k = n_k/2 log(RSS_k / n_k) + const).
Optimisation: multistart (Latin hypercube + one literature-guided start),
Powell then Nelder-Mead, fast integrator (DT = 1 h, 16 shells); the best fit
is re-evaluated with DT = 0.5 h and 24 shells.

Usage: python 07_fit_radial.py LINE N_STARTS [variants...]
Outputs: results/radial_fit_<LINE>.json, results/radial_pred_<LINE>.csv
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

import radial_model as rm

ROOT = Path(__file__).resolve().parents[1]
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
N_STARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
VARIANTS = sys.argv[3:] or ["finQ", "noQ", "instQ", "delayQ"]
# FIT_ALL=1: final estimation on every protocol after the out-of-sample check
HELD_OUT = [] if os.environ.get("FIT_ALL") else ["reox_d4"]
G1_STAGES = int(os.environ.get("G1_STAGES", "1"))  # Erlang G1 robustness checks
# fine integrator: 24 shells up to v5; v6 uses the converged 48 shells (RADIAL_FINE_M=48)
FINE_M = int(os.environ.get("RADIAL_FINE_M", "24"))
FINE_DT = float(os.environ.get("RADIAL_FINE_DT", "0.5"))
OBS = ["R_outer", "R_inhib", "R_nec", "R_pim"]

# v2 constraints (pre-specified before refitting):
#  * normoxic G1 duration 5-30 h (k1max in [1/30, 1/5] per h): total cycle 15-40 h,
#    the range reported for related FUCCI melanoma lines (C8161 21 h, WM983C 23 h,
#    1205Lu 37 h; Vittadello et al. 2018). Without it the optimiser explains slow
#    spheroid growth by a ~100 h G1, which alone creates recovery lags.
#  * baseline loss mu0 up to 0.05 per h, so slower spheroid growth can come from loss;
#  * necrosis only near anoxia (Kn fixed at 0.1% O2, cf. p_n = 0 in Murphy et al. 2023);
#  * steepness hq of the arrest switch in [1, 8] (a 2% spheroid spans only ~1-2% O2);
#  * baseline loss no longer feeds the necrotic core (radial_model.py);
#  * spheroids kept in 2% start metabolically adapted (radial_model.py).
# v3: effective pimonidazole detection threshold p_pim fitted in [0.05, 2]% O2
#     (data show R_i > R_p, i.e. arrest extends beyond detectable staining);
#     warm start from the previous version; best starts re-polished with the
#     fine integrator (DT = 0.5 h, 24 shells) because the fast one is biased.
# v4: external oxygen constant within each RK4 step (v2-v3 leaked the new
#     condition into the last stage before a switch); necrosis threshold Kn
#     fitted in [0.003, 0.3]% O2; necrotic clearance kD in 2% and kD_ox in 21%.
# v5: G1 bound widened to 5-120 h and exit times up to 1e4 h. For WM793b the
#     confounding test (11_exit_g1_confounding.py) showed that the data pin G1
#     at ~30 h by themselves, so the 30 h cap of v2-v4 is no longer needed.
LOG_BOUNDS = {"k1max": (np.log10(1 / 120), np.log10(1 / 5)), "K1": (-1.5, 1.3), "a": (-4.0, -2.0),
              "amax": (-3.0, 0.0), "bmax": (-4.0, 0.0), "Kq": (-1.0, 1.3), "hq": (0.0, 0.9),
              "mu0": (-4.0, -1.3), "mun": (-3.0, 0.0), "Kn": (-2.5, -0.5),
              "kD": (-3.5, -0.5), "kD_ox": (-3.5, -0.3), "m_h": (-1.5, 0.0), "tau_m": (0.0, 2.5),
              "R1_norm": (1.9, 2.45), "R1_hyp": (1.9, 2.45), "p_pim": (-1.3, 0.3)}
GUESS = {"k1max": 0.07, "K1": 0.3, "a": 1.6e-3, "amax": 0.1, "bmax": 0.05, "Kq": 1.6, "hq": 4.0,
         "mu0": 0.01, "mun": 0.1, "Kn": 0.1, "kD": 0.02, "kD_ox": 0.05, "m_h": 0.3, "tau_m": 12.0,
         "R1_norm": 150.0, "R1_hyp": 160.0, "p_pim": 1.3}
FIXED = ()
# initial-condition check (35_initial_condition.py): RADIAL_Q0_HYP=1 adds q0_hyp, the fraction of
# G1 cells already arrested at day 1 in low-oxygen shells of spheroids grown in 2% (finQ, delayQ);
# the default (0) leaves every earlier analysis unchanged
Q0_HYP = os.environ.get("RADIAL_Q0_HYP", "0") == "1"
LOG_BOUNDS["q0_hyp"] = (-3.0, 0.0)
GUESS["q0_hyp"] = 0.0


def free_names(variant):
    names = [n for n in rm.PARAM_NAMES if n not in FIXED]
    if variant == "noQ":
        for n in ("amax", "bmax", "Kq", "hq"):
            names.remove(n)
    if variant == "instQ":
        names.remove("bmax")  # only amax/bmax matters; bmax fixed to 1
    if Q0_HYP and variant in ("finQ", "delayQ"):
        names.append("q0_hyp")
    return names


def bounds(variant):
    names = free_names(variant)
    lo = np.array([LOG_BOUNDS[n][0] for n in names])
    hi = np.array([LOG_BOUNDS[n][1] for n in names])
    if variant == "instQ":
        i = names.index("amax")
        lo[i], hi[i] = -3.0, 3.0
    return names, lo, hi


def to_theta(x, variant):
    names = free_names(variant)
    th = dict(GUESS)
    th.update({n: 10.0 ** v for n, v in zip(names, x)})
    if variant == "instQ":
        th["bmax"] = 1.0
    if variant == "noQ":
        th["amax"], th["bmax"] = 0.0, 1.0
    return th


# ---------------------------------------------------------------- data
d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
d = d[d.cell_line == LINE].copy()
PROTS = [p for p in rm.PROTOCOLS if (d.protocol == p).any()]
TRAIN = [p for p in PROTS if p not in HELD_OUT]
DAYS = np.array(sorted(d.day.unique()))
T_OBS = np.r_[0.0, (DAYS - rm.T0_DAY) * 24.0]


def index_data(protocols):
    rows = d[d.protocol.isin(protocols)]
    pi = rows.protocol.map({p: i for i, p in enumerate(PROTS)}).values
    ti = np.searchsorted(T_OBS, (rows.day.values - rm.T0_DAY) * 24.0)
    return pi, ti, rows[OBS].values


IDX_TRAIN, IDX_TEST = index_data(TRAIN), index_data([p for p in HELD_OUT if p in PROTS])


def nll(pred, idx, sds=None):
    pi, ti, y = idx
    if len(y) == 0:
        return np.nan, None
    m = pred[pi, ti][:, :4]
    ok = np.isfinite(y)                 # missing measurements are skipped
    res = np.where(ok, y - m, 0.0)
    n = ok.sum(axis=0)
    if sds is None:
        rss = np.sum(res**2, axis=0)
        sds = np.sqrt(rss / n)
        return float(np.sum(0.5 * n * (np.log(2 * np.pi * rss / n) + 1))), sds
    return float(np.sum(0.5 * np.log(2 * np.pi * sds**2) * n + 0.5 * np.sum(res**2, 0) / sds**2)), sds


MODELS = {}


def model(variant, fine=False):
    key = (variant, fine)
    if key not in MODELS:
        rm.DT = FINE_DT if fine else 1.0
        MODELS[key] = (rm.Model(variant, M=FINE_M if fine else 16, protocols=PROTS,
                                g1_stages=G1_STAGES), rm.DT)
    return MODELS[key]


# v8 observation model: each predicted radius is the mean of the trajectory over
# t - W ... t + W (5 points, W = RADIAL_OBS_WINDOW h), representing the spread in developmental
# timing among independently formed spheroids; 0 (default) reproduces v1-v7.
OBS_WINDOW = float(os.environ.get("RADIAL_OBS_WINDOW", "0"))


def window_offsets(t, w=None):
    w = OBS_WINDOW if w is None else w
    return [0.0] if (w <= 0 or t < 24.0) else list(np.linspace(-w, w, 5))


def run_windowed(mdl, th, t_obs, w=None):
    """Model output at t_obs, averaged over the observation window."""
    t_obs = np.asarray(t_obs, float)
    t_run = np.array(sorted({t + o for t in t_obs for o in window_offsets(t, w)}))
    out = mdl.run(th, t_run)
    if out is None:
        return None
    pos = {t: i for i, t in enumerate(t_run)}
    return np.stack([out[:, [pos[t + o] for o in window_offsets(t, w)]].mean(axis=1) for t in t_obs], axis=1)


def predict(x, variant, fine=False):
    mdl, dt = model(variant, fine)
    rm.DT = dt
    if OBS_WINDOW > 0:
        return run_windowed(mdl, to_theta(x, variant), T_OBS)
    return mdl.run(to_theta(x, variant), T_OBS)


def objective(x, variant, fine=False):
    _, lo, hi = bounds(variant)
    pen = 1e3 * np.sum(np.clip(lo - x, 0, None) ** 2 + np.clip(x - hi, 0, None) ** 2)
    pred = predict(np.clip(x, lo, hi), variant, fine)
    if pred is None or not np.all(np.isfinite(pred[..., :4])):
        return 1e9
    return nll(pred, IDX_TRAIN)[0] + pen


def run_start(args):
    x0, variant = args
    r1 = minimize(objective, x0, args=(variant,), method="Powell",
                  options=dict(maxfev=2000, xtol=1e-3, ftol=1e-6))
    r2 = minimize(objective, r1.x, args=(variant,), method="Nelder-Mead",
                  options=dict(maxfev=1000, xatol=1e-4, fatol=1e-6))
    best = r2 if r2.fun < r1.fun else r1
    return variant, best.x.tolist(), float(best.fun)


def polish_fine(args):
    """Re-optimise a fast-integrator optimum with the fine integrator."""
    x0, variant = args
    r = minimize(objective, x0, args=(variant, True), method="Nelder-Mead",
                 options=dict(maxfev=700, xatol=1e-4, fatol=1e-6))
    return variant, r.x.tolist(), float(r.fun)


def warm_start(variant, names, lo, hi):
    """Best parameters of the most recent archived fit of this line, if any."""
    prev = sorted((ROOT / "results").glob(f"radial_fit_{LINE}_v*.json"))
    if not prev:
        return None
    fits = json.loads(prev[-1].read_text())
    if variant not in fits:
        return None
    th = dict(GUESS)
    th.update(fits[variant]["params"])
    return np.clip(np.log10([th[n] for n in names]), lo, hi)


if __name__ == "__main__":
    jobs = []
    for v in VARIANTS:
        names, lo, hi = bounds(v)
        g = np.clip(np.log10([GUESS[n] if not (v == "instQ" and n == "amax") else 2.0
                              for n in names]), lo, hi)
        w = warm_start(v, names, lo, hi)
        n_lhs = N_STARTS - 1 - (w is not None)
        lhs = qmc.scale(qmc.LatinHypercube(d=len(lo), seed=3).random(n_lhs), lo, hi)
        jobs += [(g, v)] + ([(w, v)] if w is not None else []) + [(x, v) for x in lhs]
    with Pool() as pool:
        out = pool.map(run_start, jobs)
        # fine-integrator polish of the three best fast optima of every variant
        top = []
        for v in VARIANTS:
            top += [(np.array(o[1]), v) for o in sorted([o for o in out if o[0] == v],
                                                         key=lambda o: o[2])[:3]]
        polished = pool.map(polish_fine, top)

    results = {}
    for v in VARIANTS:
        runs = sorted([o for o in out if o[0] == v], key=lambda o: o[2])
        fine_runs = sorted([o for o in polished if o[0] == v], key=lambda o: o[2])
        names, lo, hi = bounds(v)
        x = np.clip(np.array(fine_runs[0][1]), lo, hi)
        pred = predict(x, v, fine=True)
        tr, sds = nll(pred, IDX_TRAIN)
        te, _ = nll(pred, IDX_TEST, sds)
        k = len(names) + 4
        n_obs = int(np.isfinite(IDX_TRAIN[2]).sum())
        th = to_theta(x, v)
        at_bound = [n for n, xi, l, h in zip(names, x, lo, hi) if xi <= l + 0.02 or xi >= h - 0.02]
        results[v] = dict(params=th, free=names, at_bound=at_bound, nll_train=tr,
                          nll_train_fast=runs[0][2], sd=dict(zip(OBS, sds.tolist())),
                          n_params=k, n_obs=n_obs, AIC=2 * tr + 2 * k,
                          BIC=2 * tr + k * np.log(n_obs), nll_test=te,
                          start_nlls=[round(o[2], 2) for o in runs],
                          polished_nlls=[round(o[2], 2) for o in fine_runs])
        print(f"{LINE} {v:7s} NLL={tr:9.2f} AIC={2 * tr + 2 * k:9.2f} test={te:8.2f} "
              f"SD={np.round(sds, 1)} bounds={at_bound}")
        print("        starts:", results[v]["start_nlls"][:6], " polished:", results[v]["polished_nlls"])
        print("        params:", {n: round(val, 5) for n, val in th.items()})
    (ROOT / "results" / f"radial_fit_{LINE}.json").write_text(json.dumps(results, indent=2))

    # dense predictions for plots (fine integrator)
    rows = []
    t_dense = np.arange(0, 7 * 24 + 1, 6.0)
    for v in VARIANTS:
        names, lo, hi = bounds(v)
        th = results[v]["params"]
        rm.DT = 0.5
        mdl = rm.Model(v, M=24, protocols=list(rm.PROTOCOLS))
        out = mdl.run(th, t_dense)
        for i, p in enumerate(mdl.protocols):
            for j, t in enumerate(t_dense):
                rows.append(dict(variant=v, protocol=p, day=rm.T0_DAY + t / 24,
                                 R_outer=out[i, j, 0], R_inhib=out[i, j, 1], R_nec=out[i, j, 2],
                                 R_pim=out[i, j, 3], f_Q=out[i, j, 4]))
    pd.DataFrame(rows).to_csv(ROOT / "results" / f"radial_pred_{LINE}.csv", index=False,
                              float_format="%.5g")
