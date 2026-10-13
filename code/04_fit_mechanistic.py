"""Tier 3: fit the mechanistic model to all oxygen protocols of one cell line.

Training protocols: normoxia, hypoxia, deox_d2, reox_d2 (days 3-8).
Held-out protocol : reox_d4 (days 6 and 8), predicted without refitting.
Initial conditions: day-2 means of the normoxia and hypoxia arms (fixed).

Likelihood (per spheroid): outer radius ~ Gaussian(sd_R); arrested and
necrotic fractions ~ Gaussian with var = s^2 f(1-f) + 0.005^2. The three
noise scales are maximised out.

Usage: python 04_fit_mechanistic.py [cell_line] [n_starts]
Outputs: results/mech_fit_<line>.json, results/mech_pred_<line>.csv,
         figures/fig04_mechanistic_<line>.png
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

import mech_model as mm

ROOT = Path(__file__).resolve().parents[1]
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
N_STARTS = int(sys.argv[2]) if len(sys.argv) > 2 else 16
TRAIN = ["normoxia", "hypoxia", "deox_d2", "reox_d2"]
TEST = ["reox_d4"]
VARIANTS = ["finite", "twostage", "instant"]
SIGMA0 = 0.005
S_GRID = np.geomspace(0.02, 3.0, 160)[:, None]

# log10 bounds (per hour, volume unit 1e6 um^3, oxygen relative to 21%)
BOUNDS = {"k1bar": (-3, 0), "mu0": (-5, -1), "ell": (-3, 0), "K": (-3, 0),
          "abar": (-3, 1), "bbar": (-3, 0.5), "muh": (-4, 0), "kD": (-4, -0.5)}
LO = np.array([BOUNDS[k][0] for k in mm.PARAM_NAMES])
HI = np.array([BOUNDS[k][1] for k in mm.PARAM_NAMES])
I_BBAR = mm.PARAM_NAMES.index("bbar")

d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
d = d[d.cell_line == LINE]
IC = {}
for g in ("normoxia", "hypoxia"):
    s = d[(d.protocol == g) & (d.day == 2)]
    IC[g] = (s.V_living.mean(), s.f_arrested.mean(), s.f_necrotic.mean())


def obs_table(protocols):
    s = d[d.protocol.isin(protocols) & (d.day > 2)]
    return {p: s[s.protocol == p] for p in protocols if (s.protocol == p).any()}


DATA_TRAIN, DATA_TEST = obs_table(TRAIN), obs_table(TEST)


def frac_ll(f_model, f_obs):
    fm = np.clip(f_model, 1e-9, 1 - 1e-9)[None, :]
    var = S_GRID**2 * fm * (1 - fm) + SIGMA0**2
    return (-0.5 * np.sum(np.log(2 * np.pi * var) + (f_obs[None, :] - fm) ** 2 / var, 1)).max()


def predict(p, data, variant):
    out = {}
    for prot, s in data.items():
        tt = np.unique((s.day.values - 2) * 24.0)
        st = mm.simulate(p, prot, IC, tt, variant)
        if st is None or not np.all(np.isfinite(st)):
            return None
        r, fa, fn = mm.observables(st)
        idx = np.searchsorted(tt, (s.day.values - 2) * 24.0)
        out[prot] = (r[idx], fa[idx], fn[idx])
    return out


def nll_parts(p, data, variant, sd_r=None):
    pr = predict(p, data, variant)
    if pr is None:
        return None
    r_m = np.concatenate([pr[k][0] for k in data])
    a_m = np.concatenate([pr[k][1] for k in data])
    n_m = np.concatenate([pr[k][2] for k in data])
    r_o = np.concatenate([data[k].R_outer.values for k in data])
    a_o = np.concatenate([data[k].f_arrested.values for k in data])
    n_o = np.concatenate([data[k].f_necrotic.values for k in data])
    rss = np.sum((r_o - r_m) ** 2)
    n = r_o.size
    if sd_r is None:
        ll_r = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    else:
        ll_r = -0.5 * np.sum(np.log(2 * np.pi * sd_r**2) + (r_o - r_m) ** 2 / sd_r**2)
    return -ll_r, -frac_ll(a_m, a_o), -frac_ll(n_m, n_o), np.sqrt(rss / n)


def to_params(x, variant):
    p = 10.0 ** np.asarray(x)
    if variant == "instant":
        p[I_BBAR] = 1.0  # only abar/bbar matters; fast rates set by LAMBDA_FAST
    return p


def objective(x, variant):
    x = np.asarray(x)
    pen = np.sum(np.clip(LO - x, 0, None) ** 2 + np.clip(x - HI, 0, None) ** 2) * 1e3
    parts = nll_parts(to_params(np.clip(x, LO, HI), variant), DATA_TRAIN, variant)
    if parts is None:
        return 1e8
    return parts[0] + parts[1] + parts[2] + pen


def run_start(args):
    x0, variant = args
    best = minimize(objective, x0, args=(variant,), method="Powell",
                    options=dict(maxfev=4000, xtol=1e-4, ftol=1e-7))
    res = minimize(objective, best.x, args=(variant,), method="Nelder-Mead",
                   options=dict(maxfev=3000, xatol=1e-5, fatol=1e-7))
    if res.fun > best.fun:
        res = best
    return variant, res.x.tolist(), float(res.fun)


def n_free(variant):
    return len(mm.PARAM_NAMES) - (1 if variant == "instant" else 0) + 3  # + noise scales


if __name__ == "__main__":
    starts = qmc.scale(qmc.LatinHypercube(d=len(LO), seed=11).random(N_STARTS), LO, HI)
    jobs = [(x, v) for v in VARIANTS for x in starts]
    with Pool() as pool:
        out = pool.map(run_start, jobs)
    results = {}
    for v in VARIANTS:
        runs = sorted([o for o in out if o[0] == v], key=lambda o: o[2])
        x = np.clip(np.array(runs[0][1]), LO, HI)
        p = to_params(x, v)
        tr = nll_parts(p, DATA_TRAIN, v)
        sd_r = tr[3]
        te = nll_parts(p, DATA_TEST, v, sd_r=sd_r) if DATA_TEST else (np.nan,) * 4
        n_train = sum(len(s) for s in DATA_TRAIN.values())
        k = n_free(v)
        nll = tr[0] + tr[1] + tr[2]
        results[v] = dict(
            params=dict(zip(mm.PARAM_NAMES, p.tolist())), log10_params=x.tolist(),
            nll_train=nll, nll_train_parts=dict(radius=tr[0], arrested=tr[1], necrotic=tr[2]),
            sd_radius_um=sd_r, n_params=k, n_train_spheroids=n_train,
            AIC=2 * nll + 2 * k, BIC=2 * nll + k * np.log(3 * n_train),
            nll_test_radius=te[0], nll_test_arrested=te[1], nll_test_necrotic=te[2],
            start_nlls=[o[2] for o in runs],
        )
        if v != "instant":
            results[v]["mean_exit_time_normoxic_h"] = 1.0 / (p[I_BBAR] * 1.0 / (p[3] + 1.0))
        print(f"{v:9s} NLL={nll:9.2f} AIC={results[v]['AIC']:9.2f} "
              f"test NLL(arr)={te[1]:8.2f} sdR={sd_r:.1f}  best starts: "
              + " ".join(f"{o[2]:.1f}" for o in runs[:5]))
        print("          params:", {k_: round(v_, 5) for k_, v_ in results[v]["params"].items()})
    (ROOT / "results" / f"mech_fit_{LINE}.json").write_text(json.dumps(results, indent=2))

    # ---------------- dense predictions for plotting
    rows = []
    tt = np.linspace(0, 144, 289)
    for v in VARIANTS:
        p = np.array(list(results[v]["params"].values()))
        for prot in mm.PROTOCOLS:
            st = mm.simulate(p, prot, IC, tt, v)
            r, fa, fn = mm.observables(st)
            c = np.where((prot == "reox_d4") & (tt < 48), mm.C_HYP,
                         {"normoxia": 1.0, "deox_d2": mm.C_HYP, "hypoxia": mm.C_HYP,
                          "reox_d2": 1.0, "reox_d4": 1.0}[prot])
            ox = mm.oxygen(st, p, c)
            for i, t in enumerate(tt):
                rows.append(dict(variant=v, protocol=prot, day=2 + t / 24, R_outer=r[i],
                                 f_arrested=fa[i], f_necrotic=fn[i], N=ox[i]))
    pd.DataFrame(rows).to_csv(ROOT / "results" / f"mech_pred_{LINE}.csv", index=False,
                              float_format="%.5g")
