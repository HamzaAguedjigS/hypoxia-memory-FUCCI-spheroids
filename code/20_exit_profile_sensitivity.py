"""Exit-time profiles under the fixed modelling choices, and a consistent baseline.

Complements 17_sensitivity_fixed_choices.py, which refits finQ, instQ and noQ
freely under each fixed choice. This script adds what that run does not give:

  * baseline (eps_Q = 0.5, S/G2/M = 10 h): the stored all-data finQ optimum
    (NLL 3933.1, exit 180 h) is not the best point found; the exit-time
    profile reached NLL 3920.8 at 125 h. Here finQ and delayQ are profiled on
    a grid and the best grid point is polished with every parameter free;
    instQ and noQ are re-polished from their stored optima, so that the four
    models of the AIC table share one optimisation protocol;
  * every fixed-choice scenario: the finQ exit-time profile on the same grid
    (does the 5-11 day range survive?) and a free polish from its best point.

All fits use every protocol (FIT_ALL) and the fine integrator (DT = 0.5 h,
24 shells). Each result is appended to a .jsonl file as soon as it is done.

Usage: python 20_exit_profile_sensitivity.py [LINE] [N_WORKERS]
Outputs: results/exit_profile_sensitivity_<LINE>.csv / .json / _partial.jsonl
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

os.environ["FIT_ALL"] = "1"  # every protocol in the likelihood (also in spawned workers)
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
N_WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [sys.argv[0], LINE, "1"]
spec = importlib.util.spec_from_file_location("f7", HERE / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f7)
sys.argv = _argv
rm = f7.rm

SETTINGS = {  # name: (eps_Q, k2 per h); same scenarios as script 17
    "baseline": (0.5, 0.10),
    "epsQ_0.25": (0.25, 0.10),
    "epsQ_1.00": (1.00, 0.10),
    "k2_0.07": (0.5, 0.07),
    "k2_0.14": (0.5, 0.14),
}
EXIT_GRID = [96.0, 125.0, 180.0, 260.0, 400.0]
FITS = json.loads((f7.ROOT / "results" / f"radial_fit_{LINE}_alldata.json").read_text())
OUT = f7.ROOT / "results" / f"exit_profile_sensitivity_{LINE}"
PARTIAL = Path(str(OUT) + "_partial.jsonl")


def x_stored(variant):
    names, lo, hi = f7.bounds(variant)
    th = dict(f7.GUESS)
    th.update(FITS[variant]["params"])
    return np.clip(np.log10([th[n] for n in names]), lo, hi)


def apply(setting):
    rm.EPS_Q, rm.K2 = SETTINGS[setting]
    f7.MODELS.clear()


def fixed_exit(setting, variant, exit_h):
    apply(setting)
    names, lo, hi = f7.bounds(variant)
    jb = names.index("bmax")
    others = [i for i in range(len(names)) if i != jb]
    x0 = x_stored(variant)
    x0[jb] = np.log10(1 / exit_h)

    def obj(z):
        x = x0.copy()
        x[others] = z
        return f7.objective(x, variant, True)
    r1 = minimize(obj, x0[others], method="Powell", options=dict(maxfev=1400, xtol=1e-3, ftol=1e-6))
    r2 = minimize(obj, r1.x, method="Nelder-Mead", options=dict(maxfev=500, xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    x = x0.copy()
    x[others] = r.x
    return dict(setting=setting, variant=variant, stage="fixed_exit", exit_fixed_h=exit_h,
                nll=float(r.fun), x=x.tolist())


def free_fit(setting, variant, x0):
    apply(setting)
    r1 = minimize(f7.objective, np.asarray(x0), args=(variant, True), method="Powell",
                  options=dict(maxfev=1500, xtol=1e-3, ftol=1e-6))
    r2 = minimize(f7.objective, r1.x, args=(variant, True), method="Nelder-Mead",
                  options=dict(maxfev=700, xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    return dict(setting=setting, variant=variant, stage="free", exit_fixed_h=np.nan,
                nll=float(r.fun), x=r.x.tolist())


def run_job(args):
    return fixed_exit(*args[1:]) if args[0] == "fixed" else free_fit(*args[1:])


def run_all(pool, jobs, label):
    out = []
    for i, o in enumerate(pool.imap_unordered(run_job, jobs), 1):
        out.append(o)
        with PARTIAL.open("a") as fh:
            fh.write(json.dumps(o) + "\n")
        print(f"[{label} {i}/{len(jobs)}] {o['setting']} {o['variant']} {o['stage']} "
              f"exit={o['exit_fixed_h']} NLL={o['nll']:.2f}", flush=True)
    return out


def summarise(row):
    apply(row["setting"])
    v = row["variant"]
    names, lo, hi = f7.bounds(v)
    x = np.clip(np.array(row["x"]), lo, hi)
    th = f7.to_theta(x, v)
    pred = f7.predict(x, v, fine=True)
    nll, sds = f7.nll(pred, f7.IDX_TRAIN)  # re-evaluated at the clipped point
    k = len(names) + 4  # model parameters + one SD per radius type
    row.update(nll=float(nll), k=k, AIC=2 * float(nll) + 2 * k, G1_h=1 / th["k1max"],
               exit_h=1 / th["bmax"] if v in ("finQ", "delayQ") else np.nan,
               Kq=th["Kq"] if v != "noQ" else np.nan, m_h=th["m_h"], p_pim=th["p_pim"],
               sd_out=sds[0], sd_inh=sds[1], sd_nec=sds[2], sd_pim=sds[3],
               at_bound=";".join(n for n, xi, l, h in zip(names, x, lo, hi)
                                 if xi <= l + 0.02 or xi >= h - 0.02),
               params=th)
    return row


if __name__ == "__main__":
    PARTIAL.write_text("")
    stage1 = [("fixed", s, "finQ", e) for s in SETTINGS for e in EXIT_GRID]
    stage1 += [("fixed", "baseline", "delayQ", e) for e in (125.0, 180.0, 260.0)]
    stage1 += [("free", "baseline", v, x_stored(v).tolist()) for v in ("instQ", "noQ")]
    with Pool(N_WORKERS) as pool:
        out1 = run_all(pool, stage1, "stage 1")
        stage2 = []
        for s in SETTINGS:
            for v in ("finQ", "delayQ"):
                runs = [o for o in out1 if o["setting"] == s and o["variant"] == v]
                if runs:
                    stage2.append(("free", s, v, min(runs, key=lambda o: o["nll"])["x"]))
        out2 = run_all(pool, stage2, "stage 2")

    rows = [summarise(dict(o)) for o in out1 + out2]
    df = pd.DataFrame(rows)
    df.drop(columns=["x", "params"]).to_csv(str(OUT) + ".csv", index=False, float_format="%.6g")
    best = {}
    for s in SETTINGS:
        best[s] = {}
        for v in ("finQ", "delayQ", "instQ", "noQ"):
            sub = [r for r in rows if r["setting"] == s and r["variant"] == v]
            if sub:
                b = dict(min(sub, key=lambda r: r["nll"]))
                b.pop("x")
                best[s][v] = b
    Path(str(OUT) + ".json").write_text(json.dumps(best, indent=2, default=float))

    print(f"\nExit-time profiles and consistent baseline, {LINE} (all protocols, fine integrator)")
    for s, (eps, k2) in SETTINGS.items():
        b = best[s]
        print(f"\n{s}: eps_Q={eps}, S/G2/M={1 / k2:.1f} h")
        for v in ("finQ", "delayQ", "instQ", "noQ"):
            if v in b:
                print(f"  {v:7s} NLL={b[v]['nll']:9.2f} AIC={b[v]['AIC']:9.1f} exit={b[v]['exit_h']:6.0f} h "
                      f"G1={b[v]['G1_h']:5.1f} h bounds={b[v]['at_bound']}")
        prof = df[(df.setting == s) & (df.variant == "finQ") & (df.stage == "fixed_exit")]
        nmin = b["finQ"]["nll"]
        print("  finQ exit profile (dNLL vs best):",
              {int(e): round(n - nmin, 1) for e, n in zip(prof.exit_fixed_h, prof.nll)})
