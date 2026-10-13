"""Numerical checks reported in Appendix B.

1. Oxygen solver against Greenspan's analytic solution (uniform consumption,
   optional non-consuming core, anoxic region where oxygen would be negative).
2. Discretisation sensitivity of the final all-data fits: every model is
   re-scored (parameters unchanged) with finer time steps and more shells.
3. The baseline exit-time profile points of 20_exit_profile_sensitivity.py
   re-scored at the same finer discretisations.

Usage: python 23_numerical_checks.py   (FIT_ALL is set internally)
Output: results/numerical_checks_WM793b.txt (also printed)
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

os.environ["FIT_ALL"] = "1"
HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [sys.argv[0], "WM793b", "1"]
spec = importlib.util.spec_from_file_location("f7", HERE / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f7)
sys.argv = _argv
rm = f7.rm
RES = f7.ROOT / "results"
GRIDS = [(0.5, 24), (0.25, 24), (0.5, 32), (0.25, 32), (0.25, 48)]
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


# ---------------------------------------------------------------- 1. oxygen solver
def greenspan(r, R, pinf, q, rc=0.0):
    """Uniform consumption q for r > r_a; no consumption for r < r_a = max(core, anoxic radius)."""
    def p_all(x, ra):
        return pinf - q / 6 * (R**2 - x**2) + q * ra**3 / 3 * (1 / x - 1 / R)
    lo = max(rc, 1e-6)
    ra = rc if p_all(lo, rc) >= 0 else brentq(lambda s: p_all(s, s), lo, R)
    inner = p_all(max(ra, 1e-6), ra) if ra > 0 else 0.0
    return np.maximum(np.where(r > ra, p_all(np.maximum(r, 1e-9), ra), inner), 0.0), ra


def solver(R, pinf, q, M, rc=0.0):
    V, Vc = 4 / 3 * np.pi * R**3 / 1e6, 4 / 3 * np.pi * rc**3 / 1e6
    living = np.full((1, M), (V - Vc) / M)
    p, _, _, r_mid = rm.local_p(living, np.ones((1, M)), np.array([Vc]), np.array([q]), np.array([pinf]))
    return p[0], r_mid[0]


out("1. Oxygen solver against Greenspan's analytic profile (q/D = 0.0025 % O2 um^-2)")
out("   max |p_numeric - p_exact| at shell midpoints, % O2")
cases = [("no anoxia, R = 100 um, 21%", 100.0, 21.0, 0.0),
         ("anoxic region, R = 300 um, 21%", 300.0, 21.0, 0.0),
         ("anoxic region, R = 250 um, 2%", 250.0, 2.0, 0.0),
         ("non-consuming core 80 um + anoxic region, R = 300 um, 21%", 300.0, 21.0, 80.0)]
for name, R, pinf, rc in cases:
    errs = []
    for M in (16, 24, 32, 50, 200):
        p, rmid = solver(R, pinf, 0.0025, M, rc)
        pa, ra = greenspan(rmid, R, pinf, 0.0025, rc)
        errs.append(f"M={M}: {np.max(np.abs(p - pa)):.4f}")
    out(f"   {name:58s} (r_a = {ra:5.1f} um)  " + "; ".join(errs))

# ---------------------------------------------------------------- 2. final fits re-scored
fits = json.loads((RES / "radial_fit_WM793b_alldata.json").read_text())
out("\n2. Final all-data fits re-scored with the parameters unchanged (dAIC vs best, NLL, sd_inh / sd_pim)")
base = {}
for dt, M in GRIDS:
    row = {}
    for v, x in fits.items():
        rm.DT = dt
        pred = rm.Model(v, M=M, protocols=f7.PROTS).run(x["params"], f7.T_OBS)
        nll, sds = f7.nll(pred, f7.IDX_TRAIN)
        row[v] = (2 * nll + 2 * (len(x["free"]) + 4), nll, sds, pred)
        if (dt, M) == GRIDS[0]:
            base[v] = pred
    best = min(a for a, *_ in row.values())
    out(f"   dt = {dt} h, M = {M}: " + "; ".join(
        f"{v} {a - best:+.1f} (NLL {n:.1f}, sd {s[1]:.1f}/{s[3]:.1f})" for v, (a, n, s, _) in row.items()))
    if (dt, M) != GRIDS[0]:
        d = np.abs(row["finQ"][3][..., :4] - base["finQ"][..., :4])
        out(f"      finQ max radius change vs (0.5 h, 24): out {np.nanmax(d[..., 0]):.1f}, inh {np.nanmax(d[..., 1]):.1f}, "
            f"nec {np.nanmax(d[..., 2]):.1f}, pim {np.nanmax(d[..., 3]):.1f} um")
rm.DT = 0.5

# ---------------------------------------------------------------- 3. exit profile re-scored
partial = RES / "exit_profile_sensitivity_WM793b_partial.jsonl"
if partial.exists():
    rows = [json.loads(s) for s in partial.read_text().splitlines() if s.strip()]
    rows = [r for r in rows if r["setting"] == "baseline" and r["variant"] == "finQ"]
    names, lo, hi = f7.bounds("finQ")
    scored = {}
    for r in rows:
        th = f7.to_theta(np.clip(np.array(r["x"]), lo, hi), "finQ")
        vals = []
        for dt, M in GRIDS:
            rm.DT = dt
            pred = rm.Model("finQ", M=M, protocols=f7.PROTS).run(th, f7.T_OBS)
            vals.append(f7.nll(pred, f7.IDX_TRAIN)[0])
        scored[(1 / th["bmax"], r["stage"])] = vals
    rm.DT = 0.5
    ref = [min(v[i] for v in scored.values()) for i in range(len(GRIDS))]
    out("\n3. Baseline exit-profile points re-scored (dNLL vs best point at each discretisation)")
    out("   " + " | ".join(f"dt={dt},M={M}" for dt, M in GRIDS))
    for (e, stage), v in sorted(scored.items()):
        out(f"   exit {e:6.0f} h ({stage:10s}): " + "  ".join(f"{v[i] - ref[i]:6.1f}" for i in range(len(GRIDS))))

(RES / "numerical_checks_WM793b.txt").write_text("\n".join(lines) + "\n")
