"""Oxygen-solver and pimonidazole-boundary checks for the final (v8) discretisation.

Repeats the Greenspan comparison of 23_numerical_checks.py with the equal-thickness
shell layout used in the final analysis and adds the exact pimonidazole radius:
  1. max |p_numeric - p_exact| at shell midpoints (% O2), uniform consumption q/D;
  2. |R_pim(numeric, exact within-shell profile) - R_pim(Greenspan)| (um) for the
     thresholds 0.17 % O2 (fitted p_pim) and 1.3 % O2 (literature value).

Output: results/v8/oxygen_check_v8.txt (also printed)
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("rm", HERE / "radial_model.py")
rm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rm)
OUT = HERE.parent / "results" / "v8" / "oxygen_check_v8.txt"
Q = 0.0025                     # q/D, % O2 um^-2
lines = []


def out(s=""):
    print(s, flush=True)
    lines.append(s)


def p_all(x, ra, R, pinf):
    return pinf - Q / 6 * (R**2 - x**2) + Q * ra**3 / 3 * (1 / x - 1 / R)


def greenspan(r, R, pinf, rc=0.0):
    """Uniform consumption for r > r_a; none for r < r_a = max(core, anoxic radius)."""
    lo = max(rc, 1e-6)
    ra = rc if p_all(lo, rc, R, pinf) >= 0 else brentq(lambda s: p_all(s, s, R, pinf), lo, R)
    inner = p_all(max(ra, 1e-6), ra, R, pinf) if ra > 0 else 0.0
    return np.maximum(np.where(r > ra, p_all(np.maximum(r, 1e-9), ra, R, pinf), inner), 0.0), ra


def pim_exact(R, pinf, rc, level):
    ra = greenspan(np.array([R]), R, pinf, rc)[1]
    f = lambda x: p_all(x, ra, R, pinf) - level
    lo = max(ra, 1e-6)
    if f(lo) >= 0:
        return 0.0 if ra == 0 else ra
    return brentq(f, lo, R)


def shells(R, rc, M, layout):
    if layout == "thickness":
        e = np.linspace(rc, R, M + 1)
    else:
        e = np.cbrt(np.linspace(rc**3, R**3, M + 1))
    return (4 / 3 * np.pi * (e[1:] ** 3 - e[:-1] ** 3) / 1e6)[None, :]


cases = [("no anoxia, R = 100 um, 21%", 100.0, 21.0, 0.0),
         ("anoxic region, R = 300 um, 21%", 300.0, 21.0, 0.0),
         ("anoxic region, R = 250 um, 2%", 250.0, 2.0, 0.0),
         ("core 80 um + anoxic region, R = 300 um, 21%", 300.0, 21.0, 80.0)]
GRID = (24, 32, 64, 96, 200)
for layout in ("thickness", "volume"):
    out(f"Layout: equal {layout} shells")
    out("  1. max |p_num - p_exact| at shell midpoints (% O2)")
    for name, R, pinf, rc in cases:
        Vc = np.array([4 / 3 * np.pi * rc**3 / 1e6])
        errs = []
        for M in GRID:
            v = shells(R, rc, M, layout)
            p, _, _, rmid = rm.local_p(v, np.ones_like(v), Vc, np.array([Q]), np.array([pinf]))
            pa, ra = greenspan(rmid[0], R, pinf, rc)
            errs.append(f"M={M}: {np.max(np.abs(p[0] - pa)):.4f}")
        out(f"     {name:46s} (r_a = {ra:5.1f} um)  " + "; ".join(errs))
    out("  2. |R_pim numeric - R_pim exact| (um)")
    for level in (0.17, 1.3):
        for name, R, pinf, rc in cases[1:]:
            Vc = np.array([4 / 3 * np.pi * rc**3 / 1e6])
            ref = pim_exact(R, pinf, rc, level)
            errs = []
            for M in GRID:
                v = shells(R, rc, M, layout)
                rp = rm.pim_radius_exact(v, np.ones_like(v), Vc, np.array([Q]), np.array([pinf]), level)[0]
                errs.append(f"M={M}: {abs(rp - ref):.3f}")
            out(f"     p_pim = {level:4.2f}%, {name:38s} (exact {ref:6.1f} um)  " + "; ".join(errs))
    out()
OUT.write_text("\n".join(lines), encoding="utf8")
print(f"saved {OUT}")
