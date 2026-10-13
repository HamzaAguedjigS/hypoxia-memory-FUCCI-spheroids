"""Mechanistic view of the fitted radial model: radial profiles over time.

For the best-supported variant (or the one given), plot at several times
after reoxygenation (day 2 switch) and deoxygenation (day 2 switch):
local oxygen p(r), arrested share Q/v and green share G/v versus radius,
with the observed radii (data means) marked. This is the figure that shows
whether arrest outlives hypoxia (memory) in space and time.

Usage: python 10_radial_profiles.py LINE [variant]
Output: figures/fig07_profiles_<LINE>_<variant>.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import radial_model as rm

ROOT = Path(__file__).resolve().parents[1]
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
fits = json.loads((ROOT / "results" / f"radial_fit_{LINE}.json").read_text())
VARIANT = sys.argv[2] if len(sys.argv) > 2 else min(fits, key=lambda v: fits[v]["AIC"])
th = fits[VARIANT]["params"]
d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
d = d[d.cell_line == LINE]

M = 40
rm.DT = 0.5
CASES = [("reox_d2", "hypoxia", [2.0, 2.5, 3.0, 4.0, 6.0]),
         ("deox_d2", "normoxia", [2.0, 2.5, 3.0, 4.0, 6.0])]


def states_at(protocol, days):
    mdl = rm.Model(VARIANT, M=M, protocols=[protocol])
    S = mdl.initial(th)
    out, k_target = {}, {int(round((dd - rm.T0_DAY) * 24 / rm.DT)): dd for dd in days}
    for k in range(max(k_target) + 1):
        if k in k_target:
            out[k_target[k]] = [s.copy() for s in S]
        t = k * rm.DT
        te = t + rm.DT / 2
        k1 = mdl.derivs(t, S, th, te)
        k2 = mdl.derivs(t + rm.DT / 2, [s + rm.DT / 2 * x for s, x in zip(S, k1)], th, te)
        k3 = mdl.derivs(t + rm.DT / 2, [s + rm.DT / 2 * x for s, x in zip(S, k2)], th, te)
        k4 = mdl.derivs(t + rm.DT, [s + rm.DT * x for s, x in zip(S, k3)], th, te)
        S = [np.clip(s + rm.DT / 6 * (a + 2 * b + 2 * c + e), 0, None) if i < 5
             else s + rm.DT / 6 * (a + 2 * b + 2 * c + e)
             for i, (s, a, b, c, e) in enumerate(zip(S, k1, k2, k3, k4))]
    return mdl, out


def profile(mdl, S, t_h):
    R, E, G, Q, D, m = S
    if VARIANT == "instQ":
        living = R + G + 1e-12
        p, *_ = rm.local_p(living, (R + G) / living, D, th["a"] * m, mdl.p_inf(t_h - 1e-6))
        R, Q = mdl.split_inst(R, p, th)
    living = R + E + G + Q + 1e-12
    p, r_in, r_out, r_mid = rm.local_p(living, (R + E + G + rm.EPS_Q * Q) / living, D,
                                       th["a"] * m, mdl.p_inf(t_h - 1e-6))
    return r_mid[0], p[0], ((Q + E) / living)[0], (G / living)[0], mdl.observe(S, t_h, th)[0]


fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
cmap = plt.get_cmap("viridis")
for row, (prot, base, days) in enumerate(CASES):
    mdl, snaps = states_at(prot, days)
    for j, dd in enumerate(days):
        col = cmap(j / (len(days) - 1))
        r, p, q, g, obs = profile(mdl, snaps[dd], (dd - rm.T0_DAY) * 24)
        lab = f"day {dd:g}"
        axes[row, 0].plot(r, p, color=col, label=lab)
        axes[row, 1].plot(r, q, color=col, label=lab)
        axes[row, 2].plot(r, g / max(g.max(), 1e-12), color=col, label=lab)
        axes[row, 2].axvline(obs[1], color=col, ls=":", lw=0.8)
    p_pim = th.get("p_pim", rm.P_PIM)
    axes[row, 0].axhline(p_pim, color="k", ls="--", lw=0.7)
    axes[row, 0].axhline(th["Kq"], color="tab:red", ls=":", lw=0.7)
    axes[row, 0].text(5, p_pim * 1.3, f"fitted pimonidazole threshold ({p_pim:.2f}%)", fontsize=7)
    axes[row, 0].text(5, th["Kq"] * 1.3, f"arrest threshold K$_q$ ({th['Kq']:.2f}%)", fontsize=7,
                      color="tab:red")
    axes[row, 0].set_yscale("symlog", linthresh=1.0)
    axes[row, 2].axhline(rm.GREEN_THRESHOLD, color="k", ls="--", lw=0.7)
    for ax, lab in zip(axes[row], ["local O$_2$ (%)", "arrested share Q/v",
                                   "green share (relative to max)"]):
        ax.set_xlabel("radius (µm)")
        ax.set_ylabel(lab)
    title = {"reox_d2": "2%→21% at day 2", "deox_d2": "21%→2% at day 2"}[prot]
    axes[row, 0].set_title(f"{title}: oxygen")
    axes[row, 1].set_title(f"{title}: arrested cells")
    axes[row, 2].set_title(f"{title}: FUCCI-green (dotted: model R$_i$)")
    axes[row, 1].legend(fontsize=7, frameon=False)
fig.suptitle(f"{LINE}, fitted radial model ({VARIANT}): arrest outlives hypoxia after reoxygenation",
             fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "figures" / f"fig07_profiles_{LINE}_{VARIANT}.png", dpi=180)
print("saved", VARIANT)
