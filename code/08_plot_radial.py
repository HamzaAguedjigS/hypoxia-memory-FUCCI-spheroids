"""Figures for the radial-model fits: the four measured radii per protocol.

Usage: python 08_plot_radial.py LINE
Output: figures/fig06_radial_<LINE>.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
SUFFIX = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""   # e.g. "alldata"
pred = pd.read_csv(ROOT / "results" / f"radial_pred_{LINE}{SUFFIX}.csv")
fits = json.loads((ROOT / "results" / f"radial_fit_{LINE}{SUFFIX}.json").read_text())
d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
d = d[d.cell_line == LINE]

START = {"deox_d2": ("normoxia", [2]), "reox_d2": ("hypoxia", [2]), "reox_d4": ("hypoxia", [2, 4])}
PROTS = [p for p in ["normoxia", "hypoxia", "deox_d2", "reox_d2", "reox_d4"] if (d.protocol == p).any()]
TITLES = {"normoxia": "21% O$_2$", "hypoxia": "2% O$_2$", "deox_d2": "21%→2% at day 2",
          "reox_d2": "2%→21% at day 2",
          "reox_d4": "2%→21% at day 4" + ("" if SUFFIX else " (held out)")}
OBS = [("R_outer", "outer radius (µm)"), ("R_inhib", "inhibited radius (µm)"),
       ("R_nec", "necrotic radius (µm)"), ("R_pim", "hypoxic (PIM) radius (µm)")]
STYLE = {"noQ": ("#2ca02c", "-.", "no arrest state"),
         "instQ": ("#7f7f7f", ":", "instantaneous arrest"),
         "finQ": ("#d62728", "-", "finite arrest kinetics"),
         "delayQ": ("#1f77b4", "--", "finite + entry delay")}
SWITCH = {"deox_d2": 2, "reox_d2": 2, "reox_d4": 4}

fig, axes = plt.subplots(4, len(PROTS), figsize=(3 * len(PROTS), 10), sharex=True)
for j, prot in enumerate(PROTS):
    s = d[d.protocol == prot]
    if prot in START:
        base, days = START[prot]
        s = pd.concat([d[(d.protocol == base) & d.day.isin(days)], s])
    g = s.groupby("day")
    for i, (v, lab) in enumerate(OBS):
        ax = axes[i, j]
        ax.errorbar(g[v].mean().index, g[v].mean(), yerr=g[v].sem(), fmt="ko", ms=4,
                    capsize=2, label="data (mean ± SE)", zorder=5)
        for var, (col, ls, name) in STYLE.items():
            if var not in fits:
                continue
            p = pred[(pred.variant == var) & (pred.protocol == prot) & (pred.day >= 2)]
            ax.plot(p.day, p[v], ls, color=col, lw=1.4, label=name)
        if prot in SWITCH:
            ax.axvline(SWITCH[prot], color="grey", lw=0.6, ls="-.")
        if prot == "reox_d4" and not SUFFIX:
            ax.set_facecolor("#f5f5f5")
        if i == 0:
            ax.set_title(TITLES[prot], fontsize=9)
        if j == 0:
            ax.set_ylabel(lab)
        if i == 3:
            ax.set_xlabel("day")
axes[0, 0].legend(fontsize=7, frameon=False)
aic = "; ".join(f"{STYLE[v][2]}: AIC {fits[v]['AIC']:.0f}" for v in STYLE if v in fits)
fig.suptitle(f"{LINE} — radial model with FUCCI/pimonidazole observables. {aic}", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "figures" / f"fig06_radial_{LINE}{SUFFIX}.png", dpi=180)
