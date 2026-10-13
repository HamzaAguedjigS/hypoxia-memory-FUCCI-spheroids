"""Figure for the Tier 3 fits: data (mean +/- SE) and the three model variants.

Usage: python 05_plot_mechanistic.py [cell_line]
Output: figures/fig04_mechanistic_<line>.png
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
pred = pd.read_csv(ROOT / "results" / f"mech_pred_{LINE}.csv")
summ = pd.read_csv(ROOT / "results" / "summary_by_condition.csv")
summ = summ[summ.cell_line == LINE]
fits = json.loads((ROOT / "results" / f"mech_fit_{LINE}.json").read_text())

START = {"deox_d2": "normoxia", "reox_d2": "hypoxia", "reox_d4": "hypoxia"}
PROTS = ["normoxia", "hypoxia", "deox_d2", "reox_d2", "reox_d4"]
TITLES = {"normoxia": "21% O$_2$", "hypoxia": "2% O$_2$", "deox_d2": "21%→2% at day 2",
          "reox_d2": "2%→21% at day 2", "reox_d4": "2%→21% at day 4\n(held out: prediction)"}
VARS = [("R_outer", "outer radius (µm)"), ("f_arrested", "arrested fraction"),
        ("f_necrotic", "necrotic fraction")]
STYLE = {"finite": ("#d62728", "-", "finite exit (exponential)"),
         "twostage": ("#1f77b4", "--", "finite exit (two-stage)"),
         "instant": ("#7f7f7f", ":", "instantaneous arrest")}

fig, axes = plt.subplots(3, 5, figsize=(15, 8), sharex=True)
for j, prot in enumerate(PROTS):
    s = summ[summ.protocol == prot]
    if prot in START:
        base = summ[(summ.protocol == START[prot]) & (summ.day <= (4 if prot == "reox_d4" else 2))]
        s = pd.concat([base, s])
    s = s.sort_values("day")
    for i, (v, lab) in enumerate(VARS):
        ax = axes[i, j]
        ax.errorbar(s.day, s[v + "_mean"], yerr=s[v + "_se"], fmt="ko", ms=4, capsize=2,
                    label="data (mean ± SE)")
        for var, (col, ls, name) in STYLE.items():
            p = pred[(pred.variant == var) & (pred.protocol == prot)]
            ax.plot(p.day, p[v], ls, color=col, lw=1.5, label=name)
        if prot in ("deox_d2", "reox_d2"):
            ax.axvline(2, color="grey", lw=0.6, ls="-.")
        if prot == "reox_d4":
            ax.axvline(4, color="grey", lw=0.6, ls="-.")
            ax.set_facecolor("#f7f7f7")
        if i == 0:
            ax.set_title(TITLES[prot], fontsize=9)
        if j == 0:
            ax.set_ylabel(lab)
        if i == 2:
            ax.set_xlabel("day")
axes[0, 0].legend(fontsize=7, frameon=False)
aic = ", ".join(f"{STYLE[v][2]}: AIC {fits[v]['AIC']:.0f}" for v in STYLE)
fig.suptitle(f"{LINE}: joint fit to the available oxygen protocols; {aic}", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "figures" / f"fig04_mechanistic_{LINE}.png", dpi=200)
