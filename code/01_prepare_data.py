"""Build a tidy per-spheroid table from the Murphy et al. (2023) oxygen-switch data.

Each condition code belongs to exactly one oxygen protocol; the shared
starting points (normoxia day 2, hypoxia days 2 and 4) are attached to the
protocols that branch from them when the protocols are simulated.

Outputs
-------
data/processed/spheroids.csv      one row per retained spheroid
results/summary_by_condition.csv  mean, SD and SE per protocol and day
figures/fig01_data_overview.png   arrested / hypoxic / necrotic fractions
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Murphy2023_oxygen"

# code -> (protocol, switch day); verified against the authors' MATLAB loaders
PROTOCOL = {}
for c in (17, 18, 19, 15, 16):
    PROTOCOL[c] = ("normoxia", np.nan)
for c in (1, 2, 3, 4):
    PROTOCOL[c] = ("hypoxia", np.nan)
for c in (11, 12, 13, 14):
    PROTOCOL[c] = ("deox_d2", 2.0)
for c in (5, 6, 7, 8):
    PROTOCOL[c] = ("reox_d2", 2.0)
for c in (9, 10):
    PROTOCOL[c] = ("reox_d4", 4.0)

FILES = {"WM793b": "ExpOA_WM793b.xlsx", "WM164": "ExpOA_WM164.xlsx",
         "WM983b": "ExpOA_WM983b.xlsx"}


def load(line):
    d = pd.read_excel(RAW / FILES[line])
    d = d[(d.Keep == 1) & d.Condition.isin(PROTOCOL)].copy()
    d["cell_line"] = line
    d["protocol"] = d.Condition.map(lambda c: PROTOCOL[c][0])
    d["t_switch"] = d.Condition.map(lambda c: PROTOCOL[c][1])
    ro = d.OuterRadius.astype(float)
    rn = d.NecroticRadius.clip(lower=0).astype(float)
    ri = np.maximum(d.InhibitedRadius.clip(lower=0).astype(float), rn)
    rp = np.minimum(d.PIMOuterRadius.clip(lower=0).astype(float), ro)
    out = pd.DataFrame({
        "cell_line": d.cell_line, "condition": d.Condition, "protocol": d.protocol,
        "t_switch": d.t_switch, "day": d.Day.astype(float), "replicate": d.Replicate,
        "R_outer": ro, "R_inhib": ri, "R_nec": rn, "R_pim": rp,
    })
    vol = lambda r: 4.0 / 3.0 * np.pi * r**3 / 1e6  # 1e6 um^3
    out["V_outer"] = vol(ro)
    out["V_living"] = vol(ro) - vol(rn)
    out["f_arrested"] = (ri**3 - rn**3) / (ro**3 - rn**3)
    out["f_necrotic"] = rn**3 / ro**3
    out["f_hypoxic"] = rp**3 / ro**3
    return out


df = pd.concat([load(l) for l in FILES], ignore_index=True)
df.to_csv(ROOT / "data" / "processed" / "spheroids.csv", index=False, float_format="%.6g")

cols = ["R_outer", "V_living", "f_arrested", "f_hypoxic", "f_necrotic"]
g = df.groupby(["cell_line", "protocol", "day"])
summ = g[cols].agg(["mean", "std", "count"])
summ.columns = ["_".join(c) for c in summ.columns]
summ = summ.reset_index()
for c in cols:
    summ[c + "_se"] = summ[c + "_std"] / np.sqrt(summ[c + "_count"])
summ.to_csv(ROOT / "results" / "summary_by_condition.csv", index=False, float_format="%.5g")

# ---- overview figure: branches drawn from their shared starting point ----
BRANCH_START = {"deox_d2": ("normoxia", [2]), "reox_d2": ("hypoxia", [2]),
                "reox_d4": ("hypoxia", [2, 4])}
COLORS = {"normoxia": "#1f77b4", "hypoxia": "#ff7f0e", "deox_d2": "#2ca02c",
          "reox_d2": "#d62728", "reox_d4": "#9467bd"}
LABELS = {"normoxia": "21% O$_2$", "hypoxia": "2% O$_2$", "deox_d2": "21%→2% at day 2",
          "reox_d2": "2%→21% at day 2", "reox_d4": "2%→21% at day 4"}


def branch(line, prot, var):
    s = summ[(summ.cell_line == line) & (summ.protocol == prot)]
    if prot in BRANCH_START:
        base, days = BRANCH_START[prot]
        b = summ[(summ.cell_line == line) & (summ.protocol == base) & summ.day.isin(days)]
        s = pd.concat([b, s])
    return s.sort_values("day")


lines = ["WM793b", "WM164", "WM983b"]
vars_ = [("f_arrested", "arrested fraction\nof living volume"),
         ("f_hypoxic", "pimonidazole-positive\nfraction"),
         ("R_outer", "outer radius (µm)")]
fig, axes = plt.subplots(len(vars_), len(lines), figsize=(11, 8), sharex=True)
for j, line in enumerate(lines):
    for i, (v, lab) in enumerate(vars_):
        ax = axes[i, j]
        for prot in COLORS:
            s = branch(line, prot, v)
            if s.empty or (prot not in ("normoxia", "hypoxia") and
                           (summ[(summ.cell_line == line) & (summ.protocol == prot)].empty)):
                continue
            ax.errorbar(s.day, s[v + "_mean"], yerr=s[v + "_se"], marker="o", ms=4,
                        capsize=2, color=COLORS[prot], label=LABELS[prot])
        if i == 0:
            ax.set_title(line)
        if j == 0:
            ax.set_ylabel(lab)
        if i == len(vars_) - 1:
            ax.set_xlabel("day after seeding")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("Murphy et al. (2023) oxygen-switch spheroids: mean ± SE", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig01_data_overview.png", dpi=200)

with pd.option_context("display.width", 200):
    show = summ[["cell_line", "protocol", "day", "f_arrested_count", "R_outer_mean",
                 "f_arrested_mean", "f_arrested_se", "f_hypoxic_mean", "f_necrotic_mean"]]
    print(show.round(3).to_string(index=False))
