"""Model-free test: does cell-cycle arrest depend on oxygen history?

Under an instantaneous oxygen-threshold hypothesis (the assumption behind
Greenspan-type inhibited regions), the arrested fraction is a function of the
current oxygenation state only. We therefore compare spheroids with the same
current oxygenation (pimonidazole-positive fraction, spheroid size) but a
different oxygen history.

Outputs
-------
results/history_tests.csv
figures/fig02_hysteresis_loops.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")


def sel(line, prot, day):
    return d[(d.cell_line == line) & (d.protocol == prot) & (d.day == day)]


def bootstrap_diff(a, b, n=20000, seed=1):
    rng = np.random.default_rng(seed)
    da = rng.choice(a, (n, a.size)).mean(1) - rng.choice(b, (n, b.size)).mean(1)
    return np.percentile(da, [2.5, 97.5])


# (label, switched group, history-free reference with matched oxygenation)
COMPARISONS = [
    ("exit lag: WM793b 24 h after 2%->21%", ("WM793b", "reox_d2", 3), ("WM793b", "normoxia", 3)),
    ("exit lag: WM793b 48 h after 2%->21%", ("WM793b", "reox_d2", 4), ("WM793b", "normoxia", 4)),
    ("exit complete: WM793b 48 h after 2%->21% (day-4 switch)", ("WM793b", "reox_d4", 6), ("WM793b", "normoxia", 6)),
    ("exit lag: WM164 48 h after 2%->21% (day 4 switch)", ("WM164", "reox_d4", 6), ("WM164", "normoxia", 4)),
    ("entry lag: WM983b 24 h after 21%->2%", ("WM983b", "deox_d2", 3), ("WM983b", "hypoxia", 6)),
    ("entry lag: WM983b 48 h after 21%->2%", ("WM983b", "deox_d2", 4), ("WM983b", "hypoxia", 6)),
    ("entry lag: WM164 day 2 in 2% from seeding", ("WM164", "hypoxia", 2), ("WM164", "hypoxia", 8)),
    ("entry: WM793b 24 h after 21%->2%", ("WM793b", "deox_d2", 3), ("WM793b", "hypoxia", 4)),
]

rows = []
for label, g1, g0 in COMPARISONS:
    a, b = sel(*g1), sel(*g0)
    fa, fb = a.f_arrested.values, b.f_arrested.values
    p = mannwhitneyu(fa, fb, alternative="two-sided").pvalue
    lo, hi = bootstrap_diff(fa, fb)
    rows.append({
        "comparison": label,
        "group": "{} {} d{:g}".format(*g1), "reference": "{} {} d{:g}".format(*g0),
        "n_group": len(a), "n_ref": len(b),
        "f_hyp_group": a.f_hypoxic.mean(), "f_hyp_ref": b.f_hypoxic.mean(),
        "R_outer_group": a.R_outer.mean(), "R_outer_ref": b.R_outer.mean(),
        "f_arr_group": fa.mean(), "f_arr_ref": fb.mean(),
        "diff_f_arr": fa.mean() - fb.mean(), "diff_CI95_lo": lo, "diff_CI95_hi": hi,
        "mannwhitney_p": p,
    })
res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "history_tests.csv", index=False, float_format="%.4g")
with pd.option_context("display.width", 250, "display.max_colwidth", 60):
    print(res.round(4).to_string(index=False))

# ---------------- hysteresis figure: (hypoxic fraction, arrested fraction) ----------------
summ = pd.read_csv(ROOT / "results" / "summary_by_condition.csv")
START = {"deox_d2": ("normoxia", [2]), "reox_d2": ("hypoxia", [2]), "reox_d4": ("hypoxia", [2, 4])}
COLORS = {"normoxia": "#1f77b4", "hypoxia": "#ff7f0e", "deox_d2": "#2ca02c",
          "reox_d2": "#d62728", "reox_d4": "#9467bd"}
LABELS = {"normoxia": "21%", "hypoxia": "2%", "deox_d2": "21%→2% (day 2)",
          "reox_d2": "2%→21% (day 2)", "reox_d4": "2%→21% (day 4)"}
lines = ["WM793b", "WM164", "WM983b"]
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, line in zip(axes, lines):
    for prot, col in COLORS.items():
        s = summ[(summ.cell_line == line) & (summ.protocol == prot)]
        if s.empty:
            continue
        if prot in START:
            base, days = START[prot]
            s = pd.concat([summ[(summ.cell_line == line) & (summ.protocol == base)
                                & summ.day.isin(days)], s])
        s = s.sort_values("day")
        x, y = s.f_hypoxic_mean.values, s.f_arrested_mean.values
        ax.errorbar(x, y, xerr=s.f_hypoxic_se, yerr=s.f_arrested_se, color=col,
                    marker="o", ms=4, capsize=2, label=LABELS[prot])
        for k in range(len(x) - 1):
            ax.annotate("", xy=(x[k + 1], y[k + 1]), xytext=(x[k], y[k]),
                        arrowprops=dict(arrowstyle="->", color=col, lw=1.2))
        for xi, yi, day in zip(x, y, s.day):
            ax.text(xi + 0.008, yi + 0.008, f"{day:g}", fontsize=6, color=col)
    ax.plot([0, 0.7], [0, 0.7], ls=":", c="grey", lw=0.8)
    ax.set_title(line)
    ax.set_xlabel("pimonidazole-positive fraction (hypoxia)")
    ax.set_xlim(-0.02, 0.72)
    ax.set_ylim(-0.02, 0.62)
axes[0].set_ylabel("FUCCI-arrested fraction of living volume")
axes[0].legend(fontsize=7, frameon=False, loc="upper left")
fig.suptitle("Arrest follows oxygen with a lag: trajectories in the (hypoxia, arrest) plane; "
             "numbers = day", fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig02_hysteresis_loops.png", dpi=200)
