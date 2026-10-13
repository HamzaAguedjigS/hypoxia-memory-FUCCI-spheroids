"""Proposed replacement for the exit-profile figure, in the style of the identifiability figures.

Reads results/v8/exit_profile_all.csv, exit_profile_training.csv, final_fits.json and the
WM793b radii, and draws three panels against the mean exit time:
  (a) training profile (day-4 reoxygenation withheld), with the all-protocol profile for comparison;
  (b) score of every training-profile solution on the withheld arm, with the held-out scores of the
      M_F and M_I training optima;
  (c) inhibited radius predicted on day 6 of the withheld arm, against the observed mean.

Profiles are shown as -dNLL (0 at the best point, downward = worse), as in Figs. 4-5 and S1.
Writes figures/fig06_exit_profile_v2.pdf and .png only (the article folder is not touched).

Usage: python 32_exit_profile_figure.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "results" / "v8"
FIG = ROOT / "figures"
SUPPORT, CHI2, REJECT = 10.0, 1.92, 40.0
RED, DARK, GREY = "#c0392b", "#2b2b2b", "#7f7f7f"

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.2, "xtick.labelsize": 7.3,
                     "ytick.labelsize": 7.3, "legend.fontsize": 7.0, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "legend.frameon": False})

ALL = pd.read_csv(V8 / "exit_profile_all.csv")
TR = pd.read_csv(V8 / "exit_profile_training.csv")
FIN = json.loads((V8 / "final_fits.json").read_text())
d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
obs6 = d[(d.cell_line == "WM793b") & (d.protocol == "reox_d4") & (d.day == 6)].R_inhib
TEST_MIN = TR.nll_test.min()
EXIT_TR_OPT = 1 / FIN["train"]["finQ"]["params"]["bmax"] / 24
HO_F = FIN["train"]["finQ"]["nll_test"] - TEST_MIN
HO_I = FIN["train"]["instQ"]["nll_test"] - TEST_MIN
TICKS = [2, 4, 7, 14, 28, 56]


def tint(color, w):
    r, g, b = plt.matplotlib.colors.to_rgb(color)
    return (1 - w + w * r, 1 - w + w * g, 1 - w + w * b)


def smooth(ax, x, y, **kw):
    lx = np.log10(x)
    f = np.linspace(lx.min(), lx.max(), 300)
    ax.plot(10 ** f, PchipInterpolator(lx, y)(f), **kw)


def frame(ax, ymin, color):
    ax.axhspan(-SUPPORT, 0.6 if ymin > -60 else 2, color=tint(color, .16), lw=0, zorder=0)
    ax.axhline(-CHI2, color="#555555", lw=.6, ls=(0, (1, 1.5)), zorder=1)
    ax.axhline(-REJECT, color="#555555", lw=.7, ls=(0, (4, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(1.8, 70)
    ax.xaxis.set_major_locator(FixedLocator(TICKS))
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("mean exit time (d)")
    ax.grid(axis="y", color="#ececec", lw=.4, zorder=0)


fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw=dict(wspace=.42))

# (a) training data alone leave long exits supported
ax = axes[0]
frame(ax, -48, RED)
xa, ya = ALL.exit_h.values / 24, ALL.dNLL.values
smooth(ax, xa, -ya, color=tint(RED, .45), lw=1.2, zorder=2)
xt, yt = TR.exit_h.values / 24, TR.dTrain.values
xo = np.append(xt, EXIT_TR_OPT)          # free training optimum, dTrain = 0
yo = np.append(yt, 0.0)
o = np.argsort(xo)
smooth(ax, xo[o], -yo[o], color=RED, lw=1.9, zorder=3)
ax.scatter(xt, -yt, s=16, facecolor="white", edgecolor=RED, lw=1.1, zorder=4)
low = ya > 48
ax.scatter(xa[low], np.full(low.sum(), -46.8), marker="v", s=14, color=tint(RED, .45), zorder=4, clip_on=False)
ax.plot([EXIT_TR_OPT], [0], marker="D", ms=4.6, color=RED, mec="black", mew=.5, zorder=5)
ax.set_ylim(-48, 3)
ax.set_yticks([0, -10, -20, -30, -40])
ax.set_ylabel(r"$-\Delta$NLL")
ax.set_title("(a) training protocols", loc="left", color=RED, fontweight="bold", pad=3)
ax.plot([], [], color=RED, lw=1.9, marker="o", mfc="white", ms=4, label="day-4 arm withheld")
ax.plot([], [], color=tint(RED, .45), lw=1.2, label="all protocols")
ax.legend(loc="lower right", handlelength=1.8)

# (b) the withheld arm bounds exit from above
ax = axes[1]
frame(ax, -160, DARK)
yh = TR.dTest.values
smooth(ax, xt, -yh, color=DARK, lw=1.7, ls=(0, (4, 1.6)), zorder=3)
ax.scatter(xt, -yh, marker="s", s=15, facecolor="white", edgecolor=DARK, lw=1.1, zorder=4)
ax.axhline(-HO_I, color=GREY, lw=1.0, ls=(0, (6, 2, 1, 2)), zorder=2)
ax.text(66, -HO_I - 3, r"$\mathcal{M}_I$ optimum", ha="right", va="top", fontsize=6.6, color=GREY)
ax.plot([EXIT_TR_OPT], [-HO_F], marker="D", ms=4.6, color=RED, mec="black", mew=.5, zorder=5)
ax.annotate(r"$\mathcal{M}_F$ optimum", xy=(EXIT_TR_OPT, -HO_F), xytext=(2.1, -30), fontsize=6.6,
            color=RED, arrowprops=dict(arrowstyle="-", lw=.6, color=RED))
ax.set_ylim(-160, 6)
ax.set_yticks([0, -40, -80, -120, -160])
ax.set_ylabel(r"held-out $-\Delta$NLL")
ax.set_title("(b) withheld day-4 arm", loc="left", color=DARK, fontweight="bold", pad=3)

# (c) mechanism of the rejection: predicted day-6 inhibited radius
ax = axes[2]
m, se = obs6.mean(), obs6.sem()
ax.axhspan(m - se, m + se, color="#e6e6e6", lw=0, zorder=0)
ax.axhline(m, color="black", lw=.9, zorder=1)
ax.text(66, 2, f"observed {m:.1f} ± {se:.1f} µm", fontsize=6.6, va="bottom", ha="right")
smooth(ax, xt, TR.Rinh_d6.values, color=RED, lw=1.9, zorder=3)
ax.scatter(xt, TR.Rinh_d6.values, s=16, facecolor="white", edgecolor=RED, lw=1.1, zorder=4)
ax.set_xscale("log")
ax.set_xlim(1.8, 70)
ax.xaxis.set_major_locator(FixedLocator(TICKS))
ax.set_xticklabels([str(t) for t in TICKS])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("mean exit time (d)")
ax.set_ylim(0, 135)
ax.set_ylabel(r"predicted $R_{\mathrm{inh}}$, day 6 (µm)")
ax.set_title("(c) day-6 inhibited radius", loc="left", color=RED, fontweight="bold", pad=3)
ax.grid(axis="y", color="#ececec", lw=.4, zorder=0)

fig.savefig(FIG / "fig06_exit_profile_v2.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig06_exit_profile_v2.png", dpi=300, bbox_inches="tight")
print(f"saved figures/fig06_exit_profile_v2 (.pdf, .png); held-out dNLL of training optima: "
      f"M_F {HO_F:.1f} (exit {EXIT_TR_OPT:.2f} d), M_I {HO_I:.1f}")
