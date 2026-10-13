"""Main-text Figure 1 for Section 2: experimental design and model-free evidence."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
s = pd.read_csv(ROOT / "results" / "summary_by_condition.csv")

COL = {"normoxia": "#2878b5", "hypoxia": "#f28e2b", "deox_d2": "#2ca02c", "reox_d2": "#d62728", "reox_d4": "#9467bd"}
LAB = {"normoxia": "21%", "hypoxia": "2%", "deox_d2": "21%→2% (d2)", "reox_d2": "2%→21% (d2)", "reox_d4": "2%→21% (d4)"}
START = {"deox_d2": ("normoxia", [2]), "reox_d2": ("hypoxia", [2]), "reox_d4": ("hypoxia", [2, 4])}

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
fig = plt.figure(figsize=(7.2, 8.0), constrained_layout=True)
gs = GridSpec(4, 2, figure=fig, height_ratios=[0.78, 1.1, 0.11, 1.15])

# (a) Oxygen histories.
ax = fig.add_subplot(gs[0, 0])
protocols = [("21% chronic", 21, None, None), ("2% chronic", 2, None, None), ("21→2% at d2", 21, 2, 2), ("2→21% at d2", 2, 21, 2), ("2→21% at d4", 2, 21, 4)]
for y, (name, before, after, switch) in enumerate(protocols[::-1]):
    if switch is None:
        ax.plot([1, 8], [y, y], lw=2.4, color=COL["normoxia"] if before == 21 else COL["hypoxia"])
    else:
        ax.plot([1, switch], [y, y], lw=2.4, color=COL["normoxia"] if before == 21 else COL["hypoxia"])
        ax.plot([switch, 8], [y, y], lw=2.4, color=COL["normoxia"] if after == 21 else COL["hypoxia"])
        ax.plot([switch, switch], [y - .16, y + .16], color="black", lw=.8)
for day in [2, 3, 4, 6, 8]:
    ax.axvline(day, color="0.85", lw=.6, zorder=0)
    ax.scatter([day], [-.28], marker="v", s=15, color="0.25")
ax.set(yticks=range(5), yticklabels=[p[0] for p in protocols[::-1]], xticks=[2, 3, 4, 6, 8], xlim=(1, 8.15), ylim=(-.45, 5.0), xlabel="day after seeding")
ax.set_title("(a) Oxygen-switch protocols", loc="left", fontweight="bold")
ax.legend(handles=[Line2D([0], [0], color=COL["normoxia"], lw=2.4, label="21% O₂"), Line2D([0], [0], color=COL["hypoxia"], lw=2.4, label="2% O₂"), Line2D([0], [0], marker="v", color="0.25", lw=0, markersize=4, label="imaging")], frameon=False, loc="upper right", ncol=3, columnspacing=.8, handlelength=1.8)

# (b) Image-derived radii.
ax = fig.add_subplot(gs[0, 1], aspect="equal")
for radius, face, edge, label, angle, dashed in [(1.0, "#dbeaf5", "#2878b5", r"$R_{\mathrm{out}}$", 35, False), (.70, "#fee8c8", "#f28e2b", r"$R_{\mathrm{inh}}$", 18, False), (.49, "none", "#9467bd", r"$R_{\mathrm{pim}}$", -5, True), (.29, "#bdbdbd", "#666666", r"$R_{\mathrm{nec}}$", -55, False)]:
    ax.add_patch(Circle((0, 0), radius, facecolor=face, edgecolor=edge, lw=2, ls="--" if dashed else "-"))
    th = np.deg2rad(angle)
    text_xy = (1.18, -.88) if label == r"$R_{\mathrm{nec}}$" else (1.18, radius*np.sin(th))
    ax.annotate(label, xy=(radius*np.cos(th), radius*np.sin(th)), xytext=text_xy, arrowprops={"arrowstyle": "-", "color": edge}, color=edge, va="center")
ax.text(-.95, -1.19, r"equatorial section; green signal <20% inward of $R_{\mathrm{inh}}$", fontsize=8)
ax.set(xlim=(-1.2, 1.55), ylim=(-1.25, 1.12))
ax.axis("off")
ax.set_title("(b) Image-derived radii", loc="left", fontweight="bold")

# (c) WM793b response after reoxygenation.
ax = fig.add_subplot(gs[1, 0])
for prot in ["hypoxia", "reox_d2", "reox_d4", "normoxia"]:
    z = s[(s.cell_line == "WM793b") & (s.protocol == prot)].sort_values("day")
    if prot in START:
        base, days = START[prot]
        z = pd.concat([s[(s.cell_line == "WM793b") & (s.protocol == base) & s.day.isin(days)], z]).sort_values("day")
    ax.errorbar(z.day, z.f_arrested_mean, yerr=z.f_arrested_se, marker="o", ms=3.5, lw=1.3, capsize=2, color=COL[prot], label=LAB[prot])
    ax.errorbar(z.day, z.f_hypoxic_mean, yerr=z.f_hypoxic_se, marker="s", mfc="white", ms=3.5, lw=1.0, ls="--", capsize=2, color=COL[prot])
ax.set(xlabel="day after seeding", ylabel="volume fraction", xticks=[2, 3, 4, 6, 8], ylim=(-.02, .48))
ax.set_title("(c) WM793b:\nhypoxia clears before inhibition", loc="left", fontweight="bold", fontsize=9.5)
ax.text(.02, .96, r"filled circles: $f_{\mathrm{inh}}$; open squares: $f_{\mathrm{hyp}}$", transform=ax.transAxes, va="top", fontsize=8)

# (d) Key per-spheroid contrasts 24 h after switching.
ax = fig.add_subplot(gs[1, 1])
groups = [("WM793b\nreox d3", "WM793b", "reox_d2", 3, "#d62728", "o", -.006), ("WM793b\n21% d3", "WM793b", "normoxia", 3, "#2878b5", "o", .006), ("WM983b\ndeox d3", "WM983b", "deox_d2", 3, "#2ca02c", "s", 0), ("WM983b\n2% d6", "WM983b", "hypoxia", 6, "#f28e2b", "s", 0)]
for label, line, prot, day, color, marker, offset in groups:
    z = d[(d.cell_line == line) & (d.protocol == prot) & (d.day == day)]
    ax.scatter(z.f_hypoxic + offset, z.f_arrested, s=26, alpha=.72, color=color, marker=marker, edgecolor="white", linewidth=.3, label=label)
ax.set(xlabel=r"$f_{\mathrm{hyp}}$", ylabel=r"$f_{\mathrm{inh}}$", xlim=(-.025, .72), ylim=(-.025, .43))
ax.set_title("(d) Different histories,\ndifferent inhibition", loc="left", fontweight="bold", fontsize=9.5)
ax.legend(frameon=False, fontsize=8, ncol=2)

# Common protocol legend for panels (c) and (e).
legax = fig.add_subplot(gs[2, :])
legax.axis("off")
legax.legend(handles=[Line2D([0], [0], color=COL[p], marker="o", ms=4, lw=1.3, label=LAB[p]) for p in ["normoxia", "hypoxia", "deox_d2", "reox_d2", "reox_d4"]], loc="center", ncol=5, frameon=False, fontsize=8.5, title="Oxygen history")

# (e) Mean trajectories in the hypoxia--arrest plane.
outer = fig.add_subplot(gs[2, :])
outer.axis("off")
sub = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[3, :], wspace=.28)
for j, line in enumerate(["WM793b", "WM164", "WM983b"]):
    ax = fig.add_subplot(sub[0, j])
    for prot in COL:
        z = s[(s.cell_line == line) & (s.protocol == prot)]
        if z.empty:
            continue
        if prot in START:
            base, days = START[prot]
            z = pd.concat([s[(s.cell_line == line) & (s.protocol == base) & s.day.isin(days)], z])
        z = z.sort_values("day")
        x, y = z.f_hypoxic_mean.to_numpy(), z.f_arrested_mean.to_numpy()
        ax.plot(x, y, "-o", color=COL[prot], ms=3, lw=1.15)
        for k in range(len(x) - 1):
            ax.annotate("", (x[k+1], y[k+1]), (x[k], y[k]), arrowprops={"arrowstyle": "->", "color": COL[prot], "lw": .8})
    title = ("(e) " if j == 0 else "") + line + (" (size-associated inhibition)" if line == "WM164" else "")
    ax.set(title=title, xlabel=r"$f_{\mathrm{hyp}}$", xlim=(-.02, .72), ylim=(-.02, .62))
    ax.title.set_fontweight("bold")
    if j == 0:
        ax.set_ylabel(r"$f_{\mathrm{inh}}$")

fig.savefig(ROOT / "figures" / "fig01_section2_model_free.png", dpi=300, bbox_inches="tight")
fig.savefig(ROOT / "figures" / "fig01_section2_model_free.pdf", bbox_inches="tight")
