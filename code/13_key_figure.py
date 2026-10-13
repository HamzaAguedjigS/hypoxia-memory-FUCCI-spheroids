"""Key figure for Article A (WM793b).

(a) Data: arrest lags oxygen (hysteresis in the hypoxia/arrest plane).
(b) Data vs radial model: inhibited radius after 2%->21% at day 2 for the
    finite-kinetics, instantaneous-arrest and no-arrest variants.
(c) Fitted finite-kinetics model: share of arrested cells in the shells that
    formed the inhibited region at the switch, versus the inhibited-radius
    readout. The readout vanishes while most of these cells are still arrested.
(d) Profile likelihood of the mean exit time (G1 re-estimated, two G1 laws).

Output: figures/fig09_key_figure_WM793b.png
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import radial_model as rm

ROOT = Path(__file__).resolve().parents[1]
LINE = "WM793b"
d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
d = d[d.cell_line == LINE]
summ = pd.read_csv(ROOT / "results" / "summary_by_condition.csv")
summ = summ[summ.cell_line == LINE]
# final model: fitted to all protocols after the out-of-sample check
pred = pd.read_csv(ROOT / "results" / f"radial_pred_{LINE}_alldata.csv")
fits = json.loads((ROOT / "results" / f"radial_fit_{LINE}_alldata.json").read_text())
prof = pd.read_csv(ROOT / "results" / f"exit_profile_{LINE}.csv")

fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

# ---------------- (a) hysteresis in the data
ax = axes[0, 0]
START = {"deox_d2": ("normoxia", [2]), "reox_d2": ("hypoxia", [2])}
STY = {"normoxia": ("#1f77b4", "21%"), "hypoxia": ("#ff7f0e", "2%"),
       "deox_d2": ("#2ca02c", "21%→2% (day 2)"), "reox_d2": ("#d62728", "2%→21% (day 2)")}
for prot, (col, lab) in STY.items():
    s = summ[summ.protocol == prot]
    if prot in START:
        b, days = START[prot]
        s = pd.concat([summ[(summ.protocol == b) & summ.day.isin(days)], s])
    s = s.sort_values("day")
    x, y = s.f_hypoxic_mean.values, s.f_arrested_mean.values
    ax.errorbar(x, y, xerr=s.f_hypoxic_se, yerr=s.f_arrested_se, color=col, marker="o",
                ms=4, capsize=2, label=lab)
    for k in range(len(x) - 1):
        ax.annotate("", xy=(x[k + 1], y[k + 1]), xytext=(x[k], y[k]),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.1))
    for xi, yi, dd in zip(x, y, s.day):
        ax.text(xi + 0.006, yi + 0.008, f"d{dd:g}", fontsize=6.5, color=col)
ax.set_xlabel("pimonidazole-positive fraction")
ax.set_ylabel("FUCCI-arrested fraction of living volume")
ax.set_title("(a) Arrest lags oxygen in both directions (data)", fontsize=10, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")

# ---------------- (b) inhibited radius after reoxygenation: data vs variants
ax = axes[0, 1]
s = pd.concat([d[(d.protocol == "hypoxia") & (d.day == 2)], d[d.protocol == "reox_d2"]])
rng = np.random.default_rng(0)
ax.plot(s.day + rng.normal(0, 0.05, len(s)), s.R_inhib, "o", ms=3, color="k", alpha=0.25)
g = s.groupby("day").R_inhib
ax.errorbar(g.mean().index, g.mean(), yerr=g.sem(), fmt="ko", ms=5, capsize=3, label="data")
VSTY = {"finQ": ("#d62728", "-", "finite arrest kinetics"),
        "instQ": ("#7f7f7f", ":", "instantaneous arrest"),
        "noQ": ("#2ca02c", "-.", "no arrest state")}
for v, (col, ls, lab) in VSTY.items():
    p = pred[(pred.variant == v) & (pred.protocol == "reox_d2") & (pred.day >= 2)]
    ax.plot(p.day, p.R_inhib, ls, color=col, lw=1.8,
            label=f"{lab} (ΔAIC {fits[v]['AIC'] - fits['finQ']['AIC']:+.0f})")
ax.axvline(2, color="grey", lw=0.7, ls="-.")
ax.text(2.05, 5, "2%→21%", fontsize=7.5, color="grey")
ax.set_xlabel("day")
ax.set_ylabel("inhibited radius R$_i$ (µm)")
ax.set_title("(b) Only finite arrest kinetics keeps R$_i$ at day 3", fontsize=10, loc="left")
ax.legend(fontsize=7.5, frameon=False)

# ---------------- (c) arrested cells vs readout in the fitted model
ax = axes[1, 0]
th = fits["finQ"]["params"]
rm.DT = 0.5
M = 40
mdl = rm.Model("finQ", M=M, protocols=["reox_d2"])
S = mdl.initial(th)
t_switch = (2 - rm.T0_DAY) * 24
times, q_share, ri, rn = [], [], [], []
inner = None
for k in range(int(round(7 * 24 / rm.DT)) + 1):
    t = k * rm.DT
    if t >= t_switch - 24:
        obs = mdl.observe(S, t, th)[0]
        R, E, G, Q, D, m = S
        living = (R + E + G + Q)[0]
        if inner is None and t >= t_switch:
            # Lagrangian shells inside the inhibited region at the switch
            r_in, r_out = rm.oxygen(living[None], np.ones((1, M)), D, np.array([th["a"]]))[3:5]
            inner = r_out[0] <= obs[1] + 1e-9
        if inner is not None and inner.any():
            q_share.append(Q[0][inner].sum() / living[inner].sum())
        else:
            q_share.append(np.nan)
        times.append(rm.T0_DAY + t / 24)
        ri.append(obs[1])
        rn.append(obs[2])
    if k < int(round(7 * 24 / rm.DT)):
        te = t + rm.DT / 2
        k1 = mdl.derivs(t, S, th, te)
        k2 = mdl.derivs(t + rm.DT / 2, [x + rm.DT / 2 * y for x, y in zip(S, k1)], th, te)
        k3 = mdl.derivs(t + rm.DT / 2, [x + rm.DT / 2 * y for x, y in zip(S, k2)], th, te)
        k4 = mdl.derivs(t + rm.DT, [x + rm.DT * y for x, y in zip(S, k3)], th, te)
        S = [np.clip(x + rm.DT / 6 * (a + 2 * b + 2 * c + e), 0, None) if i < 5
             else x + rm.DT / 6 * (a + 2 * b + 2 * c + e)
             for i, (x, a, b, c, e) in enumerate(zip(S, k1, k2, k3, k4))]
times, q_share, ri, rn = map(np.array, (times, q_share, ri, rn))
sel = times >= 2
ax.plot(times[sel], q_share[sel], color="#9467bd", lw=2,
        label="arrested share of the cells that formed\nthe inhibited region at the switch")
ax.set_ylim(0, 1.05)
ax.set_xlabel("day")
ax.set_ylabel("arrested share", color="#9467bd")
ax2 = ax.twinx()
ax2.plot(times[sel], ri[sel] - rn[sel], color="#d62728", lw=1.5, ls="--",
         label="model readout: R$_i$ − R$_n$")
gd = s.assign(w=s.R_inhib - s.R_nec).groupby("day").w
ax2.errorbar(gd.mean().index, gd.mean(), yerr=gd.sem(), fmt="o", color="#d62728", ms=4,
             capsize=2, label="data: R$_i$ − R$_n$")
ax2.set_ylabel("width of inhibited ring (µm)", color="#d62728")
ax2.set_ylim(bottom=0)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper right")
cross = times[(times > 2) & (ri - rn < 0.2 * (ri - rn)[np.argmin(np.abs(times - 2))])]
if cross.size:
    q_at = q_share[np.argmin(np.abs(times - cross[0]))]
    ax.axvline(cross[0], color="grey", lw=0.6, ls=":")
    ax.text(cross[0] + 0.05, 0.08, f"readout lost at day {cross[0]:.1f}\n"
                                   f"{100 * q_at:.0f}% still arrested", fontsize=7.5)
ax.set_xlim(1.9, 6.4)
ax.text(5.15, 0.42, "(after day 6, spheroid growth\nre-creates hypoxic arrest)", fontsize=6.5,
        color="grey")
ax.set_title("(c) The FUCCI readout clears before the arrest does", fontsize=10, loc="left")

# ---------------- (d) exit-time profile
ax = axes[1, 1]
for (lab, g2), col, lw, al in zip(prof.groupby("g1_law", sort=False), ["k", "#d62728", "#1f77b4"],
                                  [2.2, 1.2, 1.2], [1.0, 0.55, 0.55]):
    ax.plot(g2.exit_time_h / 24, np.minimum(g2.delta_nll, 70), "o-", color=col, lw=lw, alpha=al,
            ms=4, label=f"{lab} (G1 ≈ {g2.g1_fit_h.median():.0f} h)")
ax.axhline(1.92, color="grey", ls="--", lw=0.8)
ax.axvspan(0, 110 / 24, color="#d62728", alpha=0.07)
ax.axvspan(300 / 24, 400, color="#d62728", alpha=0.07)
ax.text(1.05, 36, "mean exit\n< 4.6 days\nrejected", fontsize=7.5, color="#d62728")
ax.text(14, 36, "> 12.5 days\nrejected", fontsize=7.5, color="#d62728")
ax.set_xscale("log")
ax.set_xlabel("mean exit time from hypoxic arrest (days)")
ax.set_ylabel("Δ negative log-likelihood (capped at 70)")
ax.set_title("(d) Exit from hypoxic arrest takes about a week", fontsize=10, loc="left")
ax.legend(fontsize=7.5, frameon=False, loc="upper right")

fig.suptitle("WM793b melanoma spheroids: hypoxia-induced arrest has memory", fontsize=12)
fig.tight_layout()
fig.savefig(ROOT / "figures" / f"fig09_key_figure_{LINE}.png", dpi=220)
print("saved; readout loss:", cross[0] if cross.size else None)
