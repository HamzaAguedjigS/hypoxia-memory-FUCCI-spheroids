"""Tier 4 (simulation, clearly synthetic): which sampling schedule resolves exit kinetics?

Ground truth = the delayed-exponential exit law fitted to WM793b in Tier 2
(delay D, then rate 1/T; cycling growth r). Synthetic spheroids are drawn
from a Beta distribution with the fitted mean and the dispersion estimated
from the real residuals. For each design we refit the delayed and the
memoryless (D = 0) laws and record
  * relative error of the mean exit time D + T,
  * how often AIC detects the delay,
  * the bias of the exit time when the memoryless law is assumed.
Total spheroid budget after the switch is kept close to the real one (41).

Outputs: results/design_exit.csv, figures/fig05_design_exit.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
SIGMA0 = 0.005
S_GRID = np.geomspace(0.02, 2.0, 160)[:, None]
N_REP = 300
rng = np.random.default_rng(7)

fits = pd.read_csv(ROOT / "results" / "switch_kinetics_fits.csv")
row = fits[(fits.direction == "exit") & (fits.growth == "r_hat") &
           (fits.model == "delayed exp.") & fits.nll.notna()].iloc[0]
D_TRUE, T_TRUE = row.delay_h / 24, row.rate_time_h / 24
d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
s = d[(d.cell_line == "WM793b") & (d.protocol == "normoxia")]
R_GROWTH = np.polyfit(s.day, np.log(s.V_living), 1)[0]
F0 = d[(d.cell_line == "WM793b") & (d.protocol == "hypoxia") & (d.day == 2)].f_arrested.mean()


def curve(tau, f0, D, T, r=R_GROWTH):
    sft = np.clip(tau - D, 0, None)
    A = f0 * np.exp(-sft / T)
    C = (1 - f0) * np.exp(r * tau) + f0 / (1 + r * T) * (np.exp(r * sft) - np.exp(-sft / T))
    return A / (A + C)


def loglik(fm, fo):
    fm = np.clip(fm, 1e-9, 1 - 1e-9)[None, :]
    var = S_GRID**2 * fm * (1 - fm) + SIGMA0**2
    return (-0.5 * np.sum(np.log(2 * np.pi * var) + (fo[None, :] - fm) ** 2 / var, 1)).max()


# dispersion of the real data around the Tier-2 curve (WM793b, switch at day 2)
real = pd.concat([
    d[(d.cell_line == "WM793b") & (d.protocol == "hypoxia") & (d.day == 2)].assign(tau=0.0),
    d[(d.cell_line == "WM793b") & (d.protocol == "reox_d2") & d.day.isin([3, 4])]
    .assign(tau=lambda x: x.day - 2)])
fm = curve(real.tau.values, F0, D_TRUE, T_TRUE)
S_HAT = np.sqrt(np.mean((real.f_arrested.values - fm) ** 2 / (fm * (1 - fm))))
PHI = 1 / S_HAT**2 - 1


ROW_EXP = fits[(fits.direction == "exit") & (fits.growth == "r_hat") &
               (fits.model == "Erlang k=1") & fits.nll.notna()].iloc[0]
T_NULL = ROW_EXP.mean_time_h / 24  # memoryless law fitted to the real data
TRUTH = {"delayed": (D_TRUE, T_TRUE), "memoryless": (0.0, T_NULL)}


def draw(tau, n, truth="delayed"):
    D, T = TRUTH[truth]
    m = np.clip(curve(tau, F0, D, T), 1e-4, 1 - 1e-4)
    return rng.beta(m * PHI, (1 - m) * PHI, size=n)


def fit(tau, f, delay):
    if delay:
        fun = lambda th: -loglik(curve(tau, 1 / (1 + np.exp(-th[2])), np.exp(th[0]),
                                       np.exp(th[1])), f)
        x0s = [np.array([np.log(0.3), np.log(0.8), -1.0]), np.array([np.log(0.05), np.log(1.2), -1.0])]
    else:
        fun = lambda th: -loglik(curve(tau, 1 / (1 + np.exp(-th[1])), 0.0, np.exp(th[0])), f)
        x0s = [np.array([np.log(1.0), -1.0])]
    best = min((minimize(fun, x0, method="Nelder-Mead",
                         options=dict(xatol=1e-6, fatol=1e-8, maxiter=3000)) for x0 in x0s),
               key=lambda z: z.fun)
    return best


DESIGNS = {  # label: (post-switch times in days, spheroids per time)
    "current (1, 2, 4 d)": ([1, 2, 4], [15, 13, 13]),
    "12-hourly to 2 d": ([0.5, 1, 1.5, 2], [10, 10, 10, 10]),
    "6-hourly to 1.5 d + 2 d": ([0.25, 0.5, 0.75, 1, 1.25, 1.5, 2], [6] * 7),
    "6-hourly to 2 d": ([0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2], [5] * 8),
}
N_PRE = 12
truth_mean = D_TRUE + T_TRUE
rows = []
for truth in TRUTH:
    for lab, (times, ns) in DESIGNS.items():
        for rep in range(N_REP):
            tau = np.r_[np.zeros(N_PRE), np.concatenate([np.full(n, t) for t, n in zip(times, ns)])]
            f = np.r_[draw(0.0, N_PRE, truth),
                      np.concatenate([draw(t, n, truth) for t, n in zip(times, ns)])]
            fd, fe = fit(tau, f, True), fit(tau, f, False)
            mean_d = np.exp(fd.x[0]) + np.exp(fd.x[1])
            rows.append(dict(truth=truth, design=lab, rep=rep, mean_delay_model_h=24 * mean_d,
                             delay_h=24 * np.exp(fd.x[0]), mean_exp_model_h=24 * np.exp(fe.x[0]),
                             dAIC=2 * (fe.fun - fd.fun) - 2))  # >0 favours the delay
allres = pd.DataFrame(rows)
allres.to_csv(ROOT / "results" / "design_exit.csv", index=False, float_format="%.5g")
null = allres[allres.truth == "memoryless"].groupby("design", sort=False).dAIC
false_pos = null.apply(lambda x: 100 * np.mean(x > 0))
res = allres[allres.truth == "delayed"]

summ = res.groupby("design", sort=False).apply(lambda g: pd.Series({
    "rel_RMSE_mean_time_%": 100 * np.sqrt(np.mean((g.mean_delay_model_h / 24 / truth_mean - 1) ** 2)),
    "delay_detected_%": 100 * np.mean(g.dAIC > 0),
    "delay_h_median": g.delay_h.median(),
    "delay_h_IQR": f"{g.delay_h.quantile(.25):.1f}-{g.delay_h.quantile(.75):.1f}",
    "bias_exp_model_%": 100 * (g.mean_exp_model_h.median() / 24 / truth_mean - 1),
}))
summ["false_delay_%_if_memoryless"] = false_pos
print(f"truth: delay {24 * D_TRUE:.1f} h, rate time {24 * T_TRUE:.1f} h, mean {24 * truth_mean:.1f} h;"
      f" dispersion s = {S_HAT:.3f}")
print(summ.to_string())
summ.to_csv(ROOT / "results" / "design_exit_summary.csv", float_format="%.4g")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
tt = np.linspace(0, 4, 300)
axes[0].plot(tt, curve(tt, F0, D_TRUE, T_TRUE), "k-", label="delayed exit (truth)")
axes[0].plot(tt, curve(tt, F0, 0, truth_mean), "k--", label="memoryless, same mean")
axes[0].plot(real.tau + rng.normal(0, 0.03, len(real)), real.f_arrested, "o", ms=3,
             color="#d62728", alpha=0.5, label="WM793b data")
axes[0].set_xlabel("time since reoxygenation (days)")
axes[0].set_ylabel("arrested fraction")
axes[0].legend(fontsize=7, frameon=False)
labs = list(DESIGNS)
axes[1].boxplot([res[res.design == l].mean_delay_model_h for l in labs], labels=range(1, 5))
axes[1].axhline(24 * truth_mean, color="k", ls=":")
axes[1].set_ylabel("estimated mean exit time (h)")
axes[1].set_xlabel("design")
axes[2].bar(np.arange(1, 5) - 0.18, summ["delay_detected_%"], width=0.36, color="#1f77b4",
            label="truth delayed (power)")
axes[2].bar(np.arange(1, 5) + 0.18, summ["false_delay_%_if_memoryless"], width=0.36,
            color="#bbbbbb", label="truth memoryless (false alarm)")
axes[2].set_ylabel("delay selected by AIC (%)")
axes[2].legend(fontsize=7, frameon=False)
axes[2].set_xlabel("design")
for ax in axes[1:]:
    ax.set_xticks(range(1, 5))
fig.text(0.5, -0.02, " | ".join(f"{i + 1}: {l}" for i, l in enumerate(labs)), ha="center", fontsize=8)
fig.suptitle("Synthetic design study (not data): resolving the exit kinetics of arrested cells",
             fontsize=10)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig05_design_exit.png", dpi=200, bbox_inches="tight")
