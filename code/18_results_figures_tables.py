"""Publication figures and tables for the radial-model results.

Creates vector Figures 3--5 and LaTeX table fragments using the final
all-protocol WM793b fit. Visual conventions follow Figure 1.
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import radial_model as rm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
RESULTS = ROOT / "results"
LINE = "WM793b"

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
d = d[d.cell_line == LINE]
pred = pd.read_csv(RESULTS / "radial_pred_WM793b_alldata.csv")
fits = json.loads((RESULTS / "radial_fit_WM793b_alldata.json").read_text())

COL = {"finQ": "#d62728", "delayQ": "#2878b5", "instQ": "#777777", "noQ": "#2ca02c"}
LS = {"finQ": "-", "delayQ": "--", "instQ": ":", "noQ": "-."}
NAME = {"finQ": r"finite arrest, $\mathcal{M}_F$", "delayQ": r"delayed entry, $\mathcal{M}_D$",
        "instQ": r"instantaneous arrest, $\mathcal{M}_I$", "noQ": r"cycling only, $\mathcal{M}_0$"}


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "figures" / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
protocols = [p for p in ("normoxia", "hypoxia", "deox_d2", "reox_d2", "reox_d4")
             if (d.protocol == p).any()]
start = {"deox_d2": ("normoxia", [2]), "reox_d2": ("hypoxia", [2]),
         "reox_d4": ("hypoxia", [2, 4])}
switch = {"deox_d2": 2, "reox_d2": 2, "reox_d4": 4}
titles = {"normoxia": "(a) 21% O$_2$", "hypoxia": "(b) 2% O$_2$",
          "deox_d2": "(c) 21→2% at day 2", "reox_d2": "(d) 2→21% at day 2",
          "reox_d4": "(e) 2→21% at day 4"}
observables = [("R_outer", r"$R_{\mathrm{out}}$ (µm)"),
               ("R_inhib", r"$R_{\mathrm{inh}}$ (µm)"),
               ("R_nec", r"$R_{\mathrm{nec}}$ (µm)"),
               ("R_pim", r"$R_{\mathrm{pim}}$ (µm)")]

fig, axes = plt.subplots(4, len(protocols), figsize=(7.2, 7.6), sharex=True, sharey="row")
for j, protocol in enumerate(protocols):
    obs = d[d.protocol == protocol]
    if protocol in start:
        base, days = start[protocol]
        obs = pd.concat([d[(d.protocol == base) & d.day.isin(days)], obs])
    grouped = obs.groupby("day")
    for i, (variable, ylabel) in enumerate(observables):
        ax = axes[i, j]
        ax.errorbar(grouped[variable].mean().index, grouped[variable].mean(),
                    yerr=grouped[variable].sem(), fmt="o", color="black", ms=2.8,
                    capsize=1.8, elinewidth=.8, label="data: mean ± SE", zorder=8)
        for variant in ("finQ", "delayQ", "instQ", "noQ"):
            curve = pred[(pred.variant == variant) & (pred.protocol == protocol) & (pred.day >= 2)]
            ax.plot(curve.day, curve[variable], color=COL[variant], ls=LS[variant], lw=1.25,
                    label=NAME[variant])
        if protocol in switch:
            ax.axvline(switch[protocol], color="#555555", lw=.6, ls=(0, (3, 2)))
        ax.grid(axis="y", color="#dddddd", lw=.45)
        ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            ax.set_title(titles[protocol], pad=5)
        if j == 0:
            ax.set_ylabel(ylabel)
        if i == 3:
            ax.set_xlabel("day after seeding")
            ax.set_xticks([2, 4, 6, 8])
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(.5, 1.015))
fig.subplots_adjust(top=.93, left=.10, right=.99, bottom=.07, hspace=.13, wspace=.10)
save(fig, "fig03_model_fits")


# ---------------------------------------------------------------- Figure 4
# (a) all-protocol profile: grid of 11_exit_g1_confounding.py plus the baseline grid of
# 20_exit_profile_sensitivity.py; reference = best NLL found (profile or final fit).
# The threshold observables make the objective non-smooth, so no chi-square line is drawn:
# points within the optimisation-noise band are "supported", points above REJECT "rejected".
NOISE, REJECT = 20.0, 40.0
profile = pd.read_csv(RESULTS / "exit_profile_WM793b.csv")
profile = profile[profile.g1_law == "all protocols (final fit)"][["exit_time_h", "nll"]]
extra_file = RESULTS / "exit_profile_sensitivity_WM793b.csv"
if extra_file.exists():
    extra = pd.read_csv(extra_file)
    extra = extra[(extra.setting == "baseline") & (extra.variant == "finQ") & (extra.stage == "fixed_exit")]
    profile = pd.concat([profile, extra.rename(columns={"exit_fixed_h": "exit_time_h"})[["exit_time_h", "nll"]]])
profile = profile.groupby("exit_time_h", as_index=False).nll.min().sort_values("exit_time_h")
profile_ref = min(profile.nll.min(), fits["finQ"]["nll_train"])
profile["delta_nll"] = profile.nll - profile_ref
profile.to_csv(RESULTS / "exit_profile_WM793b_final.csv", index=False, float_format="%.6g")
supported = profile[profile.delta_nll <= NOISE].exit_time_h / 24
validation_file = RESULTS / "validation_profile_WM793b.csv"
has_validation = validation_file.exists()

fig, axes4 = plt.subplots(1, 2 if has_validation else 1, figsize=(7.2, 3.3), sharey=True, squeeze=False)
ax = axes4[0, 0]
x = profile.exit_time_h / 24
ax.axhspan(0, NOISE, color="#2ca02c", alpha=.08, lw=0)
ax.axhline(REJECT, color="#a51f1f", lw=.8, ls="--")
ax.plot(x, np.minimum(profile.delta_nll, 80), "o-", color="#2878b5", lw=1.7, ms=4.5)
ax.text(1.62, NOISE - 1.5, "optimisation-noise band", va="top", fontsize=7.5, color="#20752a")
ax.text(1.62, REJECT + 1.5, "rejection level", va="bottom", fontsize=7.5, color="#a51f1f")
ax.text(14.5, 6, f"supported:\n{supported.min():.1f}–{supported.max():.1f} d", fontsize=8,
        color="#20752a", va="bottom")
ax.set_xscale("log")
ax.set_xticks([2, 4, 7, 14, 28])
ax.set_xticklabels(["2", "4", "7", "14", "28"])
ax.set_xlim(1.5, 40)
ax.set_ylim(-2, 82)
ax.set_xlabel("mean exit time from arrest (days)")
ax.set_ylabel(r"$\Delta$ negative log-likelihood")
ax.set_title("(a) all protocols", loc="left", fontweight="bold")
if has_validation:
    val = pd.read_csv(validation_file).sort_values("exit_time_h")
    ax = axes4[0, 1]
    ax.axhspan(0, NOISE, color="#2ca02c", alpha=.08, lw=0)
    ax.axhline(REJECT, color="#a51f1f", lw=.8, ls="--")
    ax.plot(val.exit_time_h / 24, np.minimum(val.delta_train, 80), "o-", color="#2878b5", lw=1.6, ms=4,
            label="training protocols")
    ax.plot(val.exit_time_h / 24, np.minimum(val.delta_test, 80), "s--", color="#d62728", lw=1.6, ms=4,
            label="held-out 2→21% at day 4")
    ax.set_xscale("log")
    ax.set_xticks([2, 4, 7, 14, 28, 56])
    ax.set_xticklabels(["2", "4", "7", "14", "28", "56"])
    ax.set_xlim(1.5, 70)
    ax.set_xlabel("mean exit time from arrest (days)")
    ax.set_title("(b) day-4 reoxygenation held out", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper center", fontsize=7.2)
for ax in axes4.ravel():
    ax.grid(axis="y", color="#dddddd", lw=.5)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "fig04_exit_profile")


# ---------------------------------------------------------------- Figure 5
th = fits["finQ"]["params"]
rm.DT = .5
M = 40
mdl = rm.Model("finQ", M=M, protocols=["reox_d2"])
S = mdl.initial(th)
switch_h = 24.0
times, fq_switch, fq_total, finh, inhibited_width = [], [], [], [], []
inner = None
for step in range(int(round(5.5 * 24 / rm.DT)) + 1):
    t = step * rm.DT
    obs = mdl.observe(S, t, th)[0]
    R, E, G, Q, D, m = S
    living = (R + E + G + Q)[0]
    if inner is None and t >= switch_h:
        # Lagrangian shells forming the inhibited region at reoxygenation.
        pin = mdl.p_inf(t - 1e-6)
        p, r_in, r_out, _ = rm.local_p(living[None], np.ones((1, M)), D,
                                        np.array([th["a"] * m[0]]), pin)
        inner = r_out[0] <= obs[1] + 1e-9
    times.append(rm.T0_DAY + t / 24)
    fq_total.append(obs[4])
    denom = max(obs[0] ** 3 - obs[2] ** 3, 1e-12)
    finh.append(max(obs[1] ** 3 - obs[2] ** 3, 0) / denom)
    inhibited_width.append(max(obs[1] - obs[2], 0))
    fq_switch.append((Q[0][inner].sum() / living[inner].sum()) if inner is not None and inner.any()
                     else np.nan)
    if step == int(round(5.5 * 24 / rm.DT)):
        break
    te = t + rm.DT / 2
    k1 = mdl.derivs(t, S, th, te)
    k2v = mdl.derivs(t + rm.DT / 2, [x + rm.DT / 2 * z for x, z in zip(S, k1)], th, te)
    k3 = mdl.derivs(t + rm.DT / 2, [x + rm.DT / 2 * z for x, z in zip(S, k2v)], th, te)
    k4 = mdl.derivs(t + rm.DT, [x + rm.DT * z for x, z in zip(S, k3)], th, te)
    S = [np.clip(x + rm.DT / 6 * (a + 2*b + 2*c + e), 0, None) if i < 5
         else x + rm.DT / 6 * (a + 2*b + 2*c + e)
         for i, (x, a, b, c, e) in enumerate(zip(S, k1, k2v, k3, k4))]

times, fq_switch, fq_total, finh, inhibited_width = map(
    np.asarray, (times, fq_switch, fq_total, finh, inhibited_width))
at_switch = np.argmin(np.abs(times - 2))
threshold = .2 * inhibited_width[at_switch]
eligible = np.where((times > 2) & (inhibited_width <= threshold))[0]
lost = eligible[0]

fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(times, fq_switch, color="#9467bd", lw=2.1,
        label=r"arrested fraction in switch-region shells, $f_Q^{\mathrm{sw}}$")
ax.plot(times, fq_total, color="#2878b5", lw=1.4, ls="--",
        label=r"whole-spheroid arrested fraction, $f_Q$")
ax.plot(times, finh, color="#d62728", lw=1.8,
        label=r"image-derived inhibited fraction, $f_{\mathrm{inh}}$")

obs = pd.concat([d[(d.protocol == "hypoxia") & (d.day == 2)], d[d.protocol == "reox_d2"]]).copy()
obs["f_inh"] = np.maximum(obs.R_inhib**3 - obs.R_nec**3, 0) / np.maximum(
    obs.R_outer**3 - obs.R_nec**3, 1e-12)
g = obs.groupby("day").f_inh
ax.errorbar(g.mean().index, g.mean(), yerr=g.sem(), fmt="o", ms=4, capsize=2,
            color="#d62728", mfc="white", label=r"observed $f_{\mathrm{inh}}$")
ax.axvline(times[lost], color="#555555", lw=.8, ls=":")
ax.scatter([times[lost]], [fq_switch[lost]], s=28, color="#9467bd", zorder=7)
ax.annotate(f"readout cleared at day {times[lost]:.1f}\n{100*fq_switch[lost]:.0f}% still arrested",
            xy=(times[lost], fq_switch[lost]), xytext=(3.55, .27),
            arrowprops=dict(arrowstyle="->", lw=.8, color="#555555"), fontsize=8.2)
ax.axvline(2, color="#777777", lw=.7, ls=(0, (3, 2)))
ax.text(2.05, .05, "2→21% O$_2$", color="#666666", fontsize=8)
ax.set_xlim(1.9, 6.35)
ax.set_ylim(0, 1.03)
ax.set_xlabel("day after seeding")
ax.set_ylabel("fraction of living volume")
ax.set_title("The FUCCI readout clears before cellular arrest resolves", loc="left", fontweight="bold")
ax.grid(axis="y", color="#dddddd", lw=.5)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
save(fig, "fig05_readout_vs_arrest")


# ---------------------------------------------------------------- Tables
order = ("finQ", "delayQ", "instQ", "noQ")
model_symbol = {"finQ": "F", "delayQ": "D", "instQ": "I", "noQ": "0"}
best_aic = min(fits[v]["AIC"] for v in order)
model_lines = [
    r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
    r"Model & NLL & AIC & $\Delta$AIC & $\sigma_{\mathrm{out}}$ & $\sigma_{\mathrm{inh}}$ & $\sigma_{\mathrm{nec}}$ & $\sigma_{\mathrm{pim}}$ \\",
    r"\midrule",
]
model_rows = []
for v in order:
    x = fits[v]
    model_rows.append(dict(model=v, nll=x["nll_train"], AIC=x["AIC"],
                           delta_AIC=x["AIC"]-best_aic, sd_R_out=x["sd"]["R_outer"],
                           sd_R_inh=x["sd"]["R_inhib"], sd_R_nec=x["sd"]["R_nec"],
                           sd_R_pim=x["sd"]["R_pim"]))
    model_lines.append(
        f"$\\mathcal{{M}}_{model_symbol[v]}$ & {x['nll_train']:.1f} & {x['AIC']:.1f} & {x['AIC']-best_aic:.1f} & "
        f"{x['sd']['R_outer']:.1f} & {x['sd']['R_inhib']:.1f} & "
        f"{x['sd']['R_nec']:.1f} & {x['sd']['R_pim']:.1f}" + " \\\\")
model_lines += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "table_model_comparison.tex").write_text("\n".join(model_lines))
pd.DataFrame(model_rows).to_csv(RESULTS / "model_comparison.csv", index=False,
                                float_format="%.6g")

# Import bounds without running a fit.
spec = importlib.util.spec_from_file_location("f7_tables", ROOT / "code" / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
old_argv = sys.argv
sys.argv = [sys.argv[0], "WM793b", "1", "finQ"]
spec.loader.exec_module(f7)
sys.argv = old_argv
mapping = {
    "k1max": (r"$k_{1,\max}$", "maximal G1 progression rate"),
    "K1": (r"$K_1$", "G1 oxygen half-saturation"), "a": (r"$a$", "consumption/diffusivity ratio"),
    "amax": (r"$\alpha_{\max}$", "maximal arrest-entry rate"),
    "bmax": (r"$\beta_{\max}$", "maximal arrest-exit rate"),
    "Kq": (r"$K_q$", "arrest oxygen threshold"), "hq": (r"$h_q$", "Hill exponent"),
    "mu0": (r"$\mu_0$", "basal loss rate"), "mun": (r"$\mu_{n,\max}$", "maximal anoxic-death rate"),
    "Kn": (r"$K_n$", "anoxic-death threshold"), "kD": (r"$k_{D,h}$", r"core clearance at 2\% O$_2$"),
    "kD_ox": (r"$k_{D,o}$", r"core clearance at 21\% O$_2$"),
    "m_h": (r"$m_h$", "hypoxic consumption fraction"), "tau_m": (r"$\tau_m$", "adaptation time"),
    "R1_norm": (r"$\rho_{\mathrm{norm}}$", "initial normoxic radius"),
    "R1_hyp": (r"$\rho_{\mathrm{hyp}}$", "initial hypoxic radius"),
    "p_pim": (r"$p_{\mathrm{pim}}$", "effective PIM threshold"),
}
params = fits["finQ"]["params"]
flags = set(fits["finQ"]["at_bound"])
parameter_lines = [r"\begin{tabular}{@{}llrrl@{}}", r"\toprule",
                   r"Parameter & Interpretation & Estimate & Search interval & Status \\", r"\midrule"]
parameter_rows = []
for name in fits["finQ"]["free"]:
    lo, hi = (10**f7.LOG_BOUNDS[name][0], 10**f7.LOG_BOUNDS[name][1])
    estimate = params[name]
    if name in flags:
        status = "lower bound" if abs(np.log10(estimate)-np.log10(lo)) < abs(np.log10(hi)-np.log10(estimate)) else "upper bound"
    else:
        status = "interior"
    symbol, meaning = mapping[name]
    parameter_rows.append(dict(parameter=name, interpretation=meaning, estimate=estimate,
                               lower_bound=lo, upper_bound=hi, status=status))
    parameter_lines.append(f"{symbol} & {meaning} & {estimate:.4g} & [{lo:.4g}, {hi:.4g}] & {status}" + " \\\\")
parameter_lines += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "table_finQ_parameters.tex").write_text("\n".join(parameter_lines))
pd.DataFrame(parameter_rows).to_csv(RESULTS / "finQ_parameters.csv", index=False,
                                    float_format="%.6g")

print(f"saved Figures 3--5; readout clearance={times[lost]:.3f} d, "
      f"arrested share={100*fq_switch[lost]:.1f}%")
