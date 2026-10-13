"""Results of the initial-condition check (35_initial_condition.py) next to v8.

Compares v8 (no arrested cells at day 1) with the fit in which a fraction q0_hyp of the G1 cells
in low-oxygen shells of the 2% group starts arrested: likelihood gain, AIC/BIC ranking, q0 profile,
arrest-entry and exit profiles, day-2 hypoxic fit, early reoxygenation and readout-versus-arrest.

Writes results/v8_initial/{summary.txt, initial_condition.csv, profiles_initial.csv,
table_initial_condition.tex} and figures/figS_initial_condition.{pdf,png}. The article folder is
not touched.

Usage: python 36_initial_results.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, NullFormatter
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [str(HERE / "35_initial_condition.py")]
spec = importlib.util.spec_from_file_location("i35", HERE / "35_initial_condition.py")
i35 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(i35)
sys.argv = _argv
m = i35.CTX["all"]
rm = m.rm
ROOT = i35.p28.ROOT
OUT, FIG, V8 = ROOT / "results" / "v8_initial", ROOT / "figures", ROOT / "results" / "v8"
SUPPORT, CHI2, REJECT = 10.0, 1.92, 40.0
N_OBS = 876
RED, GREY = "#c0392b", "#9a9a9a"
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


def crossing(xs, ys, x0, side):
    """Supported-region edge beyond the outermost supported grid point (linear in log10 x)."""
    lx, l0 = np.log10(xs), np.log10(x0)
    idx = np.where(np.sign(lx - l0) == side)[0]
    idx = idx[np.argsort(np.abs(lx[idx] - l0))]
    sup = [k for k, i in enumerate(idx) if ys[i] <= SUPPORT]
    start = sup[-1] + 1 if sup else 0
    if start >= len(idx):
        return None
    px, py = (lx[idx[start - 1]], ys[idx[start - 1]]) if start else (l0, 0.0)
    i = idx[start]
    return 10 ** (px + (SUPPORT - py) / (ys[i] - py) * (lx[i] - px))


def profile(recs, fam):
    pts = {}
    for r in recs:
        if r.get("family") == fam and r["kind"] == "fix":
            if r["value"] not in pts or r["nll"] < pts[r["value"]]["nll"]:
                pts[r["value"]] = r
    v = np.array(sorted(pts), float)
    return v, np.array([pts[x]["nll"] for x in sorted(pts)])


def merged(vals, nlls, x_best, nll_best):
    """Profile points plus the free optimum, sorted; duplicates keep the lower NLL."""
    out = {}
    for x, n in list(zip(vals, nlls)) + [(x_best, nll_best)]:
        key = round(float(np.log10(x)), 6)
        if key not in out or n < out[key][1]:
            out[key] = (float(x), float(n))
    xs = np.array([out[k][0] for k in sorted(out)])
    return xs, np.array([out[k][1] for k in sorted(out)]) - nll_best


def readout_vs_arrest(th, M=64):
    rm.DT = 0.5
    mdl = rm.Model("finQ", M=M, protocols=["reox_d2"])
    S = mdl.initial(th)
    times, fq_sw, width = [], [], []
    inner = None
    for step in range(int(round(5.5 * 24 / rm.DT)) + 1):
        t = step * rm.DT
        obs = mdl.observe(S, t, th)[0]
        R, E, G, Q, D, mm = S
        living = (R + E + G + Q)[0]
        if inner is None and t >= 24.0:
            pin = mdl.p_inf(t - 1e-6)
            _, _, r_out, _ = rm.local_p(living[None], np.ones((1, M)), D, np.array([th["a"] * mm[0]]), pin)
            inner = r_out[0] <= obs[1] + 1e-9
        times.append(rm.T0_DAY + t / 24)
        width.append(max(obs[1] - obs[2], 0))
        fq_sw.append(Q[0][inner].sum() / living[inner].sum() if inner is not None and inner.any() else np.nan)
        te = t + rm.DT / 2
        k1 = mdl.derivs(t, S, th, te)
        k2 = mdl.derivs(t + rm.DT / 2, [x + rm.DT / 2 * z for x, z in zip(S, k1)], th, te)
        k3 = mdl.derivs(t + rm.DT / 2, [x + rm.DT / 2 * z for x, z in zip(S, k2)], th, te)
        k4 = mdl.derivs(t + rm.DT, [x + rm.DT * z for x, z in zip(S, k3)], th, te)
        S = [np.clip(x + rm.DT / 6 * (a + 2 * b + 2 * c + e), 0, None) if i < 5 else
             x + rm.DT / 6 * (a + 2 * b + 2 * c + e) for i, (x, a, b, c, e) in enumerate(zip(S, k1, k2, k3, k4))]
    t, w, f = np.array(times), np.array(width), np.array(fq_sw)
    i_sw = int(np.argmin(np.abs(t - 2)))
    lost = int(np.where((t > 2) & (w <= 0.2 * w[i_sw]))[0][0])
    return t[lost], 100 * f[lost]


def windowed(th, protocols, days):
    rm.DT = 0.5
    mdl = rm.Model("finQ", M=64, protocols=protocols)
    return m.run_windowed(mdl, th, (np.asarray(days) - rm.T0_DAY) * 24)


# ------------------------------------------------------------------ fits
recs = i35.records()
fin8 = json.loads((V8 / "final_fits.json").read_text())["all"]
best = {v: min((r for r in recs if r["variant"] == v), key=lambda r: r["nll"]) for v in ("finQ", "delayQ")}
NLL = {"finQ": best["finQ"]["nll"], "delayQ": best["delayQ"]["nll"],
       "instQ": fin8["instQ"]["nll_train"], "noQ": fin8["noQ"]["nll_train"]}
K = {"finQ": fin8["finQ"]["n_params"] + 1, "delayQ": fin8["delayQ"]["n_params"] + 1,
     "instQ": fin8["instQ"]["n_params"], "noQ": fin8["noQ"]["n_params"]}
AIC = {v: 2 * NLL[v] + 2 * K[v] for v in NLL}
BIC = {v: 2 * NLL[v] + K[v] * np.log(N_OBS) for v in NLL}
TH = m.to_theta(np.array(best["finQ"]["x"]), "finQ")
TH8 = dict(fin8["finQ"]["params"])
gain = fin8["finQ"]["nll_train"] - NLL["finQ"]

say("Initial-condition check: fraction q0 of G1 cells arrested at day 1 in low-oxygen shells (2% group)")
say(f"records: {len(recs)}")
say(f"M_F: NLL v8 {fin8['finQ']['nll_train']:.2f} -> {NLL['finQ']:.2f} (gain {gain:.2f} for one parameter; "
    f"dAIC vs v8 M_F {2 * (NLL['finQ'] - fin8['finQ']['nll_train']) + 2:.1f})")
say(f"M_D: NLL v8 {fin8['delayQ']['nll_train']:.2f} -> {NLL['delayQ']:.2f}")
say("ranking with q0 in M_F and M_D (M_I, M_0 unchanged): " + ", ".join(
    f"{v} dAIC {AIC[v] - AIC['finQ']:.1f} / dBIC {BIC[v] - BIC['finQ']:.1f}" for v in ("delayQ", "instQ", "noQ")))
say(f"q0_hyp = {TH['q0_hyp']:.3f}")

rows = []
qv, qn = profile(recs, "q0")
q_all, qd = merged(qv, qn, TH["q0_hyp"], NLL["finQ"])
q_lo, q_hi = crossing(q_all, qd, TH["q0_hyp"], -1), crossing(q_all, qd, TH["q0_hyp"], 1)
say("q0 profile: " + "  ".join(f"{a:g}:{b:.1f}" for a, b in zip(q_all, qd))
    + f"  -> supported [{q_lo if q_lo else 'bound'}, {q_hi if q_hi else 'bound'}]")

ev, en = profile(recs, "entry")
e_best = 1 / TH["amax"]
e_all, ed = merged(ev, en, e_best, NLL["finQ"])
e_hi = crossing(e_all, ed, e_best, 1)
e_txt = f"supported up to {e_hi:.2f} h" if e_hi else "not bounded above on the tested grid"
say("entry-time profile (h): " + "  ".join(f"{a:g}:{b:.1f}" for a, b in zip(e_all, ed))
    + f"  -> best {e_best:.2f} h, {e_txt}")

xv, xn = profile(recs, "exit")
x_best = 1 / TH["bmax"]
x_all, xd = merged(xv, xn, x_best, NLL["finQ"])
x_lo, x_hi = crossing(x_all, xd, x_best, -1), crossing(x_all, xd, x_best, 1)
d4 = float(xd[np.argmin(np.abs(x_all - 96))])
d17 = float(xd[np.argmin(np.abs(x_all - 400))])
lo_txt = f"{x_lo / 24:.2f}" if x_lo else "open"
hi_txt = f"{x_hi / 24:.2f}" if x_hi else "open"
say("exit profile (h): " + "  ".join(f"{a:.0f}:{b:.1f}" for a, b in zip(x_all, xd))
    + f"  -> best {x_best / 24:.2f} d, supported {lo_txt}-{hi_txt} d; dNLL 4 d {d4:.1f}, 17 d {d17:.1f}")
say(f"G1 {1 / TH['k1max']:.1f} h (v8 {1 / TH8['k1max']:.1f}); Kq {TH['Kq']:.3f} (v8 {TH8['Kq']:.3f}); "
    f"m_h {TH['m_h']:.3f}; tau_m {TH['tau_m']:.1f} h; p_pim {TH['p_pim']:.3f}; hq {TH['hq']:.2f}; "
    f"sd {np.round(best['finQ']['sds'], 1).tolist()} (v8 {np.round(list(fin8['finQ']['sd'].values()), 1).tolist()})")

# day-2 hypoxic fit and early reoxygenation
d793 = m.d
obs = d793.groupby(["protocol", "day"])[m.OBS].mean()
p_new = windowed(TH, ["hypoxia", "reox_d2"], [2.0, 3.0, 4.0])
p_old = windowed(TH8, ["hypoxia", "reox_d2"], [2.0, 3.0, 4.0])
for name, i, j, key in (("2%, day 2", 0, 0, ("hypoxia", 2)), ("reox d2, day 3", 1, 1, ("reox_d2", 3)),
                        ("reox d2, day 4", 1, 2, ("reox_d2", 4))):
    o_ = obs.loc[key].values
    say(f"{name:15s} observed R_out/R_inh/R_nec/R_pim {np.round(o_, 1).tolist()} | v8 "
        f"{np.round(p_old[i, j, :4], 1).tolist()} | with q0 {np.round(p_new[i, j, :4], 1).tolist()}")
c_new, a_new = readout_vs_arrest(TH)
c_old, a_old = readout_vs_arrest(TH8)
say(f"readout clearance: v8 day {c_old:.2f} with {a_old:.1f}% arrested; with q0 day {c_new:.2f} with {a_new:.1f}%")
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf8")

prof = pd.concat([pd.DataFrame(dict(family="q0", value=q_all, dNLL=qd)),
                  pd.DataFrame(dict(family="entry_h", value=e_all, dNLL=ed)),
                  pd.DataFrame(dict(family="exit_h", value=x_all, dNLL=xd))])
prof.to_csv(OUT / "profiles_initial.csv", index=False, float_format="%.6g")
pd.DataFrame([dict(model=v, NLL=NLL[v], k=K[v], dAIC=AIC[v] - AIC["finQ"], dBIC=BIC[v] - BIC["finQ"])
              for v in NLL]).to_csv(OUT / "initial_condition.csv", index=False, float_format="%.6g")

fmt = lambda v, d=1: "--" if v is None or not np.isfinite(v) else f"{v:.{d}f}"
tex = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
       r" & & \multicolumn{3}{c}{$\Delta\mathrm{AIC}$} & Entry & \multicolumn{2}{c}{Exit (d)} & G1 \\",
       r"\cmidrule(lr){3-5}\cmidrule(lr){7-8}",
       r"Initial condition & NLL & $\mathcal{M}_D$ & $\mathcal{M}_I$ & $\mathcal{M}_0$ & (h) & best & supported & (h) \\",
       r"\midrule"]
aic8 = {v: fin8[v]["AIC"] for v in fin8}
x8 = pd.read_csv(V8 / "exit_profile_all.csv")
tex.append(f"No arrest at day 1 (primary) & {fin8['finQ']['nll_train']:.1f} & {aic8['delayQ'] - aic8['finQ']:.1f} & "
           f"{aic8['instQ'] - aic8['finQ']:.1f} & {aic8['noQ'] - aic8['finQ']:.1f} & "
           f"{1 / TH8['amax']:.1f} & {1 / TH8['bmax'] / 24:.2f} & 3.4--8.6 & {1 / TH8['k1max']:.1f} \\\\")
tex.append(f"Initial arrest, $q_0={TH['q0_hyp']:.2f}$ & {NLL['finQ']:.1f} & {AIC['delayQ'] - AIC['finQ']:.1f} & "
           f"{AIC['instQ'] - AIC['finQ']:.1f} & {AIC['noQ'] - AIC['finQ']:.1f} & {e_best:.1f} & {x_best / 24:.2f} & "
           f"{fmt(x_lo / 24 if x_lo else None)}--{fmt(x_hi / 24 if x_hi else None)} & {1 / TH['k1max']:.1f} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_initial_condition.tex").write_text("\n".join(tex), encoding="utf8")

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.2, "xtick.labelsize": 7.3,
                     "ytick.labelsize": 7.3, "legend.fontsize": 7.0, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "legend.frameon": False})
fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5), gridspec_kw=dict(wspace=.55))


def frame(ax, xlabel, ticks):
    ax.axhspan(-SUPPORT, .6, color="#f3dcd9", lw=0, zorder=0)
    ax.axhline(-CHI2, color="#555555", lw=.6, ls=(0, (1, 1.5)))
    ax.axhline(-REJECT, color="#555555", lw=.7, ls=(0, (4, 2)))
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_ylim(-48, 3)
    ax.set_yticks([0, -10, -20, -30, -40])
    ax.set_xlabel(xlabel)


def curve(ax, x, y, color, lw=1.8, pts=True):
    ok = y >= -48
    f = np.linspace(np.log10(x.min()), np.log10(x.max()), 300)
    ax.plot(10 ** f, PchipInterpolator(np.log10(x), y)(f), color=color, lw=lw)
    if pts:
        ax.scatter(x[ok], y[ok], s=12, facecolor="white", edgecolor=color, lw=1, zorder=4)
        ax.scatter(x[~ok], np.full((~ok).sum(), -46.8), marker="v", s=12, color=color, zorder=4, clip_on=False)


ax = axes[0]
frame(ax, "initial arrested fraction $q_0$", [0.001, 0.01, 0.1, 1])
curve(ax, q_all, -qd, "#5b2c83")
ax.set_ylabel(r"$-\Delta$NLL")
ax.set_title("(a) initial arrest", loc="left", fontweight="bold", color="#5b2c83", pad=3)
ax = axes[1]
frame(ax, "arrest-entry time (h)", [1, 2, 4, 8, 16])
a8 = pd.read_csv(V8 / "profile_amax.csv")
a8x, a8d = merged(1 / a8.value.values, a8.nll.values, 1 / TH8["amax"], fin8["finQ"]["nll_train"])
curve(ax, a8x, -a8d, GREY, lw=1.2, pts=False)
curve(ax, e_all, -ed, "#e67e22")
ax.set_xlim(0.8, 20)
ax.set_title("(b) entry time", loc="left", fontweight="bold", color="#e67e22", pad=3)
ax = axes[2]
frame(ax, "mean exit time (d)", [2, 4, 7, 14])
curve(ax, x8.exit_h.values / 24, -x8.dNLL.values, GREY, lw=1.2, pts=False)
curve(ax, x_all / 24, -xd, RED)
ax.set_xlim(1.9, 20)
ax.set_title("(c) exit time", loc="left", fontweight="bold", color=RED, pad=3)
ax = axes[3]
t_days = np.arange(2.0, 8.01, 0.125)
for th, col, lw in ((TH8, GREY, 1.2), (TH, RED, 1.8)):
    o_ = windowed(th, ["hypoxia", "reox_d2"], t_days)
    ax.plot(t_days, o_[0, :, 1], color=col, lw=lw, ls=(0, (3, 1.5)))
    ax.plot(t_days, o_[1, :, 1], color=col, lw=lw)
for prot, mk in (("hypoxia", "s"), ("reox_d2", "o")):
    g = d793[d793.protocol == prot].groupby("day").R_inhib.agg(["mean", "sem"])
    if prot == "reox_d2":
        g = pd.concat([d793[(d793.protocol == "hypoxia") & (d793.day == 2)].groupby("day").R_inhib.agg(["mean", "sem"]), g])
    ax.errorbar(g.index, g["mean"], yerr=g["sem"], fmt=mk, ms=3.4, color="black", mfc="white", capsize=1.5, lw=.8)
ax.set_xlabel("day after seeding")
ax.set_ylabel(r"$R_{\mathrm{inh}}$ (µm)")
ax.set_title("(d) inhibited radius", loc="left", fontweight="bold", color=RED, pad=3)
ax.grid(axis="y", color="#ececec", lw=.4)
handles = [plt.Line2D([], [], color=GREY, lw=1.2, label="v8, no initial arrest"),
           plt.Line2D([], [], color=RED, lw=1.8, label="with initial arrest $q_0$"),
           plt.Line2D([], [], color="#555555", lw=1.4, ls=(0, (3, 1.5)), label="(d) 2% constant, squares"),
           plt.Line2D([], [], color="#555555", lw=1.4, label="(d) reoxygenated day 2, circles")]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.17))
fig.savefig(FIG / "figS_initial_condition.pdf", bbox_inches="tight")
fig.savefig(FIG / "figS_initial_condition.png", dpi=300, bbox_inches="tight")
say("\nsaved results/v8_initial/* and figures/figS_initial_condition (.pdf, .png)")
