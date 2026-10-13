"""Results of the observation-window sensitivity (33_window_sensitivity.py) next to v8 (W = 12 h).

For W = 6, 12 and 18 h: best fit of each model, AIC differences, M_F estimates, exit-time profile
(supported range, penalties at 4 and 17 days), lower bound of exit + G1, and the readout-versus-arrest
result after day-2 reoxygenation. W = 12 h is taken from the v8 records (results/v8/jobs.jsonl).

Writes results/v8_window/{summary.txt, window_sensitivity.csv, exit_profiles.csv, sum_profiles.csv,
table_window_sensitivity.tex} and figures/figS_window_sensitivity.{pdf,png}. The article folder is
not touched.

Usage: python 34_window_results.py
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
sys.argv = [str(HERE / "33_window_sensitivity.py")]
spec = importlib.util.spec_from_file_location("w33", HERE / "33_window_sensitivity.py")
w33 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w33)
sys.argv = _argv
p28, CTX = w33.p28, w33.CTX
rm = CTX["all"].rm
ROOT = p28.ROOT
OUT = ROOT / "results" / "v8_window"
FIG = ROOT / "figures"
V8 = ROOT / "results" / "v8"
SUPPORT, CHI2, REJECT = 10.0, 1.92, 40.0
VARIANTS = ("finQ", "delayQ", "instQ", "noQ")
WCOL = {6.0: "#e8a39b", 12.0: "#c0392b", 18.0: "#5e1a12"}
lines = []


def say(s=""):
    print(s, flush=True)
    lines.append(s)


# ------------------------------------------------------------------ records
REC = {w: [r for r in w33.records() if r["window"] == w] for w in w33.WINDOWS}
v8 = [r for r in p28.records() if r["ctx"] == "all" and tuple(r.get("scen", p28.BASE)) == p28.BASE]
fam12 = {"exit_all": "exit", "exit_plus_G1": "sum"}
REC[12.0] = []
for r in v8:
    r = dict(r, window=12.0)
    if r["kind"] == "fix":
        if r.get("family") not in fam12:
            r = dict(r, family="other")  # other v8 profiles still count for the optimum
        else:
            r["family"] = f"w12|{fam12[r['family']]}"
    REC[12.0].append(r)
WINDOWS = sorted(REC)
m = CTX["all"]
K = {v: len(m.bounds(v)[0]) + 4 for v in VARIANTS}


def theta(x, v="finQ"):
    return m.to_theta(np.array(x), v)


def crossing(xs, ys, x0, side):
    """Edge of the supported region on one side of x0, linear in log10(x); None if not reached.

    The edge is placed beyond the outermost supported grid point on that side, so that a single
    under-optimised interior point (residual optimisation noise) does not truncate the region."""
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


def readout_vs_arrest(th, M=64):
    """Single (unaveraged) M_F trajectory after day-2 reoxygenation, as in 29_v8_results.py."""
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
    return t[lost], 100 * f[lost], 100 * f[int(np.argmin(np.abs(t - 2.01)))]


def profile(recs, fam):
    pts = {}
    for r in recs:
        if r.get("family") == fam and r["kind"] == "fix":
            if r["value"] not in pts or r["nll"] < pts[r["value"]]["nll"]:
                pts[r["value"]] = r
    vals = np.array(sorted(pts), float)
    return vals, np.array([pts[v]["nll"] for v in sorted(pts)])


# ------------------------------------------------------------------ per window
rows, EXP, SUMP, BEST = [], [], [], {}
for w in WINDOWS:
    recs = REC[w]
    best = {v: min((r for r in recs if r["variant"] == v), key=lambda r: r["nll"], default=None) for v in VARIANTS}
    if any(b is None for b in best.values()) or not profile(recs, f"w{w:g}|exit")[0].size:
        print(f"W = {w:g} h: results incomplete, skipped", flush=True)
        continue
    F = best["finQ"]
    BEST[w] = F
    aic = {v: 2 * best[v]["nll"] + 2 * K[v] for v in VARIANTS if best[v]}
    th = theta(F["x"])
    ex, en = profile(recs, f"w{w:g}|exit")
    ex_all = np.append(ex, 1 / th["bmax"])
    en_all = np.append(en, F["nll"])
    o = np.argsort(ex_all)
    ex_all, dn_all = ex_all[o], en_all[o] - F["nll"]
    lo, hi = crossing(ex_all, dn_all, 1 / th["bmax"], -1), crossing(ex_all, dn_all, 1 / th["bmax"], 1)
    for T, dn in zip(ex_all, dn_all):
        EXP.append(dict(W=w, exit_h=T, dNLL=dn))
    sv, sn = profile(recs, f"w{w:g}|sum")
    S_best = (1 / th["bmax"] + 1 / th["k1max"]) / 24
    sv_all, sd_all = np.append(sv, S_best), np.append(sn, F["nll"]) - F["nll"]
    o = np.argsort(sv_all)
    sv_all, sd_all = sv_all[o], sd_all[o]
    s_lo = crossing(sv_all, sd_all, S_best, -1)
    for S_, dn in zip(sv_all, sd_all):
        SUMP.append(dict(W=w, S_d=S_, dNLL=dn))
    near = lambda T: float(dn_all[np.argmin(np.abs(ex_all - T))]) if np.min(np.abs(ex_all - T)) < 1 else np.nan
    clear_day, arrested, at_switch = readout_vs_arrest(th)
    rows.append(dict(
        W=w, nll_F=F["nll"], n_records=len(recs),
        dAIC_D=aic["delayQ"] - aic["finQ"], dAIC_I=aic["instQ"] - aic["finQ"], dAIC_0=aic["noQ"] - aic["finQ"],
        exit_d=1 / th["bmax"] / 24, exit_lo_d=lo / 24 if lo else np.nan, exit_hi_d=hi / 24 if hi else np.nan,
        G1_h=1 / th["k1max"], sum_d=S_best, sum_lo_d=s_lo if s_lo else np.nan,
        dNLL_4d=near(96), dNLL_17d=near(400), Kq=th["Kq"], hq=th["hq"], entry_h=1 / th["amax"],
        m_h=th["m_h"], tau_m=th["tau_m"], p_pim=th["p_pim"], a=th["a"],
        sd_inh=F["sds"][1], clear_day=clear_day, arrested_pct=arrested, arrested_at_switch=at_switch))

WINDOWS = [w for w in WINDOWS if w in BEST]
T = pd.DataFrame(rows)
T.to_csv(OUT / "window_sensitivity.csv", index=False, float_format="%.6g")
pd.DataFrame(EXP).to_csv(OUT / "exit_profiles.csv", index=False, float_format="%.6g")
pd.DataFrame(SUMP).to_csv(OUT / "sum_profiles.csv", index=False, float_format="%.6g")

say("Observation-window sensitivity (W = half-width of the 5-point window; W = 12 h is v8)")
say(f"records: " + ", ".join(f"W={w:g}: {len(REC[w])}" for w in WINDOWS))
say()
show = T[["W", "nll_F", "dAIC_D", "dAIC_I", "dAIC_0", "exit_d", "exit_lo_d", "exit_hi_d", "G1_h", "sum_d",
          "sum_lo_d", "dNLL_4d", "dNLL_17d"]]
say(show.round(2).to_string(index=False))
say()
say(T[["W", "Kq", "hq", "entry_h", "m_h", "tau_m", "p_pim", "a", "sd_inh"]].round(4).to_string(index=False))
say()
say(T[["W", "clear_day", "arrested_pct", "arrested_at_switch"]].round(2).to_string(index=False))
say()
for w in WINDOWS:
    e = pd.DataFrame(EXP)
    e = e[e.W == w]
    say(f"exit profile W={w:g}: " + "  ".join(f"{r.exit_h:.0f}h:{r.dNLL:.1f}" for r in e.itertuples()))
for w in WINDOWS:
    s = pd.DataFrame(SUMP)
    s = s[s.W == w]
    say(f"exit+G1 profile W={w:g}: " + "  ".join(f"{r.S_d:.2f}d:{r.dNLL:.1f}" for r in s.itertuples()))
(OUT / "summary.txt").write_text("\n".join(lines), encoding="utf8")

# ------------------------------------------------------------------ LaTeX table
fmt = lambda v, d=1: "--" if not np.isfinite(v) else f"{v:.{d}f}"
tex = [r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}rrrrrcrrrr@{}}", r"\toprule",
       r" & \multicolumn{3}{c}{$\Delta\mathrm{AIC}$} & \multicolumn{2}{c}{Exit (d)} & & Exit + G1 & "
       r"\multicolumn{2}{c}{Readout clearance} \\",
       r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){9-10}",
       r"$W$ (h) & $\mathcal{M}_D$ & $\mathcal{M}_I$ & $\mathcal{M}_0$ & best & supported & G1 (h) & "
       r"min.\ (d) & day & arrested (\%) \\", r"\midrule"]
for r in T.itertuples():
    hi = fmt(r.exit_hi_d, 1) if np.isfinite(r.exit_hi_d) else r"open"
    lo = fmt(r.exit_lo_d, 1) if np.isfinite(r.exit_lo_d) else r"open"
    tex.append(f"{r.W:.0f}{'$^{*}$' if r.W == 12 else ''} & {fmt(r.dAIC_D)} & {fmt(r.dAIC_I)} & {fmt(r.dAIC_0)} & "
               f"{fmt(r.exit_d, 2)} & {lo}--{hi} & {fmt(r.G1_h)} & {fmt(r.sum_lo_d, 1)} & "
               f"{r.clear_day:.2f} & {r.arrested_pct:.0f} \\\\")
tex += [r"\bottomrule", r"\end{tabular}"]
(OUT / "table_window_sensitivity.tex").write_text("\n".join(tex), encoding="utf8")

# ------------------------------------------------------------------ figure
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.2, "xtick.labelsize": 7.3,
                     "ytick.labelsize": 7.3, "legend.fontsize": 7.0, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "legend.frameon": False})
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=.42))
E, Sx = pd.DataFrame(EXP), pd.DataFrame(SUMP)
for ax, D, xc, xl, ticks, ttl in ((axes[0], E, "exit_h", "mean exit time (d)", [2, 4, 7, 14, 28], "(a) exit time"),
                                  (axes[1], Sx, "S_d", "exit time + G1 time (d)", [4, 6, 8, 10, 12], "(b) exit + G1")):
    ax.axhspan(-SUPPORT, .6, color="#f3dcd9", lw=0, zorder=0)
    ax.axhline(-CHI2, color="#555555", lw=.6, ls=(0, (1, 1.5)))
    ax.axhline(-REJECT, color="#555555", lw=.7, ls=(0, (4, 2)))
    for w in WINDOWS:
        d = D[D.W == w].sort_values(xc)
        x = d[xc].values / (24 if xc == "exit_h" else 1)
        y = -d.dNLL.values
        if xc == "exit_h":
            f = np.linspace(np.log10(x.min()), np.log10(x.max()), 300)
            ax.plot(10 ** f, PchipInterpolator(np.log10(x), y)(f), color=WCOL[w], lw=1.7, label=f"{w:g} h" + (" (v8)" if w == 12 else ""))
        else:
            f = np.linspace(x.min(), x.max(), 300)
            ax.plot(f, PchipInterpolator(x, y)(f), color=WCOL[w], lw=1.7)
        ok = y >= -48
        ax.scatter(x[ok], y[ok], s=12, facecolor="white", edgecolor=WCOL[w], lw=1.0, zorder=4)
        ax.scatter(x[~ok], np.full((~ok).sum(), -46.8), marker="v", s=13, color=WCOL[w], zorder=4, clip_on=False)
    if xc == "exit_h":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([str(t) for t in ticks])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(1.9, 36)
        ax.legend(loc="lower left", title="window W", title_fontsize=7.0, handlelength=1.4, borderaxespad=0.2)
    ax.set_ylim(-48, 3)
    ax.set_yticks([0, -10, -20, -30, -40])
    ax.set_ylabel(r"$-\Delta$NLL")
    ax.set_xlabel(xl)
    ax.set_title(ttl, loc="left", fontweight="bold", color="#c0392b" if xc == "exit_h" else "#5b2c83", pad=3)
# (c) fitted inhibited radius after day-2 and day-4 reoxygenation, each with its own window
ax = axes[2]
d793 = m.d
for prot, mk in (("reox_d2", "o"), ("reox_d4", "s")):
    obs = d793[d793.protocol == prot].groupby("day").R_inhib.agg(["mean", "sem"])
    ax.errorbar(obs.index, obs["mean"], yerr=obs["sem"], fmt=mk, ms=3.6, color="black", mfc="white", capsize=1.5,
                lw=.8, zorder=5)
t_days = np.arange(2.0, 8.01, 0.125)
for w in WINDOWS:
    th = theta(BEST[w]["x"])
    rm.DT = 0.5
    mdl = rm.Model("finQ", M=64, protocols=["reox_d2", "reox_d4"])
    o = m.run_windowed(mdl, th, (t_days - rm.T0_DAY) * 24, w=w)
    for i, ls in ((0, "-"), (1, (0, (3, 1.5)))):
        ax.plot(t_days, o[i, :, 1], color=WCOL[w], lw=1.5, ls=ls)
ax.set_xlabel("day after seeding")
ax.set_ylabel(r"$R_{\mathrm{inh}}$ (µm)")
ax.set_title("(c) inhibited radius", loc="left", fontweight="bold", color="#c0392b", pad=3)
ax.plot([], [], color="#777777", lw=1.4, label="reox. day 2")
ax.plot([], [], color="#777777", lw=1.4, ls=(0, (3, 1.5)), label="reox. day 4")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.6, columnspacing=1.0)
ax.grid(axis="y", color="#ececec", lw=.4)
fig.savefig(FIG / "figS_window_sensitivity.pdf", bbox_inches="tight")
fig.savefig(FIG / "figS_window_sensitivity.png", dpi=300, bbox_inches="tight")
say("\nsaved results/v8_window/* and figures/figS_window_sensitivity (.pdf, .png)")
