"""Identifiability figures with one colour per parameter (v8 results).

Reads the profile likelihoods written by 29_v8_results.py (results/v8/profile_*.csv,
profile_exit_plus_G1.csv, identifiability.json) and the bivariate grids from
results/v8/jobs.jsonl, and draws:
  * Figure 6  - nine key univariate profiles, exit x G1 map, exit + G1 profile;
  * Figure 7  - identifiability map: supported range of every parameter relative to its
                estimate, against the search interval;
  * Figure S1 - all seventeen univariate profiles, exit + G1, and the a x p_pim map.

Profiles are shown as the normalised log-likelihood -dNLL (0 at the estimate). Each parameter
keeps the same colour in every figure. The shaded band is the supported region
(dNLL <= 10, residual optimisation noise); the dotted line is the nominal chi-square 95% level
(1.92) and the dashed line strong rejection (40).

Usage: python 30_identifiability_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
V8 = ROOT / "results" / "v8"
ART = ROOT / "figures"
FIG = ROOT / "figures"
SUPPORT, CHI2, REJECT, YMIN = 10.0, 1.92, 40.0, -48.0

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.2, "xtick.labelsize": 7.3,
                     "ytick.labelsize": 7.3, "legend.fontsize": 7.2, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "legend.frameon": False})

# one colour per parameter, the same in every figure
COLOR = {
    "bmax": "#c0392b",    # exit time            - red
    "k1max": "#1f5fa8",   # G1 time              - blue
    "S": "#5b2c83",       # exit + G1            - deep violet
    "Kq": "#1e8449",      # arrest threshold     - green
    "amax": "#e67e22",    # entry time           - orange
    "hq": "#8e44ad",      # Hill exponent        - purple
    "m_h": "#8d6e63",     # hypoxic consumption  - brown
    "tau_m": "#d81b60",   # adaptation time      - magenta
    "a": "#00838f",       # consumption ratio    - teal
    "p_pim": "#9e9d24",   # PIM threshold        - olive
    "K1": "#3949ab",      # G1 half-saturation   - indigo
    "mu0": "#5d4037",     # basal loss           - dark brown
    "mun": "#ef6c00",     # anoxic death         - dark orange
    "Kn": "#7f8c8d",      # death threshold      - grey (not identified)
    "kD": "#95a5a6",      # clearance 2%         - light grey (not identified)
    "kD_ox": "#00695c",   # clearance 21%        - dark teal
    "R1_norm": "#2e86c1", # day-1 radius 21%     - light blue
    "R1_hyp": "#b03a2e",  # day-1 radius 2%      - brick
}
# display: label, symbol, transform from the parameter value, log axis, unit
TITLE = {"bmax": "exit time", "k1max": "G1 time", "amax": "entry time", "Kq": "threshold",
         "hq": "Hill exponent", "K1": "G1 half-sat.", "a": "uptake ratio", "m_h": "uptake",
         "tau_m": "adaptation", "mu0": "basal loss", "mun": "anoxic death", "Kn": "death threshold",
         "kD": "clearance, 2%", "kD_ox": "clearance, 21%", "R1_norm": "radius, 21%", "R1_hyp": "radius, 2%",
         "p_pim": "PIM threshold"}
DISPLAY = {
    "bmax": ("mean exit time", r"$T_{\mathrm{exit}}$", lambda v: 1 / v / 24, True, "d"),
    "k1max": ("normoxic G1 time", r"$T_{\mathrm{G1}}$", lambda v: 1 / v, True, "h"),
    "amax": ("arrest-entry time", r"$1/\alpha_{\max}$", lambda v: 1 / v, True, "h"),
    "Kq": ("arrest threshold", r"$K_q$", lambda v: v, True, "% O$_2$"),
    "hq": ("Hill exponent", r"$h_q$", lambda v: v, False, ""),
    "K1": ("G1 half-saturation", r"$K_1$", lambda v: v, True, "% O$_2$"),
    "a": ("consumption ratio", r"$a$", lambda v: v, True, r"% O$_2$ µm$^{-2}$"),
    "m_h": ("hypoxic consumption", r"$m_h$", lambda v: v, True, ""),
    "tau_m": ("adaptation time", r"$\tau_m$", lambda v: v, True, "h"),
    "mu0": ("basal loss", r"$\mu_0$", lambda v: v, True, r"h$^{-1}$"),
    "mun": ("anoxic death", r"$\mu_{n,\max}$", lambda v: v, True, r"h$^{-1}$"),
    "Kn": ("death threshold", r"$K_n$", lambda v: v, True, "% O$_2$"),
    "kD": ("core clearance, 2%", r"$k_{D,h}$", lambda v: v, True, r"h$^{-1}$"),
    "kD_ox": ("core clearance, 21%", r"$k_{D,o}$", lambda v: v, True, r"h$^{-1}$"),
    "R1_norm": ("day-1 radius, 21%", r"$\rho_{\mathrm{norm}}$", lambda v: v, False, "µm"),
    "R1_hyp": ("day-1 radius, 2%", r"$\rho_{\mathrm{hyp}}$", lambda v: v, False, "µm"),
    "p_pim": ("pimonidazole threshold", r"$p_{\mathrm{pim}}$", lambda v: v, True, "% O$_2$"),
}
VERD = json.loads((V8 / "identifiability.json").read_text())
FINAL = json.loads((V8 / "final_fits.json").read_text())["all"]["finQ"]
NLL_BEST = FINAL["nll_train"]
PROF = {n: pd.read_csv(V8 / f"profile_{n}.csv") for n in DISPLAY}
SUMP = pd.read_csv(V8 / "profile_exit_plus_G1.csv").groupby("S_d", as_index=False).dNLL.min()


def tint(color, w):
    r, g, b = to_rgb(color)
    return (1 - w + w * r, 1 - w + w * g, 1 - w + w * b)


def crossing(x, y, x0, side, band=SUPPORT):
    """Edge of the supported region on one side of x0 (linear interpolation), or None."""
    idx = np.where(np.sign(x - x0) == side)[0]
    idx = idx[np.argsort(np.abs(x[idx] - x0))]
    lx, ly = x0, 0.0
    for i in idx:
        if y[i] > band:
            return lx + (band - ly) / (y[i] - ly) * (x[i] - lx)
        lx, ly = x[i], y[i]
    return None


def profile_panel(ax, xv, dnll, x_mle, color, logx, title, xlabel, bounds=None, tag=None, ticks=None):
    """Simpson-style profile: normalised log-likelihood -dNLL against the parameter."""
    order = np.argsort(xv)
    xv, dnll = np.asarray(xv)[order], np.asarray(dnll)[order]
    xs = np.log10(xv) if logx else xv
    x0 = np.log10(x_mle) if logx else x_mle
    ax.axhspan(-SUPPORT, 0.6, color=tint(color, .16), lw=0, zorder=0)
    ax.axhline(-CHI2, color="#555555", lw=.6, ls=(0, (1, 1.5)), zorder=1)
    ax.axhline(-REJECT, color="#555555", lw=.7, ls=(0, (4, 2)), zorder=1)
    # smooth monotone interpolation through the profile points
    if len(xs) >= 3:
        fine = np.linspace(xs.min(), xs.max(), 300)
        yy = -PchipInterpolator(xs, dnll)(fine)
        xf = 10 ** fine if logx else fine
        ax.plot(xf, yy, color=color, lw=1.9, zorder=3, solid_capstyle="round")  # leaves the axes when below
    y_pts = -dnll
    low = y_pts < YMIN
    ax.scatter(xv[~low], y_pts[~low], s=16, facecolor="white", edgecolor=color, lw=1.1, zorder=4)
    ax.scatter(xv[low], np.full(low.sum(), YMIN + 1.2), marker="v", s=18, color=color, zorder=4, clip_on=False)
    # supported interval as a coloured bracket on the band
    lo_e, hi_e = crossing(xs, dnll, x0, -1), crossing(xs, dnll, x0, 1)
    lo_x = (10 ** lo_e if logx else lo_e) if lo_e is not None else xv.min()
    hi_x = (10 ** hi_e if logx else hi_e) if hi_e is not None else xv.max()
    ax.plot([lo_x, hi_x], [-SUPPORT, -SUPPORT], color=color, lw=3.2, alpha=.55, solid_capstyle="butt", zorder=2)
    ax.axvline(x_mle, color=color, lw=.8, ls=(0, (2, 1.5)), zorder=2)
    ax.plot([x_mle], [0], marker="D", ms=4.6, color=color, mec="black", mew=.5, zorder=5)
    if bounds is not None:
        b_lo, b_hi = sorted(bounds)
        pad = (np.log10(xv.max()) - np.log10(xv.min())) * .06 if logx else (xv.max() - xv.min()) * .06
        xlo = 10 ** (np.log10(min(xv.min(), x_mle)) - pad) if logx else min(xv.min(), x_mle) - pad
        xhi = 10 ** (np.log10(max(xv.max(), x_mle)) + pad) if logx else max(xv.max(), x_mle) + pad
        if xlo < b_lo:
            ax.axvspan(xlo, b_lo, color="#d9d9d9", alpha=.7, lw=0, zorder=0)
        if xhi > b_hi:
            ax.axvspan(b_hi, xhi, color="#d9d9d9", alpha=.7, lw=0, zorder=0)
        ax.set_xlim(xlo, xhi)
    if logx:
        ax.set_xscale("log")
        span = np.log10(ax.get_xlim()[1] / ax.get_xlim()[0])
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,) if span > 1.3 else (1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if ticks is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_ylim(YMIN, 3)
    ax.set_yticks([0, -10, -20, -30, -40])
    ax.set_title(title, loc="left", color=color, fontweight="bold", pad=3, fontsize=8)
    if tag:
        ax.set_title(tag, loc="right", color=color, style="italic", pad=3, fontsize=6.6)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", color="#ececec", lw=.4, zorder=0)


def param_panel(ax, n, letter=None, short=False):
    lab, sym, tf, logx, unit = DISPLAY[n]
    df = PROF[n]
    xv = np.array([tf(v) for v in df.value])
    vd = VERD[n]
    bounds = (tf(vd["lo_bound"]), tf(vd["hi_bound"]))
    ticks = {"bmax": [2, 4, 7, 14, 28], "kD": [3e-4, 1e-3, 3e-3]}.get(n)
    title = (f"({letter}) " if letter else "") + (sym if short else f"{sym} {TITLE[n]}")
    xlabel = f"{lab} ({unit})" if unit else lab
    profile_panel(ax, xv, df.dNLL.values, tf(vd["mle"]), COLOR[n], logx, title, xlabel, bounds, vd["verdict"], ticks)


def sum_panel(ax, letter=None, short=False):
    profile_panel(ax, SUMP.S_d.values, SUMP.dNLL.values, (1 / FINAL["params"]["bmax"] + 1 / FINAL["params"]["k1max"]) / 24,
                  COLOR["S"], False, (f"({letter}) " if letter else "") + r"$T_{\mathrm{exit}}+T_{\mathrm{G1}}$",
                  "exit time + G1 time (d)", tag="one-sided" if short else "bounded below")


def grid2d(family):
    recs = [json.loads(s) for s in (V8 / "jobs.jsonl").read_text().splitlines() if s.strip()]
    best = {}
    for r in recs:
        if r.get("family") == family:
            k = json.dumps(r["value"])
            best[k] = min(best.get(k, np.inf), r["nll"])
    vals = [json.loads(k) for k in best]
    xs, ys = sorted({v[0] for v in vals}), sorted({v[1] for v in vals})
    Z = np.full((len(ys), len(xs)), np.nan)
    for k, nll in best.items():
        v = json.loads(k)
        Z[ys.index(v[1]), xs.index(v[0])] = nll - NLL_BEST
    return np.array(xs), np.array(ys), Z


CMAP = LinearSegmentedColormap.from_list("prof", ["#123f73", "#4f86c6", "#d6e4f0", "#f7e3c8", "#d98a4b", "#8e2b1a"])


def heatmap(ax, xs, ys, Z, xl, yl, title, fx, fy, mark=None):
    im = ax.imshow(np.minimum(Z, 100), origin="lower", cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    for (i, j), v in np.ndenumerate(Z):
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                    color="white" if (v < 15 or v > 75) else "#222222", fontweight="bold" if v <= SUPPORT else None)
            if v <= SUPPORT:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="white", lw=1.4))
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([fx(v) for v in xs])
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([fy(v) for v in ys])
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title, loc="left", fontweight="bold", pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def legend_handles():
    return [plt.Rectangle((0, 0), 1, 1, color="#cfcfcf", label="supported region (ΔNLL ≤ 10), tinted in the parameter colour"),
            plt.Line2D([], [], color="#555555", ls=(0, (1, 1.5)), lw=.8, label="χ² 95% level (ΔNLL = 1.92)"),
            plt.Line2D([], [], color="#555555", ls=(0, (4, 2)), lw=.8, label="strong rejection (ΔNLL = 40)"),
            plt.Line2D([], [], color="black", marker="D", ls="", ms=4, mfc="#bbbbbb", label="estimate"),
            plt.Rectangle((0, 0), 1, 1, color="#d9d9d9", label="outside search bounds")]


def save(fig, stem):
    fig.savefig(ART / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 6
KEY = ["bmax", "k1max", "Kq", "amax", "hq", "m_h", "tau_m", "a", "p_pim"]
fig = plt.figure(figsize=(7.2, 9.0))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.25], hspace=.72, wspace=.30)
for k, n in enumerate(KEY):
    ax = fig.add_subplot(gs[k // 3, k % 3])
    param_panel(ax, n, "abcdefghi"[k])
    if k % 3 == 0:
        ax.set_ylabel(r"$-\Delta$NLL")
ax_j = fig.add_subplot(gs[3, 0:2])
EG = grid2d("exit_G1")
im = heatmap(ax_j, EG[0] / 24, EG[1], EG[2], "mean exit time (d)", "normoxic G1 time (h)",
             r"(j) $T_{\mathrm{exit}}$ × $T_{\mathrm{G1}}$: $\Delta$NLL", lambda v: f"{v:.1f}", lambda v: f"{v:g}")
cb = fig.colorbar(im, ax=ax_j, fraction=.035, pad=.015)
cb.ax.tick_params(labelsize=6.5)
cb.outline.set_visible(False)
ax_k = fig.add_subplot(gs[3, 2])
sum_panel(ax_k, "k")
fig.legend(handles=legend_handles(), ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0), fontsize=6.9,
           columnspacing=1.2, handlelength=1.8)
fig.subplots_adjust(top=.915, left=.09, right=.985, bottom=.05)
save(fig, "fig06_identifiability")

# ------------------------------------------------------------------ Figure 7: identifiability map
GROUPS = [("Arrest kinetics", ["bmax", "amax", "Kq", "hq"]),
          ("Cell cycle", ["k1max", "K1", "S"]),
          ("Oxygen and metabolism", ["a", "m_h", "tau_m", "p_pim"]),
          ("Death and necrotic core", ["mun", "kD_ox", "mu0", "Kn", "kD"]),
          ("Initial size", ["R1_norm", "R1_hyp"])]
S_BEST = (1 / FINAL["params"]["bmax"] + 1 / FINAL["params"]["k1max"]) / 24
s_lo = crossing(SUMP.S_d.values, SUMP.dNLL.values, S_BEST, -1)


def entry(n):
    """Display-scale estimate, supported interval, search interval and verdict."""
    if n == "S":
        return dict(sym=r"$T_{\mathrm{exit}}+T_{\mathrm{G1}}$", lab="exit + G1 time", est=S_BEST, lo=s_lo, hi=None,
                    b_lo=None, b_hi=None, unit="d", verdict="bounded below", hi_plot=float(SUMP.S_d.max()))
    lab, sym, tf, logx, unit = DISPLAY[n]
    vd = VERD[n]
    est = tf(vd["mle"])
    ends = [tf(v) if v is not None else None for v in (vd["low"], vd["high"])]
    bnd = sorted([tf(vd["lo_bound"]), tf(vd["hi_bound"])])
    inverted = n in ("bmax", "k1max", "amax")       # rate parameters shown as times
    lo, hi = (ends[1], ends[0]) if inverted else (ends[0], ends[1])
    if vd["verdict"] == "not identified":
        lo = hi = None
    return dict(sym=sym, lab=lab, est=est, lo=lo, hi=hi, b_lo=bnd[0], b_hi=bnd[1], unit=unit, verdict=vd["verdict"])


rows = []
for g, names in GROUPS:
    rows.append(("group", g))
    rows += [("par", n) for n in names]
fig, ax = plt.subplots(figsize=(7.2, 6.9))
XL, XH = 1 / 50, 50
y = 0
yt, yl = [], []
for kind, item in rows[::-1]:
    if kind == "group":
        ax.text(XL * 1.02, y, item, fontsize=7.8, fontweight="bold", va="center", color="#333333")
        y += 1.0
        continue
    e = entry(item)
    c = COLOR[item]
    rel = lambda v: np.clip(v / e["est"], XL, XH)
    if e["b_lo"] is not None:  # search interval
        ax.plot([rel(e["b_lo"]), rel(e["b_hi"])], [y, y], color="#e3e3e3", lw=7, solid_capstyle="butt", zorder=1)
    if e["verdict"] == "not identified":
        ax.plot([rel(e["b_lo"]), rel(e["b_hi"])], [y, y], color=tint(c, .45), lw=7, solid_capstyle="butt", zorder=2)
        txt = "not identified (flat profile)"
    else:
        lo = e["lo"] if e["lo"] is not None else (e["b_lo"] if e["b_lo"] is not None else e["est"] / 1)
        hi = e["hi"] if e["hi"] is not None else (e["b_hi"] if e["b_hi"] is not None else e.get("hi_plot", XH * e["est"]))
        ax.plot([rel(lo), rel(hi)], [y, y], color=c, lw=7, solid_capstyle="butt", zorder=2, alpha=.9)
        for end, open_, fac in ((rel(lo), e["lo"] is None, 1 / 1.5), (rel(hi), e["hi"] is None, 1.5)):
            if open_:
                ax.annotate("", xy=(end * fac, y), xytext=(end, y), annotation_clip=False,
                            arrowprops=dict(arrowstyle="-|>", color=c, lw=1.4, mutation_scale=9), zorder=3)
        fmt = lambda v: f"{v:.3g}"
        rng = f"[{fmt(e['lo']) if e['lo'] is not None else ('bound' if e['b_lo'] is not None else '…')}, " \
              f"{fmt(e['hi']) if e['hi'] is not None else ('bound' if e['b_hi'] is not None else '>' + fmt(e.get('hi_plot', 0)) + ' (not tested)')}]"
        txt = f"{fmt(e['est'])} {e['unit']}  {rng}".replace("  ", " ")
    ax.plot([1], [y], marker="D", ms=5.5, color=c, mec="black", mew=.6, zorder=4)
    ax.text(XH * 1.12, y, txt, fontsize=7, va="center", color="#222222")
    yt.append(y)
    yl.append(f"{e['sym']}  {e['lab']}")
    y += 1.0
ax.set_yticks(yt)
ax.set_yticklabels(yl, fontsize=7.6)
for t, (kind, item) in zip(ax.get_yticklabels(), [r for r in rows[::-1] if r[0] == "par"]):
    t.set_color(COLOR[item])
ax.axvline(1, color="#444444", lw=.7, ls=(0, (2, 2)), zorder=0)
ax.set_xscale("log")
ax.set_xlim(XL, XH)
ax.set_ylim(-.8, y - .2)
ax.set_xticks([1 / 50, 1 / 10, 1 / 3, 1, 3, 10, 50])
ax.set_xticklabels(["1/50", "1/10", "1/3", "1", "3", "10", "50"])
ax.set_xlabel("value relative to the estimate (fold change; supported region ΔNLL ≤ 10)")
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#eeeeee", lw=.5, zorder=0)
ax.text(XH * 1.12, y - .3, "estimate  [supported range]", fontsize=7, fontweight="bold", va="center")
handles = [plt.Line2D([], [], color="#e3e3e3", lw=7, label="search interval"),
           plt.Line2D([], [], color="#777777", lw=7, label="supported range (parameter colour)"),
           plt.Line2D([], [], color="#777777", marker=r"$\rightarrow$", ls="", ms=10, label="supported up to the search bound / grid end"),
           plt.Line2D([], [], color="#777777", marker="D", ls="", ms=5, mec="black", label="estimate")]
ax.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(.45, 1.07), fontsize=6.9)
fig.subplots_adjust(left=.27, right=.73, top=.93, bottom=.08)
save(fig, "fig07_identifiability_map")

# ------------------------------------------------------------------ Figure S1: every profile
ORDER = ["bmax", "k1max", "amax", "Kq", "hq", "K1", "a", "m_h", "tau_m", "p_pim", "mu0", "mun", "Kn", "kD",
         "kD_ox", "R1_norm", "R1_hyp"]
fig = plt.figure(figsize=(7.2, 10.2))
gs = fig.add_gridspec(5, 4, hspace=.85, wspace=.36)
LET = "abcdefghijklmnopqrs"
for k, n in enumerate(ORDER):
    ax = fig.add_subplot(gs[k // 4, k % 4])
    param_panel(ax, n, LET[k], short=True)
    if k % 4 == 0:
        ax.set_ylabel(r"$-\Delta$NLL")
sub = gs[4, 1:4].subgridspec(1, 3, width_ratios=[1, .42, 1.95], wspace=0)
ax = fig.add_subplot(sub[0])
sum_panel(ax, LET[17], short=True)
AP = grid2d("a_pim")
ax_h = fig.add_subplot(sub[2])
im = heatmap(ax_h, AP[0], AP[1], AP[2], r"consumption ratio $a$ ($\times10^{-3}$)", r"$p_{\mathrm{pim}}$ (% O$_2$)",
             r"(s) $a$ × $p_{\mathrm{pim}}$: $\Delta$NLL", lambda v: f"{1e3 * v:.2f}", lambda v: f"{v:g}")
cb = fig.colorbar(im, ax=ax_h, fraction=.04, pad=.02)
cb.ax.tick_params(labelsize=6.5)
cb.outline.set_visible(False)
fig.legend(handles=legend_handles(), ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.0), fontsize=6.9)
fig.subplots_adjust(top=.93, left=.08, right=.985, bottom=.05)
save(fig, "figS1_all_profiles")
print("saved fig06_identifiability, fig07_identifiability_map, figS1_all_profiles (PDF and PNG in figures)")
