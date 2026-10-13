"""v7 results: consolidated fits, profiles, validation, sensitivity, identifiability,
numerical checks, Figures 3-6 and S1-S3, LaTeX tables and a numeric summary.

Reads results/v7/jobs.jsonl written by 26_v7_pipeline.py (48 equal-thickness shells,
exact pimonidazole radius, dt = 0.5 h).
The best record of each (data set, model, fixed-choice scenario) is the final estimate,
whatever stage produced it (free fit, profile point or polish): every record is a valid
parameter vector evaluated with the same likelihood.

Usage: python 25_v6_results.py
"""
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [str(HERE / "26_v7_pipeline.py"), "1"]
spec = importlib.util.spec_from_file_location("p24", HERE / "26_v7_pipeline.py")
p24 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p24)
sys.argv = _argv
CTX, rm, BASE, SCEN, VARIANTS = p24.CTX, p24.rm, p24.BASE, p24.SCEN, p24.VARIANTS
ROOT = p24.ROOT
V6 = ROOT / "results" / "v7"
ART = ROOT / "figures"
FIG = ROOT / "figures"
M_FINE, DT_FINE = 48, 0.5
if os.environ.get("V6_TEST_DIR"):  # dry run on synthetic records: write nothing into the project
    V6 = ART = FIG = Path(os.environ["V6_TEST_DIR"])

# visual conventions of Figures 1 and 3-5
plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.2,
                     "pdf.fonttype": 42, "ps.fonttype": 42})
COL = {"finQ": "#d62728", "delayQ": "#2878b5", "instQ": "#777777", "noQ": "#2ca02c"}
LS = {"finQ": "-", "delayQ": "--", "instQ": ":", "noQ": "-."}
NAME = {"finQ": r"finite arrest, $\mathcal{M}_F$", "delayQ": r"delayed entry, $\mathcal{M}_D$",
        "instQ": r"instantaneous arrest, $\mathcal{M}_I$", "noQ": r"cycling only, $\mathcal{M}_0$"}
SYM = {"finQ": r"$\mathcal{M}_F$", "delayQ": r"$\mathcal{M}_D$", "instQ": r"$\mathcal{M}_I$",
       "noQ": r"$\mathcal{M}_0$"}
summary = []


def say(s=""):
    print(s, flush=True)
    summary.append(s)


def save(fig, stem, to_article=True):
    if to_article:
        fig.savefig(ART / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ records and final fits
recs = p24.records()
say(f"v6 records: {len(recs)} optimisations")


def same_scen(r, sc):
    return tuple(r.get("scen", BASE)) == tuple(sc)


def best_rec(ctx, v, sc=BASE):
    sub = [r for r in recs if r["ctx"] == ctx and r["variant"] == v and same_scen(r, sc)]
    return min(sub, key=lambda r: r["nll"]) if sub else None


def fit_entry(ctx, v, rec):
    m = CTX[ctx]
    names, lo, hi = m.bounds(v)
    x = np.array(rec["x"])
    th = m.to_theta(x, v)
    k = len(names) + 4
    n_obs = int(np.isfinite(m.IDX_TRAIN[2]).sum())
    return dict(params=th, free=names, x=rec["x"], nll_train=rec["nll"], n_params=k, n_obs=n_obs,
                AIC=2 * rec["nll"] + 2 * k, BIC=2 * rec["nll"] + k * np.log(n_obs),
                sd=dict(zip(m.OBS, rec["sds"])), source=rec["id"],
                at_bound=[n for n, xi, l, h in zip(names, x, lo, hi) if xi <= l + 0.02 or xi >= h - 0.02],
                near_bound=[n for n, xi, l, h in zip(names, x, lo, hi)
                            if (l + 0.02 < xi <= l + 0.06) or (h - 0.06 <= xi < h - 0.02)],
                nll_test=rec.get("nll_test"))


FINAL = {ctx: {v: fit_entry(ctx, v, best_rec(ctx, v)) for v in VARIANTS if best_rec(ctx, v)}
         for ctx in ("all", "train", "WM164", "WM983b")}
(V6 / "final_fits.json").write_text(json.dumps(FINAL, indent=2, default=float))
if not os.environ.get("V6_TEST_DIR"):
    (ROOT / "results" / "radial_fit_WM793b_alldata_v7.json").write_text(json.dumps(FINAL["all"], indent=2, default=float))
F = FINAL["all"]["finQ"]
NLL_BEST = F["nll_train"]
names_F, lo_F, hi_F = CTX["all"].bounds("finQ")

say("\n=== 1. Final fits (48 equal-thickness shells, exact R_pim) ===")
for ctx in ("all", "WM164", "WM983b", "train"):
    best_aic = min(e["AIC"] for e in FINAL[ctx].values())
    for v, e in FINAL[ctx].items():
        th = e["params"]
        extra = f" exit={1 / th['bmax']:.0f} h" if v in ("finQ", "delayQ") else ""
        test = f" test={e['nll_test']:.2f}" if e.get("nll_test") is not None else ""
        say(f"{ctx:7s} {v:7s} NLL={e['nll_train']:9.2f} AIC={e['AIC']:8.1f} dAIC={e['AIC'] - best_aic:6.1f}"
            f"{extra} G1={1 / th['k1max']:.1f} h sd={np.round(list(e['sd'].values()), 1).tolist()}{test} "
            f"bound={e['at_bound']} near={e['near_bound']}")


# ------------------------------------------------------------------ dense predictions
def dense(ctx, v, M=M_FINE, dt=DT_FINE, t_end=7 * 24):
    m = CTX[ctx]
    rm.DT = dt
    mdl = rm.Model(v, M=M, protocols=m.PROTS)
    t = np.arange(0, t_end + 1, 6.0)
    o = mdl.run(FINAL[ctx][v]["params"], t)
    rm.DT = DT_FINE
    rows = []
    for i, p in enumerate(mdl.protocols):
        for j, tt in enumerate(t):
            rows.append(dict(variant=v, protocol=p, day=rm.T0_DAY + tt / 24, R_outer=o[i, j, 0],
                             R_inhib=o[i, j, 1], R_nec=o[i, j, 2], R_pim=o[i, j, 3], f_Q=o[i, j, 4]))
    return rows


PRED = {}
for ctx, line in (("all", "WM793b"), ("WM164", "WM164"), ("WM983b", "WM983b")):
    PRED[line] = pd.DataFrame([row for v in FINAL[ctx] for row in dense(ctx, v)])
    PRED[line].to_csv(V6 / f"pred_{line}.csv", index=False, float_format="%.5g")

# key predictions of the inhibited radius
d793 = CTX["all"].d
say("\n=== 2. Inhibited radius at decisive points (um) ===")
for prot, day in (("reox_d2", 3), ("reox_d2", 4), ("reox_d4", 6), ("deox_d2", 3)):
    obs = d793[(d793.protocol == prot) & (d793.day == day)].R_inhib.mean()
    pr = PRED["WM793b"]
    vals = {v: float(pr[(pr.variant == v) & (pr.protocol == prot) & (np.isclose(pr.day, day))].R_inhib.iloc[0])
            for v in VARIANTS}
    say(f"{prot} day {day}: observed {obs:.1f}; " + ", ".join(f"{v} {x:.1f}" for v, x in vals.items()))

# ------------------------------------------------------------------ exit profiles and validation
fam = lambda f, **kw: [r for r in recs if r.get("family") == f and all(r.get(k) == v for k, v in kw.items())]


def profile_table(records, value_key="value"):
    best_by = {}
    for r in records:
        key = json.dumps(r[value_key])
        if key not in best_by or r["nll"] < best_by[key]["nll"]:
            best_by[key] = r
    return [best_by[k] for k in best_by]


exit_all = sorted(profile_table(fam("exit_all")), key=lambda r: r["value"])
# include the final optimum itself as a profile point
prof_all = pd.DataFrame([dict(exit_h=r["value"], nll=r["nll"],
                              G1_h=1 / CTX["all"].to_theta(np.array(r["x"]), "finQ")["k1max"]) for r in exit_all])
prof_all = pd.concat([prof_all, pd.DataFrame([dict(exit_h=1 / F["params"]["bmax"], nll=NLL_BEST,
                                                   G1_h=1 / F["params"]["k1max"])])]).sort_values("exit_h")
prof_all["dNLL"] = prof_all.nll - NLL_BEST
prof_all.to_csv(V6 / "exit_profile_all.csv", index=False, float_format="%.6g")
# start-to-start spread of the duplicated profile points: an empirical optimisation-noise level
dup = pd.DataFrame([dict(T=r["value"], nll=r["nll"]) for r in fam("exit_all")])
spread = dup.groupby("T").nll.agg(lambda s: s.max() - s.min())
say("\n=== 3. Exit-time profile, all protocols (dNLL vs optimum; G1 re-estimated) ===")
say(prof_all.round(2).to_string(index=False))
say(f"spread between the two starts per grid point: median {spread.median():.1f}, max {spread.max():.1f}")

exit_tr = sorted(profile_table(fam("exit_train")), key=lambda r: r["value"])
prof_tr = pd.DataFrame([dict(exit_h=r["value"], nll_train=r["nll"], nll_test=r["nll_test"], Rinh_d6=r["Rinh_d6"],
                             G1_h=1 / CTX["train"].to_theta(np.array(r["x"]), "finQ")["k1max"]) for r in exit_tr])
prof_tr["dTrain"] = prof_tr.nll_train - min(prof_tr.nll_train.min(), FINAL["train"]["finQ"]["nll_train"])
prof_tr["dTest"] = prof_tr.nll_test - prof_tr.nll_test.min()
prof_tr["Rinh_d6_obs"] = d793[(d793.protocol == "reox_d4") & (d793.day == 6)].R_inhib.mean()
prof_tr.to_csv(V6 / "exit_profile_training.csv", index=False, float_format="%.6g")
say("\n=== 4. Training-only exit profile scored on the withheld day-4 reoxygenation ===")
say(prof_tr.round(2).to_string(index=False))
say("held-out NLL of the training optimum of each model: " +
    ", ".join(f"{v} {FINAL['train'][v]['nll_test']:.2f}" for v in VARIANTS))

# ------------------------------------------------------------------ sensitivity
say("\n=== 5. Fixed-choice sensitivity ===")
rows = []
scen_all = {"baseline": BASE, **SCEN}
for s, sc in scen_all.items():
    bf = best_rec("all", "finQ", sc)
    bi, b0 = best_rec("all", "instQ", sc), best_rec("all", "noQ", sc)
    kF, kI, k0 = len(names_F) + 4, len(CTX["all"].bounds("instQ")[0]) + 4, len(CTX["all"].bounds("noQ")[0]) + 4
    ex = (prof_all.set_index("exit_h").nll if s == "baseline"
          else pd.Series({r["value"]: r["nll"] for r in profile_table(fam(f"scen_exit|{s}"))}))
    th = CTX["all"].to_theta(np.array(bf["x"]), "finQ")
    rows.append(dict(scenario=s, eps_Q=sc[0], k2=sc[1], nll_F=bf["nll"],
                     dAIC_I=2 * bi["nll"] + 2 * kI - (2 * bf["nll"] + 2 * kF),
                     dAIC_0=2 * b0["nll"] + 2 * k0 - (2 * bf["nll"] + 2 * kF),
                     exit_d=1 / th["bmax"] / 24, G1_h=1 / th["k1max"],
                     dNLL_4d=ex.get(96, np.nan) - bf["nll"], dNLL_17d=ex.get(400, np.nan) - bf["nll"]))
SENS = pd.DataFrame(rows)
SENS.to_csv(V6 / "sensitivity.csv", index=False, float_format="%.6g")
say(SENS.round(2).to_string(index=False))

# ------------------------------------------------------------------ univariate profiles
DISPLAY = {  # name: (label, transform from parameter value, log axis)
    "bmax": ("mean exit time (d)", lambda v: 1 / v / 24, True),
    "k1max": ("normoxic G1 time (h)", lambda v: 1 / v, True),
    "amax": ("arrest-entry time (h)", lambda v: 1 / v, True),
    "Kq": (r"arrest threshold $K_q$ (% O$_2$)", lambda v: v, True),
    "hq": (r"Hill exponent $h_q$", lambda v: v, False),
    "K1": (r"G1 half-saturation $K_1$ (% O$_2$)", lambda v: v, True),
    "a": (r"consumption ratio $a$", lambda v: v, True),
    "mu0": (r"basal loss $\mu_0$ (h$^{-1}$)", lambda v: v, True),
    "mun": (r"anoxic death $\mu_{n,\max}$ (h$^{-1}$)", lambda v: v, True),
    "Kn": (r"death threshold $K_n$ (% O$_2$)", lambda v: v, True),
    "kD": (r"core clearance, 2% ($k_{D,h}$, h$^{-1}$)", lambda v: v, True),
    "kD_ox": (r"core clearance, 21% ($k_{D,o}$, h$^{-1}$)", lambda v: v, True),
    "m_h": (r"hypoxic consumption $m_h$", lambda v: v, True),
    "tau_m": (r"adaptation time $\tau_m$ (h)", lambda v: v, True),
    "R1_norm": (r"day-1 radius, 21% (µm)", lambda v: v, False),
    "R1_hyp": (r"day-1 radius, 2% (µm)", lambda v: v, False),
    "p_pim": (r"PIM threshold $p_{\mathrm{pim}}$ (% O$_2$)", lambda v: v, True),
}
# support band: the start-to-start optimisation noise, bounded below by the nominal chi-square
# level (1.92) and above by the 20 units used in v5
SUPPORT = float(np.clip(np.ceil(spread.median()), 1.92, 20.0)) if len(spread) else 1.92
say(f"\n=== 6. Univariate profiles (support band dNLL <= {SUPPORT:.2f}) ===")
UNI = {}
for n in names_F:
    if n == "bmax":
        pts = [(1 / r["value"], r["nll"]) for r in exit_all]  # grid value is the exit time T = 1/bmax
    else:
        pts = [(r["value"], r["nll"]) for r in profile_table(fam(f"uni|{n}"))]
    pts.append((F["params"][n], NLL_BEST))
    df = pd.DataFrame(pts, columns=["value", "nll"]).groupby("value", as_index=False).nll.min().sort_values("value")
    df["dNLL"] = df.nll - NLL_BEST
    UNI[n] = df
    df.to_csv(V6 / f"profile_{n}.csv", index=False, float_format="%.6g")


def interval(df, mle, lo_b, hi_b, band):
    """Supported interval on the parameter scale, by linear interpolation of dNLL in log10."""
    x = np.log10(df.value.values)
    y = df.dNLL.values
    xm = np.log10(mle)
    out = []
    for side in (-1, 1):
        idx = np.where(np.sign(x - xm) == side)[0]
        idx = idx[np.argsort(np.abs(x[idx] - xm))]
        last_x, last_y, end = xm, 0.0, None
        for i in idx:
            if y[i] > band:
                w = (band - last_y) / (y[i] - last_y) if y[i] != last_y else 0.0
                end = last_x + w * (x[i] - last_x)
                break
            last_x, last_y = x[i], y[i]
        out.append(10 ** end if end is not None else None)
    return out


VERDICT = {}
for n, df in UNI.items():
    j = names_F.index(n)
    lo_v, hi_v = 10 ** lo_F[j], 10 ** hi_F[j]
    left, right = interval(df, F["params"][n], lo_v, hi_v, SUPPORT)
    at_lo = np.log10(F["params"][n]) <= lo_F[j] + 0.02
    at_hi = np.log10(F["params"][n]) >= hi_F[j] - 0.02
    closed = (left is not None or at_lo, right is not None or at_hi)
    if left is not None and right is not None:
        verdict = "identified"
    elif (left is not None) or (right is not None):
        verdict = "one-sided"
    else:
        verdict = "not identified"
    VERDICT[n] = dict(mle=F["params"][n], low=left, high=right, verdict=verdict, at_lo=bool(at_lo),
                      at_hi=bool(at_hi), lo_bound=lo_v, hi_bound=hi_v,
                      max_dNLL=float(df.dNLL.max()))
    lab, tf, _ = DISPLAY[n]
    rng = [tf(v) if v is not None else None for v in (left, right)]
    say(f"{n:8s} MLE={F['params'][n]:.4g}  supported=[{left if left is None else f'{left:.4g}'}, "
        f"{right if right is None else f'{right:.4g}'}]  verdict={verdict}  ({lab}: "
        f"{tf(F['params'][n]):.3g}; range {rng})")
(V6 / "identifiability.json").write_text(json.dumps(VERDICT, indent=2, default=float))

# ------------------------------------------------------------------ bivariate profiles
def grid2d(f):
    t = profile_table(fam(f))
    xs = sorted({r["value"][0] for r in t})
    ys = sorted({r["value"][1] for r in t})
    Z = np.full((len(ys), len(xs)), np.nan)
    for r in t:
        Z[ys.index(r["value"][1]), xs.index(r["value"][0])] = r["nll"] - NLL_BEST
    return np.array(xs), np.array(ys), Z


EG = grid2d("exit_G1")
AP = grid2d("a_pim")
say("\n=== 7. Bivariate profiles (dNLL vs optimum) ===")
say("exit (h) x G1 (h): rows G1 " + str(EG[1].tolist()) + ", columns exit " + str(EG[0].tolist()))
say(pd.DataFrame(EG[2], index=EG[1], columns=EG[0]).round(1).to_string())
say("a x p_pim: rows p_pim " + str(AP[1].tolist()) + ", columns a " + str(np.round(AP[0], 5).tolist()))
say(pd.DataFrame(AP[2], index=AP[1], columns=np.round(AP[0], 5)).round(1).to_string())

# ------------------------------------------------------------------ profile of exit + G1
JB, JK = names_F.index("bmax"), names_F.index("k1max")
SUMP = pd.DataFrame([dict(S_d=r["value"], nll=r["nll"], exit_d=1 / 10 ** r["x"][JB] / 24,
                          G1_h=1 / 10 ** r["x"][JK]) for r in profile_table(fam("exit_plus_G1"))])
S_BEST = (1 / F["params"]["bmax"] + 1 / F["params"]["k1max"]) / 24
SUMP = pd.concat([SUMP, pd.DataFrame([dict(S_d=S_BEST, nll=NLL_BEST, exit_d=1 / F["params"]["bmax"] / 24,
                                           G1_h=1 / F["params"]["k1max"])])]).sort_values("S_d")
SUMP["dNLL"] = SUMP.nll - NLL_BEST
SUMP.to_csv(V6 / "profile_exit_plus_G1.csv", index=False, float_format="%.6g")
say("\n=== 7b. Profile of the identifiable combination exit time + G1 time (days) ===")
say(SUMP.round(2).to_string(index=False))

# ------------------------------------------------------------------ numerical robustness
say("\n=== 8. Numerical robustness of the v7 fits (re-scored, parameters unchanged) ===")
m = CTX["all"]
NUM = []
base_pred = {}
for dt, M in ((0.5, 48), (0.25, 48), (0.5, 64), (0.5, 96)):
    row = dict(dt=dt, M=M)
    for v in VARIANTS:
        rm.DT = dt
        pred = rm.Model(v, M=M, protocols=m.PROTS).run(FINAL["all"][v]["params"], m.T_OBS)
        nll, _ = m.nll(pred, m.IDX_TRAIN)
        row[v] = nll
        if (dt, M) == (0.5, 48):
            base_pred[v] = pred
        elif v == "finQ":
            dd = np.abs(pred[..., :4] - base_pred["finQ"][..., :4])
            row.update({f"max_diff_{k}": float(np.nanmax(dd[..., i])) for i, k in enumerate(("out", "inh", "nec", "pim"))})
    rm.DT = DT_FINE
    k = {v: len(m.bounds(v)[0]) + 4 for v in VARIANTS}
    for v in ("delayQ", "instQ", "noQ"):
        row[f"dAIC_{v}"] = 2 * row[v] + 2 * k[v] - (2 * row["finQ"] + 2 * k["finQ"])
    NUM.append(row)
NUM = pd.DataFrame(NUM)
NUM.to_csv(V6 / "numerical_robustness.csv", index=False, float_format="%.6g")
say(NUM.round(2).to_string(index=False))

# ------------------------------------------------------------------ readout versus arrest
def readout_vs_arrest(th, M=M_FINE):
    rm.DT = DT_FINE
    mdl = rm.Model("finQ", M=M, protocols=["reox_d2"])
    S = mdl.initial(th)
    times, fq_sw, fq_tot, finh, width = [], [], [], [], []
    inner = None
    n_steps = int(round(5.5 * 24 / rm.DT))
    for step in range(n_steps + 1):
        t = step * rm.DT
        obs = mdl.observe(S, t, th)[0]
        R, E, G, Q, D, mm = S
        living = (R + E + G + Q)[0]
        if inner is None and t >= 24.0:
            pin = mdl.p_inf(t - 1e-6)
            _, _, r_out, _ = rm.local_p(living[None], np.ones((1, M)), D, np.array([th["a"] * mm[0]]), pin)
            inner = r_out[0] <= obs[1] + 1e-9
        times.append(rm.T0_DAY + t / 24)
        fq_tot.append(obs[4])
        finh.append(max(obs[1] ** 3 - obs[2] ** 3, 0) / max(obs[0] ** 3 - obs[2] ** 3, 1e-12))
        width.append(max(obs[1] - obs[2], 0))
        fq_sw.append(Q[0][inner].sum() / living[inner].sum() if inner is not None and inner.any() else np.nan)
        if step == n_steps:
            break
        te = t + rm.DT / 2
        k1 = mdl.derivs(t, S, th, te)
        k2 = mdl.derivs(t + rm.DT / 2, [x + rm.DT / 2 * z for x, z in zip(S, k1)], th, te)
        k3 = mdl.derivs(t + rm.DT / 2, [x + rm.DT / 2 * z for x, z in zip(S, k2)], th, te)
        k4 = mdl.derivs(t + rm.DT, [x + rm.DT * z for x, z in zip(S, k3)], th, te)
        S = [np.clip(x + rm.DT / 6 * (a + 2 * b + 2 * c + e), 0, None) if i < 5 else
             x + rm.DT / 6 * (a + 2 * b + 2 * c + e) for i, (x, a, b, c, e) in enumerate(zip(S, k1, k2, k3, k4))]
    out = {k: np.asarray(v) for k, v in dict(t=times, fq_sw=fq_sw, fq_tot=fq_tot, finh=finh, width=width).items()}
    i_sw = int(np.argmin(np.abs(out["t"] - 2)))
    lost = int(np.where((out["t"] > 2) & (out["width"] <= 0.2 * out["width"][i_sw]))[0][0])
    out["lost"] = lost
    return out


RA = readout_vs_arrest(F["params"])
say(f"\n=== 9. Readout versus arrest (day-2 reoxygenation) ===")
say(f"arrested at switch {100 * RA['fq_sw'][int(np.argmin(np.abs(RA['t'] - 2.01)))]:.1f}%; readout cleared at day "
    f"{RA['t'][RA['lost']]:.2f} with {100 * RA['fq_sw'][RA['lost']]:.1f}% of switch-region cells still arrested; "
    f"minimum {100 * np.nanmin(RA['fq_sw']):.1f}% at day {RA['t'][int(np.nanargmin(RA['fq_sw']))]:.2f}")
# the same quantity at every supported exit time (robustness of the readout result)
ra_rows = []
for r in exit_all:
    th = CTX["all"].to_theta(np.array(r["x"]), "finQ")
    if r["nll"] - NLL_BEST <= max(SUPPORT, 20.0):
        o = readout_vs_arrest(th)
        ra_rows.append(dict(exit_h=r["value"], dNLL=r["nll"] - NLL_BEST, clear_day=o["t"][o["lost"]],
                            arrested_pct=100 * o["fq_sw"][o["lost"]]))
RA_TAB = pd.DataFrame(ra_rows)
RA_TAB.to_csv(V6 / "readout_vs_arrest_by_exit.csv", index=False, float_format="%.4g")
say(RA_TAB.round(2).to_string(index=False))

# ------------------------------------------------------------------ figures
backup = FIG / "v5_backup"
backup.mkdir(exist_ok=True)
for stem in ("fig03_model_fits", "fig04_exit_profile", "fig05_readout_vs_arrest"):
    for src in (ART / f"{stem}.pdf", FIG / f"{stem}.png"):
        if src.exists() and not (backup / src.name).exists():
            shutil.copy(src, backup / src.name)


def fits_figure(line, pred, data, stem, to_article):
    protocols = [p for p in ("normoxia", "hypoxia", "deox_d2", "reox_d2", "reox_d4") if (data.protocol == p).any()]
    start = {"deox_d2": ("normoxia", [2]), "reox_d2": ("hypoxia", [2]), "reox_d4": ("hypoxia", [2, 4])}
    switch = {"deox_d2": 2, "reox_d2": 2, "reox_d4": 4}
    titles = {"normoxia": "21% O$_2$", "hypoxia": "2% O$_2$", "deox_d2": "21→2% at day 2",
              "reox_d2": "2→21% at day 2", "reox_d4": "2→21% at day 4"}
    observables = [("R_outer", r"$R_{\mathrm{out}}$ (µm)"), ("R_inhib", r"$R_{\mathrm{inh}}$ (µm)"),
                   ("R_nec", r"$R_{\mathrm{nec}}$ (µm)"), ("R_pim", r"$R_{\mathrm{pim}}$ (µm)")]
    fig, axes = plt.subplots(4, len(protocols), figsize=(1.45 * len(protocols) + .0, 7.6), sharex=True,
                             sharey="row", squeeze=False)
    for j, protocol in enumerate(protocols):
        obs = data[data.protocol == protocol]
        if protocol in start:
            base, days = start[protocol]
            obs = pd.concat([data[(data.protocol == base) & data.day.isin(days)], obs])
        g = obs.groupby("day")
        for i, (var, ylabel) in enumerate(observables):
            ax = axes[i, j]
            ax.errorbar(g[var].mean().index, g[var].mean(), yerr=g[var].sem(), fmt="o", color="black", ms=2.8,
                        capsize=1.8, elinewidth=.8, label="data: mean ± SE", zorder=8)
            for v in VARIANTS:
                c = pred[(pred.variant == v) & (pred.protocol == protocol) & (pred.day >= 2)]
                ax.plot(c.day, c[var], color=COL[v], ls=LS[v], lw=1.25, label=NAME[v])
            if protocol in switch:
                ax.axvline(switch[protocol], color="#555555", lw=.6, ls=(0, (3, 2)))
            ax.grid(axis="y", color="#dddddd", lw=.45)
            ax.spines[["top", "right"]].set_visible(False)
            if i == 0:
                ax.set_title(f"({'abcde'[j]}) {titles[protocol]}", pad=5)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == 3:
                ax.set_xlabel("day after seeding")
                ax.set_xticks([2, 4, 6, 8])
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, ncol=3 if len(protocols) < 5 else 5, frameon=False, loc="upper center",
               bbox_to_anchor=(.5, 1.04 if len(protocols) < 5 else 1.015))
    fig.subplots_adjust(top=.93, left=.12 if len(protocols) < 5 else .10, right=.99, bottom=.07, hspace=.13, wspace=.10)
    if line != "WM793b":
        fig.suptitle(line, y=1.08 if len(protocols) < 5 else 1.05, fontweight="bold")
    save(fig, stem, to_article)


fits_figure("WM793b", PRED["WM793b"], d793, "fig03_model_fits", True)
fits_figure("WM164", PRED["WM164"], CTX["WM164"].d, "figS2_fits_WM164", True)
fits_figure("WM983b", PRED["WM983b"], CTX["WM983b"].d, "figS3_fits_WM983b", True)

# Figure 4: exit profile (a) and training/held-out (b)
REJECT = 40.0
fig, axes4 = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
ax = axes4[0]
ax.axhspan(0, SUPPORT, color="#2ca02c", alpha=.12, lw=0)
ax.axhline(REJECT, color="#a51f1f", lw=.8, ls="--")
ax.plot(prof_all.exit_h / 24, np.minimum(prof_all.dNLL, 95), "o-", color="#2878b5", lw=1.7, ms=4.2)
sup = prof_all[prof_all.dNLL <= SUPPORT].exit_h / 24
ax.text(.03, .96, f"support band ΔNLL ≤ {SUPPORT:.1f}\nsupported: {sup.min():.1f}–{sup.max():.1f} d", transform=ax.transAxes,
        va="top", fontsize=7.5, color="#20752a")
ax.text(1.62, REJECT + 1.5, "rejection level", va="bottom", fontsize=7.5, color="#a51f1f")
ax.set_title("(a) all protocols", loc="left", fontweight="bold")
ax = axes4[1]
ax.axhspan(0, SUPPORT, color="#2ca02c", alpha=.12, lw=0)
ax.axhline(REJECT, color="#a51f1f", lw=.8, ls="--")
ax.plot(prof_tr.exit_h / 24, np.minimum(prof_tr.dTrain, 95), "o-", color="#2878b5", lw=1.6, ms=4,
        label="training protocols")
ax.plot(prof_tr.exit_h / 24, np.minimum(prof_tr.dTest, 95), "s--", color="#d62728", lw=1.6, ms=4,
        label="held-out 2→21% at day 4")
ax.set_title("(b) day-4 reoxygenation held out", loc="left", fontweight="bold")
ax.legend(frameon=False, loc="upper center", fontsize=7.2)
for ax, xt in zip(axes4, ([2, 4, 7, 14, 28], [2, 4, 7, 14, 28, 56])):
    ax.set_xscale("log")
    ax.set_xticks(xt)
    ax.set_xticklabels([str(v) for v in xt])
    ax.set_xlim(1.5, 40 if xt[-1] == 28 else 70)
    ax.set_ylim(-2, 97)
    ax.set_xlabel("mean exit time from arrest (days)")
    ax.grid(axis="y", color="#dddddd", lw=.5)
    ax.spines[["top", "right"]].set_visible(False)
axes4[0].set_ylabel(r"$\Delta$ negative log-likelihood")
fig.tight_layout()
save(fig, "fig04_exit_profile")

# Figure 5: readout versus arrest
fig, ax = plt.subplots(figsize=(7.2, 3.8))
t = RA["t"]
ax.plot(t, RA["fq_sw"], color="#9467bd", lw=2.1, label=r"arrested fraction in switch-region shells, $f_Q^{\mathrm{sw}}$")
ax.plot(t, RA["fq_tot"], color="#2878b5", lw=1.4, ls="--", label=r"whole-spheroid arrested fraction, $f_Q$")
ax.plot(t, RA["finh"], color="#d62728", lw=1.8, label=r"image-derived inhibited fraction, $f_{\mathrm{inh}}$")
ob = pd.concat([d793[(d793.protocol == "hypoxia") & (d793.day == 2)], d793[d793.protocol == "reox_d2"]]).copy()
ob["f_inh"] = np.maximum(ob.R_inhib**3 - ob.R_nec**3, 0) / np.maximum(ob.R_outer**3 - ob.R_nec**3, 1e-12)
g = ob.groupby("day").f_inh
ax.errorbar(g.mean().index, g.mean(), yerr=g.sem(), fmt="o", ms=4, capsize=2, color="#d62728", mfc="white",
            label=r"observed $f_{\mathrm{inh}}$")
L = RA["lost"]
ax.axvline(t[L], color="#555555", lw=.8, ls=":")
ax.scatter([t[L]], [RA["fq_sw"][L]], s=28, color="#9467bd", zorder=7)
ax.annotate(f"readout cleared at day {t[L]:.1f}\n{100 * RA['fq_sw'][L]:.0f}% still arrested",
            xy=(t[L], RA["fq_sw"][L]), xytext=(3.55, .27), arrowprops=dict(arrowstyle="->", lw=.8, color="#555555"),
            fontsize=8.2)
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

# Figure 6 and S1: identifiability
VCOL = {"identified": "#1f5fa8", "one-sided": "#d9822b", "not identified": "#7f7f7f"}
SHORT = {"bmax": "exit time", "k1max": "G1 time", "amax": "entry time", "Kq": "arrest threshold", "hq": "Hill exponent",
         "K1": "G1 half-saturation", "a": "consumption ratio", "m_h": "hypoxic consumption", "tau_m": "adaptation time",
         "mu0": "basal loss", "mun": "anoxic death", "Kn": "death threshold", "kD": "clearance, 2%",
         "kD_ox": "clearance, 21%", "R1_norm": "day-1 radius, 21%", "R1_hyp": "day-1 radius, 2%", "p_pim": "PIM threshold"}


def profile_panel(ax, n, letter=None):
    df = UNI[n]
    lab, tf, logx = DISPLAY[n]
    j = names_F.index(n)
    xv = np.array([tf(v) for v in df.value])
    order = np.argsort(xv)
    xv, yv = xv[order], df.dNLL.values[order]
    b1, b2 = sorted((tf(10 ** lo_F[j]), tf(10 ** hi_F[j])))
    ax.axhspan(0, SUPPORT, color="#2ca02c", alpha=.12, lw=0)
    ax.axhline(REJECT, color="#a51f1f", lw=.7, ls="--")
    ax.plot(xv, np.minimum(yv, 95), "-", color=VCOL[VERDICT[n]["verdict"]], lw=1.6)
    ax.plot(xv, np.minimum(yv, 95), "o", color=VCOL[VERDICT[n]["verdict"]], ms=3.2)
    xm = tf(F["params"][n])
    ax.plot([xm], [0], "D", color="black", ms=4.2, zorder=6)
    if logx:
        ax.set_xscale("log")
        span = np.log10(max(xv.max(), xm) / min(xv.min(), xm))
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1,) if span > 1.2 else (1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        if n == "bmax":
            ax.set_xticks([2, 4, 7, 14, 28])
    pad = (b2 / b1) ** .04 if logx else (b2 - b1) * .03
    xlo, xhi = (min(xv.min(), xm) / pad, max(xv.max(), xm) * pad) if logx else (min(xv.min(), xm) - pad, max(xv.max(), xm) + pad)
    for edge in (b1, b2):  # shade the region beyond a search bound
        if xlo <= edge <= xhi:
            if edge == b1:
                ax.axvspan(xlo, edge, color="#bbbbbb", alpha=.35, lw=0)
            else:
                ax.axvspan(edge, xhi, color="#bbbbbb", alpha=.35, lw=0)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-2, 97)
    ax.set_xlabel(lab, fontsize=7.6)
    ttl = VERDICT[n]["verdict"]
    ax.set_title((f"({letter}) " if letter else "") + f"{SHORT[n]}: {ttl}", loc="left", fontsize=7.8,
                 color=VCOL[ttl], fontweight="bold")
    ax.grid(axis="y", color="#e2e2e2", lw=.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


KEY = ["bmax", "k1max", "Kq", "amax", "hq", "m_h", "tau_m", "a", "p_pim"]
fig = plt.figure(figsize=(7.2, 8.4))
gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.35], hspace=.75, wspace=.32)
for k, n in enumerate(KEY):
    ax = fig.add_subplot(gs[k // 3, k % 3])
    profile_panel(ax, n, "abcdefghi"[k])
    if k % 3 == 0:
        ax.set_ylabel(r"$\Delta$NLL")
cmap = LinearSegmentedColormap.from_list("prof", ["#1f5fa8", "#9ecae1", "#fff5d6", "#f4a261", "#a51f1f"])


def heatmap(ax, xs, ys, Z, xl, yl, ttl, tx, ty):
    im_ = ax.imshow(np.minimum(Z, 100), origin="lower", cmap=cmap, vmin=0, vmax=100, aspect="auto")
    for (a_, b_), val in np.ndenumerate(Z):
        if np.isfinite(val):
            ax.text(b_, a_, f"{val:.0f}", ha="center", va="center", fontsize=6.4,
                    color="white" if val > 60 or val < 8 else "black")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([tx(v) for v in xs], fontsize=7)
    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels([ty(v) for v in ys], fontsize=7)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(ttl, loc="left", fontsize=8.5, fontweight="bold")
    return im_


ax_j = fig.add_subplot(gs[3, 0:2])
im = heatmap(ax_j, EG[0] / 24, EG[1], EG[2], "mean exit time (d)", "normoxic G1 time (h)",
             "(j) exit time × G1 duration", lambda v: f"{v:.1f}", lambda v: f"{v:g}")
# (k) profile of the identifiable combination exit + G1
ax = fig.add_subplot(gs[3, 2])
sx, sy = SUMP.S_d.values, SUMP.dNLL.values
ax.axhspan(0, SUPPORT, color="#2ca02c", alpha=.12, lw=0)
ax.axhline(REJECT, color="#a51f1f", lw=.7, ls="--")
ok = sy <= SUPPORT
s_lo, s_hi = sx[ok].min(), sx[ok].max()
both = (sy[sx < s_lo].max() if (sx < s_lo).any() else 0) > SUPPORT and (sy[sx > s_hi].max() if (sx > s_hi).any() else 0) > SUPPORT
ax.plot(sx, np.minimum(sy, 95), "-o", color=VCOL["identified" if both else "one-sided"], lw=1.6, ms=3.2)
ax.plot([S_BEST], [0], "D", color="black", ms=4.2, zorder=6)
ax.set_ylim(-2, 97)
ax.set_xlabel("exit time + G1 time (d)", fontsize=7.6)
ax.set_title(f"(k) exit + G1: {'identified' if both else 'one-sided'}", loc="left", fontsize=7.8,
             color=VCOL["identified" if both else "one-sided"], fontweight="bold")
ax.text(.97, .50, f"supported\n{s_lo:.1f}–{s_hi:.1f} d", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=7, color="#20752a")
ax.grid(axis="y", color="#e2e2e2", lw=.45)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=7)
cb = fig.colorbar(im, ax=ax_j, fraction=.035, pad=.015)
cb.ax.set_title(r"$\Delta$NLL", fontsize=6.5)
cb.ax.tick_params(labelsize=6)
handles = [plt.Line2D([], [], color=VCOL[k], lw=2, label=k) for k in VCOL] + [
    plt.Line2D([], [], color="black", marker="D", ls="", ms=4, label="estimate"),
    plt.Rectangle((0, 0), 1, 1, color="#2ca02c", alpha=.25, label=f"support band (ΔNLL ≤ {SUPPORT:.1f})"),
    plt.Line2D([], [], color="#a51f1f", ls="--", lw=.8, label="strong rejection (ΔNLL = 40)"),
    plt.Rectangle((0, 0), 1, 1, color="#bbbbbb", alpha=.5, label="outside search bounds")]
fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, .995), fontsize=7)
fig.subplots_adjust(top=.92, left=.08, right=.925, bottom=.06)
save(fig, "fig06_identifiability")

fig = plt.figure(figsize=(7.2, 9.6))
gs1 = fig.add_gridspec(5, 4, hspace=.85, wspace=.38)
order_all = ["bmax", "k1max", "amax", "Kq", "hq", "K1", "a", "m_h", "tau_m", "mu0", "mun", "Kn", "kD", "kD_ox",
             "R1_norm", "R1_hyp", "p_pim"]
for k, n in enumerate(order_all):
    ax = fig.add_subplot(gs1[k // 4, k % 4])
    profile_panel(ax, n)
    if k % 4 == 0:
        ax.set_ylabel(r"$\Delta$NLL")
im = heatmap(fig.add_subplot(gs1[4, 2:4]), AP[0], AP[1], AP[2], r"consumption ratio $a$ ($\times10^{-3}$)",
             r"$p_{\mathrm{pim}}$ (% O$_2$)", r"$a$ × $p_{\mathrm{pim}}$ (ΔNLL)", lambda v: f"{1e3 * v:.1f}",
             lambda v: f"{v:g}")
fig.legend(handles=handles, ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(.5, .995), fontsize=7)
fig.subplots_adjust(top=.93, left=.08, right=.98, bottom=.05)
save(fig, "figS1_all_profiles")

# ------------------------------------------------------------------ LaTeX tables
best_aic = min(e["AIC"] for e in FINAL["all"].values())
lines = [r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Model & NLL & AIC & $\Delta$AIC & $\sigma_{\mathrm{out}}$ & $\sigma_{\mathrm{inh}}$ & $\sigma_{\mathrm{nec}}$ & $\sigma_{\mathrm{pim}}$ \\",
         r"\midrule"]
for v in VARIANTS:
    e = FINAL["all"][v]
    s = list(e["sd"].values())
    lines.append(f"{SYM[v]} & {e['nll_train']:.1f} & {e['AIC']:.1f} & {e['AIC'] - best_aic:.1f} & "
                 f"{s[0]:.1f} & {s[1]:.1f} & {s[2]:.1f} & {s[3]:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(V6 / "table_model_comparison.tex").write_text("\n".join(lines))

TEXNAME = {"k1max": r"$k_{1,\max}$ (h$^{-1}$)", "K1": r"$K_1$ (\% O$_2$)", "a": r"$a$ (\% O$_2$~\textmu m$^{-2}$)",
           "amax": r"$\alpha_{\max}$ (h$^{-1}$)", "bmax": r"$\beta_{\max}$ (h$^{-1}$)", "Kq": r"$K_q$ (\% O$_2$)",
           "hq": r"$h_q$", "mu0": r"$\mu_0$ (h$^{-1}$)", "mun": r"$\mu_{n,\max}$ (h$^{-1}$)",
           "Kn": r"$K_n$ (\% O$_2$)", "kD": r"$k_{D,h}$ (h$^{-1}$)", "kD_ox": r"$k_{D,o}$ (h$^{-1}$)",
           "m_h": r"$m_h$", "tau_m": r"$\tau_m$ (h)", "R1_norm": r"$\rho_{\mathrm{norm}}$ (\textmu m)",
           "R1_hyp": r"$\rho_{\mathrm{hyp}}$ (\textmu m)", "p_pim": r"$p_{\mathrm{pim}}$ (\% O$_2$)"}
lines = [r"\begin{tabular}{@{}lrrrl@{}}", r"\toprule",
         r"Parameter & Estimate & Supported range & Search interval & Profile verdict \\", r"\midrule"]
fmt = lambda v: "--" if v is None else f"{v:.3g}"
for n in names_F:
    vd = VERDICT[n]
    lo_txt = fmt(vd["low"]) if vd["low"] is not None else (r"bound" if vd["at_lo"] else r"$\le$ bound")
    hi_txt = fmt(vd["high"]) if vd["high"] is not None else (r"bound" if vd["at_hi"] else r"$\ge$ bound")
    dag = r"$^{\dagger}$" if n in F["at_bound"] else ""
    lines.append(f"{TEXNAME[n]} & {vd['mle']:.4g}{dag} & [{lo_txt}, {hi_txt}] & "
                 f"[{vd['lo_bound']:.3g}, {vd['hi_bound']:.3g}] & {vd['verdict']} \\\\")
okS = SUMP.dNLL.values <= SUPPORT
lines.append(r"\addlinespace")
lines.append(f"exit + G1 time (d) & {S_BEST:.2f} & [{SUMP.S_d.values[okS].min():.2f}, {SUMP.S_d.values[okS].max():.2f}] & "
             r"-- & derived combination \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(V6 / "table_parameters.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Fixed choice & $\Delta\mathrm{AIC}_I$ & $\Delta\mathrm{AIC}_0$ & Best exit (d) & G1 (h) & "
         r"$\Delta\mathrm{NLL}_{4\mathrm{d}}$ & $\Delta\mathrm{NLL}_{17\mathrm{d}}$ \\", r"\midrule"]
lab = {"baseline": r"Baseline ($\varepsilon_Q=0.5$, 10-h S/G2/M)", "epsQ_0.25": r"$\varepsilon_Q=0.25$",
       "epsQ_1.00": r"$\varepsilon_Q=1$", "k2_0.07": "14-h S/G2/M", "k2_0.14": "7-h S/G2/M"}
for _, r in SENS.iterrows():
    lines.append(f"{lab[r.scenario]} & {r.dAIC_I:.1f} & {r.dAIC_0:.1f} & {r.exit_d:.2f} & {r.G1_h:.1f} & "
                 f"{r.dNLL_4d:.1f} & {r.dNLL_17d:.1f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(V6 / "table_sensitivity.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{@{}rrrrrrrrr@{}}", r"\toprule",
         r"$\Delta t$ (h) & $M$ & $\Delta\mathrm{AIC}_D$ & $\Delta\mathrm{AIC}_I$ & $\Delta\mathrm{AIC}_0$ & "
         r"$|\Delta\Ro|$ & $|\Delta\Ri|$ & $|\Delta\Rn|$ & $|\Delta\Rp|$ \\", r"\midrule"]
for _, r in NUM.iterrows():
    diffs = " & ".join("--" if pd.isna(r.get(f"max_diff_{k}")) else f"{r[f'max_diff_{k}']:.1f}"
                       for k in ("out", "inh", "nec", "pim"))
    lines.append(f"{r['dt']:.2f} & {int(r['M'])} & {r['dAIC_delayQ']:.1f} & {r['dAIC_instQ']:.1f} & {r['dAIC_noQ']:.1f} & {diffs} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(V6 / "table_numerical.tex").write_text("\n".join(lines))

lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Cell line & Model & NLL & $\Delta$AIC & G1 (h) & $\sigma_{\mathrm{inh}}$ (\textmu m) \\", r"\midrule"]
for ctx in ("WM164", "WM983b"):
    ba = min(e["AIC"] for e in FINAL[ctx].values())
    for k, v in enumerate(VARIANTS):
        e = FINAL[ctx][v]
        lines.append(f"{ctx if k == 0 else ''} & {SYM[v]} & {e['nll_train']:.1f} & {e['AIC'] - ba:.1f} & "
                     f"{1 / e['params']['k1max']:.1f} & {e['sd']['R_inhib']:.1f} \\\\")
    if ctx == "WM164":
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
(V6 / "table_other_lines.tex").write_text("\n".join(lines))

(V6 / "summary.txt").write_text("\n".join(summary) + "\n")
print("\nwritten: results/v7/{final_fits.json, summary.txt, tables, profiles}; figures 3-6, S1-S3")
