"""Tier 2: time scales of arrest exit (after 2%->21%) and entry (after 21%->2%).

Kinetic families (tau = time since the oxygen switch, days)
  * Erlang(k): the waiting time before a cell changes state has k exponential
    stages with mean T (k = 1 is memoryless);
  * delayed exponential: no transition before a delay D, then rate 1/T
    (mean waiting time D + T);
  * instantaneous: the oxygen-threshold hypothesis (T -> 0).

Exit: arrested cells A leave with the waiting-time law above; cycling cells C
grow at the normoxic living-volume rate r and receive the exited cells;
f = A / (A + C).
Entry: f = f_pre + (f_inf - f_pre) F(tau), F the waiting-time CDF.

Noise: heteroscedastic Gaussian, var = s^2 f(1-f) + sigma0^2 (s maximised out).
Uncertainty: profile likelihood (chi^2_1, 1.92) and a non-parametric bootstrap
that resamples spheroids within each measured condition (independent of the
noise model).

Outputs: results/switch_kinetics_fits.csv, results/switch_kinetics_bootstrap.csv,
         results/switch_kinetics_profiles.csv, figures/fig03_switch_kinetics.png
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import brentq, minimize
from scipy.stats import gamma as gamma_dist

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "data" / "processed" / "spheroids.csv")
SIGMA0 = 0.005
THRESH = 1.92
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 200
S_GRID = np.geomspace(0.02, 2.0, 160)[:, None]
ERLANG_K = (1, 2, 3, 4, 6, 8)


def loglik(f_model, f_obs, sigma0=SIGMA0):
    fm = np.clip(f_model, 1e-9, 1 - 1e-9)[None, :]
    var = S_GRID**2 * fm * (1 - fm) + sigma0**2
    ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (f_obs[None, :] - fm) ** 2 / var, axis=1)
    return ll.max()


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


# ------------------------------------------------------------ waiting-time laws
# A model is (name, n_time_params); time params are log-transformed.
def exit_curve(tau, f0, model, tp, r):
    tau = np.atleast_1d(tau).astype(float)
    name = model[0]
    if name == "instant":
        return np.where(tau > 0, 0.0, f0)
    if name == "delay":
        D, T = tp
        s = np.clip(tau - D, 0, None)
        A = f0 * np.exp(-s / T)
        C = (1 - f0) * np.exp(r * tau) + f0 / (1 + r * T) * (np.exp(r * s) - np.exp(-s / T))
        return A / (A + C)
    k = model[1]
    T = tp[0]
    M = np.zeros((k + 1, k + 1))
    rate = k / T
    for j in range(k):
        M[j, j] = -rate
        if j:
            M[j, j - 1] = rate
    M[k, k - 1], M[k, k] = rate, r
    x0 = np.zeros(k + 1)
    x0[0], x0[k] = f0, 1 - f0
    uniq, inv = np.unique(tau, return_inverse=True)
    out = np.empty(uniq.size)
    for i, t in enumerate(uniq):
        x = expm(M * t) @ x0
        out[i] = x[:k].sum() / x.sum()
    return out[inv]


def waiting_cdf(tau, model, tp):
    tau = np.atleast_1d(tau).astype(float)
    if model[0] == "instant":
        return (tau > 0).astype(float)
    if model[0] == "delay":
        D, T = tp
        return 1 - np.exp(-np.clip(tau - D, 0, None) / T)
    k = model[1]
    return gamma_dist.cdf(tau, a=k, scale=tp[0] / k)


def mean_time(model, tp):
    return 0.0 if model[0] == "instant" else float(np.sum(tp))


def n_time(model):
    return {"instant": 0, "delay": 2}.get(model[0], 1)


# ------------------------------------------------------------ objective + fit
def exit_nll(theta, model, r, branches, fixed_mean=None):
    nt = n_time(model)
    tp = np.exp(theta[:nt])
    if fixed_mean is not None:  # re-parametrise so that the mean time is fixed
        tp = fixed_tp(model, theta[:nt], fixed_mean)
    f0s = logistic(theta[nt:])
    fm = np.concatenate([exit_curve(t, f0, model, tp, r) for (t, _), f0 in zip(branches, f0s)])
    return -loglik(fm, np.concatenate([f for _, f in branches]))


def entry_nll(theta, model, data, fixed_mean=None):
    tau, f, f_pre = data
    nt = n_time(model)
    tp = np.exp(theta[:nt]) if fixed_mean is None else fixed_tp(model, theta[:nt], fixed_mean)
    f_inf = logistic(theta[nt])
    return -loglik(f_pre + (f_inf - f_pre) * waiting_cdf(tau, model, tp), f)


def fixed_tp(model, free, mean):
    """Time parameters with a prescribed mean (for profiles)."""
    if model[0] == "delay":  # free[0] = logit of the delay share of the mean
        share = logistic(free[0])
        return np.array([share * mean, (1 - share) * mean])
    return np.array([mean])


def fit(nll, x0, restarts=((0.0,), (0.7,), (-0.7,))):
    best = None
    for dx in restarts:
        res = minimize(nll, np.asarray(x0) + dx[0], method="Nelder-Mead",
                       options=dict(xatol=1e-7, fatol=1e-9, maxiter=6000))
        if best is None or res.fun < best.fun:
            best = res
    return best


def fit_exit(model, r, branches, fixed_mean=None):
    nt = n_time(model)
    if fixed_mean is not None:
        if model[0] == "delay":  # th = [logit delay share, f0 logits...]
            x0 = np.r_[0.0, np.full(len(branches), -1.0)]
            fun = lambda th: exit_nll(np.r_[th[0], 0.0, th[1:]], model, r, branches, fixed_mean)
        else:                    # th = [f0 logits...]
            x0 = np.full(len(branches), -1.0)
            fun = lambda th: exit_nll(np.r_[0.0, th], model, r, branches, fixed_mean)
        return fit(fun, x0)
    x0 = np.r_[np.log(np.full(nt, 0.5)), np.full(len(branches), -1.0)]
    return fit(lambda th: exit_nll(th, model, r, branches), x0)


def fit_entry(model, data, fixed_mean=None):
    nt = n_time(model)
    if fixed_mean is not None:
        if model[0] == "delay":  # th = [logit delay share, logit f_inf]
            x0 = np.array([0.0, -0.5])
            fun = lambda th: entry_nll(np.r_[th[0], 0.0, th[1]], model, data, fixed_mean)
        else:                    # th = [logit f_inf]
            x0 = np.array([-0.5])
            fun = lambda th: entry_nll(np.r_[0.0, th], model, data, fixed_mean)
        return fit(fun, x0)
    x0 = np.r_[np.log(np.full(nt, 0.5)), -0.5]
    return fit(lambda th: entry_nll(th, model, data), x0)


def estimate(direction, model, data, r=None):
    """Return (nll, mean time in days, fitted result)."""
    if direction == "exit":
        res = fit_exit(model, r, data)
    else:
        res = fit_entry(model, data)
    nt = n_time(model)
    return res.fun, mean_time(model, np.exp(res.x[:nt])), res


def profile_ci(direction, model, data, r, nll_min, t_hat):
    def prof(mean):
        res = fit_exit(model, r, data, mean) if direction == "exit" else fit_entry(model, data, mean)
        return res.fun - nll_min - THRESH

    def side(a, b):
        try:
            if prof(a) * prof(b) > 0:
                return np.nan
            return brentq(prof, a, b, xtol=1e-4)
        except ValueError:
            return np.nan
    return side(t_hat / 50, t_hat), side(t_hat, min(t_hat * 50, 30.0))


# ------------------------------------------------------------ data sets
def grab(line, prot, days):
    return d[(d.cell_line == line) & (d.protocol == prot) & d.day.isin(days)]


def normoxic_rate(line):
    s = d[(d.cell_line == line) & (d.protocol == "normoxia")]
    return np.polyfit(s.day, np.log(s.V_living), 1)[0]


EXIT_GROUPS = {"WM793b": [
    [("hypoxia", 2, 2.0), ("reox_d2", 3, 2.0), ("reox_d2", 4, 2.0), ("reox_d2", 6, 2.0)],
    [("hypoxia", 4, 4.0), ("reox_d4", 6, 4.0), ("reox_d4", 8, 4.0)],
]}
ENTRY_GROUPS = {line: [("deox_d2", day, 2.0) for day in (3, 4, 6, 8)]
                for line in ("WM793b", "WM164", "WM983b")}


def build_exit(line, rng=None):
    branches = []
    for grp in EXIT_GROUPS[line]:
        taus, fs = [], []
        for prot, day, ts in grp:
            s = grab(line, prot, [day]).f_arrested.values
            if rng is not None:
                s = rng.choice(s, s.size)
            taus.append(np.full(s.size, day - ts))
            fs.append(s)
        branches.append((np.concatenate(taus), np.concatenate(fs)))
    return branches


def build_entry(line, rng=None):
    taus, fs = [], []
    for prot, day, ts in ENTRY_GROUPS[line]:
        s = grab(line, prot, [day]).f_arrested.values
        if rng is not None:
            s = rng.choice(s, s.size)
        taus.append(np.full(s.size, day - ts))
        fs.append(s)
    pre = grab(line, "normoxia", [2]).f_arrested.values
    if rng is not None:
        pre = rng.choice(pre, pre.size)
    return np.concatenate(taus), np.concatenate(fs), pre.mean()


MODELS = [("erlang", k) for k in ERLANG_K] + [("delay",)]


def label(m):
    return f"Erlang k={m[1]}" if m[0] == "erlang" else ("delayed exp." if m[0] == "delay" else m[0])


# ============================================================ main fits
rows, prof_rows = [], []
cases = [("exit", "WM793b", g) for g in ("r_hat", "r=0", "2r_hat")] + \
        [("entry", l, None) for l in ("WM793b", "WM164", "WM983b")]
best_models = {}
for direction, line, growth in cases:
    if direction == "exit":
        r = {"r_hat": 1.0, "r=0": 0.0, "2r_hat": 2.0}[growth] * normoxic_rate(line)
        data = build_exit(line)
        n_obs = sum(len(b[1]) for b in data)
        n_extra = len(data)
    else:
        r, data = None, build_entry(line)
        n_obs, n_extra = len(data[1]), 1
    inst_nll = estimate(direction, ("instant",), data, r)[0]
    fits = {}
    for m in MODELS:
        nll, T, res = estimate(direction, m, data, r)
        fits[m] = (nll, T, res)
        npar = n_time(m) + n_extra + 1
        row = dict(direction=direction, cell_line=line, growth=growth, model=label(m),
                   mean_time_h=24 * T, nll=nll, AIC=2 * nll + 2 * npar, n=n_obs,
                   LRT_vs_instant=2 * (inst_nll - nll))
        if m[0] == "delay":
            tp = np.exp(res.x[:2])
            row.update(delay_h=24 * tp[0], rate_time_h=24 * tp[1])
        if direction == "entry":
            row["f_inf"] = logistic(res.x[n_time(m)])
        rows.append(row)
    rows.append(dict(direction=direction, cell_line=line, growth=growth, model="instantaneous",
                     mean_time_h=0.0, nll=inst_nll, AIC=2 * inst_nll + 2 * (n_extra + 1), n=n_obs))
    if growth in (None, "r_hat"):
        mbest = min(fits, key=lambda m: 2 * fits[m][0] + 2 * n_time(m))
        best_models[(direction, line)] = mbest
        for m in sorted({("erlang", 1), mbest}):
            nll, T, _ = fits[m]
            lo, hi = profile_ci(direction, m, data, r, nll, T)
            rows.append(dict(direction=direction, cell_line=line, growth=growth, model=label(m),
                             mean_time_h=24 * T, CI_lo_h=24 * lo, CI_hi_h=24 * hi,
                             note="profile-likelihood 95% CI of the mean waiting time"))
            for mean in np.geomspace(max(T / 20, 0.01), min(T * 20, 30), 30):
                res = (fit_exit(m, r, data, mean) if direction == "exit"
                       else fit_entry(m, data, mean))
                prof_rows.append(dict(direction=direction, cell_line=line, model=label(m),
                                      mean_time_h=24 * mean, delta_nll=res.fun - nll))

res_df = pd.DataFrame(rows)
res_df.to_csv(ROOT / "results" / "switch_kinetics_fits.csv", index=False, float_format="%.5g")
pd.DataFrame(prof_rows).to_csv(ROOT / "results" / "switch_kinetics_profiles.csv",
                               index=False, float_format="%.5g")
with pd.option_context("display.width", 250, "display.max_rows", 300):
    print(res_df.round(3).to_string(index=False))

# ============================================================ bootstrap
rng = np.random.default_rng(2026)
boot = []
for b in range(N_BOOT):
    for (direction, line), mbest in best_models.items():
        r = normoxic_rate(line) if direction == "exit" else None
        data = build_exit(line, rng) if direction == "exit" else build_entry(line, rng)
        for m in sorted({("erlang", 1), mbest}):
            nll, T, res = estimate(direction, m, data, r)
            rec = dict(rep=b, direction=direction, cell_line=line, model=label(m), mean_time_h=24 * T)
            if m[0] == "delay":
                rec["delay_h"] = 24 * np.exp(res.x[0])
            boot.append(rec)
boot = pd.DataFrame(boot)
boot.to_csv(ROOT / "results" / "switch_kinetics_bootstrap.csv", index=False, float_format="%.5g")
q = boot.groupby(["direction", "cell_line", "model"]).agg(
    median_h=("mean_time_h", "median"),
    lo_h=("mean_time_h", lambda x: np.percentile(x, 2.5)),
    hi_h=("mean_time_h", lambda x: np.percentile(x, 97.5)))
if "delay_h" in boot:
    q = q.join(boot.groupby(["direction", "cell_line", "model"]).delay_h.agg(
        delay_lo_h=lambda x: np.percentile(x.dropna(), 2.5) if x.notna().any() else np.nan,
        delay_hi_h=lambda x: np.percentile(x.dropna(), 97.5) if x.notna().any() else np.nan))
print("\nBootstrap (", N_BOOT, "replicates ) 95% intervals of the mean waiting time")
print(q.round(1).to_string())
q.to_csv(ROOT / "results" / "switch_kinetics_bootstrap_summary.csv", float_format="%.4g")

# ============================================================ figure
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
jit = np.random.default_rng(0)
ax = axes[0]
line = "WM793b"
data = build_exit(line)
r_hat = normoxic_rate(line)
mb = best_models[("exit", line)]
tt = np.linspace(0, 4, 200)
for (tau, f), col, lab in zip(data, ["#d62728", "#9467bd"], ["switch at day 2", "switch at day 4"]):
    ax.plot(tau + jit.normal(0, 0.04, tau.size), f, "o", ms=3, color=col, alpha=0.45, label=lab)
for m, ls in [(("erlang", 1), "--"), (mb, "-")]:
    _, _, res = estimate("exit", m, data, r_hat)
    nt = n_time(m)
    for f0, col in zip(logistic(res.x[nt:]), ["#d62728", "#9467bd"]):
        ax.plot(tt, exit_curve(tt, f0, m, np.exp(res.x[:nt]), r_hat), ls, color=col, lw=1.3)
    ax.plot([], [], "k" + ls, label=label(m))
ax.set_title("WM793b: arrest exit after 2%→21%")
ax.set_xlabel("time since reoxygenation (days)")
ax.set_ylabel("arrested fraction of living volume")
ax.legend(fontsize=7, frameon=False)

for ax, line in zip(axes[1:], ["WM983b", "WM164"]):
    data = build_entry(line)
    tau, f, f_pre = data
    ax.plot(tau + jit.normal(0, 0.04, tau.size), f, "o", ms=3, color="#2ca02c", alpha=0.45,
            label="21%→2% at day 2")
    tt = np.linspace(0, 6, 300)
    mb = best_models[("entry", line)]
    for m, ls in [(("erlang", 1), "--"), (mb, "-")]:
        _, _, res = estimate("entry", m, data)
        nt = n_time(m)
        ax.plot(tt, f_pre + (logistic(res.x[nt]) - f_pre) * waiting_cdf(tt, m, np.exp(res.x[:nt])),
                ls, color="k", lw=1.2, label=label(m))
    ax.set_title(f"{line}: arrest entry after 21%→2%")
    ax.set_xlabel("time since deoxygenation (days)")
    ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(ROOT / "figures" / "fig03_switch_kinetics.png", dpi=200)
