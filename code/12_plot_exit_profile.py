"""Figure: profile likelihood of the mean arrest-exit time (radial model).

For each fixed exit time, the best NLL over starts after re-optimising every
other parameter with G1 free in 5-120 h (11_exit_g1_confounding.py). Two G1
laws are compared: memoryless (exponential, files without "erlang") and an
Erlang-4 G1 with the same mean (minimal G1 duration, files "*_erlangG1_*").

Usage: python 12_plot_exit_profile.py LINE
Output: figures/fig08_exit_profile_<LINE>.png, results/exit_profile_<LINE>.csv
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
files = sorted((ROOT / "results").glob(f"exit_g1_confounding_{LINE}*_all.csv"))
groups = {"all protocols (final fit)": [f for f in files if "alldata" in f.name],
          "training only, exponential G1": [f for f in files
                                            if "erlang" not in f.name and "alldata" not in f.name],
          "training only, Erlang-4 G1": [f for f in files if "erlangG1" in f.name]}

rows = []
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax2 = ax.twinx()
for (lab, fs), col in zip(groups.items(), ["k", "#d62728", "#1f77b4"]):
    if not fs:
        continue
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    best = df.loc[df.groupby("exit_time_h").nll.idxmin()].sort_values("exit_time_h").copy()
    best["delta_nll"] = best.nll - best.nll.min()
    best["g1_law"] = lab
    rows.append(best)
    ax.plot(best.exit_time_h, best.delta_nll, "o-", color=col, label=lab)
    ax2.plot(best.exit_time_h, best.g1_fit_h, "s:", color=col, ms=3, alpha=0.6)
    inside = best[best.delta_nll <= 1.92]
    print(lab)
    print(best[["exit_time_h", "nll", "delta_nll", "g1_fit_h"]].round(2).to_string(index=False))
    print(f"  grid points within the 95% threshold: {inside.exit_time_h.tolist()}")
out = pd.concat(rows, ignore_index=True)
out.to_csv(ROOT / "results" / f"exit_profile_{LINE}.csv", index=False, float_format="%.5g")

ax.axhline(1.92, ls="--", color="grey", lw=0.8)
ax.set_xscale("log")
ax.set_yscale("symlog", linthresh=2)
ax.set_xlabel("mean exit time from hypoxic arrest (h)")
ax.set_ylabel("Δ negative log-likelihood (solid)")
ax2.set_ylabel("re-estimated G1 duration, h (dotted)")
ax2.set_ylim(0, 120)
ax.legend(fontsize=8, frameon=False, loc="upper center")
ax.set_title(f"{LINE}: profile of the exit time, G1 free in 5-120 h", fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "figures" / f"fig08_exit_profile_{LINE}.png", dpi=200)
