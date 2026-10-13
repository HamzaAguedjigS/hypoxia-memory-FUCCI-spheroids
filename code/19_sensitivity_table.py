"""Build the supplementary fixed-choice sensitivity table after script 17."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

df = pd.read_csv(RESULTS / "sensitivity_fixed_choices.csv")
base = json.loads((RESULTS / "radial_fit_WM793b_alldata.json").read_text())
rows = []
for variant in ("finQ", "instQ", "noQ"):
    x = base[variant]
    rows.append(dict(scenario="baseline", eps_Q=.5, **{"k2_h-1": .1}, variant=variant,
                     nll=x["nll_train"], AIC=x["AIC"],
                     exit_days=(1 / x["params"]["bmax"] / 24 if variant == "finQ" else None)))
all_rows = pd.concat([pd.DataFrame(rows), df], ignore_index=True)

# Merge the exit-time profiles of 20_exit_profile_sensitivity.py: keep the lowest finQ NLL
# found by either script, and record the profile at short (96 h) and long (400 h) exits.
prof_file = RESULTS / "exit_profile_sensitivity_WM793b.csv"
all_rows["dNLL_exit_4d"] = float("nan")
all_rows["dNLL_exit_17d"] = float("nan")
if prof_file.exists():
    prof = pd.read_csv(prof_file)
    for scenario in all_rows.scenario.unique():
        p = prof[(prof.setting == scenario) & (prof.variant == "finQ")]
        if p.empty:
            continue
        i = all_rows.index[(all_rows.scenario == scenario) & (all_rows.variant == "finQ")][0]
        best_p = p.loc[p.nll.idxmin()]
        if best_p.nll < all_rows.at[i, "nll"]:
            all_rows.at[i, "nll"] = best_p.nll
            all_rows.at[i, "AIC"] = best_p.AIC
            all_rows.at[i, "exit_days"] = best_p.exit_h / 24
        fixed = p[p.stage == "fixed_exit"].set_index("exit_fixed_h").nll
        all_rows.at[i, "dNLL_exit_4d"] = fixed.get(96.0, float("nan")) - all_rows.at[i, "nll"]
        all_rows.at[i, "dNLL_exit_17d"] = fixed.get(400.0, float("nan")) - all_rows.at[i, "nll"]
all_rows["delta_AIC"] = all_rows.AIC - all_rows.groupby("scenario").AIC.transform("min")
all_rows.to_csv(RESULTS / "sensitivity_fixed_choices_with_baseline.csv", index=False,
                float_format="%.6g")

scenario_labels = {
    "baseline": r"$\varepsilon_Q=0.50,\ k_2=0.10$",
    "epsQ_0.25": r"$\varepsilon_Q=0.25,\ k_2=0.10$",
    "epsQ_1.00": r"$\varepsilon_Q=1.00,\ k_2=0.10$",
    "k2_0.07": r"$\varepsilon_Q=0.50,\ k_2=0.07$",
    "k2_0.14": r"$\varepsilon_Q=0.50,\ k_2=0.14$",
}
symbols = {"finQ": r"$\mathcal{M}_F$", "instQ": r"$\mathcal{M}_I$",
           "noQ": r"$\mathcal{M}_0$"}
lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Fixed choices & Model & NLL & AIC & $\Delta$AIC & Exit time (d)"
         r" & $\Delta\ell_{\mathrm{p}}$(4 d) & $\Delta\ell_{\mathrm{p}}$(17 d) \\",
         r"\midrule"]


def fmt(value, digits=1):
    return f"{value:.{digits}f}" if pd.notna(value) else "--"


for scenario in scenario_labels:
    group = all_rows[all_rows.scenario == scenario].set_index("variant")
    for k, variant in enumerate(("finQ", "instQ", "noQ")):
        x = group.loc[variant]
        fixed = scenario_labels[scenario] if k == 0 else ""
        lines.append(f"{fixed} & {symbols[variant]} & {x.nll:.1f} & {x.AIC:.1f} & "
                     f"{x.delta_AIC:.1f} & {fmt(x.exit_days, 2)} & {fmt(x.dNLL_exit_4d)} & "
                     f"{fmt(x.dNLL_exit_17d)} \\\\")
    if scenario != list(scenario_labels)[-1]:
        lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "table_sensitivity_fixed_choices.tex").write_text("\n".join(lines))

fin = all_rows[all_rows.variant == "finQ"]
print(f"finQ selected in {(all_rows.loc[all_rows.groupby('scenario').AIC.idxmin(), 'variant'] == 'finQ').sum()}/"
      f"{all_rows.scenario.nunique()} scenarios; exit range "
      f"{fin.exit_days.min():.2f}--{fin.exit_days.max():.2f} d")
