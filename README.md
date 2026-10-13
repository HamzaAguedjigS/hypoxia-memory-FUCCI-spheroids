# Multi-day memory of hypoxia-induced cell-cycle arrest in melanoma spheroids

Code, processed data and complete optimisation records for

> H. Aguedjig, S. M. Douiri and I. El Harraki. *Mathematical modelling reveals a multi-day memory of hypoxia-induced cell-cycle arrest in melanoma spheroids.* Submitted to the Journal of the Royal Society Interface (2026).

The analysis uses the FUCCI and pimonidazole spheroid measurements published by Murphy et al. (2023, see [Source data](#source-data)). It

1. tests, without a mechanistic model, whether the FUCCI-inhibited region depends on oxygen history;
2. fits a radial oxygen–cell-cycle model with four arrest hypotheses to the four image-derived radii, and compares them by AIC/BIC, profile likelihood, a withheld reoxygenation arm and sensitivity analyses.

## Contents

```
code/               analysis scripts (numbered) and model modules
data/processed/     spheroids.csv: one row per spheroid, derived from the source data
data/raw/           location for the source data (not redistributed, see data/raw/README.md)
results/            outputs of the model-free analyses and all optimisation records
  v7/               earlier optimisation records, used only as starting points by script 28
  v8/               final analysis: fits, profiles, held-out scores, sensitivity, tables
  v8_window/        sensitivity to the observation window (+/-6 h and +/-18 h)
  v8_initial/       initial-condition check (arrested cells already present on day 1)
figures/            all figures are written here
```

## Requirements

Python 3.8 with numpy 1.24.3, scipy 1.10.0, pandas 2.0.3, matplotlib 3.6.3 and openpyxl 3.1.5:

```
pip install -r requirements.txt
```

All scripts are run from `code/`. Paths are relative to the repository, so no configuration is needed.

## Model names

| Manuscript | Code (`variant`) | Hypothesis |
|---|---|---|
| $\mathcal{M}_F$ | `finQ` | finite arrest entry and exit |
| $\mathcal{M}_D$ | `delayQ` | delayed arrest entry |
| $\mathcal{M}_I$ | `instQ` | instantaneous oxygen coupling |
| $\mathcal{M}_0$ | `noQ` | cycling without a distinct arrested state |

The radial model is implemented in `code/radial_model.py`; likelihood, parameter bounds and data handling are in `code/07_fit_radial.py`.

## Reproducing the manuscript

### A. From the saved optimisation records (about 10 minutes)

All optimisation records are included, so every figure, table and reported number can be regenerated without refitting. Script 03 takes about 8 minutes; each of the others takes at most about 1 minute.

```
cd code
python 01_prepare_data.py            # requires the source data, see data/raw/README.md
python 02_history_dependence.py      # model-free tests (section 2)
python 03_switch_kinetics.py 200     # readout time scales, 200 bootstrap samples
python 16_section2_figure.py         # Figure 1
python 29_v8_results.py              # fits, profiles, held-out scores, Tables 5 and S1, S2, S5, S6
python 30_identifiability_figures.py # Figures 4, 5 and S1 (run after 29)
python 31_oxygen_check_v8.py         # numerical checks of the oxygen solver
python 32_exit_profile_figure.py     # Figure 6
python 34_window_results.py          # Table S3, Figure S2
python 36_initial_results.py         # Table S4, Figure S3
```

`data/processed/spheroids.csv` is already provided, so step 01 can be skipped if the source data are not downloaded.

### B. Refitting from scratch (several days)

```
python 28_v8_pipeline.py 12          # four models, profiles, held-out arm, fixed-choice sensitivity (about 2 days)
python 33_window_sensitivity.py 12   # observation-window sensitivity (about 5 h)
python 35_initial_condition.py 12    # initial-condition check (about 4 h)
```

- **Workers.** The argument is the number of parallel worker processes; the times above are for 12 workers.
- **Resuming a run.** Each pipeline appends one JSON line per optimisation to `jobs.jsonl` in its results folder and skips jobs already recorded, so a run can be interrupted and resumed. To recompute from scratch, first move the corresponding `jobs.jsonl` out of the way.
- **Numerical settings.** Script 28 sets them itself: 64 equal-thickness shells, time step 0.5 h, a +/-12 h observation window and the exact pimonidazole boundary.
- **Starting points.** Script 28 uses `results/v7/jobs.jsonl` and `results/radial_fit_WM793b_alldata.json` only as starting points. Every final fit and profile point is also started from both likelihood basins (manuscript, section 4(b)).

### Optimisation records

Each line of a `jobs.jsonl` file describes one optimisation:

| Field | Content |
|---|---|
| `ctx` | data used: `all` (all WM793b protocols), `train` (day-4 reoxygenation withheld), `WM164` or `WM983b` |
| `variant` | model, named as in the table above |
| `kind`, `stage` | type of job (free fit, profile point, repair, polish) and pipeline stage |
| `x` | free parameters on a log10 scale, in the order returned by `bounds(variant)` in `07_fit_radial.py` |
| `nll` | negative log-likelihood |
| `sds` | profiled residual standard deviations of the four radii (µm) |
| `nfev`, `minutes` | cost of the optimisation |

## Where each figure and table comes from

| Manuscript | Script | Output |
|---|---|---|
| Figure 1 | `16_section2_figure.py` | `figures/fig01_section2_model_free.pdf` |
| Figure 2 | schematic drawn in LaTeX (manuscript sources) | — |
| Figure 3 | `29_v8_results.py` | `figures/fig03_model_fits.pdf` |
| Figure 4 | `30_identifiability_figures.py` | `figures/fig06_identifiability.pdf` |
| Figure 5 | `30_identifiability_figures.py` | `figures/fig07_identifiability_map.pdf` |
| Figure 6 | `32_exit_profile_figure.py` | `figures/fig06_exit_profile_v2.pdf` |
| Figure 7 | `29_v8_results.py` | `figures/fig05_readout_vs_arrest.pdf` |
| Figure S1 | `30_identifiability_figures.py` | `figures/figS1_all_profiles.pdf` |
| Figure S2 | `34_window_results.py` | `figures/figS_window_sensitivity.pdf` |
| Figure S3 | `36_initial_results.py` | `figures/figS_initial_condition.pdf` |
| Figure S4 | `29_v8_results.py` | `figures/figS2_fits_WM164.pdf` |
| Figure S5 | `29_v8_results.py` | `figures/figS3_fits_WM983b.pdf` |
| Table 5 | `29_v8_results.py` | `results/v8/table_model_comparison.tex` |
| Table S1 | `29_v8_results.py` | `results/v8/table_parameters.tex` |
| Table S2 | `29_v8_results.py` | `results/v8/table_sensitivity.tex` |
| Table S3 | `34_window_results.py` | `results/v8_window/table_window_sensitivity.tex` |
| Table S4 | `36_initial_results.py` | `results/v8_initial/table_initial_condition.tex` |
| Table S5 | `29_v8_results.py` | `results/v8/table_other_lines.tex` |
| Table S6 | `29_v8_results.py` | `results/v8/table_numerical.tex` |
| Model-free statistics (section 2) | `02_history_dependence.py` | `results/history_tests.csv` |
| Readout time scales (section 2(f), ESM appendix B) | `03_switch_kinetics.py` | `results/switch_kinetics_*.csv` |
| Numerical checks (ESM appendix C) | `31_oxygen_check_v8.py`, `29_v8_results.py` | `results/v8/oxygen_check_v8.txt`, `results/v8/numerical_robustness.csv` |
| Numbers quoted in the text | `29`, `34`, `36` | `results/v8/summary.txt`, `results/v8_window/summary.txt`, `results/v8_initial/summary.txt` |

`29_v8_results.py` also writes simpler preliminary versions of Figures 4 and S1 (`*_basic`); the published versions are produced by `30_identifiability_figures.py`. The figure file names are those of the working analysis; the manuscript numbering is given in the table.

## Development history

The final analysis consists of scripts 01–03, 16 and 28–36, together with `radial_model.py` and `07_fit_radial.py`, which script 28 imports. The other scripts are earlier stages of the work, kept for traceability:
- 04–06 and `mech_model.py`: a well-mixed ODE model;
- 08–27: radial-model versions v1–v7.

They are not needed to reproduce the manuscript, and most of their intermediate outputs are not included. In the authors' working copy, PDF figures were also written to the manuscript folder; in this repository every figure is written to `figures/`.

## Source data

The measurements were published by

> R. J. Murphy, G. Gunasingh, N. K. Haass and M. J. Simpson. Growth and adaptation mechanisms of tumour spheroids with time-dependent oxygen availability. *PLOS Computational Biology* 19(1): e1010833 (2023). https://doi.org/10.1371/journal.pcbi.1010833

They are available from the authors' repository https://github.com/ryanmurphy42/Murphy2022SpheroidOxygenAdaptation. Please cite their article when using the data or the derived table `data/processed/spheroids.csv`.

## Licence

The code is released under the MIT licence (see `LICENSE`).

## Citation

If you use this code or these records, please cite the article above and this archive: DOI to be added after the first release.
