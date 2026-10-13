# Source data

The raw measurements are not redistributed here. They belong to

> R. J. Murphy, G. Gunasingh, N. K. Haass and M. J. Simpson. Growth and adaptation mechanisms of tumour spheroids with time-dependent oxygen availability. *PLOS Computational Biology* 19(1): e1010833 (2023). https://doi.org/10.1371/journal.pcbi.1010833

To rebuild `data/processed/spheroids.csv` with `code/01_prepare_data.py`:

1. Download the three files below from the `1Data/` folder of https://github.com/ryanmurphy42/Murphy2022SpheroidOxygenAdaptation:
   - `ExpOA_WM793b.xlsx`
   - `ExpOA_WM164.xlsx`
   - `ExpOA_WM983b.xlsx`
2. Put them in `data/raw/Murphy2023_oxygen/`.
3. Run `python 01_prepare_data.py` from `code/`.

Each spheroid record gives the outer, inhibited (FUCCI), necrotic and pimonidazole-positive radii (µm) measured on an equatorial confocal section, on days 2, 3, 4, 6 and 8. The experiments cover five oxygen protocols:
- constant 21% O2;
- constant 2% O2;
- 21% to 2% on day 2;
- 2% to 21% on day 2;
- 2% to 21% on day 4.

`01_prepare_data.py` maps the experimental condition codes to these protocols, as in the authors' MATLAB scripts.
