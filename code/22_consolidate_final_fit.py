"""Keep the best all-data optimum found for every model and refresh predictions.

Candidates for each variant: the stored all-data fit
(results/radial_fit_<LINE>_alldata.json) and the baseline results of
20_exit_profile_sensitivity.py (profile grid + free polish, same protocol for
all four variants). The lower NLL wins. The previous file is kept as
radial_fit_<LINE>_alldata_pre_consolidation.json, and the dense predictions
used by the figures (radial_pred_<LINE>_alldata.csv) are regenerated.

Usage: python 22_consolidate_final_fit.py [LINE]
"""
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["FIT_ALL"] = "1"
LINE = sys.argv[1] if len(sys.argv) > 1 else "WM793b"
HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [sys.argv[0], LINE, "1"]
spec = importlib.util.spec_from_file_location("f7", HERE / "07_fit_radial.py")
f7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f7)
sys.argv = _argv
rm = f7.rm
RES = f7.ROOT / "results"
FIT_FILE = RES / f"radial_fit_{LINE}_alldata.json"
BACKUP = RES / f"radial_fit_{LINE}_alldata_pre_consolidation.json"
if not BACKUP.exists():
    shutil.copy(FIT_FILE, BACKUP)
stored = json.loads(BACKUP.read_text())
new = json.loads((RES / f"exit_profile_sensitivity_{LINE}.json").read_text())["baseline"]
rm.EPS_Q, rm.K2 = 0.5, 0.1

results = {}
for v in ("finQ", "delayQ", "instQ", "noQ"):
    cands = [("stored all-data fit", stored[v]["params"])]
    if v in new:
        cands.append(("20_exit_profile_sensitivity baseline", new[v]["params"]))
    names, lo, hi = f7.bounds(v)
    scored = []
    for src, th in cands:
        x = np.clip(np.log10([th[n] for n in names]), lo, hi)
        pred = f7.predict(x, v, fine=True)
        nll, sds = f7.nll(pred, f7.IDX_TRAIN)
        scored.append((nll, src, x, sds))
    nll, src, x, sds = min(scored, key=lambda s: s[0])
    k = len(names) + 4
    n_obs = int(np.isfinite(f7.IDX_TRAIN[2]).sum())
    th = f7.to_theta(x, v)
    results[v] = dict(params=th, free=names,
                      at_bound=[n for n, xi, l, h in zip(names, x, lo, hi) if xi <= l + 0.02 or xi >= h - 0.02],
                      nll_train=nll, sd=dict(zip(f7.OBS, sds.tolist())), n_params=k, n_obs=n_obs,
                      AIC=2 * nll + 2 * k, BIC=2 * nll + k * np.log(n_obs), nll_test=float("nan"),
                      source=src, candidate_nlls={s: round(n, 2) for n, s, _, _ in scored})
    print(f"{v:7s} NLL={nll:9.2f} AIC={2 * nll + 2 * k:8.1f} from {src}; "
          f"exit={1 / th['bmax'] if v in ('finQ', 'delayQ') else float('nan'):.0f} h "
          f"G1={1 / th['k1max']:.1f} h  candidates={results[v]['candidate_nlls']}")
FIT_FILE.write_text(json.dumps(results, indent=2))

rows = []
t_dense = np.arange(0, 7 * 24 + 1, 6.0)
for v in results:
    rm.DT = 0.5
    mdl = rm.Model(v, M=24, protocols=list(rm.PROTOCOLS))
    o = mdl.run(results[v]["params"], t_dense)
    for i, p in enumerate(mdl.protocols):
        for j, t in enumerate(t_dense):
            rows.append(dict(variant=v, protocol=p, day=rm.T0_DAY + t / 24, R_outer=o[i, j, 0],
                             R_inhib=o[i, j, 1], R_nec=o[i, j, 2], R_pim=o[i, j, 3], f_Q=o[i, j, 4]))
pd.DataFrame(rows).to_csv(RES / f"radial_pred_{LINE}_alldata.csv", index=False, float_format="%.5g")
best = min(r["AIC"] for r in results.values())
print({v: round(r["AIC"] - best, 1) for v, r in results.items()}, "(dAIC)")
