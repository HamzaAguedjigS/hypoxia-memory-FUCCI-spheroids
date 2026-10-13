"""v6: every mechanistic inference at a converged discretisation (48 shells, dt = 0.5 h).

Why v6. With 24 equal-volume shells the central shell spans 35% of the radius, so the
onset of necrosis and of the pimonidazole region is poorly resolved, and the v5 optimum
exploited these errors (finQ NLL 3922 at M = 24 against 3990 at M = 64; see
23_numerical_checks.py). With 48 shells the predicted radii agree with 64 shells to within
2.5 um (R_inh), 1.5 um (R_nec) and 5.4 um (R_pim).

Stages (every result is appended to results/v6/jobs.jsonl as soon as it is done; a
restart skips finished jobs):
  1. free fits: the four WM793b models on all protocols and on the training protocols
     (day-4 reoxygenation withheld), and the four models for WM164 and WM983b;
  2. from the stage-1 optima: the all-data exit-time profile; the training exit-time
     profile scored on the withheld arm; the fixed-choice sensitivity (eps_Q, k2) with
     exit-time profiles; univariate profiles of every M_F parameter; bivariate profiles
     exit time x G1 duration and a x p_pim;
  3. repair passes: a profile point is restarted from a neighbour with a lower NLL, and a
     free polish starts from any point that beats the current optimum.

Usage: python 24_v6_pipeline.py [N_WORKERS]
"""
import importlib.util
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

os.environ["RADIAL_FINE_M"] = "48"
os.environ["RADIAL_FINE_DT"] = "0.5"
N_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "v6"
OUT.mkdir(parents=True, exist_ok=True)
JOBS = Path(os.environ.get("V6_JOBS", str(OUT / "jobs.jsonl")))  # V6_JOBS: test override


def load(line, fit_all):
    if fit_all:
        os.environ["FIT_ALL"] = "1"
    else:
        os.environ.pop("FIT_ALL", None)
    argv = sys.argv
    sys.argv = [argv[0], line, "1"]
    spec = importlib.util.spec_from_file_location(f"f7_{line}_{int(fit_all)}", HERE / "07_fit_radial.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.argv = argv
    os.environ.pop("FIT_ALL", None)
    return mod


CTX = {"all": load("WM793b", True), "train": load("WM793b", False),
       "WM164": load("WM164", True), "WM983b": load("WM983b", True)}
assert CTX["train"].HELD_OUT == ["reox_d4"] and CTX["all"].HELD_OUT == []
rm = CTX["all"].rm
BASE = (0.5, 0.1)
SCEN = {"epsQ_0.25": (0.25, 0.1), "epsQ_1.00": (1.0, 0.1), "k2_0.07": (0.5, 0.07), "k2_0.14": (0.5, 0.14)}
VARIANTS = ("finQ", "delayQ", "instQ", "noQ")
EXIT_ALL = [48, 72, 96, 110, 125, 144, 170, 200, 230, 260, 300, 400, 500, 800]
EXIT_TRAIN = [96, 125, 144, 200, 300, 500, 800, 1500]
EXIT_SCEN = [96, 125, 180, 260, 400]
EXIT_2D = [96, 125, 180, 260, 400]
G1_2D = [20, 28, 35, 45, 60]
PPIM_2D = [0.05, 0.1, 0.3, 1.0, 1.3]


# ------------------------------------------------------------------ one optimisation
def run_job(job):
    m = CTX[job["ctx"]]
    rm.EPS_Q, rm.K2 = job.get("scen", BASE)
    v = job["variant"]
    names, lo, hi = m.bounds(v)
    x0 = np.clip(np.array(job["x0"], float), lo, hi)
    fixed = {names.index(k): val for k, val in job.get("fixed", {}).items()}
    for i, val in fixed.items():
        x0[i] = val
    free = [i for i in range(len(names)) if i not in fixed]

    def obj(z):
        x = x0.copy()
        x[free] = z
        return m.objective(x, v, True)
    t0 = time.time()
    fev = (1500, 700) if job["kind"] == "free" else (1200, 400)
    r1 = minimize(obj, x0[free], method="Powell", options=dict(maxfev=fev[0], xtol=1e-3, ftol=1e-6))
    r2 = minimize(obj, r1.x, method="Nelder-Mead", options=dict(maxfev=fev[1], xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    x = x0.copy()
    x[free] = r.x
    x = np.clip(x, lo, hi)
    pred = m.predict(x, v, True)
    nll, sds = m.nll(pred, m.IDX_TRAIN)
    rec = {k: val for k, val in job.items() if k != "x0"}
    rec.update(nll=float(nll), sds=[float(s) for s in sds], x=x.tolist(),
               nfev=int(r1.nfev + r2.nfev), minutes=round((time.time() - t0) / 60, 1))
    if job["ctx"] == "train":
        te, _ = m.nll(pred, m.IDX_TEST, sds)
        pi = m.PROTS.index("reox_d4")
        t6 = int(np.argmin(np.abs(m.T_OBS - 5 * 24)))
        t8 = int(np.argmin(np.abs(m.T_OBS - 7 * 24)))
        rec.update(nll_test=float(te), Rinh_d6=float(pred[pi, t6, 1]), Rinh_d8=float(pred[pi, t8, 1]))
    rm.EPS_Q, rm.K2 = BASE
    return rec


# ------------------------------------------------------------------ bookkeeping
def records():
    if not JOBS.exists():
        return []
    return [json.loads(s) for s in JOBS.read_text().splitlines() if s.strip()]


def run_stage(pool, jobs, label):
    done = {r["id"] for r in records()}
    todo = [j for j in jobs if j["id"] not in done]
    print(f"\n=== {label}: {len(todo)} jobs to run ({len(jobs) - len(todo)} already done) ===", flush=True)
    for i, rec in enumerate(pool.imap_unordered(run_job, todo), 1):
        with JOBS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{label} {i}/{len(todo)}] {rec['id']}  NLL={rec['nll']:.2f}"
              + (f" test={rec['nll_test']:.2f}" if "nll_test" in rec else "")
              + f"  ({rec['minutes']} min)", flush=True)


def best(recs, **match):
    sub = [r for r in recs if all(r.get(k) == v for k, v in match.items())]
    return min(sub, key=lambda r: r["nll"]) if sub else None


def xlog(m, v, params):
    names, lo, hi = m.bounds(v)
    th = dict(m.GUESS)
    th.update(params)
    return np.clip(np.log10([th[n] for n in names]), lo, hi).tolist()


def jsonfile(name):
    return json.loads((ROOT / "results" / name).read_text())


def lg(value):
    return float(np.log10(value))


# ------------------------------------------------------------------ stage definitions
def stage1():
    m = CTX["all"]
    v5, v5pre = jsonfile("radial_fit_WM793b_alldata.json"), jsonfile("radial_fit_WM793b_alldata_pre_consolidation.json")
    v5train = jsonfile("radial_fit_WM793b.json")
    p20 = [json.loads(s) for s in (ROOT / "results" / "exit_profile_sensitivity_WM793b_partial.jsonl")
           .read_text().splitlines() if s.strip()]
    p20 = [r for r in p20 if r["setting"] == "baseline" and r["stage"] == "fixed_exit"]
    jobs = []
    for v in VARIANTS:
        starts = [xlog(m, v, v5[v]["params"]), xlog(m, v, v5pre[v]["params"])]
        starts += [r["x"] for r in p20 if r["variant"] == v and r["exit_fixed_h"] in (125.0, 180.0, 260.0)]
        jobs += [dict(id=f"s1|all|{v}|{k}", stage="s1", kind="free", ctx="all", variant=v, x0=x)
                 for k, x in enumerate(starts)]
        jobs += [dict(id=f"s1|train|{v}|{k}", stage="s1", kind="free", ctx="train", variant=v, x0=x)
                 for k, x in enumerate([xlog(m, v, v5train[v]["params"]), xlog(m, v, v5[v]["params"])])]
    for line in ("WM164", "WM983b"):
        own = jsonfile(f"radial_fit_{line}.json")
        for v in VARIANTS:
            for k, params in enumerate([own[v]["params"], v5[v]["params"]]):
                jobs.append(dict(id=f"s1|{line}|{v}|{k}", stage="s1", kind="free", ctx=line, variant=v,
                                 x0=xlog(CTX[line], v, params)))
    return jobs


def profile_values(xm, lo, hi, offsets):
    vals = sorted({round(float(np.clip(xm + o, lo, hi)), 6) for o in offsets})
    return [val for val in vals if abs(val - xm) > 1e-6]


def stage2(recs):
    m = CTX["all"]
    names, lo, hi = m.bounds("finQ")
    bF = np.array(best(recs, ctx="all", variant="finQ", stage="s1")["x"])
    bT = best(recs, ctx="train", variant="finQ", stage="s1")["x"]
    jk, jb = names.index("k1max"), names.index("bmax")
    alt = bF.copy()
    alt[jk] = lg(1 / 50)
    jobs = []
    for T in EXIT_ALL:  # all-data exit profile
        for k, x in enumerate([bF, alt]):
            jobs.append(dict(id=f"s2|exitall|{T}|{k}", stage="s2", family="exit_all", kind="fix", ctx="all",
                             variant="finQ", x0=x.tolist(), fixed={"bmax": lg(1 / T)}, value=T))
    for T in EXIT_TRAIN:  # training profile scored on the held-out arm
        for k, x in enumerate([bT, bF.tolist()]):
            jobs.append(dict(id=f"s2|exittrain|{T}|{k}", stage="s2", family="exit_train", kind="fix",
                             ctx="train", variant="finQ", x0=list(x), fixed={"bmax": lg(1 / T)}, value=T))
    for s, sc in SCEN.items():  # fixed-choice sensitivity
        for T in EXIT_SCEN:
            jobs.append(dict(id=f"s2|scen|{s}|exit|{T}", stage="s2", family=f"scen_exit|{s}", kind="fix",
                             ctx="all", variant="finQ", scen=sc, x0=bF.tolist(), fixed={"bmax": lg(1 / T)},
                             value=T))
        for v in ("finQ", "instQ", "noQ"):
            jobs.append(dict(id=f"s2|scen|{s}|free|{v}", stage="s2", family=f"scen_free|{s}", kind="free",
                             ctx="all", variant=v, scen=sc,
                             x0=best(recs, ctx="all", variant=v, stage="s1")["x"]))
    for j, n in enumerate(names):  # univariate profiles of every finQ parameter
        if n == "bmax":
            continue
        offs = ([-0.1, -0.05, -0.02, 0.02, 0.05, 0.1] if n.startswith("R1")
                else [-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0])
        for val in profile_values(bF[j], lo[j], hi[j], offs):
            jobs.append(dict(id=f"s2|uni|{n}|{val:.6f}", stage="s2", family=f"uni|{n}", kind="fix", ctx="all",
                             variant="finQ", x0=bF.tolist(), fixed={n: val}, value=10 ** val))
    for T in EXIT_2D:  # bivariate exit x G1
        for g in G1_2D:
            jobs.append(dict(id=f"s2|exitG1|{T}|{g}", stage="s2", family="exit_G1", kind="fix", ctx="all",
                             variant="finQ", x0=bF.tolist(), fixed={"bmax": lg(1 / T), "k1max": lg(1 / g)},
                             value=[T, g]))
    ja = names.index("a")
    for da in (-0.3, -0.15, 0.0, 0.15, 0.3):  # bivariate a x p_pim
        aval = float(np.clip(bF[ja] + da, lo[ja], hi[ja]))
        for pp in PPIM_2D:
            jobs.append(dict(id=f"s2|aPim|{aval:.5f}|{pp}", stage="s2", family="a_pim", kind="fix", ctx="all",
                             variant="finQ", x0=bF.tolist(), fixed={"a": aval, "p_pim": lg(pp)},
                             value=[10 ** aval, pp]))
    return jobs


def stage3(recs, round_no):
    """Repair profile points stuck above a better neighbour; polish from new optima."""
    jobs = []
    fams = sorted({r["family"] for r in recs if r.get("stage") in ("s2", "s3") and r["kind"] == "fix"})
    for fam in fams:
        sub = [r for r in recs if r.get("family") == fam and r["kind"] == "fix"]
        pts = {}
        for r in sub:  # best record per grid value
            key = json.dumps(r["value"])
            if key not in pts or r["nll"] < pts[key]["nll"]:
                pts[key] = r
        keys = list(pts)
        vals = [json.loads(k) for k in keys]
        two_d = isinstance(vals[0], list)
        for k, val in zip(keys, vals):
            if two_d:
                xs = sorted({v_[0] for v_ in vals})
                ys = sorted({v_[1] for v_ in vals})
                i, j = xs.index(val[0]), ys.index(val[1])
                nb = [[xs[a], ys[b]] for a, b in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                      if 0 <= a < len(xs) and 0 <= b < len(ys)]
            else:
                order = sorted(vals)
                i = order.index(val)
                nb = [order[a] for a in (i - 1, i + 1) if 0 <= a < len(order)]
            nb = [pts[json.dumps(n_)] for n_ in nb if json.dumps(n_) in pts]
            if not nb:
                continue
            cand = min(nb, key=lambda r: r["nll"])
            if cand["nll"] < pts[k]["nll"] - 1.0:
                me = pts[k]
                jobs.append(dict(id=f"s3r{round_no}|{me['id']}", stage="s3", family=fam, kind="fix",
                                 ctx=me["ctx"], variant=me["variant"], scen=me.get("scen", BASE),
                                 x0=cand["x"], fixed=me["fixed"], value=me["value"]))
    # free polish from the lowest point of each context/scenario if it beats the free optimum
    groups = [("all", "finQ", None), ("train", "finQ", None)] + [("all", "finQ", s) for s in SCEN]
    for ctx, v, s in groups:
        sc = SCEN[s] if s else BASE
        pool_ = [r for r in recs if r["ctx"] == ctx and r["variant"] == v and tuple(r.get("scen", BASE)) == sc]
        free_ = [r for r in pool_ if r["kind"] == "free"]
        if not pool_ or not free_:
            continue
        low, bestfree = min(pool_, key=lambda r: r["nll"]), min(free_, key=lambda r: r["nll"])
        if low["nll"] < bestfree["nll"] - 0.5:
            jobs.append(dict(id=f"s3p{round_no}|{ctx}|{v}|{s}", stage="s3", kind="free", ctx=ctx, variant=v,
                             scen=sc, x0=low["x"], family=f"polish|{s or 'baseline'}"))
    return jobs


if __name__ == "__main__":
    print(f"v6 pipeline, 48 shells, dt = 0.5 h, {N_WORKERS} workers, started {time.ctime()}", flush=True)
    with Pool(N_WORKERS) as pool:
        run_stage(pool, stage1(), "stage 1 (free fits)")
        run_stage(pool, stage2(records()), "stage 2 (profiles, validation, sensitivity)")
        for k in (1, 2):
            jobs = stage3(records(), k)
            if not jobs:
                break
            run_stage(pool, jobs, f"stage 3 round {k} (repair and polish)")
    print(f"\nv6 pipeline finished {time.ctime()}", flush=True)
