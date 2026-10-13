"""v8 (final): v7 plus the observation window, at 64 shells.

Observation window (RADIAL_OBS_WINDOW=12): every predicted radius is the mean of the model
trajectory at t-12, t-6, t, t+6, t+12 h. Independently formed spheroids differ by hours in
developmental timing, and the threshold radii (R_pim appears with the radius of the core, the
inhibited boundary falls abruptly) switch within hours in a single trajectory; without the
window the v7 optimum placed such a switch at an imaging day, and its NLL changed by 32 units
between 48 and 64 shells. With the window the change is 5 units at the same parameters, and
64 and 96 shells agree to 0.1 units.

Everything below is the v7 design (warm starts now come from the v7 records):
v7: every mechanistic inference with converged numerics and two-basin profiling.

Numerics (validated in 23_numerical_checks.py and the v6 analysis):
  * initial shells of equal THICKNESS (RADIAL_SHELL_LAYOUT=thickness): with equal volumes the
    central shell spans 28-35% of the radius and the onset of necrosis and of the
    pimonidazole region is poorly resolved (v6 optimum: NLL 3916 at M = 48, 3938 at M = 96);
    with equal thickness the radii at M = 48 agree with M = 96 to within 1.3 um and the
    likelihood to within 0.1;
  * pimonidazole radius from the exact piecewise oxygen profile (RADIAL_PIM_ANALYTIC=1);
  * M = 48 shells, dt = 0.5 h.

Two basins. v6 revealed that a short exit with a long G1 (4 d, 67 h) and a longer exit
with a shorter G1 (5-11 d, 35 h) both fit: after reoxygenation the green signal returns only
after exit AND G1 transit. Every profile is therefore started from both basins, and a
profile of the identifiable combination S = exit time + G1 time is added.

Stages (results appended to results/v7/jobs.jsonl; a restart skips finished jobs):
  1. free fits of the four models: WM793b all protocols and training protocols, WM164, WM983b;
  2. profiles from both basins: exit time (all data; training data scored on the withheld
     arm), S = exit + G1, every M_F parameter, bivariate exit x G1 and a x p_pim, and the
     fixed-choice sensitivity (eps_Q, k2) with exit profiles;
  3. repair passes (restart from a better neighbour) and free polish from any point that
     beats the current optimum.

Usage: python 28_v8_pipeline.py [N_WORKERS]
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

os.environ["RADIAL_FINE_M"] = "64"
os.environ["RADIAL_OBS_WINDOW"] = "12"
os.environ["RADIAL_FINE_DT"] = "0.5"
os.environ["RADIAL_SHELL_LAYOUT"] = "thickness"
os.environ["RADIAL_PIM_ANALYTIC"] = "1"
N_WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "v8"
OUT.mkdir(parents=True, exist_ok=True)
JOBS = Path(os.environ.get("V8_JOBS", str(OUT / "jobs.jsonl")))


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
rm = CTX["all"].rm
assert CTX["train"].HELD_OUT == ["reox_d4"] and CTX["all"].HELD_OUT == []
assert rm.SHELL_LAYOUT == "thickness" and rm.PIM_ANALYTIC and CTX["all"].FINE_M == 64 and CTX["all"].OBS_WINDOW == 12
BASE = (0.5, 0.1)
SCEN = {"epsQ_0.25": (0.25, 0.1), "epsQ_1.00": (1.0, 0.1), "k2_0.07": (0.5, 0.07), "k2_0.14": (0.5, 0.14)}
VARIANTS = ("finQ", "delayQ", "instQ", "noQ")
EXIT_ALL = [48, 72, 96, 110, 125, 144, 170, 200, 230, 260, 300, 400, 500, 800]
EXIT_TRAIN = [72, 96, 125, 144, 200, 300, 500, 800, 1500]
EXIT_SCEN = [72, 96, 125, 180, 260, 400]
SUM_DAYS = [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 12.0]
EXIT_2D = [72, 96, 125, 180, 260, 400]
G1_2D = [25, 35, 45, 55, 67, 80]
PPIM_2D = [0.05, 0.1, 0.3, 1.0, 1.3]
KEY = ("k1max", "Kq", "amax", "hq", "m_h", "tau_m", "a", "p_pim")


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
    S = job.get("sum_h")  # profile of exit time + G1 time = S (hours)
    if S is not None:
        jb, jk = names.index("bmax"), names.index("k1max")
        free = [i for i in free if i not in (jb, jk)]
        u0 = (1 / 10 ** x0[jb]) / (1 / 10 ** x0[jb] + 1 / 10 ** x0[jk])

        def unpack(z):
            x = x0.copy()
            x[free] = z[1:]
            u = 1 / (1 + np.exp(-z[0]))
            x[jb], x[jk] = np.log10(1 / (u * S)), np.log10(1 / ((1 - u) * S))
            return x
        z0 = np.r_[np.log(u0 / (1 - u0)), x0[free]]
    else:
        def unpack(z):
            x = x0.copy()
            x[free] = z
            return x
        z0 = x0[free]

    def obj(z):
        return m.objective(unpack(z), v, True)
    t0 = time.time()
    fev = (1500, 700) if job["kind"] == "free" else (1200, 400)
    r1 = minimize(obj, z0, method="Powell", options=dict(maxfev=fev[0], xtol=1e-3, ftol=1e-6))
    r2 = minimize(obj, r1.x, method="Nelder-Mead", options=dict(maxfev=fev[1], xatol=1e-4, fatol=1e-6))
    r = r2 if r2.fun < r1.fun else r1
    x = np.clip(unpack(r.x), lo, hi)
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


def lg(value):
    return float(np.log10(value))


def best(recs, **match):
    sub = [r for r in recs if all(r.get(k) == v for k, v in match.items()) and tuple(r.get("scen", BASE)) == BASE]
    return min(sub, key=lambda r: r["nll"]) if sub else None


def g1_of(x, m=None):
    m = m or CTX["all"]
    return 1 / 10 ** x[m.bounds("finQ")[0].index("k1max")]


def basins(recs, ctx="all"):
    """Best finQ vector with G1 < 50 h (basin A) and with G1 >= 50 h (basin B)."""
    sub = [r for r in recs if r["ctx"] == ctx and r["variant"] == "finQ" and tuple(r.get("scen", BASE)) == BASE]
    a = [r for r in sub if g1_of(r["x"]) < 50]
    b = [r for r in sub if g1_of(r["x"]) >= 50]
    out = []
    for grp in (a, b):
        if grp:
            out.append(min(grp, key=lambda r: r["nll"])["x"])
    return out


# ------------------------------------------------------------------ stage definitions
def v6_records():  # warm starts: the v7 records
    f = ROOT / "results" / "v7" / "jobs.jsonl"
    return [json.loads(s) for s in f.read_text().splitlines() if s.strip()] if f.exists() else []


def stage1():
    v6 = v6_records()
    v5 = json.loads((ROOT / "results" / "radial_fit_WM793b_alldata.json").read_text())
    jobs = []
    for ctx in ("all", "train", "WM164", "WM983b"):
        m = CTX[ctx]
        for v in VARIANTS:
            starts = []
            sub = [r for r in v6 if r["ctx"] == ctx and r["variant"] == v and tuple(r.get("scen", BASE)) == BASE]
            if v == "finQ" and ctx in ("all", "train"):
                starts += basins(v6, ctx)
            elif sub:
                starts.append(min(sub, key=lambda r: r["nll"])["x"])
            if ctx in ("all", "train"):  # a second, independent start
                names, lo, hi = m.bounds(v)
                th = dict(m.GUESS)
                th.update(v5[v]["params"])
                starts.append(np.clip(np.log10([th[n] for n in names]), lo, hi).tolist())
            elif len(sub) > 1:
                starts.append(sorted(sub, key=lambda r: r["nll"])[1]["x"])
            jobs += [dict(id=f"s1|{ctx}|{v}|{k}", stage="s1", kind="free", ctx=ctx, variant=v, x0=x)
                     for k, x in enumerate(starts)]
    return jobs


def profile_values(xm, lo, hi, offsets):
    vals = sorted({round(float(np.clip(xm + o, lo, hi)), 6) for o in offsets})
    return [val for val in vals if abs(val - xm) > 1e-6]


def fix_job(jid, family, x0, fixed, value, ctx="all", scen=BASE, **extra):
    return dict(id=jid, stage="s2", family=family, kind="fix", ctx=ctx, variant="finQ", scen=scen,
                x0=list(x0), fixed=fixed, value=value, **extra)


def stage2(recs):
    m = CTX["all"]
    names, lo, hi = m.bounds("finQ")
    B = basins(recs, "all")
    BT = basins(recs, "train")
    xbest = np.array(best(recs, ctx="all", variant="finQ")["x"])
    jobs = []
    for T in EXIT_ALL:
        for k, x in enumerate(B):
            jobs.append(fix_job(f"s2|exitall|{T}|b{k}", "exit_all", x, {"bmax": lg(1 / T)}, T))
    for T in EXIT_TRAIN:
        for k, x in enumerate(BT + B):
            jobs.append(fix_job(f"s2|exittrain|{T}|b{k}", "exit_train", x, {"bmax": lg(1 / T)}, T, ctx="train"))
    for Sd in SUM_DAYS:  # identifiable combination: exit + G1
        for k, x in enumerate(B):
            jobs.append(dict(id=f"s2|sum|{Sd}|b{k}", stage="s2", family="exit_plus_G1", kind="fix", ctx="all",
                             variant="finQ", scen=BASE, x0=list(x), fixed={}, sum_h=24.0 * Sd, value=Sd))
    for j, n in enumerate(names):  # univariate profiles of every parameter, from both basins
        if n == "bmax":
            continue
        if n.startswith("R1"):
            offs = [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1]
        elif n in KEY:
            offs = [-1.0, -0.5, -0.25, -0.1, 0.1, 0.25, 0.5, 1.0]
        else:
            offs = [-1.0, -0.4, -0.15, 0.15, 0.4, 1.0]
        starts = B if (n in KEY or n in ("mu0", "K1")) else [xbest]  # G1-related parameters: both basins
        for val in profile_values(xbest[j], lo[j], hi[j], offs):
            for k, x in enumerate(starts):
                jobs.append(fix_job(f"s2|uni|{n}|{val:.6f}|b{k}", f"uni|{n}", x, {n: val}, 10 ** val))
    for T in EXIT_2D:  # bivariate exit x G1: start from the basin with the closer G1
        for g in G1_2D:
            x = min(B, key=lambda xx: abs(np.log(g1_of(xx) / g)))
            jobs.append(fix_job(f"s2|exitG1|{T}|{g}", "exit_G1", x, {"bmax": lg(1 / T), "k1max": lg(1 / g)}, [T, g]))
    ja = names.index("a")
    for da in (-0.3, -0.15, 0.0, 0.15, 0.3):
        aval = float(np.clip(xbest[ja] + da, lo[ja], hi[ja]))
        for pp in PPIM_2D:
            jobs.append(fix_job(f"s2|aPim|{aval:.5f}|{pp}", "a_pim", xbest, {"a": aval, "p_pim": lg(pp)},
                                [10 ** aval, pp]))
    for s, sc in SCEN.items():  # fixed-choice sensitivity, both basins
        for T in EXIT_SCEN:
            for k, x in enumerate(B):
                jobs.append(fix_job(f"s2|scen|{s}|exit|{T}|b{k}", f"scen_exit|{s}", x, {"bmax": lg(1 / T)}, T,
                                    scen=sc))
        for v in ("finQ", "instQ", "noQ"):
            xs = B if v == "finQ" else [best(recs, ctx="all", variant=v)["x"]]
            for k, x in enumerate(xs):
                jobs.append(dict(id=f"s2|scen|{s}|free|{v}|b{k}", stage="s2", family=f"scen_free|{s}", kind="free",
                                 ctx="all", variant=v, scen=sc, x0=list(x)))
    return jobs


def stage3(recs, round_no):
    """Repair profile points stuck above a better neighbour; polish from new optima."""
    jobs = []
    fams = sorted({r["family"] for r in recs if r.get("kind") == "fix" and r.get("family")})
    for fam in fams:
        sub = [r for r in recs if r.get("family") == fam and r["kind"] == "fix"]
        pts = {}
        for r in sub:
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
                job = dict(id=f"s3r{round_no}|{me['id']}", stage="s3", family=fam, kind="fix", ctx=me["ctx"],
                           variant=me["variant"], scen=me.get("scen", BASE), x0=cand["x"], fixed=me["fixed"],
                           value=me["value"])
                if "sum_h" in me:
                    job["sum_h"] = me["sum_h"]
                jobs.append(job)
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
    print(f"v8 pipeline: 64 equal-thickness shells, exact R_pim, +/-12 h observation window, dt = 0.5 h, "
          f"{N_WORKERS} workers, "
          f"started {time.ctime()}", flush=True)
    with Pool(N_WORKERS) as pool:
        run_stage(pool, stage1(), "stage 1 (free fits)")
        run_stage(pool, stage2(records()), "stage 2 (profiles, validation, sensitivity)")
        for k in (1, 2):
            jobs = stage3(records(), k)
            if not jobs:
                break
            run_stage(pool, jobs, f"stage 3 round {k} (repair and polish)")
    print(f"\nv8 pipeline finished {time.ctime()}", flush=True)
