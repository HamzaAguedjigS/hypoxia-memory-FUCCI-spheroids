"""Sensitivity of the v8 results to the width of the observation window (+/-6 h and +/-18 h).

The v8 likelihood averages every predicted radius over t-W ... t+W (5 points, W = 12 h). Here the
complete WM793b data set is refitted with W = 6 h and W = 18 h, everything else as in v8
(64 equal-thickness shells, exact pimonidazole radius, dt = 0.5 h, eps_Q = 0.5, k2 = 0.1):
  stage 1  free fits of M_F (both v8 basins), M_D, M_I and M_0 (warm starts: v8 optima);
           exit-time profile (8 points x 2 basins) and exit + G1 profile (5 points x 2 basins);
  stage 2  up to two rounds of repair (restart a profile point from a better neighbour) and
           polish (free M_F fit from any profile point that beats the free optimum).

Records: results/v8_window/jobs.jsonl (one line per optimisation; the run can be resumed).
Analysis: 34_window_results.py.

Usage: python 33_window_sensitivity.py [N_WORKERS]
"""
import importlib.util
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_argv = sys.argv
sys.argv = [str(HERE / "28_v8_pipeline.py"), "1"]
spec = importlib.util.spec_from_file_location("p28", HERE / "28_v8_pipeline.py")
p28 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p28)
sys.argv = _argv
CTX, BASE = p28.CTX, p28.BASE

OUT = p28.ROOT / "results" / "v8_window"
OUT.mkdir(parents=True, exist_ok=True)
JOBS = OUT / "jobs.jsonl"
WINDOWS = (6.0, 18.0)
EXIT_W = [72, 96, 125, 150, 180, 220, 260, 400]
SUM_W = [5.0, 6.0, 7.0, 8.0, 10.0]


def run_job(job):
    """v8 optimisation with the observation window of the job."""
    m = CTX[job["ctx"]]
    w_old = m.OBS_WINDOW
    m.OBS_WINDOW = float(job["window"])
    try:
        return p28.run_job(job)
    finally:
        m.OBS_WINDOW = w_old


def records():
    if not JOBS.exists():
        return []
    return [json.loads(s) for s in JOBS.read_text().splitlines() if s.strip()]


def run_stage(pool, jobs, label):
    done = {r["id"] for r in records()}
    todo = [j for j in jobs if j["id"] not in done]
    todo.sort(key=lambda j: j["variant"] != "instQ")  # slowest fits first
    print(f"\n=== {label}: {len(todo)} jobs to run ({len(jobs) - len(todo)} already done) ===", flush=True)
    for i, rec in enumerate(pool.imap_unordered(run_job, todo), 1):
        with JOBS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{label} {i}/{len(todo)}] {rec['id']}  NLL={rec['nll']:.2f}  ({rec['minutes']} min)  "
              f"{time.strftime('%H:%M')}", flush=True)


def stage1():
    v8 = p28.records()
    B = p28.basins(v8, "all")
    jobs = []
    for w in WINDOWS:
        tag = f"w{w:g}"
        for k, x in enumerate(B):
            jobs.append(dict(id=f"{tag}|free|finQ|b{k}", stage="w1", kind="free", ctx="all", variant="finQ",
                             scen=BASE, window=w, family=f"{tag}|free", x0=list(x)))
        for v in ("delayQ", "instQ", "noQ"):
            x = p28.best(v8, ctx="all", variant=v)["x"]
            jobs.append(dict(id=f"{tag}|free|{v}|b0", stage="w1", kind="free", ctx="all", variant=v,
                             scen=BASE, window=w, family=f"{tag}|free", x0=list(x)))
        for T in EXIT_W:
            for k, x in enumerate(B):
                jobs.append(dict(id=f"{tag}|exit|{T}|b{k}", stage="w1", kind="fix", ctx="all", variant="finQ",
                                 scen=BASE, window=w, family=f"{tag}|exit", x0=list(x),
                                 fixed={"bmax": p28.lg(1 / T)}, value=T))
        for Sd in SUM_W:
            for k, x in enumerate(B):
                jobs.append(dict(id=f"{tag}|sum|{Sd}|b{k}", stage="w1", kind="fix", ctx="all", variant="finQ",
                                 scen=BASE, window=w, family=f"{tag}|sum", x0=list(x), fixed={},
                                 sum_h=24.0 * Sd, value=Sd))
    return jobs


def stage2(recs, round_no):
    """Repair profile points stuck above a better neighbour; polish from new optima."""
    jobs = []
    for fam in sorted({r["family"] for r in recs if r["kind"] == "fix"}):
        pts = {}
        for r in recs:
            if r.get("family") == fam and r["kind"] == "fix":
                if r["value"] not in pts or r["nll"] < pts[r["value"]]["nll"]:
                    pts[r["value"]] = r
        order = sorted(pts)
        for i, val in enumerate(order):
            nb = [pts[order[a]] for a in (i - 1, i + 1) if 0 <= a < len(order)]
            cand = min(nb, key=lambda r: r["nll"])
            me = pts[val]
            if cand["nll"] < me["nll"] - 1.0:
                job = dict(id=f"w2r{round_no}|{me['id']}", stage="w2", kind="fix", ctx="all", variant="finQ",
                           scen=BASE, window=me["window"], family=fam, x0=cand["x"], fixed=me["fixed"],
                           value=val)
                if "sum_h" in me:
                    job["sum_h"] = me["sum_h"]
                jobs.append(job)
    for w in WINDOWS:
        pool_ = [r for r in recs if r["window"] == w and r["variant"] == "finQ"]
        free_ = [r for r in pool_ if r["kind"] == "free"]
        low, bestfree = min(pool_, key=lambda r: r["nll"]), min(free_, key=lambda r: r["nll"])
        if low["nll"] < bestfree["nll"] - 0.5:
            jobs.append(dict(id=f"w2p{round_no}|w{w:g}|finQ", stage="w2", kind="free", ctx="all", variant="finQ",
                             scen=BASE, window=w, family=f"w{w:g}|free", x0=low["x"]))
    return jobs


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"window sensitivity (W = {', '.join(f'{w:g}' for w in WINDOWS)} h; v8 otherwise): "
          f"{n_workers} workers, started {time.ctime()}", flush=True)
    with Pool(n_workers) as pool:
        run_stage(pool, stage1(), "stage 1 (free fits and profiles)")
        for k in (1, 2):
            jobs = stage2(records(), k)
            if not jobs:
                break
            run_stage(pool, jobs, f"stage 2 round {k} (repair and polish)")
    print(f"\nwindow sensitivity finished {time.ctime()}", flush=True)
