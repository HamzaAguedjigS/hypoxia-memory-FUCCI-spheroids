"""Initial-condition check: are rapid entry and the exit range artefacts of starting without arrest?

In v8, spheroids grown in 2% start at day 1 with no arrested cells. Here M_F and M_D get one extra
parameter, q0_hyp: at day 1 a fraction q0_hyp * (1 - s(p)) of the G1 cells of each shell of the
2% group is already arrested (s = oxygen switch, p = day-1 oxygen profile), so initial arrest sits
where oxygen is low. Everything else is v8 (64 equal-thickness shells, exact R_pim, +/-12 h window,
dt = 0.5 h, all WM793b data). M_I and M_0 have no arrested state and keep their v8 fits.

  stage 1  free fits of M_F (both v8 basins x q0 = 0.05, 0.5) and M_D (q0 = 0.3);
           profiles of q0_hyp (5 values), arrest-entry time (2-16 h) and exit time (8 values),
           each from both basins;
  stage 2  up to two rounds of repair (restart from a better neighbour) and polish.

Records: results/v8_initial/jobs.jsonl (resumable). Analysis: 36_initial_results.py.

Usage: python 35_initial_condition.py [N_WORKERS]
"""
import os

os.environ["RADIAL_Q0_HYP"] = "1"   # must be set before the fitting module is loaded

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
assert CTX["all"].Q0_HYP and CTX["all"].bounds("finQ")[0][-1] == "q0_hyp"

OUT = p28.ROOT / "results" / "v8_initial"
OUT.mkdir(parents=True, exist_ok=True)
JOBS = OUT / "jobs.jsonl"
Q0_PROFILE = [0.001, 0.03, 0.1, 0.3, 1.0]
ENTRY_H = [2.0, 4.0, 8.0, 16.0]
EXIT_H = [72, 96, 125, 150, 180, 220, 260, 400]


def run_job(job):
    return p28.run_job(job)


def records():
    if not JOBS.exists():
        return []
    return [json.loads(s) for s in JOBS.read_text().splitlines() if s.strip()]


def run_stage(pool, jobs, label):
    done = {r["id"] for r in records()}
    todo = [j for j in jobs if j["id"] not in done]
    todo.sort(key=lambda j: j["kind"] != "free")  # free fits (longest) first
    print(f"\n=== {label}: {len(todo)} jobs to run ({len(jobs) - len(todo)} already done) ===", flush=True)
    for i, rec in enumerate(pool.imap_unordered(run_job, todo), 1):
        with JOBS.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[{label} {i}/{len(todo)}] {rec['id']}  NLL={rec['nll']:.2f}  ({rec['minutes']} min)  "
              f"{time.strftime('%H:%M')}", flush=True)


def with_q0(x, q0):
    return list(x) + [float(np.log10(q0))]


def fix(jid, fam, x0, fixed, value):
    return dict(id=jid, stage="i1", kind="fix", ctx="all", variant="finQ", scen=BASE, family=fam,
                x0=list(x0), fixed=fixed, value=value)


def stage1():
    v8 = p28.records()
    B = p28.basins(v8, "all")
    jobs = []
    for k, x in enumerate(B):
        for q0 in (0.05, 0.5):
            jobs.append(dict(id=f"i|free|finQ|b{k}|q{q0:g}", stage="i1", kind="free", ctx="all", variant="finQ",
                             scen=BASE, family="free", x0=with_q0(x, q0)))
    xd = p28.best(v8, ctx="all", variant="delayQ")["x"]
    jobs.append(dict(id="i|free|delayQ|b0", stage="i1", kind="free", ctx="all", variant="delayQ",
                     scen=BASE, family="free", x0=with_q0(xd, 0.3)))
    for k, x in enumerate(B):
        for q0 in Q0_PROFILE:
            jobs.append(fix(f"i|q0|{q0:g}|b{k}", "q0", with_q0(x, 0.3), {"q0_hyp": float(np.log10(q0))}, q0))
        for T in ENTRY_H:
            jobs.append(fix(f"i|entry|{T:g}|b{k}", "entry", with_q0(x, 0.3), {"amax": float(np.log10(1 / T))}, T))
        for T in EXIT_H:
            jobs.append(fix(f"i|exit|{T}|b{k}", "exit", with_q0(x, 0.3), {"bmax": float(np.log10(1 / T))}, T))
    return jobs


def stage2(recs, round_no):
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
                jobs.append(dict(id=f"i2r{round_no}|{me['id']}", stage="i2", kind="fix", ctx="all",
                                 variant="finQ", scen=BASE, family=fam, x0=cand["x"], fixed=me["fixed"],
                                 value=val))
    pool_ = [r for r in recs if r["variant"] == "finQ"]
    free_ = [r for r in pool_ if r["kind"] == "free"]
    low, bestfree = min(pool_, key=lambda r: r["nll"]), min(free_, key=lambda r: r["nll"])
    if low["nll"] < bestfree["nll"] - 0.5:
        jobs.append(dict(id=f"i2p{round_no}|finQ", stage="i2", kind="free", ctx="all", variant="finQ",
                         scen=BASE, family="free", x0=low["x"]))
    return jobs


if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"initial-condition check (q0_hyp free for M_F and M_D; v8 otherwise): {n_workers} workers, "
          f"started {time.ctime()}", flush=True)
    with Pool(n_workers) as pool:
        run_stage(pool, stage1(), "stage 1 (free fits and profiles)")
        for k in (1, 2):
            jobs = stage2(records(), k)
            if not jobs:
                break
            run_stage(pool, jobs, f"stage 2 round {k} (repair and polish)")
    print(f"\ninitial-condition check finished {time.ctime()}", flush=True)
