"""
experiments.py
==============
Three extra studies that make the results more reliable (run after train.py):

  A. ROBUSTNESS   – fresh motors never used before, tested with the saved networks:
                    accuracy vs sensor noise, vs load level, vs fault severity
  B. REPEATED RUNS – every network trained 10 times (new 3-set split + new initial weights each time)
                    -> mean ± standard deviation instead of a single, possibly lucky, number
  C. HIDDEN NEURONS – MLP with H = 1 ... 100 hidden neurons: under-fitting -> good -> over-fitting

Run:   python experiments.py            (≈ 10–15 min on 2 cores; uses all CPU cores)
       python experiments.py --quick    (smaller version, ≈ 3 min, to test)
Output: results/experiments.json  +  figures/exp_*.png   (shown on the "Experiments" page of the app)
"""
from __future__ import annotations

import os
import json
import time
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from motor_model import MotorParams, simulate
from faults import CLASSES, make_fault_state
from features import add_noise, extract_features
from networks import MODEL_NAMES, MLP
import train as T
import plots

NOISE_LEVELS = [0.0025, 0.005, 0.01, 0.02, 0.03, 0.05]
LOAD_BINS = [(0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.1)]
HIDDEN = [1, 2, 4, 8, 15, 30, 60, 100]


# ------------------------------------------------------------------ A. robustness
def _pool_motor(args):
    """Simulate ONE fresh motor and return its features for every noise level (same motor, different sensors)."""
    c, sev, load, seed = args
    rng = np.random.default_rng(seed)
    mp = MotorParams()
    mp.Rs *= rng.uniform(0.95, 1.05)
    mp.Rr *= rng.uniform(0.95, 1.05)
    sig = simulate(mp, make_fault_state(c, sev, mp, rng), load=load, t_end=3.0)
    return [extract_features(add_noise(sig, nz, np.random.default_rng(seed + 1)), mp) for nz in NOISE_LEVELS]


def robustness(n_per_class, jobs, seed=12345):
    rng = np.random.default_rng(seed)
    meta, args = [], []
    for c in range(len(CLASSES)):
        for k in range(n_per_class):
            sev = 0 if c == 0 else k % 3 + 1
            load = float(rng.uniform(0.3, 1.1))
            meta.append((c, sev, load))
            args.append((c, sev, load, int(rng.integers(1 << 30))))
    t = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        feats = list(ex.map(_pool_motor, args, chunksize=4))
    print(f"  robustness pool: {len(args)} fresh motors simulated in {time.time()-t:.0f}s")
    feats = np.array(feats)                      # motors x noise levels x 19
    meta = pd.DataFrame(meta, columns=["label", "severity", "load"])
    b = T.load_bundle(os.path.join(T.MOD, "diagnosis_bundle.pkl"))
    sc = b["scaler"]
    noise_rows, sev_rows, load_rows = [], [], []
    i_ref = NOISE_LEVELS.index(0.01)              # breakdowns at the nominal 1 % noise
    for name, m in b["models"].items():
        for j, nz in enumerate(NOISE_LEVELS):
            pred = m.predict(sc.transform(feats[:, j]))
            noise_rows.append({"model": name, "noise": nz, "accuracy": float(np.mean(pred == meta.label))})
        pred = m.predict(sc.transform(feats[:, i_ref]))
        ok = pred == meta.label.values
        for s in range(4):
            msk = meta.severity.values == s
            sev_rows.append({"model": name, "severity": s, "accuracy": float(ok[msk].mean())})
        for lo, hi in LOAD_BINS:
            msk = (meta.load.values >= lo) & (meta.load.values < hi)
            load_rows.append({"model": name, "load": f"{int(lo*100)}–{int(hi*100)} %", "accuracy": float(ok[msk].mean())})
    return {"n_motors": len(args), "noise": noise_rows, "severity": sev_rows, "load": load_rows}


# ------------------------------------------------------------------ B. repeated runs
def _one_run(args):
    df, task, seed = args
    prep = T.prepare(df, task, seed)
    out = []
    for kind in MODEL_NAMES:
        m = T.train_one(kind, prep, {"seed": seed})
        r = T.result_row(m, prep)
        out.append({"task": task, "seed": seed, "model": kind, "accuracy": r["accuracy"], "f1": r["f1"],
                    "train_time_s": r["train_time_s"], "best_epoch_A": r["best_epoch_A"]})
    return out


def repeated_runs(df, n_runs, jobs):
    t = time.time()
    tasks = [(df, task, s) for task in ("diagnosis", "detection") for s in range(n_runs)]
    rows = []
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(_one_run, tasks):
            rows += r
    print(f"  repeated runs: {len(tasks)} runs x 7 networks in {time.time()-t:.0f}s")
    summ = (pd.DataFrame(rows).groupby(["task", "model"])
            .agg(acc_mean=("accuracy", "mean"), acc_std=("accuracy", "std"), acc_min=("accuracy", "min"),
                 acc_max=("accuracy", "max"), f1_mean=("f1", "mean"), time_mean=("train_time_s", "mean"))
            .reset_index())
    return {"n_runs": n_runs, "runs": rows, "summary": summ.to_dict(orient="records")}


# ------------------------------------------------------------------ C. hidden neurons
def _one_hidden(args):
    df, H, seed = args
    prep = T.prepare(df, "diagnosis", seed)
    m = MLP(H, eta=0.02, alpha=0.8, epochs=300, seed=seed)
    m.fit(prep["X_train"], prep["y_train"], prep["X_test"], prep["y_test"], n_classes=prep["n_classes"])
    h = m.history
    return {"H": H, "seed": seed, "train_E_final": h["train_E"][-1], "test_E_final": h["test_E"][-1],
            "test_E_best": min(h["test_E"]), "best_epoch": m.best_epoch,
            "train_acc": float(np.mean(m.predict(prep["X_train"]) == prep["y_train"])),
            "val_acc": float(np.mean(m.predict(prep["X_val"]) == prep["y_val"]))}


def hidden_sweep(df, n_seeds, jobs):
    t = time.time()
    tasks = [(df, H, s) for H in HIDDEN for s in range(n_seeds)]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        rows = list(ex.map(_one_hidden, tasks))
    print(f"  hidden sweep: {len(tasks)} trainings in {time.time()-t:.0f}s")
    return {"n_seeds": n_seeds, "rows": rows}


# ------------------------------------------------------------------ figures + main
SEV_LABELS = {0: "Healthy", 1: "Severity 1\n(early)", 2: "Severity 2\n(medium)", 3: "Severity 3\n(severe)"}


def make_figures(res):
    if "robustness" in res:
        r = res["robustness"]
        T.savefig(plots.noise_robustness_plot(r["noise"]), "exp_noise_robustness.png")
        T.savefig(plots.breakdown_plot(r["severity"], "severity", SEV_LABELS,
                                       "Diagnosis accuracy vs fault severity (fresh motors, 1 % noise)"),
                  "exp_severity.png")
        T.savefig(plots.breakdown_plot(r["load"], "load", {}, "Diagnosis accuracy vs load (fresh motors, 1 % noise)"),
                  "exp_load.png")
    if "repeated" in res:
        for task in ("diagnosis", "detection"):
            T.savefig(plots.repeated_runs_box(res["repeated"]["runs"], task), f"exp_repeated_{task}.png")
    if "hidden" in res:
        T.savefig(plots.hidden_sweep_plot(res["hidden"]["rows"]), "exp_hidden_neurons.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small version to test quickly")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--only", choices=["robustness", "repeated", "hidden"], default=None)
    a = ap.parse_args()
    df = pd.read_csv(T.DATA)
    path = os.path.join(T.RES, "experiments.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    q = a.quick
    if a.only in (None, "robustness"):
        print("A. robustness ...")
        res["robustness"] = robustness(10 if q else 100, a.jobs)
    if a.only in (None, "repeated"):
        print("B. repeated runs ...")
        res["repeated"] = repeated_runs(df, 2 if q else 10, a.jobs)
    if a.only in (None, "hidden"):
        print("C. hidden neurons ...")
        res["hidden"] = hidden_sweep(df, 1 if q else 3, a.jobs)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    make_figures(res)
    print(f"saved {path} and figures/exp_*.png")


if __name__ == "__main__":
    main()
