"""
dataset.py
==========
Generates the labelled MCSA dataset by running many motor simulations
(random fault class, severity, load and noise), and splits it into the three
sets of the course's cross-validation protocol:

    TRAINING set   -> used to adjust the weights
    TEST set       -> learning is suspended periodically and the test error is measured;
                      used for EARLY STOPPING (keep the weights at the test-error minimum, "point A")
    VALIDATION set -> never seen during learning; final, unbiased evaluation of the network
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from motor_model import MotorParams, simulate
from faults import CLASSES, make_fault_state
from features import FEATURE_NAMES, add_noise, extract_features


def _one_sample(args):
    fault, severity, load, noise, seed, t_end = args
    rng = np.random.default_rng(seed)
    mp = MotorParams()
    # +/-5 % parameter spread: every motor is slightly different (and gets warm -> resistances change)
    mp.Rs *= rng.uniform(0.95, 1.05)
    mp.Rr *= rng.uniform(0.95, 1.05)
    fs = make_fault_state(fault, severity, mp, rng)
    sig = simulate(mp, fs, load=load, t_end=t_end)
    sig = add_noise(sig, noise, rng)
    return extract_features(sig, mp)


def make_jobs(n_per_class: int, noise: float = 0.01, load_range=(0.3, 1.1), seed: int = 0, t_end: float = 3.0):
    rng = np.random.default_rng(seed)
    jobs, meta = [], []
    for c in range(len(CLASSES)):
        for k in range(n_per_class):
            sev = 0 if c == 0 else int(k % 3) + 1            # balanced severities 1,2,3
            load = float(rng.uniform(*load_range))
            nz = float(noise * rng.uniform(0.5, 2.0))
            jobs.append((c, sev, load, nz, int(rng.integers(1 << 31)), t_end))
            meta.append((c, sev, load, nz))
    return jobs, meta


def generate_dataset(n_per_class: int = 150, noise: float = 0.01, load_range=(0.3, 1.1), seed: int = 0,
                     n_jobs: int = 1, t_end: float = 3.0, progress=None) -> pd.DataFrame:
    """Return a DataFrame: one row per simulated motor, FEATURE_NAMES + label, severity, load, noise."""
    jobs, meta = make_jobs(n_per_class, noise, load_range, seed, t_end)
    feats = []
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for i, f in enumerate(ex.map(_one_sample, jobs, chunksize=4)):
                feats.append(f)
                if progress:
                    progress(i + 1, len(jobs))
    else:
        for i, j in enumerate(jobs):
            feats.append(_one_sample(j))
            if progress:
                progress(i + 1, len(jobs))
    df = pd.DataFrame(np.array(feats), columns=FEATURE_NAMES)
    m = np.array(meta)
    df["label"] = m[:, 0].astype(int)
    df["class"] = [CLASSES[i] for i in df["label"]]
    df["severity"] = m[:, 1].astype(int)
    df["load"] = m[:, 2]
    df["noise"] = m[:, 3]
    return df


def split_3sets(df: pd.DataFrame, frac=(0.6, 0.2, 0.2), seed: int = 0):
    """Stratified split (same class proportions in every set) into training / test / validation."""
    rng = np.random.default_rng(seed)
    idx = {"train": [], "test": [], "val": []}
    for c in sorted(df["label"].unique()):
        ids = rng.permutation(np.where(df["label"].values == c)[0])
        n = len(ids)
        n_tr, n_te = int(round(frac[0] * n)), int(round(frac[1] * n))
        idx["train"] += list(ids[:n_tr])
        idx["test"] += list(ids[n_tr:n_tr + n_te])
        idx["val"] += list(ids[n_tr + n_te:])
    return {k: df.iloc[rng.permutation(v)].reset_index(drop=True) for k, v in idx.items()}


def detection_view(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Balanced binary dataset for FAULT DETECTION: all healthy rows + the same number of faulty
    rows (equally drawn from every fault class). label: 0 = healthy, 1 = faulty."""
    rng = np.random.default_rng(seed)
    h = df[df.label == 0]
    n_per = int(np.ceil(len(h) / (df.label.nunique() - 1)))
    f = pd.concat([g.sample(min(n_per, len(g)), random_state=int(rng.integers(1e6)))
                   for c, g in df[df.label > 0].groupby("label")])
    f = f.iloc[:len(h)]
    out = pd.concat([h, f]).copy()
    out["label"] = (out["label"] > 0).astype(int)
    out["class"] = np.where(out["label"] == 0, "Healthy", "Faulty")
    return out.reset_index(drop=True)


if __name__ == "__main__":
    import time, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="samples per class")
    ap.add_argument("--noise", type=float, default=0.01)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--out", default="data/dataset.csv")
    a = ap.parse_args()
    t = time.time()
    df = generate_dataset(a.n, a.noise, n_jobs=a.jobs,
                          progress=lambda i, n: print(f"\r{i}/{n}", end="", flush=True))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nsaved {a.out}: {df.shape} in {time.time()-t:.0f}s")
