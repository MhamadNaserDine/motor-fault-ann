"""
train.py
========
End-to-end experiment for the report:

  1. load (or generate) the simulated MCSA dataset
  2. FAULT DETECTION  (healthy vs faulty, 1 output neuron)  and
     FAULT DIAGNOSIS  (5 classes, one output neuron per class)
  3. 3-set cross-validation: training (60 %) / test (20 %, early stopping) / validation (20 %, final score)
  4. train Perceptron, ADALINE, MLP-BP, MLP-momentum, MLP-adaptive η, MLP-LM, RBF
     (+ scikit-learn MLP as a sanity check)
  5. save models (models/), figures (figures/) and metrics + summary (results/)

Run:  python train.py              (uses data/dataset.csv, generates it if missing)
      python train.py --regen --n 150
"""
from __future__ import annotations

import os
import json
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from features import FEATURE_NAMES, StandardScaler
from faults import CLASSES, SHORT
from dataset import generate_dataset, split_3sets, detection_view
from networks import build_model, MODEL_NAMES, metrics, MLP, RBFNet
import plots

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "dataset.csv")
FIG = os.path.join(ROOT, "figures")
MOD = os.path.join(ROOT, "models")
RES = os.path.join(ROOT, "results")

DEFAULT_CFG = dict(n_hidden=15, eta=0.02, alpha=0.8, mu=0.01, epochs=300, n_centers=40, hidden_act="tanh", seed=0)


# ------------------------------------------------------------------ data preparation
def prepare(df: pd.DataFrame, task: str = "diagnosis", seed: int = 0):
    """Split into the 3 sets and normalise (scaler fitted on the TRAINING set only)."""
    if task == "detection":
        df = detection_view(df, seed)
        labels = ["Healthy", "Faulty"]
    else:
        labels = CLASSES
    sets = split_3sets(df, seed=seed)
    scaler = StandardScaler().fit(sets["train"][FEATURE_NAMES].values)
    out = {"task": task, "labels": labels, "n_classes": len(labels), "scaler": scaler, "sets": sets}
    for k, s in sets.items():
        out["X_" + k] = scaler.transform(s[FEATURE_NAMES].values)
        out["y_" + k] = s["label"].values.astype(int)
    return out


def evaluate(model, X, y, n_classes):
    r = metrics(y, model.predict(X), n_classes)
    return r


def train_one(kind, prep, cfg=None, callback=None):
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    m = build_model(kind, **cfg)
    m.fit(prep["X_train"], prep["y_train"], prep["X_test"], prep["y_test"],
          n_classes=prep["n_classes"], callback=callback)
    return m


def result_row(m, prep):
    rv = evaluate(m, prep["X_val"], prep["y_val"], prep["n_classes"])
    rt = evaluate(m, prep["X_test"], prep["y_test"], prep["n_classes"])
    rtr = evaluate(m, prep["X_train"], prep["y_train"], prep["n_classes"])
    return {"model": m.name, "accuracy": rv["accuracy"], "precision": rv["precision"], "recall": rv["recall"],
            "f1": rv["f1"], "train_acc": rtr["accuracy"], "test_acc": rt["accuracy"],
            "train_time_s": m.train_time, "epochs_run": m.epochs_run, "best_epoch_A": m.best_epoch,
            "cm": rv["cm"].tolist(), "per_class_recall": rv["per_class_recall"].tolist()}


def sklearn_check(prep, cfg):
    """Sanity check with a library MLP (same 1-hidden-layer architecture)."""
    from sklearn.neural_network import MLPClassifier
    t = time.time()
    clf = MLPClassifier(hidden_layer_sizes=(cfg["n_hidden"],), activation="tanh", solver="adam",
                        max_iter=2000, early_stopping=False, random_state=cfg["seed"])
    clf.fit(prep["X_train"], prep["y_train"])
    rv = metrics(prep["y_val"], clf.predict(prep["X_val"]), prep["n_classes"])
    return {"model": "scikit-learn MLP (check)", "accuracy": rv["accuracy"], "precision": rv["precision"],
            "recall": rv["recall"], "f1": rv["f1"], "train_acc": clf.score(prep["X_train"], prep["y_train"]),
            "test_acc": clf.score(prep["X_test"], prep["y_test"]), "train_time_s": time.time() - t,
            "epochs_run": clf.n_iter_, "best_epoch_A": None, "cm": rv["cm"].tolist(),
            "per_class_recall": rv["per_class_recall"].tolist()}


def save_bundle(models: dict, prep, path):
    bundle = {"models": models, "scaler": prep["scaler"], "labels": prep["labels"],
              "feature_names": FEATURE_NAMES, "task": prep["task"]}
    with open(path, "wb") as f:
        pickle.dump(bundle, f)


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main experiment
def run_task(df, task, cfg, verbose=True):
    prep = prepare(df, task, cfg["seed"])
    labels = prep["labels"]
    short = SHORT if task == "diagnosis" else ["Healthy", "Faulty"]
    if verbose:
        print(f"\n=== {task.upper()} | train {len(prep['y_train'])}  test {len(prep['y_test'])}  "
              f"validation {len(prep['y_val'])}  | inputs {prep['X_train'].shape[1]}  classes {len(labels)}")
    models, rows = {}, []
    for kind in MODEL_NAMES:
        m = train_one(kind, prep, cfg)
        models[kind] = m
        r = result_row(m, prep)
        r["key"] = kind
        rows.append(r)
        if verbose:
            print(f"  {kind:13s} val acc {r['accuracy']*100:6.2f} %  F1 {r['f1']*100:6.2f} %  "
                  f"time {r['train_time_s']:6.2f}s  epochs {r['epochs_run']:4d}  A={r['best_epoch_A']}")
    sk = sklearn_check(prep, cfg)
    sk["key"] = "sklearn-MLP"
    rows.append(sk)
    if verbose:
        print(f"  {'sklearn-MLP':13s} val acc {sk['accuracy']*100:6.2f} %  F1 {sk['f1']*100:6.2f} %")
    table = pd.DataFrame(rows).set_index("key")

    # save
    save_bundle(models, prep, os.path.join(MOD, f"{task}_bundle.pkl"))
    table.drop(columns=["cm", "per_class_recall"]).to_csv(os.path.join(RES, f"{task}_metrics.csv"))
    with open(os.path.join(RES, f"{task}_results.json"), "w") as f:
        json.dump(table.reset_index().to_dict(orient="records"), f, indent=1, default=float)

    # figures
    savefig(plots.learning_curves_grid({k: m for k, m in models.items() if k != "RBF"},
                                       f"{task.capitalize()}: training / test error and early-stopping point A"),
            f"{task}_learning_curves.png")
    savefig(plots.learning_curve(models["MLP-BP"], f"{task.capitalize()} – MLP-BP: early stopping (point A)"),
            f"{task}_mlp_bp_point_A.png")
    savefig(plots.comparison_bars(table, f"{task.capitalize()}: validation-set metrics"), f"{task}_comparison.png")
    savefig(plots.time_bars(table, f"{task.capitalize()}: training time"), f"{task}_training_time.png")
    savefig(plots.confusion_grid({k: r for k, r in table.iterrows()}, short,
                                 f"{task.capitalize()}: confusion matrices (validation set)"),
            f"{task}_confusion_matrices.png")
    best_key = table.drop(index="sklearn-MLP")["f1"].idxmax()
    savefig(plots.confusion(np.array(table.loc[best_key, "cm"]), short,
                            f"{task.capitalize()} – {best_key} (validation)"), f"{task}_confusion_best.png")
    savefig(plots.adaptive_eta(models["MLP-adaptive"]), f"{task}_adaptive_eta.png")
    savefig(plots.adaptive_eta(models["MLP-LM"], "Levenberg-Marquardt: 1/μ per iteration"), f"{task}_lm_mu.png")
    return prep, models, table


def nonlinear_separability(df, seed=0):
    """2-feature detection problem (healthy vs {broken bars, inter-turn short}) using
    I2/I1 (sees the short) and the lower BRB sideband (sees broken bars).
    The healthy class sits in the 'low-low' corner -> the faulty region is an OR of two
    half-planes (L-shape), which a single-layer network cannot represent (cf. XOR in the course)."""
    from networks import Perceptron, Adaline
    from features import StandardScaler
    feats = ["I2_I1_negseq", "brb_lsb_dB"]
    rng = np.random.default_rng(seed)
    h = df[df.label == 0]
    f = pd.concat([df[df.label == c].sample(len(h) // 2, random_state=seed) for c in (1, 2)])
    d = pd.concat([h, f]).copy()
    d["label"] = (d["label"] > 0).astype(int)
    s = split_3sets(d, seed=seed)
    sc = StandardScaler().fit(s["train"][feats].values)
    X = {k: sc.transform(v[feats].values) for k, v in s.items()}
    y = {k: v["label"].values for k, v in s.items()}
    models = {"Perceptron": Perceptron(0.01, 200, seed), "ADALINE (LMS)": Adaline(0.005, 200, seed),
              "MLP-BP + momentum (10 hidden)": MLP(10, eta=0.05, alpha=0.8, epochs=300, seed=seed)}
    res = {}
    for k, m in models.items():
        m.fit(X["train"], y["train"], X["test"], y["test"], n_classes=2)
        res[k] = float(np.mean(m.predict(X["val"]) == y["val"]))
    Xall = np.vstack([X["train"], X["test"], X["val"]])
    yall = np.concatenate([y["train"], y["test"], y["val"]])
    savefig(plots.decision_regions(models, Xall, yall, feats, labels=("Healthy", "BRB or ITSC"),
                                   title="Healthy vs (broken bars OR inter-turn short) with 2 features"),
            "nonlinear_separability.png")
    return res


def extra_experiments(prep, cfg, df=None):
    """Course illustrations: eta trade-off, momentum effect, RBF vs number of centres,
    non-linear separability (single layer vs MLP)."""
    out = {}
    if df is not None:
        out["nonlinear_2feat_val_acc"] = nonlinear_separability(df, cfg["seed"])
    eta_models = {}
    for eta in (0.001, 0.05, 5.0):   # too small / good / too large (oscillation)
        m = MLP(cfg["n_hidden"], eta=eta, epochs=150, seed=cfg["seed"])
        m.fit(prep["X_train"], prep["y_train"], prep["X_test"], prep["y_test"], n_classes=prep["n_classes"])
        eta_models[f"η = {eta}"] = m
        out[f"eta_{eta}"] = {"final_train_E": m.history["train_E"][-1],
                            "val_acc": float(np.mean(m.predict(prep["X_val"]) == prep["y_val"]))}
    savefig(plots.eta_tradeoff(eta_models), "diagnosis_eta_tradeoff.png")

    mom = {}
    for a in (0.0, 0.5, 0.9):
        m = MLP(cfg["n_hidden"], eta=0.005, alpha=a, epochs=150, seed=cfg["seed"])
        m.fit(prep["X_train"], prep["y_train"], prep["X_test"], prep["y_test"], n_classes=prep["n_classes"])
        mom[f"α = {a}  (η = 0.005)"] = m
    savefig(plots.eta_tradeoff(mom, "Effect of the momentum term α"), "diagnosis_momentum_effect.png")

    Ms = [5, 10, 20, 40, 80, 120, 200, 300]
    tr, te = [], []
    for M in Ms:
        r = RBFNet(M, seed=cfg["seed"]).fit(prep["X_train"], prep["y_train"], n_classes=prep["n_classes"])
        tr.append(1 - np.mean(r.predict(prep["X_train"]) == prep["y_train"]))
        te.append(1 - np.mean(r.predict(prep["X_test"]) == prep["y_test"]))
    savefig(plots.rbf_vs_centers(Ms, tr, te), "diagnosis_rbf_vs_M.png")
    out["rbf_M"] = {"M": Ms, "train_err": tr, "test_err": te}
    return out


def motor_figures():
    """Signals, spectra and Park's vector for the report."""
    from motor_model import MotorParams, simulate
    from faults import make_fault_state
    from features import add_noise
    rng = np.random.default_rng(3)
    mp = MotorParams()
    sig = simulate(mp, make_fault_state(0, 0, mp, rng, natural_spread=False), load=1.0, t_end=1.5,
                   start_from_standstill=True)
    savefig(plots.motor_signals(sig, "Direct-on-line start, rated load (healthy motor)", t_show=1.5),
            "motor_start_up.png")
    sigs = {}
    for c, sev in [(0, 0), (1, 2), (2, 2), (3, 3), (4, 2)]:
        s = simulate(mp, make_fault_state(c, sev, mp, rng), load=0.9, t_end=3.0)
        s = add_noise(s, 0.005, rng)
        sigs[CLASSES[c]] = s
        savefig(plots.current_spectrum(s, c if c else None, s["slip"],
                                       title=f"Current spectrum – {CLASSES[c]} (90 % load)"),
                f"spectrum_{SHORT[c]}.png")
    savefig(plots.park_vector({k: sigs[k] for k in ["Healthy", "Stator inter-turn short", "Supply voltage unbalance"]}),
            "park_vector.png")
    savefig(plots.motor_signals(sigs["Bearing fault"], "Bearing fault – steady state", t_show=0.1),
            "signals_bearing.png")
    savefig(plots.motor_signals(sigs["Broken rotor bars"], "Broken rotor bars – steady state", t_show=0.1),
            "signals_brb.png")


def write_summary(tables, extra, df):
    d, g = tables["detection"], tables["diagnosis"]

    def md(t):
        cols = ["accuracy", "precision", "recall", "f1", "train_time_s", "epochs_run", "best_epoch_A"]
        s = "| Model | Acc % | Prec % | Rec % | F1 % | Time (s) | Epochs | Point A |\n|---|---|---|---|---|---|---|---|\n"
        for k, r in t.iterrows():
            a = "—" if r["best_epoch_A"] is None or pd.isna(r["best_epoch_A"]) else int(r["best_epoch_A"])
            s += (f"| {r['model']} | {r['accuracy']*100:.1f} | {r['precision']*100:.1f} | {r['recall']*100:.1f} | "
                  f"{r['f1']*100:.1f} | {r['train_time_s']:.2f} | {int(r['epochs_run'])} | {a} |\n")
        return s
    txt = f"""# Results summary

Dataset: {len(df)} simulated motors ({df.label.value_counts().min()} per class), loads 30–110 % of rated,
measurement noise ~0.5–1.5 % of RMS, ±5 % parameter spread. 19 MCSA features.
Split: training 60 % / test 20 % (early stopping) / validation 20 % (final, unbiased score).

## Fault detection (healthy vs faulty) – validation set
{md(d)}
## Fault diagnosis (5 classes) – validation set
{md(g)}
## Learning-rate experiment (classic BP, 150 epochs)
""" + "\n".join(f"- {k}: final training cost {v['final_train_E']:.4f}, validation accuracy {v['val_acc']*100:.1f} %"
                for k, v in extra.items() if k.startswith("eta")) + \
        "\n\n## Non-linear separability (2 features: I2/I1 and BRB sideband; healthy vs BRB-or-ITSC), validation accuracy\n" + \
        "\n".join(f"- {k}: {v*100:.1f} %" for k, v in extra.get("nonlinear_2feat_val_acc", {}).items()) + "\n\n## RBF: test error vs number of centres M\n" + \
        "\n".join(f"- M = {M}: training error {a*100:.1f} %, test error {b*100:.1f} %"
                  for M, a, b in zip(extra["rbf_M"]["M"], extra["rbf_M"]["train_err"], extra["rbf_M"]["test_err"]))
    with open(os.path.join(RES, "summary.md"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regen", action="store_true", help="regenerate the dataset")
    ap.add_argument("--n", type=int, default=150, help="samples per class when generating")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--no-motor-figs", action="store_true")
    a = ap.parse_args()
    for d in (FIG, MOD, RES, os.path.dirname(DATA)):
        os.makedirs(d, exist_ok=True)
    if a.regen or not os.path.exists(DATA):
        print("Generating dataset ...")
        df = generate_dataset(a.n, n_jobs=a.jobs, progress=lambda i, n: print(f"\r  {i}/{n}", end="", flush=True))
        df.to_csv(DATA, index=False)
        print()
    df = pd.read_csv(DATA)
    cfg = dict(DEFAULT_CFG)

    savefig(plots.feature_boxplots(df, ["I_unbalance_%", "I2_I1_negseq", "V2_V1_negseq", "park_eccentricity",
                                    "brb_lsb_dB", "brb_usb_dB", "brg_lsb_dB", "brg_usb_dB", "h3_dB"]),
            "feature_boxplots.png")
    tables = {}
    preps = {}
    for task in ("detection", "diagnosis"):
        preps[task], _, tables[task] = run_task(df, task, cfg)
    extra = extra_experiments(preps["diagnosis"], cfg, df)
    with open(os.path.join(RES, "extra_experiments.json"), "w") as f:
        json.dump(extra, f, indent=1, default=float)
    if not a.no_motor_figs:
        motor_figures()
    print("\n" + write_summary(tables, extra, df))


if __name__ == "__main__":
    main()
