"""
plots.py -- all matplotlib figures (used by train.py to save PNGs for the report and by app.py).
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faults import CLASSES, SHORT, fault_frequencies
from features import spectrum

COLORS = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#8d5ac4", "#f4a261", "#457b9d", "#6c757d"]
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def motor_signals(sig, title="", t_show=0.2):
    """Stator currents (zoom), speed and electromagnetic torque."""
    t = sig["t"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 6.5), sharex=False)
    m = t <= t[0] + t_show
    for k, (lab, c) in enumerate(zip("abc", COLORS)):
        ax[0].plot(t[m], sig["i_abc"][k][m], color=c, lw=1, label=f"i_{lab}")
    ax[0].set_ylabel("Stator current (A)")
    ax[0].legend(loc="upper right", ncol=3)
    ax[0].set_title(title or "Stator currents (first %.0f ms)" % (t_show * 1000))
    ax[1].plot(t, sig["speed_rpm"], color=COLORS[2], lw=1)
    ax[1].set_ylabel("Speed (rpm)")
    ax[2].plot(t, sig["torque"], color=COLORS[1], lw=0.8, label="T_e")
    ax[2].plot(t, sig["load_torque"], color="k", lw=0.8, ls="--", label="T_load")
    ax[2].set_ylabel("Torque (N·m)")
    ax[2].set_xlabel("Time (s)")
    ax[2].legend(loc="upper right")
    fig.tight_layout()
    return fig


def current_spectrum(sig, fault=None, slip=None, fmax=200, title=""):
    """Phase-a current spectrum (dB rel. fundamental) with the theoretical fault frequencies marked."""
    f, A = spectrum(sig["i_abc"][0], sig["f_sample"])
    A_db = 20 * np.log10(np.maximum(A, 1e-12) / A.max())
    fig, ax = plt.subplots(figsize=(9, 3.6))
    m = f <= fmax
    ax.plot(f[m], A_db[m], color=COLORS[2], lw=0.8)
    ax.set_ylim(-110, 5)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    ax.set_title(title or "Stator current spectrum (MCSA)")
    if fault and slip is not None:
        for fr in fault_frequencies(fault, sig["f_s"], slip):
            if fr <= fmax:
                ax.axvline(fr, color=COLORS[1], ls="--", lw=1, alpha=0.8)
                ax.text(fr, 0, f"{fr:.1f}", color=COLORS[1], fontsize=7, ha="center", va="bottom", rotation=90)
        ax.plot([], [], color=COLORS[1], ls="--", label=f"{CLASSES[fault]} frequencies")
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def park_vector(sigs: dict, title="Park's vector pattern"):
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    for k, (lab, sig) in enumerate(sigs.items()):
        i3 = sig["i_abc"]
        iD = np.sqrt(2 / 3) * i3[0] - (1 / np.sqrt(6)) * (i3[1] + i3[2])
        iQ = (1 / np.sqrt(2)) * (i3[1] - i3[2])
        ax.plot(iD, iQ, lw=0.6, color=COLORS[k % len(COLORS)], label=lab)
    ax.set_aspect("equal")
    ax.set_xlabel("i_D (A)")
    ax.set_ylabel("i_Q (A)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def learning_curve(model, title=None):
    """Training and test error vs epochs, with early-stopping point A (test-error minimum)."""
    h = model.history
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ep = np.array(h["epoch"])
    ax.plot(ep, h["train_E"], color=COLORS[2], lw=1.4, label="Training set error")
    if h["test_E"]:
        ax.plot(ep, h["test_E"], color=COLORS[1], lw=1.4, label="Test set error")
        if model.best_epoch:
            i = list(ep).index(model.best_epoch)
            ax.plot(ep[i], h["test_E"][i], "o", ms=8, mfc="none", mec="k", mew=1.5)
            ax.annotate(f"A (epoch {ep[i]})", (ep[i], h["test_E"][i]), textcoords="offset points",
                        xytext=(8, 10), fontsize=8)
            ax.axvline(ep[i], color="k", ls=":", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean cost  E = ½ Σ e²")
    ax.set_title(title or model.name)
    ax.legend()
    fig.tight_layout()
    return fig


def learning_curves_grid(models: dict, title=""):
    n = len(models)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, m) in zip(axes, models.items()):
        h = m.history
        ep = np.array(h["epoch"])
        ax.plot(ep, h["train_E"], color=COLORS[2], label="training")
        ax.plot(ep, h["test_E"], color=COLORS[1], label="test")
        if m.best_epoch and len(ep) > 1:
            i = list(ep).index(m.best_epoch)
            ax.plot(ep[i], h["test_E"][i], "ko", mfc="none", ms=7)
            ax.annotate("A", (ep[i], h["test_E"][i]), xytext=(4, 6), textcoords="offset points")
        ax.set_yscale("log")
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def confusion(cm, labels, title="", ax=None):
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    if own:
        fig.tight_layout()
        return fig


def confusion_grid(results: dict, labels, title=""):
    n = len(results)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(13, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, r) in zip(axes, results.items()):
        confusion(np.array(r["cm"]), labels, f"{name}\nacc = {r['accuracy']*100:.1f} %", ax=ax)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout(h_pad=3.0)
    return fig


def comparison_bars(table, title=""):
    """table: pandas DataFrame indexed by model with accuracy/precision/recall/f1 columns."""
    mets = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    x = np.arange(len(table))
    w = 0.2
    for k, m in enumerate(mets):
        ax.bar(x + (k - 1.5) * w, table[m] * 100, w, label=m, color=COLORS[k])
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=20, ha="right")
    ax.set_ylabel("%  (validation set)")
    lo = max(0, (table[mets].values.min() * 100) - 10)
    ax.set_ylim(lo, 101)
    ax.legend(ncol=4, loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def time_bars(table, title="Training time"):
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(table.index, table["train_time_s"], color=COLORS[6])
    ax.set_yscale("log")
    ax.set_ylabel("seconds (log)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def eta_tradeoff(models: dict, title="Learning-rate trade-off (classic BP)"):
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k, (lab, m) in enumerate(models.items()):
        ax.plot(m.history["epoch"], m.history["train_E"], color=COLORS[k], label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training cost E")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def adaptive_eta(model, title="Adaptive learning rate η(n)"):
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(model.history["epoch"], model.history["eta"], color=COLORS[4])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("η(n)" if "Levenberg" not in model.name else "1/μ")
    ax.set_yscale("log")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def rbf_vs_centers(Ms, tr_err, te_err, title="RBF: error vs number of hidden neurons M"):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(Ms, np.array(tr_err) * 100, "o-", color=COLORS[2], label="training set")
    ax.plot(Ms, np.array(te_err) * 100, "s-", color=COLORS[1], label="test set")
    ax.set_xlabel("M (number of Gaussian neurons)")
    ax.set_ylabel("Classification error (%)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def feature_boxplots(df, feats, title="Feature distribution per class"):
    n = len(feats)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.8 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        data = [df.loc[df.label == c, f].values for c in range(len(CLASSES))]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], COLORS):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(CLASSES) + 1))
        ax.set_xticklabels(SHORT, fontsize=7)
        ax.set_title(f, fontsize=9)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def class_distribution(df, title="Class distribution"):
    fig, ax = plt.subplots(figsize=(6, 3))
    counts = df["class"].value_counts().reindex([c for c in CLASSES if c in set(df["class"])])
    ax.bar(range(len(counts)), counts.values, color=COLORS[:len(counts)])
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels([SHORT[CLASSES.index(c)] for c in counts.index])
    ax.set_ylabel("samples")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def confidence_bars(proba, labels, title="Network output (confidence)"):
    fig, ax = plt.subplots(figsize=(6, 2.6))
    order = np.argsort(proba)
    ax.barh(np.array(labels)[order], np.array(proba)[order] * 100,
            color=[COLORS[0] if i == order[-1] else "#bbbbbb" for i in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("%")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def decision_regions(models: dict, X, y, feat_names, scaler=None, labels=("Healthy", "Faulty"),
                     title="Decision regions: linear vs non-linear separation"):
    """2-D decision regions (inputs normalised) of several trained networks side by side."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 4))
    axes = np.atleast_1d(axes)
    pad = 0.5
    x0, x1 = X[:, 0].min() - pad, X[:, 0].max() + pad
    y0, y1 = X[:, 1].min() - pad, X[:, 1].max() + pad
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    G = np.column_stack([gx.ravel(), gy.ravel()])
    for ax, (name, m) in zip(axes, models.items()):
        Z = m.predict(G).reshape(gx.shape)
        ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5], colors=["#cfe8e4", "#f7d3c8"], alpha=0.9)
        ax.contour(gx, gy, Z, levels=[0.5], colors="k", linewidths=1)
        for c, (lab, col, mk) in enumerate(zip(labels, [COLORS[0], COLORS[1]], ["o", "^"])):
            ax.scatter(X[y == c, 0], X[y == c, 1], s=12, c=col, marker=mk, edgecolors="k", linewidths=0.3, label=lab)
        acc = np.mean(m.predict(X) == y) * 100
        ax.set_title(f"{name}\naccuracy {acc:.1f} %", fontsize=9)
        ax.set_xlabel(feat_names[0] + " (normalised)")
        ax.set_ylabel(feat_names[1] + " (normalised)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(False)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def add_verdict(fig, text, faulty):
    """Write the diagnosis in a coloured banner on top of a figure (green = healthy, red = faulty)."""
    color = "#c0392b" if faulty else "#2a9d8f"
    fig.suptitle(text, color="white", fontsize=12, fontweight="bold", y=0.995,
                 bbox=dict(boxstyle="round,pad=0.45", fc=color, ec="none"))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ====================================================================== experiments (experiments.py)
MODEL_ORDER = ["Perceptron", "ADALINE", "MLP-BP", "MLP-momentum", "MLP-adaptive", "MLP-LM", "RBF"]
MODEL_COLORS = dict(zip(MODEL_ORDER, ["#9aa5b1", "#c9b37e", "#264653", "#2a9d8f", "#8d5ac4", "#e76f51", "#e9c46a"]))


def repeated_runs_box(runs, task, title=None):
    """runs: list of dicts {model, seed, accuracy, ...}. Box plot of validation accuracy over the seeds."""
    import pandas as pd
    df = pd.DataFrame(runs)
    df = df[df.task == task]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    data = [df.loc[df.model == m, "accuracy"].values * 100 for m in MODEL_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, medianprops=dict(color="k"))
    for patch, m in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.85)
    for k, d in enumerate(data, 1):
        ax.scatter(np.random.default_rng(k).normal(k, 0.06, len(d)), d, s=10, color="k", alpha=0.5, zorder=3)
        ax.text(k, np.min(d) - 1.2, f"{d.mean():.1f}±{d.std(ddof=1):.1f}",
                ha="center", va="top", fontsize=7.5)
    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels(MODEL_ORDER, rotation=15)
    ax.set_ylabel("Validation accuracy (%)")
    lo = min(np.min(d) for d in data)
    ax.set_ylim(lo - 4, 100.5)
    ax.set_title(title or f"{task.capitalize()}: accuracy over {df.seed.nunique()} independent runs "
                          f"(new split + new initial weights each run)")
    fig.tight_layout()
    return fig


def hidden_sweep_plot(sweep, title="MLP size vs generalisation (number of hidden neurons)"):
    """sweep: list of dicts {H, seed, train_E_final, test_E_final, test_E_best, val_acc, train_acc}."""
    import pandas as pd
    df = pd.DataFrame(sweep).groupby("H").agg(["mean", "std"])
    H = df.index.values
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for col, lab, c in [("train_E_final", "training set (end of training)", COLORS[2]),
                        ("test_E_final", "test set (end of training, no early stopping)", COLORS[1]),
                        ("test_E_best", "test set at point A (early stopping)", COLORS[0])]:
        m, s = df[(col, "mean")], df[(col, "std")].fillna(0)
        ax[0].plot(H, m, "o-", color=c, label=lab)
        ax[0].fill_between(H, m - s, m + s, color=c, alpha=0.15)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("hidden neurons H")
    ax[0].set_ylabel("mean cost E = ½Σe²")
    ax[0].set_title("Error vs network size")
    ax[0].legend(fontsize=7)
    for col, lab, c in [("train_acc", "training accuracy", COLORS[2]), ("val_acc", "validation accuracy", COLORS[1])]:
        m, s = df[(col, "mean")] * 100, df[(col, "std")].fillna(0) * 100
        ax[1].plot(H, m, "o-", color=c, label=lab)
        ax[1].fill_between(H, m - s, m + s, color=c, alpha=0.15)
    va = df[("val_acc", "mean")]
    plateau = va.index[va >= va.max() - 0.01].min()          # smallest network within 1 point of the best
    ax[1].axvline(plateau, color="k", ls=":", lw=1)
    ax[1].annotate(f"plateau from H ≈ {plateau}", (plateau, va.loc[plateau] * 100), xytext=(8, -26),
                   textcoords="offset points", fontsize=8)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("hidden neurons H")
    ax[1].set_ylabel("%")
    ax[1].set_title("Accuracy vs network size")
    ax[1].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def noise_robustness_plot(rows, title="Robustness to sensor noise (fresh test motors, networks trained at 0.25–2 %)"):
    """rows: list of dicts {model, noise, accuracy}."""
    import pandas as pd
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for m in MODEL_ORDER:
        d = df[df.model == m].sort_values("noise")
        if len(d):
            ax.plot(d.noise * 100, d.accuracy * 100, "o-", color=MODEL_COLORS[m], label=m,
                    lw=2.2 if m in ("MLP-BP", "MLP-LM") else 1.2)
    ax.axvspan(0.25, 2.0, color="#2a9d8f", alpha=0.08)
    ax.text(1.1, 22, "training\nnoise range", ha="center", fontsize=7, color="#2a9d8f")
    ax.set_xlabel("sensor noise (% of RMS)")
    ax.set_ylabel("diagnosis accuracy (%)")
    ax.set_ylim(0, 101)
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


def breakdown_plot(rows, by, labels, title):
    """rows: list of dicts {model, <by>, accuracy}; grouped bars for the main networks."""
    import pandas as pd
    df = pd.DataFrame(rows)
    models = [m for m in ["Perceptron", "ADALINE", "MLP-BP", "MLP-LM", "RBF"] if m in set(df.model)]
    keys = sorted(df[by].unique())
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(models)
    x = np.arange(len(keys))
    for k, m in enumerate(models):
        d = df[df.model == m].set_index(by).reindex(keys)
        ax.bar(x + (k - (len(models) - 1) / 2) * w, d.accuracy * 100, w, color=MODEL_COLORS[m], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(k, str(k)) for k in keys])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=5, fontsize=7, loc="lower right")
    ax.set_title(title)
    fig.tight_layout()
    return fig
