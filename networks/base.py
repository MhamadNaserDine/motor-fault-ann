"""
Common pieces shared by all networks: activation functions, target coding, metrics,
and a small base class implementing early stopping on the TEST set.

Course conventions used everywhere
----------------------------------
* neuron:            y = phi(v),   v = sum_i w_i x_i - theta
* the threshold theta is handled as a weight w_0 = theta applied to a constant input x_0 = -1
* instantaneous cost: E(n) = 1/2 * sum_j e_j(n)^2 ,  e_j = y*_j - y_j
"""
from __future__ import annotations

import time
import pickle
import numpy as np


# ------------------------------------------------------------------ activations
class Logistic:
    """phi(v) = 1 / (1 + exp(-a v))     phi'(v) = a * y * (1 - y)"""
    name = "logistic"

    def __init__(self, a: float = 1.0):
        self.a = a

    def f(self, v):
        return 1.0 / (1.0 + np.exp(-self.a * np.clip(v, -50, 50)))

    def df(self, y):           # derivative expressed with the OUTPUT y (cheap)
        return self.a * y * (1.0 - y)


class Tanh:
    """phi(v) = a tanh(b v)            phi'(v) = (b/a) (a - y)(a + y)"""
    name = "tanh"

    def __init__(self, a: float = 1.0, b: float = 1.0):
        self.a, self.b = a, b

    def f(self, v):
        return self.a * np.tanh(self.b * v)

    def df(self, y):
        return (self.b / self.a) * (self.a - y) * (self.a + y)


ACTIVATIONS = {"logistic": Logistic, "tanh": Tanh}


def add_bias(X):
    """Prepend the constant input x_0 = -1 (so that w_0 = theta, the threshold)."""
    return np.hstack([-np.ones((X.shape[0], 1)), X])


def encode_targets(y, n_classes, low=0.0, high=1.0):
    """Desired outputs y*: 1 output neuron for 2 classes, one-hot (one neuron per class) otherwise."""
    y = np.asarray(y, int)
    if n_classes == 2:
        return np.where(y[:, None] == 1, high, low).astype(float)
    T = np.full((len(y), n_classes), low, float)
    T[np.arange(len(y)), y] = high
    return T


def decode_outputs(Y, low=0.0, high=1.0):
    """Network outputs -> class index (threshold for 1 output, winner-takes-all otherwise)."""
    if Y.shape[1] == 1:
        return (Y[:, 0] > (low + high) / 2).astype(int)
    return Y.argmax(axis=1)


def outputs_to_confidence(Y, low=0.0, high=1.0):
    """Turn raw outputs into a probability-like confidence vector (for the GUI)."""
    Z = (Y - low) / (high - low)
    Z = np.clip(Z, 1e-6, None)
    if Z.shape[1] == 1:
        p = np.clip(Z[:, 0], 0, 1)
        return np.column_stack([1 - p, p])
    return Z / Z.sum(axis=1, keepdims=True)


# ------------------------------------------------------------------ metrics
def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def metrics(y_true, y_pred, n_classes):
    """accuracy + macro-averaged precision / recall / F1 (+ per-class values)."""
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    if n_classes == 2:   # detection: report the "faulty" class (positive class = 1)
        P, R, F = prec[1], rec[1], f1[1]
    else:
        P, R, F = prec.mean(), rec.mean(), f1.mean()
    return {"accuracy": tp.sum() / cm.sum(), "precision": P, "recall": R, "f1": F,
            "per_class_precision": prec, "per_class_recall": rec, "per_class_f1": f1, "cm": cm}


# ------------------------------------------------------------------ base class
class NeuralNet:
    """Base class: stores training history and implements early stopping on the test set."""
    name = "net"
    low, high = 0.0, 1.0   # target coding levels

    def __init__(self):
        self.history = {"epoch": [], "train_E": [], "test_E": [], "train_acc": [], "test_acc": [], "eta": []}
        self.best_epoch = None
        self.train_time = 0.0
        self.epochs_run = 0

    # to implement in subclasses
    def forward(self, X):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def set_weights(self, w):
        raise NotImplementedError

    # common helpers
    def cost(self, X, T):
        """Mean over the patterns of E(n) = 1/2 sum_j e_j^2."""
        e = T - self.forward(X)
        return 0.5 * np.mean(np.sum(e ** 2, axis=1))

    def predict(self, X):
        return decode_outputs(self.forward(X), self.low, self.high)

    def predict_proba(self, X):
        return outputs_to_confidence(self.forward(X), self.low, self.high)

    def _log_epoch(self, ep, Xtr, Ttr, ytr, Xte, Tte, yte, eta=np.nan):
        h = self.history
        h["epoch"].append(ep)
        h["train_E"].append(self.cost(Xtr, Ttr))
        h["train_acc"].append(np.mean(self.predict(Xtr) == ytr))
        if Xte is not None:
            h["test_E"].append(self.cost(Xte, Tte))
            h["test_acc"].append(np.mean(self.predict(Xte) == yte))
        h["eta"].append(eta)

    def _early_stopping_update(self, best):
        """Keep the weights with the minimum TEST error ('point A' of the course figure)."""
        h = self.history
        cur = h["test_E"][-1] if h["test_E"] else h["train_E"][-1]
        if best is None or cur < best[0]:
            return (cur, h["epoch"][-1], self.get_weights())
        return best

    def _finish(self, best, t0):
        if best is not None:
            self.set_weights(best[2])
            self.best_epoch = best[1]
        self.train_time = time.time() - t0
        self.epochs_run = self.history["epoch"][-1] if self.history["epoch"] else 0

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)
