"""
Single-layer Perceptron (Rosenblatt) -- course Chapter 2.

Neuron j:   y_j = sign(v_j),  v_j = sum_i w_ji x_i - theta_j      (hard limiter, outputs +/-1)
Learning:   Delta w_ji = eta * (d_j - y_j) * x_i                     (error-correction rule)

One neuron per class ("one-vs-all"); for detection a single neuron (healthy = -1, faulty = +1).
The convergence theorem only holds for LINEARLY SEPARABLE classes: on our motor data the
error does not go to zero -> this is the baseline that motivates the MLP.
"""
from __future__ import annotations

import time
import numpy as np
from .base import NeuralNet, add_bias, encode_targets


class Perceptron(NeuralNet):
    name = "Perceptron"
    low, high = -1.0, 1.0

    def __init__(self, eta: float = 0.01, epochs: int = 100, seed: int = 0):
        super().__init__()
        self.eta, self.epochs, self.seed = eta, epochs, seed

    def net_input(self, X):
        return add_bias(X) @ self.W.T                      # v = W [-1, x]

    def forward(self, X):
        # for the cost / decision we use the sign output; argmax on v breaks ties for one-vs-all
        return np.where(self.net_input(X) >= 0, 1.0, -1.0)

    def predict(self, X):
        v = self.net_input(X)
        return (v[:, 0] >= 0).astype(int) if v.shape[1] == 1 else v.argmax(axis=1)

    def get_weights(self):
        return self.W.copy()

    def set_weights(self, w):
        self.W = w.copy()

    def fit(self, Xtr, ytr, Xte=None, yte=None, n_classes=None, callback=None):
        rng = np.random.default_rng(self.seed)
        n_classes = n_classes or int(max(ytr.max(), (yte.max() if yte is not None else 0)) + 1)
        Ttr = encode_targets(ytr, n_classes, -1, 1)
        Tte = encode_targets(yte, n_classes, -1, 1) if Xte is not None else None
        m = Ttr.shape[1]
        self.W = rng.normal(0, 0.1, (m, Xtr.shape[1] + 1))
        Xa = add_bias(Xtr)
        t0, best = time.time(), None
        for ep in range(1, self.epochs + 1):
            for n in rng.permutation(len(Xa)):           # sequential (pattern-by-pattern) mode
                y = np.where(self.W @ Xa[n] >= 0, 1.0, -1.0)
                e = Ttr[n] - y                                # e = d - y  (0 or +/-2)
                self.W += self.eta * np.outer(e, Xa[n])        # Delta w = eta (d - y) x
            self._log_epoch(ep, Xtr, Ttr, ytr, Xte, Tte, yte, self.eta)
            best = self._early_stopping_update(best)
            if callback:
                callback(self, ep)
        self._finish(best, t0)
        return self
