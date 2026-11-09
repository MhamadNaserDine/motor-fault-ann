"""
ADALINE + LMS (Widrow-Hoff) -- course Chapter 2.

Linear combiner:  u = sum_i w_i x_i - theta      (the error is taken BEFORE the hard limiter)
Error:            e = d - u
LMS / delta rule: Delta w = eta * e * x          (stochastic gradient descent on J = 1/2 E[e^2])
Decision:         y = sign(u)   (or winner-takes-all over the class neurons)

Being linear, ADALINE minimises a bowl-shaped error surface (single global minimum) but can only
draw HYPERPLANES between the classes -> limited accuracy on non-linearly-separable fault data.
"""
from __future__ import annotations

import time
import numpy as np
from .base import NeuralNet, add_bias, encode_targets


class Adaline(NeuralNet):
    name = "ADALINE (LMS)"
    low, high = -1.0, 1.0

    def __init__(self, eta: float = 0.005, epochs: int = 100, seed: int = 0):
        super().__init__()
        self.eta, self.epochs, self.seed = eta, epochs, seed

    def forward(self, X):
        return add_bias(X) @ self.W.T          # linear output u

    def get_weights(self):
        return self.W.copy()

    def set_weights(self, w):
        self.W = w.copy()

    def fit(self, Xtr, ytr, Xte=None, yte=None, n_classes=None, callback=None):
        rng = np.random.default_rng(self.seed)
        n_classes = n_classes or int(ytr.max() + 1)
        Ttr = encode_targets(ytr, n_classes, -1, 1)
        Tte = encode_targets(yte, n_classes, -1, 1) if Xte is not None else None
        self.W = rng.normal(0, 0.05, (Ttr.shape[1], Xtr.shape[1] + 1))
        Xa = add_bias(Xtr)
        t0, best = time.time(), None
        for ep in range(1, self.epochs + 1):
            for n in rng.permutation(len(Xa)):
                u = self.W @ Xa[n]
                e = Ttr[n] - u                          # e = d - u
                self.W += self.eta * np.outer(e, Xa[n])  # LMS:  Delta w = eta e x
            self._log_epoch(ep, Xtr, Ttr, ytr, Xte, Tte, yte, self.eta)
            best = self._early_stopping_update(best)
            if callback:
                callback(self, ep)
        self._finish(best, t0)
        return self
