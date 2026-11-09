"""
Radial Basis Function network, FIXED-CENTERS method -- course Chapter 3.

Hidden neuron j (Gaussian, isotropic):
    phi_j(x) = exp( -(M / d^2) * ||x - t_j||^2 )        i.e. sigma = d / sqrt(2M)
    t_j : centres chosen RANDOMLY among the training vectors, j = 1..M
    d   : maximum distance between the selected centres
Output layer (linear):  y = sum_j w_j phi_j(x)  (- theta, optional bias)
Weights by the PSEUDO-INVERSE:   G w = y*   ->   w = G+ y*     with g_ij = phi_j(x_i)

No iterative learning -> training is one matrix operation (very fast), but each hidden neuron
only "sees" a local region (local approximation) -> needs more neurons than the MLP.
"""
from __future__ import annotations

import time
import numpy as np
from .base import NeuralNet, encode_targets


class RBFNet(NeuralNet):
    name = "RBF (fixed centres + pseudo-inverse)"
    low, high = 0.0, 1.0

    def __init__(self, n_centers: int = 60, use_bias: bool = True, seed: int = 0):
        super().__init__()
        self.M, self.use_bias, self.seed = n_centers, use_bias, seed

    def design_matrix(self, X):
        d2 = ((X[:, None, :] - self.centers[None, :, :]) ** 2).sum(-1)    # ||x_i - t_j||^2
        G = np.exp(-(self.M / self.d ** 2) * d2)                           # g_ij = phi_j(x_i)
        if self.use_bias:
            G = np.hstack([G, -np.ones((len(X), 1))])
        return G

    def forward(self, X):
        return self.design_matrix(X) @ self.W

    def get_weights(self):
        return self.W.copy()

    def set_weights(self, w):
        self.W = w.copy()

    def fit(self, Xtr, ytr, Xte=None, yte=None, n_classes=None, callback=None):
        rng = np.random.default_rng(self.seed)
        t0 = time.time()
        n_classes = n_classes or int(ytr.max() + 1)
        Ttr = encode_targets(ytr, n_classes)
        Tte = encode_targets(yte, n_classes) if Xte is not None else None
        M = min(self.M, len(Xtr))
        self.M = M
        self.centers = Xtr[rng.choice(len(Xtr), M, replace=False)]           # random training vectors
        dc = np.sqrt(((self.centers[:, None] - self.centers[None]) ** 2).sum(-1))
        self.d = dc.max()                                                     # max distance between centres
        self.sigma = self.d / np.sqrt(2 * M)                                  # sigma = d / sqrt(2M)
        G = self.design_matrix(Xtr)
        self.W = np.linalg.pinv(G) @ Ttr                                      # w = G+ y*
        self._log_epoch(1, Xtr, Ttr, ytr, Xte, Tte, yte)
        self.best_epoch = 1
        self.train_time = time.time() - t0
        self.epochs_run = 1
        if callback:
            callback(self, 1)
        return self
