"""
MLP trained with the Levenberg-Marquardt algorithm -- course Chapter 3.

Gauss-Newton + gradient-descent hybrid (batch mode, all training patterns at once):

    Delta w = -(J^T J + mu I)^-1 J^T e

    e : error vector of ALL outputs for ALL patterns  (length N*m),  e = y* - y
    J = de/dw : Jacobian of the error vector w.r.t. the weight vector (N*m x P)
    mu large  -> Delta w ~ -(1/mu) J^T e     = gradient descent with eta = 1/mu
    mu small  -> Delta w ~ -(J^T J)^-1 J^T e = Gauss-Newton

Course rule: decrease mu after a successful iteration (cost decreased), increase it otherwise.
"""
from __future__ import annotations

import time
import numpy as np
from .base import add_bias, encode_targets
from .mlp_bp import MLP


class MLP_LM(MLP):
    def __init__(self, n_hidden: int = 12, mu: float = 0.01, mu_dec: float = 0.1, mu_inc: float = 10.0,
                 mu_max: float = 1e10, epochs: int = 100, hidden_act: str = "tanh", seed: int = 0):
        super().__init__(n_hidden=n_hidden, epochs=epochs, hidden_act=hidden_act, seed=seed,
                         name="MLP-Levenberg-Marquardt")
        self.mu0, self.mu_dec, self.mu_inc, self.mu_max = mu, mu_dec, mu_inc, mu_max

    # weight vector <-> matrices
    def _pack(self):
        return np.concatenate([self.W1.ravel(), self.W2.ravel()])

    def _unpack(self, w):
        n1 = self.W1.size
        self.W1 = w[:n1].reshape(self.W1.shape).copy()
        self.W2 = w[n1:].reshape(self.W2.shape).copy()

    def jacobian(self, X, T):
        """Analytic Jacobian J = de/dw (rows: pattern n, output j; columns: all weights)."""
        yj, yk, Xa, yka = self.forward(X, return_hidden=True)
        N, m = yj.shape
        H = self.n_hidden
        do = self.phi_o.df(yj)                          # phi_o'(v_j)   (N x m)
        dh = self.phi_h.df(yk)                          # phi_h'(v_k)   (N x H)
        n_in1 = Xa.shape[1]
        J1 = np.zeros((N, m, H, n_in1))
        J2 = np.zeros((N, m, m, H + 1))
        for j in range(m):
            # dy_j/dW2[j,:] = phi_o'(v_j) * [-1, y_k]
            J2[:, j, j, :] = do[:, j:j + 1] * yka
            # dy_j/dW1[k,i] = phi_o'(v_j) * w_jk * phi_h'(v_k) * x_i
            g = do[:, j:j + 1] * self.W2[j, 1:][None, :] * dh          # N x H
            J1[:, j] = g[:, :, None] * Xa[:, None, :]
        Jy = np.concatenate([J1.reshape(N * m, -1), J2.reshape(N * m, -1)], axis=1)
        e = (T - yj).reshape(-1)
        return -Jy, e                                   # de/dw = -dy/dw

    def fit(self, Xtr, ytr, Xte=None, yte=None, n_classes=None, callback=None):
        rng = np.random.default_rng(self.seed)
        n_classes = n_classes or int(ytr.max() + 1)
        Ttr = encode_targets(ytr, n_classes)
        Tte = encode_targets(yte, n_classes) if Xte is not None else None
        self.init_weights(Xtr.shape[1], Ttr.shape[1], rng)
        mu = self.mu0
        t0, best = time.time(), None
        self.mu_history = []
        for ep in range(1, self.epochs + 1):
            J, e = self.jacobian(Xtr, Ttr)
            E = 0.5 * e @ e
            w = self._pack()
            JtJ, Jte = J.T @ J, J.T @ e
            I = np.eye(len(w))
            while True:
                dw = -np.linalg.solve(JtJ + mu * I, Jte)    # Delta w = -(J^T J + mu I)^-1 J^T e
                self._unpack(w + dw)
                e_new = (Ttr - self.forward(Xtr)).reshape(-1)
                if 0.5 * e_new @ e_new < E:                  # success -> accept, decrease mu
                    mu = max(mu * self.mu_dec, 1e-12)
                    break
                mu *= self.mu_inc                           # failure -> reject, increase mu
                if mu > self.mu_max:
                    self._unpack(w)
                    break
            self.mu_history.append(mu)
            self._log_epoch(ep, Xtr, Ttr, ytr, Xte, Tte, yte, 1.0 / mu)  # equivalent eta = 1/mu
            best = self._early_stopping_update(best)
            if callback:
                callback(self, ep)
            if mu > self.mu_max:
                break
        self._finish(best, t0)
        return self
