"""
Multilayer Perceptron trained with the Error Back-Propagation Algorithm (BPA) -- course Chapter 3.

Architecture: inputs x_i -> hidden neurons k (phi_h) -> output neurons j (phi_o, logistic)
    v_k = sum_i w_ki x_i - theta_k ,  y_k = phi_h(v_k)
    v_j = sum_k w_jk y_k - theta_j ,  y_j = phi_o(v_j)
Cost:  E(n) = 1/2 sum_j e_j(n)^2 ,  e_j = y*_j - y_j

Back-propagation (sequential / pattern-by-pattern mode):
    output layer:  delta_j = e_j * phi'(v_j)                       Delta w_jk = eta delta_j y_k
    hidden layer:  delta_k = (sum_j delta_j w_jk) * phi'(v_k)      Delta w_ki = eta delta_k x_i

Improvements (slides "Improvement of the learning algorithm"):
    momentum:          Delta w(n) = eta delta y + alpha Delta w(n-1)          (alpha > 0)
    adaptive eta(n):   Delta w(n) = eta(n) delta y [+ alpha Delta w(n-1)]
        rule used here (evaluated once per epoch on the training cost E):
          E decreased              -> eta <- eta * eta_inc   (go faster)
          E increased > max_ratio  -> reject the epoch (restore weights), eta <- eta * eta_dec,
                                      reset the momentum memory (we were oscillating)
"""
from __future__ import annotations

import time
import numpy as np
from .base import NeuralNet, ACTIVATIONS, Logistic, add_bias, encode_targets


class MLP(NeuralNet):
    low, high = 0.0, 1.0

    def __init__(self, n_hidden: int = 12, eta: float = 0.1, alpha: float = 0.0, adaptive: bool = False,
                 epochs: int = 300, hidden_act: str = "tanh", eta_inc: float = 1.05, eta_dec: float = 0.7,
                 max_ratio: float = 1.04, seed: int = 0, name: str | None = None):
        super().__init__()
        self.n_hidden, self.eta, self.alpha, self.adaptive = n_hidden, eta, alpha, adaptive
        self.epochs, self.seed = epochs, seed
        self.eta_inc, self.eta_dec, self.max_ratio = eta_inc, eta_dec, max_ratio
        self.phi_h = ACTIVATIONS[hidden_act]()
        self.phi_o = Logistic()
        self.name = name or ("MLP-BP" + (" + momentum" if alpha > 0 else "") + (" + adaptive η" if adaptive else ""))

    # ---------------------------------------------------------------- forward
    def init_weights(self, n_in, n_out, rng):
        # small random weights (uniform +/- 1/sqrt(fan-in)) so that neurons start in the linear zone
        self.W1 = rng.uniform(-1, 1, (self.n_hidden, n_in + 1)) / np.sqrt(n_in + 1)
        self.W2 = rng.uniform(-1, 1, (n_out, self.n_hidden + 1)) / np.sqrt(self.n_hidden + 1)

    def forward(self, X, return_hidden=False):
        Xa = add_bias(X)
        yk = self.phi_h.f(Xa @ self.W1.T)            # hidden outputs y_k
        yka = add_bias(yk)
        yj = self.phi_o.f(yka @ self.W2.T)           # network outputs y_j
        return (yj, yk, Xa, yka) if return_hidden else yj

    def get_weights(self):
        return (self.W1.copy(), self.W2.copy())

    def set_weights(self, w):
        self.W1, self.W2 = w[0].copy(), w[1].copy()

    # ---------------------------------------------------------------- learning
    def fit(self, Xtr, ytr, Xte=None, yte=None, n_classes=None, callback=None):
        rng = np.random.default_rng(self.seed)
        n_classes = n_classes or int(ytr.max() + 1)
        Ttr = encode_targets(ytr, n_classes)
        Tte = encode_targets(yte, n_classes) if Xte is not None else None
        self.init_weights(Xtr.shape[1], Ttr.shape[1], rng)
        dW1_prev = np.zeros_like(self.W1)   # Delta w(n-1) for the momentum term
        dW2_prev = np.zeros_like(self.W2)
        eta = self.eta
        E_prev = self.cost(Xtr, Ttr)
        t0, best = time.time(), None
        Xa_all = add_bias(Xtr)

        for ep in range(1, self.epochs + 1):
            w_saved = self.get_weights()
            for n in rng.permutation(len(Xtr)):
                xa = Xa_all[n]
                # --- forward pass
                yk = self.phi_h.f(self.W1 @ xa)
                yka = np.concatenate([[-1.0], yk])
                yj = self.phi_o.f(self.W2 @ yka)
                # --- backward pass
                e = Ttr[n] - yj                                   # e_j = y*_j - y_j
                delta_j = e * self.phi_o.df(yj)                    # local gradient of output layer: delta_j = e_j phi'(v_j)
                delta_k = (self.W2[:, 1:].T @ delta_j) * self.phi_h.df(yk)  # hidden: delta_k = sum_j delta_j w_jk . phi'(v_k)
                # --- weight corrections: Delta w(n) = eta delta y + alpha Delta w(n-1)
                dW2 = eta * np.outer(delta_j, yka) + self.alpha * dW2_prev
                dW1 = eta * np.outer(delta_k, xa) + self.alpha * dW1_prev
                self.W2 += dW2
                self.W1 += dW1
                dW2_prev, dW1_prev = dW2, dW1

            E = self.cost(Xtr, Ttr)
            if self.adaptive:
                if E > E_prev * self.max_ratio:      # error went up too much -> step rejected
                    self.set_weights(w_saved)
                    eta *= self.eta_dec
                    dW1_prev[:] = 0
                    dW2_prev[:] = 0
                    E = E_prev
                elif E < E_prev:                     # error went down -> accelerate
                    eta *= self.eta_inc
            E_prev = E

            self._log_epoch(ep, Xtr, Ttr, ytr, Xte, Tte, yte, eta)
            best = self._early_stopping_update(best)
            if callback:
                callback(self, ep)
        self.final_eta = eta
        self._finish(best, t0)
        return self
