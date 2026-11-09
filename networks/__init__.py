"""From-scratch NumPy implementations of the networks seen in the course."""
from .base import NeuralNet, metrics, confusion_matrix
from .perceptron import Perceptron
from .adaline import Adaline
from .mlp_bp import MLP
from .lm import MLP_LM
from .rbf import RBFNet

MODEL_NAMES = ["Perceptron", "ADALINE", "MLP-BP", "MLP-momentum", "MLP-adaptive", "MLP-LM", "RBF"]


def build_model(kind: str, n_hidden: int = 12, eta: float = 0.05, alpha: float = 0.8, mu: float = 0.01,
                epochs: int = 300, n_centers: int = 80, hidden_act: str = "tanh", seed: int = 0):
    """Factory used by train.py and the Streamlit app."""
    if kind == "Perceptron":
        return Perceptron(eta=eta, epochs=epochs, seed=seed)
    if kind == "ADALINE":
        return Adaline(eta=min(eta, 0.01), epochs=epochs, seed=seed)
    if kind == "MLP-BP":
        return MLP(n_hidden, eta=eta, alpha=0.0, epochs=epochs, hidden_act=hidden_act, seed=seed)
    if kind == "MLP-momentum":
        return MLP(n_hidden, eta=eta, alpha=alpha, epochs=epochs, hidden_act=hidden_act, seed=seed)
    if kind == "MLP-adaptive":
        return MLP(n_hidden, eta=eta, alpha=alpha, adaptive=True, epochs=epochs, hidden_act=hidden_act,
                   seed=seed, name="MLP-BP + momentum + adaptive η")
    if kind == "MLP-LM":
        return MLP_LM(n_hidden, mu=mu, epochs=max(10, epochs // 3), hidden_act=hidden_act, seed=seed)
    if kind == "RBF":
        return RBFNet(n_centers=n_centers, seed=seed)
    raise ValueError(kind)
