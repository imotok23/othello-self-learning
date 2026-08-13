"""A tiny NumPy-only MLP used as the position value function.

Two layers (input -> ReLU hidden -> tanh output), trained with Adam on a
plain MSE loss between predicted value and the actual game outcome
observed after a self-play game.
"""
import numpy as np


class ValueNetwork:
    def __init__(self, input_dim, hidden_dim=64, lr=0.001, seed=None):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_dim), (hidden_dim, 1))
        self.b2 = np.zeros(1)
        self.lr = lr
        self._t = 0
        self._adam = {name: {"m": np.zeros_like(p), "v": np.zeros_like(p)}
                      for name, p in self._params().items()}

    def _params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        z2 = h @ self.W2 + self.b2
        out = np.tanh(z2)
        return out.reshape(-1), (x, z1, h)

    def predict(self, x):
        out, _ = self.forward(np.atleast_2d(x))
        return out

    def train_step(self, x, y):
        x = np.atleast_2d(x)
        y = np.atleast_1d(y).astype(np.float64)
        out, (x_, z1, h) = self.forward(x)
        n = x.shape[0]

        d_out = ((out - y) * (1 - out ** 2) / n).reshape(-1, 1)
        dW2 = h.T @ d_out
        db2 = d_out.sum(axis=0)
        dh = d_out @ self.W2.T
        dz1 = dh * (z1 > 0)
        dW1 = x_.T @ dz1
        db1 = dz1.sum(axis=0)

        self._apply_adam({"W1": dW1, "b1": db1, "W2": dW2, "b2": db2})
        return float(np.mean((out - y) ** 2))

    def _apply_adam(self, grads, beta1=0.9, beta2=0.999, eps=1e-8):
        self._t += 1
        for name, g in grads.items():
            state = self._adam[name]
            state["m"] = beta1 * state["m"] + (1 - beta1) * g
            state["v"] = beta2 * state["v"] + (1 - beta2) * (g ** 2)
            m_hat = state["m"] / (1 - beta1 ** self._t)
            v_hat = state["v"] / (1 - beta2 ** self._t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, getattr(self, name) - update)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self._adam = {name: {"m": np.zeros_like(p), "v": np.zeros_like(p)}
                      for name, p in self._params().items()}
        self._t = 0
