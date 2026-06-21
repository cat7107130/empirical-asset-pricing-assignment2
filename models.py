"""
models.py
=========
The seven GKX models required by the assignment, behind one uniform interface:

    model.fit(X_train, y_train, X_val, y_val)   # validation used for tuning
    model.predict(X)            -> np.ndarray   # works on any matrix of the
                                                #   same feature columns

    OLS+H     : Huber-loss linear regression on the full feature set
    OLS-3+H   : Huber-loss linear regression on 3 features (size, bm, mom)
                (same class; just fed the 3-column design matrix in main.py)
    PCR       : principal-component regression (n_components tuned)
    ENet+H    : elastic-net with Huber loss (alpha, l1_ratio tuned)
    RF        : random forest (n_estimators, max_depth, max_features tuned)
    NN2 / NN4 : feed-forward nets, geometric-pyramid (32,16) / (32,16,8,4),
                ReLU + batch-norm + dropout + L1 penalty + Adam + early
                stopping, ensembled over several seeds (GKX recipe)

Hyperparameter grids start from GKX defaults and are exposed in ``ModelConfig``
so the full run and the lightweight pilot share one code path.

Tuning protocol (GKX): fit on the training window, score MSE on the validation
window, keep the training-fitted model with the best validation MSE.  The
validation rows are NOT folded back into training.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import itertools
import numpy as np

from sklearn.linear_model import HuberRegressor, SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Configuration                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """All tunable knobs.  Defaults follow GKX (2020); the pilot overrides
    these with smaller grids/epochs via main.py."""
    # Huber threshold (xi) shared by OLS+H, OLS-3+H, ENet+H.
    huber_epsilon: float = 1.35
    ols_l2: float = 1e-4                     # tiny ridge so Huber stays solvable

    # PCR: number of principal components to try.
    pcr_n_components: tuple = (5, 10, 20, 30, 50)

    # ENet+H grid (alpha = overall penalty, l1_ratio = elastic-net mix).
    enet_alpha: tuple = (1e-4, 1e-3, 1e-2)
    enet_l1_ratio: tuple = (0.1, 0.5, 0.9)
    enet_max_iter: int = 2000

    # Random forest grid.  Reduced for speed (was 300 trees x 6 grid points).
    rf_n_estimators: tuple = (200,)
    rf_max_depth: tuple = (4, 6)
    rf_max_features: tuple = (0.3,)
    rf_n_jobs: int = -1

    # Neural-net training.
    nn_l1_lambda: tuple = (1e-4,)            # tuned on validation
    nn_learning_rate: float = 1e-3
    nn_dropout: float = 0.1
    nn_batch_size: int = 10_000
    nn_max_epochs: int = 50                  # reduced for speed (early-stopped anyway)
    nn_patience: int = 5                     # early-stopping patience
    nn_ensemble: int = 3                     # seeds averaged (reduced from 10 for speed)
    nn_device: str = "cuda" if torch.cuda.is_available() else "cpu"

    seed: int = 0


def _to_np(X):
    """Accept a DataFrame or array, return contiguous float32 ndarray."""
    if hasattr(X, "values"):
        X = X.values
    return np.ascontiguousarray(X, dtype=np.float32)


def _mse(y, yhat):
    return float(np.mean((y - yhat) ** 2))


# --------------------------------------------------------------------------- #
# Linear family (OLS+H, OLS-3+H, ENet+H, PCR)                                  #
# --------------------------------------------------------------------------- #
class SklearnModel:
    """Wraps a scikit-learn estimator/pipeline with a small validation grid
    search.  ``param_grid`` maps estimator-step parameter names to value lists.
    """

    def __init__(self, name, make_estimator, param_grid):
        self.name = name
        self._make = make_estimator          # callable(**params) -> estimator
        self._grid = param_grid
        self.best_ = None
        self.best_params_ = None

    def fit(self, Xtr, ytr, Xval, yval):
        Xtr, ytr = _to_np(Xtr), _to_np(ytr).ravel()
        Xval, yval = _to_np(Xval), _to_np(yval).ravel()
        keys = list(self._grid.keys())
        best_score = np.inf
        for combo in itertools.product(*[self._grid[k] for k in keys]):
            params = dict(zip(keys, combo))
            est = self._make(**params)
            est.fit(Xtr, ytr)
            score = _mse(yval, est.predict(Xval))
            if score < best_score:
                best_score, self.best_, self.best_params_ = score, est, params
        return self

    def predict(self, X):
        return self.best_.predict(_to_np(X))


def make_olsh(config: ModelConfig) -> SklearnModel:
    """OLS with Huber loss (also used for OLS-3+H on the 3-feature matrix).
    Huber epsilon is fixed from config, so the grid is a single dummy point."""
    def make(_):
        return HuberRegressor(epsilon=config.huber_epsilon,
                              alpha=config.ols_l2, max_iter=500)
    return SklearnModel("OLS+H", make, {"_": [0]})


def make_pcr(config: ModelConfig) -> SklearnModel:
    """Principal-component regression: standardize -> PCA(k) -> OLS."""
    def make(n_components):
        return Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=config.seed)),
            ("ols", LinearRegression()),
        ])
    return SklearnModel("PCR", make, {"n_components": list(config.pcr_n_components)})


def make_eneth(config: ModelConfig) -> SklearnModel:
    """Elastic-net with Huber loss via SGD (alpha & l1_ratio tuned)."""
    def make(alpha, l1_ratio):
        return Pipeline([
            ("scale", StandardScaler()),
            ("sgd", SGDRegressor(loss="huber", epsilon=config.huber_epsilon,
                                 penalty="elasticnet", alpha=alpha,
                                 l1_ratio=l1_ratio, max_iter=config.enet_max_iter,
                                 tol=1e-4, random_state=config.seed)),
        ])
    return SklearnModel("ENet+H", make,
                        {"alpha": list(config.enet_alpha),
                         "l1_ratio": list(config.enet_l1_ratio)})


def make_rf(config: ModelConfig) -> SklearnModel:
    """Random forest regressor."""
    def make(n_estimators, max_depth, max_features):
        return RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            max_features=max_features, n_jobs=config.rf_n_jobs,
            random_state=config.seed)
    return SklearnModel("RF", make,
                        {"n_estimators": list(config.rf_n_estimators),
                         "max_depth": list(config.rf_max_depth),
                         "max_features": list(config.rf_max_features)})


# --------------------------------------------------------------------------- #
# Neural networks (NN2 / NN4)                                                  #
# --------------------------------------------------------------------------- #
class _MLP(nn.Module):
    """Geometric-pyramid MLP: Linear -> BatchNorm -> ReLU -> Dropout per hidden
    layer, then a linear output unit."""

    def __init__(self, in_dim, hidden_sizes, dropout):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NNModel:
    """Feed-forward net with L1 penalty, Adam, early stopping, and seed
    ensembling.  ``nn_l1_lambda`` is tuned on the validation set; the chosen
    lambda is then used to train the full ensemble that backs ``predict``."""

    def __init__(self, name, hidden_sizes, config: ModelConfig):
        self.name = name
        self.hidden = hidden_sizes
        self.cfg = config
        self.scaler_ = StandardScaler()
        self.members_ = []                   # list of trained nn.Modules
        self.best_lambda_ = None

    # ---- one training run for a single seed --------------------------------
    def _train_one(self, Xtr, ytr, Xval, yval, l1_lambda, seed):
        dev = self.cfg.nn_device
        torch.manual_seed(seed)
        net = _MLP(Xtr.shape[1], self.hidden, self.cfg.nn_dropout).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=self.cfg.nn_learning_rate)
        lossf = nn.MSELoss()

        Xtr_t = torch.tensor(Xtr, device=dev)
        ytr_t = torch.tensor(ytr, device=dev)
        Xval_t = torch.tensor(Xval, device=dev)
        yval_t = torch.tensor(yval, device=dev)
        n = Xtr.shape[0]
        bs = self.cfg.nn_batch_size

        best_val, best_state, wait = np.inf, None, 0
        for _ in range(self.cfg.nn_max_epochs):
            net.train()
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                pred = net(Xtr_t[idx])
                loss = lossf(pred, ytr_t[idx])
                # L1 penalty on all weights (GKX regularization).
                l1 = sum(p.abs().sum() for p in net.parameters())
                (loss + l1_lambda * l1).backward()
                opt.step()
            # Early stopping on validation MSE.
            net.eval()
            with torch.no_grad():
                v = lossf(net(Xval_t), yval_t).item()
            if v < best_val - 1e-7:
                best_val, best_state, wait = v, {k: t.detach().clone() for k, t in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.cfg.nn_patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_val

    def fit(self, Xtr, ytr, Xval, yval):
        Xtr = self.scaler_.fit_transform(_to_np(Xtr))
        Xval = self.scaler_.transform(_to_np(Xval))
        ytr = _to_np(ytr).ravel()
        yval = _to_np(yval).ravel()

        # 1) Tune L1 lambda with a single seed.
        best_lambda, best_score = None, np.inf
        for lam in self.cfg.nn_l1_lambda:
            _, vscore = self._train_one(Xtr, ytr, Xval, yval, lam, seed=self.cfg.seed)
            if vscore < best_score:
                best_score, best_lambda = vscore, lam
        self.best_lambda_ = best_lambda

        # 2) Train the ensemble at the chosen lambda over multiple seeds.
        self.members_ = []
        for s in range(self.cfg.nn_ensemble):
            net, _ = self._train_one(Xtr, ytr, Xval, yval, best_lambda,
                                     seed=self.cfg.seed + 1 + s)
            self.members_.append(net)
        return self

    def predict(self, X):
        dev = self.cfg.nn_device
        Xs = self.scaler_.transform(_to_np(X))
        Xt = torch.tensor(Xs, device=dev)
        preds = np.zeros(Xs.shape[0], dtype=np.float64)
        with torch.no_grad():
            for net in self.members_:
                net.eval()
                # Predict in chunks to bound GPU memory on large test sets.
                out = []
                for i in range(0, Xt.shape[0], 200_000):
                    out.append(net(Xt[i:i + 200_000]).cpu().numpy())
                preds += np.concatenate(out)
        return preds / len(self.members_)


# --------------------------------------------------------------------------- #
# Factory                                                                      #
# --------------------------------------------------------------------------- #
def build_model(name: str, config: ModelConfig):
    """Return a fresh model instance for the given GKX model name."""
    if name in ("OLS+H", "OLS-3+H"):
        m = make_olsh(config); m.name = name; return m
    if name == "PCR":
        return make_pcr(config)
    if name == "ENet+H":
        return make_eneth(config)
    if name == "RF":
        return make_rf(config)
    if name == "NN2":
        return NNModel("NN2", (32, 16), config)
    if name == "NN4":
        return NNModel("NN4", (32, 16, 8, 4), config)
    raise ValueError(f"unknown model: {name}")
