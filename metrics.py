"""
metrics.py
==========
Out-of-sample R^2 (GKX Table 1 definition) and model-based variable importance
(GKX Figure 4 definition).
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def r2_oos(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """GKX pooled out-of-sample R^2:

        R2 = 1 - sum((r - r_hat)^2) / sum(r^2)

    The denominator uses raw returns (NO demeaning), exactly as GKX, because
    historical-mean benchmarking artificially inflates predictability.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum(y_true ** 2)
    return 1.0 - sse / sst


def variable_importance(model, X, y, feat_cols, char_cols,
                        baseline=None, sample_cap: int = 20000,
                        rng_seed: int = 0) -> dict:
    """GKX-style variable importance for one fitted model on one training sample.

    For each characteristic we set its column to 0 (its cross-sectional median,
    i.e. an uninformative value) and measure the resulting INCREASE in MSE
    relative to the unperturbed prediction.  Industry dummies are not scored
    (Figure 4 reports characteristic importance).  Larger MSE increase => more
    important variable.

    Returns {characteristic_name: raw_importance}.  Normalization to sum 1 is
    done after averaging across refits (see aggregate_importance).
    """
    rng = np.random.default_rng(rng_seed)
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    yv = np.asarray(y, dtype=np.float64).ravel()
    # Subsample rows to keep the (n_features x n_rows) sweep affordable.
    if Xv.shape[0] > sample_cap:
        sel = rng.choice(Xv.shape[0], sample_cap, replace=False)
        Xv, yv = Xv[sel], yv[sel]

    base_pred = model.predict(Xv)
    base_mse = np.mean((yv - base_pred) ** 2) if baseline is None else baseline

    col_index = {c: i for i, c in enumerate(feat_cols)}
    imp = {}
    for c in char_cols:
        j = col_index[c]
        saved = Xv[:, j].copy()
        Xv[:, j] = 0.0
        pred = model.predict(Xv)
        Xv[:, j] = saved
        imp[c] = max(np.mean((yv - pred) ** 2) - base_mse, 0.0)
    return imp


def aggregate_importance(per_refit: list[dict], char_cols: list[str]) -> pd.Series:
    """Average raw importances across recursive refits, then normalize to sum 1.

    Returns a Series indexed by characteristic, descending by importance.
    """
    df = pd.DataFrame(per_refit).reindex(columns=char_cols).fillna(0.0)
    mean_imp = df.mean(axis=0)
    total = mean_imp.sum()
    if total > 0:
        mean_imp = mean_imp / total
    return mean_imp.sort_values(ascending=False)
