"""
portfolio.py
============
Builds the GKX Figure-9 machine-learning portfolios from out-of-sample
predictions.

Each month, stocks are sorted into deciles on the model's predicted return.
Decile portfolios are formed and a zero-cost long-short H-L (decile 10 minus
decile 1) portfolio is computed.  Cumulative log/compounded return of the H-L
leg is what Figure 9 plots.

Weighting (VW vs EW) is exposed via config; the default is value-weighted by
``market_equity``, documented in the architecture file.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    s = w.sum()
    if s <= 0:
        return float(np.mean(values))
    return float(np.sum(values * w) / s)


def decile_long_short(preds: pd.DataFrame, n_deciles: int = 10,
                      weighting: str = "VW") -> pd.DataFrame:
    """Form monthly decile portfolios and the H-L spread.

    Parameters
    ----------
    preds : DataFrame with columns
        ['eom', 'yhat', 'ret_exc_lead1m', 'market_equity']  (realized return is
        the next-month excess return that the prediction targets).
    weighting : 'VW' (market_equity) or 'EW'.

    Returns DataFrame indexed by eom with columns
        ['lo' (decile 1), 'hi' (decile 10), 'hml' (hi - lo)] of realized
        monthly excess returns.
    """
    rows = []
    for eom, g in preds.groupby("eom", sort=True):
        if len(g) < n_deciles:
            continue
        # Rank predictions into deciles (labels 0..n-1); 'first' breaks ties.
        d = pd.qcut(g["yhat"].rank(method="first"), n_deciles,
                    labels=False, duplicates="drop")
        g = g.assign(decile=d)
        lo = g[g["decile"] == 0]
        hi = g[g["decile"] == d.max()]
        if weighting == "VW":
            r_lo = _weighted_mean(lo["ret_exc_lead1m"].values, lo["market_equity"].values)
            r_hi = _weighted_mean(hi["ret_exc_lead1m"].values, hi["market_equity"].values)
        else:
            r_lo = float(lo["ret_exc_lead1m"].mean())
            r_hi = float(hi["ret_exc_lead1m"].mean())
        rows.append((eom, r_lo, r_hi, r_hi - r_lo))
    out = pd.DataFrame(rows, columns=["eom", "lo", "hi", "hml"]).set_index("eom")
    return out.sort_index()


def cumulative_return(monthly: pd.Series, log: bool = False) -> pd.Series:
    """Compounded cumulative return of a monthly return series.

    GKX Figure 9 plots cumulative return of a $1 investment; we compound
    (1+r).cumprod() - 1.  Set ``log`` for cumulative log return instead.
    """
    if log:
        return np.log1p(monthly).cumsum()
    return (1.0 + monthly).cumprod() - 1.0
