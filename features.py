"""
features.py
===========
Builds the GKX-style design matrix from the filtered panel.

Deviation from Gu, Kelly & Xiu (2020) — documented in the architecture file:
GKX use macro predictors x_t and form the Kronecker interaction z = x_t ⊗ (1, c_it).
We do NOT use macro predictors (Welch-Goyal series are intentionally excluded),
so there are no interaction terms.  The design matrix is simply:

        X = [ 151 rank-normalized characteristics | 48 ff49 industry dummies ]

Pre-processing follows GKX exactly:
  * Each characteristic is cross-sectionally ranked within each month and mapped
    to [-1, 1].  This is a contemporaneous (same-month) transform, so it uses no
    future information.
  * Missing values are set to 0 after normalization (the cross-sectional median).
This rank mapping is the reason no train/test leakage occurs from normalization:
it is computed independently within each month's cross-section.
"""

from __future__ import annotations
import pandas as pd
import numpy as np


# OLS-3 uses three predictors only.  We map GKX's size / book-to-market /
# momentum to JKP fields.  size = log(market_equity); bm = be_me; mom = ret_12_1.
OLS3_RAW = {"size": "market_equity", "bm": "be_me", "mom": "ret_12_1"}


def rank_normalize(df: pd.DataFrame, char_cols: list[str],
                   time_col: str = "eom") -> pd.DataFrame:
    """Cross-sectionally rank-normalize each characteristic to [-1, 1] within
    each month, then impute missing values to 0.

    rank mapping: pct-rank in (0, 1]  ->  2*pct - 1 in (-1, 1], median ~ 0.
    Returns a new DataFrame (same index) with the normalized characteristics.
    """
    g = df.groupby(time_col, sort=False)
    # pct=True gives the average percentile rank, robust to ties; NaNs stay NaN.
    ranked = g[char_cols].rank(pct=True)
    normalized = ranked * 2.0 - 1.0
    normalized = normalized.fillna(0.0)            # missing -> cross-sectional median
    return normalized


def build_industry_dummies(df: pd.DataFrame, industry_col: str = "ff49",
                           drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode the ff49 industry code.

    A consistent, fixed set of dummy columns is produced for ALL industries
    1..49 (plus a 'missing' bucket) so that the design matrix has the same
    columns in every recursive train/validation/test split.  One category is
    dropped to avoid perfect collinearity with the intercept.
    """
    codes = df[industry_col].fillna(0).astype(int)
    categories = [0] + list(range(1, 50))         # 0 = missing/unknown bucket
    cat = pd.Categorical(codes, categories=categories)
    dummies = pd.get_dummies(cat, prefix="ff49", drop_first=drop_first).astype(np.float32)
    dummies.index = df.index
    return dummies


def build_design_matrix(df: pd.DataFrame, char_cols: list[str]):
    """Assemble the full feature matrix and companion bookkeeping frame.

    Returns
    -------
    X        : DataFrame of features [normalized characteristics | ff49 dummies]
    meta     : DataFrame with ['id', 'eom', 'ret_exc_lead1m', 'market_equity']
    feat_cols: list of feature column names (characteristics first, then dummies)
    """
    norm = rank_normalize(df, char_cols)
    dummies = build_industry_dummies(df)
    X = pd.concat([norm.astype(np.float32), dummies], axis=1)
    meta = df[["id", "eom", "ret_exc_lead1m", "market_equity"]].copy()
    feat_cols = list(norm.columns) + list(dummies.columns)
    return X, meta, feat_cols


def build_ols3_matrix(df: pd.DataFrame):
    """Build the 3-feature OLS-3 design matrix (size, bm, momentum).

    The three raw predictors are themselves rank-normalized to [-1, 1] each
    month (size first transformed by log to match GKX's 'size = log market cap').
    """
    tmp = df[["id", "eom"]].copy()
    tmp["size"] = np.log(df["market_equity"].where(df["market_equity"] > 0))
    tmp["bm"] = df["be_me"]
    tmp["mom"] = df["ret_12_1"]
    X3 = rank_normalize(tmp, ["size", "bm", "mom"])
    meta = df[["id", "eom", "ret_exc_lead1m", "market_equity"]].copy()
    return X3.astype(np.float32), meta, ["size", "bm", "mom"]
