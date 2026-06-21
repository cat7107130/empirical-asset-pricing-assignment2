"""
plots.py
========
Renders the three required GKX outputs for one sample set (Set1 or Set2):

  Table 1  : bar chart of monthly out-of-sample R^2 by model (+ a CSV table).
  Figure 4 : variable-importance heatmap (characteristics x models).
  Figure 9 : cumulative return of the H-L machine-learning portfolios, with the
             Ken French Mkt-RF benchmark (SP500-Rf proxy) and NBER recession
             shading.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # headless: write files, no display
import matplotlib.pyplot as plt


def plot_table1(r2_by_model: dict, out_png: str, out_csv: str, title: str):
    """Bar chart + CSV of pooled OOS R^2 (in percent) by model.

    The CSV mirrors GKX Table 1: models are columns, a single ``All`` row holds
    the pooled monthly out-of-sample R^2 expressed in percent."""
    models = list(r2_by_model.keys())
    vals = [r2_by_model[m] * 100.0 for m in models]      # to percent
    # GKX Table-1 layout: one "All" row, models as columns, values in percent.
    pd.DataFrame({m: [r2_by_model[m] * 100.0] for m in models},
                 index=["All"]).round(3).to_csv(out_csv)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#c0392b" if v < 0 else "#2c7fb8" for v in vals]
    ax.bar(models, vals, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(r"$R^2_{oos}$ (%)")
    ax.set_title(title)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def plot_figure4(importance_by_model: dict, out_png: str, title: str,
                 top_n: int = 20):
    """GKX Figure 4: per-model horizontal bar charts of variable importance.

    importance_by_model : {model_name: pd.Series(importance indexed by char)}.
    For each model we draw its own panel listing the top_n most influential
    characteristics, sorted descending, with importance normalized to sum to one
    within that model (exactly as in Figure 4 of GKX 2020).
    """
    models = list(importance_by_model.keys())
    n = len(models)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 0.32 * top_n + 1.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, m in zip(axes, models):
        s = importance_by_model[m].astype(float)
        total = s.sum()
        if total > 0:                                    # normalize to sum one
            s = s / total
        s = s.sort_values(ascending=False).head(top_n)
        s = s.iloc[::-1]                                 # largest at the top
        ax.barh(range(len(s)), s.values, color="#2c7fb8")
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels(s.index, fontsize=7)
        ax.set_title(m, fontsize=10, weight="bold")
        ax.tick_params(axis="x", labelsize=7)
    for ax in axes[n:]:                                  # hide unused panels
        ax.axis("off")

    fig.suptitle(title, fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return models


def _recession_spans(recession: pd.DataFrame, index: pd.DatetimeIndex):
    """Convert the monthly USREC 0/1 flag into (start, end) shaded spans that
    overlap the plotted date range."""
    rec = recession.set_index("eom")["usrec"].reindex(index).fillna(0).astype(int)
    spans, in_rec, start = [], False, None
    for dt, v in rec.items():
        if v == 1 and not in_rec:
            in_rec, start = True, dt
        elif v == 0 and in_rec:
            in_rec = False
            spans.append((start, dt))
    if in_rec:
        spans.append((start, rec.index[-1]))
    return spans


def plot_figure9(cum_by_model: dict, benchmark_cum: pd.Series,
                 recession: pd.DataFrame, out_png: str, title: str):
    """Cumulative H-L portfolio returns by model + benchmark + NBER shading.

    cum_by_model  : {model_name: pd.Series(cumulative return indexed by eom)}.
    benchmark_cum : pd.Series cumulative Mkt-RF (SP500-Rf proxy).
    """
    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Union index for recession shading.
    all_idx = sorted(set().union(*[s.index for s in cum_by_model.values()]))
    all_idx = pd.DatetimeIndex(all_idx)
    for (a, b) in _recession_spans(recession, all_idx):
        ax.axvspan(a, b, color="0.85", zorder=0)

    for name, s in cum_by_model.items():
        ax.plot(s.index, s.values, label=name, lw=1.4)
    if benchmark_cum is not None and len(benchmark_cum):
        ax.plot(benchmark_cum.index, benchmark_cum.values,
                label="Mkt-RF (SP500-Rf proxy)", color="black",
                lw=1.6, ls="--")

    ax.set_ylabel("Cumulative return")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
