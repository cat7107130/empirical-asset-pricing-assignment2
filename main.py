"""
main.py
=======
Entry point that replicates GKX (2020) Table 1, Figure 4 and Figure 9 for the
two required sample sets:

    Set1 : test period 2001-2016   (training/validation 1971-...)
    Set2 : test period 2001-2025

Because the recursive train/validation windows are identical for both sets, we
run the full recursion ONCE (test years 2001..end_year) and then slice the
collected predictions/importances to build Set1 (<=2016) and Set2 (all).

Run a quick end-to-end pilot:
    python main.py --pilot
Run the full replication:
    python main.py --full
"""

from __future__ import annotations
import os
import json
import argparse
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd

import data_loader
import features
import splits as splits_mod
import metrics
import portfolio
import plots
from models import ModelConfig, build_model


# Required model lists per output (from the assignment).
TABLE1_MODELS = ["OLS+H", "OLS-3+H", "PCR", "ENet+H", "RF", "NN2", "NN4"]
FIG4_MODELS   = ["PCR", "ENet+H", "RF", "NN2", "NN4"]
FIG9_MODELS   = ["OLS-3+H", "PCR", "ENet+H", "RF", "NN2", "NN4"]
# OLS-3+H is fed the 3-feature design matrix; everyone else the full matrix.
THREE_FEATURE_MODELS = {"OLS-3+H"}


@dataclass
class RunConfig:
    """All run-level knobs.  ``--pilot`` swaps in small windows / subsample /
    cheap ModelConfig so the same code path completes in minutes."""
    data_path: str = "data.csv"
    ff3_path: str = "ff3.csv"
    usrec_path: str = "USREC.csv"
    cache_path: str = "panel_nyse.parquet"
    out_dir: str = "outputs"

    start_year: int = 1971
    end_year: int = 2025
    train_years: int = 18
    val_years: int = 12
    set1_last_test_year: int = 2016          # Set1 truncation

    weighting: str = "VW"                    # Figure-9 portfolio weighting
    max_firms: int | None = None             # subsample firms (None = all)
    firm_seed: int = 42
    importance_sample_cap: int = 5000

    model_config: ModelConfig = field(default_factory=ModelConfig)


def fast2021_config() -> RunConfig:
    """Compute-limited variant requested by the assignment ("if computational
    resources are limited, first reduce the sample period, e.g. start from
    2006").  The OOS test period is restricted to 2021-2025 (characteristic
    data is available through 2021).  Training starts in 2006 with a GKX-style
    1.5:1 train:val ratio (9 train / 6 val) so the first test year is 2021:

        first_test = 2006 + 9 + 6 = 2021
        test 2021 -> train 2006-2014, val 2015-2020
        ...
        test 2025 -> train 2006-2018, val 2019-2024

    Model grids are the reduced (speed-first) ModelConfig defaults.  The two
    required output sets are mapped onto this shorter window as Set1=2021-2023
    (shorter OOS window) and Set2=2021-2025 (full available OOS window)."""
    return RunConfig(
        start_year=2006, end_year=2025, train_years=9, val_years=6,
        set1_last_test_year=2023,
        out_dir="outputs",
    )


def pilot_config() -> RunConfig:
    """Lightweight settings: short windows, 500 firms, tiny models.  Goal is to
    prove the pipeline runs end-to-end and emits all three outputs, NOT to
    produce accurate numbers."""
    mc = ModelConfig(
        pcr_n_components=(5, 10),
        enet_alpha=(1e-3,), enet_l1_ratio=(0.5,), enet_max_iter=500,
        rf_n_estimators=(100,), rf_max_depth=(3,), rf_max_features=(0.5,),
        nn_l1_lambda=(1e-4,), nn_max_epochs=8, nn_patience=3, nn_ensemble=2,
        nn_batch_size=4096,
    )
    return RunConfig(
        start_year=2008, end_year=2014, train_years=3, val_years=2,
        set1_last_test_year=2013, max_firms=500, model_config=mc,
        out_dir="outputs_pilot",
    )


# --------------------------------------------------------------------------- #
# Data preparation                                                            #
# --------------------------------------------------------------------------- #
def prepare(cfg: RunConfig):
    """Load panel, optionally subsample firms, build both design matrices, and
    return everything the recursion needs (as numpy for speed)."""
    df, char_cols = data_loader.load_panel(cfg.data_path, cfg.cache_path)

    # Restrict to the configured calendar range up front.
    yr = df["eom"].dt.year
    df = df[(yr >= cfg.start_year) & (yr <= cfg.end_year)].reset_index(drop=True)

    # Optional unbiased firm subsample (random ids), for pilot / limited compute.
    if cfg.max_firms is not None:
        ids = df["id"].unique()
        rng = np.random.default_rng(cfg.firm_seed)
        keep = set(rng.choice(ids, min(cfg.max_firms, len(ids)), replace=False))
        df = df[df["id"].isin(keep)].reset_index(drop=True)

    # Full design matrix and 3-feature OLS-3 matrix (both rank-normalized).
    X, meta, feat_cols = features.build_design_matrix(df, char_cols)
    X3, _, feat3 = features.build_ols3_matrix(df)

    Xnp = X.values.astype(np.float32)
    X3np = X3.values.astype(np.float32)
    y = meta["ret_exc_lead1m"].values.astype(np.float32)
    year = meta["eom"].dt.year.values
    return dict(meta=meta, year=year, y=y,
                Xnp=Xnp, feat_cols=feat_cols, char_cols=char_cols,
                X3np=X3np, feat3=feat3)


# --------------------------------------------------------------------------- #
# Recursive run for a single model                                            #
# --------------------------------------------------------------------------- #
def run_model(name: str, data: dict, cfg: RunConfig, want_importance: bool):
    """Run the recursive expanding-window scheme for one model across all test
    years.  Returns (predictions DataFrame, importance-per-refit dict)."""
    use3 = name in THREE_FEATURE_MODELS
    X = data["X3np"] if use3 else data["Xnp"]
    feats = data["feat3"] if use3 else data["feat_cols"]
    char_cols, y, year, meta = data["char_cols"], data["y"], data["year"], data["meta"]

    # Per-window checkpoints: a long run can be killed/resumed without losing
    # finished windows.  Each test year writes one preds parquet and (optionally)
    # one importance json; a restart skips years already on disk.
    ck = os.path.join(cfg.out_dir, "ckpt", name.replace("+", "p").replace("-", "_"))
    os.makedirs(ck, exist_ok=True)

    pred_parts, imp_per_refit = [], {}
    for sp in splits_mod.generate_splits(cfg.start_year, cfg.end_year,
                                         cfg.train_years, cfg.val_years):
        ck_pred = os.path.join(ck, f"pred_{sp.test_year}.parquet")
        ck_imp = os.path.join(ck, f"imp_{sp.test_year}.json")
        # Resume: reuse a completed window.
        if os.path.exists(ck_pred) and (not want_importance or os.path.exists(ck_imp)):
            pred_parts.append(pd.read_parquet(ck_pred))
            if want_importance:
                with open(ck_imp) as fh:
                    imp_per_refit[sp.test_year] = json.load(fh)
            print(f"    [{name}] test {sp.test_year}: resumed from checkpoint", flush=True)
            continue

        tr = (year >= sp.train_start) & (year <= sp.train_end)
        va = (year >= sp.val_start) & (year <= sp.val_end)
        te = (year == sp.test_year)
        if tr.sum() == 0 or va.sum() == 0 or te.sum() == 0:
            continue

        model = build_model(name, cfg.model_config)
        model.fit(X[tr], y[tr], X[va], y[va])
        yhat = model.predict(X[te])

        part = meta.loc[te, ["id", "eom", "ret_exc_lead1m", "market_equity"]].copy()
        part["yhat"] = yhat
        part["test_year"] = sp.test_year
        part.to_parquet(ck_pred, index=False)          # checkpoint predictions
        pred_parts.append(part)

        # GKX variable importance: evaluated on the training sample of each refit.
        if want_importance:
            imp = metrics.variable_importance(
                model, X[tr], y[tr], feats, char_cols,
                sample_cap=cfg.importance_sample_cap,
                rng_seed=cfg.model_config.seed)
            with open(ck_imp, "w") as fh:
                json.dump(imp, fh)                      # checkpoint importance
            imp_per_refit[sp.test_year] = imp
        print(f"    [{name}] test {sp.test_year}: "
              f"train={tr.sum()} val={va.sum()} test={te.sum()}", flush=True)

    preds = pd.concat(pred_parts, ignore_index=True)
    return preds, imp_per_refit


# --------------------------------------------------------------------------- #
# Output assembly per sample set                                              #
# --------------------------------------------------------------------------- #
def assemble_set(set_name: str, last_test_year: int,
                 all_preds: dict, all_imp: dict, benchmark: pd.DataFrame,
                 recession: pd.DataFrame, cfg: RunConfig):
    """Build Table 1, Figure 4, Figure 9 for one set by slicing the full run to
    test_year <= last_test_year."""
    os.makedirs(cfg.out_dir, exist_ok=True)
    tag = set_name

    # ---- Table 1: pooled OOS R^2 for all 7 models ------------------------- #
    r2 = {}
    for m in TABLE1_MODELS:
        p = all_preds[m]
        p = p[p["test_year"] <= last_test_year]
        r2[m] = metrics.r2_oos(p["ret_exc_lead1m"].values, p["yhat"].values)
    plots.plot_table1(r2, f"{cfg.out_dir}/table1_{tag}.png",
                      f"{cfg.out_dir}/table1_{tag}.csv",
                      f"Table 1 - Monthly OOS R^2 ({tag})")

    # ---- Figure 4: variable importance for the 5 models ------------------- #
    # Average the per-refit importances over the refits in this set's range.
    char_cols = list(next(iter(all_imp[FIG4_MODELS[0]].values())).keys())
    imp_by_model = {
        m: metrics.aggregate_importance(
            [d for ty, d in all_imp[m].items() if ty <= last_test_year], char_cols)
        for m in FIG4_MODELS
    }
    plots.plot_figure4(imp_by_model, f"{cfg.out_dir}/figure4_{tag}.png",
                       f"Figure 4 - Variable importance ({tag})")

    # ---- Figure 9: cumulative H-L portfolio returns ----------------------- #
    cum_by_model = {}
    for m in FIG9_MODELS:
        p = all_preds[m]
        p = p[p["test_year"] <= last_test_year]
        ls = portfolio.decile_long_short(p, weighting=cfg.weighting)
        cum_by_model[m] = portfolio.cumulative_return(ls["hml"])
    # Benchmark over the same months.
    months = cum_by_model[FIG9_MODELS[0]].index
    bench = benchmark.set_index("eom")["mkt_rf"].reindex(months).fillna(0.0)
    bench_cum = portfolio.cumulative_return(bench)
    plots.plot_figure9(cum_by_model, bench_cum, recession,
                       f"{cfg.out_dir}/figure9_{tag}.png",
                       f"Figure 9 - Cumulative ML portfolio return ({tag})")

    print(f"  [{tag}] R2_oos:", {k: round(v, 4) for k, v in r2.items()})


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="quick end-to-end run")
    ap.add_argument("--full", action="store_true", help="full 1971-2025 run")
    ap.add_argument("--fast2021", action="store_true",
                    help="compute-limited run: OOS test period 2021-2025")
    args = ap.parse_args()
    if not (args.pilot or args.full or args.fast2021):
        args.pilot = True                    # default to the safe pilot

    if args.fast2021:
        cfg, mode = fast2021_config(), "FAST2021"
    elif args.full:
        cfg, mode = RunConfig(), "FULL"
    else:
        cfg, mode = pilot_config(), "PILOT"
    os.makedirs(cfg.out_dir, exist_ok=True)
    print(f"=== {mode} run -> {cfg.out_dir} ===")

    data = prepare(cfg)
    print(f"prepared: rows={len(data['y'])} feats={len(data['feat_cols'])} "
          f"chars={len(data['char_cols'])}")

    benchmark = data_loader.load_ff_benchmark(cfg.ff3_path)
    recession = data_loader.load_recession(cfg.usrec_path)

    # Run every model once over the full recursion.
    all_preds, all_imp = {}, {}
    for m in TABLE1_MODELS:
        print(f"  running {m} ...", flush=True)
        want_imp = m in FIG4_MODELS
        preds, imp = run_model(m, data, cfg, want_imp)
        all_preds[m] = preds
        if want_imp:
            all_imp[m] = imp
        preds.to_csv(f"{cfg.out_dir}/preds_{m.replace('+','p').replace('-','_')}.csv",
                     index=False)

    # Build Set1 (<=2016) and Set2 (<=end_year).
    assemble_set("Set1", cfg.set1_last_test_year, all_preds, all_imp,
                 benchmark, recession, cfg)
    assemble_set("Set2", cfg.end_year, all_preds, all_imp,
                 benchmark, recession, cfg)
    print("=== done ===")


if __name__ == "__main__":
    main()
