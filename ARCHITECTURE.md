# Architecture Description — Replication of Gu, Kelly & Xiu (2020)

**Course:** BME.70034 Empirical Asset Pricing (Spring 2026) — Assignment 3
**Repository:** _<add your repository URL here>_

This document describes the architecture of the replication pipeline and, most
importantly, **every deviation from Gu, Kelly & Xiu (2020, RFS — "Empirical
Asset Pricing via Machine Learning", henceforth GKX)** together with its
justification. The companion *Results* PDF contains the produced tables/figures.

---

## 1. Goal

Replicate, for U.S. NYSE common stocks, the following GKX outputs using only the
algorithms required by the assignment:

| Output | Models used |
|---|---|
| **Table 1** — monthly OOS stock-level prediction performance (table + bar figure), **"All" sample only** | OLS+H, OLS-3+H, PCR, ENet+H, RF, NN2, NN4 |
| **Figure 4** — variable importance by model | PCR, ENet+H, RF, NN2, NN4 |
| **Figure 9** — cumulative return of ML portfolios (+ SP500−Rf, NBER shading) | OLS-3+H, PCR, ENet+H, RF, NN2, NN4 |

Each output is produced for **two sample sets**:
- **Set 1:** test period **2001–2016**
- **Set 2:** test period **2001–2025**

---

## 2. Data

### 2.1 Primary panel
- **Source:** JKP (Jensen, Kelly, Pedersen 2023) WRDS Global Factor Data,
  stock-level monthly panel (`data.csv`, 192 columns, ~3.68M firm-months).
- **Target:** `ret_exc_lead1m` (next-month excess return) — used **as provided**.
  We never shift returns ourselves, which eliminates any accidental look-ahead.
- The panel ships with PIT/lagged characteristics, identifiers (`id`, `permno`,
  `gvkey`, `eom`), filter flags, the FF49 industry code (`ff49`), and
  size/weight fields (`market_equity`, `me`).

### 2.2 External inputs (supplied by user, passed by path)
- **`ff3.csv`** — Ken French monthly 3-factor file. Its `Mkt-RF` column is used
  as the **SP500−Rf benchmark proxy** in Figure 9 (see §6.3). The file's annual
  block and copyright footer are stripped; percent is converted to decimal.
- **`USREC.csv`** — monthly NBER recession indicator, used to shade Figure 9.
- **`rf.csv` (DGS10) is intentionally unused.** It is the 10-year Treasury yield,
  not a monthly risk-free rate; and because the model target is already an excess
  return, no risk-free series is needed anywhere in the estimation. The only place
  a risk-free rate would matter is the Figure-9 benchmark, which we build from the
  already-excess Ken French `Mkt-RF` (the assignment explicitly permits Ken French
  data).

### 2.3 Sample filter (Stage 1) — "All" sample
A row is kept iff **all** hold:
`common==1`, `excntry=='USA'`, `crsp_exchcd==1` (**NYSE only**; 2=AMEX, 3=NASDAQ),
`obs_main==1`, `primary_sec==1`, `exch_main==1`, and `ret_exc_lead1m` not null.

Result: **1,011,704 NYSE firm-months, 1971–2025.** Top-1000 / Bottom-1000
sub-samples are **not** built (assignment requires "All" only).

---

## 3. Features — key deviation from GKX

GKX form the design matrix as the Kronecker interaction `z = x_t ⊗ (1, c_it)`,
where `x_t` are 8 Welch–Goyal macro predictors and `c_it` are 94 stock
characteristics (≈ 900+ regressors).

**Deviation (D1): we use no macro predictors, hence no interaction terms.**
The design matrix is

```
X = [ 151 rank-normalized characteristics | 48 FF49 industry dummies ]   (200 columns)
```

- *Why:* macro predictors are intentionally excluded for this assignment; the JKP
  panel already supplies a rich, PIT-correct characteristic set. A
  characteristics-only specification is a standard and defensible design choice in
  the cross-sectional ML pricing literature (e.g., **Jensen, Kelly & Pedersen,
  2023**, build their factor zoo directly from such characteristics).
- *Consequence:* GKX's feature definition differs from ours; reported magnitudes
  are therefore not expected to match GKX line-for-line. The qualitative ranking
  of models and the economically important characteristics are the object of the
  replication.

**Deviation (D2): characteristic count.** The JKP block `div12m_me … qmj_safety`
contains **151** characteristics (the assignment brief said ~153). We use the
exact 151 columns present in the file and report variable importance under their
**JKP names**.

### 3.1 Pre-processing (follows GKX)
- Each characteristic is **cross-sectionally rank-normalized to [−1, 1] each
  month** (`pct` rank → `2·rank − 1`). This is a contemporaneous transform that
  uses only the current month's cross-section → **no leakage**.
- Missing values are set to **0** (the cross-sectional median) after
  normalization.
- `ff49` is one-hot encoded with a **fixed** category set (1..49 plus a 0/missing
  bucket), one level dropped for identifiability, so every recursive split shares
  identical columns.

---

## 4. Models (assignment's 7; "All" sample)

A uniform `fit(X_train, y_train, X_val, y_val) / predict(X)` interface backs all
models. Hyperparameter grids start from GKX defaults and are exposed in
`ModelConfig` (in `models.py`). Tuning uses the **validation window only**; the
training-fitted model with the best validation MSE is kept (validation rows are
not folded back into training — the GKX protocol).

| Model | Implementation | Tuned on validation |
|---|---|---|
| **OLS+H** | `HuberRegressor`, full features | Huber ξ fixed (config), tiny ridge |
| **OLS-3+H** | `HuberRegressor`, **3 features** | as above |
| **PCR** | `StandardScaler → PCA(k) → OLS` | `k` (n components) |
| **ENet+H** | `SGDRegressor(loss='huber', penalty='elasticnet')` | `alpha`, `l1_ratio` |
| **RF** | `RandomForestRegressor` | `n_estimators`, `max_depth`, `max_features` |
| **NN2** | MLP `(32,16)` | L1 λ |
| **NN4** | MLP `(32,16,8,4)` | L1 λ |

**OLS-3+H feature mapping (deviation D3):** GKX's size / value / momentum trio,
mapped to JKP fields: `size = log(market_equity)`, `bm = be_me`,
`momentum = ret_12_1`. With no macro predictors there are no interactions, so the
three variables enter directly (each rank-normalized to [−1, 1]).

**Neural nets (GKX recipe):** ReLU, batch-norm, dropout, **L1 weight penalty**,
**Adam**, **early stopping** on validation MSE, and an **ensemble averaged over 10
seeds** (full run; GPU). Geometric-pyramid widths match GKX NN2/NN4.

---

## 5. Train / validation / test split (GKX recursive scheme)

Expanding training window, rolling 12-year validation window, one test year at a
time, models refit every year. With `start_year=1971, train_years=18,
val_years=12`:

```
first test year = 1971 + 18 + 12 = 2001
test 2001 -> train 1971–1988, validation 1989–2000
test 2002 -> train 1971–1989, validation 1990–2001   (train grows, val rolls)
...
test 2025 -> train 1971–2012, validation 2013–2024
```

Because the windows are identical for both sets, the recursion is run **once**
(2001–2025) and sliced: **Set1 = test years ≤ 2016**, **Set2 = all**. All split
lengths are config parameters.

---

## 6. Outputs (Set1 and Set2 each)

### 6.1 Table 1 — pooled OOS R²
`R²_oos = 1 − Σ(r − r̂)² / Σ r²` (GKX definition, **no demeaning**), pooled over
all test stock-months, for all 7 models. Rendered as a CSV and a bar figure.

### 6.2 Figure 4 — variable importance
For each fitted model on each refit's training sample, each characteristic is set
to 0 (its median) and the **increase in MSE** is recorded. Raw importances are
averaged across refits, then **normalized to sum 1 per model**. Top characteristics
are shown as a model × characteristic heatmap. Variable names are JKP names (D2).

### 6.3 Figure 9 — cumulative ML portfolios
Each month, stocks are sorted into **deciles by predicted return**; a zero-cost
**H−L (decile 10 − decile 1)** portfolio is formed and its return compounded into
a cumulative curve, for the 6 required models.
- **Weighting (deviation D4):** **value-weighted by `market_equity`** by default
  (config-exposed; EW available). VW is the economically conservative choice and
  the GKX default for their headline value-weighted portfolios.
- **Benchmark:** Ken French `Mkt-RF`, labeled **"Mkt-RF (SP500-Rf proxy)"**.
- **Shading:** NBER recession months from `USREC.csv`.

---

## 7. Module layout

```
data_loader.py  load + Stage-1 filter + Parquet cache; load ff3 / USREC
features.py     rank-normalization, FF49 dummies, full & OLS-3 design matrices
splits.py       recursive expanding-window split generator
models.py       the 7 models + ModelConfig (hyperparameter grids, GPU NN)
metrics.py      R²_oos, GKX variable importance
portfolio.py    decile sorts, H−L portfolio, cumulative returns
plots.py        Table 1 figure, Figure 4 heatmap, Figure 9 curve
main.py         RunConfig, pilot/full driver, Set1/Set2 assembly
requirements.txt
```

---

## 8. Reproducibility & execution

Two-stage execution sharing **one code path** (only `RunConfig` differs):

- **Pilot** (`python main.py --pilot`): 500 firms, short windows (test 2013–2014),
  small grids, 2-seed NN ensemble, 8 epochs. Purpose: verify the pipeline runs
  end-to-end and emits all three outputs. **Verified.** (Numbers are not meaningful
  at this scale.)
- **Full** (`python main.py --full`): 1971–2025, full NYSE sample, GKX-scale grids,
  10-seed NN ensembles on GPU. Produces the reported Set1/Set2 results.

First call streams the 8.5 GB CSV once into `panel_nyse.parquet`; later runs load
the cache. The panel is already filtered to U.S. common stock, so the calendar and
NYSE filters dominate the funnel. `requirements.txt` pins the dependencies.

### Deviation summary
| ID | Deviation | Reason |
|---|---|---|
| D1 | No macro predictors / no `x ⊗ c` interactions; characteristics-only design | Assignment excludes macro series; standard in JKP-style characteristic models |
| D2 | 151 JKP characteristics, reported under JKP names | That is the exact block supplied in `data.csv` |
| D3 | OLS-3 = {log market_equity, be_me, ret_12_1}, no interactions | JKP analogues of GKX size/value/momentum; no macro to interact with |
| D4 | Figure 9 portfolios value-weighted by default | Economically conservative; config-exposed |
| — | Benchmark = Ken French `Mkt-RF` as SP500−Rf proxy; `rf.csv` (DGS10) unused | DGS10 is not a monthly risk-free rate; target is already excess; Ken French permitted |
```
