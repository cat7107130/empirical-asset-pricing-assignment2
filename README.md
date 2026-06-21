# Empirical Asset Pricing via Machine Learning — GKX (2020) Replication

Replication of selected tables and figures from Gu, Kelly, and Xiu (2020),
*"Empirical Asset Pricing via Machine Learning"*, RFS 33(5), for the
KAIST BME.70034 Assignment.

Outputs produced (for **"All"** sample only, per the assignment):

- **Table 1** — Monthly out-of-sample stock-level prediction performance
  (pooled R²ₒₒₛ), as both a table and a bar figure.
- **Figure 4** — Variable importance by model (per-model bar charts).
- **Figure 9** — Cumulative return of machine-learning H-L portfolios, with the
  SP500-Rf proxy (Ken French Mkt-RF) and NBER recession shading.

Models: `OLS+H, OLS-3+H, PCR, ENet+H, RF, NN2, NN4`.

---

## Repository layout

| File | Purpose |
|------|---------|
| `main.py` | Entry point; runs the full recursion and assembles all outputs |
| `data_loader.py` | Loads the JKP stock panel, Ken French factors, NBER USREC |
| `features.py` | Builds the rank-normalized design matrices |
| `splits.py` | Expanding-window train / validation / test splits |
| `models.py` | The 7 models (linear family + RF + neural nets) |
| `metrics.py` | Out-of-sample R² and variable importance |
| `portfolio.py` | Decile long-short portfolio construction |
| `plots.py` | Renders Table 1 / Figure 4 / Figure 9 |
| `make_pdfs.py` | Builds the two submission PDFs |
| `profile_data.py` | One-off helper to profile the raw panel (optional) |
| `ARCHITECTURE.md` | Architecture description (rendered into `architecture.pdf`) |
| `requirements.txt` | Python dependencies |

---

## Data sources (download separately — not committed)

The raw data files are large / externally hosted and are excluded via
`.gitignore`. Place them in the project root before running:

| File | Source |
|------|--------|
| `data.csv` | JKP (Jensen, Kelly, Pedersen 2023) WRDS Global Factor monthly stock panel (~8.5 GB). Obtain from WRDS / the JKP Global Factor Data project (https://jkpfactors.com). |
| `ff3.csv` | Ken French Data Library — "Fama/French 3 Factors" monthly (https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Used for the SP500-Rf proxy (Mkt-RF). |
| `USREC.csv` | NBER recession indicator (USREC) from FRED (https://fred.stlouisfed.org/series/USREC). Monthly 0/1 flag for Figure 9 shading. |

On first run, `data_loader.py` streams `data.csv`, applies the NYSE "All"
sample filter, and writes a compact `panel_nyse.parquet` cache for fast reuse.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Run

```bash
# Compute-limited run used for the submitted results: OOS test period 2021-2025
python main.py --fast2021

# Build the two submission PDFs from the produced outputs/
python make_pdfs.py --out_dir outputs
```

Other modes:

```bash
python main.py --pilot   # quick end-to-end smoke test (tiny windows / subsample)
python main.py --full    # full 1971-2025 GKX recursion (very long)
```

Artifacts are written to `outputs/` (`table1_*`, `figure4_*`, `figure9_*`,
`preds_*.csv`) and the submission PDFs `architecture.pdf` / `results.pdf` to the
project root.

---

## Submitted deliverables

- `architecture.pdf` — architecture description + this repository URL.
- `results.pdf` — Set 1 and Set 2 outputs (Table 1, Figure 4, Figure 9).

## Notes on deviations (compute-limited, assignment-permitted)

The assignment allows reducing the sample period and/or firm count when compute
is limited. This submission uses:

1. **OOS test period 2021–2025** (characteristic data is available through 2021;
   sample starts 2006 with a 9-year train / 6-year validation expanding window).
   Set 1 = 2021–2023, Set 2 = 2021–2025.
2. **No macro × characteristic interactions** — the design matrix uses the
   stock-level characteristics directly (GKX's full 920-covariate interaction
   set is not constructed).
3. **Reduced model grids** — RF: 200 trees, smaller depth/feature grid;
   neural nets: 3-seed ensemble, 50 epochs (early-stopped).

See `fast2021_config()` in `main.py` for exact settings.
