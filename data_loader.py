"""
data_loader.py
==============
Loads the JKP (Jensen, Kelly, Pedersen 2023) WRDS Global Factor monthly stock
panel, applies the Stage-1 sample filter, and merges the externally supplied
series needed for Figure 9 (Ken French Mkt-RF benchmark proxy and the NBER
recession indicator).

Design notes
------------
* The raw panel (``data.csv``) is ~8.5 GB / 192 columns / ~3.68M firm-months.
  We never load it whole more than once: the first call streams it in chunks,
  keeps only the columns and rows we need, and writes a compact Parquet cache.
  Subsequent calls read the cache instantly.
* The prediction target is ``ret_exc_lead1m`` (next-month excess return), already
  provided by JKP.  We never shift returns ourselves (avoids look-ahead).
* ``rf.csv`` (DGS10, the 10-year Treasury yield) is intentionally NOT used: it is
  not a monthly risk-free rate, and the target is already an excess return, so a
  risk-free series is not needed anywhere in the models.  The Figure-9 benchmark
  uses Ken French Mkt-RF instead (see ``load_ff_benchmark``).
"""

from __future__ import annotations
import os
import pandas as pd
import numpy as np


# Identifier / bookkeeping columns we always carry alongside the characteristics.
ID_COLS = ["id", "permno", "eom"]
# Columns used by the Stage-1 filter.
FILTER_COLS = ["common", "excntry", "crsp_exchcd", "obs_main", "primary_sec", "exch_main"]
# Extra columns needed downstream: target, size/weight, industry.
EXTRA_COLS = ["ret_exc_lead1m", "market_equity", "ff49"]


def get_characteristic_columns(data_path: str) -> list[str]:
    """Return the JKP characteristic block: every column from div12m_me to
    qmj_safety inclusive (151 columns in this file)."""
    header = pd.read_csv(data_path, nrows=0).columns.tolist()
    i0, i1 = header.index("div12m_me"), header.index("qmj_safety")
    return header[i0:i1 + 1]


def load_panel(data_path: str,
               cache_path: str = "panel_nyse.parquet",
               force_rebuild: bool = False,
               chunksize: int = 500_000) -> tuple[pd.DataFrame, list[str]]:
    """Load the filtered NYSE 'All' sample as a DataFrame.

    Stage-1 filter (all must hold):
        common == 1, excntry == 'USA', crsp_exchcd == 1 (NYSE only),
        obs_main == 1, primary_sec == 1, exch_main == 1,
        ret_exc_lead1m not null.

    Returns
    -------
    (df, char_cols) where df has columns ID_COLS + EXTRA_COLS + char_cols and
    char_cols is the list of 151 characteristic names.
    """
    char_cols = get_characteristic_columns(data_path)

    if os.path.exists(cache_path) and not force_rebuild:
        df = pd.read_parquet(cache_path)
        return df, char_cols

    usecols = list(dict.fromkeys(ID_COLS + FILTER_COLS + EXTRA_COLS + char_cols))
    parts = []
    for ch in pd.read_csv(data_path, usecols=usecols, chunksize=chunksize):
        # Apply the Stage-1 filter chunk by chunk to keep memory bounded.
        m = (
            (ch["common"] == 1)
            & (ch["excntry"] == "USA")
            & (ch["crsp_exchcd"] == 1)          # 1 = NYSE (2 = AMEX, 3 = NASDAQ)
            & (ch["obs_main"] == 1)
            & (ch["primary_sec"] == 1)
            & (ch["exch_main"] == 1)
            & ch["ret_exc_lead1m"].notna()
        )
        sub = ch.loc[m, ID_COLS + EXTRA_COLS + char_cols].copy()
        parts.append(sub)

    df = pd.concat(parts, ignore_index=True)
    df["eom"] = pd.to_datetime(df["eom"])
    df = df.sort_values(["eom", "id"]).reset_index(drop=True)
    df.to_parquet(cache_path, index=False)
    return df, char_cols


def load_ff_benchmark(ff3_path: str) -> pd.DataFrame:
    """Load Ken French monthly factors and return the market excess return.

    The Ken French CSV stores monthly rows keyed by YYYYMM, followed by an
    annual block and a copyright footer; we keep only the 6-digit monthly keys
    and convert percent -> decimal.  Mkt-RF serves as the SP500-Rf proxy in
    Figure 9 (the assignment explicitly permits Ken French data).

    Returns DataFrame with columns ['eom', 'mkt_rf', 'rf'] (decimal, monthly).
    """
    raw = pd.read_csv(ff3_path)
    raw = raw.rename(columns={raw.columns[0]: "ym"})
    # Keep only rows whose key is a 6-digit YYYYMM month.
    key = raw["ym"].astype(str).str.strip()
    monthly = raw[key.str.fullmatch(r"\d{6}")].copy()
    monthly["ym"] = monthly["ym"].astype(str).str.strip()
    monthly["eom"] = pd.to_datetime(monthly["ym"], format="%Y%m") + pd.offsets.MonthEnd(0)
    out = pd.DataFrame({
        "eom": monthly["eom"].values,
        "mkt_rf": monthly["Mkt-RF"].astype(float).values / 100.0,
        "rf": monthly["RF"].astype(float).values / 100.0,
    })
    return out.sort_values("eom").reset_index(drop=True)


def load_recession(usrec_path: str) -> pd.DataFrame:
    """Load the monthly NBER recession indicator (USREC) used to shade Figure 9.

    Returns DataFrame with columns ['eom', 'usrec'] aligned to month-end."""
    raw = pd.read_csv(usrec_path)
    raw.columns = [c.strip().lower() for c in raw.columns]
    raw["eom"] = pd.to_datetime(raw["observation_date"]) + pd.offsets.MonthEnd(0)
    return raw[["eom", "usrec"]].sort_values("eom").reset_index(drop=True)


if __name__ == "__main__":
    # Quick smoke test of the loader on the real files.
    df, chars = load_panel("data.csv")
    print("panel rows:", len(df), "characteristics:", len(chars))
    print("eom range:", df["eom"].min().date(), "->", df["eom"].max().date())
    bench = load_ff_benchmark("ff3.csv")
    rec = load_recession("USREC.csv")
    print("benchmark months:", len(bench), "recession months:", int(rec["usrec"].sum()))
