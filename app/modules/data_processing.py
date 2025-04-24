"""
app/modules/data_processing.py
==============================
UNIFIED DATA‑INGESTION & PREPROCESSING PIPELINE
------------------------------------------------

This module ingests **all raw inputs** (APIs + Excel uploads),
cleans / interpolates / reshapes them, and persists canonical tables
inside   data/combined.db   for downstream feature‑engineering,
vector‑database builds, ML forecasting, and Streamlit visualisation.

Pipelines implemented
---------------------
1. Global Imports & Exports  (Kpler API)
2. US‑specific Imports & Exports  (Kpler API)
3. Weekly Petroleum Report (Excel → fully‑interpolated truth & sliding tables)
4. US Crude Pipeline + Movements (Excel → daily pivoted tables)
5. Macro & Market series (FRED API  → bond_stocks table)
6. Pricing Vector (VectorDBPricing.xlsm → pricing_vector table)

All heavy I/O is wrapped in small helper functions so that you can:
    • run each pipeline on demand from notebooks or Streamlit
    • call  update_all()  to refresh every API‑based dataset
    • upload updated Excel files via the web UI and trigger their
      corresponding import_*() functions.

Environment variables
---------------------
• KPLER_EMAIL / KPLER_TOKEN          – Kpler SDK credentials
• FRED_API_KEY                       – FRED API key
• PROJECT_ROOT (optional override)   – root path of the repository
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()     

import os
import sys
import time
from pathlib import Path
from datetime import date, datetime, timedelta

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# ------------------------------------------------------------------------------
# 0.  PATHS & SQLITE ENGINE
# ------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# path to the local SQLite file (used if DATABASE_URL is not set)
DB_PATH = DATA_DIR / "combined.db"

DB_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{DB_PATH}"
)
ENGINE = create_engine(DB_URL, echo=False)


# ==============================================================================
# 1.  GLOBAL & US‑SPECIFIC IMPORTS / EXPORTS  (Kpler API)
# ==============================================================================
try:
    from kpler.sdk import Platform
    from kpler.sdk.configuration import Configuration
    from kpler.sdk.resources.flows import Flows
    from kpler.sdk import (
        FlowsDirection, FlowsSplit, FlowsPeriod, FlowsMeasurementUnit
    )
except ImportError:  # lazily skip if SDK not present
    Flows = None
    print("⚠️  Install `kpler‑sdk` to enable Kpler pipelines.")

KPLER_EMAIL  = os.getenv("KPLER_EMAIL",  "YOUR_EMAIL")
KPLER_TOKEN  = os.getenv("KPLER_TOKEN",  "YOUR_API_TOKEN")

USIE_START   = date(2021, 11, 15)
GLOBAL_START = date(2021, 11, 15)        # same start for global flows

USIE_TABLE   = "us_imports_exports"
GLOBAL_TABLE = "global_imports_exports"


def _kpler_client() -> "Flows":
    if Flows is None:
        raise ImportError("kpler‑sdk not installed.")
    cfg = Configuration(Platform.Liquids, KPLER_EMAIL, KPLER_TOKEN)
    return Flows(cfg)


# --------------------------------------------------------------------------- #
# 1A.  US‑ONLY IMPORTS & EXPORTS (daily)
# --------------------------------------------------------------------------- #
def fetch_us_imports_exports() -> pd.DataFrame:
    """
    Update / create **us_imports_exports** table and return it.

    Columns
    -------
    Date, United States Exports, Imports From <country>...
    """
    client = _kpler_client()

    try:
        df_exist = pd.read_sql_table(USIE_TABLE, ENGINE, parse_dates=["Date"])
        last_dt  = df_exist["Date"].max().date()
    except Exception:
        df_exist = pd.DataFrame()
        last_dt  = USIE_START - timedelta(days=1)

    start_dt  = last_dt + timedelta(days=1)
    end_dt    = date.today()
    if start_dt > end_dt:
        print("[USIE] already up‑to‑date.")
        return df_exist

    # Exports FROM United States
    resp_exp = client.get(
        flow_direction=[FlowsDirection.Export],
        from_zones=['United States'],
        split=[FlowsSplit.OriginCountries],
        granularity=[FlowsPeriod.Daily],
        unit=[FlowsMeasurementUnit.T],
        start_date=start_dt,
        end_date=end_dt
    )

    # Imports TO United States
    resp_imp = client.get(
        flow_direction=[FlowsDirection.Import],
        to_zones=['United States'],
        split=[FlowsSplit.OriginCountries],
        granularity=[FlowsPeriod.Daily],
        unit=[FlowsMeasurementUnit.T],
        start_date=start_dt,
        end_date=end_dt
    )

    exp_df = resp_exp[['Date', 'United States']].copy()
    exp_df.rename(columns={'United States': 'United States Exports'}, inplace=True)

    imp_keep = [c for c in resp_imp.columns if c != 'Date']
    imp_df   = resp_imp[['Date'] + imp_keep].copy()
    for col in imp_df.columns[1:]:
        imp_df.rename(columns={col: f'Imports From {col}'}, inplace=True)

    new_window = exp_df.merge(imp_df, on="Date", how="outer").sort_values("Date")

    full_df = (
        pd.concat([df_exist, new_window])
          .drop_duplicates(subset="Date", keep="last")
          .sort_values("Date")
    )

    full_df.to_sql(USIE_TABLE, ENGINE, if_exists="replace", index=False)
    print(f"[USIE] up‑to {full_df['Date'].max().date()}  ({len(full_df):,} rows)")
    return full_df


# --------------------------------------------------------------------------- #
# 1B.  GLOBAL IMPORTS & EXPORTS (Top‑15 each direction, daily)
# --------------------------------------------------------------------------- #
def fetch_global_imports_exports(n_top: int = 15) -> pd.DataFrame:
    """
    Pull **global** export & import flows, keep the N most important
    origin/destination countries (by cumulative volume), and persist to SQL.

    Table name:  global_imports_exports
    """
    client = _kpler_client()

    try:
        df_exist = pd.read_sql_table(GLOBAL_TABLE, ENGINE, parse_dates=["Date"])
        last_dt  = df_exist["Date"].max().date()
    except Exception:
        df_exist = pd.DataFrame()
        last_dt  = GLOBAL_START - timedelta(days=1)

    start_dt  = last_dt + timedelta(days=1)
    end_dt    = date.today()
    if start_dt > end_dt:
        print("[GLOBAL_IE] already up‑to‑date.")
        return df_exist

    # --- Exports by origin country ----------------------------------------
    resp_exp = client.get(
        flow_direction=[FlowsDirection.Export],
        split=[FlowsSplit.OriginCountries],
        granularity=[FlowsPeriod.Daily],
        unit=[FlowsMeasurementUnit.T],
        start_date=start_dt,
        end_date=end_dt
    )
    top_exp_countries = (
        resp_exp.drop(columns='Date')
                .sum(numeric_only=True)
                .sort_values(ascending=False)
                .head(n_top)
                .index.tolist()
    )
    global_exports = resp_exp[['Date'] + top_exp_countries]
    for c in global_exports.columns[1:]:
        global_exports.rename(columns={c: f"{c} Exports"}, inplace=True)

    # --- Imports by destination country -----------------------------------
    resp_imp = client.get(
        flow_direction=[FlowsDirection.Import],
        split=[FlowsSplit.DestinationCountries],
        granularity=[FlowsPeriod.Daily],
        unit=[FlowsMeasurementUnit.T],
        start_date=start_dt,
        end_date=end_dt
    )
    top_imp_countries = (
        resp_imp.drop(columns='Date')
                .sum(numeric_only=True)
                .sort_values(ascending=False)
                .head(n_top)
                .index.tolist()
    )
    global_imports = resp_imp[['Date'] + top_imp_countries]
    for c in global_imports.columns[1:]:
        global_imports.rename(columns={c: f"{c} Imports"}, inplace=True)

    new_window = pd.merge(global_exports, global_imports,
                          on='Date', how='outer').sort_values('Date')

    full_df = (
        pd.concat([df_exist, new_window])
          .drop_duplicates(subset="Date", keep="last")
          .sort_values("Date")
    )

    full_df.to_sql(GLOBAL_TABLE, ENGINE, if_exists="replace", index=False)
    print(f"[GLOBAL_IE] up‑to {full_df['Date'].max().date()}  ({len(full_df):,} rows)")
    return full_df


# ==============================================================================
# 2.  WEEKLY PETROLEUM REPORT (Excel upload → daily truth/sliding tables)
# ==============================================================================
WPR_TRUTH_TABLE   = "wpr_truth"
WPR_SLIDING_TABLE = "wpr_sliding"

# --- static column lists copied verbatim from notebook ------------------------
_WPR_KEEP_COLS   : list[str]
_WPR_STOCK_COLS  : list[str]
_WPR_FLOW_COLS   : list[str]
#  (keeping the lists identical to the notebook, omitted here for brevity)
#  Paste your full stock_cols / flow_cols lists here ↓↓↓
_WPR_KEEP_COLS = [  # shortened example; include full list in production
    'EIA CUSHING- OK CRUDE EXCL SPR STK', 'EIA US CRUDE EXCL SPR STK',
    'EIA US CRUDE STK', 'EIA PADD1 CRUDE EXCL SPR STK',
    'EIA PADD5 CRUDE EXCL SPR STK', 'EIA US CRUDE STK IN SPR',
    'EIA PADD2 CRUDE EXCL SPR STK', 'EIA PADD3 CRUDE EXCL SPR STK',
    'EIA US CRUDE/PETRO EXCL SPR STK',
    'EIA US CRUDE STK AND PETROLEUM PRODUCTS',
    'EIA PADD4 CRUDE EXCL SPR STK', 'EIA US CRUDE OIL PRODUCTION',
    'EIA US GROSS INPUTS REFINERIES', 'EIA PADD1 GROSS INPUTS REFINERIES',
    'EIA PADD2 GROSS INPUTS REFINERIES', 'EIA PADD3 GROSS INPUTS REFINERIES',
    'EIA PADD4 GROSS INPUTS REFINERIES', 'EIA PADD5 GROSS INPUTS REFINERIES',
    'EIA PADD1 CRUDE DISTILLATION CAP', 'EIA PADD5 CRUDE DISTILLATION CAP',
    'EIA PADD2 CRUDE DISTILLATION CAP', 'EIA PADD3 CRUDE DISTILLATION CAP',
    'EIA PADD4 CRUDE DISTILLATION CAP', 'EIA US EXPORTS OF CRUDE OIL',
    'EIA US PROPANE/PROPYLENE STK', 'EIA US GAS STK', 'EIA PADD1 GAS STK',
    'EIA PADD2 GAS STK', 'EIA PADD3 GAS STK', 'EIA PADD5 GAS STK',
    'EIA PADD4 GAS STK', 'EIA US DIST STK', 'EIA US RESIDUAL FUEL STK',
    'EIA US KEROSENE JET FUEL IMP', 'EIA US DIST IMP',
    'EIA US RESIDUAL FUEL IMP', 'EIA US CRUDE/PETRO EXCL SPR STK.1',
    'EIA US CRUDE STK AND PETROLEUM PRODUCTS.1', 'EIA US ETHANOL STK',
    'EIA US OTHERS EXCLUDE ETHANOL STK', 'EIA ALASKA CRUDE OIL PRODUCTION',
    'EIA LOWER 48 CRUDE OIL PRODUCTION', 'EIA US REFINERY PERCENT UTIL',
    'EIA PADD1 REFINERY PERCENT UTIL', 'EIA PADD5 REFINERY PERCENT UTIL',
    'EIA PADD2 REFINERY PERCENT UTIL', 'EIA PADD3 REFINERY PERCENT UTIL',
    'EIA PADD4 REFINERY PERCENT UTIL',
    'Midwest (PADD 2) Ending Stocks of Distillate Fuel Oil, Weekly',
    'East Coast (PADD 1) Ending Stocks of Distillate Fuel Oil, Weekly',
    'West Coast (PADD 5) Ending Stocks of Distillate Fuel Oil, Weekly',
    'Gulf Coast (PADD 3) Ending Stocks of Distillate Fuel Oil, Weekly',
    'Rocky Mountain (PADD 4) Ending Stocks of Distillate Fuel Oil, Weekly',
    'U.S. Product Supplied of Propane and Propylene, 4 Week Avg',
    'U.S. Product Supplied of Petroleum Products, 4 Week Avg',
    'U.S. Product Supplied of Other Oils, 4 Week Avg',
    'U.S. Product Supplied of Distillate Fuel Oil, 4 Week Avg',
    'U.S. Product Supplied of Residual Fuel Oil, 4 Week Avg',
    'U.S. Product Supplied of Finished Motor Gasoline, 4 Week Avg',
    'U.S. Product Supplied of Kerosene-Type Jet Fuel, 4 Week Avg',
    'EIA PADD1 DIST STK', 'EIA PADD2 DIST STK', 'EIA PADD5 DIST STK',
    'EIA PADD3 DIST STK', 'EIA PADD4 DIST STK',
    'EIA US PETROLEUM PRODUCTS SUPPLIED', 'EIA US OTHER OILS SUPPLIED',
    'EIA US PROPANE/PROPYLENE SUPPLIED', 'EIA US DISTILLATE FUEL OIL SUPPLIED',
    'EIA US RESIDUAL FUEL OIL SUPPLIED', 'EIA US FIN M GAS SUPPLIED',
    'EIA US KEROSENE JET FUEL SUPPLIED', 'EIA US REFINER NET CRUDE INPUT',
    'EIA PADD1 REFINER NET CRUDE INPUT', 'EIA PADD2 REFINER NET CRUDE INPUT',
    'EIA PADD3 REFINER NET CRUDE INPUT', 'EIA PADD4 REFINER NET CRUDE INPUT',
    'EIA PADD5 REFINER NET CRUDE INPUT',
    # ...
]
_WPR_STOCK_COLS = ['EIA CUSHING- OK CRUDE EXCL SPR STK',
    'EIA US CRUDE EXCL SPR STK',
    'EIA US CRUDE STK',
    'EIA PADD1 CRUDE EXCL SPR STK',
    'EIA PADD5 CRUDE EXCL SPR STK',
    'EIA US CRUDE STK IN SPR',
    'EIA PADD2 CRUDE EXCL SPR STK',
    'EIA PADD3 CRUDE EXCL SPR STK',
    'EIA US CRUDE/PETRO EXCL SPR STK',
    'EIA US CRUDE STK AND PETROLEUM PRODUCTS',
    'EIA PADD4 CRUDE EXCL SPR STK',
    'EIA PADD1 CRUDE DISTILLATION CAP',
    'EIA PADD5 CRUDE DISTILLATION CAP',
    'EIA PADD2 CRUDE DISTILLATION CAP',
    'EIA PADD3 CRUDE DISTILLATION CAP',
    'EIA PADD4 CRUDE DISTILLATION CAP',
    'EIA US PROPANE/PROPYLENE STK',
    'EIA US GAS STK',
    'EIA PADD1 GAS STK',
    'EIA PADD2 GAS STK',
    'EIA PADD3 GAS STK',
    'EIA PADD5 GAS STK',
    'EIA PADD4 GAS STK',
    'EIA US DIST STK',
    'EIA US RESIDUAL FUEL STK',
    'EIA US CRUDE/PETRO EXCL SPR STK.1',
    'EIA US CRUDE STK AND PETROLEUM PRODUCTS.1',
    'EIA US ETHANOL STK',
    'EIA US OTHERS EXCLUDE ETHANOL STK',
    'EIA US REFINERY PERCENT UTIL',
    'EIA PADD1 REFINERY PERCENT UTIL',
    'EIA PADD5 REFINERY PERCENT UTIL',
    'EIA PADD2 REFINERY PERCENT UTIL',
    'EIA PADD3 REFINERY PERCENT UTIL',
    'EIA PADD4 REFINERY PERCENT UTIL',
    'Midwest (PADD 2) Ending Stocks of Distillate Fuel Oil, Weekly',
    'East Coast (PADD 1) Ending Stocks of Distillate Fuel Oil, Weekly',
    'West Coast (PADD 5) Ending Stocks of Distillate Fuel Oil, Weekly',
    'Gulf Coast (PADD 3) Ending Stocks of Distillate Fuel Oil, Weekly',
    'Rocky Mountain (PADD 4) Ending Stocks of Distillate Fuel Oil, Weekly',
    'EIA PADD1 DIST STK',
    'EIA PADD2 DIST STK',
    'EIA PADD5 DIST STK',
    'EIA PADD3 DIST STK',
    'EIA PADD4 DIST STK']   # split according to your notebook
_WPR_FLOW_COLS  = ['EIA US CRUDE OIL PRODUCTION',
    'EIA US GROSS INPUTS REFINERIES',
    'EIA PADD1 GROSS INPUTS REFINERIES',
    'EIA PADD2 GROSS INPUTS REFINERIES',
    'EIA PADD3 GROSS INPUTS REFINERIES',
    'EIA PADD4 GROSS INPUTS REFINERIES',
    'EIA PADD5 GROSS INPUTS REFINERIES',
    'EIA US EXPORTS OF CRUDE OIL',
    'EIA US KEROSENE JET FUEL IMP',
    'EIA US DIST IMP',
    'EIA US RESIDUAL FUEL IMP',
    'EIA ALASKA CRUDE OIL PRODUCTION',
    'EIA LOWER 48 CRUDE OIL PRODUCTION',
    'U.S. Product Supplied of Propane and Propylene, 4 Week Avg',
    'U.S. Product Supplied of Petroleum Products, 4 Week Avg',
    'U.S. Product Supplied of Other Oils, 4 Week Avg',
    'U.S. Product Supplied of Distillate Fuel Oil, 4 Week Avg',
    'U.S. Product Supplied of Residual Fuel Oil, 4 Week Avg',
    'U.S. Product Supplied of Finished Motor Gasoline, 4 Week Avg',
    'U.S. Product Supplied of Kerosene-Type Jet Fuel, 4 Week Avg',
    'EIA US PETROLEUM PRODUCTS SUPPLIED',
    'EIA US OTHER OILS SUPPLIED',
    'EIA US PROPANE/PROPYLENE SUPPLIED',
    'EIA US DISTILLATE FUEL OIL SUPPLIED',
    'EIA US RESIDUAL FUEL OIL SUPPLIED',
    'EIA US FIN M GAS SUPPLIED',
    'EIA US KEROSENE JET FUEL SUPPLIED',
    'EIA US REFINER NET CRUDE INPUT',
    'EIA PADD1 REFINER NET CRUDE INPUT',
    'EIA PADD2 REFINER NET CRUDE INPUT',
    'EIA PADD3 REFINER NET CRUDE INPUT',
    'EIA PADD4 REFINER NET CRUDE INPUT',
    'EIA PADD5 REFINER NET CRUDE INPUT']


def import_weekly_petroleum_report(
    excel_path_or_file, *,
    save_sql: bool = True,
    start_date: str | pd.Timestamp = "2021-11-15"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert the *Weekly Petroleum Report* Excel (Fri rows) to

    • truth_df   – fully‑interpolated Fri→Fri blocks
    • sliding_df – “real‑time” view shifted so newest row == yesterday

    Both are optionally persisted to SQL.
    """
    raw = (
        pd.read_excel(excel_path_or_file, skiprows=2)
          .rename(columns={'Unnamed: 0': 'Date'})
    )
    raw['Date'] = pd.to_datetime(raw['Date'])
    raw = raw.set_index('Date')[_WPR_KEEP_COLS].copy()

    start_dt = pd.to_datetime(start_date)
    fridays  = raw.index[raw.index.weekday == 4].sort_values()
    today    = pd.Timestamp.today().normalize()

    # 1️⃣ build daily truth table (linear inter‑ or forward‑fill)
    daily_idx = pd.date_range(fridays[0], today, freq='D')
    truth = pd.DataFrame(index=daily_idx, columns=_WPR_KEEP_COLS)

    # Fri→Fri interpolation blocks
    for prev_fri, next_fri in zip(fridays[:-1], fridays[1:]):
        rng = pd.date_range(prev_fri, next_fri, freq='D')
        truth.loc[rng, _WPR_STOCK_COLS] = (
            raw.loc[[prev_fri, next_fri], _WPR_STOCK_COLS]
                .reindex(rng).interpolate('time')
        )
        truth.loc[rng, _WPR_FLOW_COLS] = (
            (raw.loc[[prev_fri, next_fri], _WPR_FLOW_COLS] / 7.0)
                .reindex(rng).interpolate('time')
        )

    # last partial week (flat / ÷7)
    last_fri  = fridays[-1]
    partial   = pd.date_range(last_fri, today, freq='D')
    truth.loc[partial, _WPR_STOCK_COLS] = raw.loc[last_fri, _WPR_STOCK_COLS].values
    truth.loc[partial, _WPR_FLOW_COLS]  = (raw.loc[last_fri, _WPR_FLOW_COLS] / 7.0).values

    # 2️⃣ build sliding calendar (cutoff = yesterday US/Eastern)
    cutoff     = (pd.Timestamp.now(tz='US/Eastern').normalize()
                  - pd.Timedelta(days=1)).tz_localize(None)
    offset_days = (cutoff - last_fri).days
    sliding     = (truth.shift(offset_days, freq='D')
                         .loc[start_dt:cutoff])

    # 3️⃣ helper columns
    def _add_helpers(df: pd.DataFrame) -> pd.DataFrame:
        wk_start = df.index.to_series().dt.to_period('W-FRI').dt.start_time
        df['is_release_day']   = (df.index.weekday == 2).astype(int)
        df['fraction_of_week'] = ((df.index - wk_start).dt.days / 7.0)
        df['model_weight']     = 1.0
        df.loc[df['is_release_day'] == 1, 'model_weight'] = 0.1
        return df

    truth   = _add_helpers(truth).loc[start_dt - timedelta(days=7):]
    sliding = _add_helpers(sliding)

    if save_sql:
        truth.to_sql(WPR_TRUTH_TABLE, ENGINE, if_exists="replace", index=True)
        sliding.to_sql(WPR_SLIDING_TABLE, ENGINE, if_exists="replace", index=True)
        print(f"[WPR] truth:{len(truth):,}  sliding:{len(sliding):,} rows saved.")

    return truth, sliding


# ==============================================================================
# 3.  CRUDE PIPELINE & MOVEMENTS  (Excel upload)
# ==============================================================================
PIPELINE_TABLE  = "daily_pipeline"
MOVEMENT_TABLE  = "daily_movement"

# --- helper conversion functions from notebook -------------------------------
def _weekly_to_daily_per_pipeline(df, flow_col, cap_col, heavy_col, light_col, name_col):
    df['Date (Week)'] = pd.to_datetime(df['Date (Week)'])
    out = []
    for name, grp in df.groupby(name_col):
        grp = grp.sort_values('Date (Week)').set_index('Date (Week)')
        daily_idx = pd.date_range(grp.index.min(), grp.index.max(), freq='D')
        daily = grp.reindex(daily_idx, method='ffill')
        daily.index.name = 'Date'
        daily['Daily Flow']            = daily[flow_col]  / 7
        daily['PipelineDailyCapacity'] = daily[cap_col]   / 7
        daily['PipelineHeavyDaily']    = daily[heavy_col] / 7
        daily['PipelineLightDaily']    = daily[light_col] / 7
        daily[name_col] = name
        out.append(daily)
    return pd.concat(out).reset_index()

def _weekly_to_daily_per_movement(df, heavy_col, light_col, name_col):
    df['Date (Week)'] = pd.to_datetime(df['Date (Week)'])
    out = []
    for name, grp in df.groupby(name_col):
        grp = grp.sort_values('Date (Week)').set_index('Date (Week)')
        daily_idx = pd.date_range(grp.index.min(), grp.index.max(), freq='D')
        daily = grp.reindex(daily_idx, method='ffill')
        daily.index.name = 'Date'
        daily['MovementHeavyDaily'] = daily[heavy_col]
        daily['MovementLightDaily'] = daily[light_col]
        daily[name_col] = name
        out.append(daily)
    return pd.concat(out).reset_index()


def import_pipeline_flow_excel(
    excel_path_or_file, *, save_sql: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert   CRUDE SD.xlsm   to two wide daily tables and optionally write to SQL.
    Returns (pivot_movement_df, pivot_pipeline_df).
    """
    FILE = excel_path_or_file
    pipeline_df = pd.read_excel(FILE, sheet_name='Pipelines', usecols='B:AC',
                                header=0, skiprows=4)
    movement_df = pd.read_excel(FILE, sheet_name='Movements', usecols='B:R',
                                header=0, skiprows=4)

    daily_pipeline_df = _weekly_to_daily_per_pipeline(
        pipeline_df,
        flow_col='SBM Published', cap_col='SBM Capacity',
        heavy_col='SBM Published Heavy', light_col='SBM Published Light',
        name_col='SBM Name'
    )
    daily_movement_df = _weekly_to_daily_per_movement(
        movement_df,
        heavy_col='SBM Heavy (Mbpd)', light_col='SBM Light (Mbpd)',
        name_col='SBM Name'
    )

    # Clean up / recodes (identical to notebook)
    movement_cols_to_drop = [
        'EIA Monthly Heavy Implied (Mbpd)', 'EIA Monthly Light Implied (Mbpd)',
        'EIA Monthly (Mbpd)', 'SBM Heavy %', 'SBM Heavy (Mbpd)',
        'SBM Light (Mbpd)', 'SBM Total (Mbpd)', 'Date (Month)', 'Year',
        'Data Filter (Weekly)', 'Date Filter (Monthly)'
    ]
    daily_movement_df.drop(columns=movement_cols_to_drop, inplace=True)
    daily_movement_df['From PADD'] = (
        daily_movement_df['From PADD'].str.replace('PADD ', '', regex=False).astype(int)
    )
    daily_movement_df['To PADD'] = (
        daily_movement_df['To PADD'].str.replace('PADD ', '', regex=False).astype(int)
    )

    pipeline_cols_to_drop = [
        'Date (Month)', 'SBM Filter Cushing', 'SBM Filter Date Range',
        'Date Filter (Weekly)', 'Date Filter (Monthly)',
        'Genscape Pipeline Name Regulatory', 'EA Attribute', 'Genscape Capacity',
        'Genscape Published', 'Genscape Revised', 'Genscape Change',
        'Regulatory Published', 'SBM Regulatory Adjusted Flow', 'Regulatory Revised',
        'SBM Heavy %', 'SBM Capacity', 'SBM Published Heavy', 'SBM Published Light',
        'SBM Published', 'SBM Revised Override', 'SBM Revised Heavy',
        'SBM Revised Light', 'SBM Revised', 'EA', 'Platts'
    ]
    daily_pipeline_df.drop(columns=pipeline_cols_to_drop, inplace=True)
    daily_pipeline_df['SBM Direction Cushing'] = (
        daily_pipeline_df['SBM Direction Cushing'].map({'Outbound': 0, 'Inbound': 1})
    )

    # Pivot wide by SBM Name
    pivot_movement = (
        daily_movement_df
        .pivot_table(
            index='Date',
            columns='SBM Name',
            values=['MovementHeavyDaily', 'MovementLightDaily'],
            aggfunc='sum'
        ).fillna(0)
    )
    pivot_movement.columns = [f"{a}_{b}" if b else a
                              for a, b in pivot_movement.columns.to_flat_index()]

    pivot_pipeline = (
        daily_pipeline_df
        .pivot_table(
            index='Date',
            columns='SBM Name',
            values=['Daily Flow', 'PipelineDailyCapacity',
                    'PipelineHeavyDaily', 'PipelineLightDaily'],
            aggfunc='sum'
        ).fillna(0)
    )
    pivot_pipeline.columns = [f"{a}_{b}" if b else a
                              for a, b in pivot_pipeline.columns.to_flat_index()]

    # Re‑index pipeline back to full daily span (bfill to 2021‑11‑15)
    pivot_pipeline = pivot_pipeline.reindex(
        pd.date_range(start='2021-11-15',
                      end=pivot_pipeline.index.max(), freq='D')
    )
    pivot_pipeline.index.name = 'Date'
    pivot_pipeline = pivot_pipeline.bfill()

    # Align on common start date
    common_start = max(pivot_movement.index.min(), pivot_pipeline.index.min())
    pivot_movement = pivot_movement[pivot_movement.index >= common_start]
    pivot_pipeline = pivot_pipeline[pivot_pipeline.index >= common_start]

    if save_sql:
        pivot_movement.to_sql(MOVEMENT_TABLE, ENGINE, if_exists="replace", index=True)
        pivot_pipeline.to_sql(PIPELINE_TABLE, ENGINE, if_exists="replace", index=True)
        print(f"[PIPELINE] movement:{pivot_movement.shape}  pipeline:{pivot_pipeline.shape}")

    return pivot_movement, pivot_pipeline


# ==============================================================================
# 4.  MACRO / MARKET SERIES  (FRED API)
# ==============================================================================
try:
    from fredapi import Fred
except ImportError:
    Fred = None
    print("⚠️  Install `fredapi` to enable macro pipeline.")

FRED_API_KEY = os.getenv("FRED_API_KEY", "YOUR_FRED_KEY")
FRED_SERIES  = {
    "VIX (Volatility)": "VIXCLS",
    "NASDAQ Composite": "NASDAQCOM",
    "Nikkei 225":       "NIKKEI225",
    "OVX":              "OVXCLS",
    "S&P 500":          "SP500",
    "T10Y3M Spread":    "T10Y3M",
    "T10Y2Y Spread":    "T10Y2Y", 
    "WTI Crude Oil":  'DCOILWTICO'
}
MACRO_TABLE  = "bond_stocks"
MACRO_START  = "2021-11-15"


def _fred_client() -> "Fred":
    if Fred is None:
        raise ImportError("fredapi not installed.")
    return Fred(api_key=FRED_API_KEY)


def update_macro_table() -> pd.DataFrame:
    """
    Download FRED macro series and persist to SQL   (bond_stocks table).
    """
    end = datetime.today().strftime("%Y-%m-%d")
    frames = []
    fred   = _fred_client()

    for nice, code in FRED_SERIES.items():
        ser = fred.get_series(code, observation_start=MACRO_START).rename(nice)
        frames.append(ser.to_frame())

    df = pd.concat(frames, axis=1)
    df.index.name = "date"

    # Nikkei trades ~1 day earlier than US close; align by shifting ‑1
    if "Nikkei 225" in df.columns:
        df["Nikkei 225"] = df["Nikkei 225"].shift(-1)

    df.sort_index(inplace=True)
    df.ffill(inplace=True)

    df.to_sql(MACRO_TABLE, ENGINE, if_exists="replace", index=True)
    print(f"[MACRO] stored {len(df):,} rows  → {MACRO_TABLE}")
    return df


# ==============================================================================
# 5.  PRICING VECTOR (VectorDBPricing.xlsm)
# ==============================================================================
PRICING_SHEETS = {
    "NETBACK AND REFINERY PRICING": "netback",
    "DIRTY TANKER PRICING":         "freight",
    "ArgDailyPricingRaw":           "pricing"
}
BAD_DATE_COL  = "combined_x000D_\ndescription"
PRICING_TABLE = "pricing_vector"


def import_vector_db_pricing(
    excel_path_or_file, *, save_sql: bool = True
) -> pd.DataFrame:
    """
    Merge the three sheets of VectorDBPricing.xlsm and (optionally) write to SQL.
    """
    dfs = {}
    for sheet, tag in PRICING_SHEETS.items():
        df = pd.read_excel(
            excel_path_or_file, sheet_name=sheet,
            skiprows=1, header=0, engine="openpyxl"
        )
        df.rename(columns={BAD_DATE_COL: "Date"}, inplace=True)
        df["Date"] = pd.to_datetime(df["Date"])
        dfs[tag] = df

    merged = (
        dfs["netback"]
            .merge(dfs["freight"], on="Date", how="right")
            .merge(dfs["pricing"], on="Date", how="right")
            .dropna()
            .set_index("Date")
            .sort_index()
    )

    if save_sql:
        merged.to_sql(PRICING_TABLE, ENGINE, if_exists="replace", index=True)
        print(f"[PRICING] {len(merged):,} rows  → {PRICING_TABLE}")
    return merged


# ==============================================================================
# 6.  CONVENIENCE DRIVERS
# ==============================================================================
def update_all(api_only: bool = True):
    """
    Refresh every **API‑driven** dataset and write to SQL.

    Excel‑based imports (WPR, Pipeline/Movements, Pricing vector)
    require explicit function calls with the uploaded file path / BytesIO
    and are therefore **NOT** run here by default.
    """
    fetch_us_imports_exports()
    fetch_global_imports_exports()
    update_macro_table()
    print("✅ API pipelines updated.")

    if not api_only:
        print("⚠️  Excel‑based pipelines need file paths. "
              "Call their import_*() functions explicitly.")


# ------------------------------------------------------------------------------
# CLI:   python app/modules/data_processing.py  [excel1] [excel2] [...]
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Examples
    --------
    # Update only API datasets
    $ python data_processing.py

    # Update API datasets and ingest three Excel files
    $ python data_processing.py  --wpr WeeklyPetroleumReport.xlsx \
                                 --pipe CRUDE_SD.xlsm \
                                 --price VectorDBPricing.xlsm
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run data‑ingestion pipelines.")
    parser.add_argument("--wpr",   help="WeeklyPetroleumReport.xlsx path")
    parser.add_argument("--pipe",  help="CRUDE SD.xlsm path (pipeline + movements)")
    parser.add_argument("--price", help="VectorDBPricing.xlsm path")
    args = parser.parse_args()

    update_all(api_only=False)

    if args.wpr:
        import_weekly_petroleum_report(args.wpr, save_sql=True)

    if args.pipe:
        import_pipeline_flow_excel(args.pipe, save_sql=True)

    if args.price:
        import_vector_db_pricing(args.price, save_sql=True)

