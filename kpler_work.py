import pandas as pd
from datetime import date, timedelta
import os
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

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
