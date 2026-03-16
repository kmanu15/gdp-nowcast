"""
data/ingest.py — FRED data ingestion with vintage tracking.

Pulls all configured series from FRED, applies transformations,
and saves a timestamped vintage snapshot so backtests can reconstruct
the information set available on any past date.
"""

import os
import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import numpy as np

from config import INDICATORS, SeriesConfig, TransformCode, DATA_DIR, VINTAGE_DIR

logger = logging.getLogger(__name__)


def get_fred_api():
    """Return a fredapi.Fred instance. Requires FRED_API_KEY env var."""
    try:
        from fredapi import Fred
    except ImportError:
        raise ImportError("Install fredapi: pip install fredapi")

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Set FRED_API_KEY environment variable. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=api_key)


def apply_transform(series: pd.Series, transform: TransformCode) -> pd.Series:
    """Apply a stationarity-inducing transformation to a raw series."""
    if transform == "none":
        return series
    elif transform == "log":
        return np.log(series)
    elif transform == "diff":
        return series.diff()
    elif transform == "log_diff":
        return np.log(series).diff()
    elif transform == "pct_change":
        return series.pct_change() * 100
    elif transform == "yoy":
        return series.pct_change(periods=4 if series.index.freqstr.startswith("Q") else 12) * 100
    else:
        raise ValueError(f"Unknown transform: {transform}")


def fetch_series(fred, config: SeriesConfig) -> pd.Series:
    """Fetch a single series from FRED and apply transformation."""
    logger.info(f"Fetching {config.fred_id} ({config.name})")
    raw = fred.get_series(config.fred_id)
    raw.name = config.fred_id
    transformed = apply_transform(raw, config.transform)
    transformed.name = config.fred_id
    return transformed


def build_panel(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """
    Merge all series into a single DataFrame with a monthly DatetimeIndex.
    Daily/weekly series are resampled to month-end averages.
    Quarterly series are forward-filled to monthly frequency.
    """
    monthly = {}
    freq_map = {s.fred_id: s.frequency for s in INDICATORS}

    for fred_id, series in series_dict.items():
        freq = freq_map.get(fred_id, "M")
        if freq in ("D", "W"):
            monthly[fred_id] = series.resample("ME").mean()
        elif freq == "Q":
            monthly[fred_id] = series.resample("ME").last().ffill()
        else:
            series.index = series.index.to_period("M").to_timestamp("M")
            monthly[fred_id] = series

    panel = pd.DataFrame(monthly)
    panel.index.name = "date"
    return panel


def save_vintage(panel: pd.DataFrame, vintage_dir: str = VINTAGE_DIR) -> Path:
    """
    Save a timestamped copy of the panel (a 'vintage').
    Filename encodes the download date so backtests can load
    the correct vintage for any evaluation date.
    """
    Path(vintage_dir).mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()
    path = Path(vintage_dir) / f"vintage_{today}.parquet"
    panel.to_parquet(path)
    logger.info(f"Saved vintage: {path}")

    index_path = Path(vintage_dir) / "vintage_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {}
    index[today] = str(path)
    index_path.write_text(json.dumps(index, indent=2))

    return path


def load_vintage(as_of: str, vintage_dir: str = VINTAGE_DIR) -> pd.DataFrame:
    """
    Load the most recent vintage available as of a given date.
    Enables real-time backtesting without look-ahead bias.
    """
    index_path = Path(vintage_dir) / "vintage_index.json"
    if not index_path.exists():
        raise FileNotFoundError("No vintages found. Run ingest() first.")

    index = json.loads(index_path.read_text())
    available = sorted(k for k in index.keys() if k <= as_of)
    if not available:
        raise ValueError(f"No vintage available as of {as_of}")

    chosen = available[-1]
    logger.info(f"Loading vintage from {chosen} (requested as_of={as_of})")
    return pd.read_parquet(index[chosen])


def ingest(save: bool = True) -> pd.DataFrame:
    """
    Main entry point. Fetches all configured series from FRED,
    builds a monthly panel, optionally saves a vintage, and returns
    the DataFrame.
    """
    fred = get_fred_api()
    series_dict = {}

    for config in INDICATORS:
        try:
            series_dict[config.fred_id] = fetch_series(fred, config)
        except Exception as e:
            logger.warning(f"Failed to fetch {config.fred_id}: {e}")

    panel = build_panel(series_dict)

    raw_dir = Path(DATA_DIR) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for fred_id, series in series_dict.items():
        series.to_csv(raw_dir / f"{fred_id}.csv", header=True)

    if save:
        save_vintage(panel)

    logger.info(f"Panel shape: {panel.shape} | Date range: {panel.index[0]} to {panel.index[-1]}")
    return panel


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    panel = ingest()
    print(panel.tail())
    print(f"\nMissing data by series:\n{panel.isna().sum()}")
