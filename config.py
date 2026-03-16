"""
config.py — Series definitions, transformations, and model parameters.
Edit this file to add/remove indicators or tune the model.
"""

from dataclasses import dataclass, field
from typing import Literal

TransformCode = Literal["none", "log", "diff", "log_diff", "pct_change", "yoy"]


@dataclass
class SeriesConfig:
    fred_id: str
    name: str
    frequency: Literal["D", "W", "M", "Q"]
    transform: TransformCode
    release_lag_days: int  # typical days after period end before release
    group: str             # used for news decomposition grouping


INDICATORS: list[SeriesConfig] = [
    SeriesConfig(
        fred_id="A191RL1Q225SBEA",
        name="Real GDP growth (QoQ ann.)",
        frequency="Q",
        transform="none",
        release_lag_days=30,
        group="target",
    ),
    SeriesConfig(
        fred_id="PAYEMS",
        name="Nonfarm payrolls",
        frequency="M",
        transform="diff",
        release_lag_days=7,
        group="labor",
    ),
    SeriesConfig(
        fred_id="ICSA",
        name="Initial jobless claims",
        frequency="W",
        transform="log",
        release_lag_days=5,
        group="labor",
    ),
    SeriesConfig(
        fred_id="PCE",
        name="Real personal consumption",
        frequency="M",
        transform="pct_change",
        release_lag_days=30,
        group="consumption",
    ),
    SeriesConfig(
        fred_id="RSXFS",
        name="Retail sales ex. food services",
        frequency="M",
        transform="pct_change",
        release_lag_days=14,
        group="consumption",
    ),
    SeriesConfig(
        fred_id="INDPRO",
        name="Industrial production",
        frequency="M",
        transform="pct_change",
        release_lag_days=16,
        group="production",
    ),
    SeriesConfig(
        fred_id="MANEMP",
        name="Manufacturing employment",
        frequency="M",
        transform="diff",
        release_lag_days=7,
        group="production",
    ),
    SeriesConfig(
        fred_id="T10Y2Y",
        name="10Y-2Y yield spread",
        frequency="D",
        transform="none",
        release_lag_days=1,
        group="financial",
    ),
    SeriesConfig(
        fred_id="BAMLH0A0HYM2",
        name="HY credit spread (OAS)",
        frequency="D",
        transform="none",
        release_lag_days=1,
        group="financial",
    ),
]

TARGET_SERIES_ID = "A191RL1Q225SBEA"

INDICATOR_IDS = [s.fred_id for s in INDICATORS if s.group != "target"]


MODEL_PARAMS = {
    "bridge": {
        "min_obs": 20,           # minimum observations for OLS estimation
        "window": None,          # None = expanding window, int = rolling
    },
    "dfm": {
        "k_factors": 2,          # number of common factors
        "error_order": 1,        # AR order for idiosyncratic errors
        "em_iter": 500,          # max EM iterations
        "em_tol": 1e-6,
    },
}

BACKTEST = {
    "start_date": "2005-01-01",
    "end_date": "2019-12-31",   # stop before COVID distortions
    "eval_frequency": "QS",     # evaluate once per quarter-start
}

DATA_DIR = "data"
VINTAGE_DIR = "data/vintages"
OUTPUT_DIR = "output"
