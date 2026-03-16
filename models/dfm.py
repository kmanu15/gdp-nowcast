"""
models/dfm.py — Dynamic Factor Model wrapper.

Wraps statsmodels DynamicFactorMQ, which handles mixed-frequency
data and the ragged edge natively via the Kalman filter.

Reference: Banbura & Modugno (2014), "Maximum likelihood estimation
of factor models on datasets with arbitrary pattern of missing data."
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import TARGET_SERIES_ID, INDICATOR_IDS, MODEL_PARAMS, INDICATORS

logger = logging.getLogger(__name__)


def build_dfm_input(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Prepare the mixed-frequency panel for DynamicFactorMQ.
    - Monthly indicators: pass as-is
    - Quarterly target: statsmodels expects a specific column spec

    Returns (data, freq_map) where freq_map tells the model
    which columns are monthly vs quarterly.
    """
    freq_map = {s.fred_id: s.frequency for s in INDICATORS}

    cols = [c for c in panel.columns if c in INDICATOR_IDS + [TARGET_SERIES_ID]]
    data = panel[cols].copy()

    column_info = {}
    for col in cols:
        f = freq_map.get(col, "M")
        if f in ("D", "W", "M"):
            column_info[col] = {"freq": "M"}
        else:
            column_info[col] = {"freq": "Q"}

    return data, column_info


class DFMNowcaster:
    """
    Dynamic Factor Model for real-time GDP nowcasting.

    The model extracts k common factors from a panel of indicators
    using the EM algorithm + Kalman smoother. Because the Kalman filter
    handles missing values naturally, it can work with the ragged edge
    at the end of the sample without imputation.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or MODEL_PARAMS["dfm"]
        self.model = None
        self.result = None
        self._fitted = False

    def fit(self, panel: pd.DataFrame) -> "DFMNowcaster":
        """
        Fit the DFM via EM algorithm.
        Note: this can take 1-5 minutes on a full sample — normal.
        """
        try:
            from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
        except ImportError:
            raise ImportError("statsmodels >= 0.13 required: pip install statsmodels")

        data, col_info = build_dfm_input(panel)

        factors = {"Global": {"variables": list(data.columns), "factors_order": 1}}

        logger.info(
            f"Fitting DFM: {self.params['k_factors']} factor(s), "
            f"{data.shape[1]} series, {data.shape[0]} periods"
        )

        self.model = DynamicFactorMQ(
            data,
            factors=self.params["k_factors"],
            idiosyncratic_ar1=True,
        )

        self.result = self.model.fit(
            disp=False,
            maxiter=self.params["em_iter"],
        )
        self._fitted = True
        logger.info("DFM fitted successfully")
        return self

    def nowcast(self, panel: pd.DataFrame) -> dict:
        """
        Generate a nowcast for the current quarter.
        Appends the latest data to the fitted model and runs
        the Kalman filter forward to the end of the ragged edge.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .nowcast()")

        applied = self.result.apply(panel, refit=False)
        smoothed = applied.smoother_results

        target_col_idx = list(panel.columns).index(TARGET_SERIES_ID)
        filtered_state = smoothed.smoothed_state

        fitted_raw = self.result.fittedvalues[TARGET_SERIES_ID].dropna()
        gdp_actual = panel[TARGET_SERIES_ID].dropna()
        gdp_mean = float(gdp_actual.mean())
        gdp_std = float(gdp_actual.std())
        fitted_rescaled = fitted_raw * gdp_std + gdp_mean
        cutoff = fitted_rescaled.index[-1] - pd.DateOffset(months=3)
        recent = fitted_rescaled.loc[fitted_rescaled.index >= cutoff]
        nowcast_val = float(recent.mean()) if len(recent) > 0 else float(fitted_rescaled.iloc[-1])

        factors_df = pd.DataFrame(
            filtered_state[:self.params["k_factors"]].T,
            index=panel.index,
            columns=[f"Factor_{i+1}" for i in range(self.params["k_factors"])],
        )

        return {
            "nowcast": round(nowcast_val, 3),
            "factors": factors_df,
            "fitted": self.result.fittedvalues[TARGET_SERIES_ID],
        }

    def forecast_errors(self) -> pd.Series:
        """In-sample forecast errors on the target series."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        actual = self.model.endog[TARGET_SERIES_ID]
        fitted = self.result.fittedvalues[TARGET_SERIES_ID]
        return (actual - fitted).dropna()

    @property
    def factor_loadings(self) -> pd.DataFrame:
        """Factor loadings for each series (how much each indicator loads on each factor)."""
        if not self._fitted:
            raise RuntimeError("Model not fitted")
        params = self.result.params
        loading_keys = [k for k in params.index if "loading" in k.lower()]
        return pd.Series(params[loading_keys], name="loading")
