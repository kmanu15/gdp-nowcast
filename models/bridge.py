"""
models/bridge.py — Bridge equation nowcasting model.

Estimates current-quarter GDP growth by regressing quarterly GDP
on monthly averages of each indicator, then aggregating predictions.
A natural starting point: simple, interpretable, and easy to explain
in an interview.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from config import TARGET_SERIES_ID, INDICATOR_IDS, MODEL_PARAMS

logger = logging.getLogger(__name__)


def to_quarterly(monthly: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate a monthly panel to quarterly frequency by taking
    the within-quarter mean of each indicator.
    GDP (the target) is already quarterly — take the last value.
    """
    q = monthly.resample("QE").mean()
    if TARGET_SERIES_ID in monthly.columns:
        q[TARGET_SERIES_ID] = monthly[TARGET_SERIES_ID].resample("QE").last()
    return q


def get_quarter_to_date_means(monthly: pd.DataFrame, eval_date: pd.Timestamp) -> pd.Series:
    """
    For the current (incomplete) quarter, compute the mean of each
    indicator using only the months observed so far.
    This is the 'ragged edge' — not all months are available yet.
    """
    q_start = eval_date.to_period("Q").start_time
    current_q = monthly.loc[q_start:eval_date]
    return current_q.mean()


class BridgeModel:
    """
    One OLS regression per indicator: GDP_Q ~ mean(X_monthly in Q).
    Final nowcast is the equal-weighted average of each bridge equation's
    prediction. A simple but production-relevant benchmark.
    """

    def __init__(self, params: Optional[dict] = None):
        self.params = params or MODEL_PARAMS["bridge"]
        self.models: dict[str, LinearRegression] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.in_sample_r2: dict[str, float] = {}

    def fit(self, quarterly: pd.DataFrame) -> "BridgeModel":
        """Fit one OLS bridge equation per indicator."""
        target = quarterly[TARGET_SERIES_ID].dropna()

        for indicator in INDICATOR_IDS:
            if indicator not in quarterly.columns:
                continue

            xy = pd.concat([quarterly[indicator], target], axis=1).dropna()
            if len(xy) < self.params["min_obs"]:
                logger.warning(f"Skipping {indicator}: only {len(xy)} obs")
                continue

            X = xy[[indicator]].values
            y = xy[TARGET_SERIES_ID].values

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LinearRegression()
            model.fit(X_scaled, y)

            self.scalers[indicator] = scaler
            self.models[indicator] = model
            self.in_sample_r2[indicator] = model.score(X_scaled, y)

        logger.info(f"Fitted {len(self.models)} bridge equations")
        return self

    def predict(self, current_q_means: pd.Series) -> dict:
        """
        Generate a nowcast from quarter-to-date indicator means.
        Returns the ensemble mean and each individual bridge prediction.
        """
        predictions = {}

        for indicator, model in self.models.items():
            if indicator not in current_q_means.index or pd.isna(current_q_means[indicator]):
                continue
            x = np.array([[current_q_means[indicator]]])
            x_scaled = self.scalers[indicator].transform(x)
            predictions[indicator] = model.predict(x_scaled)[0]

        if not predictions:
            return {"nowcast": np.nan, "components": {}}

        nowcast = np.mean(list(predictions.values()))
        return {
            "nowcast": round(nowcast, 3),
            "components": {k: round(v, 3) for k, v in predictions.items()},
            "n_equations": len(predictions),
        }

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame of R² and coefficients for each bridge equation."""
        rows = []
        for ind, model in self.models.items():
            rows.append({
                "indicator": ind,
                "r2": round(self.in_sample_r2.get(ind, np.nan), 3),
                "coef": round(model.coef_[0], 4),
                "intercept": round(model.intercept_, 4),
            })
        return pd.DataFrame(rows).sort_values("r2", ascending=False)
