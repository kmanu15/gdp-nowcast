"""
models/news.py — News decomposition engine.

Computes how much each new data release moved the nowcast,
and in which direction. This is the feature that makes the
model feel like a real macro research tool.

Methodology: Banbura, Giannone, Modugno & Reichlin (2013).
"Now-casting and the real-time data flow."

The key insight: the nowcast revision between two dates equals
the weighted sum of "news" (actual minus expectation) across all
releases in that window. We approximate expectations as the
model's one-step-ahead forecast for each series.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config import INDICATORS, TARGET_SERIES_ID

logger = logging.getLogger(__name__)

SERIES_NAMES = {s.fred_id: s.name for s in INDICATORS}
SERIES_GROUPS = {s.fred_id: s.group for s in INDICATORS}


@dataclass
class Release:
    """A single data release event."""
    date: pd.Timestamp
    series_id: str
    series_name: str
    actual: float
    expectation: float
    surprise: float
    nowcast_before: float
    nowcast_after: float
    contribution: float
    group: str


def compute_news(
    panel_before: pd.DataFrame,
    panel_after: pd.DataFrame,
    model,
    release_date: Optional[pd.Timestamp] = None,
) -> list[Release]:
    """
    Decompose the change in nowcast between two vintage panels
    into contributions from each newly released series.

    Args:
        panel_before: Data panel as of the earlier date
        panel_after:  Data panel as of the later date
        model:        A fitted DFMNowcaster or BridgeModel
        release_date: The date of the data release (for labeling)

    Returns:
        List of Release objects, one per series with new data
    """
    nowcast_before_result = model.nowcast(panel_before)
    nowcast_after_result = model.nowcast(panel_after)

    nowcast_before = nowcast_before_result["nowcast"]
    nowcast_after = nowcast_after_result["nowcast"]
    total_revision = nowcast_after - nowcast_before

    new_data_cols = []
    for col in panel_after.columns:
        if col == TARGET_SERIES_ID:
            continue
        last_before = panel_before[col].last_valid_index()
        last_after = panel_after[col].last_valid_index()
        if last_after is not None and (last_before is None or last_after > last_before):
            new_data_cols.append(col)

    if not new_data_cols:
        logger.info("No new data between the two panels")
        return []

    releases = []
    for col in new_data_cols:
        last_after_idx = panel_after[col].last_valid_index()
        actual = panel_after[col].loc[last_after_idx]

        panel_minus_one = panel_after.copy()
        panel_minus_one.loc[last_after_idx, col] = np.nan
        nowcast_minus_one = model.nowcast(panel_minus_one)["nowcast"]

        contribution = nowcast_after - nowcast_minus_one

        fitted_before = nowcast_before_result.get("fitted")
        if fitted_before is not None and last_after_idx in fitted_before.index:
            expectation = fitted_before.loc[last_after_idx]
        else:
            prev_vals = panel_before[col].dropna()
            expectation = prev_vals.iloc[-1] if len(prev_vals) > 0 else actual

        releases.append(Release(
            date=release_date or last_after_idx,
            series_id=col,
            series_name=SERIES_NAMES.get(col, col),
            actual=round(actual, 4),
            expectation=round(expectation, 4),
            surprise=round(actual - expectation, 4),
            nowcast_before=round(nowcast_before, 3),
            nowcast_after=round(nowcast_after, 3),
            contribution=round(contribution, 3),
            group=SERIES_GROUPS.get(col, "other"),
        ))

    releases.sort(key=lambda r: abs(r.contribution), reverse=True)
    return releases


def news_table(releases: list[Release]) -> pd.DataFrame:
    """Format news decomposition results as a clean DataFrame."""
    if not releases:
        return pd.DataFrame()

    rows = []
    for r in releases:
        rows.append({
            "Series": r.series_name,
            "Group": r.group,
            "Actual": r.actual,
            "Expected": r.expectation,
            "Surprise": r.surprise,
            "Nowcast contribution (pp)": r.contribution,
        })

    df = pd.DataFrame(rows)
    total = df["Nowcast contribution (pp)"].sum()
    summary = pd.DataFrame([{
        "Series": "TOTAL REVISION",
        "Group": "",
        "Actual": np.nan,
        "Expected": np.nan,
        "Surprise": np.nan,
        "Nowcast contribution (pp)": round(total, 3),
    }])
    return pd.concat([df, summary], ignore_index=True)


def narrative_summary(releases: list[Release], current_quarter: str) -> str:
    """
    Generate a plain-English summary of the news decomposition,
    suitable for a research note or dashboard.
    """
    if not releases:
        return "No new data releases to report."

    total = sum(r.contribution for r in releases)
    direction = "revised up" if total > 0 else "revised down"
    abs_total = abs(total)

    top = max(releases, key=lambda r: abs(r.contribution))
    top_dir = "above" if top.surprise > 0 else "below"

    lines = [
        f"The {current_quarter} GDP nowcast was {direction} by {abs_total:.2f}pp "
        f"following this week's data releases.",
        "",
        f"The largest mover was {top.series_name}, which came in {abs(top.surprise):.2f} "
        f"units {top_dir} model expectations, contributing {top.contribution:+.2f}pp to the revision.",
    ]

    by_group = {}
    for r in releases:
        by_group.setdefault(r.group, []).append(r.contribution)

    group_contribs = {g: round(sum(v), 3) for g, v in by_group.items() if sum(v) != 0}
    if len(group_contribs) > 1:
        group_strs = [f"{g}: {v:+.2f}pp" for g, v in sorted(group_contribs.items(), key=lambda x: -abs(x[1]))]
        lines.append("")
        lines.append("Contributions by category: " + ", ".join(group_strs) + ".")

    return "\n".join(lines)
