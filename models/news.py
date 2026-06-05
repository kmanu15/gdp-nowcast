"""
models/news.py — News decomposition engine.

Computes how much each new data release moved the Bridge Model nowcast,
and in which direction.

Methodology: Bańbura & Modugno (2014) leave-one-out attribution,
rescaled so contributions exactly sum to the total revision.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from config import INDICATORS, TARGET_SERIES_ID

logger = logging.getLogger(__name__)

SERIES_NAMES = {s.fred_id: s.name for s in INDICATORS}
SERIES_GROUPS = {s.fred_id: s.group for s in INDICATORS}

# Where to persist the last decomposition
_CACHE_PATH = Path("data/last_news_decomposition.json")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_nowcast(model, panel: pd.DataFrame) -> dict:
    from models.dfm import DFMNowcaster
    from models.bridge import BridgeModel, get_quarter_to_date_means

    if isinstance(model, DFMNowcaster):
        return model.nowcast_from_panel(panel)
    elif isinstance(model, BridgeModel):
        qtd_means = get_quarter_to_date_means(panel, panel.index[-1])
        return model.predict(qtd_means)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


def _detect_new_observations(
    panel_before: pd.DataFrame,
    panel_after: pd.DataFrame,
) -> dict[str, list[pd.Timestamp]]:
    """
    Detect both newly appended observations AND revised observations
    for each series between two vintage panels.

    Returns a dict mapping series_id -> list of new/revised timestamps.
    """
    changed: dict[str, list[pd.Timestamp]] = {}

    for col in panel_after.columns:
        if col == TARGET_SERIES_ID:
            continue

        after_series = panel_after[col].dropna()
        before_series = panel_before[col].dropna()

        new_dates = []

        # Case 1: Newly appended observations
        new_obs_dates = after_series.index.difference(before_series.index)
        new_dates.extend(new_obs_dates.tolist())

        # Case 2: Revised observations (same date, different value)
        common_dates = after_series.index.intersection(before_series.index)
        for date in common_dates:
            old_val = float(before_series.loc[date])
            new_val = float(after_series.loc[date])
            if not np.isclose(old_val, new_val, rtol=1e-6, atol=1e-9):
                new_dates.append(date)

        if new_dates:
            changed[col] = sorted(new_dates)

    return changed


def _get_expectation(
    col: str,
    date: pd.Timestamp,
    panel_before: pd.DataFrame,
    nowcast_before_result: dict,
) -> float:
    """
    Estimate the model's expectation for series `col` at `date`.

    Priority:
    1. DFM Kalman-smoother fitted values
    2. AR(1) extrapolation from series history (Bridge Model fallback)
    3. Last observed value (last resort)
    """
    fitted_all = nowcast_before_result.get("fitted_all")

    if fitted_all is not None and col in fitted_all.columns:
        fitted_col = fitted_all[col].dropna()
        if date in fitted_col.index:
            return float(fitted_col.loc[date])
        if not fitted_col.empty:
            return float(fitted_col.iloc[-1])

    prev_vals = panel_before[col].dropna()
    if len(prev_vals) >= 4:
        mu = prev_vals.mean()
        rho = float(np.clip(prev_vals.autocorr(lag=1) or 0.0, -0.95, 0.95))
        return float(mu + rho * (prev_vals.iloc[-1] - mu))
    elif len(prev_vals) >= 1:
        return float(prev_vals.iloc[-1])

    return float("nan")


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class Release:
    """A single data release event (one series, one observation date)."""
    release_date: pd.Timestamp
    obs_date: pd.Timestamp
    series_id: str
    series_name: str
    actual: float
    expectation: float
    surprise: float
    is_revision: bool
    nowcast_before: float
    nowcast_after: float
    contribution: float
    contribution_raw: float
    group: str


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _releases_to_json(releases: list[Release]) -> list[dict]:
    """Serialise Release objects to JSON-safe dicts."""
    out = []
    for r in releases:
        d = asdict(r)
        # Timestamps are not JSON-serialisable — convert to ISO strings
        d["release_date"] = r.release_date.isoformat() if pd.notna(r.release_date) else None
        d["obs_date"] = r.obs_date.isoformat() if pd.notna(r.obs_date) else None
        # Replace NaN with None so json.dumps doesn't choke
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                d[k] = None
        out.append(d)
    return out


def _releases_from_json(data: list[dict]) -> list[Release]:
    """Deserialise Release objects from JSON dicts."""
    releases = []
    for d in data:
        d["release_date"] = pd.Timestamp(d["release_date"]) if d["release_date"] else pd.NaT
        d["obs_date"] = pd.Timestamp(d["obs_date"]) if d["obs_date"] else pd.NaT
        for k, v in d.items():
            if v is None and k not in ("release_date", "obs_date"):
                d[k] = float("nan")
        releases.append(Release(**d))
    return releases


def save_news_cache(releases: list[Release], computed_at: pd.Timestamp) -> None:
    """
    Persist the latest news decomposition to disk.

    Called automatically by compute_news when new data is found.
    You can also call it explicitly if needed.
    """
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "computed_at": computed_at.isoformat(),
        "releases": _releases_to_json(releases),
    }
    with open(_CACHE_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("News decomposition cached to %s", _CACHE_PATH)


def load_news_cache() -> tuple[list[Release], pd.Timestamp | None]:
    """
    Load the most recently saved news decomposition from disk.

    Returns:
        (releases, computed_at) — releases is empty list if no cache exists.
    """
    if not _CACHE_PATH.exists():
        logger.info("No news cache found at %s", _CACHE_PATH)
        return [], None

    with open(_CACHE_PATH) as f:
        payload = json.load(f)

    computed_at = pd.Timestamp(payload["computed_at"])
    releases = _releases_from_json(payload["releases"])
    logger.info(
        "Loaded %d cached releases from %s (computed %s)",
        len(releases), _CACHE_PATH, computed_at.date(),
    )
    return releases, computed_at


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def compute_news(
    panel_before: pd.DataFrame,
    panel_after: pd.DataFrame,
    model,
    release_date: Optional[pd.Timestamp] = None,
) -> list[Release]:
    """
    Decompose the change in nowcast between two vintage panels
    into contributions from each newly released (or revised) data point.

    If new data is found, contributions are computed and the result is
    automatically saved to disk so it can be retrieved later via
    get_news_decomposition().

    Args:
        panel_before:  Data panel as of the earlier vintage
        panel_after:   Data panel as of the later vintage
        model:         A fitted DFMNowcaster or BridgeModel
        release_date:  Wall-clock date of the release (for labeling)

    Returns:
        List of Release objects sorted by |contribution|, largest first.
        Empty list if no new or revised data was detected.
    """
    nowcast_before_result = _get_nowcast(model, panel_before)
    nowcast_after_result = _get_nowcast(model, panel_after)

    nowcast_before = nowcast_before_result["nowcast"]
    nowcast_after = nowcast_after_result["nowcast"]
    total_revision = nowcast_after - nowcast_before

    changed = _detect_new_observations(panel_before, panel_after)

    if not changed:
        logger.info("No new or revised data between the two panels.")
        return []

    # --- Leave-one-out attribution ---
    raw_releases: list[Release] = []

    for col, obs_dates in changed.items():
        for obs_date in obs_dates:
            actual = float(panel_after[col].loc[obs_date])

            before_val = panel_before[col].get(obs_date, np.nan)
            is_revision = (
                not pd.isna(before_val)
                and not np.isclose(float(before_val), actual, rtol=1e-6, atol=1e-9)
            )

            panel_loo = panel_after.copy()
            panel_loo.loc[obs_date, col] = np.nan
            nowcast_loo = _get_nowcast(model, panel_loo)["nowcast"]
            contribution_raw = nowcast_after - nowcast_loo

            expectation = _get_expectation(
                col, obs_date, panel_before, nowcast_before_result
            )

            raw_releases.append(Release(
                release_date=release_date or obs_date,
                obs_date=obs_date,
                series_id=col,
                series_name=SERIES_NAMES.get(col, col),
                actual=round(actual, 4),
                expectation=round(expectation, 4) if not np.isnan(expectation) else np.nan,
                surprise=round(actual - expectation, 4) if not np.isnan(expectation) else np.nan,
                is_revision=is_revision,
                nowcast_before=round(nowcast_before, 3),
                nowcast_after=round(nowcast_after, 3),
                contribution=0.0,
                contribution_raw=round(contribution_raw, 4),
                group=SERIES_GROUPS.get(col, "other"),
            ))

    # --- Proportional rescaling so contributions sum exactly to total_revision ---
    sum_raw = sum(r.contribution_raw for r in raw_releases)
    scale = total_revision / sum_raw if abs(sum_raw) > 1e-9 else 1.0
    for r in raw_releases:
        r.contribution = round(r.contribution_raw * scale, 3)

    raw_releases.sort(key=lambda r: abs(r.contribution), reverse=True)

    # Persist so the dashboard can show it even when there's no new data
    computed_at = release_date or pd.Timestamp.now()
    save_news_cache(raw_releases, computed_at)

    return raw_releases


def get_news_decomposition(
    panel_before: pd.DataFrame,
    panel_after: pd.DataFrame,
    model,
    release_date: Optional[pd.Timestamp] = None,
) -> tuple[list[Release], bool, pd.Timestamp | None]:
    """
    High-level entry point for the dashboard.

    Tries to compute a fresh decomposition. If no new data is found,
    falls back to the most recently cached result.

    Returns:
        releases:     List of Release objects (fresh or cached)
        is_cached:    True if the result came from cache (no new data today)
        computed_at:  When the decomposition was originally computed
    """
    releases = compute_news(panel_before, panel_after, model, release_date)

    if releases:
        computed_at = release_date or pd.Timestamp.now()
        return releases, False, computed_at

    # No new data — fall back to cache
    cached_releases, computed_at = load_news_cache()
    if cached_releases:
        logger.info("No new data; showing cached decomposition from %s", computed_at)
        return cached_releases, True, computed_at

    # Nothing fresh, nothing cached
    return [], False, None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def news_table(releases: list[Release], is_cached: bool = False) -> pd.DataFrame:
    """
    Format news decomposition results as a clean DataFrame.
    Includes a TOTAL REVISION row at the bottom.
    """
    if not releases:
        return pd.DataFrame()

    rows = []
    for r in releases:
        rows.append({
            "Series": r.series_name,
            "Group": r.group,
            "Obs. Date": r.obs_date.strftime("%Y-%m-%d") if pd.notna(r.obs_date) else "",
            "Is Revision": "✓" if r.is_revision else "",
            "Actual": r.actual,
            "Expected": r.expectation,
            "Surprise": r.surprise,
            "Contribution (pp)": r.contribution,
        })

    df = pd.DataFrame(rows)

    total = df["Contribution (pp)"].sum()
    summary = pd.DataFrame([{
        "Series": "TOTAL REVISION",
        "Group": "",
        "Obs. Date": "",
        "Is Revision": "",
        "Actual": np.nan,
        "Expected": np.nan,
        "Surprise": np.nan,
        "Contribution (pp)": round(total, 3),
    }])

    return pd.concat([df, summary], ignore_index=True)


def narrative_summary(
    releases: list[Release],
    current_quarter: str,
    is_cached: bool = False,
    computed_at: Optional[pd.Timestamp] = None,
) -> str:
    """
    Generate a plain-English summary of the news decomposition.
    """
    if not releases:
        return "No news decomposition available."

    total = sum(r.contribution for r in releases)
    direction = "revised up" if total > 0 else "revised down"
    abs_total = abs(total)

    top = max(releases, key=lambda r: abs(r.contribution))
    top_dir = "above" if (top.surprise or 0) > 0 else "below"
    top_surprise_str = (
        f"{abs(top.surprise):.2f} units {top_dir} model expectations"
        if top.surprise is not None and not np.isnan(top.surprise)
        else "no model expectation available"
    )

    revisions = [r for r in releases if r.is_revision]

    # Stale-data banner
    if is_cached and computed_at:
        header = (
            f"*No new data releases since last update. "
            f"Showing decomposition from {computed_at.strftime('%d %b %Y')}.*\n"
        )
    else:
        header = ""

    lines = [
        header,
        f"The {current_quarter} GDP nowcast was {direction} by "
        f"{abs_total:.2f}pp following the latest data releases.",
        "",
        f"The largest contributor was **{top.series_name}** ({top_surprise_str}), "
        f"contributing {top.contribution:+.2f}pp to the revision.",
    ]

    if revisions:
        rev_names = ", ".join(r.series_name for r in revisions[:3])
        rev_contrib = sum(r.contribution for r in revisions)
        lines.append("")
        lines.append(
            f"Data revisions ({rev_names}) contributed a combined "
            f"{rev_contrib:+.2f}pp."
        )

    by_group: dict[str, list[float]] = {}
    for r in releases:
        by_group.setdefault(r.group, []).append(r.contribution)

    group_contribs = {
        g: round(sum(v), 3)
        for g, v in by_group.items()
        if abs(sum(v)) > 1e-4
    }
    if len(group_contribs) > 1:
        group_strs = [
            f"{g}: {v:+.2f}pp"
            for g, v in sorted(group_contribs.items(), key=lambda x: -abs(x[1]))
        ]
        lines.append("")
        lines.append("Contributions by category: " + ", ".join(group_strs) + ".")

    return "\n".join(lines)