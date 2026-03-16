"""
pipeline.py — Main orchestration script.

Run this to execute the full nowcasting pipeline:
  1. Ingest latest data from FRED and save a vintage
  2. Fit the bridge model
  3. Fit the Dynamic Factor Model
  4. Generate a nowcast for the current quarter
  5. Run news decomposition vs. last vintage
  6. Print a research-style summary

Usage:
    python pipeline.py
    python pipeline.py --backtest          # run historical evaluation
    python pipeline.py --no-dfm            # skip DFM (faster, bridge only)
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def current_quarter_label() -> str:
    today = date.today()
    q = (today.month - 1) // 3 + 1
    return f"Q{q} {today.year}"


def run_nowcast(panel: pd.DataFrame, use_dfm: bool = True) -> dict:
    from models.bridge import BridgeModel, to_quarterly, get_quarter_to_date_means
    from config import TARGET_SERIES_ID, INDICATOR_IDS

    indicators = [c for c in panel.columns if c in INDICATOR_IDS]
    target = panel[[TARGET_SERIES_ID]] if TARGET_SERIES_ID in panel.columns else None

    quarterly = to_quarterly(panel)
    bridge = BridgeModel()
    bridge.fit(quarterly)
    qtd_means = get_quarter_to_date_means(panel, panel.index[-1])
    bridge_result = bridge.predict(qtd_means)

    result = {
        "quarter": current_quarter_label(),
        "eval_date": panel.index[-1].date().isoformat(),
        "bridge": bridge_result,
        "bridge_model": bridge,
        "dfm": None,
        "dfm_model": None,
        "panel": panel,
    }

    if use_dfm:
        try:
            from models.dfm import DFMNowcaster
            dfm = DFMNowcaster()
            dfm.fit(panel)
            dfm_result = dfm.nowcast(panel)
            result["dfm"] = dfm_result
            result["dfm_model"] = dfm
        except Exception as e:
            logger.warning(f"DFM failed: {e}. Falling back to bridge only.")

    return result


def run_backtest(start: str, end: str) -> pd.DataFrame:
    from models.bridge import BridgeModel, to_quarterly
    from config import TARGET_SERIES_ID, BACKTEST
    from data.ingest import load_vintage

    eval_dates = pd.date_range(start, end, freq=BACKTEST["eval_frequency"])
    panel = load_vintage(date.today().isoformat())
    gdp_actual = panel[TARGET_SERIES_ID].resample("QE").last().dropna()

    records = []
    for eval_date in eval_dates:
        try:
            simulated = panel.loc[:eval_date].copy()

            quarterly = to_quarterly(simulated)

            bridge = BridgeModel()
            bridge.fit(quarterly)

            indicator_cols = [c for c in simulated.columns if c != TARGET_SERIES_ID]
            qtd_means = simulated[indicator_cols].iloc[-3:].mean()

            pred = bridge.predict(qtd_means)

            current_q_end = eval_date.to_period("Q").end_time
            mask = gdp_actual.index <= current_q_end + pd.DateOffset(months=1)
            actual_candidates = gdp_actual[mask]
            actual = float(actual_candidates.iloc[-1]) if len(actual_candidates) > 0 else np.nan

            records.append({
                "eval_date": eval_date.date(),
                "quarter": eval_date.to_period("Q").strftime("Q%q %Y"),
                "nowcast_bridge": pred["nowcast"],
                "actual": round(actual, 3) if not np.isnan(actual) else np.nan,
                "error": round(pred["nowcast"] - actual, 3) if not np.isnan(actual) and not np.isnan(pred["nowcast"]) else np.nan,
            })
        except Exception as e:
            import traceback
            logger.warning(f"Backtest failed for {eval_date.date()}: {e}")
            logger.warning(traceback.format_exc())

    df = pd.DataFrame(records)
    if len(df) > 0 and "error" in df.columns and df["error"].notna().any():
        rmse = np.sqrt((df["error"] ** 2).mean())
        logger.info(f"Backtest RMSE (bridge): {rmse:.3f}pp")
    return df


def print_summary(result: dict):
    q = result["quarter"]
    eval_date = result["eval_date"]
    bridge = result["bridge"]
    dfm = result["dfm"]

    print("\n" + "=" * 58)
    print(f"  GDP NOWCAST SUMMARY — {q}")
    print(f"  As of {eval_date}")
    print("=" * 58)

    print(f"\n  Bridge model:       {bridge['nowcast']:+.2f}% (QoQ ann.)")
    print(f"  Equations active:   {bridge['n_equations']}")

    if dfm:
        print(f"  DFM nowcast:        {dfm['nowcast']:+.2f}% (QoQ ann.)")

    print("\n  Bridge contributions by indicator:")
    for ind, val in sorted(bridge["components"].items(), key=lambda x: -abs(x[1])):
        bar = "+" * int(abs(val) * 2) if val > 0 else "-" * int(abs(val) * 2)
        print(f"    {ind:<35} {val:+.3f}  {bar}")

    print("\n" + "=" * 58 + "\n")


def main():
    parser = argparse.ArgumentParser(description="GDP Nowcasting Pipeline")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--no-dfm", action="store_true", help="Skip DFM (faster)")
    parser.add_argument("--no-ingest", action="store_true", help="Use cached data")
    args = parser.parse_args()

    if not args.no_ingest:
        logger.info("Ingesting latest data from FRED...")
        from data.ingest import ingest
        panel = ingest(save=True)
    else:
        logger.info("Loading most recent cached vintage...")
        from data.ingest import load_vintage
        panel = load_vintage(date.today().isoformat())

    if args.backtest:
        from config import BACKTEST
        logger.info("Running backtest...")
        bt = run_backtest(BACKTEST["start_date"], BACKTEST["end_date"])
        bt_path = Path("output") / "backtest_results.csv"
        bt_path.parent.mkdir(exist_ok=True)
        bt.to_csv(bt_path, index=False)
        logger.info(f"Backtest results saved to {bt_path}")
        print(bt.tail(8).to_string(index=False))
    else:
        result = run_nowcast(panel, use_dfm=not args.no_dfm)
        print_summary(result)


if __name__ == "__main__":
    main()
