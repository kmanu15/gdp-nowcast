"""
dashboard.py — Streamlit nowcasting dashboard.

Displays the current nowcast, historical track record,
news decomposition, and factor dynamics.

Run with:
    streamlit run dashboard.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="GDP Nowcast",
    page_icon="",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("GDP Nowcasting Model")
st.sidebar.markdown("Real-time estimates of U.S. GDP growth using mixed-frequency data.")

use_dfm = st.sidebar.toggle("Use Dynamic Factor Model", value=False,
                             help="DFM is more accurate but slower. Bridge equations are instant.")
show_backtest = st.sidebar.toggle("Show backtest results", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data source:** FRED (St. Louis Fed)  \n**Model:** Bridge + DFM")


# ── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_panel():
    try:
        from data.ingest import ingest
        return ingest(save=True)
    except Exception as e:
        st.error(f"Data ingestion failed: {e}")
        return None


@st.cache_data(ttl=3600)
def run_models(use_dfm_flag: bool):
    from pipeline import run_nowcast
    panel = load_panel()
    if panel is None:
        return None
    return run_nowcast(panel, use_dfm=use_dfm_flag)


with st.spinner("Loading data and running models..."):
    result = run_models(use_dfm)

if result is None:
    st.error("Could not load data. Check your FRED_API_KEY.")
    st.stop()

panel = result["panel"]
bridge = result["bridge"]
dfm = result["dfm"]
quarter = result["quarter"]
eval_date = result["eval_date"]

from config import TARGET_SERIES_ID, INDICATORS
SERIES_NAMES = {s.fred_id: s.name for s in INDICATORS}
SERIES_GROUPS = {s.fred_id: s.group for s in INDICATORS}


# ── Header ───────────────────────────────────────────────────────────────────

st.title(f"U.S. GDP Nowcast — {quarter}")
st.caption(f"As of {eval_date}  ·  Data via FRED")

col1, col2, col3 = st.columns(3)

bridge_val = bridge["nowcast"]
col1.metric(
    "Bridge nowcast",
    f"{bridge_val:+.2f}%",
    help="Equal-weighted average of bridge equation predictions (QoQ annualized)"
)

if dfm:
    col2.metric("DFM nowcast", f"{dfm['nowcast']:+.2f}%",
                help="Dynamic Factor Model with Kalman filter")
    ensemble = round((bridge_val + dfm["nowcast"]) / 2, 2)
    col3.metric("Ensemble", f"{ensemble:+.2f}%", help="Simple average of Bridge and DFM")
else:
    col2.metric("DFM nowcast", "—", help="Enable DFM in sidebar")
    col3.metric("Active equations", str(bridge["n_equations"]))

st.divider()


# ── Nowcast chart ─────────────────────────────────────────────────────────────

st.subheader("Nowcast vs. actual GDP")

gdp_actual = panel[TARGET_SERIES_ID].resample("QE").last().dropna()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=gdp_actual.index, y=gdp_actual.values,
    name="Actual GDP growth",
    marker_color=["#1D9E75" if v >= 0 else "#D85A30" for v in gdp_actual.values],
    opacity=0.7,
))

current_q_end = pd.Timestamp(eval_date).to_period("Q").end_time
fig.add_trace(go.Scatter(
    x=[current_q_end], y=[bridge_val],
    mode="markers+text",
    name="Bridge nowcast",
    marker=dict(size=14, color="#534AB7", symbol="diamond"),
    text=[f"{bridge_val:+.1f}%"],
    textposition="top center",
))

fig.update_layout(
    height=320,
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    yaxis_title="% QoQ annualized",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)", zeroline=True, zerolinecolor="rgba(128,128,128,0.3)")
st.plotly_chart(fig, use_container_width=True)


# ── News decomposition ────────────────────────────────────────────────────────

st.subheader("Bridge equation contributions")
st.caption("How much each indicator is pulling the nowcast up or down")

if bridge["components"]:
    comp_df = pd.DataFrame(
        [(SERIES_NAMES.get(k, k), SERIES_GROUPS.get(k, "other"), v)
         for k, v in bridge["components"].items()],
        columns=["Indicator", "Group", "Contribution (pp)"]
    ).sort_values("Contribution (pp)")

    colors = ["#D85A30" if v < 0 else "#1D9E75" for v in comp_df["Contribution (pp)"]]

    fig2 = go.Figure(go.Bar(
        x=comp_df["Contribution (pp)"],
        y=comp_df["Indicator"],
        orientation="h",
        marker_color=colors,
    ))
    fig2.update_layout(
        height=max(200, len(comp_df) * 36),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Contribution to nowcast (pp)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig2.update_xaxes(gridcolor="rgba(128,128,128,0.1)", zeroline=True, zerolinecolor="rgba(128,128,128,0.3)")
    fig2.update_yaxes(showgrid=False)
    st.plotly_chart(fig2, use_container_width=True)


# ── Indicator panel ───────────────────────────────────────────────────────────

st.subheader("Indicator dashboard")
st.caption("Recent history of key monthly indicators")

indicator_ids = [c for c in panel.columns if c != TARGET_SERIES_ID]
n_cols = 3
rows = [indicator_ids[i:i+n_cols] for i in range(0, len(indicator_ids), n_cols)]

for row in rows:
    cols = st.columns(n_cols)
    for col, ind_id in zip(cols, row):
        series = panel[ind_id].dropna().tail(24)
        if series.empty:
            continue
        name = SERIES_NAMES.get(ind_id, ind_id)
        last_val = series.iloc[-1]
        prev_val = series.iloc[-2] if len(series) > 1 else last_val
        delta = last_val - prev_val

        fig_sm = go.Figure(go.Scatter(
            x=series.index, y=series.values,
            mode="lines",
            line=dict(width=1.5, color="#534AB7"),
            fill="tozeroy",
            fillcolor="rgba(83,74,183,0.08)",
        ))
        fig_sm.update_layout(
            height=100,
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        with col:
            st.caption(name)
            st.metric("Latest", f"{last_val:.2f}", f"{delta:+.2f}")
            st.plotly_chart(fig_sm, use_container_width=True)


# ── Backtest ──────────────────────────────────────────────────────────────────

if show_backtest:
    st.divider()
    st.subheader("Historical backtest")
    bt_path = "output/backtest_results.csv"
    try:
        bt = pd.read_csv(bt_path)
        bt["error_bridge"] = pd.to_numeric(bt["error_bridge"], errors="coerce")
        bt["error_dfm"] = pd.to_numeric(bt["error_dfm"], errors="coerce")
        rmse_bridge = np.sqrt((bt["error_bridge"] ** 2).mean())
        rmse_dfm = np.sqrt((bt["error_dfm"] ** 2).mean())

        c1, c2, c3 = st.columns(3)
        c1.metric("Bridge RMSE", f"{rmse_bridge:.2f}pp")
        c2.metric("DFM RMSE", f"{rmse_dfm:.2f}pp")
        c3.metric("Sample", "2005–2019")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt["eval_date"], y=bt["actual"], name="Actual GDP", line=dict(color="#1D9E75", width=2)))
        fig_bt.add_trace(go.Scatter(x=bt["eval_date"], y=bt["nowcast_bridge"], name="Bridge model", line=dict(color="#534AB7", dash="dash")))
        fig_bt.add_trace(go.Scatter(x=bt["eval_date"], y=bt["nowcast_dfm"], name="DFM", line=dict(color="#D85A30", dash="dot")))
        fig_bt.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="% QoQ annualized",
        )
        fig_bt.update_xaxes(showgrid=False)
        fig_bt.update_yaxes(gridcolor="rgba(128,128,128,0.1)", zeroline=True, zerolinecolor="rgba(128,128,128,0.3)")
        st.plotly_chart(fig_bt, use_container_width=True)
        st.dataframe(bt.tail(12), use_container_width=True)

    except FileNotFoundError:
        st.info("Run `python pipeline.py --backtest` first to generate backtest results.")