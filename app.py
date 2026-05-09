"""
Maritime Anomaly Detection Dashboard
=====================================
Interactive Streamlit application that streams live AIS data, detects
anomalies using Isolation Forest + DBSCAN, and visualises results on a
global map with supplementary charts.

Falls back to realistic sample data when the live WebSocket feed is
unavailable (e.g. invalid/expired API key).
"""

import asyncio
import logging
from typing import Optional

import pandas as pd
import streamlit as st

from config import AppConfig
from data_stream import AISDataStream
from data_cleaner import DataCleaner
from anomaly_detector import (
    BehaviourAnomalyDetector,
    SpatialClusterDetector,
    AISGapDetector,
    AnomalyScorer,
)
from visualizations import (
    create_anomaly_map,
    create_trend_chart,
    create_distribution_chart,
    create_correlation_heatmap,
)
from utils import highlight_anomalies

# ---------------------------------------------------------------------- #
#  Logging
# ---------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------- #
#  Streamlit page config
# ---------------------------------------------------------------------- #
cfg = AppConfig()
st.set_page_config(
    page_title=cfg.streamlit.page_title,
    page_icon=cfg.streamlit.page_icon,
    layout=cfg.streamlit.layout,
)

# ---------------------------------------------------------------------- #
#  Sidebar controls
# ---------------------------------------------------------------------- #
st.sidebar.title("Controls")

data_source = st.sidebar.radio(
    "Data Source",
    options=["Live AIS Stream", "Sample Data (Demo)"],
    index=0,
    help="Use Live AIS Stream for real-time data. "
         "Switch to Sample Data if your API key is expired or the feed is down.",
)

max_messages = st.sidebar.slider(
    "Messages to collect", 100, 5000, 1000, step=100
)
refresh_btn = st.sidebar.button("Refresh Data")

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **About**

    This dashboard ingests live AIS data via WebSocket, cleans it,
    runs ML anomaly detection (Isolation Forest + DBSCAN), and
    visualises suspicious vessel activity on a global map.
    """
)

# ---------------------------------------------------------------------- #
#  Async runner (Streamlit-safe)
# ---------------------------------------------------------------------- #
def _run_async(coro):
    """
    Run an async coroutine in a fresh event loop.

    Streamlit executes each script in its own thread, so we cannot
    rely on asyncio.get_event_loop() -- we must create a new one.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

# ---------------------------------------------------------------------- #
#  Data acquisition helpers
# ---------------------------------------------------------------------- #
def _fetch_live_data(max_msgs: int) -> Optional[pd.DataFrame]:
    """Connect to the live AIS WebSocket and return raw messages as a DataFrame."""
    try:
        streamer = AISDataStream(cfg)
        messages = _run_async(streamer.connect(max_messages=max_msgs))
        df = streamer.to_dataframe(messages)

        if df.empty:
            return None

        logger.info("Live stream returned %d records.", len(df))
        return df

    except Exception as exc:
        logger.error("Live stream failed: %s", exc)
        return None


def _fetch_sample_data() -> pd.DataFrame:
    """Load realistic sample data with seeded anomalies."""
    from sample_data import generate_sample_data_with_gaps
    df = generate_sample_data_with_gaps(n_vessels=200, seed=42)
    logger.info("Loaded %d sample records.", len(df))
    return df

# ---------------------------------------------------------------------- #
#  ML pipeline (shared by both data sources)
# ---------------------------------------------------------------------- #
def _run_ml_pipeline(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Clean, detect anomalies, and score the DataFrame."""
    try:
        # 1. Clean ----------------------------------------------------
        cleaner = DataCleaner(cfg)
        df = cleaner.clean(df)

        if df.empty:
            st.warning("All records were filtered during cleaning.")
            return None

        # 2. Behaviour anomalies (Isolation Forest) -------------------
        if_detector = BehaviourAnomalyDetector(cfg)
        if_detector.fit(df)
        df = if_detector.predict(df)

        # 3. Spatial clustering (DBSCAN) -----------------------------
        spatial_detector = SpatialClusterDetector(cfg)
        df = spatial_detector.detect(df)

        # 4. AIS gap detection ----------------------------------------
        gap_detector = AISGapDetector(cfg)
        df = gap_detector.detect(df)

        # 5. Combined scoring -----------------------------------------
        scorer = AnomalyScorer(cfg)
        df = scorer.score(df)

        return df

    except Exception as exc:
        st.error(f"ML pipeline error: {exc}")
        logger.error("Pipeline error", exc_info=exc)
        return None

# ---------------------------------------------------------------------- #
#  Main UI
# ---------------------------------------------------------------------- #
st.title(cfg.streamlit.page_title)

if data_source == "Live AIS Stream":
    st.markdown(
        """
        Real-time detection of abnormal cargo vessel behaviour using
        **Isolation Forest** (speed / heading anomalies) and
        **DBSCAN** (spatial clustering for STS / loitering).
        """
    )
else:
    st.markdown(
        """
        **Demo mode** -- running on realistic sample data with seeded
        anomalies.  Switch to *Live AIS Stream* in the sidebar when
        your API key is active.
        """
    )

# -- Data acquisition -------------------------------------------------- #
should_refresh = (
    "df" not in st.session_state
    or refresh_btn
    or st.session_state.get("_source") != data_source
)

if should_refresh:
    with st.spinner(
        f"Collecting up to {max_messages} AIS messages ..."
        if data_source == "Live AIS Stream"
        else "Loading sample data ..."
    ):
        if data_source == "Live AIS Stream":
            raw_df = _fetch_live_data(max_messages)

            # Auto-fallback: if live stream returns nothing, try sample
            if raw_df is None or raw_df.empty:
                st.warning(
                    "Live AIS stream returned no data. "
                    "This usually means the API key is expired or inactive. "
                    "Falling back to sample data so you can still explore the dashboard."
                )
                raw_df = _fetch_sample_data()
                data_source = "Sample Data (Demo)"

        else:
            raw_df = _fetch_sample_data()

        st.session_state.df = _run_ml_pipeline(raw_df)
        st.session_state._source = data_source

df = st.session_state.get("df")

if df is None or df.empty:
    st.info("No data available. Click **Refresh Data** in the sidebar to start.")
    st.stop()

# -- KPI row ----------------------------------------------------------- #
st.subheader("Key Metrics")
c1, c2, c3, c4 = st.columns(4)
total = len(df)
high = int((df["severity"] == "High").sum()) if "severity" in df.columns else 0
medium = int((df["severity"] == "Medium").sum()) if "severity" in df.columns else 0
unique_vessels = int(df["mmsi"].nunique()) if "mmsi" in df.columns else 0

c1.metric("Total Records", total)
c2.metric("Unique Vessels", unique_vessels)
c3.metric("High Anomaly", high, delta=f"Medium: {medium}")
c4.metric(
    "Anomaly Rate",
    f"{(high + medium) / total * 100:.1f}%" if total else "0%",
)

st.markdown("---")

# -- MAP --------------------------------------------------------------- #
st.subheader("Global Anomaly Map")
map_fig = create_anomaly_map(df, cfg)
st.plotly_chart(map_fig, use_container_width=True)

# -- Three required visualisations ------------------------------------- #
st.subheader("Analytics")
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(create_trend_chart(df, cfg), use_container_width=True)
with col_b:
    st.plotly_chart(create_distribution_chart(df, cfg), use_container_width=True)

st.plotly_chart(create_correlation_heatmap(df, cfg), use_container_width=True)

# -- Top anomalies table ----------------------------------------------- #
st.subheader("Top Flagged Vessels")
top = highlight_anomalies(df, top_n=20)
if not top.empty:
    st.dataframe(top, use_container_width=True, hide_index=True)
else:
    st.info("No anomalies detected in the current window.")

# -- Raw data preview -------------------------------------------------- #
with st.expander("Raw Data Preview"):
    st.dataframe(df.head(50), use_container_width=True)

# -- Footer ------------------------------------------------------------ #
st.markdown("---")
source_label = "Live AIS Stream" if st.session_state.get("_source") == "Live AIS Stream" else "Sample Data (Demo)"
st.caption(
    f"Maritime Anomaly Detector | {source_label} | "
    "Powered by Streamlit, scikit-learn, Plotly & AIS Stream"
)
