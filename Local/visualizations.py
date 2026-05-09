"""
Visualization helpers for the Maritime Anomaly Dashboard.
Uses Plotly for interactive charts and a Plotly Express scatter map.
"""

import logging
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import AppConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- #
#  Map
# ----------------------------------------------------------------------- #

def create_anomaly_map(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
) -> go.Figure:
    """
    Interactive global scatter-map with vessels colour-coded by anomaly
    severity.  Hovering a marker shows IMO, MMSI, flagged reason and
    anomaly score.
    """
    config = config or AppConfig()
    colors = config.visualization.anomaly_colors

    severity_to_color = {
        "Normal": colors["normal"],
        "Low": colors["low_anomaly"],
        "Medium": colors["medium_anomaly"],
        "High": colors["high_anomaly"],
    }

    if df.empty or "latitude" not in df.columns:
        return _empty_map()

    df = df.copy()

    # Build a clean hover text column
    hover_text = []
    for _, row in df.iterrows():
        lines = [
            "<b>{}</b>".format(row.get("name", "Unknown")),
            "IMO: {}".format(row.get("imo", "N/A") or "N/A"),
            "MMSI: {}".format(row.get("mmsi", "N/A") or "N/A"),
            "Type: {}".format(row.get("vessel_type", "Unknown")),
            "Speed: {:.1f} kn".format(row.get("sog", 0) or 0),
            "Severity: {}".format(row.get("severity", "Normal")),
            "Score: {:.2f}".format(row.get("anomaly_score", 0) or 0),
            "Flag: {}".format(row.get("anomaly_reason", "Normal")),
        ]
        hover_text.append("<br>".join(lines))

    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="severity" if "severity" in df.columns else None,
        color_discrete_map=severity_to_color,
        size="sog",
        size_max=12,
        hover_name=df["name"].fillna("Unknown"),
        hover_data={
            "imo": True,
            "mmsi": True,
            "vessel_type": True,
            "sog": ":.1f",
            "severity": True,
            "anomaly_score": ":.2f",
            "anomaly_reason": True,
        },
        center={"lat": config.visualization.map_center[0],
                "lon": config.visualization.map_center[1]},
        zoom=config.visualization.map_zoom,
        opacity=0.85,
    )

    # Override with rich hover text
    fig.update_traces(hovertext=hover_text, hoverinfo="text")

    fig.update_layout(
        mapbox_style="carto-positron",
        title="Global Vessel Anomaly Map",
        title_x=0.5,
        margin=dict(l=0, r=0, t=50, b=0),
        legend_title_text="Severity",
    )

    return fig


def _empty_map() -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No vessel data available",
        showarrow=False,
        font_size=20,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
    )
    return fig


# ----------------------------------------------------------------------- #
#  Time-series trend line
# ----------------------------------------------------------------------- #

def create_trend_chart(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
) -> go.Figure:
    """Time-series of anomaly severity counts over the observation window."""
    config = config or AppConfig()
    if df.empty or "timestamp" not in df.columns:
        return _empty_chart("Trend Chart")

    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("h")

    agg = (
        df.groupby(["hour", "severity"], observed=True)
        .size()
        .reset_index(name="count")
    )

    fig = px.line(
        agg,
        x="hour",
        y="count",
        color="severity",
        markers=True,
        title="Anomaly Severity Trend Over Time",
        labels={"hour": "Time", "count": "Vessel Count"},
    )
    fig.update_traces(line_width=2)
    fig.update_layout(xaxis_title="Time", yaxis_title="Vessel Count")
    return fig


# ----------------------------------------------------------------------- #
#  Distribution plot
# ----------------------------------------------------------------------- #

def create_distribution_chart(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
) -> go.Figure:
    """Histogram of anomaly scores with a vertical line at the high threshold."""
    config = config or AppConfig()
    if df.empty or "anomaly_score" not in df.columns:
        return _empty_chart("Distribution Chart")

    fig = px.histogram(
        df,
        x="anomaly_score",
        color="severity" if "severity" in df.columns else None,
        title="Distribution of Anomaly Scores",
        labels={"anomaly_score": "Anomaly Score"},
        nbins=50,
        opacity=0.7,
    )
    threshold = config.visualization.anomaly_thresholds[2]
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        annotation_text="High threshold ({})".format(threshold),
        annotation_position="top right",
    )
    return fig


# ----------------------------------------------------------------------- #
#  Correlation heatmap
# ----------------------------------------------------------------------- #

def create_correlation_heatmap(
    df: pd.DataFrame,
    config: Optional[AppConfig] = None,
) -> go.Figure:
    """Heatmap of correlations between numeric navigation features and anomaly score."""
    config = config or AppConfig()
    numeric_cols = [
        "sog", "cog", "rot", "heading", "heading_cog_delta",
        "rot_abs", "anomaly_score",
    ]
    available = [c for c in numeric_cols if c in df.columns]
    if len(available) < 2:
        return _empty_chart("Correlation Heatmap")

    corr = df[available].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Heatmap",
        labels=dict(color="Correlation"),
    )
    return fig


# ----------------------------------------------------------------------- #
#  Helpers
# ----------------------------------------------------------------------- #

def _empty_chart(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="{} -- no data".format(title),
        showarrow=False,
        font_size=16,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
    )
    return fig
