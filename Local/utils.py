"""
Utility helpers shared across modules.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str) -> pd.DataFrame:
    """Merge two DataFrames, logging a warning if keys are missing."""
    if on not in left.columns:
        logger.warning("Merge key %r not in left DataFrame", on)
        return left
    if on not in right.columns:
        logger.warning("Merge key %r not in right DataFrame", on)
        return left
    return left.merge(right, on=on, how="left")


def highlight_anomalies(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Return the top-N most anomalous vessels sorted by score."""
    if "anomaly_score" not in df.columns:
        return pd.DataFrame()
    return (
        df.nlargest(top_n, "anomaly_score")[
            ["name", "imo", "mmsi", "vessel_type", "latitude", "longitude",
             "sog", "anomaly_score", "severity", "anomaly_reason"]
        ]
        .reset_index(drop=True)
    )


def format_mmsi(value) -> str:
    """Format an MMSI as a zero-padded 9-digit string."""
    try:
        return f"{int(value):09d}"
    except (ValueError, TypeError):
        return str(value) if value is not None else "N/A"
