"""
Data cleaning and transformation for AIS data.
Handles validation, filtering, imputation, and feature engineering.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config import AppConfig

logger = logging.getLogger(__name__)


class DataCleaner:
    """Cleans, validates, and engineers features from raw AIS DataFrames."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full cleaning pipeline."""
        if df.empty:
            logger.warning("Empty DataFrame -- nothing to clean.")
            return df

        initial = len(df)
        logger.info("Cleaning %d records ...", initial)

        df = self._drop_duplicates(df)
        df = self._validate_geographic_bounds(df)
        df = self._clean_numerics(df)
        df = self._filter_unrealistic_speed(df)
        df = self._impute_missing(df)
        df = self._engineer_features(df)
        df = self._parse_timestamps(df)

        logger.info("Cleaning done: %d -> %d records.", initial, len(df))
        return df

    # ------------------------------------------------------------------ #
    #  Pipeline steps
    # ------------------------------------------------------------------ #

    @staticmethod
    def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
        if {"mmsi", "timestamp"}.issubset(df.columns):
            df = df.drop_duplicates(subset=["mmsi", "timestamp"], keep="last")
        else:
            df = df.drop_duplicates(keep="last")
        return df

    def _validate_geographic_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.data_cleaning
        mask = (
            (df["latitude"] >= cfg.min_latitude)
            & (df["latitude"] <= cfg.max_latitude)
            & (df["longitude"] >= cfg.min_longitude)
            & (df["longitude"] <= cfg.max_longitude)
        )
        return df[mask]

    def _clean_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config.data_cleaning
        for col in (
            "sog", "cog", "rot", "latitude", "longitude",
            "heading", "length", "width",
        ):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "sog" in df.columns:
            df = df[(df["sog"] >= cfg.min_speed) & (df["sog"] <= cfg.max_speed)]
        if "cog" in df.columns:
            df = df[(df["cog"] >= cfg.min_cog) & (df["cog"] <= cfg.max_cog)]
        if "rot" in df.columns:
            df = df[(df["rot"] >= cfg.min_rot) & (df["rot"] <= cfg.max_rot)]
        return df

    def _filter_unrealistic_speed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove records with SOG > 30 knots unless the vessel is a
        high-speed craft type.
        """
        cfg = self.config.data_cleaning
        if "sog" not in df.columns or "vessel_type" not in df.columns:
            return df

        valid_speed = df["sog"] <= cfg.max_speed_normal
        is_high_speed = df["vessel_type"].astype(str).str.upper().isin(
            [t.upper() for t in cfg.high_speed_craft_types]
        )
        mask = valid_speed | is_high_speed
        return df[mask]

    @staticmethod
    def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
        """Fill / cap missing numeric values with sensible defaults."""
        if "sog" in df.columns:
            df["sog"] = df["sog"].fillna(0.0)
        if "cog" in df.columns:
            df["cog"] = df["cog"].fillna(df["cog"].median())
        if "rot" in df.columns:
            df["rot"] = df["rot"].fillna(0.0)
        if "heading" in df.columns:
            df["heading"] = df["heading"].fillna(df["heading"].median())
        return df

    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns useful for anomaly detection."""
        # Speed category
        if "sog" in df.columns:
            df["speed_category"] = pd.cut(
                df["sog"],
                bins=[-0.01, 0, 3, 10, 20, 100],
                labels=["Stopped", "Drifting", "Slow", "Cruising", "Fast"],
            )

        # Is the vessel moving?
        if "sog" in df.columns:
            df["is_moving"] = df["sog"] > 1.0

        # Heading vs COG divergence (erratic manoeuvring signal)
        if {"heading", "cog"}.issubset(df.columns):
            df["heading_cog_delta"] = (
                (df["heading"] - df["cog"]).abs().clip(upper=180)
            )

        # ROT magnitude
        if "rot" in df.columns:
            df["rot_abs"] = df["rot"].abs()

        return df

    @staticmethod
    def _parse_timestamps(df: pd.DataFrame) -> pd.DataFrame:
        """Convert timestamp column to datetime, handling errors."""
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            # Drop rows where timestamp could not be parsed
            df = df.dropna(subset=["timestamp"])
        return df
