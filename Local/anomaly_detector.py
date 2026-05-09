"""
Machine-learning anomaly detection for maritime AIS data.

Models
------
1. Isolation Forest  – speed / heading / turn-rate anomalies
2. DBSCAN           – spatial clustering (loitering, STS proximity)
3. AIS-gap detector – vessels that stop transmitting
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

from config import AppConfig

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------- #
#  Isolation Forest  – behavioural anomalies
# ----------------------------------------------------------------------- #

class BehaviourAnomalyDetector:
    """
    Detects vessels moving at unusual speeds or exhibiting erratic
    heading / turn-rate behaviour using an Isolation Forest.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        cfg = self.config.isolation_forest
        self.model = IsolationForest(
            n_estimators=cfg.n_estimators,
            max_samples=cfg.max_samples,
            contamination=cfg.contamination,
            random_state=cfg.random_state,
        )
        self.scaler = StandardScaler()
        self.feature_cols = cfg.feature_columns
        self._fitted = False

    # -- public ---------------------------------------------------------- #

    def fit(self, df: pd.DataFrame) -> "BehaviourAnomalyDetector":
        """Fit the model on a representative sample of vessel data."""
        X = self._prepare_features(df)
        if X is None or X.shape[0] < 10:
            logger.warning(
                "Not enough data to fit Isolation Forest (%d rows).", 
                X.shape[0] if X is not None else 0,
            )
            return self
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled)
        self._fitted = True
        logger.info("Isolation Forest fitted on %d samples.", X.shape[0])
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add columns ``if_label`` (-1 anomaly, 1 normal) and
        ``if_anomaly_score`` (lower = more anomalous) to the DataFrame.
        """
        if not self._fitted:
            logger.warning("Isolation Forest not fitted yet; returning df unchanged.")
            return df

        X = self._prepare_features(df)
        if X is None:
            return df

        X_scaled = self.scaler.transform(X)
        df = df.copy()
        df["if_label"] = self.model.predict(X_scaled)
        df["if_anomaly_score"] = -self.model.score_samples(X_scaled)
        # Higher score -> more anomalous (negate so that high = bad)
        return df

    # -- internals ------------------------------------------------------- #

    def _prepare_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract and scale feature columns, returning None on failure."""
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning("Missing IF feature columns: %s", missing)
            return None
        X = df[self.feature_cols].values.astype(float)
        # Replace any NaN with 0 (should already be imputed, but belt-and-suspenders)
        X = np.nan_to_num(X, nan=0.0)
        return X


# ----------------------------------------------------------------------- #
#  DBSCAN  – spatial clustering
# ----------------------------------------------------------------------- #

class SpatialClusterDetector:
    """
    Identifies vessels loitering in restricted zones or forming unusual
    clusters (potential dark-fleet / STS activity).
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        cfg = self.config.dbscan
        self.eps = cfg.eps
        self.min_samples = cfg.min_samples
        self.feature_cols = cfg.feature_columns
        self.labels_: Optional[np.ndarray] = None

    # -- public ---------------------------------------------------------- #

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run DBSCAN on lat/lon and annotate each row with
        ``dbscan_cluster`` (-1 = noise / outlier, >= 0 = cluster id).
        """
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            logger.warning("Missing DBSCAN feature columns: %s", missing)
            df["dbscan_cluster"] = -1
            df["cluster_size"] = 0
            return df

        coords = df[self.feature_cols].values.astype(float)
        coords = np.nan_to_num(coords, nan=0.0)

        clustering = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            algorithm="ball_tree",
            metric="haversine",
        )
        # DBSCAN haversine expects radians
        lat_rad = np.radians(coords[:, 0])
        lon_rad = np.radians(coords[:, 1])
        coords_rad = np.column_stack([lat_rad, lon_rad])

        self.labels_ = clustering.fit_predict(coords_rad)

        df = df.copy()
        df["dbscan_cluster"] = self.labels_

        # Compute cluster sizes
        cluster_counts = pd.Series(self.labels_).value_counts()
        df["cluster_size"] = df["dbscan_cluster"].map(cluster_counts).fillna(0).astype(int)

        # Flag vessels in suspiciously large clusters (potential STS)
        median_size = cluster_counts.median() if len(cluster_counts) > 0 else 0
        df["suspicious_cluster"] = (
            (df["dbscan_cluster"] >= 0)
            & (df["cluster_size"] > max(median_size * 2, 5))
        )

        logger.info(
            "DBSCAN found %d clusters (%d noise points).",
            len(cluster_counts),
            int(cluster_counts.get(-1, 0)),
        )
        return df

    # -- convenience ----------------------------------------------------- #

    def get_cluster_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a summary DataFrame of detected clusters."""
        if "dbscan_cluster" not in df.columns:
            return pd.DataFrame()
        summary = (
            df.groupby("dbscan_cluster")
            .agg(
                size=("mmsi", "nunique"),
                mean_lat=("latitude", "mean"),
                mean_lon=("longitude", "mean"),
                mean_sog=("sog", "mean"),
            )
            .reset_index()
        )
        summary = summary[summary["dbscan_cluster"] >= 0]  # exclude noise
        return summary


# ----------------------------------------------------------------------- #
#  AIS Gap Detector  – transmission silence
# ----------------------------------------------------------------------- #

class AISGapDetector:
    """
    Detects periods where a vessel stops transmitting AIS, which may
    indicate intentional AIS manipulation.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.threshold_min = self.config.ais_gap_threshold_minutes

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each MMSI, compute the maximum gap between consecutive
        messages.  Flag vessels whose gap exceeds the threshold.
        """
        if {"mmsi", "timestamp"}.issubset(df.columns) is False:
            df["max_gap_minutes"] = 0.0
            df["ais_gap_flagged"] = False
            return df

        df = df.copy()
        df = df.sort_values(["mmsi", "timestamp"])

        gaps = (
            df.groupby("mmsi")["timestamp"]
            .diff()
            .dt.total_seconds()
            .div(60)
            .groupby(df["mmsi"])
            .max()
            .rename("max_gap_minutes")
        )

        df = df.merge(gaps, on="mmsi", how="left")
        df["max_gap_minutes"] = df["max_gap_minutes"].fillna(0.0)
        df["ais_gap_flagged"] = df["max_gap_minutes"] >= self.threshold_min

        flagged = int(df["ais_gap_flagged"].sum())
        logger.info("AIS-gap detector flagged %d vessels.", flagged)
        return df


# ----------------------------------------------------------------------- #
#  Combined Anomaly Scorer
# ----------------------------------------------------------------------- #

class AnomalyScorer:
    """
    Combines signals from all detectors into a single anomaly score
    per vessel and a human-readable flag reason.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add ``anomaly_score`` (0-1) and ``anomaly_reason`` columns.
        """
        df = df.copy()
        df["anomaly_score"] = 0.0
        reasons: List[str] = []

        # --- IF contribution ---
        if "if_anomaly_score" in df.columns:
            if_col = df["if_anomaly_score"]
            if_col_norm = self._min_max_norm(if_col)
            df["anomaly_score"] += if_col_norm * 0.40
            mask = if_col_norm > self.config.visualization.anomaly_thresholds[1]
            reasons.extend(
                ["Speed/Heading Anomaly"] * mask.sum()
                + [""] * (~mask).sum()
            )

        # --- DBSCAN contribution ---
        if "suspicious_cluster" in df.columns:
            cluster_signal = df["suspicious_cluster"].astype(float)
            df["anomaly_score"] += cluster_signal * 0.35
            reasons_b = (
                ["Suspicious Cluster (STS?)"] if s else ""
                for s in df["suspicious_cluster"]
            )

        # --- AIS gap contribution ---
        if "ais_gap_flagged" in df.columns:
            gap_signal = df["ais_gap_flagged"].astype(float)
            df["anomaly_score"] += gap_signal * 0.25
            reasons_c = (
                ["AIS Transmission Gap"] if g else ""
                for g in df["ais_gap_flagged"]
            )

        # Build reason strings
        if "if_anomaly_score" in df.columns:
            reason_a = [
                "Speed/Heading Anomaly" if v > self.config.visualization.anomaly_thresholds[1]
                else ""
                for v in self._min_max_norm(df["if_anomaly_score"])
            ]
        else:
            reason_a = [""] * len(df)

        if "suspicious_cluster" in df.columns:
            reason_b = [
                "Suspicious Cluster (STS?)" if s else ""
                for s in df["suspicious_cluster"]
            ]
        else:
            reason_b = [""] * len(df)

        if "ais_gap_flagged" in df.columns:
            reason_c = [
                "AIS Transmission Gap" if g else ""
                for g in df["ais_gap_flagged"]
            ]
        else:
            reason_c = [""] * len(df)

        df["anomaly_reason"] = [
            ", ".join(filter(None, r)) or "Normal"
            for r in zip(reason_a, reason_b, reason_c)
        ]

        # Clip to [0, 1]
        df["anomaly_score"] = df["anomaly_score"].clip(0, 1)

        # Severity bucket
        t = self.config.visualization.anomaly_thresholds
        df["severity"] = pd.cut(
            df["anomaly_score"],
            bins=[-0.01, t[0], t[1], t[2], 1.01],
            labels=["Normal", "Low", "Medium", "High"],
        )

        return df

    @staticmethod
    def _min_max_norm(series: pd.Series) -> pd.Series:
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series(0.0, index=series.index)
        return (series - mn) / (mx - mn)
