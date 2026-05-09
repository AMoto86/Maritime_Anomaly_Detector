"""
Configuration module for the Maritime Anomaly Detection System.
Centralizes all settings, API keys, thresholds, and model hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class WebSocketConfig:
    """WebSocket connection configuration."""
    url: str = "wss://stream.aisstream.io/v0/stream"
    api_key: str = "7816ef206a09ac6155a92ea4ce38b278721d6430"
    reconnect_delay: int = 5
    max_retries: int = 3


@dataclass
class DataCleaningConfig:
    """Configuration for data cleaning thresholds."""
    max_speed_normal: float = 30.0
    high_speed_craft_types: List[str] = field(default_factory=lambda: [
        "HIGH_SPEED_CRAFT", "PILOT_BOAT", "TENDER", "FAST_FERRY"
    ])
    min_latitude: float = -90.0
    max_latitude: float = 90.0
    min_longitude: float = -180.0
    max_longitude: float = 180.0
    min_speed: float = 0.0
    max_speed: float = 90.0
    min_cog: float = 0.0
    max_cog: float = 360.0
    min_rot: float = -50.0
    max_rot: float = 50.0


@dataclass
class IsolationForestConfig:
    """Configuration for Isolation Forest anomaly detection."""
    contamination: float = 0.05
    n_estimators: int = 100
    max_samples: str = "auto"
    random_state: int = 42
    feature_columns: List[str] = field(default_factory=lambda: [
        "sog", "cog", "rot"
    ])


@dataclass
class DBSCANConfig:
    """Configuration for DBSCAN spatial clustering."""
    eps: float = 0.05
    min_samples: int = 3
    feature_columns: List[str] = field(default_factory=lambda: [
        "latitude", "longitude"
    ])


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    map_center: Tuple[float, float] = field(default_factory=lambda: (20.0, 0.0))
    map_zoom: int = 2
    anomaly_colors: dict = field(default_factory=lambda: {
        "normal": "#2ecc71",
        "low_anomaly": "#f1c40f",
        "medium_anomaly": "#e67e22",
        "high_anomaly": "#e74c3c"
    })
    anomaly_thresholds: Tuple[float, float, float] = field(
        default_factory=lambda: (0.2, 0.5, 0.8)
    )


@dataclass
class StreamlitConfig:
    """Streamlit dashboard configuration."""
    page_title: str = "🚢 Maritime Anomaly Detection Dashboard"
    page_icon: str = "🚢"
    layout: str = "wide"
    refresh_interval_seconds: int = 30


@dataclass
class AppConfig:
    """Main application configuration aggregating all sub-configs."""
    websocket: WebSocketConfig = field(default_factory=WebSocketConfig)
    data_cleaning: DataCleaningConfig = field(default_factory=DataCleaningConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    dbscan: DBSCANConfig = field(default_factory=DBSCANConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    streamlit: StreamlitConfig = field(default_factory=StreamlitConfig)

    target_vessel_types: List[str] = field(default_factory=lambda: [
        "CARGO", "CONTAINER_SHIP", "BULK_CARRIER", "TANKER",
        "GENERAL_CARGO", "MULTIPURPOSE", "CHEMICAL_TANKER",
        "OIL_TANKER", "GAS_TANKER"
    ])

    ais_gap_threshold_minutes: int = 30
