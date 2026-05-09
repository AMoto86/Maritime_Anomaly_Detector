"""
Generates realistic sample AIS data for demonstration when the live
WebSocket feed is unavailable.

Produces a mix of normal vessels and seeded anomalies so every ML
detector has something to find.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


VESSEL_TYPES = [
    "CARGO", "CONTAINER_SHIP", "BULK_CARRIER", "TANKER",
    "GENERAL_CARGO", "MULTIPURPOSE", "CHEMICAL_TANKER",
]

# Major shipping lanes / choke points for realistic clustering
SHIPPING_LANES = [
    (26.8, 56.0),   # Strait of Hormuz
    (31.2, 32.5),   # Suez Canal approach
    (1.3, 104.0),   # Singapore Strait
    (9.2, -79.8),   # Panama Canal approach
    (50.8, 1.3),    # English Channel
    (54.5, 2.0),    # North Sea
    (5.0, 3.0),     # Gulf of Guinea
    (12.0, 113.0),  # South China Sea
    (38.5, 20.0),   # Mediterranean
    (15.0, -40.0),  # Atlantic mid-ocean
]


def generate_sample_data(n_vessels: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic AIS snapshot with seeded anomalies.

    Parameters
    ----------
    n_vessels : int
        Number of vessel records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns matching the live-stream schema.
    """
    rng = np.random.default_rng(seed)
    now = datetime.utcnow()

    records: list[dict] = []

    for i in range(n_vessels):
        lane_lat, lane_lon = rng.choice(SHIPPING_LANES)

        # Base position with scatter
        lat = round(lane_lat + rng.normal(0, 0.3), 6)
        lon = round(lane_lon + rng.normal(0, 0.5), 6)

        # Normal vessel attributes
        imo = str(rng.integers(9000000, 9999999))
        mmsi = str(rng.integers(200000000, 999999999))
        name = (
            f"MV {rng.choice(['PACIFIC', 'ATLANTIC', 'NORDIC', 'ASIA', "
            f"'GLOBAL', 'OCEAN', 'STAR', 'HORIZON'])} {rng.integers(1, 99)}"
        )
        vessel_type = rng.choice(VESSEL_TYPES)
        length = round(rng.uniform(80, 350), 1)
        width = round(length * rng.uniform(0.1, 0.15), 1)
        destination = rng.choice([
            "SINGAPORE", "ROTTERDAM", "SHANGHAI", "HOUSTON",
            "FUJAIRAH", "BUSAN", "HAMBURG",
        ])

        # Timestamp within last 2 hours
        ts = now - timedelta(minutes=int(rng.integers(0, 120)))

        # --- Decide anomaly type (15% of vessels get an anomaly) ---
        anomaly_roll = rng.random()

        if anomaly_roll < 0.05:
            # SPEED ANOMALY: cargo ship moving way too fast
            sog = round(rng.uniform(25, 38), 1)
            cog = round(rng.uniform(0, 360), 1)
            rot = round(rng.normal(0, 5), 2)
            heading = round(cog + rng.normal(0, 10), 1)

        elif anomaly_roll < 0.10:
            # STS CLUSTER: group of vessels very close together, nearly stopped
            sog = round(rng.uniform(0, 2), 1)
            cog = round(rng.uniform(0, 360), 1)
            rot = round(rng.normal(0, 1), 2)
            heading = round(cog + rng.normal(0, 5), 1)
            # Bunch them together tightly
            lat = round(lane_lat + rng.normal(0, 0.01), 6)
            lon = round(lane_lon + rng.normal(0, 0.01), 6)

        elif anomaly_roll < 0.15:
            # AIS GAP: simulate a large timestamp gap
            sog = round(rng.uniform(5, 18), 1)
            cog = round(rng.uniform(0, 360), 1)
            rot = round(rng.normal(0, 2), 2)
            heading = round(cog + rng.normal(0, 5), 1)
            # Push timestamp far back to trigger gap detection
            ts = now - timedelta(minutes=int(rng.integers(180, 480)))

        else:
            # NORMAL vessel
            sog = round(rng.uniform(0, 20), 1)
            cog = round(rng.uniform(0, 360), 1)
            rot = round(rng.normal(0, 2), 2)
            heading = round(cog + rng.normal(0, 5), 1)

        status = rng.choice(["Under way using engine", "At anchor", "Moored"])

        records.append({
            "imo": imo,
            "mmsi": mmsi,
            "name": name,
            "vessel_type": vessel_type,
            "latitude": lat,
            "longitude": lon,
            "sog": sog,
            "cog": cog,
            "heading": heading,
            "rot": rot,
            "length": length,
            "width": width,
            "status": status,
            "destination": destination,
            "timestamp": ts.isoformat(),
        })

    df = pd.DataFrame(records)
    return df


def generate_sample_data_with_gaps(n_vessels: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate sample data where some MMSIs have multiple timestamps,
    allowing the AIS gap detector to find real gaps.
    """
    rng = np.random.default_rng(seed)
    now = datetime.utcnow()

    records: list[dict] = []
    unique_mmsis = [
        str(rng.integers(200000000, 999999999))
        for _ in range(n_vessels // 3)
    ]

    for mmsi in unique_mmsis:
        imo = str(rng.integers(9000000, 9999999))
        name = (
            f"MV {rng.choice(['PACIFIC', 'ATLANTIC', 'NORDIC'])} "
            f"{rng.integers(1, 99)}"
        )
        vessel_type = rng.choice(VESSEL_TYPES)
        lane_lat, lane_lon = rng.choice(SHIPPING_LANES)

        n_pings = int(rng.integers(3, 6))
        has_gap = rng.random() < 0.2

        for j in range(n_pings):
            if has_gap and j == n_pings - 1:
                ts = now - timedelta(minutes=int(rng.integers(180, 480)))
            else:
                ts = now - timedelta(minutes=int(rng.integers(0, 60)))

            records.append({
                "imo": imo,
                "mmsi": mmsi,
                "name": name,
                "vessel_type": vessel_type,
                "latitude": round(lane_lat + rng.normal(0, 0.1), 6),
                "longitude": round(lane_lon + rng.normal(0, 0.15), 6),
                "sog": round(rng.uniform(0, 20), 1),
                "cog": round(rng.uniform(0, 360), 1),
                "heading": round(rng.uniform(0, 360), 1),
                "rot": round(rng.normal(0, 2), 2),
                "length": round(rng.uniform(80, 350), 1),
                "width": round(rng.uniform(10, 45), 1),
                "status": rng.choice(["Under way using engine", "At anchor"]),
                "destination": rng.choice(["SINGAPORE", "ROTTERDAM", "SHANGHAI"]),
                "timestamp": ts.isoformat(),
            })

    return pd.DataFrame(records)
