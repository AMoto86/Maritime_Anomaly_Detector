"""
WebSocket connection handler for AIS data streaming.
Manages connection, subscription handshake, message parsing, and buffering.

Protocol (per aisstream.io documentation):
    1. Connect to wss://stream.aisstream.io/v0/stream (no key in URL)
    2. Send JSON subscription message within 3 seconds:
       - "APIKey" (capital K) for authentication
       - "BoundingBoxes" as array of corner pairs: [[[lat1,lon1],[lat2,lon2]], ...]
    3. Receive AIS messages with nested structure: Message[Type] + MetaData
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import websockets

from config import AppConfig

logger = logging.getLogger(__name__)


class AISDataStream:
    """Handles WebSocket connection to the AIS stream API."""

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.data_buffer: List[Dict[str, Any]] = []
        self.is_connected: bool = False
        self._connection_start_time: Optional[float] = None
        self._message_count: int = 0

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    async def connect(self, max_messages: int = 1000) -> List[Dict[str, Any]]:
        """
        Open a WebSocket connection and collect AIS messages.

        Args:
            max_messages: Stop after this many valid messages.

        Returns:
            List of parsed AIS message dicts.
        """
        retries = 0
        while retries < self.config.websocket.max_retries:
            try:
                # Connect WITHOUT key in URL (per docs)
                url = self.config.websocket.url.rstrip("/")
                logger.info("Connecting to AIS stream: %s", url)

                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=20
                ) as ws:
                    self.is_connected = True
                    self._connection_start_time = time.time()
                    logger.info("Connected. Sending subscription within 3s...")

                    # Send subscription message (EXACT format from docs)
                    # Note: "APIKey" with capital K, bounding boxes as array of pairs
                    subscription = {
                        "APIKey": self.config.websocket.api_key,
                        "BoundingBoxes": [[[-90.0, -180.0], [90.0, 180.0]]],
                        "FiltersShipMMSI": [],
                        "FilterMessageTypes": ["PositionReport"],
                    }

                    await ws.send(json.dumps(subscription))
                    logger.info("Subscription sent.")

                    messages: List[Dict[str, Any]] = []
                    try:
                        async for raw in ws:
                            if max_messages and len(messages) >= max_messages:
                                break
                            parsed = self._parse_ais_message(raw)
                            if parsed:
                                messages.append(parsed)
                                self._message_count += 1
                    except asyncio.TimeoutError:
                        logger.warning("Connection timed out.")
                    except websockets.exceptions.ConnectionClosed as exc:
                        logger.warning("Connection closed: %s", exc)

                    self.data_buffer.extend(messages)
                    return messages

            except (
                websockets.exceptions.InvalidURI,
                websockets.exceptions.InvalidHandshake,
                ConnectionRefusedError,
            ) as exc:
                retries += 1
                logger.error(
                    "Connection attempt %d failed: %s", retries, exc
                )
                if retries < self.config.websocket.max_retries:
                    wait = self.config.websocket.reconnect_delay * retries
                    logger.info("Retrying in %d s ...", wait)
                    await asyncio.sleep(wait)
                else:
                    raise ConnectionError(
                        "Failed to connect after max retries."
                    ) from exc
            except Exception as exc:
                logger.error("Unexpected error: %s", exc)
                raise

        return self.data_buffer

    def to_dataframe(
        self, messages: Optional[List[Dict[str, Any]]] = None
    ) -> pd.DataFrame:
        """Convert message dicts (or the internal buffer) to a DataFrame."""
        source = messages if messages is not None else self.data_buffer
        if not source:
            return pd.DataFrame()
        df = pd.DataFrame(source)
        if "raw_data" in df.columns:
            df.drop(columns=["raw_data"], inplace=True)
        return df

    # ------------------------------------------------------------------ #
    #  Parsing helpers (handles nested AISStream message format)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_ais_message(raw: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single raw WebSocket message into a normalised dict.

        AISStream messages have this structure:
        {
            "Message": {"PositionReport": {...}},
            "MessageType": "PositionReport",
            "MetaData": {
                "MMSI": 123456789,
                "ShipName": "VESSEL",
                "latitude": 12.34,
                "longitude": -56.78,
                "time_utc": "2024-..."
            }
        }
        """
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        # Extract from MetaData (primary source for position/identity)
        meta = data.get("MetaData", {})
        msg_body = {}

        # Get the actual AIS message body (nested under MessageType)
        msg_type = data.get("MessageType", "")
        if msg_type and "Message" in data:
            msg_body = data["Message"].get(msg_type, {})

        parsed = {
            "imo": AISDataStream._extract(
                msg_body, ("ImoNumber", "IMO")
            ) or AISDataStream._extract(meta, ("imo",)),
            "mmsi": str(
                meta.get("MMSI") or msg_body.get("UserID", "")
            ) if meta.get("MMSI") or msg_body.get("UserID") else None,
            "callsign": AISDataStream._extract(
                msg_body, ("CallSign",)
            ),
            "name": meta.get("ShipName") or AISDataStream._extract(
                msg_body, ("Name",)
            ),
            "latitude": AISDataStream._to_float(
                meta.get("latitude") or msg_body.get("Latitude")
            ),
            "longitude": AISDataStream._to_float(
                meta.get("longitude") or msg_body.get("Longitude")
            ),
            "sog": AISDataStream._to_float(
                msg_body.get("Sog") or meta.get("sog")
            ),
            "cog": AISDataStream._to_float(
                msg_body.get("Cog") or meta.get("cog")
            ),
            "heading": AISDataStream._to_float(
                msg_body.get("TrueHeading") or meta.get("heading")
            ),
            "rot": AISDataStream._to_float(
                msg_body.get("RateOfTurn") or meta.get("rot")
            ),
            "vessel_type": AISDataStream._extract(
                msg_body, ("Type",)
            ),
            "length": AISDataStream._to_float(
                msg_body.get("Dimension", {}).get("A")
            ),
            "width": AISDataStream._to_float(
                msg_body.get("Dimension", {}).get("B")
            ),
            "status": AISDataStream._extract(
                msg_body, ("NavigationalStatus",)
            ),
            "destination": AISDataStream._extract(
                msg_body, ("Destination",)
            ),
            "timestamp": meta.get("time_utc") or datetime.now().isoformat(),
        }

        # Require at least lat/lon to be valid
        if parsed["latitude"] is None or parsed["longitude"] is None:
            return None

        # Validate coordinate ranges
        if not (-90 <= parsed["latitude"] <= 90):
            return None
        if not (-180 <= parsed["longitude"] <= 180):
            return None

        return parsed

    @staticmethod
    def _extract(data: dict, keys: tuple, default=None):
        for k in keys:
            if k in data and data[k] is not None:
                return data[k]
        return default

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        """Return connection / buffer statistics."""
        return {
            "is_connected": self.is_connected,
            "message_count": self._message_count,
            "buffer_size": len(self.data_buffer),
            "uptime_seconds": (
                round(time.time() - self._connection_start_time, 1)
                if self._connection_start_time
                else 0
            ),
        }
