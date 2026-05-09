"""
WebSocket connection handler for AIS data streaming.
Manages connection, reconnection, message parsing, and buffering.
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
                url = (
                    f"{self.config.websocket.url}"
                    f"?api_key={self.config.websocket.api_key}"
                )
                logger.info("Connecting to AIS stream ...")

                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=20
                ) as ws:
                    self.is_connected = True
                    self._connection_start_time = time.time()
                    logger.info("Connected successfully.")

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
                logger.error("Connection attempt %d failed: %s", retries, exc)
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
    #  Parsing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_ais_message(raw: Any) -> Optional[Dict[str, Any]]:
        """Parse a single raw WebSocket message into a normalised dict."""
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        parsed = {
            "imo": AISDataStream._extract(data, ("imo", "IMO")),
            "mmsi": AISDataStream._extract(data, ("mmsi", "MMSI")),
            "callsign": AISDataStream._extract(data, ("callsign", "callSign")),
            "name": AISDataStream._extract(data, ("name", "shipname", "vesselName")),
            "latitude": AISDataStream._to_float(
                AISDataStream._extract(data, ("lat", "latitude", "LAT"))
            ),
            "longitude": AISDataStream._to_float(
                AISDataStream._extract(data, ("lon", "longitude", "LON"))
            ),
            "sog": AISDataStream._to_float(
                AISDataStream._extract(data, ("sog", "speed", "SOG"))
            ),
            "cog": AISDataStream._to_float(
                AISDataStream._extract(data, ("cog", "course", "COG"))
            ),
            "heading": AISDataStream._to_float(
                AISDataStream._extract(data, ("heading", "hdg", "HEAD"))
            ),
            "rot": AISDataStream._to_float(
                AISDataStream._extract(data, ("rot", "rateOfTurn", "ROT"))
            ),
            "vessel_type": AISDataStream._extract(
                data, ("type", "vesselType", "shipType", "vessel_type")
            ),
            "length": AISDataStream._to_float(
                AISDataStream._extract(data, ("length", "loa", "lengthOverall"))
            ),
            "width": AISDataStream._to_float(
                AISDataStream._extract(data, ("width", "beam", "widthOverall"))
            ),
            "status": AISDataStream._extract(
                data, ("status", "navStatus", "navigationStatus")
            ),
            "destination": AISDataStream._extract(data, ("dest", "destination")),
            "timestamp": AISDataStream._get_timestamp(data),
        }

        # Require at least lat/lon
        if parsed["latitude"] is None or parsed["longitude"] is None:
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

    @staticmethod
    def _get_timestamp(data: dict) -> str:
        for k in ("timestamp", "time", "ts", "receivedTime", "t"):
            if k in data and data[k]:
                try:
                    if isinstance(data[k], (int, float)):
                        return datetime.fromtimestamp(data[k]).isoformat()
                    return str(data[k])
                except (ValueError, OSError):
                    continue
        return datetime.now().isoformat()

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
