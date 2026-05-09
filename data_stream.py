"""
WebSocket connection handler for AIS data streaming.
Manages connection, subscription handshake, message parsing, and buffering.

Protocol (per aisstream.io documentation):
    1. Connect to wss://stream.aisstream.io/v0/stream (no key in URL)
    2. Send JSON subscription message WITHIN 3 SECONDS:
       - "APIKey" (capital K) is required
       - BoundingBoxes format: [[[lat1, lon1], [lat2, lon2]], ...]
    3. Receive AIS messages as nested JSON:
       - Position in Message.PositionReport.Latitude/Longitude
       - Metadata in MetaData (MMSI, ShipName, time_utc)
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
                url = self.config.websocket.url.rstrip("/")
                logger.info("Connecting to AIS stream: %s", url)

                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=20
                ) as ws:
                    self.is_connected = True
                    self._connection_start_time = time.time()

                    # Send subscription IMMEDIATELY (must be within 3 seconds)
                    subscription = self._build_subscription()
                    await ws.send(json.dumps(subscription))
                    logger.info("Subscription sent: %s", json.dumps(subscription))

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

    def _build_subscription(self) -> Dict[str, Any]:
        """Build the subscription message per aisstream.io docs."""
        # Global bounding box: [[[lat_min, lon_min], [lat_max, lon_max]]]
        # Note the triple nesting: list of boxes, each box has 2 corners
        subscription = {
            "APIKey": self.config.websocket.api_key,
            "BoundingBoxes": [[[-90.0, -180.0], [90.0, 180.0]]],
        }
        return subscription

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
    #  Parsing helpers (updated for nested AIS message structure)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_ais_message(raw: Any) -> Optional[Dict[str, Any]]:
        """
        Parse a single raw WebSocket message into a normalised dict.

        AISStream messages have this structure:
        {
            "Message": {"PositionReport": {"Latitude": ..., "Longitude": ...}},
            "MessageType": "PositionReport",
            "MetaData": {"MMSI": ..., "ShipName": ..., "latitude": ..., ...}
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

        # Extract nested message and metadata
        message = data.get("Message", {})
        meta = data.get("MetaData", {})
        msg_type = data.get("MessageType", "")

        # Get position report fields (or other message type)
        pos_report = message.get("PositionReport", {})

        parsed = {
            "imo": AISDataStream._extract(meta, ("IMO", "ImoNumber")) or \
                   AISDataStream._extract(pos_report, ("ImoNumber",)),
            "mmsi": str(AISDataStream._extract(
                meta, ("MMSI",)
            ) or AISDataStream._extract(pos_report, ("UserID",))) \
                    if AISDataStream._extract(meta, ("MMSI",)) or \
                       AISDataStream._extract(pos_report, ("UserID",)) else None,
            "callsign": AISDataStream._extract(meta, ("CallSign",)),
            "name": AISDataStream._extract(
                meta, ("ShipName", "Name")
            ),
            # Latitude/Longitude from both MetaData and PositionReport
            "latitude": AISDataStream._to_float(
                AISDataStream._extract(meta, ("latitude",)) or \
                AISDataStream._to_float(AISDataStream._extract(
                    pos_report, ("Latitude",)
                ))
            ),
            "longitude": AISDataStream._to_float(
                AISDataStream._extract(meta, ("longitude",)) or \
                AISDataStream._to_float(AISDataStream._extract(
                    pos_report, ("Longitude",)
                ))
            ),
            "sog": AISDataStream._to_float(
                AISDataStream._extract(pos_report, ("Sog",)) or \
                AISDataStream._extract(meta, ("sog", "speed"))
            ),
            "cog": AISDataStream._to_float(
                AISDataStream._extract(pos_report, ("Cog",)) or \
                AISDataStream._extract(meta, ("cog", "course"))
            ),
            "heading": AISDataStream._to_float(
                AISDataStream._extract(pos_report, ("TrueHeading",)) or \
                AISDataStream._extract(meta, ("heading", "hdg"))
            ),
            "rot": AISDataStream._to_float(
                AISDataStream._extract(pos_report, ("RateOfTurn",)) or \
                AISDataStream._extract(meta, ("rot", "rateOfTurn"))
            ),
            "vessel_type": AISDataStream._extract(
                meta, ("ShipType", "type", "vesselType")
            ),
            "length": AISDataStream._to_float(
                AISDataStream._extract(meta, ("Length",)) or \
                AISDataStream._to_float(AISDataStream._extract(
                    pos_report, ("Dimension", {}).get("A") if isinstance(
                        AISDataStream._extract(pos_report, ("Dimension",)), dict
                    ) else None
                ))
            ),
            "width": AISDataStream._to_float(
                AISDataStream._extract(meta, ("Width",)) or \
                AISDataStream._to_float(AISDataStream._extract(
                    pos_report, ("Dimension", {}).get("B") if isinstance(
                        AISDataStream._extract(pos_report, ("Dimension",)), dict
                    ) else None
                ))
            ),
            "status": AISDataStream._extract(
                meta, ("NavigationalStatus",)
            ) or AISDataStream._to_float(AISDataStream._extract(
                pos_report, ("NavigationalStatus",)
            )),
            "destination": AISDataStream._extract(meta, ("Destination",)),
            "timestamp": meta.get("time_utc") or datetime.now().isoformat(),
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
