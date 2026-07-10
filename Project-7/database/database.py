"""
SQLite database access layer.

Provides a thin, dependency-free wrapper around :mod:`sqlite3` for the four
persisted concerns required by the project brief:

    * User queries
    * Generated itineraries
    * Flight selections
    * Hotel selections
    * Tool/search logs

The module exposes a single :class:`TravelDatabase` class. All SQL lives in
``database/schema.sql`` and is applied idempotently on first use.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from configs import settings
from utils.logger import get_logger

logger = get_logger("database")


class TravelDatabase:
    """SQLite-backed persistence layer for the Travel AI Assistant.

    Args:
        db_path: Path to the SQLite database file. Defaults to
            ``settings.DATABASE_PATH``.
        schema_path: Path to the SQL schema file. Defaults to
            ``settings.SCHEMA_PATH``.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = settings.DATABASE_PATH,
        schema_path: Union[str, Path] = settings.SCHEMA_PATH,
    ) -> None:
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON;")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise_schema(self) -> None:
        if not self.schema_path.exists():
            logger.warning("Schema file not found at {}; skipping init.", self.schema_path)
            return
        sql_script = self.schema_path.read_text(encoding="utf-8")
        with self._connect() as conn:
            conn.executescript(sql_script)
        logger.info("Database schema ensured at {}", self.db_path)

    # ------------------------------------------------------------------
    # user_queries
    # ------------------------------------------------------------------
    def insert_user_query(
        self,
        session_id: str,
        source_city: str,
        destination_city: str,
        start_date: str,
        num_days: int,
        budget: float,
        travel_style: str,
        num_travellers: int = 1,
        raw_query: Optional[str] = None,
    ) -> int:
        """Insert a user query and return its new ``query_id``."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO user_queries (
                    session_id, raw_query, source_city, destination_city,
                    start_date, num_days, budget, travel_style, num_travellers
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    raw_query,
                    source_city,
                    destination_city,
                    start_date,
                    num_days,
                    budget,
                    travel_style,
                    num_travellers,
                ),
            )
            query_id = int(cursor.lastrowid)
        logger.info("Inserted user_query id={} session={}", query_id, session_id)
        return query_id

    def get_user_queries(self, session_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent user queries, optionally filtered by session."""
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM user_queries WHERE session_id = ? "
                    "ORDER BY query_id DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM user_queries ORDER BY query_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # itineraries
    # ------------------------------------------------------------------
    def insert_itinerary(
        self,
        query_id: int,
        trip_title: str,
        itinerary: Union[dict, str],
        total_cost: Optional[float] = None,
        budget_category: Optional[str] = None,
    ) -> int:
        """Persist a generated itinerary as JSON and return its ``itinerary_id``."""
        itinerary_json = itinerary if isinstance(itinerary, str) else json.dumps(itinerary, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO itineraries (
                    query_id, trip_title, itinerary_json, total_cost, budget_category
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (query_id, trip_title, itinerary_json, total_cost, budget_category),
            )
            itinerary_id = int(cursor.lastrowid)
        logger.info("Inserted itinerary id={} for query_id={}", itinerary_id, query_id)
        return itinerary_id

    def get_itineraries(self, query_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Return generated itineraries, optionally filtered by query."""
        with self._connect() as conn:
            if query_id is not None:
                rows = conn.execute(
                    "SELECT * FROM itineraries WHERE query_id = ? ORDER BY itinerary_id DESC LIMIT ?",
                    (query_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM itineraries ORDER BY itinerary_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        results = []
        for row in rows:
            record = dict(row)
            try:
                record["itinerary_json"] = json.loads(record["itinerary_json"])
            except (TypeError, json.JSONDecodeError):
                pass
            results.append(record)
        return results

    # ------------------------------------------------------------------
    # flight_selections
    # ------------------------------------------------------------------
    def insert_flight_selection(
        self,
        query_id: int,
        flight_id: str,
        airline: str,
        source_city: str,
        destination_city: str,
        price: float,
        duration_minutes: int,
        selection_reason: Optional[str] = None,
    ) -> int:
        """Persist the flight chosen for a query and return the row id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO flight_selections (
                    query_id, flight_id, airline, source_city, destination_city,
                    price, duration_minutes, selection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    flight_id,
                    airline,
                    source_city,
                    destination_city,
                    price,
                    duration_minutes,
                    selection_reason,
                ),
            )
            return int(cursor.lastrowid)

    # ------------------------------------------------------------------
    # hotel_selections
    # ------------------------------------------------------------------
    def insert_hotel_selection(
        self,
        query_id: int,
        hotel_id: str,
        hotel_name: str,
        city: str,
        stars: int,
        price_per_night: float,
        selection_reason: Optional[str] = None,
    ) -> int:
        """Persist the hotel chosen for a query and return the row id."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO hotel_selections (
                    query_id, hotel_id, hotel_name, city, stars,
                    price_per_night, selection_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (query_id, hotel_id, hotel_name, city, stars, price_per_night, selection_reason),
            )
            return int(cursor.lastrowid)

    # ------------------------------------------------------------------
    # search_logs
    # ------------------------------------------------------------------
    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        input_payload: Any,
        output_status: str,
        duration_ms: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> int:
        """Insert a row into ``search_logs`` for observability."""
        payload_text = input_payload if isinstance(input_payload, str) else json.dumps(
            input_payload, default=str, ensure_ascii=False
        )
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO search_logs (
                    session_id, tool_name, input_payload, output_status,
                    duration_ms, error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, tool_name, payload_text, output_status, duration_ms, error_message),
            )
            return int(cursor.lastrowid)

    def get_search_logs(self, session_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Return recent tool invocation logs."""
        with self._connect() as conn:
            if session_id:
                rows = conn.execute(
                    "SELECT * FROM search_logs WHERE session_id = ? ORDER BY log_id DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM search_logs ORDER BY log_id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]


_default_db: Optional[TravelDatabase] = None


def get_database() -> TravelDatabase:
    """Return a process-wide singleton :class:`TravelDatabase` instance."""
    global _default_db
    if _default_db is None:
        _default_db = TravelDatabase()
    return _default_db
