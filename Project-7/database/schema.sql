-- ===========================================================================
-- Travel AI Assistant -- SQLite Schema
--
-- Normalised tables for persisting user queries, generated itineraries,
-- flight/hotel selections, and tool search logs. Designed for a
-- single-tenant SQLite deployment; foreign keys cascade on delete so
-- removing a query also removes its dependent itinerary/selection rows.
-- ===========================================================================

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- user_queries: every trip-planning request submitted to the agent.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS user_queries (
    query_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    raw_query       TEXT,
    source_city     TEXT NOT NULL,
    destination_city TEXT NOT NULL,
    start_date      TEXT NOT NULL,
    num_days        INTEGER NOT NULL,
    budget          REAL NOT NULL,
    travel_style    TEXT NOT NULL,
    num_travellers  INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_user_queries_session
    ON user_queries (session_id);

CREATE INDEX IF NOT EXISTS idx_user_queries_route
    ON user_queries (source_city, destination_city);


-- ---------------------------------------------------------------------------
-- itineraries: the generated itinerary (JSON payload) for a query.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS itineraries (
    itinerary_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id        INTEGER NOT NULL,
    trip_title      TEXT NOT NULL,
    itinerary_json  TEXT NOT NULL,
    total_cost      REAL,
    budget_category TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (query_id) REFERENCES user_queries (query_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_itineraries_query
    ON itineraries (query_id);


-- ---------------------------------------------------------------------------
-- flight_selections: the flight chosen for a given query.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS flight_selections (
    selection_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id        INTEGER NOT NULL,
    flight_id       TEXT NOT NULL,
    airline         TEXT NOT NULL,
    source_city     TEXT NOT NULL,
    destination_city TEXT NOT NULL,
    price           REAL NOT NULL,
    duration_minutes INTEGER NOT NULL,
    selection_reason TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (query_id) REFERENCES user_queries (query_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_flight_selections_query
    ON flight_selections (query_id);


-- ---------------------------------------------------------------------------
-- hotel_selections: the hotel chosen for a given query.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS hotel_selections (
    selection_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id        INTEGER NOT NULL,
    hotel_id        TEXT NOT NULL,
    hotel_name      TEXT NOT NULL,
    city            TEXT NOT NULL,
    stars           INTEGER NOT NULL,
    price_per_night REAL NOT NULL,
    selection_reason TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (query_id) REFERENCES user_queries (query_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_hotel_selections_query
    ON hotel_selections (query_id);


-- ---------------------------------------------------------------------------
-- search_logs: a log of every tool invocation, for observability/debugging.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS search_logs (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    input_payload   TEXT,
    output_status   TEXT NOT NULL,
    duration_ms     REAL,
    error_message   TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_search_logs_session
    ON search_logs (session_id);

CREATE INDEX IF NOT EXISTS idx_search_logs_tool
    ON search_logs (tool_name);
