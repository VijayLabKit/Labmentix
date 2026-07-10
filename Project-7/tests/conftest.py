"""Shared pytest fixtures for the travel assistant test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is importable as a package root when running
# `pytest` from any working directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.helpers import load_json_dataset  # noqa: E402
from configs import settings  # noqa: E402


@pytest.fixture(scope="session")
def flights_data():
    """Raw flights dataset, loaded once per test session."""
    return load_json_dataset(settings.FLIGHTS_FILE)


@pytest.fixture(scope="session")
def hotels_data():
    """Raw hotels dataset, loaded once per test session."""
    return load_json_dataset(settings.HOTELS_FILE)


@pytest.fixture(scope="session")
def places_data():
    """Raw places dataset, loaded once per test session."""
    return load_json_dataset(settings.PLACES_FILE)


@pytest.fixture()
def temp_database(tmp_path, monkeypatch):
    """Provide a :class:`TravelDatabase` backed by a temporary SQLite file.

    Patches :data:`configs.settings.DATABASE_PATH` and clears the cached
    singleton in :mod:`database.database` so tests do not write to the
    project's real database file.
    """
    import database.database as database_module

    db_path = tmp_path / "test_travel_assistant.db"
    monkeypatch.setattr(settings, "DATABASE_PATH", db_path)

    # NOTE: TravelDatabase's constructor binds ``settings.DATABASE_PATH`` as a
    # default *at function-definition time*, so monkeypatching the setting
    # alone would not redirect ``get_database()``. Construct the instance
    # explicitly with the temp path and install it as the cached singleton so
    # any code that calls ``get_database()`` during the test also uses it.
    db = database_module.TravelDatabase(db_path=db_path, schema_path=settings.SCHEMA_PATH)
    monkeypatch.setattr(database_module, "_default_db", db)

    yield db

    monkeypatch.setattr(database_module, "_default_db", None)
