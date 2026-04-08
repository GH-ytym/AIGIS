"""Database engine/session helpers for local sqlite."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aigis_agent.core.config import settings
from aigis_agent.db.base import Base


def _ensure_sqlite_parent_dir() -> None:
    """Create sqlite parent directory if missing."""
    prefix = "sqlite:///"
    if settings.sqlite_url.startswith(prefix):
        db_path = Path(settings.sqlite_url[len(prefix) :])
        db_path.parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_parent_dir()
engine = create_engine(settings.sqlite_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    """Create tables for MVP logs and metadata."""
    import aigis_agent.db.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
