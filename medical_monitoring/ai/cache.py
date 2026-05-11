"""
Semantic cache backed by SQLite.

Medical terminology mappings are highly cacheable -- the same AE name maps
to the same MedDRA PT deterministically.  Expected cache hit rate > 70%
within a single trial (many repeated AE terms across subjects).
"""

from __future__ import annotations

import json
import sqlite3
import time
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SemanticCache:
    """SQLite-backed key-value cache with TTL expiry."""

    _DDL = """
    CREATE TABLE IF NOT EXISTS cache (
        key     TEXT PRIMARY KEY,
        value   TEXT NOT NULL,
        model   TEXT,
        created REAL NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_cache_created ON cache(created);
    """

    def __init__(self, db_path: str | Path = ".ai_cache.db", ttl_days: int = 30):
        self.db_path = str(db_path)
        self.ttl_seconds = ttl_days * 86400
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self.db_path)
        self._conn.executescript(self._DDL)
        self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        """Return cached value or None if missing / expired."""
        if self._conn is None:
            return None
        cursor = self._conn.execute(
            "SELECT value, created FROM cache WHERE key = ?", (key,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        value_str, created = row
        if time.time() - created > self.ttl_seconds:
            self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            self._conn.commit()
            return None
        try:
            return json.loads(value_str)
        except json.JSONDecodeError:
            return None

    def put(self, key: str, value: dict[str, Any]) -> None:
        """Insert or replace a cache entry."""
        if self._conn is None:
            return
        value_str = json.dumps(value, ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, created) VALUES (?, ?, ?)",
            (key, value_str, time.time()),
        )
        self._conn.commit()

    def clear_expired(self) -> int:
        """Remove all expired entries. Returns number of rows deleted."""
        if self._conn is None:
            return 0
        cutoff = time.time() - self.ttl_seconds
        cursor = self._conn.execute("DELETE FROM cache WHERE created < ?", (cutoff,))
        self._conn.commit()
        return cursor.rowcount

    def clear_all(self) -> None:
        """Wipe the entire cache."""
        if self._conn is None:
            return
        self._conn.execute("DELETE FROM cache")
        self._conn.commit()

    def stats(self) -> dict[str, int]:
        """Return basic cache statistics."""
        if self._conn is None:
            return {"total": 0, "expired": 0}
        total = self._conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
        cutoff = time.time() - self.ttl_seconds
        expired = self._conn.execute(
            "SELECT COUNT(*) FROM cache WHERE created < ?", (cutoff,)
        ).fetchone()[0]
        return {"total": total, "expired": expired, "active": total - expired}

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
