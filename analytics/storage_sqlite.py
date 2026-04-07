from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SessionRow:
    track_id: int
    time_in: float
    time_out: float
    duration_seconds: float
    model_path: str
    source: str


class SQLiteStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        return con

    def _init_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  track_id INTEGER NOT NULL,
                  time_in REAL NOT NULL,
                  time_out REAL NOT NULL,
                  duration_seconds REAL NOT NULL,
                  model_path TEXT NOT NULL,
                  source TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_sessions_time_in
                ON sessions(time_in)
                """
            )

    def insert_session(self, row: SessionRow) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO sessions(track_id, time_in, time_out, duration_seconds, model_path, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (row.track_id, row.time_in, row.time_out, row.duration_seconds, row.model_path, row.source),
            )

    def report_by_day(self) -> List[Tuple[str, int, float]]:
        """Return list of (YYYY-MM-DD, total_sessions, avg_duration_seconds)."""
        with self._connect() as con:
            cur = con.execute(
                """
                SELECT
                  date(time_in, 'unixepoch') AS d,
                  COUNT(*) AS total_sessions,
                  AVG(duration_seconds) AS avg_duration
                FROM sessions
                GROUP BY d
                ORDER BY d DESC
                """
            )
            return [(r[0], int(r[1]), float(r[2] or 0.0)) for r in cur.fetchall()]

    def report_by_hour(self, day: Optional[str] = None) -> List[Tuple[str, int, float]]:
        """Return list of (YYYY-MM-DD HH:00, total_sessions, avg_duration_seconds)."""
        with self._connect() as con:
            if day:
                cur = con.execute(
                    """
                    SELECT
                      strftime('%Y-%m-%d %H:00', time_in, 'unixepoch') AS h,
                      COUNT(*) AS total_sessions,
                      AVG(duration_seconds) AS avg_duration
                    FROM sessions
                    WHERE date(time_in, 'unixepoch') = ?
                    GROUP BY h
                    ORDER BY h ASC
                    """,
                    (day,),
                )
            else:
                cur = con.execute(
                    """
                    SELECT
                      strftime('%Y-%m-%d %H:00', time_in, 'unixepoch') AS h,
                      COUNT(*) AS total_sessions,
                      AVG(duration_seconds) AS avg_duration
                    FROM sessions
                    GROUP BY h
                    ORDER BY h DESC
                    LIMIT 200
                    """
                )
            return [(r[0], int(r[1]), float(r[2] or 0.0)) for r in cur.fetchall()]

