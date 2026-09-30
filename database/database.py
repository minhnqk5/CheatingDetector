"""SQLite đơn giản (thư viện chuẩn, không cần cài thêm)."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime

import config
from database.models import EventRecord, SessionRecord


class Database:
    def __init__(self, path=None):
        path = path or config.DB_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, mode TEXT,
                    started_at TEXT, finished_at TEXT);
                CREATE TABLE IF NOT EXISTS events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER, person_id INTEGER,
                    behavior TEXT, status TEXT, start_time REAL, end_time REAL, duration REAL);
                CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
                """
            )

    def create_session(self, source: str, mode: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO sessions(source, mode, started_at) VALUES (?,?,?)",
                (source, mode, datetime.now().isoformat(timespec="seconds")))
            return cur.lastrowid

    def finish_session(self, session_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE sessions SET finished_at=? WHERE id=?",
                               (datetime.now().isoformat(timespec="seconds"), session_id))

    def add_event(self, session_id: int, person_id: int, behavior: str, status: str,
                  start: float, end: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO events(session_id, person_id, behavior, status, start_time, end_time, duration)"
                " VALUES (?,?,?,?,?,?,?)",
                (session_id, person_id, behavior, status, start, end, end - start))

    def list_events(self, session_id: int) -> list[EventRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE session_id=? ORDER BY start_time", (session_id,)).fetchall()
        return [EventRecord(**dict(r)) for r in rows]

    def list_sessions(self) -> list[SessionRecord]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
        return [SessionRecord(**dict(r)) for r in rows]
