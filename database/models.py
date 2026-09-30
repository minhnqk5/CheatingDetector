"""Mô hình dữ liệu lưu trong SQLite."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class SessionRecord:
    id: Optional[int]
    source: str
    mode: str            # "live" | "video"
    started_at: str
    finished_at: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class EventRecord:
    id: Optional[int]
    session_id: int
    person_id: int
    behavior: str
    status: str
    start_time: float
    end_time: float
    duration: float

    def to_dict(self):
        return asdict(self)
