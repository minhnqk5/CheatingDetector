"""
Phân tích theo thời gian: biến tín hiệu thô từng khung (True/False/None) thành các
đoạn hành vi [start, end], có ngưỡng thời lượng tối thiểu và chống nhấp nháy.
"""
from __future__ import annotations

from typing import Optional, Tuple

import config
from behavior.behavior_state import Behavior, BehaviorEvent


class SegmentTracker:
    """idle -> pending (đang nghi) -> active (đã đủ thời lượng) -> idle (kết thúc)."""

    def __init__(self, min_duration: float, release: float, hold: float):
        self.min_duration, self.release, self.hold = min_duration, release, hold
        self.state = "idle"
        self.start: Optional[float] = None
        self.last_active: Optional[float] = None

    @property
    def is_active(self) -> bool:
        return self.state == "active"

    def update(self, ts: float, active: Optional[bool]) -> Optional[Tuple[float, float]]:
        if active:
            if self.state == "idle":
                self.state, self.start = "pending", ts
            self.last_active = ts
            if self.state == "pending" and ts - self.start >= self.min_duration:
                self.state = "active"
            return None
        if self.state == "idle":
            return None
        limit = self.release + (self.hold if active is None else 0.0)
        if ts - self.last_active >= limit:
            return self._end()
        return None

    def close(self) -> Optional[Tuple[float, float]]:
        return self._end()

    def _end(self):
        seg = (self.start, self.last_active) if self.state == "active" else None
        self.state, self.start, self.last_active = "idle", None, None
        return seg


class TemporalAnalyzer:
    def __init__(self):
        self._params = {
            Behavior.TURN_LEFT: (config.TURN_MIN_SEC, config.TURN_RELEASE_SEC),
            Behavior.TURN_RIGHT: (config.TURN_MIN_SEC, config.TURN_RELEASE_SEC),
            Behavior.LEFT_SEAT: (config.LEAVE_MIN_SEC, config.LEAVE_RELEASE_SEC),
            Behavior.PHONE: (config.PHONE_MIN_SEC, config.PHONE_RELEASE_SEC),
        }
        self._trk: dict = {}

    def _get(self, pid: int, b: Behavior) -> SegmentTracker:
        key = (pid, b)
        if key not in self._trk:
            mn, rel = self._params[b]
            self._trk[key] = SegmentTracker(mn, rel, config.NO_INFO_HOLD_SEC)
        return self._trk[key]

    def update(self, pid: int, b: Behavior, ts: float, active: Optional[bool]) -> Optional[BehaviorEvent]:
        seg = self._get(pid, b).update(ts, active)
        return BehaviorEvent(pid, b, seg[0], seg[1]) if seg else None

    def is_active(self, pid: int, b: Behavior) -> bool:
        t = self._trk.get((pid, b))
        return bool(t and t.is_active)

    def close_person(self, pid: int) -> list:
        out = []
        for (p, b), t in list(self._trk.items()):
            if p == pid:
                seg = t.close()
                if seg:
                    out.append(BehaviorEvent(p, b, seg[0], seg[1]))
                del self._trk[(p, b)]
        return out

    def close_all(self) -> list:
        out = []
        for pid in {p for p, _ in self._trk}:
            out.extend(self.close_person(pid))
        return sorted(out, key=lambda e: e.start_time)
