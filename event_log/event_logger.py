"""Ghi các sự kiện hành vi ra events.json (và SQLite nếu bật).

Lưu ý: thư mục này KHÔNG đặt tên `logging/` vì sẽ đè lên module `logging` của Python.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from behavior.behavior_state import BehaviorEvent, Status


def timecode(sec: float) -> str:
    m, s = divmod(max(sec, 0.0), 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}"


class EventLogger:
    def __init__(self, json_path: Path, source: str, mode: str, fps: Optional[float] = None,
                 db=None, session_id: Optional[int] = None):
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        self.source, self.mode, self.fps = source, mode, fps
        self.db, self.session_id = db, session_id
        self.started_at = datetime.now()
        self.events: list[dict] = []
        self.save()

    def log(self, ev: BehaviorEvent) -> dict:
        rec = {
            "person_id": ev.person_id,
            "behavior": ev.behavior.value,
            "status": Status.SUSPICIOUS.value,
            "start_time": round(ev.start_time, 2),
            "end_time": round(ev.end_time, 2),
            "duration": round(ev.end_time - ev.start_time, 2),
            "start_timecode": timecode(ev.start_time),
            "end_timecode": timecode(ev.end_time),
        }
        if self.fps:
            rec["start_frame"] = int(round(ev.start_time * self.fps))
            rec["end_frame"] = int(round(ev.end_time * self.fps))
        if self.mode == "live":
            rec["start_datetime"] = (self.started_at.timestamp() + ev.start_time)
            rec["start_datetime"] = datetime.fromtimestamp(rec["start_datetime"]).isoformat(timespec="seconds")
        self.events.append(rec)
        if self.db is not None and self.session_id is not None:
            self.db.add_event(self.session_id, ev.person_id, rec["behavior"], rec["status"],
                              ev.start_time, ev.end_time)
        self.save()
        return rec

    def save(self) -> None:
        summary: dict = {}
        for e in self.events:
            summary[e["behavior"]] = summary.get(e["behavior"], 0) + 1
        doc = {
            "source": self.source,
            "mode": self.mode,
            "fps": self.fps,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "total_events": len(self.events),
            "summary": summary,
            "events": sorted(self.events, key=lambda e: (e["start_time"], e["person_id"])),
        }
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.json_path)
