"""Lõi xử lý dùng chung cho CLI (live/video) và web."""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2

import config
from behavior.behavior_detector import BehaviorDetector
from camera.camera import VideoSource
from event_log.event_logger import EventLogger
from tracking.person_tracker import PersonTracker
from visualizer import draw_frame

log = logging.getLogger(__name__)


class _SharedDetector:
    """Model nạp 1 lần, dùng chung giữa các luồng (có khoá để an toàn)."""
    _inst = None
    _guard = threading.Lock()

    def __init__(self, person, objects):
        self.person, self.objects, self.lock = person, objects, threading.Lock()

    @classmethod
    def get(cls) -> "_SharedDetector":
        with cls._guard:
            if cls._inst is None:
                from detection.person_detector import PersonDetector
                obj = None
                if config.ENABLE_OBJECT_DETECTOR:
                    from detection.object_detector import ObjectDetector
                    obj = ObjectDetector()
                cls._inst = cls(PersonDetector(), obj)
            return cls._inst

    def detect(self, frame):
        with self.lock:
            return self.person.detect(frame)

    def detect_objects(self, frame):
        if self.objects is None:
            return None
        with self.lock:
            return self.objects.detect(frame)


def get_detector() -> _SharedDetector:
    return _SharedDetector.get()


class ExamMonitor:
    """Một phiên giám sát: detect -> track -> hành vi -> log -> vẽ."""

    def __init__(self, source_name: str, mode: str, fps: float, events_path: Path,
                 detector=None, db=None):
        self.detector = detector or get_detector()
        self.tracker = PersonTracker()
        self.behavior = BehaviorDetector()
        self.session_id = db.create_session(source_name, mode) if db else None
        self.db = db
        self.logger = EventLogger(events_path, source_name, mode, fps, db, self.session_id)
        self.last_ts = 0.0

        self.last_detection_ts = -1.0
        self.last_dets = []


    def process_frame(self, frame, ts: float):
        self.last_ts = ts

        if (
            self.last_detection_ts < 0
            or ts - self.last_detection_ts >= 1.0 / config.POSE_DETECT_FPS
        ):
            self.last_dets = self.detector.detect(frame)
            self.last_detection_ts = ts

        dets = self.last_dets

        out = self.tracker.update(dets, ts)

        states, closed = self.behavior.update(
            out.visible,
            out.lost,
            ts,
            None
        )

        for pid in out.removed_ids:
            closed += self.behavior.drop_person(pid)

        for ev in closed:
            self.logger.log(ev)

        return draw_frame(
            frame.copy(),
            out.visible,
            out.lost,
            states,
            ts,
            None
        )

    def finish(self) -> None:
        for ev in self.behavior.flush():
            self.logger.log(ev)
        self.logger.save()
        if self.db and self.session_id:
            self.db.finish_session(self.session_id)


def new_run_dir(base: Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base / f"{Path(name).stem}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_db():
    if not config.ENABLE_DB:
        return None
    from database.database import Database
    return Database()


def transcode_h264(src: Path, dst: Path) -> bool:
    """mp4v của OpenCV không phát được trong trình duyệt -> chuyển sang H.264."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
    if not exe:
        return False
    cmd = [exe, "-y", "-i", str(src), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Chuyển H.264 thất bại: %s", exc)
        return False


def run_video(input_path: str, out_dir: Path, progress_cb: Optional[Callable[[int, int], None]] = None,
              stop_event: Optional[threading.Event] = None, detector=None, show: bool = False) -> dict:
    """Xử lý video upload -> annotated.mp4 + events.json trong out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = VideoSource(str(input_path))
    events_path = out_dir / "events.json"
    raw_path, final_path = out_dir / "_raw.mp4", out_dir / "annotated.mp4"
    monitor = ExamMonitor(Path(input_path).name, "video", src.fps, events_path, detector, make_db())
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (src.width, src.height))
    try:
        while not (stop_event and stop_event.is_set()):
            item = src.read()
            if item is None:
                break
            frame, ts, idx = item
            out = monitor.process_frame(frame, ts)
            writer.write(out)
            if progress_cb:
                progress_cb(idx + 1, src.frame_count)
            if show:
                cv2.imshow("Exam Monitoring", out)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        writer.release()
        src.release()
        monitor.finish()
    if transcode_h264(raw_path, final_path):
        raw_path.unlink(missing_ok=True)
    else:
        raw_path.replace(final_path)
    return {"video": str(final_path), "events": str(events_path), "total_events": len(monitor.logger.events)}
