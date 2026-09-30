"""API + trang web: upload video, xem webcam trực tiếp, tải events.json."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import cv2
from flask import Blueprint, Response, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

import config
from camera.camera import VideoSource
from pipeline import ExamMonitor, make_db, run_video

bp = Blueprint("main", __name__)

ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
JOBS: dict = {}
_process_lock = threading.Lock()          # xử lý lần lượt từng video (tránh tràn GPU/RAM)
LIVE = {"stop": None, "monitor": None}
_live_lock = threading.Lock()


# --------------------------------------------------------------------- trang
@bp.get("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------ upload + xử lý
def _worker(job_id: str, in_path: Path, out_dir: Path) -> None:
    job = JOBS[job_id]
    with _process_lock:
        job["status"] = "processing"
        try:
            def progress(done, total):
                job["progress"] = round(100 * done / total, 1) if total else 0.0
            res = run_video(str(in_path), out_dir, progress)
            job.update(status="done", progress=100.0, total_events=res["total_events"])
        except Exception as exc:  # noqa: BLE001
            job.update(status="error", error=str(exc))


@bp.post("/api/upload")
def upload():
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify(error="Chưa chọn file"), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify(error=f"Định dạng không hỗ trợ ({ext})"), 400
    job_id = uuid.uuid4().hex[:10]
    in_path = config.UPLOAD_DIR / f"{job_id}_{secure_filename(f.filename)}"
    f.save(in_path)
    out_dir = config.OUTPUT_DIR / job_id
    JOBS[job_id] = {"status": "queued", "progress": 0.0, "out_dir": str(out_dir), "name": f.filename}
    threading.Thread(target=_worker, args=(job_id, in_path, out_dir), daemon=True).start()
    return jsonify(job_id=job_id)


@bp.get("/api/jobs/<job_id>")
def job_status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Không tìm thấy job"), 404
    return jsonify({k: v for k, v in job.items() if k != "out_dir"})


@bp.get("/api/jobs/<job_id>/video")
def job_video(job_id):
    job = JOBS.get(job_id)
    p = Path(job["out_dir"]) / "annotated.mp4" if job else None
    if not p or not p.exists():
        return jsonify(error="Chưa có video"), 404
    return send_file(p, mimetype="video/mp4", conditional=True)


@bp.get("/api/jobs/<job_id>/events")
def job_events(job_id):
    job = JOBS.get(job_id)
    p = Path(job["out_dir"]) / "events.json" if job else None
    if not p or not p.exists():
        return jsonify(error="Chưa có events"), 404
    if request.args.get("download"):
        return send_file(p, as_attachment=True, download_name="events.json")
    return Response(p.read_text(encoding="utf-8"), mimetype="application/json")


# ------------------------------------------------------------ webcam trực tiếp
def _mjpeg(src: VideoSource, monitor: ExamMonitor, stop: threading.Event):
    interval = 1.0 / config.LIVE_STREAM_FPS
    next_frame_time = time.monotonic()

    try:
        while not stop.is_set():
            item = src.read()

            if item is None:
                break

            frame, ts, idx = item

            out = monitor.process_frame(frame, ts)

            ok, buf = cv2.imencode(
                ".jpg",
                out,
                [cv2.IMWRITE_JPEG_QUALITY, config.JPEG_QUALITY]
            )

            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )

            next_frame_time += interval
            delay = next_frame_time - time.monotonic()

            if delay > 0:
                time.sleep(delay)
            else:
                next_frame_time = time.monotonic()

    finally:
        monitor.finish()
        src.release()
        with _live_lock:
            LIVE.update(stop=None, monitor=None)


@bp.get("/live/feed")
def live_feed():
    with _live_lock:
        if LIVE["stop"] is not None:
            return jsonify(error="Đang có một phiên webcam khác"), 409
        try:
            src = VideoSource(config.CAMERA_INDEX)
        except RuntimeError as exc:
            return jsonify(error=str(exc)), 503
        stop = threading.Event()
        run_id = uuid.uuid4().hex[:8]
        monitor = ExamMonitor(f"webcam:{config.CAMERA_INDEX}", "live", src.fps,
                              config.EVENTS_DIR / f"live_{run_id}" / "events.json", db=make_db())
        LIVE.update(stop=stop, monitor=monitor)
    return Response(_mjpeg(src, monitor, stop), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.post("/api/live/stop")
def live_stop():
    with _live_lock:
        if LIVE["stop"] is not None:
            LIVE["stop"].set()
    return jsonify(ok=True)


@bp.get("/api/live/events")
def live_events():
    m = LIVE.get("monitor")
    return jsonify(events=list(m.logger.events) if m else [])
