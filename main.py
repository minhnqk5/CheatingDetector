"""
Điểm vào của hệ thống.

  python main.py live  [--source 0] [--record out.mp4]   # webcam trực tiếp (bbox + khung xương)
  python main.py video input.mp4 [--out-dir data/outputs/x] [--show]
  python main.py web                                       # giao diện web (upload video / xem webcam)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import cv2

import config


def setup_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(config.LOG_DIR / "app.log", encoding="utf-8")])


def cmd_live(args) -> None:
    from camera.camera import VideoSource
    from pipeline import ExamMonitor, make_db
    src_id = int(args.source) if str(args.source).isdigit() else args.source
    src = VideoSource(src_id)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    events_path = config.EVENTS_DIR / f"live_{stamp}" / "events.json"
    monitor = ExamMonitor(f"webcam:{args.source}", "live", src.fps, events_path, db=make_db())
    writer = None
    if args.record:
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (src.width, src.height))
    logging.info("Đang chạy. Nhấn 'q' hoặc ESC để dừng. Events: %s", events_path)
    try:
        while True:
            item = src.read()
            if item is None:
                break
            frame, ts, _ = item
            out = monitor.process_frame(frame, ts)
            if writer:
                writer.write(out)
            cv2.imshow("Exam Monitoring (q = thoat)", out)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        monitor.finish()
        src.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    logging.info("Đã lưu %s", events_path)


def cmd_video(args) -> None:
    from pathlib import Path
    from pipeline import new_run_dir, run_video

    out_dir = Path(args.out_dir) if args.out_dir else new_run_dir(config.OUTPUT_DIR, args.input)

    def progress(done: int, total: int) -> None:
        if total and done % 30 == 0:
            print(f"\r{done}/{total} khung ({100 * done / total:.0f}%)", end="", flush=True)

    res = run_video(args.input, out_dir, progress, show=args.show)
    print(f"\nVideo: {res['video']}\nEvents: {res['events']} ({res['total_events']} sự kiện)")


def cmd_web(_args) -> None:
    from web.app import create_app
    create_app().run(host=config.WEB_HOST, port=config.WEB_PORT, threaded=True, debug=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Exam monitoring - phát hiện hành vi gian lận")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("live"); p.add_argument("--source", default=str(config.CAMERA_INDEX))
    p.add_argument("--record", help="lưu video đã vẽ (mp4)"); p.set_defaults(fn=cmd_live)
    p = sub.add_parser("video"); p.add_argument("input"); p.add_argument("--out-dir")
    p.add_argument("--show", action="store_true"); p.set_defaults(fn=cmd_video)
    p = sub.add_parser("web"); p.set_defaults(fn=cmd_web)
    args = ap.parse_args()
    setup_logging()
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
