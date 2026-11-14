"""Analyze every available source frame.
Camera uses the newest eligible frame without building a stale queue.
Analysis can be paused independently of capture.
"""
import os
os.environ.setdefault('OPENCV_FORCE_DNN_ENGINE', '4')
import threading
import time
from pathlib import Path
from uuid import uuid4
from rules import Rules
from live_capture import LatestCamera


class Engine:
    def __init__(self, store, evidence_dir):
        self.store = store
        self.evidence_dir = Path(evidence_dir); self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock(); self.state_lock = threading.RLock()
        self.frame_ready = threading.Condition(self.state_lock)
        self.source_info = {}
        self.monitor = None; self.cap = None; self.camera = None; self.thread = None
        self.running = True; self.analysis_enabled = False; self.finished = False
        self.frame_stride = max(1, int(os.environ.get('AI_FRAME_STRIDE', '1')))
        self.session_id = None; self.mode = 'idle'; self.source_fps = 25.0
        self.frame_no = 0; self.last_sequence = 0; self.media = 0.; self.skipped_frames = 0; self.samples = 0
        self.pending_frame = None; self.jpg = None; self.video_start_time = None
        self.rules = Rules(store.settings())
        self.status = dict(ready=False, error='', persons=0, ai_fps=0, tracks=[], mode='idle', timings={})

    def start_worker(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self.loop, daemon=True, name='ExamAIWorker'); self.thread.start()

    def load(self):
        if self.monitor is None:
            from config import MODEL_PATH, FACE_DETECTOR_MODEL, POSE_MODEL_PATH, HEAD_BACKEND
            head_path = POSE_MODEL_PATH if HEAD_BACKEND == "pose" else FACE_DETECTOR_MODEL
            missing = [str(p) for p in (MODEL_PATH, head_path) if not p.is_file()]
            if missing: raise ValueError('Thiếu model: ' + ', '.join(missing))
            from core import Monitor
            self.monitor = Monitor(self.store)

    def reset(self):
        self.rules.reset()
        if self.monitor is not None:
            from core import Pose
            m = self.monitor; m.pose = Pose(); m.track_history.clear(); m.cache.clear(); m.lastrec.clear(); m.events.clear()
            m.last_tracks = []; m.objects = []; m.frame_count = 0; m._event_times.clear(); m.positions.reset()
            for tracker in getattr(getattr(m.model, 'predictor', None), 'trackers', []): tracker.reset()

    @staticmethod
    def encode(frame, width=1280, quality=80):
        import cv2
        h, w = frame.shape[:2]
        if width and w > width: frame = cv2.resize(frame, (width, round(h * width / w)))
        ok, data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok: raise RuntimeError('Không mã hóa được ảnh')
        return data.tobytes()

    def source(self, source, mode):
        import cv2
        cv2.setNumThreads(max(1, int(os.environ.get('CV_THREADS', '2'))))
        with self.lock:
            if isinstance(source, int):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            elif mode == 'video': cap = cv2.VideoCapture(str(source))
            else:
                cap = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
            if not cap.isOpened():
                cap.release(); raise ValueError('Không mở được nguồn hình ảnh')
            pending = None; preview = None
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                if isinstance(source, int):
                    fourcc_setting = os.environ.get('CAMERA_FOURCC', '').strip()
                    if fourcc_setting:
                        if len(fourcc_setting) != 4: raise ValueError('CAMERA_FOURCC phải có 4 ký tự')
                        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_setting))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(os.environ.get('CAMERA_WIDTH', '1280')))
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(os.environ.get('CAMERA_HEIGHT', '720')))
                    cap.set(cv2.CAP_PROP_FPS, max(1, min(60, int(os.environ.get('CAMERA_FPS', '30')))))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if mode == 'video':
                    ok, pending = cap.read()
                    if not ok: raise ValueError('Video không có frame đọc được')
                    preview = self.encode(pending)
            except Exception:
                cap.release(); raise
            self.end_session()
            if self.camera: self.camera.close()
            elif self.cap is not None: self.cap.release()
            self.camera = None; self.cap = cap; self.mode = mode
            self.source_fps = fps if 1 <= fps <= 240 else 25.
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.source_info = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                  height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                  reported_fps=round(float(fps), 2) if 0 < fps < 1000 else None,
                                  fourcc=''.join(chr((fourcc >> (8 * i)) & 255) for i in range(4)))
            self.pending_frame = pending; self.frame_no = 0; self.last_sequence = 0
            self.skipped_frames = 0; self.samples = 0; self.media = 0.; self.finished = False
            self.video_start_time = None
            self.reset()
            if mode != 'video':
                self.camera = LatestCamera(cap,
                    preview_fps=max(1, min(60, int(os.environ.get('PREVIEW_FPS', '30')))),
                    preview_width=max(320, int(os.environ.get('PREVIEW_WIDTH', '960'))),
                    jpeg_quality=int(os.environ.get('PREVIEW_JPEG_QUALITY', '70')))
            with self.state_lock:
                self.jpg = preview
                self.status = dict(ready=True, error='', persons=0, ai_fps=0, tracks=[], mode=mode, timings={})
                self.frame_ready.notify_all()
            self.store.audit('source_change', {'mode': mode})

    def set_analysis(self, enabled):
        with self.lock:
            if not enabled:
                self.analysis_enabled = False; self.rules.reset()
                if self.camera: self.camera.clear_overlay()
                return
            if self.analysis_enabled: return
            if self.cap is None or self.finished: raise ValueError('Hãy tải video hoặc mở camera trước')
            if self.camera:
                packet, error, _ = self.camera.latest()
                if error or packet is None or time.monotonic() - packet[1] > 2:
                    raise ValueError(error or 'Camera chưa có ảnh mới')
            self.load(); self.reset()
            if self.session_id is None:
                self.session_id = self.store.start('Phân tích ' + time.strftime('%Y-%m-%d %H:%M:%S'))
            self.analysis_enabled = True
            with self.state_lock: self.status.update(ready=True, error='')

    def end_session(self):
        self.analysis_enabled = False
        if self.session_id is not None: self.store.end(self.session_id)
        self.session_id = None; self.rules.reset()
        if self.camera: self.camera.clear_overlay()

    def snapshot(self):
        with self.state_lock:
            result = dict(self.status, analysis_enabled=self.analysis_enabled, finished=self.finished,
                        frame_stride=self.frame_stride, skipped_frames=self.skipped_frames, source_info=dict(self.source_info),
                        sampled_frames=self.samples, source_frame=self.frame_no,
                        source_fps=self.source_fps, settings=dict(self.rules.settings),
                        media_time=round(self.media, 3) if self.mode == 'video' else None)
        if self.camera:
            packet, error, fps = self.camera.latest()
            age = time.monotonic() - packet[1] if packet else None
            result.update(capture_fps=round(fps, 1), frame_age_ms=round(age * 1000, 1) if age is not None else None)
            result.update(self.camera.metrics())
            if not error and age is not None and age > 2: error = 'Camera không có ảnh mới trong hơn 2 giây'
            if error: result.update(ready=False, error=error, tracks=[], persons=0)
        return result

    def preview(self, analysis=False):
        if self.camera: return self.camera.preview(analysis=analysis and self.analysis_enabled)
        with self.state_lock: return self.jpg

    def wait_preview(self, last, analysis=False):
        camera = self.camera
        if camera:
            return camera.wait_preview(last, timeout=.5, analysis=analysis and self.analysis_enabled)
        with self.frame_ready:
            self.frame_ready.wait_for(lambda: self.jpg is not last or not self.running, timeout=.5)
            return self.jpg

    def next_video_sample(self):
        if self.pending_frame is not None:
            frame = self.pending_frame; self.pending_frame = None; self.frame_no = 1
            self.media = 0.
            return frame
        while True:
            if not self.cap.grab(): return None
            self.frame_no += 1
            if (self.frame_no - 1) % self.frame_stride:
                self.skipped_frames += 1
                continue
            ok, frame = self.cap.retrieve()
            if not ok: raise RuntimeError('Không giải mã được frame ' + str(self.frame_no))
            self.media = (self.frame_no - 1) / self.source_fps
            return frame

    def loop(self):
        while self.running:
            time.sleep(.005)
            with self.lock:
                if self.cap is None or not self.analysis_enabled: continue
                started = time.perf_counter()
                try:
                    if self.camera:
                        packet, error, _ = self.camera.latest()
                        if error: raise RuntimeError(error)
                        if packet is None: continue
                        seq, captured, frame = packet
                        if time.monotonic() - captured > 2: raise RuntimeError('Camera không có ảnh mới trong hơn 2 giây')
                        if seq <= self.last_sequence: continue
                        if self.last_sequence and seq - self.last_sequence < self.frame_stride: continue
                        self.skipped_frames += max(0, seq - self.last_sequence - 1)
                        self.last_sequence = seq; self.frame_no = seq; self.media = captured
                    else:
                        frame = self.next_video_sample()
                        if frame is None:
                            self.cap.release(); self.cap = None; self.finished = True; self.end_session()
                            with self.state_lock: self.status.update(error='', ready=True, tracks=[], persons=0)
                            continue
                        
                        if self.mode == 'video' and self.source_fps > 0:
                            if self.video_start_time is None:
                                self.video_start_time = time.monotonic() - self.media
                            
                            target_time = self.video_start_time + self.media
                            sleep_duration = target_time - time.monotonic()
                            
                            if sleep_duration > 0:
                                time.sleep(sleep_duration)
                            elif sleep_duration < -0.5:
                                self.video_start_time = time.monotonic() - self.media

                    inference_start = time.perf_counter()
                    tracks = self.monitor.process(frame, media_time=self.media, mode=self.mode)
                    inference_ms = (time.perf_counter() - inference_start) * 1000
                    h, w = frame.shape[:2]
                    events = self.rules.update(tracks, self.monitor.objects, {}, [], w, h, self.media)
                    
                    # Thu thập sự kiện trực tiếp từ monitor
                    if self.monitor.events:
                        events.extend(self.monitor.events)
                        self.monitor.events.clear()

                    for track in tracks:
                        track['alert_confirmed'] = track['track_id'] in self.rules.alerted_tracks
                    if self.camera:
                        self.camera.set_overlay(tracks, self.monitor.objects, frame.shape)
                    output = self.monitor.draw(frame.copy(), tracks)
                    jpg = self.encode(output) if not self.camera else None
                    if events:
                        name = uuid4().hex + '.jpg'
                        (self.evidence_dir / name).write_bytes(self.encode(output, width=None, quality=85))
                        for event in events:
                            self.store.event(self.session_id, self.media if self.mode == 'video' else None, event, name)
                    self.samples += 1
                    elapsed = max(time.perf_counter() - started, .001)
                    with self.state_lock:
                        if jpg is not None: self.jpg = jpg
                        self.status.update(ready=True, error='', persons=len(tracks), tracks=tracks,
                            ai_fps=round(1 / elapsed, 2), inference_ms=round(inference_ms, 1),
                            timings=getattr(self.monitor, 'timings', {}),
                            latency_ms=round((time.monotonic() - self.media) * 1000, 1) if self.camera else None)
                        self.frame_ready.notify_all()
                except Exception as exc:
                    self.analysis_enabled = False; self.rules.reset()
                    if self.camera: self.camera.clear_overlay()
                    with self.state_lock: self.status.update(ready=False, error=str(exc), tracks=[], persons=0)

    def close(self):
        self.running = False
        with self.frame_ready: self.frame_ready.notify_all()
        with self.lock:
            self.end_session()
            if self.camera: self.camera.close(); self.camera = None
            elif self.cap is not None: self.cap.release()
            self.cap = None