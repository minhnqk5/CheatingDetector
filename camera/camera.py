"""Nguồn video: webcam trực tiếp (luồng đọc riêng, luôn lấy khung mới nhất) hoặc file video."""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np

import config


class VideoSource:
    def __init__(self, source: Union[int, str], width: Optional[int] = None, height: Optional[int] = None):
        self.is_live = isinstance(source, int) or str(source).isdigit()

        self._next_process_time = 0.0

        src = int(source) if self.is_live else str(source)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn video: {source}")
        if self.is_live:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width or config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height or config.CAMERA_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 1 else (30.0 if self.is_live else 25.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else 0
        self._idx = 0
        if self.is_live:
            self._cond = threading.Condition()
            self._frame, self._ts, self._seq, self._last_seq = None, 0.0, 0, 0
            self._stop = False
            self._t0 = time.monotonic()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()


    def _loop(self) -> None:
        fails = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if not ok:
                fails += 1
                if fails > 100:
                    break
                time.sleep(0.02)
                continue
            fails = 0
            with self._cond:
                self._frame, self._ts = frame, time.monotonic() - self._t0
                self._seq += 1
                self._cond.notify_all()
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    def read(self) -> Optional[Tuple[np.ndarray, float, int]]:
        """Trả về (frame, timestamp_giây, chỉ_số_khung) hoặc None khi hết video/camera lỗi."""
        if not self.is_live:
            ok, frame = self.cap.read()
            if not ok:
                return None
            ts = self._idx / self.fps
            self._idx += 1
            return frame, ts, self._idx - 1
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._last_seq or self._stop, timeout=5.0)
            if self._seq == self._last_seq:
                return None
            self._last_seq = self._seq
            return self._frame.copy(), self._ts, self._seq

    def release(self) -> None:
        if self.is_live:
            self._stop = True
            self._thread.join(timeout=1.0)
        self.cap.release()
