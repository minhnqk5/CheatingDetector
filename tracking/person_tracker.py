"""
Tracker nhiều người + bộ nhớ chỗ ngồi tự học.

Chống "người ma":
  * Track mới chỉ được tạo từ detection có điểm cao (DET_CONF_NEW).
  * Track "tentative" phải khớp TRACK_MIN_HITS khung liên tiếp mới được cấp ID/hiển thị;
    bị lỡ 1 khung là huỷ ngay (không tiêu tốn ID).
  * Chỉ vẽ bbox dự đoán tối đa TRACK_DRAW_MISS_FRAMES khung khi bị lỡ.

Chỗ ngồi KHÔNG cố định trong config: mỗi người tự "chốt" chỗ ngồi sau khi ngồi yên
SEAT_CALIB_SEC giây (SeatMemory). Khi người rời đi rồi quay lại đúng chỗ, ID cũ được nhận lại.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from detection.person_detector import Detection


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-6)


class SeatMemory:
    """Tự học chỗ ngồi: khi điểm thân người ổn định đủ lâu -> chốt làm 'anchor'."""

    def __init__(self):
        self.samples: deque = deque()     # (ts, x, y, scale)
        self.anchor: np.ndarray | None = None
        self.scale: float | None = None
        self._last_ts: float | None = None

    @property
    def ready(self) -> bool:
        return self.anchor is not None

    def update(self, ts: float, pos: np.ndarray, scale: float) -> None:
        dt = 0.0 if self._last_ts is None else max(ts - self._last_ts, 0.0)
        self._last_ts = ts
        if self.anchor is None:
            self.samples.append((ts, pos[0], pos[1], scale))
            while self.samples and ts - self.samples[0][0] > config.SEAT_CALIB_SEC:
                self.samples.popleft()
            if len(self.samples) >= 8 and ts - self.samples[0][0] >= config.SEAT_CALIB_SEC * 0.95:
                arr = np.array(self.samples)
                med = np.median(arr[:, 1:3], axis=0)
                sc = float(np.median(arr[:, 3]))
                d = np.linalg.norm(arr[:, 1:3] - med, axis=1)
                if np.percentile(d, 90) < config.SEAT_STABLE_FACTOR * sc:
                    self.anchor, self.scale = med, sc
                    self.samples.clear()
            return
        if np.linalg.norm(pos - self.anchor) < config.SEAT_DRIFT_FACTOR * self.scale:
            a = 1.0 - math.exp(-dt / config.SEAT_DRIFT_TAU_SEC) if dt > 0 else 0.0
            self.anchor = self.anchor + a * (pos - self.anchor)
            self.scale = self.scale + a * (scale - self.scale)


class Track:
    def __init__(self, det: Detection, ts: float):
        self.id: int | None = None
        self.confirmed = False
        self.bbox = det.bbox.astype(float).copy()
        self.kp = det.kp.copy()
        self.kc = det.kc.copy()
        self.score = det.score
        self.score_sum = det.score
        self.hits = 1
        self.misses = 0
        self.first_ts = ts
        self.last_ts = ts
        self.vel = np.zeros(2)
        self.anchor = det.anchor_point()
        self.scale = det.raw_scale()
        self.seat = SeatMemory()

    # -- tiện ích
    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    def predicted_bbox(self, ts: float) -> np.ndarray:
        if self.misses == 0:
            return self.bbox
        shift = self.vel * (ts - self.last_ts)
        lim = 0.5 * self.width
        shift = np.clip(shift, -lim, lim)
        return self.bbox + np.array([shift[0], shift[1], shift[0], shift[1]])

    def update(self, det: Detection, ts: float) -> None:
        dt = max(ts - self.last_ts, 1e-3)
        old_c = self.center
        a = config.BOX_SMOOTH
        self.bbox = a * det.bbox + (1 - a) * self.bbox
        self.vel = 0.5 * self.vel + 0.5 * (self.center - old_c) / dt
        self.kp, self.kc, self.score = det.kp.copy(), det.kc.copy(), det.score
        self.score_sum += det.score
        self.hits += 1
        self.misses = 0
        self.last_ts = ts
        self.anchor = 0.5 * self.anchor + 0.5 * det.anchor_point()
        s = det.raw_scale()                      # tăng nhanh, giảm chậm (tránh co khi người xoay)
        self.scale += (0.3 if s > self.scale else 0.02) * (s - self.scale)
        self.seat.update(ts, self.anchor, self.scale)

    def revive_from(self, other: "Track") -> None:
        """Nhận lại track đã mất bằng dữ liệu của track tentative mới."""
        self.bbox, self.kp, self.kc = other.bbox.copy(), other.kp.copy(), other.kc.copy()
        self.score = other.score
        self.score_sum += other.score_sum
        self.hits += other.hits
        self.misses = 0
        self.last_ts = other.last_ts
        self.vel = other.vel.copy()
        self.anchor = other.anchor.copy()
        self.seat.update(other.last_ts, self.anchor, self.scale)


@dataclass
class TrackerOutput:
    visible: list = field(default_factory=list)     # track để vẽ (đã xác nhận, còn thấy)
    lost: list = field(default_factory=list)        # track đã xác nhận nhưng đang mất dấu
    removed_ids: list = field(default_factory=list)  # ID bị xoá hẳn ở khung này


class PersonTracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, dets: list[Detection], ts: float) -> TrackerOutput:
        removed: list[int] = []
        cands = [t for t in self.tracks if ts - t.last_ts <= config.TRACK_MAX_MISS_SEC]
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()

        if cands and dets:
            tb = np.array([t.predicted_bbox(ts) for t in cands])
            db = np.array([d.bbox for d in dets])
            iou = iou_matrix(tb, db)
            tc = np.stack([(tb[:, 0] + tb[:, 2]) / 2, (tb[:, 1] + tb[:, 3]) / 2], 1)
            dc = np.stack([(db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2], 1)
            wmax = np.maximum((tb[:, 2] - tb[:, 0])[:, None], (db[:, 2] - db[:, 0])[None, :])
            dist = np.linalg.norm(tc[:, None] - dc[None], axis=2) / np.maximum(wmax, 1.0)
            allowed = (iou >= config.TRACK_IOU_GATE) | (dist <= config.TRACK_CENTER_GATE)
            cost = 0.7 * (1 - iou) + 0.3 * np.minimum(dist, 1.5) / 1.5
            cost[~allowed] = 1e6
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < 1e5:
                    cands[r].update(dets[c], ts)
                    matched_tracks.add(id(cands[r]))
                    matched_dets.add(c)

        # Track không khớp
        for t in list(self.tracks):
            if id(t) in matched_tracks:
                continue
            if not t.confirmed:
                self.tracks.remove(t)          # tentative bị lỡ -> huỷ ngay
            else:
                t.misses += 1

        # Track mới (chỉ từ detection điểm cao)
        for i, d in enumerate(dets):
            if i not in matched_dets and d.score >= config.DET_CONF_NEW:
                self.tracks.append(Track(d, ts))

        # Xác nhận
        for t in list(self.tracks):
            if (not t.confirmed and t.hits >= config.TRACK_MIN_HITS
                    and t.score_sum / t.hits >= config.TRACK_MIN_AVG_SCORE):
                self._confirm(t)

        # Xoá track mất quá lâu
        for t in list(self.tracks):
            if not t.confirmed:
                continue
            gone = ts - t.last_ts
            limit = config.LOST_KEEP_SEC if t.seat.ready else config.LOST_NO_SEAT_KEEP_SEC
            if gone > limit:
                self.tracks.remove(t)
                removed.append(t.id)

        out = TrackerOutput(removed_ids=removed)
        for t in self.tracks:
            if not t.confirmed:
                continue
            if t.misses <= config.TRACK_DRAW_MISS_FRAMES and ts - t.last_ts <= config.TRACK_MAX_MISS_SEC:
                out.visible.append(t)
            elif ts - t.last_ts > config.TRACK_MAX_MISS_SEC:
                out.lost.append(t)
        return out

    # ------------------------------------------------------------------ nội bộ
    def _confirm(self, t: Track) -> None:
        lost = self._find_recoverable(t)
        if lost is not None:
            lost.revive_from(t)
            self.tracks.remove(t)
            return
        t.id = self._next_id
        self._next_id += 1
        t.confirmed = True

    def _find_recoverable(self, new: Track):
        best, best_d = None, None
        for c in self.tracks:
            if not c.confirmed or not c.seat.ready or c.last_ts >= new.last_ts:
                continue
            if new.last_ts - c.last_ts <= config.TRACK_MAX_MISS_SEC:
                continue
            d = float(np.linalg.norm(new.anchor - c.seat.anchor)) / c.seat.scale
            if d <= config.RECOVER_DIST_FACTOR and (best_d is None or d < best_d):
                best, best_d = c, d
        return best
