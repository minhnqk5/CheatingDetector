"""Phát hiện người + khung xương (YOLO-pose) kèm bộ lọc chống bbox "người ma"."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import config

log = logging.getLogger(__name__)

# 17 keypoint theo chuẩn COCO
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI, \
    L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK = range(17)

SKELETON = [
    (L_SHO, R_SHO), (L_SHO, L_ELB), (L_ELB, L_WRI), (R_SHO, R_ELB), (R_ELB, R_WRI),
    (L_SHO, L_HIP), (R_SHO, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNE), (L_KNE, L_ANK), (R_HIP, R_KNE), (R_KNE, R_ANK),
    (NOSE, L_EYE), (NOSE, R_EYE), (L_EYE, L_EAR), (R_EYE, R_EAR),
]


@dataclass
class Detection:
    bbox: np.ndarray     # (4,) x1,y1,x2,y2
    score: float
    kp: np.ndarray       # (17,2)
    kc: np.ndarray       # (17,) confidence từng keypoint

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])

    def shoulder_width(self):
        if self.kc[L_SHO] >= config.KP_CONF and self.kc[R_SHO] >= config.KP_CONF:
            return float(np.linalg.norm(self.kp[L_SHO] - self.kp[R_SHO]))
        return None

    def anchor_point(self) -> np.ndarray:
        """Điểm đại diện vị trí thân người: trung điểm hai vai (fallback: đỉnh bbox)."""
        if self.kc[L_SHO] >= config.KP_CONF and self.kc[R_SHO] >= config.KP_CONF:
            return (self.kp[L_SHO] + self.kp[R_SHO]) / 2
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2, y1 + 0.25 * (y2 - y1)])

    def raw_scale(self) -> float:
        """Thước đo kích thước người (px) để chuẩn hoá theo phối cảnh."""
        sw = self.shoulder_width()
        base = sw if sw else 0.5 * self.width
        return max(base, 0.3 * self.width, 1.0)


def resolve_weights(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() or p.exists():
        return p
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return config.MODELS_DIR / p.name


def load_yolo(name: str):
    from ultralytics import YOLO
    path = resolve_weights(name)
    try:
        return YOLO(str(path))          # tự tải về models/ nếu chưa có
    except Exception as exc:            # noqa: BLE001
        log.warning("Không nạp được %s (%s); thử tải theo tên", path, exc)
        return YOLO(name)


# ---------------------------------------------------------------- Lọc "người ma"
def _has_valid_skeleton(d: Detection) -> bool:
    vis = d.kc >= config.KP_CONF
    if int(vis[:7].sum()) < config.MIN_UPPER_KEYPOINTS:
        return False
    if not (vis[L_SHO] or vis[R_SHO]):
        return False
    pts = d.kp[vis]
    m = 0.15 * max(d.width, d.height)
    x1, y1, x2, y2 = d.bbox
    inside = (pts[:, 0] >= x1 - m) & (pts[:, 0] <= x2 + m) & (pts[:, 1] >= y1 - m) & (pts[:, 1] <= y2 + m)
    return float(inside.mean()) >= config.KP_INSIDE_RATIO


def _inter_area(a: np.ndarray, b: np.ndarray) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0.0) * max(h, 0.0)


def _dedup(dets: list[Detection]) -> list[Detection]:
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda x: -x.score):
        area_d = max(d.width * d.height, 1e-6)
        dup = False
        for k in kept:
            inter = _inter_area(d.bbox, k.bbox)
            area_k = max(k.width * k.height, 1e-6)
            iou = inter / (area_d + area_k - inter)
            ioa = inter / min(area_d, area_k)
            if iou > config.DEDUP_IOU or ioa > config.DEDUP_IOA:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def filter_detections(dets: list[Detection], frame_shape) -> list[Detection]:
    """Loại các phát hiện vô lý: quá nhỏ/to, sai tỉ lệ, khung xương không hợp lệ, trùng lặp."""
    H, W = frame_shape[:2]
    kept = []
    for d in dets:
        w, h = d.width, d.height
        if d.score < config.DET_CONF_LOW:
            continue
        if h < config.MIN_BOX_HEIGHT_RATIO * H or w < config.MIN_BOX_WIDTH_PX:
            continue
        if w * h > config.MAX_BOX_AREA_RATIO * W * H:
            continue
        if not (config.ASPECT_MIN <= w / max(h, 1e-6) <= config.ASPECT_MAX):
            continue
        if not _has_valid_skeleton(d):
            continue
        kept.append(d)
    return _dedup(kept)


class PersonDetector:
    def __init__(self):
        self.model = load_yolo(config.POSE_MODEL)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        res = self.model.predict(
            frame, imgsz=config.IMGSZ, conf=config.DET_CONF_LOW, iou=config.DET_IOU,
            device=config.DEVICE, verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        kxy = res.keypoints.xy.cpu().numpy()
        kcf = res.keypoints.conf
        kcf = kcf.cpu().numpy() if kcf is not None else np.ones(kxy.shape[:2], dtype=np.float32)
        dets = [Detection(xyxy[i].astype(float), float(conf[i]), kxy[i].astype(float), kcf[i].astype(float))
                for i in range(len(xyxy))]
        return filter_detections(dets, frame.shape)
