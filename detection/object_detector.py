"""(Tuỳ chọn) Phát hiện đồ vật đáng ngờ như điện thoại. Bật bằng config.ENABLE_OBJECT_DETECTOR."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from detection.person_detector import load_yolo


@dataclass
class ObjectDet:
    label: str
    score: float
    bbox: np.ndarray

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2])


class ObjectDetector:
    def __init__(self):
        self.model = load_yolo(config.OBJECT_MODEL)

    def detect(self, frame: np.ndarray) -> list[ObjectDet]:
        res = self.model.predict(
            frame, imgsz=config.IMGSZ, conf=config.OBJECT_CONF, device=config.DEVICE,
            classes=list(config.OBJECT_CLASSES), verbose=False,
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        cls = res.boxes.cls.cpu().numpy().astype(int)
        return [ObjectDet(config.OBJECT_CLASSES.get(int(c), str(c)), float(s), b.astype(float))
                for b, s, c in zip(xyxy, conf, cls)]
