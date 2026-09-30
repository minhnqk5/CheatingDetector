"""Kiểm thử logic không cần model/GPU: dùng detector giả sinh khung xương tổng hợp.
Chạy:  python tests/test_logic.py   (hoặc pytest)
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from behavior.behavior_detector import estimate_head_yaw  # noqa: E402
from detection.person_detector import Detection, filter_detections  # noqa: E402
from pipeline import ExamMonitor  # noqa: E402


def make_person(cx, cy, yaw=0.0, sw=60.0, score=0.9):
    """Khung xương COCO tổng hợp. yaw>0: mũi lệch sang phải ảnh."""
    torso = 1.4 * sw
    R = config.HEAD_R_TORSO * torso
    hx, hy = cx, cy - 0.7 * sw
    kp = np.zeros((17, 2)); kc = np.full(17, 0.9)
    kp[0] = (hx + R * math.sin(math.radians(yaw)), hy + 0.1 * R)          # mũi
    kp[1] = (kp[0, 0] - 0.35 * R, hy - 0.2 * R); kp[2] = (kp[0, 0] + 0.35 * R, hy - 0.2 * R)
    c = R * math.cos(math.radians(yaw))
    kp[3] = (hx - c, hy); kp[4] = (hx + c, hy)                             # tai trái/phải ảnh
    kc[4] = 0.9 if yaw < 50 else 0.05
    kc[3] = 0.9 if yaw > -50 else 0.05
    kp[5] = (cx - sw / 2, cy); kp[6] = (cx + sw / 2, cy)
    kp[11] = (cx - sw * 0.4, cy + torso); kp[12] = (cx + sw * 0.4, cy + torso)
    kp[7:11] = [(cx - sw, cy + 40), (cx + sw, cy + 40), (cx - sw, cy + 90), (cx + sw, cy + 90)]
    kp[13:] = [(cx - sw * 0.4, cy + 2 * torso)] * 4
    kc[13:] = 0.1
    x1, x2 = cx - sw * 1.1, cx + sw * 1.1
    return Detection(np.array([x1, hy - 1.5 * R, x2, cy + torso * 1.1]), score, kp, kc)


class FakeDetector:
    def __init__(self, fn):
        self.fn = fn
        self.t = 0.0

    def detect(self, frame):
        return filter_detections(self.fn(self.t), frame.shape)

    def detect_objects(self, frame):
        return None


def test_yaw_estimator():
    for yaw in (-40, -20, 0, 20, 40):
        d = make_person(300, 300, yaw)
        est = estimate_head_yaw(d.kp, d.kc)
        assert est is not None and abs(est - yaw) < 3, (yaw, est)
    assert estimate_head_yaw(*[make_person(300, 300, 70).kp, make_person(300, 300, 70).kc]) == config.ONE_EAR_YAW_DEG
    assert estimate_head_yaw(*[make_person(300, 300, -70).kp, make_person(300, 300, -70).kc]) == -config.ONE_EAR_YAW_DEG


def test_ghost_filter():
    good = make_person(300, 300)
    tiny = make_person(100, 100, sw=3)                       # quá nhỏ
    no_skel = make_person(500, 300); no_skel.kc[:] = 0.05      # khung xương rỗng
    dup = make_person(302, 301, score=0.7)                    # trùng lặp
    kept = filter_detections([good, tiny, no_skel, dup], (720, 1280, 3))
    assert len(kept) == 1 and kept[0] is good


def run_sim(script, duration=30.0, fps=30):
    tmp = Path(tempfile.mkdtemp())
    det = FakeDetector(script)
    mon = ExamMonitor("sim", "video", fps, tmp / "events.json", detector=det, db=None)
    frame = np.zeros((720, 1280, 3), np.uint8)
    for i in range(int(duration * fps)):
        det.t = i / fps
        mon.process_frame(frame, det.t)
    mon.finish()
    return mon


def test_end_to_end():
    def script(t):
        dets = [make_person(300, 300, yaw=45 if 8 <= t < 12 else 0)]          # A: quay phải 8-12s
        if not (15 <= t < 19):                                                # B: vắng mặt 15-19s
            dets.append(make_person(700, 320))
        if 22 <= t < 23:                                                      # C: 'người ma' 1s
            dets.append(make_person(1000, 300, score=0.6))
        if 5 <= t < 5.05:                                                     # D: ma 1.5 khung
            dets.append(make_person(1100, 300, score=0.7))
        return dets

    mon = run_sim(script, duration=30)
    ev = mon.logger.events
    ids = {e["person_id"] for e in ev}
    print(*ev, sep="\n")
    turn = [e for e in ev if e["behavior"] == "turn_right"]
    assert len(turn) == 1 and abs(turn[0]["start_time"] - 8) < 0.5 and abs(turn[0]["end_time"] - 12) < 0.5
    left = [e for e in ev if e["behavior"] == "left_seat"]
    assert len(left) == 1 and 15 <= left[0]["start_time"] < 17.5 and 18 <= left[0]["end_time"] <= 19.5, left
    assert left[0]["person_id"] == 2                       # về chỗ vẫn giữ nguyên ID
    assert ids <= {1, 2}
    assert max(t.id for t in mon.tracker.tracks if t.confirmed) == 2   # không có ID cho người ma


def test_walk_away():
    def script(t):
        x = 300 + (0 if t < 10 else min((t - 10) * 150, 450))     # đi ngang 3s rồi đứng ở xa
        return [make_person(x, 300)]
    mon = run_sim(script, duration=20)
    left = [e for e in mon.logger.events if e["behavior"] == "left_seat"]
    assert len(left) == 1 and left[0]["start_time"] < 12.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("OK", name)
