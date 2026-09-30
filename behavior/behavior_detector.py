"""
Phát hiện hành vi từ khung xương:
  * quay đầu trái/phải  (góc yaw ước lượng từ mũi/tai + tỉ lệ thân người dùng vai & hông)
  * rời khỏi vị trí     (lệch khỏi chỗ ngồi tự học của chính người đó)
  * (tuỳ chọn) điện thoại trong vùng người

Ước lượng yaw đầu
-----------------
Đầu xem như hình cầu bán kính R. Khi nhìn thẳng, mũi nằm giữa hai tai; khi quay góc θ,
mũi lệch khỏi trung điểm hai tai một đoạn R·sinθ  =>  θ = asin((mũi_x - tai_giữa_x) / R).
R phải là thước đo KHÔNG co lại khi người xoay: chiều dài thân (trung điểm vai -> trung điểm hông)
không bị co theo yaw, còn bề rộng vai thì co (cosθ) nên chỉ dùng làm dự phòng khi hông bị che.
Chỉ thấy 1 tai (tai kia conf rất thấp) => quay mạnh (~ONE_EAR_YAW_DEG).
Góc cuối cùng được trừ đi "hướng nhìn quen thuộc" của từng người (baseline) để camera lệch
hoặc người ngồi chéo không bị báo nhầm.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import config
from behavior.behavior_state import Behavior, BehaviorEvent, PersonBehaviorState, Status
from behavior.temporal_analyzer import TemporalAnalyzer
from detection.person_detector import (
    L_EAR, L_EYE, L_HIP, L_SHO, NOSE, R_EAR, R_EYE, R_HIP, R_SHO
)


# ------------------------------------------------------------ Ước lượng yaw 2D
def estimate_head_yaw(kp: np.ndarray, kc: np.ndarray) -> Optional[float]:
    """
    Ước lượng góc quay đầu tương đối với hướng thân, hoàn toàn từ 2D.

    Ý tưởng:
      1. face_dir = mũi - trung điểm hai mắt: hướng mặt trong ảnh.
      2. body_forward = pháp tuyến của đoạn hai vai, chọn phía hướng về mũi.
      3. yaw = góc có dấu giữa body_forward và face_dir.

    Điểm quan trọng: không lấy trung điểm hai tai làm "mốc thế giới".
    Khi người ngồi chéo 30-60 độ, cả vai và đầu cùng xoay theo camera;
    so sánh đầu với thân giúp triệt bớt phần xoay do tư thế ngồi.
    """
    thr = config.KP_CONF

    if not all(kc[i] >= thr for i in (NOSE, L_EYE, R_EYE, L_SHO, R_SHO)):
        return None

    eye_mid = (kp[L_EYE] + kp[R_EYE]) * 0.5
    shoulder_mid = (kp[L_SHO] + kp[R_SHO]) * 0.5

    face_vec = kp[NOSE] - eye_mid
    shoulder_vec = kp[R_SHO] - kp[L_SHO]

    face_len = float(np.linalg.norm(face_vec))
    eye_dist = float(np.linalg.norm(kp[R_EYE] - kp[L_EYE]))
    shoulder_len = float(np.linalg.norm(shoulder_vec))

    # Loại các frame mà mặt quá nhỏ / keypoint quá nhiễu.
    if face_len < max(1.5, 0.10 * eye_dist) or shoulder_len < 3.0:
        return None

    # Hai pháp tuyến của đường vai.
    n1 = np.array([-shoulder_vec[1], shoulder_vec[0]], dtype=float)
    n2 = -n1

    # Chọn pháp tuyến hướng về phía mũi.
    nose_from_shoulders = kp[NOSE] - shoulder_mid
    body_forward = n1 if np.dot(n1, nose_from_shoulders) >= np.dot(n2, nose_from_shoulders) else n2

    body_forward /= max(np.linalg.norm(body_forward), 1e-6)
    face_vec /= max(face_len, 1e-6)

    # atan2(cross_z, dot): dương = quay về bên phải theo hệ tọa độ ảnh.
    cross = body_forward[0] * face_vec[1] - body_forward[1] * face_vec[0]
    dot = float(np.clip(np.dot(body_forward, face_vec), -1.0, 1.0))
    angle = math.degrees(math.atan2(cross, dot))

    # Chỉ quan tâm quay trái/phải, không để góc > 90° làm đảo trạng thái.
    return float(np.clip(angle, -90.0, 90.0))

# --------------------------------------------------------------- Trạng thái mỗi người
@dataclass
class _Runtime:
    yaw_buf: deque = field(
        default_factory=lambda: deque(
            maxlen=config.YAW_MEDIAN_WINDOW
        )
    )

    last_ts: Optional[float] = None

    left_on: bool = False
    right_on: bool = False
    away_on: bool = False

    rel_yaw: Optional[float] = None

def _body_frame(t):
    """
    Tạo hệ trục 2D của thân người.

    right:
        hướng sang bên phải của người, dựa trên 2 vai.

    up:
        hướng từ hông lên vai nếu có đủ hông,
        nếu không thì lấy pháp tuyến của đường vai.

    Trả về:
        body_right, body_up, scale
    """

    kp = t.kp
    kc = t.kc
    thr = config.KP_CONF

    # --------------------------------------------------------
    # 1. Hai vai bắt buộc phải có
    # --------------------------------------------------------
    if kc[L_SHO] < thr or kc[R_SHO] < thr:
        return None

    left_sh = kp[L_SHO].astype(float)
    right_sh = kp[R_SHO].astype(float)

    shoulder_vec = right_sh - left_sh
    shoulder_len = float(np.linalg.norm(shoulder_vec))

    if shoulder_len < config.MIN_SHOULDER_PX:
        return None

    shoulder_right = shoulder_vec / shoulder_len

    # --------------------------------------------------------
    # 2. Hướng "up" của thân
    # --------------------------------------------------------
    up = None

    if (
        kc[L_HIP] >= config.HIP_CONF
        and kc[R_HIP] >= config.HIP_CONF
    ):
        hip_mid = (
            kp[L_HIP].astype(float)
            + kp[R_HIP].astype(float)
        ) / 2.0

        shoulder_mid = (
            left_sh + right_sh
        ) / 2.0

        torso_vec = shoulder_mid - hip_mid
        torso_len = float(np.linalg.norm(torso_vec))

        if torso_len >= config.MIN_TORSO_PX:
            # Không lấy thân nếu hình học quá kỳ quặc.
            # Trong ảnh, vai thường nằm phía trên hông.
            if torso_vec[1] < 0:
                up = torso_vec / torso_len

    # --------------------------------------------------------
    # 3. Không có hông -> pháp tuyến đường vai
    # --------------------------------------------------------
    if up is None:
        # vector vuông góc shoulder_right
        up = np.array(
            [-shoulder_right[1], shoulder_right[0]],
            dtype=float,
        )

        # ép hướng lên trên ảnh
        if up[1] > 0:
            up = -up

    # --------------------------------------------------------
    # 4. scale
    # --------------------------------------------------------
    scale = max(
        shoulder_len,
        0.5 * t.width,
        1.0,
    )

    return shoulder_right, up, scale


def _relative_head_yaw(t):
    """
    Ước lượng yaw của đầu RELATIVE với thân.

    Không đo:
        head -> camera

    Mà đo:
        head direction -> body direction

    Vì vậy người ngồi chéo camera sẽ không tự động bị
    coi là TURN_LEFT / TURN_RIGHT.

    Giá trị:
        > 0 : đầu lệch về phía phải của người
        < 0 : đầu lệch về phía trái của người

    Trả về:
        yaw_deg hoặc None
    """

    kp = t.kp
    kc = t.kc
    thr = config.KP_CONF

    # --------------------------------------------------------
    # Body frame
    # --------------------------------------------------------
    frame = _body_frame(t)

    if frame is None:
        return None

    body_right, body_up, scale = frame

    # --------------------------------------------------------
    # Nose
    # --------------------------------------------------------
    if kc[NOSE] < thr:
        return None

    nose = kp[NOSE].astype(float)

    # --------------------------------------------------------
    # Cue 1: nose so với midpoint của 2 mắt
    #
    # Đây là cue quan trọng nhất.
    #
    # Khi đầu thẳng theo thân:
    #
    # eye_mid
    #    |
    #    |
    #   nose
    #
    # vector này gần song song body_up.
    #
    # Khi quay đầu:
    #
    # eye_mid ---- nose
    #
    # xuất hiện thành phần theo body_right.
    # --------------------------------------------------------
    eye_valid = (
        kc[L_EYE] >= thr
        and kc[R_EYE] >= thr
    )

    if not eye_valid:
        return None

    left_eye = kp[L_EYE].astype(float)
    right_eye = kp[R_EYE].astype(float)

    eye_mid = (left_eye + right_eye) / 2.0

    eye_dist = float(
        np.linalg.norm(left_eye - right_eye)
    )

    if eye_dist < config.MIN_EYE_PX:
        return None

    face_vec = nose - eye_mid

    # Thành phần ngang theo thân
    lateral = float(face_vec @ body_right)

    # Thành phần dọc
    vertical = float(face_vec @ body_up)

    # --------------------------------------------------------
    # Không dùng pixel tuyệt đối.
    #
    # Chuẩn hóa theo khoảng cách 2 mắt.
    # Điều này rất quan trọng khi người gần/xa camera.
    # --------------------------------------------------------
    ratio = lateral / max(eye_dist, 1.0)

    # --------------------------------------------------------
    # Chuyển ratio -> góc tương đối.
    #
    # Không coi ratio là sin chính xác tuyệt đối.
    # Đây là "pseudo angle" ổn định cho 2D.
    # --------------------------------------------------------
    yaw_deg = math.degrees(
        math.atan2(
            ratio,
            config.HEAD_FORWARD_RATIO
        )
    )

    # giới hạn nhiễu cực đoan
    yaw_deg = float(
        np.clip(
            yaw_deg,
            -89.0,
            89.0,
        )
    )

    # --------------------------------------------------------
    # Cue phụ: mũi so với midpoint vai
    #
    # Chỉ dùng nhẹ để củng cố.
    # --------------------------------------------------------
    shoulder_mid = (
        kp[L_SHO].astype(float)
        + kp[R_SHO].astype(float)
    ) / 2.0

    nose_body = nose - shoulder_mid

    shoulder_lateral = float(
        nose_body @ body_right
    )

    shoulder_ratio = (
        shoulder_lateral / max(scale, 1.0)
    )

    shoulder_deg = math.degrees(
        math.atan2(
            shoulder_ratio,
            config.HEAD_SHOULDER_FORWARD_RATIO
        )
    )

    shoulder_deg = float(
        np.clip(
            shoulder_deg,
            -89.0,
            89.0,
        )
    )

    # --------------------------------------------------------
    # Fuse:
    #
    # face cue mạnh
    # shoulder cue nhẹ
    # --------------------------------------------------------
    yaw = (
        yaw_deg * config.HEAD_EYE_WEIGHT
        + shoulder_deg * config.HEAD_SHOULDER_WEIGHT
    )

    return float(
        np.clip(yaw, -89.0, 89.0)
    )


class BehaviorDetector:
    def __init__(self):
        self.analyzer = TemporalAnalyzer()
        self.rt: dict[int, _Runtime] = {}

    # ---------------------------------------------------------------- chính
    def update(self, visible, lost, ts: float, objects=None):
        """Trả về (states: {pid: PersonBehaviorState}, events đã kết thúc ở khung này)."""
        events: list[BehaviorEvent] = []
        states: dict[int, PersonBehaviorState] = {}
        phone_ids = self._assign_phones(visible, objects) if objects is not None else None

        def emit(ev):
            if ev:
                events.append(ev)

        for t in visible:
            pid = t.id
            rt = self.rt.setdefault(pid, _Runtime())
            if t.misses > 0:                       # đang ngoại suy, không có dữ liệu mới
                states[pid] = self._state(t, rt, present=True)
                continue
            dt = 0.0 if rt.last_ts is None else max(ts - rt.last_ts, 0.0)
            rt.last_ts = ts

            left, right, rt.rel_yaw = self._turn_flags(t, rt, ts, dt)
            emit(self.analyzer.update(pid, Behavior.TURN_LEFT, ts, left))
            emit(self.analyzer.update(pid, Behavior.TURN_RIGHT, ts, right))

            away = None
            if t.seat.ready:
                off = self._seat_offset(t)
                if off >= config.LEAVE_ENTER_FACTOR:
                    rt.away_on = True
                elif off <= config.LEAVE_EXIT_FACTOR:
                    rt.away_on = False
                away = rt.away_on
            emit(self.analyzer.update(pid, Behavior.LEFT_SEAT, ts, away))

            if phone_ids is not None:
                emit(self.analyzer.update(pid, Behavior.PHONE, ts, pid in phone_ids))
            states[pid] = self._state(t, rt, present=True)

        for t in lost:                              # mất dấu: nếu đã có chỗ ngồi -> coi là đang vắng
            pid = t.id
            rt = self.rt.setdefault(pid, _Runtime())
            emit(self.analyzer.update(pid, Behavior.LEFT_SEAT, ts, True if t.seat.ready else None))
            for b in (Behavior.TURN_LEFT, Behavior.TURN_RIGHT):
                emit(self.analyzer.update(pid, b, ts, None))
            states[pid] = self._state(t, rt, present=False)
        return states, events

    def drop_person(self, pid: int) -> list[BehaviorEvent]:
        self.rt.pop(pid, None)
        return self.analyzer.close_person(pid)

    def flush(self) -> list[BehaviorEvent]:
        return self.analyzer.close_all()




    def _turn_flags(
        self,
        t,
        rt: _Runtime,
        ts: float,
        dt: float,
    ):
        """
        Detect turn dựa trên HEAD YAW RELATIVE TO BODY.

        Không có baseline.
        Không phụ thuộc người đang ngồi chéo bao nhiêu độ
        so với camera.
        """

        yaw = _relative_head_yaw(t)

        if yaw is not None:
            rt.yaw_buf.append(yaw)

        if not rt.yaw_buf:
            return None, None, rt.rel_yaw

        # median nhỏ để chống rung YOLO
        rel = float(
            np.median(rt.yaw_buf)
        )

        rt.rel_yaw = rel

        # --------------------------------------------------------
        # Hysteresis
        # --------------------------------------------------------

        # LEFT
        if rel <= -config.YAW_ENTER_DEG:
            rt.left_on = True
        elif rel >= -config.YAW_EXIT_DEG:
            rt.left_on = False

        # RIGHT
        if rel >= config.YAW_ENTER_DEG:
            rt.right_on = True
        elif rel <= config.YAW_EXIT_DEG:
            rt.right_on = False

        return (
            rt.left_on,
            rt.right_on,
            rel,
        )



    

    @staticmethod
    def _seat_offset(t) -> float:
        return float(np.linalg.norm(t.anchor - t.seat.anchor)) / t.seat.scale

    def _state(self, t, rt: _Runtime, present: bool) -> PersonBehaviorState:
        active = [b for b in Behavior if self.analyzer.is_active(t.id, b)]
        st = PersonBehaviorState(
            person_id=t.id,
            status=Status.SUSPICIOUS if active else Status.NORMAL,
            active=active, present=present, head_yaw=rt.rel_yaw,
        )
        if t.seat.ready:
            st.seat_anchor = (float(t.seat.anchor[0]), float(t.seat.anchor[1]))
            st.seat_scale = float(t.seat.scale)
            if present:
                st.seat_offset = self._seat_offset(t)
        return st

    @staticmethod
    def _assign_phones(visible, objects) -> set:
        ids = set()
        for o in objects:
            c = o.center
            best, best_d = None, None
            for t in visible:
                x1, y1, x2, y2 = t.bbox
                mx, my = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
                if x1 - mx <= c[0] <= x2 + mx and y1 - my <= c[1] <= y2 + my:
                    d = float(np.linalg.norm(c - t.center))
                    if best_d is None or d < best_d:
                        best, best_d = t.id, d
            if best is not None:
                ids.add(best)
        return ids
