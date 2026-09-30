"""Vẽ bbox, ID, khung xương và trạng thái normal/suspicious lên khung hình."""
from __future__ import annotations

import cv2
import numpy as np

from behavior.behavior_state import Status
from detection.person_detector import SKELETON
import config

GREEN, RED, WHITE, YELLOW = (0, 200, 0), (0, 0, 255), (255, 255, 255), (0, 220, 255)


def _text(img, txt, org, color, scale, thick, bg=None):
    (w, h), base = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(org[1])
    if bg is not None:
        cv2.rectangle(img, (x, y - h - base), (x + w + 4, y + base), bg, -1)
    cv2.putText(img, txt, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _fmt(sec: float) -> str:
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def draw_frame(img, visible, lost, states, ts, objects=None):
    H = img.shape[0]
    fs = max(0.45, H / 1080 * 0.7)
    th = max(1, int(round(H / 540)))

    for o in objects or []:
        x1, y1, x2, y2 = o.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), YELLOW, th)
        _text(img, o.label, (x1, y1 - 3), (0, 0, 0), fs * 0.8, th, bg=YELLOW)

    n_susp = 0
    for t in visible:
        st = states.get(t.id)
        susp = st is not None and st.status == Status.SUSPICIOUS
        n_susp += susp
        color = RED if susp else GREEN
        x1, y1, x2, y2 = t.bbox.astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, th + (1 if susp else 0))

        for a, b in SKELETON:                       # khung xương
            if t.kc[a] >= 0.3 and t.kc[b] >= 0.3:
                cv2.line(img, tuple(t.kp[a].astype(int)), tuple(t.kp[b].astype(int)), color, th, cv2.LINE_AA)
        for i in range(17):
            if t.kc[i] >= 0.3:
                cv2.circle(img, tuple(t.kp[i].astype(int)), th + 1, WHITE, -1, cv2.LINE_AA)

        yaw_text = ""

        if st is not None and st.head_yaw is not None:
            yaw_text = f" | yaw={st.head_yaw:.1f}"
        else:
            yaw_text = " | yaw=None"

        label = (
            f"ID {t.id} | "
            + (
                "SUSPICIOUS: " + ",".join(b.value for b in st.active)
                if susp else "normal"
            )
            + yaw_text
        )
        _text(
            img,
            label,
            (x1, max(y1 - 4, 14)),
            WHITE,
            fs,
            th,
            bg=color
        )

        

    for t in lost:       # người đã rời chỗ: đánh dấu tại chỗ ngồi cũ
        st = states.get(t.id)
        if st and st.seat_anchor and st.status == Status.SUSPICIOUS:
            n_susp += 1
            ax, ay = int(st.seat_anchor[0]), int(st.seat_anchor[1])
            r = int(max(st.seat_scale or 20, 10) * 1.5)
            cv2.circle(img, (ax, ay), r, RED, th, cv2.LINE_AA)
            cv2.line(img, (ax - r, ay - r), (ax + r, ay + r), RED, th)
            cv2.line(img, (ax - r, ay + r), (ax + r, ay - r), RED, th)
            _text(img, f"ID {t.id} | LEFT SEAT", (ax - r, max(ay - r - 4, 14)), WHITE, fs, th, bg=RED)

    hud = f"t={_fmt(ts)}  people={len(visible)}  suspicious={n_susp}"
    _text(img, hud, (10, int(28 * fs / 0.7)), WHITE, fs, th, bg=(40, 40, 40))
    return img
