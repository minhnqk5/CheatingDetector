"""Kiểu dữ liệu dùng chung cho phân tích hành vi."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Behavior(str, Enum):
    TURN_LEFT = "turn_left"        # quay đầu sang trái (theo toạ độ ảnh)
    TURN_RIGHT = "turn_right"      # quay đầu sang phải (theo toạ độ ảnh)
    LEFT_SEAT = "left_seat"        # rời khỏi vị trí ngồi
    PHONE = "phone_nearby"         # có điện thoại ở vùng người (tuỳ chọn)


class Status(str, Enum):
    NORMAL = "normal"
    SUSPICIOUS = "suspicious"


@dataclass
class BehaviorEvent:
    person_id: int
    behavior: Behavior
    start_time: float
    end_time: float


@dataclass
class PersonBehaviorState:
    person_id: int
    status: Status = Status.NORMAL
    active: list = field(default_factory=list)      # danh sách Behavior đang diễn ra
    present: bool = True                            # False = đang mất dấu (rời chỗ)
    head_yaw: Optional[float] = None                # góc quay đầu tương đối (độ)
    seat_offset: Optional[float] = None             # độ lệch khỏi chỗ ngồi (đơn vị vai)
    seat_anchor: Optional[tuple] = None
    seat_scale: Optional[float] = None
