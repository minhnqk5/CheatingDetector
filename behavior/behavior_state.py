from dataclasses import dataclass
from datetime import datetime


@dataclass
class BehaviorState:

    behavior_type: str
    start_time: datetime
    last_seen: datetime
    confidence: float

    @property
    def duration(self):
        return (
            self.last_seen - self.start_time
        ).total_seconds()