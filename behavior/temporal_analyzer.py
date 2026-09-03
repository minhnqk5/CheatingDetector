from datetime import datetime

from config import MIN_BEHAVIOR_DURATION
from .behavior_state import BehaviorState


class TemporalAnalyzer:

    def __init__(self):
        self.active = {}

    def update(self, behaviors, now=None):

        if now is None:
            now = datetime.now()

        events = []

        # Start / update behaviors
        for behavior in behaviors:

            if behavior not in self.active:

                self.active[behavior] = BehaviorState(
                    behavior_type=behavior,
                    start_time=now,
                    last_seen=now,
                    confidence=1.0
                )

            else:
                state = self.active[behavior]
                state.last_seen = now

        # Detect ended behaviors
        active_behaviors = set(behaviors)

        for behavior in list(self.active):

            if behavior not in active_behaviors:

                state = self.active.pop(behavior)

                if state.duration >= MIN_BEHAVIOR_DURATION:

                    events.append({
                        "behavior_type":
                            state.behavior_type,

                        "start_time":
                            state.start_time,

                        "end_time":
                            state.last_seen,

                        "duration":
                            state.duration,

                        "confidence":
                            state.confidence
                    })

        return events