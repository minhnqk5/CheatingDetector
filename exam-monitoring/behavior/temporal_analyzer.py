import time
from config import BEHAVIOR_MIN_DURATION


class TemporalAnalyzer:

    def __init__(self):
        self.states = {}

    def update(self, behaviors):

        now = time.time()

        events = []

        current_behaviors = set(behaviors)

        # Bắt đầu timer cho behavior mới
        for behavior in current_behaviors:

            if behavior not in self.states:
                self.states[behavior] = now

        # Kiểm tra các behavior đang tồn tại
        for behavior in list(self.states.keys()):

            if behavior in current_behaviors:
                continue

            start_time = self.states[behavior]
            end_time = now

            duration = end_time - start_time

            if duration >= BEHAVIOR_MIN_DURATION:

                events.append({
                    "behavior": behavior,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration
                })

            del self.states[behavior]

        return events

    def close(self):

        now = time.time()

        events = []

        for behavior, start_time in self.states.items():

            duration = now - start_time

            if duration >= BEHAVIOR_MIN_DURATION:

                events.append({
                    "behavior": behavior,
                    "start_time": start_time,
                    "end_time": now,
                    "duration": duration
                })

        self.states.clear()

        return events