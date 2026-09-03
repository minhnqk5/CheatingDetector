import json
from pathlib import Path


class EventLogger:

    def __init__(self, log_dir="data/logs"):

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.log_file = self.log_dir / "events.jsonl"

    def log(self, event):

        data = {
            "behavior_type":
                event["behavior_type"],

            "start_time":
                event["start_time"].isoformat(),

            "end_time":
                event["end_time"].isoformat(),

            "duration":
                event["duration"],

            "confidence":
                event["confidence"]
        }

        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as f:

            f.write(
                json.dumps(data)
                + "\n"
            )