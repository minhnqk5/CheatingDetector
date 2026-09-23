import json
import os
from datetime import datetime


class EventLogger:

    def __init__(self, log_file="exam-monitoring/data/logs/events.json"):

        self.log_file = log_file

        os.makedirs(
            os.path.dirname(log_file),
            exist_ok=True
        )

        # Create file if not exists
        if not os.path.exists(self.log_file):

            with open(
                self.log_file,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    [],
                    f,
                    indent=4
                )

    # ======================================================
    # Log event
    # ======================================================

    def log(self, event):

        if event is None:
            return

        data = self._load()

        log_entry = {
            "behavior": event["behavior"],

            "start_time": datetime.fromtimestamp(
                event["start_time"]
            ).strftime("%Y-%m-%d %H:%M:%S"),

            "end_time": datetime.fromtimestamp(
                event["end_time"]
            ).strftime("%Y-%m-%d %H:%M:%S"),

            "duration": round(
                event["duration"],
                2
            )
        }

        data.append(log_entry)

        self._save(data)

        # Terminal
        print(
            f"\n[EVENT] "
            f"{log_entry['behavior']} | "
            f"duration={log_entry['duration']}s"
        )

    # ======================================================
    # Load
    # ======================================================

    def _load(self):

        with open(
            self.log_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    # ======================================================
    # Save
    # ======================================================

    def _save(self, data):

        with open(
            self.log_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False
            )