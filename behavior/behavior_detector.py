from config import (
    LOOKING_AWAY_YAW,
    LOOKING_DOWN_PITCH
)


class BehaviorDetector:

    def detect(
        self,
        faces,
        pose,
        objects,
        people
    ):

        behaviors = []

        # No face
        if len(faces) == 0:
            behaviors.append("NO_FACE")

        # Multiple people
        if len(people) > 1:
            behaviors.append("MULTIPLE_PERSON")

        # Head direction
        if pose:

            if abs(pose["yaw"]) > LOOKING_AWAY_YAW:
                behaviors.append("LOOKING_AWAY")

            if pose["pitch"] > LOOKING_DOWN_PITCH:
                behaviors.append("LOOKING_DOWN")

        # Objects
        for obj in objects:

            if obj["label"] == "cell phone":
                behaviors.append("PHONE_DETECTED")

        return behaviors