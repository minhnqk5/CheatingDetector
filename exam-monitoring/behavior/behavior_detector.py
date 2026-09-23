from config import (
    LOOKING_AWAY_YAW,
    LOOKING_DOWN_PITCH,
    LOOKING_UP_PITCH
)


class BehaviorDetector:

    def __init__(self):
        pass


    def classify_pose(
        self,
        pose
    ):

        if pose is None:
            return None


        yaw = pose["yaw"]
        pitch = pose["pitch"]


        if abs(yaw) > LOOKING_AWAY_YAW:

            return "LOOKING_AWAY"


        if pitch > LOOKING_DOWN_PITCH:

            return "LOOKING_AWAY"


        if pitch < LOOKING_UP_PITCH:

            return "LOOKING_AWAY"


        return None


    def is_identity_pose_reliable(
        self,
        pose
    ):

        if pose is None:
            return False


        yaw = pose["yaw"]
        pitch = pose["pitch"]


        if abs(yaw) > LOOKING_AWAY_YAW:
            return False


        if pitch < LOOKING_UP_PITCH:
            return False


        if pitch > LOOKING_DOWN_PITCH:
            return False


        return True


    def detect(
        self,
        pose=None,
        faces=None
    ):

        behaviors = []


        # =====================================
        # POSE
        # =====================================

        pose_behavior = (
            self.classify_pose(
                pose
            )
        )


        if pose_behavior is not None:

            behaviors.append(
                pose_behavior
            )


        # =====================================
        # NO FACE
        # =====================================

        if not faces:

            behaviors.append(
                "NO_FACE_DETECTED"
            )

            return behaviors


        # =====================================
        # IDENTITY RELIABILITY
        # =====================================

        identity_reliable = (
            self.is_identity_pose_reliable(
                pose
            )
        )


        # =====================================
        # FACE IDENTITY
        # =====================================

        for face in faces:

            identity = face.get(
                "identity",
                "UNKNOWN"
            )


            if identity == "DIFFERENT_PERSON":

                if identity_reliable:

                    behaviors.append(
                        "POSSIBLE_SUBSTITUTION"
                    )


            elif identity == "UNCERTAIN":

                behaviors.append(
                    "IDENTITY_UNCERTAIN"
                )


        return behaviors