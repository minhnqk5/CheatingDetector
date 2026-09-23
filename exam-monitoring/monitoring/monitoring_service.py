import time

import numpy as np

from behavior.behavior_detector import BehaviorDetector
from behavior.temporal_analyzer import TemporalAnalyzer
from detection.face_detector import FaceDetector
from detection.pose_detector import PoseDetector
from event_logging.event_logger import EventLogger


class MonitoringService:

    def __init__(
        self,
        reference_embedding
    ):

        self.face_detector = FaceDetector(
            similarity_threshold=0.45,
            uncertain_threshold=0.35,
            detect_every=3
        )

        self.face_detector.set_reference_embedding(reference_embedding)

        self.pose_detector = PoseDetector(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.behavior_detector = (
            BehaviorDetector()
        )

        self.temporal_analyzer = (
            TemporalAnalyzer()
        )

        self.event_logger = EventLogger()

        self.started_at = time.time()


    def process_frame(self, frame):

        # =====================================
        # 1. FACE
        # =====================================

        faces = self.face_detector.detect(
            frame
        )


        # =====================================
        # 2. POSE
        # =====================================

        pose = self.pose_detector.detect(
            frame
        )


        # =====================================
        # 3. BEHAVIOR
        # =====================================

        behaviors = (
            self.behavior_detector.detect(
                pose=pose,
                faces=faces
            )
        )


        # =====================================
        # 4. TEMPORAL ANALYSIS
        # =====================================

        completed_events = (
            self.temporal_analyzer.update(
                behaviors
            )
        )


        # =====================================
        # 5. LOG EVENTS
        # =====================================

        logged_events = []

        for event in completed_events:


            if not self._is_suspicious_event(
                event["behavior"]
            ):
                continue

            self.event_logger.log(
                event
            )

            logged_events.append(
                self._serialize_event(event)
            )


        # =====================================
        # 6. CURRENT STATUS
        # =====================================

        status = self._build_status(
            faces,
            pose,
            behaviors
        )


        return {
            "status": status,
            "events": logged_events
        }


    def _is_suspicious_event(
        self,
        behavior
    ):

        return behavior in {
            "LOOKING_AWAY",
            "POSSIBLE_SUBSTITUTION"
        }


    def _build_status(
        self,
        faces,
        pose,
        behaviors
    ):

        if not faces:

            return {
                "face": "NO_FACE",
                "identity": None,
                "similarity": None,
                "pose": None,
                "behaviors": []
            }


        primary_face = faces[0]

        identity = primary_face.get(
            "identity"
        )

        similarity = primary_face.get(
            "similarity"
        )


        pose_name = (
            self._get_pose_name(
                pose
            )
        )


        return {
            "face": "DETECTED",
            "identity": identity,
            "similarity": similarity,
            "pose": pose_name,
            "behaviors": [
                behavior
                for behavior in behaviors
                if behavior
                not in {
                    "NO_FACE_DETECTED"
                }
            ]
        }


    def _get_pose_name(
        self,
        pose
    ):

        if pose is None:
            return None

        yaw = pose["yaw"]
        pitch = pose["pitch"]

        if abs(yaw) > 0.3:

            if yaw > 0:
                return "LOOKING_LEFT"

            return "LOOKING_RIGHT"


        if pitch > 0.5:
            return "LOOKING_DOWN"


        if pitch < 0.3:
            return "LOOKING_UP"


        return "FORWARD"


    def _serialize_event(
        self,
        event
    ):

        return {
            "behavior": event["behavior"],
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "duration": round(
                event["duration"],
                2
            )
        }


    def close(self):

        final_events = (
            self.temporal_analyzer.close()
        )


        for event in final_events:

            if self._is_suspicious_event(
                event["behavior"]
            ):

                self.event_logger.log(
                    event
                )


        self.pose_detector.close()

        self.face_detector.close()