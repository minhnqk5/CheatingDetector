import cv2
from datetime import datetime

from camera.camera import Camera

from detection.face_detector import FaceDetector
from detection.pose_detector import PoseDetector
from detection.object_detector import ObjectDetector

from tracking.person_tracker import PersonTracker

from behavior.behavior_detector import BehaviorDetector
from behavior.temporal_analyzer import TemporalAnalyzer

from logging.event_logger import EventLogger


def main():

    camera = Camera()

    face_detector = FaceDetector()
    pose_detector = PoseDetector()
    object_detector = ObjectDetector()

    tracker = PersonTracker()

    behavior_detector = BehaviorDetector()
    temporal_analyzer = TemporalAnalyzer()

    event_logger = EventLogger()

    print("Exam monitoring started.")

    try:

        while True:

            frame = camera.read()

            # Detection
            faces = face_detector.detect(frame)

            pose = pose_detector.detect(frame)

            objects = object_detector.detect(frame)

            people = tracker.update(faces)

            # Behavior detection
            behaviors = behavior_detector.detect(
                faces,
                pose,
                objects,
                people
            )

            # Temporal analysis
            events = temporal_analyzer.update(
                behaviors,
                datetime.now()
            )

            # Event logging
            for event in events:

                event_logger.log(event)

                print(
                    f"[EVENT] "
                    f"{event['behavior_type']} "
                    f"{event['duration']:.2f}s"
                )

            # Display
            cv2.imshow(
                "Exam Monitoring",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()