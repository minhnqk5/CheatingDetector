import cv2
from datetime import datetime

from web.app import create_app

app = create_app()

from camera.camera import Camera

from detection.face_detector import FaceDetector
from detection.pose_detector import PoseDetector
from detection.object_detector import ObjectDetector

from tracking.tracker import PersonTracker

from behavior.behavior_detector import BehaviorDetector
from behavior.temporal_analyzer import TemporalAnalyzer

from event_logging.event_logger import EventLogger



if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

