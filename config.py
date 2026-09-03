CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Processing FPS.
PROCESS_FPS = 10

# Behavior thresholds
LOOKING_AWAY_YAW = 30
LOOKING_DOWN_PITCH = 25

MIN_BEHAVIOR_DURATION = 1.0

# YOLO
YOLO_MODEL = "models/yolov8n.pt"
YOLO_CONFIDENCE = 0.5

# Database
DATABASE_URL = "sqlite:///data/exam_monitoring.db"

# Evidence
EVIDENCE_DIR = "data/evidence"

# Web
HOST = "127.0.0.1"
PORT = 8000


CAMERA_FPS = 30
PROCESS_FPS = 10