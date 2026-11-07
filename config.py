"""Cấu hình mặc định; ghi đè đường dẫn/model bằng biến môi trường."""
from pathlib import Path
import os
import torch
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('EXAM_DATA', BASE_DIR / 'data'))
MODEL_PATH = Path(os.environ.get('YOLO_MODEL', BASE_DIR / 'models/yolo26x.pt'))
FACE_DETECTOR_MODEL = Path(os.environ.get('YUNET_MODEL', BASE_DIR / 'models/face_detection_yunet_2026may.onnx'))
UPLOAD_DIR = DATA_DIR / 'uploads'
OUTPUT_DIR = DATA_DIR / 'evidence'
for p in (DATA_DIR, UPLOAD_DIR, OUTPUT_DIR):
    p.mkdir(parents=True, exist_ok=True)
CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = 0 if CUDA_AVAILABLE else 'cpu'
HALF = CUDA_AVAILABLE
# Worker chạy AI từng frame đã lấy. Không bỏ frame thêm lần nữa trong Monitor.
PROCESS_EVERY_N_FRAMES = 1
IMG_SIZE = int(os.environ.get('AI_IMAGE_SIZE', '640'))
CONFIDENCE = 0.10
OBJECT_CONFIDENCE = float(os.environ.get("OBJECT_CONFIDENCE", "0.45"))
FACE_MAX_SIZE = int(os.environ.get("FACE_MAX_SIZE", "1920"))
TRACKER_PATH = str(BASE_DIR / "bytetrack_exam.yaml")
IOU = 0.45
DETECTION_CLASSES = [0, 63, 67, 73]
LOOKING_AWAY_SECONDS = 2.0  # Quay khỏi hướng giữa liên tục 2 giây mới ghi cảnh báo

# Giảm báo nhầm trái/phải khi thí sinh cúi viết
HEAD_YAW_SIGN = -1.0 if os.environ.get('HEAD_YAW_SIGN', '1').strip() == '-1' else 1.0
# Mặc định 1: giữ chiều trái/phải; đặt biến môi trường thành -1 để đảo chiều

HEAD_YAW_THRESHOLD = 20.0       # Góc yaw từ 20° trở lên mới tính là quay trái/phải
HEAD_YAW_CENTER_THRESHOLD = 15.0  # Ngưỡng trở về hướng giữa, giúp nhãn đỡ nhấp nháy
HEAD_YAW_MAX_ABS_PITCH = 25.0   # Cúi/ngẩng quá 25° thì không phân loại trái/phải
HEAD_PITCH_THRESHOLD = 15.0     # Ngưỡng góc cúi/ngẩng
HEAD_YAW_SMOOTH_ALPHA = 0.35    # Mức làm mượt yaw; thấp hơn thì mượt hơn nhưng phản ứng chậm hơn
HEAD_DIRECTION_STABLE_FRAMES = 2  # Hai mẫu AI liên tiếp để giảm nhãn nhấp nháy.
PHONE_CLASS_IDS = {67}
BOOK_CLASS_IDS = {73}
LAPTOP_CLASS_IDS = {63}

# pose: dùng điểm đầu trên bộ xương, không cần detector khuôn mặt YuNet.
HEAD_BACKEND = os.environ.get('HEAD_BACKEND', 'pose').lower()
POSE_MODEL_PATH = Path(os.environ.get('POSE_MODEL', BASE_DIR / 'models/yolo26n-pose.pt'))
POSE_IMAGE_SIZE = int(os.environ.get('POSE_IMAGE_SIZE', '960'))
POSE_KEYPOINT_CONFIDENCE = 0.45
HEAD_TURN_RATIO = 0.22  # Tỷ lệ 2D, không phải góc yaw.

# Chỉ thử crop khi toàn khung chưa xác định được hướng.
POSE_CROP_RETRY = os.environ.get('POSE_CROP_RETRY','1') == '1'
POSE_CROP_SIZE = int(os.environ.get('POSE_CROP_SIZE','640'))
POSE_DEBUG = os.environ.get('POSE_DEBUG','1') == '1'
HEAD_STATE_MAX_GAP = 1.5
