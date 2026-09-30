"""
Cấu hình trung tâm của hệ thống giám sát phòng thi.

Quy ước: mọi ngưỡng thời gian tính bằng GIÂY (không phụ thuộc FPS).
Không có toạ độ chỗ ngồi cố định nào ở đây: vị trí ngồi của từng người
được hệ thống tự học (xem tracking/person_tracker.py -> SeatMemory).
"""
from pathlib import Path

# ----------------------------------------------------------------- Đường dẫn
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
EVENTS_DIR = DATA_DIR / "events"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "exam_monitoring.db"
ENABLE_DB = True

# --------------------------------------------------------------------- Model
# Ảnh toàn cảnh phòng thi -> người nhỏ, nên dùng model >= "s" và imgsz lớn.
# Gợi ý: yolov8n-pose (nhanh, kém) < yolov8s-pose < yolov8m-pose < yolov8l-pose (chậm, tốt)
POSE_MODEL = "yolov8m-pose.pt"
IMGSZ = 960              # tăng lên 1280 nếu người trong ảnh rất nhỏ
DEVICE = None            # None = tự chọn; hoặc "cpu", "cuda:0", "mps"

# Bộ phát hiện đồ vật (điện thoại) - TẮT mặc định
ENABLE_OBJECT_DETECTOR = False
OBJECT_MODEL = "yolov8s.pt"
OBJECT_CONF = 0.35
OBJECT_CLASSES = {67: "cell phone"}    # COCO id -> tên
PHONE_MIN_SEC = 1.0
PHONE_RELEASE_SEC = 1.5

# ------------------------------------------------------------------- Camera
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# -------------------------------------------- Chống bbox "người ma" (detection)
DET_CONF_LOW = 0.30          # ngưỡng tối thiểu để CẬP NHẬT một track đã có
DET_CONF_NEW = 0.55          # ngưỡng cao hơn để TẠO track mới (hysteresis)
DET_IOU = 0.5                # NMS của YOLO
DEDUP_IOU = 0.6              # loại bbox trùng lặp sau YOLO
DEDUP_IOA = 0.85             # loại bbox nằm gần hết bên trong bbox khác
MIN_BOX_HEIGHT_RATIO = 0.04  # chiều cao bbox tối thiểu so với chiều cao khung hình
MIN_BOX_WIDTH_PX = 12
MAX_BOX_AREA_RATIO = 0.5
ASPECT_MIN, ASPECT_MAX = 0.25, 2.2       # w/h hợp lệ của người (ngồi/đứng)
KP_CONF = 0.35               # keypoint có conf >= giá trị này coi là "nhìn thấy"
KP_HIDDEN_CONF = 0.15        # conf < giá trị này coi là "chắc chắn bị che"
MIN_UPPER_KEYPOINTS = 4      # số keypoint phần trên (mũi/mắt/tai/vai) tối thiểu (trong 7)
KP_INSIDE_RATIO = 0.8        # tỉ lệ keypoint phải nằm trong bbox

# ------------------------------------------------------------------ Tracking
TRACK_MIN_HITS = 5           # cần khớp >= N khung liên tiếp mới được cấp ID & hiển thị
TRACK_MIN_AVG_SCORE = 0.50   # điểm trung bình tối thiểu khi xác nhận
TRACK_MAX_MISS_SEC = 1.0     # quá thời gian này không thấy -> coi là "mất dấu"
TRACK_DRAW_MISS_FRAMES = 2   # chỉ vẽ bbox dự đoán tối đa N khung khi bị lỡ
TRACK_IOU_GATE = 0.10
TRACK_CENTER_GATE = 0.5      # (khoảng cách tâm / bề rộng bbox) tối đa để ghép
BOX_SMOOTH = 0.6             # làm mượt bbox (1 = không mượt)
LOST_KEEP_SEC = 300.0        # giữ track đã mất để nhận lại đúng ID khi về chỗ
LOST_NO_SEAT_KEEP_SEC = 3.0  # track chưa học được chỗ ngồi thì bỏ nhanh
RECOVER_DIST_FACTOR = 1.2    # bán kính nhận lại ID quanh chỗ ngồi (đơn vị: vai)

# -------------------------------------------------- Chỗ ngồi tự học (không cố định)
SEAT_CALIB_SEC = 3.0         # đứng yên bao lâu thì "chốt" chỗ ngồi
SEAT_STABLE_FACTOR = 0.5     # độ dao động cho phép khi chốt (đơn vị: vai)
SEAT_DRIFT_FACTOR = 0.6      # chỉ cập nhật chỗ ngồi khi còn gần chỗ (đơn vị: vai)
SEAT_DRIFT_TAU_SEC = 30.0    # chỗ ngồi trôi chậm theo tư thế ngồi

# ---------------------------------------------------------------- Hành vi
# Quay đầu trái/phải (theo toạ độ ẢNH: trái = mũi lệch về bên trái khung hình)
# ----------------------------------------------------------------
# Head yaw relative to body
# ----------------------------------------------------------------

# YOLO keypoint phải đủ lớn mới tính hình học.
MIN_SHOULDER_PX = 10.0
MIN_TORSO_PX = 12.0
MIN_EYE_PX = 4.0

# Hông dùng để tạo trục thân nếu đủ confidence.
HIP_CONF = 0.35
USE_HIPS = True

# ---------------------------------------------------------------
# 2D head-vs-body yaw
# ---------------------------------------------------------------

# Cue mắt -> mũi là cue chính.
HEAD_EYE_WEIGHT = 0.80

# Cue mũi -> vai là cue phụ.
HEAD_SHOULDER_WEIGHT = 0.20

# Giá trị càng nhỏ thì yaw càng nhạy.
#
# atan2(ratio, forward_ratio)
#
# Ví dụ:
# ratio = 0.5
# forward = 0.8
# => khoảng 32°
HEAD_FORWARD_RATIO = 0.80

HEAD_SHOULDER_FORWARD_RATIO = 0.55

# ---------------------------------------------------------------
# Temporal smoothing
# ---------------------------------------------------------------

# 1 = phản ứng nhanh nhất.
# 3 = chống rung nhẹ.
# 5 = ổn định hơn nhưng có delay hơn.
YAW_MEDIAN_WINDOW = 3










# THRESHOLD
YAW_ENTER_DEG = 25.0
YAW_EXIT_DEG = 20.0










# Thời gian phải quay liên tục
TURN_MIN_SEC = 0.2
TURN_RELEASE_SEC = 0.6


# Rời vị trí
LEAVE_ENTER_FACTOR = 1.6     # lệch khỏi chỗ ngồi > 1.6 bề rộng vai
LEAVE_EXIT_FACTOR = 1.0
LEAVE_MIN_SEC = 2.0
LEAVE_RELEASE_SEC = 1.0

NO_INFO_HOLD_SEC = 2.0       # mất thông tin (vd. che mặt) giữ nguyên trạng thái tối đa

# ---------------------------------------------------------------------- Web
WEB_HOST = "0.0.0.0"
WEB_PORT = 5000
MAX_UPLOAD_MB = 4096
JPEG_QUALITY = 70


POSE_DETECT_FPS = 5
CAMERA_PROCESS_FPS = 10
LIVE_STREAM_FPS = 15
# ---------------------------------------------------------------- Pose method
POSE_METHOD = '2d_head_vs_body'
