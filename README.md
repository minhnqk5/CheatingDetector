# Exam Monitoring - 2D Head/Body Relative Pose

Bản này giữ nguyên pipeline/tracker/temporal analyzer của project gốc nhưng thay cách tính turn left/right.

## Cách tính
- `face_dir = nose - midpoint(left_eye, right_eye)`
- `body_forward = pháp tuyến của đường nối hai vai`, chọn phía hướng về mũi.
- `yaw = signed_angle(body_forward, face_dir)`

Vì so đầu với **thân của chính người đó**, một người ngồi chéo camera khoảng 30-60° ít bị coi nhầm là đang quay đầu chỉ vì toàn thân đã xoay.

Không dùng yaw baseline. Không cần model 3D hay thư viện mới.

## Chạy
```bash
pip install -r requirements.txt
python main.py video input.mp4 --show
python main.py web
```

## Lưu ý
Đây vẫn là hình học 2D nên không thể loại bỏ hoàn toàn méo phối cảnh. Nó được thiết kế ưu tiên tốc độ và giữ nguyên YOLO-pose hiện tại.
