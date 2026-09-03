from ultralytics import YOLO

from config import YOLO_MODEL, YOLO_CONFIDENCE


class ObjectDetector:

    def __init__(self):
        self.model = YOLO(YOLO_MODEL)

    def detect(self, frame):

        results = self.model(
            frame,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )

        objects = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                label = self.model.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                objects.append({
                    "label": label,
                    "confidence": confidence,
                    "bbox": (x1, y1, x2, y2)
                })

        return objects