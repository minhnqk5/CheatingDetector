import mediapipe as mp


class FaceDetector:

    def __init__(self):
        self.mp_face = mp.solutions.face_detection

        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

    def detect(self, frame):
        rgb = frame[:, :, ::-1]

        results = self.detector.process(rgb)

        faces = []

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                faces.append({
                    "bbox": bbox,
                    "confidence": detection.score[0]
                })

        return faces