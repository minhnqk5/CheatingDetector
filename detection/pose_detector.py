import cv2
import numpy as np
import mediapipe as mp


class PoseDetector:

    def __init__(
        self,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    # ==========================================================
    # Detect
    # ==========================================================

    def detect(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]

        height, width, _ = frame.shape

        # ======================================================
        # 6 landmarks
        # ======================================================

        nose = self._get_point(
            landmarks,
            1,
            width,
            height
        )

        left_eye = self._get_point(
            landmarks,
            33,
            width,
            height
        )

        right_eye = self._get_point(
            landmarks,
            263,
            width,
            height
        )

        left_mouth = self._get_point(
            landmarks,
            61,
            width,
            height
        )

        right_mouth = self._get_point(
            landmarks,
            291,
            width,
            height
        )

        chin = self._get_point(
            landmarks,
            152,
            width,
            height
        )

        # ======================================================
        # YAW
        # ======================================================
        #
        # Trung tâm hai mắt
        #
        # Khi nhìn thẳng:
        #
        #       nose
        #         |
        #     eye center
        #
        # Khi quay:
        #
        #      nose
        #        \
        #         eye center
        #
        # Chuẩn hóa theo khoảng cách hai mắt.
        # ======================================================

        eye_center = (
            left_eye +
            right_eye
        ) / 2.0

        eye_distance = np.linalg.norm(
            right_eye -
            left_eye
        )

        if eye_distance < 1e-6:
            return None

        yaw = (
            nose[0] -
            eye_center[0]
        ) / eye_distance

        # ======================================================
        # PITCH
        # ======================================================
        #
        # Dùng vị trí tương đối của:
        #
        # eye center
        # nose
        # mouth center
        # chin
        #
        # để giảm ảnh hưởng của kích thước khuôn mặt.
        # ======================================================

        mouth_center = (
            left_mouth +
            right_mouth
        ) / 2.0

        face_height = np.linalg.norm(
            chin -
            eye_center
        )

        if face_height < 1e-6:
            return None

        nose_vertical = (
            nose[1] -
            eye_center[1]
        )

        pitch = (
            nose_vertical /
            face_height
        )

        # ======================================================
        # Return
        # ======================================================

        return {
            "yaw": float(yaw),
            "pitch": float(pitch)
        }

    # ==========================================================
    # Get landmark
    # ==========================================================

    def _get_point(
        self,
        landmarks,
        index,
        width,
        height
    ):

        landmark = landmarks.landmark[index]

        return np.array(
            [
                landmark.x * width,
                landmark.y * height
            ],
            dtype=np.float64
        )

    # ==========================================================

    def close(self):

        self.face_mesh.close()