import numpy as np
import cv2

from insightface.app import FaceAnalysis


class FaceEmbeddingService:

    def __init__(self):

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )


    # ==================================================
    # Generate embedding from image
    # ==================================================

    def generate_embedding(
        self,
        image_path
    ):

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise ValueError(
                "Không thể đọc ảnh."
            )

        faces = self.app.get(
            image
        )

        # --------------------------------------------------
        # Không có mặt
        # --------------------------------------------------

        if len(faces) == 0:

            raise ValueError(
                "Không phát hiện được khuôn mặt."
            )

        # --------------------------------------------------
        # Có nhiều mặt
        # --------------------------------------------------

        if len(faces) > 1:

            raise ValueError(
                "Ảnh phải chỉ có một khuôn mặt."
            )

        face = faces[0]

        embedding = face.normed_embedding

        if embedding is None:

            raise ValueError(
                "Không thể tạo face embedding."
            )

        # --------------------------------------------------
        # Normalize
        # --------------------------------------------------

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm < 1e-8:

            raise ValueError(
                "Face embedding không hợp lệ."
            )

        embedding = embedding / norm

        return embedding


    # ==================================================
    # Convert embedding -> bytes
    # ==================================================

    @staticmethod
    def embedding_to_bytes(
        embedding
    ):

        return np.asarray(
            embedding,
            dtype=np.float32
        ).tobytes()


    # ==================================================
    # Convert bytes -> embedding
    # ==================================================

    @staticmethod
    def bytes_to_embedding(data):

        if data is None:
            return None


        embedding = np.frombuffer(
            data,
            dtype=np.float32
        ).copy()


        norm = np.linalg.norm(
            embedding
        )


        if norm < 1e-8:

            raise ValueError(
                "Invalid face embedding."
            )


        embedding = (
            embedding / norm
        )


        return embedding.astype(
            np.float32
        )
