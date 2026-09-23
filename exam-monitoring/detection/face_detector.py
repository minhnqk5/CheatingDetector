import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:

    def __init__(
        self,
        reference_image_path=None,
        similarity_threshold=0.45,
        uncertain_threshold=0.35,
        detect_every=3
    ):
        # ==================================================
        # InsightFace
        # ==================================================

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )



        self.reference_embedding = None

        self.similarity_threshold = similarity_threshold
        self.uncertain_threshold = uncertain_threshold



        self.frame_count = 0
        self.detect_every = detect_every


        self.last_faces = []




    # ======================================================
    # Detect faces
    # ======================================================

    def detect(self, frame):

        self.frame_count += 1

        # --------------------------------------------------
        # Skip frame
        # --------------------------------------------------

        if self.frame_count % self.detect_every != 0:
            return self.last_faces

        # ==================================================
        # InsightFace detection + embedding
        # ==================================================

        detected_faces = self.app.get(frame)

        faces = []

        # ==================================================
        # No face
        # ==================================================

        if not detected_faces:

            self.last_faces = []

            return self.last_faces

        height, width = frame.shape[:2]

        # ==================================================
        # Process every face
        # ==================================================

        for face in detected_faces:

            # --------------------------------------------------
            # Bounding box
            # --------------------------------------------------

            x1, y1, x2, y2 = face.bbox.astype(int)

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # --------------------------------------------------
            # Detection confidence
            # --------------------------------------------------

            confidence = float(face.det_score)

            # --------------------------------------------------
            # Embedding
            # --------------------------------------------------

            embedding = face.normed_embedding

            identity = "NO_REFERENCE"
            similarity = None

            if embedding is not None:

                embedding = embedding.astype(
                    np.float32
                )

                # ==========================================
                # Compare with reference
                # ==========================================

                if self.reference_embedding is not None:

                    similarity = float(
                        np.dot(
                            self.reference_embedding,
                            embedding
                        )
                    )



                    if similarity >= self.similarity_threshold:

                        identity = "SAME_PERSON"



                    elif similarity >= self.uncertain_threshold:

                        identity = "UNCERTAIN"


                    else:

                        identity = "DIFFERENT_PERSON"

            # ==================================================
            # Save face
            # ==================================================

            faces.append({

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                ),

                "confidence": confidence,

                "identity": identity,

                "similarity": similarity,

                # InsightFace embedding
                "embedding": embedding
            })

        # ==================================================
        # Save result
        # ==================================================

        self.last_faces = faces

        return faces


    # ======================================================
    # Check reference
    # ======================================================

    def has_reference(self):

        return self.reference_embedding is not None


    # ======================================================
    # Set reference embedding
    # ======================================================

    def set_reference_embedding(self, embedding):

        if embedding is None:
            self.reference_embedding = None
            return

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # ------------------------------------------
        # Normalize
        # ------------------------------------------

        norm = np.linalg.norm(
            embedding
        )

        if norm < 1e-8:
            raise ValueError(
                "Invalid reference embedding."
            )

        self.reference_embedding = (
            embedding / norm
        ).astype(np.float32)


    # ======================================================
    # Clear reference
    # ======================================================

    def reset_reference(self):

        self.reference_embedding = None


    # ======================================================
    # Close
    # ======================================================

    def close(self):

        self.app = None

