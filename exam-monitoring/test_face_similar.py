import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==========================================
# CONFIG
# ==========================================

REFERENCE_IMAGE = "data/face_images/5769405614ab4b56b729665ffc7588be.jpg"

SIMILARITY_THRESHOLD = 0.45
UNCERTAIN_THRESHOLD = 0.35

CAMERA_INDEX = 0


# ==========================================
# LOAD INSIGHTFACE
# ==========================================

print("[INFO] Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=[
        "CPUExecutionProvider"
    ]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("[INFO] InsightFace loaded.")


# ==========================================
# LOAD REFERENCE FACE
# ==========================================

print("[INFO] Loading reference image...")

reference_image = cv2.imread(
    REFERENCE_IMAGE
)

if reference_image is None:
    raise ValueError(
        f"Cannot read reference image: "
        f"{REFERENCE_IMAGE}"
    )

reference_faces = app.get(
    reference_image
)

if len(reference_faces) == 0:
    raise ValueError(
        "No face detected in reference image."
    )

if len(reference_faces) > 1:
    raise ValueError(
        "Reference image must contain "
        "exactly one face."
    )

reference_embedding = (
    reference_faces[0].normed_embedding
)

reference_embedding = np.asarray(
    reference_embedding,
    dtype=np.float32
)

reference_embedding /= np.linalg.norm(
    reference_embedding
)

print(
    "[INFO] Reference face loaded."
)

print(
    "[INFO] Embedding shape:",
    reference_embedding.shape
)


# ==========================================
# OPEN WEBCAM
# ==========================================

print("[INFO] Opening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():
    raise RuntimeError(
        "Cannot open webcam."
    )

print("[INFO] Webcam started.")
print("[INFO] Press Q to quit.")


# ==========================================
# WEBCAM LOOP
# ==========================================

DETECT_EVERY = 5
frame_count = 0
last_faces = []


while True:

    ret, frame = cap.read()

    if not ret:
        print(
            "[ERROR] Cannot read webcam frame."
        )
        break

    # InsightFace detection
    frame_count += 1

    if frame_count % DETECT_EVERY == 0:
        last_faces = app.get(frame)

    faces = last_faces

    for face in faces:

        # ----------------------------------
        # Bounding box
        # ----------------------------------

        x1, y1, x2, y2 = (
            face.bbox.astype(int)
        )

        # ----------------------------------
        # Face embedding
        # ----------------------------------

        embedding = face.normed_embedding

        if embedding is None:
            continue

        embedding = np.asarray(
            embedding,
            dtype=np.float32
        )

        # ----------------------------------
        # Cosine similarity
        #
        # Because both embeddings are
        # normalized, dot product =
        # cosine similarity.
        # ----------------------------------

        similarity = float(
            np.dot(
                reference_embedding,
                embedding
            )
        )

        # ----------------------------------
        # Identity classification
        # ----------------------------------

        if (
            similarity
            >= SIMILARITY_THRESHOLD
        ):

            identity = "SAME_PERSON"

        elif (
            similarity
            >= UNCERTAIN_THRESHOLD
        ):

            identity = "UNCERTAIN"

        else:

            identity = "DIFFERENT_PERSON"

        # ----------------------------------
        # Face detection confidence
        # ----------------------------------

        detection_confidence = float(
            face.det_score
        )

        # ----------------------------------
        # Draw bounding box
        # ----------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # ----------------------------------
        # Text
        # ----------------------------------

        text_identity = (
            f"{identity}"
        )

        text_similarity = (
            f"Similarity: "
            f"{similarity:.3f}"
        )

        text_confidence = (
            f"Detection: "
            f"{detection_confidence:.3f}"
        )

        # Identity
        cv2.putText(
            frame,
            text_identity,
            (x1, y1 - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Similarity
        cv2.putText(
            frame,
            text_similarity,
            (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Detection confidence
        cv2.putText(
            frame,
            text_confidence,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # ======================================
    # GLOBAL INFO
    # ======================================

    cv2.putText(
        frame,
        "InsightFace Webcam Test",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # Number of detected faces
    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    # ======================================
    # SHOW
    # ======================================

    cv2.imshow(
        "InsightFace Webcam Test",
        frame
    )

    # Q to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


# ==========================================
# CLEANUP
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("[INFO] Webcam test finished.")