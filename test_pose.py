import cv2

from detection.pose_detector import PoseDetector


# ==========================================================
# Camera
# ==========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()


pose_detector = PoseDetector()


# ==========================================================
# Threshold
# ==========================================================

YAW_THRESHOLD = 0.3

PITCH_UP_THRESHOLD = 0.3
PITCH_DOWN_THRESHOLD = 0.5


# ==========================================================
# Classification
# ==========================================================

def classify_yaw(yaw):

    if yaw < -YAW_THRESHOLD:
        return "LEFT"

    elif yaw > YAW_THRESHOLD:
        return "RIGHT"

    else:
        return "CENTER"


def classify_pitch(pitch):

    if pitch < PITCH_UP_THRESHOLD:
        return "UP"

    elif pitch > PITCH_DOWN_THRESHOLD:
        return "DOWN"

    else:
        return "CENTER"


# ==========================================================
# Main
# ==========================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Mirror webcam
    frame = cv2.flip(
        frame,
        1
    )

    pose = pose_detector.detect(frame)

    if pose is not None:

        yaw = pose["yaw"]
        pitch = pose["pitch"]

        yaw_state = classify_yaw(
            yaw
        )

        pitch_state = classify_pitch(
            pitch
        )

        # ==================================================
        # Display values
        # ==================================================

        cv2.putText(
            frame,
            f"Yaw   : {yaw:+.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Pitch : {pitch:.3f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2
        )

        # ==================================================
        # Direction
        # ==================================================

        cv2.putText(
            frame,
            f"Yaw state   : {yaw_state}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Pitch state : {pitch_state}",
            (20, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # ==================================================
        # Combined state
        # ==================================================

        if (
            yaw_state == "CENTER"
            and
            pitch_state == "CENTER"
        ):

            state = "LOOK CENTER"

        elif yaw_state != "CENTER":

            state = f"TURN {yaw_state}"

        else:

            state = f"LOOK {pitch_state}"

        cv2.putText(
            frame,
            f"STATE: {state}",
            (20, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # ==================================================
        # Terminal
        # ==================================================

        print(
            f"\r"
            f"Yaw={yaw:+.3f} | "
            f"Pitch={pitch:.3f} | "
            f"{state:<18}",
            end=""
        )

    else:

        cv2.putText(
            frame,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        print(
            "\rNO FACE                         ",
            end=""
        )

    # ======================================================
    # Show
    # ======================================================

    cv2.imshow(
        "Head Pose Test",
        frame
    )

    # ======================================================
    # Quit
    # ======================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


print()

cap.release()

cv2.destroyAllWindows()

pose_detector.close()