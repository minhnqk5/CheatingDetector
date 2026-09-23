import cv2
import numpy as np
import time

from detection.face_detector import FaceDetector
from detection.pose_detector import PoseDetector
from behavior.behavior_detector import BehaviorDetector
from behavior.temporal_analyzer import TemporalAnalyzer
from event_logging.event_logger import EventLogger


# ==================================================
# Configuration
# ==================================================

YAW_CENTER_THRESHOLD = 0.3

PITCH_CENTER_MIN = 0.3
PITCH_CENTER_MAX = 0.5

REFERENCE_MIN_DURATION = 1.5



face_detector = FaceDetector()
pose_detector = PoseDetector()
behavior_detector = BehaviorDetector()
temporal_analyzer = TemporalAnalyzer()
event_logger = EventLogger()


cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Không thể mở camera")

    exit()



looking_center_start = None

reference_locked = False


print("========================================")
print("PERSON SUBSTITUTION TEST")
print("========================================")
print("Hãy nhìn thẳng vào camera.")
print(
    f"Giữ nhìn thẳng {REFERENCE_MIN_DURATION:.1f} giây "
    "để đăng ký Reference Person."
)
print("========================================")


# ==================================================
# Main loop
# ==================================================

try:

    while True:

        ret, frame = cap.read()


        if not ret or frame is None:

            print("\nKhông thể đọc frame")

            break


        # ==================================================
        # Normalize frame
        # ==================================================

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)


        if (
            len(frame.shape) != 3
            or frame.shape[2] != 3
        ):

            continue


        frame = np.ascontiguousarray(
            frame
        )


        # ==================================================
        # Face detection
        # ==================================================

        try:

            faces = face_detector.detect(
                frame
            )

        except Exception as e:

            print(
                f"\nFACE DETECTOR ERROR: "
                f"{type(e).__name__}: {e}"
            )

            continue


        # ==================================================
        # Pose detection
        # ==================================================

        try:

            pose = pose_detector.detect(
                frame
            )

        except Exception as e:

            print(
                f"\nPOSE DETECTOR ERROR: "
                f"{type(e).__name__}: {e}"
            )

            pose = None


        # ==================================================
        # Reference registration
        # ==================================================

        if not reference_locked:


            # --------------------------------------------------
            # Need exactly one face
            # --------------------------------------------------

            if len(faces) == 1 and pose is not None:

                yaw = pose["yaw"]

                pitch = pose["pitch"]


                # --------------------------------------------------
                # Check looking center
                # --------------------------------------------------

                looking_center = (

                    abs(yaw)
                    <= YAW_CENTER_THRESHOLD

                    and

                    PITCH_CENTER_MIN
                    <= pitch
                    <= PITCH_CENTER_MAX

                )


                # ==================================================
                # Looking center
                # ==================================================

                if looking_center:


                    if looking_center_start is None:

                        looking_center_start = (
                            time.monotonic()
                        )


                    elapsed = (
                        time.monotonic()
                        - looking_center_start
                    )


                    remaining = max(
                        0,
                        REFERENCE_MIN_DURATION
                        - elapsed
                    )


                    # --------------------------------------------------
                    # Capture reference
                    # --------------------------------------------------

                    if elapsed >= REFERENCE_MIN_DURATION:

                        success = (
                            face_detector.capture_reference(
                                faces[0]
                            )
                        )


                        if success:

                            reference_locked = True

                            print(
                                "\n\n"
                                "========================================"
                            )

                            print(
                                "REFERENCE PERSON LOCKED!"
                            )

                            print(
                                "========================================"
                            )


                            # Reset timer

                            looking_center_start = None


                    else:

                        status = (
                            f"REGISTERING REFERENCE "
                            f"{elapsed:.1f}/"
                            f"{REFERENCE_MIN_DURATION:.1f}s"
                        )


                # ==================================================
                # Not looking center
                # ==================================================

                else:

                    looking_center_start = None

                    status = (
                        "LOOK STRAIGHT "
                        f"(yaw={yaw:+.2f}, "
                        f"pitch={pitch:+.2f})"
                    )


            # ==================================================
            # No suitable face
            # ==================================================

            else:

                looking_center_start = None


                if len(faces) == 0:

                    status = "NO FACE"


                elif len(faces) > 1:

                    status = (
                        "MULTIPLE FACES - "
                        "ONLY ONE PERSON ALLOWED"
                    )


                elif pose is None:

                    status = "POSE NOT DETECTED"


                else:

                    status = "WAITING"


            # ==================================================
            # Display registration status
            # ==================================================

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


        # ==================================================
        # Reference already locked
        # ==================================================

        else:


            # --------------------------------------------------
            # Behavior detection
            # --------------------------------------------------

            behaviors = (
                behavior_detector.detect(
                    pose=pose,
                    faces=faces
                )
            )


            # --------------------------------------------------
            # Temporal analysis
            # --------------------------------------------------

            events = (
                temporal_analyzer.update(
                    behaviors
                )
            )


            for event in events:

                event_logger.log(
                    event
                )


            # ==================================================
            # Display behavior
            # ==================================================

            if behaviors:

                behavior_text = ", ".join(
                    str(b)
                    for b in behaviors
                )

            else:

                behavior_text = "NORMAL"


            cv2.putText(
                frame,
                f"BEHAVIOR: {behavior_text}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                "REFERENCE LOCKED",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )


        # ==================================================
        # Draw faces
        # ==================================================

        for index, face in enumerate(faces):


            identity = face.get(
                "identity",
                "UNKNOWN"
            )


            similarity = face.get(
                "similarity"
            )


            # --------------------------------------------------
            # Text
            # --------------------------------------------------

            if similarity is not None:

                text = (
                    f"{identity} "
                    f"{similarity:.3f}"
                )

            else:

                text = identity


            cv2.putText(
                frame,
                text,
                (
                    20,
                    170 + index * 30
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


            # ==================================================
            # Bounding box
            # ==================================================

            bbox = face["bbox"]

            height, width = frame.shape[:2]


            x1 = max(
                0,
                int(
                    bbox.xmin * width
                )
            )

            y1 = max(
                0,
                int(
                    bbox.ymin * height
                )
            )


            x2 = min(
                width,
                int(
                    (bbox.xmin + bbox.width)
                    * width
                )
            )

            y2 = min(
                height,
                int(
                    (bbox.ymin + bbox.height)
                    * height
                )
            )


            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # ==================================================
        # Display pose
        # ==================================================

        if pose is not None:

            yaw = pose["yaw"]

            pitch = pose["pitch"]


            cv2.putText(
                frame,
                f"Yaw: {yaw:+.3f}",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"Pitch: {pitch:+.3f}",
                (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )


        # ==================================================
        # Terminal
        # ==================================================

        if not reference_locked:

            if (
                len(faces) == 1
                and pose is not None
            ):

                yaw = pose["yaw"]

                pitch = pose["pitch"]


                looking_center = (

                    abs(yaw)
                    <= YAW_CENTER_THRESHOLD

                    and

                    PITCH_CENTER_MIN
                    <= pitch
                    <= PITCH_CENTER_MAX

                )


                if looking_center:

                    if looking_center_start is not None:

                        elapsed = (
                            time.monotonic()
                            - looking_center_start
                        )

                        print(
                            f"\rREGISTERING REFERENCE: "
                            f"{elapsed:.1f}/"
                            f"{REFERENCE_MIN_DURATION:.1f}s",
                            end=""
                        )

                else:

                    print(
                        "\rLOOK STRAIGHT INTO CAMERA"
                        + " " * 20,
                        end=""
                    )

            else:

                print(
                    "\rWAITING FOR ONE FACE..."
                    + " " * 20,
                    end=""
                )


        else:

            if faces:

                first_face = faces[0]

                identity = first_face.get(
                    "identity",
                    "UNKNOWN"
                )

                similarity = first_face.get(
                    "similarity"
                )


                if similarity is not None:

                    print(
                        f"\rFACES={len(faces)} | "
                        f"{identity} | "
                        f"similarity={similarity:.3f}",
                        end=""
                    )

                else:

                    print(
                        f"\rFACES={len(faces)} | "
                        f"{identity}",
                        end=""
                    )

            else:

                print(
                    "\rNO FACE"
                    + " " * 30,
                    end=""
                )


        # ==================================================
        # Show frame
        # ==================================================

        cv2.imshow(
            "Person Substitution Test",
            frame
        )


        # ==================================================
        # Quit
        # ==================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ==================================================
# Keyboard interrupt
# ==================================================

except KeyboardInterrupt:

    print(
        "\n\nĐã dừng chương trình."
    )


# ==================================================
# Unexpected error
# ==================================================

except Exception as e:

    print(
        "\n\n========================================"
    )

    print(
        "UNEXPECTED ERROR"
    )

    print(
        "========================================"
    )

    print(
        type(e).__name__
    )

    print(
        str(e)
    )

    print(
        "========================================"
    )


# ==================================================
# Close current event
# ==================================================

finally:

    try:

        events = temporal_analyzer.close()

        for event in events:

            event_logger.log(
                event
            )

    except Exception as e:

        print(
            f"\nLỗi khi đóng temporal analyzer: {e}"
        )


    # ==================================================
    # Cleanup
    # ==================================================

    cap.release()

    cv2.destroyAllWindows()


    try:

        face_detector.close()

    except Exception:

        pass


    try:

        pose_detector.close()

    except Exception:

        pass