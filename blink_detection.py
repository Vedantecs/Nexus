"""
NEXUS - gestures/blink_detection.py
Step 6: measure how open each eye is (Eye Aspect Ratio, EAR) in real time.

Run from the NEXUS folder (with venv active):
    python gestures/blink_detection.py
Press 'q' in the video window to quit.
"""

import time

import cv2
import mediapipe as mp
import numpy as np

# 6 landmarks per eye, in this order:
# [outer corner, upper-1, upper-2, inner corner, lower-2, lower-1]
# The image is mirrored, so "screen left" is your own left eye.
SCREEN_LEFT_EYE = [33, 160, 158, 133, 153, 144]
SCREEN_RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR below this value counts as "closed".
# Measured on Ved's webcam: open eyes about 0.24-0.26, closed about 0.06-0.09.
EAR_CLOSED_THRESHOLD = 0.16

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def get_points(face, indices, width, height):
    """Return landmark positions in pixels as a list of (x, y) arrays."""
    return [
        np.array([face.landmark[i].x * width, face.landmark[i].y * height])
        for i in indices
    ]


def eye_aspect_ratio(face, indices, width, height):
    """EAR = (vertical distance 1 + vertical distance 2) / (2 * horizontal distance)."""
    p1, p2, p3, p4, p5, p6 = get_points(face, indices, width, height)
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open the webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Blink test started. Press 'q' in the video window to quit.")
    previous_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Could not read a frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]

            y_text = 30
            for name, indices in (("Left eye", SCREEN_LEFT_EYE),
                                  ("Right eye", SCREEN_RIGHT_EYE)):
                ear = eye_aspect_ratio(face, indices, width, height)
                closed = ear < EAR_CLOSED_THRESHOLD
                color = RED if closed else GREEN

                # Draw the 6 eye points and the text
                for point in get_points(face, indices, width, height):
                    cv2.circle(frame, (int(point[0]), int(point[1])), 2, color, -1)
                state = "CLOSED" if closed else "OPEN"
                cv2.putText(frame, f"{name}: {state}  EAR {ear:.2f}", (10, y_text),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y_text += 30
        else:
            cv2.putText(frame, "NO FACE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

        now = time.time()
        fps = 1.0 / max(now - previous_time, 1e-6)
        previous_time = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        cv2.imshow("NEXUS - Blink Test (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()