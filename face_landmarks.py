"""
NEXUS - vision/face_landmarks.py
Milestone 1: open the webcam, detect one face, track facial landmarks,
and highlight both eyes, both irises and the mouth in real time.

Run from the NEXUS folder (with venv active):
    python vision/face_landmarks.py
Press 'q' in the video window to quit.
"""

import time

import cv2
import mediapipe as mp
import numpy as np

# ---------------------------------------------------------------------------
# MediaPipe Face Mesh landmark indices (468 face points + 10 iris points).
# The camera image is mirrored below, so we name the eyes by where they appear
# on the screen: SCREEN_LEFT_EYE is the eye on the left side of the window,
# which is your own left eye (like looking in a mirror).
# ---------------------------------------------------------------------------
SCREEN_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155,
                   133, 173, 157, 158, 159, 160, 161, 246]
SCREEN_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249,
                    263, 466, 388, 387, 386, 385, 384, 398]
MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
               291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

SCREEN_LEFT_IRIS_CENTER = 468
SCREEN_RIGHT_IRIS_CENTER = 473

# Mouth opening is measured with 4 points:
MOUTH_TOP, MOUTH_BOTTOM = 13, 14      # inner upper / lower lip
MOUTH_LEFT, MOUTH_RIGHT = 61, 291     # mouth corners

# Mouth Aspect Ratio above this value counts as "OPEN".
# This is only a starting guess; we will tune it by testing.
MOUTH_OPEN_THRESHOLD = 0.08

# Colors are in BGR order (OpenCV style)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
BLUE = (255, 150, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def to_pixels(face, indices, width, height):
    """Convert normalized landmark positions (0..1) into pixel coordinates."""
    points = [
        [int(face.landmark[i].x * width), int(face.landmark[i].y * height)]
        for i in indices
    ]
    return np.array(points, dtype=np.int32)


def mouth_aspect_ratio(face, width, height):
    """Mouth height divided by mouth width. Bigger value = mouth more open."""
    top, bottom, left, right = to_pixels(
        face, [MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT], width, height
    )
    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return float(vertical / horizontal) if horizontal > 0 else 0.0


def main():
    # CAP_DSHOW makes the webcam start faster and more reliably on Windows.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open the webcam. Close other apps using the "
              "camera (Zoom, Teams, browser) and try again.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,        # adds the 10 iris landmarks we need for gaze
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("Webcam started. Press 'q' in the video window to quit.")
    previous_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Could not read a frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)                     # mirror view, feels natural
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # MediaPipe expects RGB
        result = face_mesh.process(rgb)

        status_text = "NO FACE"
        status_color = RED
        mouth_text = ""

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            status_text = "FACE DETECTED"
            status_color = GREEN

            # Eyes
            for eye in (SCREEN_LEFT_EYE, SCREEN_RIGHT_EYE):
                cv2.polylines(frame, [to_pixels(face, eye, width, height)],
                              True, GREEN, 1)

            # Irises (centers)
            for iris in (SCREEN_LEFT_IRIS_CENTER, SCREEN_RIGHT_IRIS_CENTER):
                center = to_pixels(face, [iris], width, height)[0]
                cv2.circle(frame, tuple(center), 3, BLUE, -1)

            # Mouth
            cv2.polylines(frame, [to_pixels(face, MOUTH_OUTER, width, height)],
                          True, YELLOW, 1)
            mar = mouth_aspect_ratio(face, width, height)
            state = "OPEN" if mar > MOUTH_OPEN_THRESHOLD else "CLOSED"
            mouth_text = f"Mouth: {state} (MAR {mar:.2f})"

        # Frames per second
        now = time.time()
        fps = 1.0 / max(now - previous_time, 1e-6)
        previous_time = now

        cv2.putText(frame, status_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, mouth_text, (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        cv2.imshow("NEXUS - Face Landmarks (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()