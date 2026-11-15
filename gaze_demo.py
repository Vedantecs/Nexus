"""
NEXUS - cursor/gaze_demo.py
Live test of the trained gaze model. Shows a small map of your screen in the
camera window; the red dot is where the model thinks you are looking.
This does NOT move your mouse.

Run from the NEXUS folder (with venv active):
    python -m cursor.gaze_demo
Press 'q' in the video window to quit.
"""

import time

import cv2
import mediapipe as mp

from cursor.gaze_estimator import GazeEstimator

SMOOTHING = 0.3          # 0..1, lower = smoother but slower
MAP_W, MAP_H = 320, 180  # size of the small screen map

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def main():
    estimator = GazeEstimator()

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

    smooth = None
    print("Gaze demo started. Look at the corners of your screen. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Could not read a frame from the webcam.")
            break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        status, latency_ms = "NO FACE", 0.0
        if result.multi_face_landmarks:
            t0 = time.time()
            pred = estimator.predict(frame, result.multi_face_landmarks[0])
            latency_ms = (time.time() - t0) * 1000
            if pred is not None:
                x = min(max(pred[0], 0.0), 1.0)
                y = min(max(pred[1], 0.0), 1.0)
                smooth = (x, y) if smooth is None else (
                    SMOOTHING * x + (1 - SMOOTHING) * smooth[0],
                    SMOOTHING * y + (1 - SMOOTHING) * smooth[1])
                status = f"GAZE x={smooth[0]:.2f} y={smooth[1]:.2f}"

        # Small screen map (top right) with 3 x 3 zone lines
        x0, y0 = width - MAP_W - 10, 10
        cv2.rectangle(frame, (x0, y0), (x0 + MAP_W, y0 + MAP_H), WHITE, 1)
        for i in (1, 2):
            cv2.line(frame, (x0 + i * MAP_W // 3, y0), (x0 + i * MAP_W // 3, y0 + MAP_H), (120, 120, 120), 1)
            cv2.line(frame, (x0, y0 + i * MAP_H // 3), (x0 + MAP_W, y0 + i * MAP_H // 3), (120, 120, 120), 1)
        if smooth is not None:
            cv2.circle(frame, (x0 + int(smooth[0] * MAP_W), y0 + int(smooth[1] * MAP_H)), 7, RED, -1)

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    GREEN if smooth is not None else RED, 2)
        cv2.putText(frame, f"Model time: {latency_ms:.0f} ms", (10, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        cv2.imshow("NEXUS - Gaze Demo (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()