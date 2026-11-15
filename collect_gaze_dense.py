"""
NEXUS - training/collect_gaze_dense.py
Step 18: collect gaze data on a DENSE grid of dots (for a regression model
that predicts the exact screen position you are looking at).

Run from the NEXUS folder (with venv active):
    python -m training.collect_gaze_dense            normal grid: 6 x 4 = 24 dots
    python -m training.collect_gaze_dense shifted    in-between grid: 5 x 3 = 15 dots
                                                     (used as TEST data, positions the
                                                     model never saw in training)
Press SPACE to start, ESC to stop at any time.
Output: data/gaze_dense/images/ and data/gaze_dense/labels.csv
"""

import csv
import ctypes
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from gestures.blink_detection import (
    SCREEN_LEFT_EYE,
    SCREEN_RIGHT_EYE,
    eye_aspect_ratio,
)

# ---- Settings ----
COLS, ROWS = 6, 4        # normal grid size
ROUNDS_NORMAL = 4        # times every dot is visited (normal grid)
ROUNDS_SHIFTED = 6       # times every dot is visited (shifted grid)
SETTLE_S = 0.8           # time to move your eyes to the dot (nothing saved)
CAPTURE_S = 1.5          # time while images are saved
MIN_EAR_TO_SAVE = 0.13   # skip frames where the eyes are clearly closed
CROP_W, CROP_H = 128, 48

# Face landmarks saved as numbers: nose tip, forehead, chin, face sides,
# eye corners, and both iris centers.
LANDMARKS = [1, 10, 152, 234, 454, 33, 133, 362, 263, 468, 473]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "gaze_dense"
IMAGE_DIR = DATA_DIR / "images"


def get_screen_size():
    """Real screen size in pixels (Windows)."""
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return (ctypes.windll.user32.GetSystemMetrics(0),
                ctypes.windll.user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def dot_positions(shifted):
    """List of (x, y) dot positions as fractions of the screen."""
    xs = np.linspace(0.05, 0.95, COLS)
    ys = np.linspace(0.07, 0.93, ROWS)
    if shifted:   # move to the middle between the normal dots
        xs = xs[:-1] + (xs[1] - xs[0]) / 2
        ys = ys[:-1] + (ys[1] - ys[0]) / 2
    return [(float(x), float(y)) for y in ys for x in xs]


def eye_strip(gray, face, width, height):
    """Cut a strip around both eyes and resize it. Returns None if it leaves the frame."""
    xs, ys = [], []
    for i in SCREEN_LEFT_EYE + SCREEN_RIGHT_EYE:
        xs.append(face.landmark[i].x * width)
        ys.append(face.landmark[i].y * height)
    box_w = (max(xs) - min(xs)) * 1.3
    box_h = box_w * CROP_H / CROP_W
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    x1, x2 = int(cx - box_w / 2), int(cx + box_w / 2)
    y1, y2 = int(cy - box_h / 2), int(cy + box_h / 2)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        return None
    return cv2.resize(gray[y1:y2, x1:x2], (CROP_W, CROP_H))


def landmark_features(face):
    row = []
    for i in LANDMARKS:
        row += [round(face.landmark[i].x, 5), round(face.landmark[i].y, 5)]
    return row


def draw_screen(screen_w, screen_h, dots, current, phase, status):
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    for (x, y) in dots:   # faint dots show all positions
        cv2.circle(canvas, (int(x * screen_w), int(y * screen_h)), 4, (70, 70, 70), -1)
    if current is not None:   # active dot: red = get ready, green = saving
        x, y = dots[current]
        cx, cy = int(x * screen_w), int(y * screen_h)
        color = (0, 0, 255) if phase == "settle" else (0, 255, 0)
        cv2.circle(canvas, (cx, cy), 14, color, -1)
        cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1)
    cv2.putText(canvas, status, (int(screen_w * 0.15), screen_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main():
    shifted = "shifted" in sys.argv[1:]
    dots = dot_positions(shifted)
    rounds = ROUNDS_SHIFTED if shifted else ROUNDS_NORMAL
    screen_w, screen_h = get_screen_size()
    prefix = "t" if shifted else "d"
    session = datetime.now().strftime(prefix + "%Y%m%d_%H%M%S")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = DATA_DIR / "labels.csv"
    new_file = (not labels_path.exists()) or labels_path.stat().st_size == 0
    header = ["file", "point", "session", "target_x", "target_y"]
    header += [f"lm{i}_{axis}" for i in LANDMARKS for axis in ("x", "y")]

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

    window = "NEXUS dense gaze data"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    visits = []
    for _ in range(rounds):
        order = list(range(len(dots)))
        random.shuffle(order)
        visits += order

    started = False
    visit_index = 0
    phase_start = 0.0
    saved = 0
    per_point = [0] * len(dots)
    kind = "SHIFTED (test)" if shifted else "NORMAL"

    with open(labels_path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("ERROR: Could not read a frame from the webcam.")
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            now = time.time()
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            face = result.multi_face_landmarks[0] if result.multi_face_landmarks else None

            current, phase = None, "idle"
            if not started:
                status = f"{kind} grid, {len(dots)} dots. Press SPACE to start, ESC to quit."
            else:
                elapsed = now - phase_start
                if elapsed >= SETTLE_S + CAPTURE_S:      # move on to the next dot
                    visit_index += 1
                    phase_start = now
                    elapsed = 0.0
                if visit_index >= len(visits):
                    break
                current = visits[visit_index]
                phase = "settle" if elapsed < SETTLE_S else "capture"
                status = f"Dot {visit_index + 1}/{len(visits)}   saved {saved}"
                if face is None:
                    status += "   NO FACE!"

                if phase == "capture" and face is not None:
                    left_ear = eye_aspect_ratio(face, SCREEN_LEFT_EYE, width, height)
                    right_ear = eye_aspect_ratio(face, SCREEN_RIGHT_EYE, width, height)
                    if min(left_ear, right_ear) > MIN_EAR_TO_SAVE:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        strip = eye_strip(gray, face, width, height)
                        if strip is not None:
                            name = f"{session}_{saved:05d}.png"
                            cv2.imwrite(str(IMAGE_DIR / name), strip)
                            tx, ty = dots[current]
                            writer.writerow([f"images/{name}", current, session,
                                             round(tx, 4), round(ty, 4)]
                                            + landmark_features(face))
                            saved += 1
                            per_point[current] += 1

            cv2.imshow(window, draw_screen(screen_w, screen_h, dots, current, phase, status))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                                # ESC
                break
            if key == 32 and not started:                # SPACE
                started = True
                phase_start = time.time()

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Session {session} ({kind}): saved {saved} images in {DATA_DIR}")
    print("Images per dot:", per_point)


if __name__ == "__main__":
    main()
    