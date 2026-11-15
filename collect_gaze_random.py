"""
NEXUS - training/collect_gaze_random.py
Step 20: collect gaze data at RANDOM screen positions (60 dots per run).
Every run places one random dot inside each cell of a 10 x 6 grid, so the
dots cover the whole screen and every run uses new positions.

Run from the NEXUS folder (with venv active):
    python -m training.collect_gaze_random                 training run
    python -m training.collect_gaze_random test            test run
    python -m training.collect_gaze_random glasses         training run, glasses on
    python -m training.collect_gaze_random test glasses    test run, glasses on
Press SPACE to start, ESC to stop at any time.
Output: data/gaze_random/images/ and data/gaze_random/labels.csv

Session names tell the runs apart:
    r... training, no glasses    rg... training, glasses
    t... test, no glasses        tg... test, glasses
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
CELLS_X, CELLS_Y = 10, 6                  # 60 dots per run
X_MIN, X_MAX = 0.03, 0.97                 # screen area used (fractions)
Y_MIN, Y_MAX = 0.05, 0.95
SETTLE_S = 0.8                            # time to move your eyes (nothing saved)
CAPTURE_S = 1.2                           # time while images are saved
CROP_W, CROP_H = 128, 48

# Face landmarks saved as numbers: nose tip, forehead, chin, face sides,
# eye corners, and both iris centers.
LANDMARKS = [1, 10, 152, 234, 454, 33, 133, 362, 263, 468, 473]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "gaze_random"
IMAGE_DIR = DATA_DIR / "images"


def get_screen_size():
    """Real screen size in pixels (Windows)."""
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return (ctypes.windll.user32.GetSystemMetrics(0),
                ctypes.windll.user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


def make_dots():
    """One random dot inside every cell of the grid: even coverage, new positions each run."""
    cell_w = (X_MAX - X_MIN) / CELLS_X
    cell_h = (Y_MAX - Y_MIN) / CELLS_Y
    dots = []
    for row in range(CELLS_Y):
        for col in range(CELLS_X):
            x = X_MIN + (col + random.uniform(0.1, 0.9)) * cell_w
            y = Y_MIN + (row + random.uniform(0.1, 0.9)) * cell_h
            dots.append((x, y))
    return dots


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


def draw_screen(screen_w, screen_h, dot, phase, status):
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    if dot is not None:   # red = get ready, green = saving
        cx, cy = int(dot[0] * screen_w), int(dot[1] * screen_h)
        color = (0, 0, 255) if phase == "settle" else (0, 255, 0)
        cv2.circle(canvas, (cx, cy), 14, color, -1)
        cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1)
    cv2.putText(canvas, status, (int(screen_w * 0.15), screen_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main():
    is_test = "test" in sys.argv[1:]
    glasses = "glasses" in sys.argv[1:]
    dots = make_dots()
    screen_w, screen_h = get_screen_size()
    prefix = ("t" if is_test else "r") + ("g" if glasses else "")
    session = datetime.now().strftime(prefix + "%Y%m%d_%H%M%S")
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    labels_path = DATA_DIR / "labels.csv"
    new_file = (not labels_path.exists()) or labels_path.stat().st_size == 0
    header = ["file", "point", "session", "target_x", "target_y", "left_ear", "right_ear"]
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

    window = "NEXUS random gaze data"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    visits = list(range(len(dots)))
    random.shuffle(visits)

    started = False
    visit_index = 0
    phase_start = 0.0
    saved = 0
    bands = [0, 0, 0, 0]                       # images per screen quarter, top to bottom
    skipped = {"no face": 0, "eyes outside frame": 0}
    kind = ("TEST run" if is_test else "TRAINING run") + (" (glasses)" if glasses else "")

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

            dot, phase = None, "idle"
            if not started:
                status = f"{kind}, {len(dots)} dots. Press SPACE to start, ESC to quit."
            else:
                elapsed = now - phase_start
                if elapsed >= SETTLE_S + CAPTURE_S:      # move on to the next dot
                    visit_index += 1
                    phase_start = now
                    elapsed = 0.0
                if visit_index >= len(visits):
                    break
                point = visits[visit_index]
                dot = dots[point]
                phase = "settle" if elapsed < SETTLE_S else "capture"
                status = f"Dot {visit_index + 1}/{len(visits)}   saved {saved}"
                if face is None:
                    status += "   NO FACE!"

                if phase == "capture":
                    if face is None:
                        skipped["no face"] += 1
                    else:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        strip = eye_strip(gray, face, width, height)
                        if strip is None:
                            skipped["eyes outside frame"] += 1
                        else:
                            left_ear = eye_aspect_ratio(face, SCREEN_LEFT_EYE, width, height)
                            right_ear = eye_aspect_ratio(face, SCREEN_RIGHT_EYE, width, height)
                            name = f"{session}_{saved:05d}.png"
                            cv2.imwrite(str(IMAGE_DIR / name), strip)
                            writer.writerow([f"images/{name}", point, session,
                                             round(dot[0], 4), round(dot[1], 4),
                                             round(left_ear, 3), round(right_ear, 3)]
                                            + landmark_features(face))
                            saved += 1
                            bands[min(int(dot[1] * 4), 3)] += 1

            cv2.imshow(window, draw_screen(screen_w, screen_h, dot, phase, status))
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
    print("Images by screen quarter (top to bottom):", bands)
    print("Skipped frames:", skipped)


if __name__ == "__main__":
    main()