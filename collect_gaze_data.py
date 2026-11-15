"""
NEXUS - training/collect_gaze_data.py
Step 16: collect gaze training data for the 9-zone gaze CNN.

A dot appears in one of 9 screen zones. You look at it, and for every frame
the script saves a small grayscale crop of your eyes plus face landmark
numbers, labelled with the zone number.

Zones:  0 top-left     1 top      2 top-right
        3 left         4 center   5 right
        6 bottom-left  7 bottom   8 bottom-right

Run from the NEXUS folder (with venv active):
    python -m training.collect_gaze_data
Press SPACE to start, ESC to stop at any time.
Output: data/gaze/zone_0 ... zone_8 (images) and data/gaze/labels.csv
"""

import csv
import ctypes
import random
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
ROUNDS = 6              # how many times every zone is visited (random order)
SETTLE_S = 1.0          # time to move your eyes to the dot (nothing saved)
CAPTURE_S = 2.0         # time while images are saved
MIN_EAR_TO_SAVE = 0.13  # skip frames where the eyes are clearly closed (blinks)
CROP_W, CROP_H = 128, 48

# Dot positions as fractions of the screen: columns and rows
DOT_X = [0.06, 0.50, 0.94]
DOT_Y = [0.08, 0.50, 0.92]

# Face landmarks saved as numbers: nose tip, forehead, chin, face sides,
# eye corners, and both iris centers.
LANDMARKS = [1, 10, 152, 234, 454, 33, 133, 362, 263, 468, 473]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "gaze"


def get_screen_size():
    """Real screen size in pixels (Windows)."""
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return (ctypes.windll.user32.GetSystemMetrics(0),
                ctypes.windll.user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


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


def draw_screen(screen_w, screen_h, zone, phase, status):
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    for z in range(9):    # faint dots show the 9 zone positions
        cx, cy = int(DOT_X[z % 3] * screen_w), int(DOT_Y[z // 3] * screen_h)
        cv2.circle(canvas, (cx, cy), 8, (70, 70, 70), -1)
    if zone is not None:  # the active dot: red = get ready, green = saving
        cx, cy = int(DOT_X[zone % 3] * screen_w), int(DOT_Y[zone // 3] * screen_h)
        color = (0, 0, 255) if phase == "settle" else (0, 255, 0)
        cv2.circle(canvas, (cx, cy), 22, color, -1)
        cv2.circle(canvas, (cx, cy), 6, (255, 255, 255), -1)
    cv2.putText(canvas, status, (int(screen_w * 0.15), screen_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def main():
    screen_w, screen_h = get_screen_size()
    session = datetime.now().strftime("s%Y%m%d_%H%M%S")
    for z in range(9):
        (DATA_DIR / f"zone_{z}").mkdir(parents=True, exist_ok=True)

    labels_path = DATA_DIR / "labels.csv"
    new_file = (not labels_path.exists()) or labels_path.stat().st_size == 0
    header = ["file", "zone", "session", "target_x", "target_y"]
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

    window = "NEXUS gaze data"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    visits = []
    for _ in range(ROUNDS):
        order = list(range(9))
        random.shuffle(order)
        visits += order

    started = False
    visit_index = 0
    phase_start = 0.0
    saved = 0
    per_zone = [0] * 9

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

            zone, phase = None, "idle"
            if not started:
                status = "Sit comfortably. Press SPACE to start, ESC to quit."
            else:
                elapsed = now - phase_start
                if elapsed >= SETTLE_S + CAPTURE_S:     # move on to the next dot
                    visit_index += 1
                    phase_start = now
                    elapsed = 0.0
                if visit_index >= len(visits):
                    break
                zone = visits[visit_index]
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
                            cv2.imwrite(str(DATA_DIR / f"zone_{zone}" / name), strip)
                            writer.writerow([f"zone_{zone}/{name}", zone, session,
                                             DOT_X[zone % 3], DOT_Y[zone // 3]]
                                            + landmark_features(face))
                            saved += 1
                            per_zone[zone] += 1

            cv2.imshow(window, draw_screen(screen_w, screen_h, zone, phase, status))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:                                # ESC
                break
            if key == 32 and not started:                # SPACE
                started = True
                phase_start = time.time()

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Session {session}: saved {saved} images in {DATA_DIR}")
    print("Images per zone (0-8):", per_zone)


if __name__ == "__main__":
    main()