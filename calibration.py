"""
NEXUS - cursor/calibration.py
15-dot calibration. You look at 15 dots; we compare what the gaze model guessed
with where the dots really were, and fit a small correction (an affine map).
The correction is then applied to every later guess.
"""

import time
from pathlib import Path

import cv2
import numpy as np

CAL_X = [0.05, 0.275, 0.50, 0.725, 0.95]   # 5 columns (closer to the edges than before)
CAL_Y = [0.07, 0.50, 0.93]                 # 3 rows -> 15 dots
SETTLE_S = 0.8        # time to move your eyes to the dot (nothing recorded)
CAPTURE_S = 1.2       # time while guesses are recorded
MIN_FRAMES = 8        # a dot needs at least this many good frames
MIN_DOTS = 10         # calibration needs at least this many good dots
PROFILE_PATH = Path(__file__).resolve().parent.parent / "models" / "calibration.npz"


def apply_calibration(W, xy):
    """Correct a raw gaze guess (x, y). The result is kept inside the screen."""
    x, y = float(xy[0]), float(xy[1])
    if W is not None:
        x, y = np.array([x, y, 1.0]) @ W
    return float(min(max(x, 0.0), 1.0)), float(min(max(y, 0.0), 1.0))


def save_profile(W, ref):
    np.savez(PROFILE_PATH, W=W, ref=ref)


def load_profile():
    if not PROFILE_PATH.exists():
        return None
    data = np.load(PROFILE_PATH)
    return data["W"], data["ref"]


def _draw(screen_w, screen_h, dot, phase, status):
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cx, cy = int(dot[0] * screen_w), int(dot[1] * screen_h)
    color = (0, 0, 255) if phase == "settle" else (0, 255, 0)   # red = get ready, green = recording
    cv2.circle(canvas, (cx, cy), 16, color, -1)
    cv2.circle(canvas, (cx, cy), 4, (255, 255, 255), -1)
    cv2.putText(canvas, status, (int(screen_w * 0.15), screen_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return canvas


def run_calibration(cap, face_mesh, estimator, gaze_features, screen_w, screen_h):
    """Show 15 dots and fit the correction. Returns (W, ref) or None if it failed / was cancelled."""
    dots = [(x, y) for y in CAL_Y for x in CAL_X]
    window = "NEXUS calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    raw_points, true_points, features = [], [], []
    for index, (tx, ty) in enumerate(dots):
        start = time.time()
        guesses = []
        while True:
            elapsed = time.time() - start
            if elapsed >= SETTLE_S + CAPTURE_S:
                break
            ok, frame = cap.read()
            if not ok:
                cv2.destroyWindow(window)
                return None
            frame = cv2.flip(frame, 1)
            result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            phase = "settle" if elapsed < SETTLE_S else "capture"
            if phase == "capture" and result.multi_face_landmarks:
                face = result.multi_face_landmarks[0]
                guess = estimator.predict(frame, face)
                if guess is not None:
                    guesses.append(guess)
                    features.append(gaze_features(face))
            status = (f"Calibration: dot {index + 1}/{len(dots)} - look at the dot, "
                      f"keep your head still (ESC = cancel)")
            cv2.imshow(window, _draw(screen_w, screen_h, (tx, ty), phase, status))
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(window)
                return None
        if len(guesses) >= MIN_FRAMES:
            raw_points.append(np.median(np.array(guesses), axis=0))
            true_points.append([tx, ty])
    cv2.destroyWindow(window)

    if len(raw_points) < MIN_DOTS:
        return None
    A = np.c_[np.array(raw_points), np.ones(len(raw_points))]
    W, *_ = np.linalg.lstsq(A, np.array(true_points), rcond=None)
    if not (0.3 < W[0, 0] < 4.0 and 0.3 < W[1, 1] < 4.0):    # reject a crazy correction
        return None
    # Remember where your face was (position and size) so we can warn if you move.
    ref = np.median(np.array(features)[:, [7, 8, 9]], axis=0)
    return W.astype(np.float32), ref.astype(np.float32)