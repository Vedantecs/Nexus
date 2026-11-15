"""
NEXUS - cursor/gaze_cursor.py
Move the REAL mouse cursor: eyes make the big jump, head movements fine-tune it,
and eye winks click.

Run from the NEXUS folder (with venv active):
    python -m cursor.gaze_cursor

How it works:
    1. Look at a target. If it is far from the cursor, the cursor jumps there.
    2. Move your head slightly to nudge the cursor exactly onto the target.
    3. Wink to click.
Eyes (only while the cursor is ON):
    Left wink    = left click          Right wink    = right click
    Double blink = double click        Triple blink  = cursor ON / OFF
In the video window:
    SPACE = calibrate (15 dots, about 25 seconds)     L = load the last calibration
    R     = calibrate again
SAFETY: HOLD the ESC key (works anywhere on the screen) to quit at any time.
The cursor starts OFF.
"""

import ctypes
import time

import cv2
import mediapipe as mp
import numpy as np

from cursor.calibration import apply_calibration, load_profile, run_calibration, save_profile
from cursor.gaze_estimator import GazeEstimator, gaze_features
from cursor.smoothing import GazeSmoother, edge_snap, widen
from gestures.blink_detection import (
    EAR_CLOSED_THRESHOLD,
    SCREEN_LEFT_EYE,
    SCREEN_RIGHT_EYE,
    eye_aspect_ratio,
)
from gestures.blink_gestures import BlinkEventDetector
from gestures.blink_patterns import BlinkPatternDetector
from training.collect_gaze_random import get_screen_size

# ---- Settings you can tune ----
SMOOTH_WINDOW = 8         # gaze averaging: more = steadier but slower (try 5 to 12)
JUMP_PX = 250             # gaze move bigger than this = "looked somewhere new"
EDGE_GAIN = 1.10          # stretch gaze away from the centre (try 1.0 to 1.25)
EDGE_SNAP = 0.06          # within this fraction of an edge, stick to the edge (0.03 to 0.12)
WARP_PX = 300             # gaze must be farther than this from the cursor to make it jump
WARP_FRAMES = 6           # ...for this many frames in a row (about 0.25 s)
HEAD_GAIN = 5.0           # head nudging strength: higher = cursor moves more (try 3 to 8, 0 = off)
STILL_PX = 4              # the cursor counts as "still" if it moves less than this per frame
NO_FACE_OFF_S = 2.0       # no face for this long -> cursor switches OFF
POST_BLINK_FREEZE_S = 0.15  # keep the cursor still this long after your eyes reopen
REQUIRE_STABLE = True     # a click needs a still cursor first
STABLE_S = 0.2            # the cursor must not have moved for this long
CLICK_COOLDOWN_S = 0.5    # minimum time between clicks
PLACE_WINDOW = True       # move the video window to the bottom-right corner

MAP_W, MAP_H = 320, 180
WINDOW = "NEXUS - Gaze Cursor"
VK_ESCAPE = 0x1B
NOSE_TIP = 1

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

# ---- Windows helpers (mouse, keyboard state, window under the cursor) ----
user32 = ctypes.windll.user32


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


user32.WindowFromPoint.argtypes = [POINT]
user32.WindowFromPoint.restype = ctypes.c_void_p
user32.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
user32.GetAncestor.restype = ctypes.c_void_p
user32.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
user32.GetCursorPos.argtypes = [ctypes.POINTER(POINT)]


def esc_held():
    """True while the ESC key is held down, even if another window is in front."""
    return bool(user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)


def cursor_position():
    point = POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def window_title_at(x, y):
    """Title of the top-level window under the screen point (x, y)."""
    hwnd = user32.WindowFromPoint(POINT(int(x), int(y)))
    if not hwnd:
        return ""
    root = user32.GetAncestor(hwnd, 2)          # 2 = the top-level window
    buffer = ctypes.create_unicode_buffer(256)
    user32.GetWindowTextW(root or hwnd, buffer, 256)
    return buffer.value


def mouse_click(button="left", times=1):
    """Press and release the mouse button at the current cursor position."""
    down, up = (0x0002, 0x0004) if button == "left" else (0x0008, 0x0010)
    for i in range(times):
        user32.mouse_event(down, 0, 0, 0, 0)
        time.sleep(0.03)
        user32.mouse_event(up, 0, 0, 0, 0)
        if i < times - 1:
            time.sleep(0.05)


def main():
    estimator = GazeEstimator()
    screen_w, screen_h = get_screen_size()
    screen_max = np.array([screen_w - 1, screen_h - 1], dtype=np.float64)

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

    events, patterns = BlinkEventDetector(), BlinkPatternDetector()
    smoother = GazeSmoother(window=SMOOTH_WINDOW, jump_px=JUMP_PX)
    W, ref = None, None          # calibration correction and face reference
    active = False               # is the cursor ON?
    smooth = None                # smoothed gaze (x, y) as fractions of the screen
    cursor_pos = None            # where we put the cursor (pixels)
    far_frames = 0               # how long the gaze has been far from the cursor
    nose_prev = None             # nose position in the previous frame
    last_move_time = 0.0         # when the cursor last moved
    last_click_time = 0.0
    freeze_until = 0.0           # cursor stays still until this time
    last_face_time = time.time()
    message, message_time = "", 0.0
    window_placed = False

    print("Gaze cursor started. Click the video window, press SPACE to calibrate.")
    print("SAFETY: hold ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Could not read a frame from the webcam.")
            break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()
        calibrated = W is not None
        warning = ""
        frozen = True
        head_delta = np.zeros(2)

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            last_face_time = now

            # --- head movement since the previous frame (nose tip, in camera pixels) ---
            nose = np.array([face.landmark[NOSE_TIP].x * 640.0, face.landmark[NOSE_TIP].y * 480.0])
            if nose_prev is not None:
                head_delta = nose - nose_prev
            nose_prev = nose

            # --- eyes: freeze the cursor while they are closed (and a moment after) ---
            left_ear = eye_aspect_ratio(face, SCREEN_LEFT_EYE, width, height)
            right_ear = eye_aspect_ratio(face, SCREEN_RIGHT_EYE, width, height)
            if min(left_ear, right_ear) < EAR_CLOSED_THRESHOLD:
                freeze_until = now + POST_BLINK_FREEZE_S
            frozen = now < freeze_until

            # --- gaze guess -> calibration -> stretch -> smoothing ---
            if not frozen:
                pred = estimator.predict(frame, face)
                if pred is not None:
                    x, y = apply_calibration(W, pred)
                    x, y = widen(x, y, EDGE_GAIN)
                    sx, sy = smoother.update(x * screen_w, y * screen_h)
                    smooth = (sx / screen_w, sy / screen_h)

            # --- warn if your head moved a lot since the calibration ---
            if ref is not None:
                f = gaze_features(face)
                if (abs(f[7] - ref[0]) > 0.08 or abs(f[8] - ref[1]) > 0.08
                        or abs(f[9] - ref[2]) / ref[2] > 0.15):
                    warning = "HEAD MOVED - press R to recalibrate"

            # --- blink gestures ---
            pattern, action = None, None
            event = events.update(left_ear, right_ear, now)
            if event:
                if event[0] == "BLINK":
                    pattern = patterns.add_blink(now)
                else:
                    patterns.cancel()           # a wink breaks any blink pattern
                    action = "LEFT" if event[0] == "LEFT_WINK" else "RIGHT"
            if pattern is None:
                pattern = patterns.check_timeout(now)
            if pattern == "DOUBLE_BLINK":
                action = "DOUBLE"

            if pattern == "TRIPLE_BLINK" and calibrated:
                active = not active
                cursor_pos, far_frames = None, 0     # next time ON, start from the gaze
                message, message_time = ("CURSOR ON" if active else "CURSOR OFF"), now

            # --- clicks (only while the cursor is ON) ---
            if action and active and calibrated:
                stable = (not REQUIRE_STABLE) or (now - last_move_time >= STABLE_S)
                if now - last_click_time >= CLICK_COOLDOWN_S:
                    if not stable:
                        message, message_time = "Not stable - hold still", now
                    else:
                        cx, cy = cursor_position()
                        title = window_title_at(cx, cy)
                        if action == "LEFT":
                            mouse_click("left")
                        elif action == "RIGHT":
                            mouse_click("right")
                        else:
                            mouse_click("left", times=2)
                        last_click_time = now
                        shown = title if title else "taskbar/other"
                        if title == WINDOW:
                            message = "CLICK HIT THE NEXUS WINDOW - move it"
                        else:
                            message = f"{action} CLICK on {shown[:24]}"
                        message_time = now
                        print(f"{action} click at ({cx}, {cy}) on window: '{title}'")
        else:
            events.reset()
            patterns.cancel()
            nose_prev = None
            if active and now - last_face_time > NO_FACE_OFF_S:
                active = False
                cursor_pos, far_frames = None, 0
                message, message_time = "No face - cursor OFF", now

        # --- move the real mouse: eyes make big jumps, the head fine-tunes ---
        if active and calibrated and smooth is not None and not frozen:
            gaze_px = np.array([edge_snap(smooth[0], EDGE_SNAP) * screen_max[0],
                                edge_snap(smooth[1], EDGE_SNAP) * screen_max[1]])
            if cursor_pos is None:
                new_pos = gaze_px                                   # first position: go to the gaze
            else:
                far_frames = far_frames + 1 if np.linalg.norm(gaze_px - cursor_pos) > WARP_PX else 0
                if far_frames >= WARP_FRAMES:
                    new_pos = gaze_px                               # you looked far away: jump there
                    far_frames = 0
                else:
                    new_pos = cursor_pos + HEAD_GAIN * head_delta   # head nudges the cursor
            new_pos = np.clip(new_pos, 0.0, screen_max)

            previous = cursor_pos
            cursor_pos = new_pos
            if previous is None or (int(new_pos[0]), int(new_pos[1])) != (int(previous[0]), int(previous[1])):
                user32.SetCursorPos(int(new_pos[0]), int(new_pos[1]))
            if previous is None or np.linalg.norm(new_pos - previous) > STILL_PX:
                last_move_time = now

        # --- drawing: small screen map (red = gaze guess, green = cursor) and status text ---
        x0, y0 = width - MAP_W - 10, 10
        cv2.rectangle(frame, (x0, y0), (x0 + MAP_W, y0 + MAP_H), WHITE, 1)
        if smooth is not None:
            cv2.circle(frame, (x0 + int(smooth[0] * MAP_W), y0 + int(smooth[1] * MAP_H)), 6, RED, -1)
        if active and cursor_pos is not None:
            cv2.circle(frame, (x0 + int(cursor_pos[0] / screen_w * MAP_W),
                               y0 + int(cursor_pos[1] / screen_h * MAP_H)), 5, GREEN, -1)
        cv2.putText(frame, "CURSOR: ON" if active else "CURSOR: OFF", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN if active else RED, 2)
        if not calibrated:
            lines = ["SPACE = calibrate (about 25 s)", "L = load saved calibration", "Hold ESC = quit"]
        else:
            lines = ["Triple blink = cursor ON/OFF", "Eyes jump, head fine-tunes",
                     "L wink = click | R wink = right click", "Double blink = double click",
                     "R = recalibrate | Hold ESC = quit"]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (10, 58 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        if warning:
            cv2.putText(frame, warning, (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
        if message and now - message_time < 2.5:
            cv2.putText(frame, message, (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

        cv2.imshow(WINDOW, frame)
        if PLACE_WINDOW and not window_placed:
            cv2.moveWindow(WINDOW, max(screen_w - 670, 0), max(screen_h - 570, 0))
            window_placed = True
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or esc_held():
            break

        if (key == ord(" ") and not calibrated) or key == ord("r"):
            outcome = run_calibration(cap, face_mesh, estimator, gaze_features, screen_w, screen_h)
            if outcome is None:
                message, message_time = "Calibration failed or cancelled", time.time()
            else:
                W, ref = outcome
                save_profile(W, ref)
                message, message_time = "Calibration done", time.time()
            active, smooth, cursor_pos, far_frames, nose_prev = False, None, None, 0, None
            smoother.reset()
            events.reset()
            patterns.cancel()

        if key == ord("l") and not calibrated:
            profile = load_profile()
            if profile is None:
                message, message_time = "No saved calibration yet", time.time()
            else:
                W, ref = profile
                message, message_time = "Loaded saved calibration", time.time()

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()