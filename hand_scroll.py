"""
NEXUS - gestures/hand_scroll.py
Autoscroll with two fingers, like the middle mouse button.

    Show TWO FINGERS (index + middle up, like a V sign) and hold still:
    that spot becomes the neutral line.
    Move the fingers UP   from the line -> the page keeps scrolling DOWN (faster the further you go).
    Move the fingers DOWN from the line -> the page keeps scrolling UP.
    Back at the line -> stops.   Fist / hand out of view -> stops.

Run from the NEXUS folder (with venv active):
    python -m gestures.hand_scroll
Put your normal mouse pointer over a long web page first.
Press 'q' in the video window (or HOLD ESC) to quit.
"""

import ctypes
import time

import cv2
import mediapipe as mp
import numpy as np

# ---- Settings you can tune ----
SEND_SCROLL = True     # False = only show detection, do not scroll anything
INVERT = False         # True = swap the directions
DEAD_BAND = 0.06       # no scrolling this close to the neutral line (fraction of image height)
GAIN = 5000.0          # scroll speed per distance from the line (wheel units per second)
MAX_SPEED = 2000.0     # fastest scroll speed (wheel units per second; 120 = one wheel notch)
SMOOTH = 0.6           # 0..1, lower = smoother fingertip position
CHUNK = 40             # mouse-wheel units sent at a time
GRACE_S = 0.25         # keep scroll mode this long if the two-finger pose is lost for a moment
EXTEND_RATIO = 1.45    # a finger counts as "up" if its tip is this much farther from the wrist than its base

user32 = ctypes.windll.user32
VK_ESCAPE = 0x1B

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREY = (150, 150, 150)
WHITE = (255, 255, 255)


def esc_held():
    return bool(user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)


def mouse_wheel(units):
    """Positive = scroll up, negative = scroll down."""
    user32.mouse_event(0x0800, 0, 0, int(units), 0)


class HandScroller:
    """Watches one hand and says how fast to scroll."""

    def __init__(self):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.last_pose_time = -1e9
        self.last_time = None
        self._reset()

    def _reset(self):
        self.y_smooth = None
        self.anchor = None          # the neutral line (fraction of image height)
        self.accumulated = 0.0

    def process(self, rgb_frame, now):
        """Returns a dict: found, scroll_mode, hand, anchor, y, speed, wheel (units to send now)."""
        info = {"found": False, "scroll_mode": False, "hand": None,
                "anchor": None, "y": None, "x": None, "speed": 0.0, "wheel": 0}
        dt = 0.0 if self.last_time is None else min(now - self.last_time, 0.1)
        self.last_time = now

        result = self.hands.process(rgb_frame)
        if not result.multi_hand_landmarks:          # hand out of view -> stop
            self._reset()
            return info

        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark
        info["found"], info["hand"] = True, hand

        def point(i):
            return np.array([lm[i].x * 640.0, lm[i].y * 480.0, lm[i].z * 640.0])

        wrist = point(0)

        def extended(tip, base):         # fingertip much farther from the wrist than the finger base
            return np.linalg.norm(point(tip) - wrist) > EXTEND_RATIO * np.linalg.norm(point(base) - wrist)

        pose_ok = (extended(8, 5) and extended(12, 9)
                   and not extended(16, 13) and not extended(20, 17))
        if pose_ok:
            self.last_pose_time = now
        in_mode = pose_ok or (self.anchor is not None and now - self.last_pose_time < GRACE_S)
        info["scroll_mode"] = in_mode
        if not in_mode:                              # fist or other pose -> stop
            self._reset()
            return info

        tip_y = (lm[8].y + lm[12].y) / 2             # average height of the two fingertips
        self.y_smooth = tip_y if self.y_smooth is None else SMOOTH * tip_y + (1 - SMOOTH) * self.y_smooth
        if self.anchor is None:
            self.anchor = self.y_smooth              # the first position becomes the neutral line

        offset = self.y_smooth - self.anchor         # positive = fingers moved down
        excess = abs(offset) - DEAD_BAND
        speed = 0.0
        if excess > 0:
            speed = min(GAIN * excess, MAX_SPEED) * (1.0 if offset > 0 else -1.0)
            if INVERT:
                speed = -speed
        # fingers up -> negative -> page scrolls down; fingers down -> positive -> page scrolls up

        self.accumulated = self.accumulated + speed * dt if speed != 0 else 0.0
        wheel = 0
        while abs(self.accumulated) >= CHUNK:
            step = CHUNK if self.accumulated > 0 else -CHUNK
            wheel += step
            self.accumulated -= step

        info.update(anchor=self.anchor, y=self.y_smooth, x=(lm[8].x + lm[12].x) / 2,
                    speed=speed, wheel=int(wheel))
        return info


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("ERROR: Could not open the webcam.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    scroller = HandScroller()
    drawing = mp.solutions.drawing_utils
    connections = mp.solutions.hands.HAND_CONNECTIONS
    previous_time = time.time()
    last_direction = None

    print("Hand autoscroll started. Show two fingers, hold still, then move up or down. Press q or hold ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Could not read a frame from the webcam.")
            break
        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        now = time.time()
        info = scroller.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), now)

        if info["hand"] is not None:
            drawing.draw_landmarks(frame, info["hand"], connections)
        if info["wheel"] != 0 and SEND_SCROLL:
            mouse_wheel(info["wheel"])

        speed = info["speed"]
        direction = "UP" if speed > 0 else "DOWN" if speed < 0 else None
        if direction != last_direction:
            print(f"Scrolling {direction}" if direction else "Stopped")
            last_direction = direction

        # neutral line, dead band and fingertip marker
        if info["anchor"] is not None:
            ay = int(info["anchor"] * height)
            band = int(DEAD_BAND * height)
            cv2.line(frame, (0, ay), (width, ay), YELLOW, 2)
            cv2.line(frame, (0, ay - band), (width, ay - band), GREY, 1)
            cv2.line(frame, (0, ay + band), (width, ay + band), GREY, 1)
            cv2.circle(frame, (int(info["x"] * width), int(info["y"] * height)), 9, GREEN, -1)

        if not info["found"]:
            status, color = "NO HAND - stopped", RED
        elif info["scroll_mode"]:
            status, color = "SCROLL MODE (two fingers)", GREEN
        else:
            status, color = "Hand seen - show TWO FINGERS", WHITE
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "Fingers ABOVE the yellow line = page DOWN", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        cv2.putText(frame, "Fingers BELOW the line = page UP | back at the line = stop", (10, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        if direction:
            cv2.putText(frame, f"SCROLLING {'UP' if direction == 'UP' else 'DOWN'}  ({abs(speed):.0f}/s)",
                        (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, YELLOW, 2)
        fps = 1.0 / max(now - previous_time, 1e-6)
        previous_time = now
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        cv2.imshow("NEXUS - Hand Autoscroll (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q") or esc_held():
            break

    scroller.hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()