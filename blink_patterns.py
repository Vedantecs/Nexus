"""
NEXUS - gestures/blink_patterns.py
Step 14: double blink, triple blink and the NEXUS ON/OFF switch.

    Triple blink -> switch NEXUS between ACTIVE and INACTIVE
    Double blink -> SELECT (simulated for now)
    Left wink    -> LEFT CLICK (simulated, only while ACTIVE)
    Right wink   -> RIGHT CLICK (simulated, only while ACTIVE)

Run from the NEXUS folder (with venv active):
    python -m gestures.blink_patterns
Press 'q' in the video window to quit.
"""

import time

import cv2
import mediapipe as mp

from gestures.blink_detection import (
    EAR_CLOSED_THRESHOLD,
    SCREEN_LEFT_EYE,
    SCREEN_RIGHT_EYE,
    eye_aspect_ratio,
)
from gestures.blink_gestures import BlinkEventDetector

# ---- Timing settings (seconds). We will tune these by testing. ----
MAX_GAP_S = 0.6        # max time between two blinks of the same pattern
COOLDOWN_S = 1.0       # ignore new blinks for this long after a pattern
MESSAGE_SHOW_S = 1.5   # how long a message stays on screen

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


class BlinkPatternDetector:
    """Counts quick blinks and reports DOUBLE_BLINK or TRIPLE_BLINK."""

    def __init__(self):
        self.blink_times = []
        self.last_pattern_time = -999.0

    def cancel(self):
        """Forget any blinks counted so far."""
        self.blink_times = []

    def add_blink(self, now):
        """Call when a normal two-eye BLINK happens. Returns TRIPLE_BLINK or None."""
        if now - self.last_pattern_time < COOLDOWN_S:
            return None
        # If the previous blink was too long ago, start counting again.
        if self.blink_times and now - self.blink_times[-1] > MAX_GAP_S:
            self.blink_times = []
        self.blink_times.append(now)
        if len(self.blink_times) >= 3:
            self._finish(now)
            return "TRIPLE_BLINK"
        return None

    def check_timeout(self, now):
        """Call every frame. A double blink is confirmed only after no third blink follows."""
        if self.blink_times and now - self.blink_times[-1] > MAX_GAP_S:
            count = len(self.blink_times)
            self.blink_times = []
            if count == 2:
                self.last_pattern_time = now
                return "DOUBLE_BLINK"
        return None

    def _finish(self, now):
        self.blink_times = []
        self.last_pattern_time = now


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

    events = BlinkEventDetector()
    patterns = BlinkPatternDetector()
    active = False
    message = ""
    message_time = 0.0

    def show(text, now):
        nonlocal message, message_time
        message, message_time = text, now
        print(text)

    print("Blink patterns started. Triple blink = ON/OFF. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: Could not read a frame from the webcam.")
            break

        frame = cv2.flip(frame, 1)
        height, width = frame.shape[:2]
        result = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        now = time.time()

        if result.multi_face_landmarks:
            face = result.multi_face_landmarks[0]
            left_ear = eye_aspect_ratio(face, SCREEN_LEFT_EYE, width, height)
            right_ear = eye_aspect_ratio(face, SCREEN_RIGHT_EYE, width, height)

            pattern = None
            event = events.update(left_ear, right_ear, now)
            if event:
                name, _duration = event
                if name == "BLINK":
                    pattern = patterns.add_blink(now)
                else:
                    patterns.cancel()          # a wink breaks any blink pattern
                    if active:
                        click = "LEFT" if name == "LEFT_WINK" else "RIGHT"
                        show(f"{click} CLICK (simulated)", now)
                    else:
                        show(f"{name} ignored (NEXUS inactive)", now)
            if pattern is None:
                pattern = patterns.check_timeout(now)

            if pattern == "TRIPLE_BLINK":
                active = not active
                show("NEXUS ACTIVATED" if active else "NEXUS DEACTIVATED", now)
            elif pattern == "DOUBLE_BLINK":
                if active:
                    show("DOUBLE BLINK -> SELECT (simulated)", now)
                else:
                    show("DOUBLE BLINK ignored (NEXUS inactive)", now)

            for y, label, ear in ((95, "L", left_ear), (118, "R", right_ear)):
                closed = ear < EAR_CLOSED_THRESHOLD
                cv2.putText(frame, f"{label}: {'CLOSED' if closed else 'OPEN'} {ear:.2f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            RED if closed else GREEN, 1)
        else:
            events.reset()
            patterns.cancel()
            cv2.putText(frame, "NO FACE", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2)

        state_text = "NEXUS: ACTIVE" if active else "NEXUS: INACTIVE"
        cv2.putText(frame, state_text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, GREEN if active else RED, 3)
        cv2.putText(frame, "Triple blink = ON/OFF", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)

        if message and now - message_time < MESSAGE_SHOW_S:
            cv2.putText(frame, message, (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, YELLOW, 2)

        cv2.imshow("NEXUS - Blink Patterns (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()