"""
NEXUS - gestures/blink_gestures.py
Step 11: turn eye OPEN/CLOSED readings into blink EVENTS.

One event is reported each time the eyes finish closing and open again:
    BLINK        both eyes closed briefly (normal or deliberate blink)
    LEFT_WINK    only your left eye closed for a deliberate time
    RIGHT_WINK   only your right eye closed for a deliberate time

Run from the NEXUS folder (with venv active):
    python -m gestures.blink_gestures
Press 'q' in the video window to quit.
"""

import time

import cv2
import mediapipe as mp

# We reuse the eye measurement code from our previous file.
from gestures.blink_detection import (
    EAR_CLOSED_THRESHOLD,
    SCREEN_LEFT_EYE,
    SCREEN_RIGHT_EYE,
    eye_aspect_ratio,
)

# ---- Timing settings (in seconds). We will tune these by testing. ----
MIN_EYE_CLOSED_S = 0.04   # an eye must be closed at least this long to count
WINK_MIN_S = 0.25         # a wink must last at least this long (ignores natural blinks)
MAX_EPISODE_S = 2.0       # eyes closed longer than this are ignored
EVENT_SHOW_S = 1.5        # how long the last event stays on screen

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


class BlinkEventDetector:
    """Watches both eyes every frame and reports finished blink events."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.in_episode = False      # True while at least one eye is closed
        self.start_time = 0.0
        self.left_closed_s = 0.0     # how long the left eye was closed
        self.right_closed_s = 0.0    # how long the right eye was closed
        self.last_time = None

    def update(self, left_ear, right_ear, now):
        """Call once per frame. Returns (event_name, duration) or None."""
        dt = 0.0 if self.last_time is None else now - self.last_time
        self.last_time = now

        left_closed = left_ear < EAR_CLOSED_THRESHOLD
        right_closed = right_ear < EAR_CLOSED_THRESHOLD

        if left_closed or right_closed:
            if not self.in_episode:              # a new closing has started
                self.in_episode = True
                self.start_time = now
                self.left_closed_s = 0.0
                self.right_closed_s = 0.0
            if left_closed:
                self.left_closed_s += dt
            if right_closed:
                self.right_closed_s += dt
            return None

        # Both eyes are open now.
        if not self.in_episode:
            return None
        self.in_episode = False
        return self._classify(now - self.start_time)

    def _classify(self, episode_s):
        left, right = self.left_closed_s, self.right_closed_s
        if episode_s > MAX_EPISODE_S:
            return None
        if left >= WINK_MIN_S and right < MIN_EYE_CLOSED_S:
            return ("LEFT_WINK", episode_s)
        if right >= WINK_MIN_S and left < MIN_EYE_CLOSED_S:
            return ("RIGHT_WINK", episode_s)
        if left >= MIN_EYE_CLOSED_S and right >= MIN_EYE_CLOSED_S:
            return ("BLINK", episode_s)
        return None                              # unclear movement, ignore it


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

    detector = BlinkEventDetector()
    counts = {"BLINK": 0, "LEFT_WINK": 0, "RIGHT_WINK": 0}
    last_event_text = ""
    last_event_time = 0.0

    print("Blink events started. Press 'q' in the video window to quit.")

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

            event = detector.update(left_ear, right_ear, now)
            if event:
                name, duration = event
                counts[name] += 1
                last_event_text = f"{name} ({duration:.2f} s)"
                last_event_time = now
                print(last_event_text)

            for y, label, ear in ((30, "Left eye", left_ear),
                                  (58, "Right eye", right_ear)):
                closed = ear < EAR_CLOSED_THRESHOLD
                cv2.putText(frame, f"{label}: {'CLOSED' if closed else 'OPEN'}  EAR {ear:.2f}",
                            (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            RED if closed else GREEN, 2)
        else:
            detector.reset()
            cv2.putText(frame, "NO FACE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, RED, 2)

        # Show the last event for a short time
        if last_event_text and now - last_event_time < EVENT_SHOW_S:
            cv2.putText(frame, last_event_text, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, YELLOW, 3)

        cv2.putText(frame,
                    f"Blinks {counts['BLINK']}  L-wink {counts['LEFT_WINK']}  R-wink {counts['RIGHT_WINK']}",
                    (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

        cv2.imshow("NEXUS - Blink Events (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()