"""
NEXUS - fusion/fusion_engine.py
The decision layer. The perception modules (gaze, blinks, hand, voice) send EVENTS here;
the fusion engine applies the safety rules and returns the ACTION to perform, or None.
Every decision is logged for the HUD and for testing.
"""

import time
from collections import deque, namedtuple

Action = namedtuple("Action", ["kind", "data"])

MIN_CONFIDENCE = 0.35       # voice below this is ignored
LOCK_CONFIDENCE = 0.70      # locking the PC needs a clearer voice
CLICK_COOLDOWN_S = 0.5      # minimum time between clicks
STABLE_S = 0.2              # the cursor must have been still this long before a click
SYSTEM_COMMANDS = ("VOLUME_UP", "VOLUME_DOWN", "MUTE", "UNDO", "REDO", "ADDRESS_BAR")
CLICK_COMMANDS = {"CLICK": "LEFT", "DOUBLE_CLICK": "DOUBLE", "RIGHT_CLICK": "RIGHT"}


class FusionEngine:
    def __init__(self):
        self.calibrated = False
        self.cursor_on = False
        self.mode = "CONTROL"                  # CONTROL or TYPING
        self.log = deque(maxlen=20)            # recent decisions: (time, text)
        self.counts = {"clicks": 0, "rejected": 0, "typed_phrases": 0, "voice_commands": 0}
        self._last_click = 0.0
        self._typed = []                       # length of each typed phrase (for "delete that")

    @property
    def cursor_active(self):
        """The cursor follows the gaze only when it is ON, calibrated and not in TYPING mode."""
        return self.cursor_on and self.calibrated and self.mode == "CONTROL"

    def note(self, text):
        self.log.append((time.time(), text))
        print(f"[fusion] {text}")

    def reject(self, text):
        self.counts["rejected"] += 1
        self.note(text)

    # ---------------- state changes ----------------
    def emergency_stop(self, reason):
        self.cursor_on = False
        self.mode = "CONTROL"
        self._typed.clear()
        self.note(f"EMERGENCY STOP ({reason})")

    def on_triple_blink(self):
        if not self.calibrated:
            self.note("triple blink ignored (calibrate first)")
            return
        self.cursor_on = not self.cursor_on
        self.note("cursor ON" if self.cursor_on else "cursor OFF")

    def no_face_timeout(self):
        if self.cursor_on:
            self.cursor_on = False
            self.note("no face: cursor OFF")

    # ---------------- clicks (eyes or voice) ----------------
    def on_click_gesture(self, gesture, now, cursor_still_s):
        """gesture is 'LEFT', 'RIGHT' or 'DOUBLE'. Returns an Action or None."""
        if not self.cursor_active:
            self.reject(f"{gesture} click ignored (cursor not active)")
            return None
        if now - self._last_click < CLICK_COOLDOWN_S:
            return None
        if cursor_still_s < STABLE_S:
            self.reject(f"{gesture} click rejected (cursor not stable)")
            return None
        self._last_click = now
        self.counts["clicks"] += 1
        self.note(f"{gesture} click")
        return Action("click", gesture)

    # ---------------- hand scrolling ----------------
    def on_hand_scroll(self, wheel_units):
        return Action("wheel", wheel_units) if wheel_units else None

    # ---------------- voice ----------------
    def on_voice(self, text, confidence, command, typing_ok, window_title, cursor_still_s):
        """Decide what to do with one spoken phrase. Returns an Action or None."""
        if len(text) < 2 or confidence < MIN_CONFIDENCE:
            self.note(f"voice ignored (unsure {confidence:.2f}): {text[:25]!r}")
            return None
        if command == "STOP":
            self.emergency_stop("voice: stop")
            return Action("beep", "low")
        if command in ("HUD_BIG", "HUD_SMALL"):
            self.note("HUD " + ("big" if command == "HUD_BIG" else "small"))
            return Action("hud", "BIG" if command == "HUD_BIG" else "SMALL")
        if self.mode == "CONTROL":
            return self._voice_control(text, confidence, command, cursor_still_s)
        return self._voice_typing(text, command, typing_ok, window_title)

    def _voice_control(self, text, confidence, command, cursor_still_s):
        if command == "START_TYPING":
            self.mode = "TYPING"
            self.note("TYPING mode (cursor paused)")
            return Action("beep", "high")
        if command == "CALIBRATE":
            self.cursor_on = False
            self.note("voice: calibrate")
            return Action("calibrate", None)
        if command in CLICK_COMMANDS:
            return self.on_click_gesture(CLICK_COMMANDS[command], time.time(), cursor_still_s)
        if command in SYSTEM_COMMANDS:
            self.counts["voice_commands"] += 1
            self.note(f"voice command: {command}")
            return Action("system", command)
        if command == "LOCK":
            if confidence < LOCK_CONFIDENCE:
                self.reject(f"lock rejected (unsure {confidence:.2f})")
                return None
            self.counts["voice_commands"] += 1
            self.note("voice command: LOCK")
            return Action("system", "LOCK")
        self.note(f"voice ignored in CONTROL mode: {text[:25]!r}")
        return None

    def _voice_typing(self, text, command, typing_ok, window_title):
        if command == "STOP_TYPING":
            self.mode = "CONTROL"
            self._typed.clear()
            self.note("CONTROL mode")
            return Action("beep", "low")
        if not typing_ok:
            self.reject(f"typing BLOCKED (window: {window_title[:20]})")
            return None
        if command == "NEW_LINE":
            self._typed.append(1)
            self.note("new line")
            return Action("key", "ENTER")
        if command == "DELETE_LAST":
            if not self._typed:
                return None
            self.note("delete last phrase")
            return Action("backspace", self._typed.pop())
        phrase = text + " "
        self._typed.append(len(phrase))
        self.counts["typed_phrases"] += 1
        self.note(f"typed: {text[:30]}")
        return Action("type", phrase)