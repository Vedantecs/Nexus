"""
NEXUS - dashboard/monitor.py
Draws the NEXUS monitor panel: what NEXUS sees, what it decides, and how fast it runs.
"""

import time

import cv2
import numpy as np

PANEL_W, PANEL_H = 440, 340

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)
GREY = (150, 150, 150)


def _put(img, text, y, color=WHITE, x=8, scale=0.45):
    cv2.putText(img, text[:60], (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_panel(s):
    """s is a dictionary with the current status values (built in app/main_loop.py)."""
    img = np.full((PANEL_H, PANEL_W, 3), 28, dtype=np.uint8)
    _put(img, "NEXUS MONITOR", 18, CYAN, scale=0.55)
    _put(img, f"FPS {s['fps']:.0f}", 18, WHITE, x=PANEL_W - 80)

    calibration = "" if s["calibrated"] else "  (not calibrated)"
    rows = [
        ("AUTH", "not implemented (prototype)", GREY),
        ("MODE", s["mode"], YELLOW if s["mode"] == "TYPING" else GREEN),
        ("CURSOR", ("ON" if s["cursor_on"] else "OFF") + calibration, GREEN if s["cursor_on"] else RED),
        ("GAZE", s["gaze_text"], WHITE),
        ("HEAD", s["head_text"], RED if s["head_warn"] else GREEN),
        ("EYES", s["eyes_text"], WHITE),
        ("HAND", s["hand_text"], GREEN if s["hand_active"] else WHITE),
        ("VOICE", s["voice_text"], YELLOW if s["speaking"] else WHITE),
        ("SPEED", s["speed_text"], WHITE),
    ]
    y = 42
    for label, value, color in rows:
        _put(img, label, y, GREY)
        _put(img, value, y, color, x=72)
        y += 20

    y += 6
    _put(img, "DECISIONS (newest first)", y, CYAN)
    y += 18
    now = time.time()
    for t, text in list(s["log"])[-4:][::-1]:
        _put(img, f"{now - t:4.0f}s  {text}", y)
        y += 18
    _put(img, s["stats_text"], PANEL_H - 8, GREY)
    return img