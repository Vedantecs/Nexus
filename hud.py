"""
NEXUS - dashboard/hud.py
The NEXUS monitor: a clean, instrument-style display of everything NEXUS senses and decides,
with the image-processing pipeline along the bottom.
"""

import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from cursor.gaze_estimator import MODEL_PATH, gaze_features
from gestures.blink_detection import EAR_CLOSED_THRESHOLD
from training.collect_gaze_random import eye_strip

HUD_W, HUD_H = 1280, 880
RENDER_EVERY_N = 2        # redraw the display every N frames (1 = every frame, 3 = lighter on the CPU)
MOUTH_OPEN = 0.08

# layout: (x, y, width, height)
CAM = (24, 76, 640, 480)
EYES = (688, 76, 280, 160)
GAZE = (976, 76, 280, 160)
HAND = (688, 248, 280, 160)
MOUTH = (976, 248, 280, 160)
SYSTEM = (688, 420, 568, 136)
VOICE = (24, 568, 640, 112)
DECISIONS = (688, 568, 568, 112)
PIPE = (24, 692, 1232, 160)

# colours (BGR): a calm, low-saturation palette
BG = (20, 16, 13)
PANEL = (31, 26, 21)
BORDER = (68, 58, 48)
TEXT = (222, 216, 208)
MUTED = (150, 140, 128)
ACCENT = (196, 168, 70)         # steel blue
ACCENT_HI = (232, 208, 120)     # lighter blue for outlines
OK = (120, 190, 100)
WARN = (60, 165, 230)
ALERT = (85, 85, 215)

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _pairs(connections):
    return np.array(sorted(connections), dtype=np.int32)


def _t(img, s, x, y, color=TEXT, scale=0.5, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def _segments(img, pts, pairs, color, thick=1):
    cv2.polylines(img, list(pts[pairs]), False, color, thick, cv2.LINE_AA)


def _panel(img, rect, title):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), PANEL, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), BORDER, 1)
    cv2.line(img, (x, y), (x + 28, y), ACCENT, 2)
    _t(img, title, x + 12, y + 20, MUTED, 0.45)
    cv2.line(img, (x + 12, y + 28), (x + w - 12, y + 28), BORDER, 1)


def _tag(img, text, x, y, color=MUTED):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
    cv2.rectangle(img, (x, y), (x + tw + 12, y + th + 10), PANEL, -1)
    _t(img, text, x + 6, y + th + 4, color, 0.42)


def _bar(img, x, y, w, fraction, color, tick=None):
    fraction = min(max(float(fraction), 0.0), 1.0)
    cv2.rectangle(img, (x, y), (x + w, y + 8), BORDER, 1)
    fill = int((w - 2) * fraction)
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill, y + 7), color, -1)
    if tick is not None:
        tx = x + int(w * tick)
        cv2.line(img, (tx, y - 3), (tx, y + 11), MUTED, 1)


def _chip(img, x, y, label, color):
    (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    cv2.rectangle(img, (x, y), (x + tw + 16, y + 24), color, 1)
    _t(img, label, x + 8, y + 17, color, 0.5)


def _callout(img, anchor, text, color, side, x1, x2):
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
    ax, ay = int(anchor[0]), int(anchor[1])
    if side == "L":
        bx = max(CAM[0] + 8, int(x1) - tw - 14)
        lx = bx + tw + 4
    else:
        bx = min(CAM[0] + CAM[2] - tw - 8, int(x2) + 14)
        lx = bx - 4
    cv2.line(img, (ax, ay), (lx, ay), color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (bx - 3, ay - th - 4), (bx + tw + 3, ay + 4), PANEL, -1)
    _t(img, text, bx, ay, color, 0.42)


def _fit(img, tw=160, th=90):
    """Resize an image to fit a tw x th tile, keeping its shape (black bars around it)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
    tile = np.zeros((th, tw, 3), np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    tile[oy:oy + nh, ox:ox + nw] = resized
    return tile


def _log_color(message):
    m = message.lower()
    if "emergency" in m:
        return ALERT
    if "blocked" in m or "rejected" in m or "ignored" in m or "unsure" in m:
        return WARN
    return TEXT


def _load_feature_stats():
    """Mean and spread of the 12 gaze features, stored inside the trained model file."""
    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        return (np.asarray(checkpoint["G_mean"], np.float32),
                np.asarray(checkpoint["G_std"], np.float32))
    except Exception:
        return None, None


class Hud:
    def __init__(self):
        face, hand = mp.solutions.face_mesh, mp.solutions.hands
        self.oval = _pairs(face.FACEMESH_FACE_OVAL)
        self.eyes = [_pairs(face.FACEMESH_LEFT_EYE), _pairs(face.FACEMESH_RIGHT_EYE)]
        self.lips = _pairs(face.FACEMESH_LIPS)
        self.lip_idx = np.unique(self.lips)
        self.hand_links = _pairs(hand.HAND_CONNECTIONS)
        self.trail = deque(maxlen=24)
        self.f_mean, self.f_std = _load_feature_stats()
        self.static = self._make_static()
        self.last = None
        self._count = 0
        self._pts = None

    @staticmethod
    def _make_static():
        """Everything that never changes is drawn once here."""
        img = np.full((HUD_H, HUD_W, 3), BG, np.uint8)
        _t(img, "NEXUS", 24, 42, TEXT, 1.0, 2)
        cv2.line(img, (132, 18), (132, 46), BORDER, 1)
        _t(img, "Neural Expressive Multimodal User-interface System", 148, 34, TEXT, 0.5)
        _t(img, "Gaze  /  Hand  /  Voice  /  Image processing", 148, 52, MUTED, 0.42)
        cv2.line(img, (0, 62), (HUD_W, 62), BORDER, 1)
        x, y, w, h = CAM
        cv2.rectangle(img, (x - 1, y - 1), (x + w, y + h), BORDER, 1)
        for rect, title in ((EYES, "EYE TRACKING"), (GAZE, "GAZE ESTIMATE"), (HAND, "HAND SCROLL"),
                            (MOUTH, "MOUTH & SPEECH ACTIVITY"), (SYSTEM, "SYSTEM"),
                            (VOICE, "VOICE"), (DECISIONS, "FUSION DECISIONS"),
                            (PIPE, "IMAGE PROCESSING PIPELINE")):
            _panel(img, rect, title)
        _t(img, "HOLD ESC = QUIT   |   SAY 'STOP' = EMERGENCY STOP   |   H = RESIZE", 24, 872, MUTED, 0.4)
        return img

    # ------------------------------------------------------------------
    def render(self, s):
        self._count += 1
        if self.last is not None and self._count % RENDER_EVERY_N:
            return self.last                       # reuse the last picture (saves CPU)
        face = s["face"]
        self._pts = (np.array([(l.x, l.y) for l in face.landmark], np.float32)
                     if face is not None else None)
        img = self.static.copy()
        self._header(img, s)
        self._camera(img, s)
        self._eyes(img, s)
        self._gaze(img, s)
        self._hand(img, s)
        self._mouth(img, s)
        self._system(img, s)
        self._voice(img, s)
        self._decisions(img, s)
        self._pipeline(img, s)
        self.last = img
        return img

    # ------------------------------------------------------------------
    def _header(self, img, s):
        _t(img, time.strftime("%H:%M:%S"), HUD_W - 124, 40, TEXT, 0.7)
        items = [(f"FPS {s['fps']:.0f}", TEXT),
                 ("CURSOR " + ("ON" if s["cursor_on"] else "OFF"), OK if s["cursor_on"] else MUTED),
                 (s["mode"] + " MODE", WARN if s["mode"] == "TYPING" else OK)]
        x = HUD_W - 144
        for label, color in items:                 # drawn from right to left
            (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
            x -= tw + 16
            _chip(img, x, 18, label, color)
            x -= 10

    def _camera(self, img, s):
        x0, y0, w, h = CAM
        frame = s["frame"]
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        img[y0:y0 + h, x0:x0 + w] = frame
        _tag(img, "OPTICAL INPUT  640x480  MIRRORED", x0 + 8, y0 + 8)

        if s["face"] is not None:
            pc = (self._pts * (w, h)).astype(np.int32) + (x0, y0)
            _segments(img, pc, self.oval, ACCENT_HI)
            for eye in self.eyes:
                _segments(img, pc, eye, ACCENT_HI)
            _segments(img, pc, self.lips, ACCENT_HI)
            for p in pc[::5]:
                cv2.circle(img, (int(p[0]), int(p[1])), 1, TEXT, -1)
            if len(pc) > 474:                      # iris rings
                for centre_i, edge_i in ((468, 469), (473, 474)):
                    c = pc[centre_i]
                    r = int(np.linalg.norm(pc[edge_i] - c)) + 2
                    cv2.circle(img, (int(c[0]), int(c[1])), r, ACCENT_HI, 1, cv2.LINE_AA)
                    cv2.circle(img, (int(c[0]), int(c[1])), 1, TEXT, -1)
            lo, hi = pc.min(axis=0) - 14, pc.max(axis=0) + 14
            x1, y1 = max(int(lo[0]), x0 + 4), max(int(lo[1]), y0 + 4)
            x2, y2 = min(int(hi[0]), x0 + w - 4), min(int(hi[1]), y0 + h - 4)
            cv2.rectangle(img, (x1, y1), (x2, y2), ACCENT, 1)
            _tag(img, "FACE", x1, max(y1 - 22, y0 + 30), ACCENT_HI)
            left_eye, right_eye = (pc[33] + pc[133]) / 2, (pc[362] + pc[263]) / 2
            _callout(img, left_eye, f"L EAR {s['left_ear']:.2f}", ACCENT_HI, "L", x1, x2)
            _callout(img, right_eye, f"R EAR {s['right_ear']:.2f}", ACCENT_HI, "R", x1, x2)
            _callout(img, pc[13], f"MAR {s['mar']:.2f}", ACCENT_HI, "R", x1, x2)
        else:
            _t(img, "NO FACE DETECTED", x0 + 220, y0 + 240, ALERT, 0.8, 2)

        hand = s["hand"]
        if hand is not None:
            hp = (np.array([(l.x, l.y) for l in hand.landmark], np.float32)
                  * (w, h)).astype(np.int32) + (x0, y0)
            _segments(img, hp, self.hand_links, OK, 2)
            for p in hp:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, OK, -1)
            info = s["hand_info"]
            if info.get("scroll_mode"):
                for tip in (8, 12):
                    cv2.circle(img, (int(hp[tip][0]), int(hp[tip][1])), 11, WARN, 1, cv2.LINE_AA)
                if info.get("anchor") is not None:
                    ay = int(y0 + info["anchor"] * h)
                    for dx in range(0, w, 18):     # dashed neutral line
                        cv2.line(img, (x0 + dx, ay), (x0 + dx + 9, ay), WARN, 1)
                    _tag(img, "NEUTRAL", x0 + 8, ay - 22, WARN)

    def _eyes(self, img, s):
        x, y, w, h = EYES
        for i, (name, ear) in enumerate((("LEFT EYE", s["left_ear"]), ("RIGHT EYE", s["right_ear"]))):
            top = y + 54 + i * 44
            closed = ear < EAR_CLOSED_THRESHOLD
            color = ALERT if closed else OK
            _t(img, name, x + 14, top, MUTED, 0.45)
            _t(img, f"{ear:.2f}  {'CLOSED' if closed else 'OPEN'}", x + w - 112, top, color, 0.5)
            _bar(img, x + 14, top + 8, w - 28, ear / 0.35, color, EAR_CLOSED_THRESHOLD / 0.35)
        if s["last_event"] and s["now"] - s["event_time"] < 2.5:
            _t(img, "EVENT  " + s["last_event"], x + 14, y + h - 12, WARN, 0.45)
        else:
            _t(img, "EVENT  none", x + 14, y + h - 12, MUTED, 0.45)

    def _gaze(self, img, s):
        x, y, w, h = GAZE
        mx, my, mw, mh = x + 14, y + 38, w - 28, h - 56
        cv2.rectangle(img, (mx, my), (mx + mw, my + mh), BORDER, 1)
        for i in (1, 2):
            cv2.line(img, (mx + i * mw // 3, my), (mx + i * mw // 3, my + mh), BORDER, 1)
            cv2.line(img, (mx, my + i * mh // 3), (mx + mw, my + i * mh // 3), BORDER, 1)
        gaze, cursor = s["gaze"], s["cursor"]
        if gaze is not None:
            self.trail.append(gaze)
        for i, (tx, ty) in enumerate(self.trail):
            cv2.circle(img, (int(mx + tx * mw), int(my + ty * mh)), 1, MUTED, -1)
        if gaze is not None:
            gx, gy = int(mx + gaze[0] * mw), int(my + gaze[1] * mh)
            cv2.circle(img, (gx, gy), 6, ACCENT_HI, 1, cv2.LINE_AA)
            cv2.line(img, (gx - 9, gy), (gx + 9, gy), ACCENT_HI, 1)
            cv2.line(img, (gx, gy - 9), (gx, gy + 9), ACCENT_HI, 1)
        if cursor is not None:
            cx, cy = int(mx + cursor[0] * mw), int(my + cursor[1] * mh)
            cv2.rectangle(img, (cx - 3, cy - 3), (cx + 3, cy + 3), OK, -1)
        _t(img, "circle = gaze estimate    square = cursor", x + 14, y + h - 8, MUTED, 0.38)

    def _hand(self, img, s):
        x, y, w, h = HAND
        tx, top, bottom = x + 34, y + 44, y + h - 16
        mid = (top + bottom) // 2
        cv2.line(img, (tx, top), (tx, bottom), BORDER, 2)
        cv2.line(img, (tx - 12, mid), (tx + 12, mid), WARN, 1)
        info = s["hand_info"]
        text_x = x + 76
        if not info["found"]:
            _t(img, "NO HAND", text_x, y + 80, MUTED, 0.6)
        elif not info["scroll_mode"]:
            _t(img, "HAND DETECTED", text_x, y + 74, TEXT, 0.5)
            _t(img, "show two fingers to scroll", text_x, y + 98, MUTED, 0.42)
        else:
            if info["anchor"] is not None and info["y"] is not None:
                offset = float(np.clip((info["y"] - info["anchor"]) / 0.3, -1, 1))
                cv2.circle(img, (tx, int(mid + offset * (bottom - top) / 2)), 7, OK, -1)
            speed = info["speed"]
            if speed == 0:
                _t(img, "SCROLL MODE", text_x, y + 74, OK, 0.55)
                _t(img, "neutral", text_x, y + 98, MUTED, 0.45)
            else:
                down = speed < 0                   # fingers up -> the page goes down
                _t(img, "PAGE DOWN" if down else "PAGE UP", text_x, y + 74, OK, 0.55)
                _t(img, f"speed {abs(speed):.0f} units/s", text_x, y + 98, TEXT, 0.45)

    def _mouth(self, img, s):
        x, y, w, h = MOUTH
        mar = s["mar"]
        _t(img, "MOUTH ASPECT RATIO", x + 14, y + 54, MUTED, 0.45)
        _t(img, f"{mar:.2f}", x + w - 60, y + 54, TEXT, 0.55)
        _bar(img, x + 14, y + 62, w - 28, mar / 0.7, ACCENT, MOUTH_OPEN / 0.7)
        mouth_open = mar > MOUTH_OPEN
        speaking = s["voice_state"] == "speaking"
        for i, (label, on) in enumerate((("MOUTH", mouth_open), ("VOICE", speaking),
                                         ("SYNC", mouth_open and speaking))):
            cx, cy = x + 32 + i * 86, y + 112
            cv2.circle(img, (cx, cy), 6, OK if on else BORDER, -1 if on else 1, cv2.LINE_AA)
            _t(img, label, cx + 12, cy + 5, TEXT if on else MUTED, 0.45)
        _t(img, "speech activity = mouth movement + microphone", x + 14, y + h - 10, MUTED, 0.38)

    def _system(self, img, s):
        x, y, w, h = SYSTEM
        top = [("MODE", s["mode"], WARN if s["mode"] == "TYPING" else OK),
               ("CURSOR", "ON" if s["cursor_on"] else "OFF", OK if s["cursor_on"] else MUTED),
               ("CALIBRATION", "OK" if s["calibrated"] else "NOT DONE", OK if s["calibrated"] else WARN),
               ("AUTHENTICATION", "N/A", MUTED),
               ("HEAD", "MOVED" if s["head_warn"] else "OK", WARN if s["head_warn"] else OK)]
        bottom = [("FPS", f"{s['fps']:.0f}"), ("LOOP", f"{s['loop_ms']:.0f} ms"),
                  ("GAZE MODEL", f"{s['gaze_ms']:.0f} ms"), ("WHISPER", f"{s['whisper_s']:.1f} s"),
                  ("CLICKS", str(s["counts"]["clicks"]))]
        for i, (label, value, color) in enumerate(top):
            _t(img, label, x + 14 + i * 110, y + 52, MUTED, 0.4)
            _t(img, value, x + 14 + i * 110, y + 74, color, 0.58)
        for i, (label, value) in enumerate(bottom):
            _t(img, label, x + 14 + i * 110, y + 100, MUTED, 0.4)
            _t(img, value, x + 14 + i * 110, y + 122, TEXT, 0.58)

    def _voice(self, img, s):
        x, y, w, h = VOICE
        state = s["voice_state"]
        color = {"speaking": WARN, "processing": ACCENT_HI, "listening": ACCENT,
                 "loading": MUTED, "off": MUTED, "error": ALERT}[state]
        threshold = max(s["voice_threshold"], 1e-4)
        base = y + 66
        cv2.line(img, (x + 14, base), (x + 14 + 64 * 7, base), BORDER, 1)
        for i, level in enumerate(list(s["voice_levels"])[-64:]):
            hgt = int(min(level / (threshold * 6), 1.0) * 22) + 1
            bx = x + 16 + i * 7
            cv2.line(img, (bx, base - hgt), (bx, base + hgt), color, 2)
        _chip(img, x + 500, y + 38, state.upper(), color)
        target = s["target_title"] or "-"
        _t(img, "TARGET  " + target[:18], x + 500, y + 82, OK if s["target_ok"] else WARN, 0.42)
        if s["voice_text"]:
            shown = s["voice_text"][:max(0, int((s["now"] - s["voice_time"]) * 45))]
            blink = "|" if int(s["now"] * 2) % 2 == 0 else " "
            _t(img, f"> {shown[:60]}{blink}   [{s['voice_conf']:.2f}]", x + 14, y + 104, TEXT, 0.48)
        else:
            _t(img, "> say 'start typing' ...", x + 14, y + 104, MUTED, 0.48)

    def _decisions(self, img, s):
        x, y, w, h = DECISIONS
        for i, (t, message) in enumerate(list(s["log"])[-4:][::-1]):
            _t(img, f"{s['now'] - t:3.0f}s   {message}"[:68], x + 14, y + 50 + i * 18,
               _log_color(message), 0.46)

    # ------------------------------------------------------------------
    def _pipeline(self, img, s):
        x, y, w, h = PIPE
        frame, face = s["frame"], s["face"]
        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blank = np.zeros((90, 160, 3), np.uint8)
        tiles = [("01  INPUT", _fit(frame), "camera frame, mirrored"),
                 ("02  GRAYSCALE", _fit(gray), "single channel")]

        # 03 landmarks (face points and hand skeleton on black)
        lm = blank.copy()
        caption = "no landmarks"
        if self._pts is not None:
            px = np.clip((self._pts[:, 0] * 120 + 20).astype(int), 0, 159)
            py = np.clip((self._pts[:, 1] * 90).astype(int), 0, 89)
            lm[py, px] = ACCENT_HI
            caption = f"{len(self._pts)} face pts"
        if s["hand"] is not None:
            hp = np.array([(l.x, l.y) for l in s["hand"].landmark], np.float32)
            hx, hy = (hp[:, 0] * 120 + 20).astype(int), (hp[:, 1] * 90).astype(int)
            for a, b in self.hand_links:
                cv2.line(lm, (int(hx[a]), int(hy[a])), (int(hx[b]), int(hy[b])), OK, 1)
            caption += " + 21 hand pts" if self._pts is not None else "21 hand pts"
        tiles.append(("03  LANDMARKS", lm, caption))

        # 04 eye region and 05 gaze features (the model inputs)
        roi_tile, feat_tile = blank.copy(), blank.copy()
        roi_caption, feat_caption = "no face", "no face"
        if face is not None:
            strip = eye_strip(gray, face, fw, fh)
            if strip is not None:
                roi = cv2.resize(strip, (160, 60), interpolation=cv2.INTER_NEAREST)
                roi_tile[15:75] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                roi_caption = "128x48 gray, CNN input"
            if self.f_mean is not None:
                z = np.clip((gaze_features(face) - self.f_mean) / self.f_std, -3, 3)
                cv2.line(feat_tile, (0, 45), (159, 45), BORDER, 1)
                for i, v in enumerate(z):
                    bx, hgt = 4 + i * 13, int(abs(float(v)) / 3 * 40)
                    if v >= 0:
                        cv2.rectangle(feat_tile, (bx, 45 - hgt), (bx + 9, 45), ACCENT, -1)
                    else:
                        cv2.rectangle(feat_tile, (bx, 45), (bx + 9, 45 + hgt), WARN, -1)
                feat_caption = "12 gaze features (z-score)"
        tiles.append(("04  EYE REGION", roi_tile, roi_caption))
        tiles.append(("05  GAZE FEATURES", feat_tile, feat_caption))

        # 06 mouth region
        mouth_tile, mouth_caption = blank.copy(), "no face"
        if self._pts is not None:
            lip = self._pts[self.lip_idx] * (fw, fh)
            x1, y1 = lip.min(axis=0)
            x2, y2 = lip.max(axis=0)
            pad = 0.35 * (x2 - x1)
            crop = frame[max(int(y1 - pad), 0):min(int(y2 + pad), fh),
                         max(int(x1 - pad), 0):min(int(x2 + pad), fw)]
            if crop.size:
                mouth_tile = _fit(crop)
                mouth_caption = f"mouth ratio {s['mar']:.2f}"
        tiles.append(("06  MOUTH REGION", mouth_tile, mouth_caption))

        # 07 brightness histogram (a quick check of the lighting)
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten()
        hist = hist / max(float(hist.max()), 1.0)
        hist_tile = blank.copy()
        for i, v in enumerate(hist):
            hgt = int(v * 80)
            cv2.rectangle(hist_tile, (4 + i * 5, 88 - hgt), (7 + i * 5, 88), ACCENT, -1)
        tiles.append(("07  LUMINANCE", hist_tile, f"mean {gray.mean():.0f} of 255"))

        for i, (label, tile, caption) in enumerate(tiles):
            tx, ty = x + 20 + i * 172, y + 52
            img[ty:ty + 90, tx:tx + 160] = tile
            cv2.rectangle(img, (tx - 1, ty - 1), (tx + 160, ty + 90), BORDER, 1)
            _t(img, label, tx, y + 46, TEXT, 0.45)
            _t(img, caption[:32], tx, y + 154, MUTED, 0.38)
            if i < len(tiles) - 1:                 # arrow to the next stage
                ax, ay = tx + 166, ty + 45
                cv2.polylines(img, [np.array([[ax - 2, ay - 5], [ax + 3, ay], [ax - 2, ay + 5]], np.int32)],
                              False, MUTED, 1, cv2.LINE_AA)