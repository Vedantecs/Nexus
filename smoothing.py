"""
NEXUS - cursor/smoothing.py
Makes the gaze cursor steadier.

GazeSmoother: heavy averaging while you look at one place (stops the shaking),
              but a fast reset when you look somewhere new (no lag).
widen():      stretches a position away from the screen centre so corners are reachable.
edge_snap():  pulls a value that is very close to a screen edge onto the edge.
"""

from collections import deque

import numpy as np


class GazeSmoother:
    def __init__(self, window=8, jump_px=250, jump_frames=3):
        self.window = window            # how many recent guesses are averaged
        self.jump_px = jump_px          # a guess this far from the average counts as "far"
        self.jump_frames = jump_frames  # this many far guesses in a row = you looked elsewhere
        self.samples = deque(maxlen=window)
        self.far_count = 0

    def reset(self):
        self.samples.clear()
        self.far_count = 0

    def update(self, x, y):
        """x, y in pixels. Returns the smoothed (x, y) in pixels."""
        if self.samples:
            mean = np.mean(self.samples, axis=0)
            far = np.hypot(x - mean[0], y - mean[1]) > self.jump_px
            self.far_count = self.far_count + 1 if far else 0
        self.samples.append((x, y))
        if self.far_count >= self.jump_frames:
            # You looked somewhere new: keep only the newest guesses.
            recent = list(self.samples)[-self.jump_frames:]
            self.samples = deque(recent, maxlen=self.window)
            self.far_count = 0
        mean = np.mean(self.samples, axis=0)
        return float(mean[0]), float(mean[1])


def widen(x, y, gain):
    """Stretch (x, y), given as fractions 0..1, away from the centre by 'gain'."""
    x = 0.5 + gain * (x - 0.5)
    y = 0.5 + gain * (y - 0.5)
    return min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)


def edge_snap(value, margin):
    """If a 0..1 value is within 'margin' of an edge, move it onto the edge."""
    if value < margin:
        return 0.0
    if value > 1.0 - margin:
        return 1.0
    return value