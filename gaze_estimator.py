"""
NEXUS - cursor/gaze_estimator.py
Loads the trained gaze CNN (models/gaze_model.pt) and turns a webcam frame plus
MediaPipe face landmarks into an estimated screen position (x, y), where
(0, 0) is the top-left of the screen and (1, 1) is the bottom-right.

The eye-strip crop and the landmark numbers come from the SAME code used to
collect the training data (training/collect_gaze_random.py), so live input is
prepared exactly like the training input.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

from training.collect_gaze_random import eye_strip, landmark_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "gaze_model.pt"


class GazeNet(nn.Module):
    """Same structure as the model trained in Colab (eye image + gaze features)."""

    def __init__(self, n_feats):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2))

        self.cnn = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 64))
        self.fc_img = nn.Sequential(nn.Flatten(), nn.Dropout(0.3),
                                    nn.Linear(64 * 3 * 8, 128), nn.ReLU())
        self.fc_feat = nn.Sequential(nn.Linear(n_feats, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + 64, 128), nn.ReLU(),
                                  nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, img, feats):
        return self.head(torch.cat([self.fc_img(self.cnn(img)), self.fc_feat(feats)], dim=1))


def gaze_features(face):
    """The 12 gaze features (iris position in each eye, head yaw/pitch/roll, face position and size)."""
    f = np.array(landmark_features(face), dtype=np.float32).reshape(11, 2)
    f = f * np.array([640.0, 480.0], dtype=np.float32)       # same scaling as training
    nose, fore, chin, side_a, side_b = f[0], f[1], f[2], f[3], f[4]
    ea_out, ea_in, eb_in, eb_out, iris_a, iris_b = f[5], f[6], f[7], f[8], f[9], f[10]

    def iris_offset(c1, c2, iris):
        axis = c2 - c1
        length = float(np.linalg.norm(axis)) + 1e-6
        u = axis / length
        n = np.array([-u[1], u[0]], dtype=np.float32)
        rel = iris - (c1 + c2) / 2
        return float(rel @ u) / length, float(rel @ n) / length, length

    ta, sa, la = iris_offset(ea_out, ea_in, iris_a)
    tb, sb, lb = iris_offset(eb_in, eb_out, iris_b)

    face_w = float(np.linalg.norm(side_b - side_a)) + 1e-6
    face_h = float(np.linalg.norm(chin - fore)) + 1e-6
    mid_sides = (side_a + side_b) / 2
    mid_vert = (fore + chin) / 2
    yaw = (nose[0] - mid_sides[0]) / face_w
    pitch = (nose[1] - mid_vert[1]) / face_h
    roll = np.arctan2(side_b[1] - side_a[1], side_b[0] - side_a[0])
    cx, cy = mid_sides[0] / 640.0, mid_vert[1] / 480.0

    return np.array([ta, sa, tb, sb, yaw, pitch, roll, cx, cy,
                     face_w / 640.0, la / face_w, lb / face_w], dtype=np.float32)


class GazeEstimator:
    def __init__(self, model_path=MODEL_PATH):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.g_mean = np.asarray(checkpoint["G_mean"], dtype=np.float32)
        self.g_std = np.asarray(checkpoint["G_std"], dtype=np.float32)
        self.model = GazeNet(len(self.g_mean))
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def predict(self, frame, face):
        """frame: the MIRRORED BGR camera image. Returns (x, y) or None."""
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        strip = eye_strip(gray, face, width, height)
        if strip is None:
            return None
        img = torch.from_numpy(strip).float().div(255.0).sub(0.5).div(0.5)
        img = img.unsqueeze(0).unsqueeze(0)                       # (1, 1, 48, 128)
        feats = (gaze_features(face) - self.g_mean) / self.g_std
        feats = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.model(img, feats)[0]
        return float(out[0]), float(out[1])