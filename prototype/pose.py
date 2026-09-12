"""Pose estimation (MediaPipe PoseLandmarker / BlazePose) + geometry helpers.

Everything below `PoseEstimator` is plain math on keypoints, so it ports 1:1
to TypeScript in the React Native app.
"""
from __future__ import annotations

import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

MODEL_DIR = Path(__file__).parent / "models"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task")

VIS_THRESHOLD = 0.5

# BlazePose landmark indices (33 total). "Left" is the person's left.
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28

SIDES = {
    "left": dict(shoulder=L_SHOULDER, elbow=L_ELBOW, wrist=L_WRIST, hip=L_HIP, knee=L_KNEE, ankle=L_ANKLE),
    "right": dict(shoulder=R_SHOULDER, elbow=R_ELBOW, wrist=R_WRIST, hip=R_HIP, knee=R_KNEE, ankle=R_ANKLE),
}

SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST), (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]


def ensure_model(variant: str) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"pose_landmarker_{variant}.task"
    if not path.exists():
        print(f"Downloading {path.name} ...")
        urllib.request.urlretrieve(MODEL_URL.format(v=variant), path)
    return path


@dataclass
class Pose:
    pts: np.ndarray  # (33, 2) pixel coords, or (33, 3) world coords in metres when using 3D angles
    px: np.ndarray   # (33, 2) pixel coords, for drawing
    vis: np.ndarray  # (33,) visibility 0..1

    def visible(self, *idx: int) -> bool:
        return all(self.vis[i] >= VIS_THRESHOLD for i in idx)

    def best_side(self, *joints: str) -> dict | None:
        """Joint map of the side whose `joints` are most visible, or None if neither side is usable."""
        side = max(SIDES.values(), key=lambda s: sum(self.vis[s[j]] for j in joints))
        return side if self.visible(*(side[j] for j in joints)) else None


class PoseEstimator:
    def __init__(self, variant: str = "full", use_3d: bool = False):
        options = vision.PoseLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_model(variant))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._use_3d = use_3d
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Pose | None:
        ts = max(int(timestamp_ms), self._last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
        self._last_ts = ts
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
        if not result.pose_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        image_lms = result.pose_landmarks[0]
        px = np.array([[lm.x * w, lm.y * h] for lm in image_lms], dtype=np.float32)
        vis = np.array([lm.visibility or 0.0 for lm in image_lms], dtype=np.float32)
        if self._use_3d:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]], dtype=np.float32)
        else:
            pts = px
        return Pose(pts, px, vis)

    def close(self) -> None:
        self._landmarker.close()


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees, at vertex b. Works for 2D or 3D points."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def angle_from_vertical(top: np.ndarray, bottom: np.ndarray) -> float:
    """Angle in degrees between segment bottom->top and straight up (0 = upright, 90 = horizontal)."""
    v = top - bottom
    cos = -v[1] / (np.linalg.norm(v) + 1e-9)  # "up" is -y in both image and MediaPipe world coords
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


class EMA:
    """Exponential moving average to damp keypoint jitter."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, x: float) -> float:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value
        return self.value
