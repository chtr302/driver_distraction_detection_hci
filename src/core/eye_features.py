"""Per-frame eye + head features for the drowsiness model (runs on the Pi: numpy only).

One implementation for training and runtime, so the model sees the same numbers in both places:
  raw_features()  -> 13 raw values per frame (8 blendshapes, EAR L/R, pitch/yaw/roll in degrees)
  assemble()      -> 15 model inputs: raw minus the person's baseline, pitch rate (deg/s), face flag
  EyeFeatureStream wraps both for one frame at a time (keeps the previous pitch for the rate).
Face lost: all features 0 and flag 0, so the model can keep running through the gap.
"""
from __future__ import annotations

import numpy as np

BLEND_FEATURES = ["eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
                  "eyeWideLeft", "eyeWideRight", "eyeLookDownLeft", "eyeLookDownRight"]
RAW_NAMES = BLEND_FEATURES + ["earLeft", "earRight", "pitch", "yaw", "roll"]
FEATURE_NAMES = [f"d_{n}" for n in RAW_NAMES] + ["pitch_rate", "face_found"]
NUM_RAW, NUM_FEATURES = len(RAW_NAMES), len(FEATURE_NAMES)  # 13, 15

# MediaPipe 478-point mesh, 6 points per eye in the usual EAR order p1..p6 (corner, top, top, corner, bottom, bottom).
EAR_LEFT = [362, 385, 387, 263, 373, 380]
EAR_RIGHT = [33, 160, 158, 133, 153, 144]
EAR_POINTS = EAR_LEFT + EAR_RIGHT  # the 12 landmarks raw_features() needs, in this order

RATE_MAX_GAP_S = 0.5  # pitch rate is 0 when the previous face-found frame is older than this


def _ear(p: np.ndarray) -> np.ndarray:
    """p: (N, 6, 2) pixel coordinates -> (N,) eye aspect ratio."""
    v = np.linalg.norm(p[:, 1] - p[:, 5], axis=1) + np.linalg.norm(p[:, 2] - p[:, 4], axis=1)
    h = np.linalg.norm(p[:, 0] - p[:, 3], axis=1)
    return np.where(h > 0, v / (2 * np.maximum(h, 1e-9)), 0.0)


def pose_deg(tf: np.ndarray) -> np.ndarray:
    """tf: (N, 16) row-major 4x4 face transform -> (N, 3) pitch, yaw, roll in degrees."""
    r = tf.reshape(-1, 4, 4)[:, :3, :3]
    r = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-9)  # drop scale from the columns
    pitch = np.arctan2(r[:, 2, 1], r[:, 2, 2])
    yaw = np.arctan2(-r[:, 2, 0], np.hypot(r[:, 2, 1], r[:, 2, 2]))
    roll = np.arctan2(r[:, 1, 0], r[:, 0, 0])
    return np.degrees(np.stack([pitch, yaw, roll], axis=1))


def raw_features(blend: np.ndarray, eye_xy: np.ndarray, tf: np.ndarray, width: int, height: int) -> np.ndarray:
    """blend (N, 8) in BLEND_FEATURES order, eye_xy (N, 12, 2) normalized EAR_POINTS, tf (N, 16) -> (N, 13)."""
    px = eye_xy * np.array([width, height], dtype=np.float64)
    ear = np.stack([_ear(px[:, :6]), _ear(px[:, 6:])], axis=1)
    return np.concatenate([blend, ear, pose_deg(tf)], axis=1)


def baseline(raw: np.ndarray, found: np.ndarray) -> np.ndarray:
    """Median of the raw features over calibration frames with a face -> (13,). Awake frames only, by the caller."""
    sel = raw[found.astype(bool)]
    if len(sel) == 0:
        raise ValueError("no face in the calibration frames")
    return np.median(sel, axis=0)


def pitch_rate(pitch: np.ndarray, t_ms: np.ndarray, found: np.ndarray) -> np.ndarray:
    """deg/s against the previous face-found frame; 0 at gaps longer than RATE_MAX_GAP_S or without a face."""
    out = np.zeros(len(pitch))
    last_p, last_t = None, None
    for i in range(len(pitch)):
        if not found[i]:
            continue
        if last_t is not None and 0 < t_ms[i] - last_t <= RATE_MAX_GAP_S * 1000:
            out[i] = (pitch[i] - last_p) / ((t_ms[i] - last_t) / 1000)
        last_p, last_t = pitch[i], t_ms[i]
    return out


def assemble(raw: np.ndarray, t_ms: np.ndarray, found: np.ndarray, base: np.ndarray) -> np.ndarray:
    """(N, 13) raw -> (N, 15) model inputs."""
    f = found.astype(bool)
    x = np.zeros((len(raw), NUM_FEATURES), dtype=np.float32)
    x[:, :NUM_RAW] = raw - base
    x[:, NUM_RAW] = pitch_rate(raw[:, 10], t_ms, f)
    x[:, NUM_RAW + 1] = f
    x[~f, :] = 0.0
    return x


class EyeFeatureStream:
    """Runtime: one frame at a time, same numbers as assemble() over the whole sequence."""

    def __init__(self, base: np.ndarray, width: int, height: int):
        self.base, self.width, self.height = np.asarray(base, dtype=np.float64), width, height
        self._last_p: float | None = None
        self._last_t: float | None = None

    def step(self, blend: np.ndarray, eye_xy: np.ndarray, tf: np.ndarray, found: bool, t_ms: float) -> np.ndarray:
        x = np.zeros(NUM_FEATURES, dtype=np.float32)
        if not found:
            return x
        raw = raw_features(blend[None], eye_xy[None], tf[None], self.width, self.height)[0]
        rate = 0.0
        if self._last_t is not None and 0 < t_ms - self._last_t <= RATE_MAX_GAP_S * 1000:
            rate = (raw[10] - self._last_p) / ((t_ms - self._last_t) / 1000)
        self._last_p, self._last_t = float(raw[10]), float(t_ms)
        x[:NUM_RAW] = raw - self.base
        x[NUM_RAW] = rate
        x[NUM_RAW + 1] = 1.0
        return x



# --- Eyelid Angle (ELA), experimental input (arXiv 2511.19519): angle between planes fitted to the upper and the
# lower eyelid in 3D; reported to depend less on the camera angle than EAR. Seven MediaPipe points per lid,
# ordered inner corner -> outer corner. Corners are not part of the lids.
LID_LEFT_UPPER = [398, 384, 385, 386, 387, 388, 466]
LID_LEFT_LOWER = [382, 381, 380, 374, 373, 390, 249]
LID_RIGHT_UPPER = [173, 157, 158, 159, 160, 161, 246]
LID_RIGHT_LOWER = [155, 154, 153, 145, 144, 163, 7]
LID_POINTS = LID_LEFT_UPPER + LID_LEFT_LOWER + LID_RIGHT_UPPER + LID_RIGHT_LOWER  # 28


def _lid_normal(p: np.ndarray) -> np.ndarray:
    """p: (N, 7, 3) -> (N, 3) unit normal of the least-squares plane, sign fixed by the point order."""
    a = p - p.mean(axis=1, keepdims=True)
    n = np.linalg.svd(a)[2][:, -1, :]
    s = np.einsum("nk,nk->n", n, np.cross(a[:, 1:], a[:, :-1]).sum(axis=1))
    return n * np.where(s < 0, -1.0, 1.0)[:, None]


def ela_deg(lids: np.ndarray, width: int, height: int) -> np.ndarray:
    """lids: (N, 28, 3) normalized LID_POINTS (x/width, y/height, z in width units) -> (N, 2) ELA left, right.
    y is rescaled to width units so the three axes share one scale (the paper's extra z factor is not used)."""
    p = lids * np.array([1.0, height / width, 1.0])
    out = []
    for k in (0, 14):
        nu, nl = _lid_normal(p[:, k:k + 7]), _lid_normal(p[:, k + 7:k + 14])
        out.append(np.degrees(np.arccos(np.clip(np.einsum("nk,nk->n", nu, nl), -1, 1))))
    return np.stack(out, axis=1)
