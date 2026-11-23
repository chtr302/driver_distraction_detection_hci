"""Face pipeline output shared by every backend (no mediapipe / onnxruntime import here)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NUM_LANDMARKS = 478
# Output order of face_landmarker.task (verified against the model, 2026-10-18)
BLENDSHAPE_NAMES = (
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthFunnel",
    "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower", "mouthRollUpper", "mouthShrugLower",
    "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
)
FACE_BACKENDS = ("onnx", "mediapipe")


@dataclass
class FaceResult:
    landmarks: np.ndarray | None    # (478, 3) float32, x/y normalized to image size, z on the x scale
    blendshapes: np.ndarray | None  # (52,) float32, BLENDSHAPE_NAMES order
    matrix: np.ndarray | None       # (4, 4) float32, canonical face -> camera (cm), MediaPipe axes

    @property
    def found(self) -> bool:
        return self.landmarks is not None


def open_face_pipeline(backend: str = "onnx"):
    """Imported lazily so the ONNX path never loads mediapipe."""
    if backend == "onnx":
        from src.core.face_onnx import FaceOnnx
        return FaceOnnx()
    if backend == "mediapipe":
        from src.core.landmarker import Landmarker
        return Landmarker(running_mode="video")
    raise ValueError(f"unknown face backend: {backend}")
