import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Sequence, List
from src.core.config import FACE_LANDMARKER_PATH, ALL_POINTS

def build_face_detector() -> vision.FaceLandmarker:
    if not FACE_LANDMARKER_PATH.exists():
        raise FileNotFoundError(f"Model not found at {FACE_LANDMARKER_PATH}")
    
    base_options = python.BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)

def extract_landmarks(detection_result) -> List[List[float]] | None:
    if not detection_result.face_landmarks:
        return None
    landmarks = detection_result.face_landmarks[0]
    return [[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in ALL_POINTS]

def normalize_landmarks(points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    center = np.mean(pts, axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    scale = float(np.max(dist))
    if scale < 1e-6: scale = 1.0
    normalized = (pts - center) / scale
    return normalized.flatten().astype(np.float32)
