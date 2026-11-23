"""Face pipeline on ONNX Runtime: MediaPipe's own weights, MediaPipe's tracking recipe, no mediapipe import.

Same contract as src/core/landmarker.py (FaceResult: 478 landmarks normalized like MediaPipe, 52 blendshapes,
4x4 head pose), so the collector and the runtime share one input definition.

Per frame (one face):
  1. ROI: from the previous frame's landmarks (tracking) or, when there is none, from BlazeFace.
     Square on the long side, x1.5, rotated so the eye line is level (MediaPipe face_landmarks_detector_graph).
  2. 478-point mesh on a 256x256 crop; presence = sigmoid(flag) >= presence_threshold, else re-detect once.
  3. Blendshapes from the 146-point subset in image pixels (MediaPipe face_blendshapes_graph).
  4. Head pose: MediaPipe's geometry pipeline (geometry_pipeline.cc): landmarks -> metric space under a
     63 deg vertical-FOV camera, weighted Procrustes against the canonical face (cm); stateless per frame.
No landmark smoothing: raw per-frame values, temporal modelling belongs to the model downstream.

Detector decoding (anchors, weighted NMS) follows yakhyo/mediapipe-face-mesh-onnx (Apache-2.0), which
re-implements MediaPipe's SSD post-processing. Models: tools/convert_face_models.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.core.config import MODELS_DIR
from src.core.face_types import BLENDSHAPE_NAMES, NUM_LANDMARKS, FaceResult

FACE_ONNX_DIR = MODELS_DIR / "face_onnx"
DET_SIZE, MESH_SIZE = 128, 256
ROI_SCALE = 1.5
EYE_START, EYE_END = 33, 263  # rotation vector for tracking ROIs (outer eye corners)
# 146 of the 478 landmarks fed to the blendshape model (face_blendshapes_graph.cc, kLandmarksSubsetIdxs)
BLENDSHAPE_SUBSET = np.array([
    0, 1, 4, 5, 6, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67, 70, 78, 80,
    81, 82, 84, 87, 88, 91, 93, 95, 103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153,
    154, 155, 157, 158, 159, 160, 161, 162, 163, 168, 172, 173, 176, 178, 181, 185, 191, 195, 197, 234, 246, 249,
    251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311, 312, 314,
    317, 318, 321, 323, 324, 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380, 381, 382,
    384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409, 415, 454, 466, 468, 469, 470, 471, 472, 473,
    474, 475, 476, 477])
# MediaPipe's geometry pipeline assumes this perspective camera (face_geometry environment)
MP_VERTICAL_FOV_DEG = 63.0


def _anchors() -> np.ndarray:
    """SSD anchor centers of face_detection_short_range at 128x128: strides [8, 16, 16, 16] -> 896 anchors."""
    out = []
    for stride, per_cell in ((8, 2), (16, 6)):
        cells = DET_SIZE // stride
        for y in range(cells):
            for x in range(cells):
                out.extend([((x + 0.5) / cells, (y + 0.5) / cells)] * per_cell)
    return np.array(out)


def _weighted_nms(rows: np.ndarray, iou_threshold: float = 0.3) -> np.ndarray:
    """MediaPipe WEIGHTED NMS on rows of (score, cx, cy, w, h, kp...): overlapping candidates are score-averaged."""
    rows = rows[np.argsort(-rows[:, 0])]
    out = []
    while len(rows):
        top = rows[0]
        x1, y1 = rows[:, 1] - rows[:, 3] / 2, rows[:, 2] - rows[:, 4] / 2
        x2, y2 = rows[:, 1] + rows[:, 3] / 2, rows[:, 2] + rows[:, 4] / 2
        tx1, ty1, tx2, ty2 = top[1] - top[3] / 2, top[2] - top[4] / 2, top[1] + top[3] / 2, top[2] + top[4] / 2
        inter = np.clip(np.minimum(x2, tx2) - np.maximum(x1, tx1), 0, None) * \
            np.clip(np.minimum(y2, ty2) - np.maximum(y1, ty1), 0, None)
        iou = inter / np.maximum((x2 - x1) * (y2 - y1) + (tx2 - tx1) * (ty2 - ty1) - inter, 1e-9)
        group = rows[iou > iou_threshold]
        blended = top.copy()
        blended[1:] = (group[:, 1:] * group[:, :1]).sum(0) / group[:, 0].sum()
        out.append(blended)
        rows = rows[iou <= iou_threshold]
    return np.array(out)


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _roi(center: np.ndarray, size: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> tuple[float, float, float, float]:
    """(cx, cy, side, angle_deg): square on the long side x1.5, rotated so p0 -> p1 is horizontal."""
    dx, dy = p1 - p0
    return float(center[0]), float(center[1]), ROI_SCALE * float(size.max()), float(np.degrees(np.arctan2(dy, dx)))


class FaceOnnx:
    def __init__(self, model_dir: Path = FACE_ONNX_DIR, threads: int = 0, spinning: bool = False,
                 detection_threshold: float = 0.5, presence_threshold: float = 0.5):
        self.model_dir = Path(model_dir)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads  # 0 = onnxruntime default (one per core)
        # three small sessions run back to back; spinning worker threads of one starve the next
        opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if spinning else "0")
        self._threads, self._spinning = threads, spinning
        make = lambda name: ort.InferenceSession(str(self.model_dir / f"{name}.onnx"), sess_options=opts,
                                                 providers=["CPUExecutionProvider"])
        self._det, self._mesh, self._bs = make("face_detector"), make("face_landmarks_detector"), make("face_blendshapes")
        self._det_in, self._mesh_in, self._bs_in = (s.get_inputs()[0].name for s in (self._det, self._mesh, self._bs))
        geo = np.load(self.model_dir / "canonical_face.npz")
        self._canonical = geo["vertices"].astype(np.float64)
        self._canonical_t = self._canonical.T
        weights = np.zeros(len(self._canonical))
        weights[geo["basis_ids"]] = geo["basis_weights"]
        self._sqrt_w, self._total_w = np.sqrt(weights), float(weights.sum())
        self._anchors = _anchors()
        self.detection_threshold, self.presence_threshold = detection_threshold, presence_threshold
        self._roi: tuple[float, float, float, float] | None = None
        self.stats = {"frames": 0, "detector_runs": 0, "lost": 0}

    # --- stages -------------------------------------------------------------------------------------------------
    def _detect(self, rgb: np.ndarray):
        """BlazeFace on a letterboxed 128x128 copy; returns the ROI of the best face or None."""
        h, w = rgb.shape[:2]
        scale = DET_SIZE / max(h, w)
        pad = np.array([(DET_SIZE - w * scale) / 2, (DET_SIZE - h * scale) / 2])
        canvas = cv2.warpAffine(rgb, np.array([[scale, 0, pad[0]], [0, scale, pad[1]]]), (DET_SIZE, DET_SIZE))
        x = (canvas.astype(np.float32) - 127.5) / 127.5
        reg, logits = self._det.run(None, {self._det_in: x[None]})
        self.stats["detector_runs"] += 1
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits[0, :, 0].astype(np.float64), -100, 100)))
        keep = scores >= self.detection_threshold
        if not keep.any():
            return None
        r, a = reg[0][keep].astype(np.float64) / DET_SIZE, self._anchors[keep]
        rows = np.empty((int(keep.sum()), 17))
        rows[:, 0] = scores[keep]
        rows[:, 1:3] = r[:, 0:2] + a
        rows[:, 3:5] = r[:, 2:4]
        rows[:, 5:] = (r[:, 4:16].reshape(-1, 6, 2) + a[:, None]).reshape(-1, 12)
        best = _weighted_nms(rows)[0]
        to_img = lambda xy: (xy * DET_SIZE - pad) / scale
        size = best[3:5] * DET_SIZE / scale
        kp = to_img(best[5:].reshape(6, 2))  # 0 right eye, 1 left eye (subject), i.e. viewer left -> right
        return _roi(to_img(best[1:3]), size, kp[0], kp[1])

    def _landmarks(self, rgb: np.ndarray, roi) -> tuple[np.ndarray, float]:
        """478 x (x, y, z) in image pixels (z on the x scale) and the presence probability."""
        cx, cy, side, angle = roi
        m = cv2.getRotationMatrix2D((cx, cy), angle, MESH_SIZE / side)
        m[:, 2] += (MESH_SIZE / 2 - cx, MESH_SIZE / 2 - cy)
        crop = cv2.warpAffine(rgb, m, (MESH_SIZE, MESH_SIZE)).astype(np.float32) / 255.0
        lm_raw, flag = self._mesh.run(None, {self._mesh_in: crop[None]})[:2]
        lm = lm_raw.reshape(NUM_LANDMARKS, 3).astype(np.float64)
        inv = cv2.invertAffineTransform(m)
        lm[:, :2] = lm[:, :2] @ inv[:, :2].T + inv[:, 2]
        lm[:, 2] *= side / MESH_SIZE
        return lm, _sigmoid(float(np.ravel(flag)[0]))

    def _next_roi(self, lm: np.ndarray):
        lo, hi = lm[:, :2].min(0), lm[:, :2].max(0)
        return _roi((lo + hi) / 2, hi - lo, lm[EYE_START, :2], lm[EYE_END, :2])

    def _blendshapes(self, lm: np.ndarray) -> np.ndarray:
        pts = lm[BLENDSHAPE_SUBSET, :2].astype(np.float32)[None]
        return np.asarray(self._bs.run(None, {self._bs_in: pts})[0], dtype=np.float32).reshape(-1)

    def _procrustes(self, target: np.ndarray) -> np.ndarray:
        """Weighted orthogonal Procrustes canonical -> target (3xN), 4x4 with uniform scale (MediaPipe solver)."""
        src, sw = self._canonical_t, self._sqrt_w
        ws, wt = src * sw, target * sw
        center = (ws * sw).sum(1, keepdims=True) / self._total_w
        cws = ws - center * sw
        u, _, vt = np.linalg.svd(wt @ cws.T)
        if np.linalg.det(u) * np.linalg.det(vt) < 0:
            u[:, 2] *= -1
        rot = u @ vt
        scale = ((rot @ cws) * wt).sum() / (cws * ws).sum()
        trans = (((wt - scale * rot @ ws) * sw).sum(1)) / self._total_w
        out = np.eye(4)
        out[:3, :3], out[:3, 3] = scale * rot, trans
        return out

    def _pose(self, norm: np.ndarray, w: int, h: int) -> np.ndarray:
        """MediaPipe geometry pipeline (screen -> metric space, two scale passes, Procrustes); stateless."""
        near = 1.0
        height = 2 * near * np.tan(np.radians(MP_VERTICAL_FOV_DEG) / 2)
        width = w * height / h
        left, bottom = -width / 2, -height / 2
        scr = norm[:len(self._canonical)].T.astype(np.float64).copy()  # 3xN: x, y, z normalized like MediaPipe
        scr[1] = 1.0 - scr[1]                                          # origin top-left -> bottom-left
        scr *= np.array([[width], [height], [width]])
        scr[0] += left
        scr[1] += bottom
        depth_offset = scr[2].mean()

        def metric(scale: float) -> np.ndarray:
            m = scr.copy()
            m[2] = (m[2] - depth_offset + near) / scale
            m[0] *= m[2] / near
            m[1] *= m[2] / near
            m[2] *= -1
            return m

        first = scr.copy()
        first[2] *= -1
        s1 = np.linalg.norm(self._procrustes(first)[:3, 0])
        s2 = np.linalg.norm(self._procrustes(metric(s1))[:3, 0])
        return self._procrustes(metric(s1 * s2)).astype(np.float32)

    # --- public -------------------------------------------------------------------------------------------------
    def detect(self, rgb: np.ndarray, t_ms: int | None = None) -> FaceResult:
        """One RGB uint8 frame -> FaceResult. t_ms is accepted for interface parity and unused."""
        h, w = rgb.shape[:2]
        self.stats["frames"] += 1
        tracked = self._roi is not None
        for attempt in range(2):
            roi = self._roi if (tracked and attempt == 0) else self._detect(rgb)
            if roi is None:
                break
            lm, presence = self._landmarks(rgb, roi)
            if presence >= self.presence_threshold:
                self._roi = self._next_roi(lm)
                norm = lm.copy()
                norm[:, 0] /= w
                norm[:, 1] /= h
                norm[:, 2] /= w
                return FaceResult(norm.astype(np.float32), self._blendshapes(lm), self._pose(norm, w, h))
            if not tracked or attempt == 1:
                break
        if tracked:
            self.stats["lost"] += 1
        self._roi = None
        return FaceResult(None, None, None)

    def reset(self) -> None:
        self._roi = None

    def info(self) -> dict:
        conv = json.loads((self.model_dir / "conversion.json").read_text())
        return {
            "backend": "onnxruntime", "version": ort.__version__, "running_mode": "tracking",
            "source_task_sha256": conv["source_sha256"],
            "models": {n: hashlib.sha256((self.model_dir / f"{n}.onnx").read_bytes()).hexdigest()
                       for n in ("face_detector", "face_landmarks_detector", "face_blendshapes")},
            "presence_threshold": self.presence_threshold, "detection_threshold": self.detection_threshold,
            "smoothing": "none", "pose": f"mediapipe geometry pipeline, vfov {MP_VERTICAL_FOV_DEG} deg",
            "threads": self._threads, "spinning": self._spinning,
        }

    def close(self) -> None:
        pass


assert len(BLENDSHAPE_SUBSET) == 146 and len(BLENDSHAPE_NAMES) == 52
