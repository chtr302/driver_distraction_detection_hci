from __future__ import annotations

# --- FIX PROTOBUF ERROR ON PYTHON 3.11+ ---
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# ------------------------------------------

import argparse
import sys
from pathlib import Path
from typing import Sequence
import time

import cv2
import numpy as np
import torch

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
MODEL_PATH = SRC_DIR / "models" / "face_landmarker.task"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from muscle.head_pose import HeadPoseCalibrator, estimate_head_pose, head_pose_state


# Landmarks used in data collection (must match training CSV)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311]
ALL_POINTS = LEFT_EYE + RIGHT_EYE + MOUTH_INNER


class LandmarkClassifier(torch.nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: list[int],
        out_dim: int,
        dropout: float = 0.2,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(torch.nn.Linear(prev, h))
            if batch_norm:
                layers.append(torch.nn.BatchNorm1d(h))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            prev = h
        layers.append(torch.nn.Linear(prev, out_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InputNormalizer(torch.nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class NormalizedModel(torch.nn.Module):
    def __init__(self, base: torch.nn.Module, mean: np.ndarray, std: np.ndarray) -> None:
        super().__init__()
        self.norm = InputNormalizer(mean, std)
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(self.norm(x))


def _infer_mlp_from_state_dict(state_dict: dict) -> tuple[int, int, list[int], bool, bool] | None:
    linear_layers: list[tuple[int, torch.Tensor]] = []
    has_bn = False
    for key, value in state_dict.items():
        if key.startswith("base.net.") and key.endswith("running_mean"):
            has_bn = True
        if (
            key.startswith("base.net.")
            and key.endswith("weight")
            and isinstance(value, torch.Tensor)
            and value.dim() == 2
        ):
            try:
                idx = int(key.split(".")[2])
            except (IndexError, ValueError):
                continue
            linear_layers.append((idx, value))

    if not linear_layers:
        return None

    linear_layers.sort(key=lambda item: item[0])
    input_dim = int(linear_layers[0][1].shape[1])
    out_dim = int(linear_layers[-1][1].shape[0])
    hidden = [int(weight.shape[0]) for _, weight in linear_layers[:-1]]

    dropout_present = False
    if len(linear_layers) >= 2:
        diffs = [linear_layers[i + 1][0] - linear_layers[i][0] for i in range(len(linear_layers) - 1)]
        step = min(diffs)
        if has_bn:
            dropout_present = step >= 4
        else:
            dropout_present = step >= 3

    return input_dim, out_dim, hidden, has_bn, dropout_present


def normalize_landmarks(points: Sequence[Sequence[float]]) -> np.ndarray:
    pts = np.array(points, dtype=np.float32)
    center = np.mean(pts, axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    scale = float(np.max(dist))
    if scale < 1e-6:
        scale = 1.0
    normalized = (pts - center) / scale
    return normalized.flatten().astype(np.float32)


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        inferred = _infer_mlp_from_state_dict(state_dict)
        if inferred is None:
            raise ValueError("Could not infer MLP architecture from checkpoint.")

        inf_input_dim, inf_out_dim, inf_hidden, has_bn, has_dropout = inferred
        input_dim = int(checkpoint.get("input_dim", inf_input_dim))
        if input_dim != inf_input_dim:
            input_dim = inf_input_dim

        hidden = list(checkpoint.get("hidden", []))
        if hidden != inf_hidden:
            hidden = inf_hidden

        out_dim = inf_out_dim
        binary_output = out_dim == 1

        mean = np.asarray(checkpoint.get("mean", np.zeros(input_dim, dtype=np.float32)))
        std = np.asarray(checkpoint.get("std", np.ones(input_dim, dtype=np.float32)))

        dropout = float(checkpoint.get("dropout", 0.2)) if has_dropout else 0.0
        base = MLPClassifier(
            in_dim=input_dim,
            hidden=hidden,
            out_dim=out_dim,
            dropout=dropout,
            batch_norm=has_bn,
        )
        model = NormalizedModel(base, mean, std)
        model.load_state_dict(state_dict)
        return model, input_dim, out_dim, binary_output

    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported checkpoint format.")

    state_dict = checkpoint
    weight_tensors = [v for v in state_dict.values() if isinstance(v, torch.Tensor) and v.dim() == 2]
    if not weight_tensors:
        raise ValueError("Could not infer model dimensions from checkpoint.")

    input_dim = int(weight_tensors[0].shape[1])
    out_dim = int(weight_tensors[-1].shape[0])
    model = LandmarkClassifier(input_dim, out_dim)
    model.load_state_dict(state_dict)
    binary_output = out_dim == 1
    return model, input_dim, out_dim, binary_output


def build_detector() -> vision.FaceLandmarker:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing face landmarker model at: {MODEL_PATH}. "
            "Download: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


def build_label_names(num_classes: int, names: str | None) -> list[str]:
    if names:
        items = [n.strip() for n in names.split(",") if n.strip()]
        if len(items) == num_classes:
            return items
    if num_classes == 2:
        return ["NORMAL", "ALERT"]
    return [f"CLASS_{i}" for i in range(num_classes)]

def resolve_model_path(model_arg: Path | None) -> Path:
    if model_arg is not None:
        return model_arg
    latest = SRC_DIR / "models" / "latest_model.txt"
    if latest.exists():
        text = latest.read_text(encoding="utf-8").strip()
        if text:
            return Path(text)
    return SRC_DIR / "models" / "wakeguard_mlp.pt"



def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime head pose + classifier demo.")
    parser.add_argument("--model", type=Path, default=None, help="Path to .pt model. If omitted, uses latest_model.txt.")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--class-names", type=str, default=None)
    parser.add_argument("--show-landmarks", action="store_true")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    device = torch.device("cpu")
    model, input_dim, out_dim, binary_output = load_model(model_path, device)
    model.eval()

    label_names = build_label_names(out_dim if out_dim > 1 else 2, args.class_names)

    detector = build_detector()

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened() and args.cam == 0:
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    pose_calibrator = HeadPoseCalibrator(calibrate_seconds=3.0)

    print("Realtime demo running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

    
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        h, w = frame.shape[:2]
        label_text = "NO FACE"
        pose_text = ""

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            selected = [[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in ALL_POINTS]
            features = normalize_landmarks(selected)
            if features.shape[0] != input_dim:
                raise ValueError(f"Feature dim mismatch: {features.shape[0]} vs model {input_dim}")

            with torch.no_grad():
                x = torch.from_numpy(features).unsqueeze(0)
                logits = model(x)
                if binary_output:
                    prob = float(torch.sigmoid(logits).item())
                    pred = 1 if prob >= 0.5 else 0
                    label_text = f"{label_names[pred]} ({prob:.2f})"
                else:
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    pred = int(np.argmax(probs))
                    label_text = f"{label_names[pred]} ({probs[pred]:.2f})"

            try:
                yaw, pitch, ok = estimate_head_pose(landmarks, w, h)
                if not ok:
                    pose_text = "Pose invalid"
                else:
                    pose_calibrator.update(yaw, pitch)
                    if pose_calibrator.ready:
                        yaw_norm, pitch_norm = pose_calibrator.normalize(yaw, pitch)
                        direction = head_pose_state(yaw_norm, pitch_norm)
                        pose_text = f"Yaw:{yaw_norm:.1f} Pitch:{pitch_norm:.1f} | {direction}"
                    else:
                        elapsed = 0.0 if pose_calibrator.start_time is None else (time.time() - pose_calibrator.start_time)
                        remain = max(0.0, pose_calibrator.calibrate_seconds - elapsed)
                        pose_text = f"Calibrating head pose... {remain:.1f}s"
            except (ValueError, IndexError):
                pose_text = "Pose invalid"

            if args.show_landmarks:
                for p in selected:
                    cv2.circle(frame, (int(p[0] * w), int(p[1] * h)), 2, (0, 255, 255), -1)

        cv2.putText(frame, label_text, (20, 40), 1, 0.9, (255, 215, 0), 2)
        if pose_text:
            cv2.putText(frame, pose_text, (20, 80), 1, 0.8, (0, 255, 0), 2)

        cv2.imshow("Realtime Demo", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
