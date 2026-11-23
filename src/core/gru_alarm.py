"""Per-frame drowsiness decision from the GRU v3 ensemble (runtime, Pi path: numpy + onnxruntime only).

    alarm = GruAlarm("src/models/eye_gru_v3")
    drowsy = alarm.step(x, face_found)            # x: 15 features (src/core/eye_features.FEATURE_NAMES)
    score, level = alert_manager.update(is_distracted=drowsy, head_pose_ok=True)

Rule (chosen by tools/tune_operating_point.py, docs/design/pipeline-gru.md "Final model"):
  per model m: z_m = episode logit - offset_m - threshold_m
    offset_m = median episode logit over the driver's first CALIB_S seconds with a face (trip-start calibration)
  z = mean of z_m over the models; s = EMA of z over face frames (alpha); drowsy = face and s > delta.
The water tank (+5 / -2, warning 50, critical 100) stays in src/core/alert_logic.AlertManager, unchanged.
During calibration the decision is always "not drowsy" (the offset is not known yet).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


class GruAlarm:
    def __init__(self, model_dir: str | Path, threads: int = 1):
        d = Path(model_dir)
        self.rule = json.loads((d / "rule.json").read_text(encoding="utf-8"))
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = [ort.InferenceSession(str(d / m["file"]), opts, providers=["CPUExecutionProvider"])
                     for m in self.rule["models"]]
        self.thr = np.array([m["threshold_offset_logit"] for m in self.rule["models"]], np.float32)
        self.hid = int(self.rule["hidden"])
        self.calib_frames = int(self.rule["calib_s"] * self.rule["fps"])
        self.reset()

    def reset(self, keep_calibration: bool = False) -> None:
        """New trip: zero GRU state and smoothing. New driver (default): also recalibrate.
        The offset must come from awake driving: a calibration taken while drowsy hides the drowsiness."""
        self.h = [np.zeros((1, self.hid), np.float32) for _ in self.sess]
        self.s: float | None = None
        self.last_z: float | None = None
        if not keep_calibration:
            self.calib: list[np.ndarray] = []
            self.offset: np.ndarray | None = None

    @property
    def calibrating(self) -> bool:
        return self.offset is None

    def step(self, x: np.ndarray, face_found: bool) -> bool:
        x = np.asarray(x, np.float32).reshape(1, -1)
        z = np.empty(len(self.sess), np.float32)
        for i, sess in enumerate(self.sess):  # the GRU state runs on every frame, face or not (as in training)
            ze, _, self.h[i] = sess.run(None, {"x": x, "h": self.h[i]})
            z[i] = ze[0]
        if not face_found:
            return False
        if self.offset is None:
            self.calib.append(z)
            if len(self.calib) >= self.calib_frames:
                self.offset = np.median(np.stack(self.calib), axis=0)
                self.calib = []
            return False
        zz = float(np.mean(z - self.offset - self.thr))
        self.last_z = zz  # before smoothing; for display only
        a = float(self.rule["ema_alpha"])
        self.s = zz if self.s is None else self.s + a * (zz - self.s)
        return self.s > float(self.rule["delta"])
