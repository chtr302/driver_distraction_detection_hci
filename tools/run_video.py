"""Run the GRU v3 pipeline on video files, frame by frame, and write one CSV per video (model comparison).

    python -m tools.run_video VIDEO [VIDEO ...] --out runs/bench_gru_v3
    python -m tools.run_video data/clips/*.mp4 --color rgb

Same path as the runtime (numpy + onnxruntime only, no torch):
  frame -> resample to 12 fps by timestamp -> gray3 -> FaceOnnx -> 13 raw features
  -> first CALIB_S seconds: per-driver baseline + GRU offset (driver must be awake there)
  -> 15 inputs -> GruAlarm (3 ONNX models, offset, EMA, delta) -> drowsy per frame.
Every video is treated as one driver and one trip, calibrated on its own first CALIB_S seconds.
Videos shorter than CALIB_S + a few seconds give no decisions.

CSV columns (one row per processed frame after calibration):
  t_s         video time in seconds
  face        1 if a face was found
  z           ensemble logit after offset and threshold (before smoothing); empty without a face
  ema         smoothed score; drowsy = ema > delta (rule.json, 2.0)
  drowsy      0/1, the model's per-frame decision (no water tank, no alert rule)
  ms_face     time of FaceOnnx on this frame
  ms_model    time of features + 3 GRU models on this frame
Rows with an empty ema: the GRU is still calibrating (fewer than 360 face frames so far), drowsy is 0 there;
leave them out of any metric. Videos with no face in the first CALIB_S seconds are skipped (note in summary.json).
A summary line per video and summary.json are printed/written: frames, face ratio, drowsy ratio, median and p95 ms.
To compare another model, produce the same per-frame columns (t_s, drowsy, optionally a score) on the same videos.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.core.capture import apply_color_mode
from src.core.eye_features import BLEND_FEATURES, EAR_POINTS, EyeFeatureStream, baseline, raw_features
from src.core.face_onnx import FaceOnnx
from src.core.face_types import BLENDSHAPE_NAMES
from src.core.gru_alarm import GruAlarm

BLEND_IDX = [BLENDSHAPE_NAMES.index(n) for n in BLEND_FEATURES]


def face_inputs(res):
    """FaceResult -> (blend (8,), eye_xy (12, 2), tf (16,)) or None."""
    if not res.found:
        return None
    return (res.blendshapes[BLEND_IDX].astype(np.float64), res.landmarks[EAR_POINTS, :2].astype(np.float64),
            res.matrix.reshape(16).astype(np.float64))


def frames_at(path: Path, fps: float):
    """Yield (t_ms, rgb) resampled to `fps` by video timestamp (drops extra frames, never duplicates)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    i, next_ms = 0, 0.0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                return
            t_ms = i * 1000.0 / src_fps
            i += 1
            if t_ms + 1e-6 < next_ms:
                continue
            next_ms += 1000.0 / fps
            yield t_ms, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def run(path: Path, model_dir: str, color: str, out_dir: Path) -> dict:
    alarm = GruAlarm(model_dir)
    fps, calib_s = float(alarm.rule["fps"]), float(alarm.rule["calib_s"])
    face = FaceOnnx()
    calib, stream, rows = [], None, []
    for t_ms, rgb in frames_at(path, fps):
        img = apply_color_mode(rgb, color)
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        inp = face_inputs(face.detect(img))
        t1 = time.perf_counter()
        if stream is None:
            calib.append((inp, t_ms))
            if t_ms / 1000 < calib_s:
                continue
            found = np.array([c[0] is not None for c in calib])
            if not found.any():  # no baseline possible: skip this video, keep the batch running
                face.close()
                return {"video": path.name, "frames": 0, "note": f"no face in the first {calib_s:.0f} s"}
            if found.mean() < 0.5:
                print(f"{path.name}: face in only {found.mean():.0%} of calibration frames, results unreliable")
            raw = np.stack([raw_features(c[0][0][None], c[0][1][None], c[0][2][None], w, h)[0]
                            if c[0] is not None else np.zeros(13) for c in calib])
            stream = EyeFeatureStream(baseline(raw, found), w, h)
            alarm.reset()
            for c, tm in calib:  # replay: the GRU takes its offsets from the same awake stretch
                x = stream.step(*c, found=True, t_ms=tm) if c is not None else np.zeros(15, np.float32)
                alarm.step(x, c is not None)
            continue
        x = stream.step(*inp, found=True, t_ms=t_ms) if inp is not None else np.zeros(15, np.float32)
        drowsy = alarm.step(x, inp is not None)
        t2 = time.perf_counter()
        rows.append({"t_s": round(t_ms / 1000, 3), "face": int(inp is not None),
                     "z": "" if inp is None or alarm.last_z is None else round(alarm.last_z, 4),
                     "ema": "" if alarm.s is None else round(alarm.s, 4), "drowsy": int(drowsy),
                     "ms_face": round((t1 - t0) * 1000, 2), "ms_model": round((t2 - t1) * 1000, 2)})
    face.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{path.stem}.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=["t_s", "face", "z", "ema", "drowsy", "ms_face", "ms_model"])
        wr.writeheader()
        wr.writerows(rows)
    if not rows:
        return {"video": path.name, "frames": 0, "note": f"shorter than calibration ({calib_s:.0f} s)"}
    mf = np.array([r["ms_face"] for r in rows])
    mm = np.array([r["ms_model"] for r in rows])
    return {"video": path.name, "frames": len(rows), "seconds": rows[-1]["t_s"] - rows[0]["t_s"],
            "face_ratio": float(np.mean([r["face"] for r in rows])),
            "drowsy_ratio": float(np.mean([r["drowsy"] for r in rows])),
            "ms_face_median": float(np.median(mf)), "ms_face_p95": float(np.percentile(mf, 95)),
            "ms_model_median": float(np.median(mm)), "ms_model_p95": float(np.percentile(mm, 95))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", type=Path)
    ap.add_argument("--model", default="src/models/eye_gru_v3")
    ap.add_argument("--color", default="gray3", help="gray3 (as the training data) or rgb")
    ap.add_argument("--out", type=Path, default=Path("runs/bench_gru_v3"))
    a = ap.parse_args()
    summary = []
    for v in a.videos:
        s = run(v, a.model, a.color, a.out)
        summary.append(s)
        print(json.dumps(s), flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{len(summary)} videos, {sum(1 for s in summary if s['frames'])} with results; summary.json in {a.out}")


if __name__ == "__main__":
    main()
