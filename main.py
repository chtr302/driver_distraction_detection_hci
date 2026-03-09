import os
import argparse
import cv2
import threading
import time
import numpy as np
from flask import Flask, Response, jsonify, render_template
from pathlib import Path

# --- CORE LOGIC IMPORTS ---
from src.core.config import (
    DEFAULT_MODEL_PATH, LATEST_MODEL_INFO,
    CALIBRATION_SECONDS, MAX_SCORE,
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT
)
from src.core.models import load_onnx_model
from src.core.vision import build_face_detector, extract_landmarks, normalize_landmarks
from src.core.alert_logic import AlertManager
from src.muscle.head_pose import HeadPoseCalibrator, estimate_head_pose, head_pose_state
from src.muscle.feedback import FeedbackHAL

import mediapipe as mp
import logging

# --- SUPPRESS LOGS ---
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__, template_folder="templates")

output_frame = None
lock = threading.Lock()

system_state = {
    'muted': False,
    'calibration_requested': False,
    'score': 0.0,
    'level': 0,
    'status': 'SAFE',
    'head_pose': {'yaw': 0.0, 'pitch': 0.0, 'state': 'UNKNOWN'},
    'is_distracted': False,
    'calibrated': False,
    'face_detected': False,
    'uptime': 0,
    'alerts_count': {'warning': 0, 'critical': 0}
}
state_lock = threading.Lock()

def resolve_model_path(model_arg: Path | None) -> Path:
    if model_arg and model_arg.exists():
        return model_arg
    if LATEST_MODEL_INFO.exists():
        p = Path(LATEST_MODEL_INFO.read_text().strip())
        if p.exists():
            return p
    return DEFAULT_MODEL_PATH

def vision_thread_func(args):
    global output_frame, system_state

    model_path = resolve_model_path(args.model)
    classifier = load_onnx_model(model_path)
    detector = build_face_detector()
    alert_manager = AlertManager()
    feedback = FeedbackHAL()
    pose_calibrator = HeadPoseCalibrator(calibrate_seconds=CALIBRATION_SECONDS)

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    start_time = time.time()
    last_level = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            with state_lock:
                if system_state['calibration_requested']:
                    pose_calibrator = HeadPoseCalibrator(calibrate_seconds=CALIBRATION_SECONDS)
                    system_state['calibration_requested'] = False
                    system_state['calibrated'] = False

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            # --- RESET LOGIC STATES ---
            is_distracted = False
            face_detected = False
            head_pose_ok = True
            yaw, pitch, pose_state = 0.0, 0.0, "UNKNOWN"

            if result.face_landmarks:
                face_detected = True
                pts = extract_landmarks(result)
                if pts:
                    pred, _ = classifier.predict(normalize_landmarks(pts))
                    is_distracted = (pred == 1)

                    landmarks = result.face_landmarks[0]
                    yaw, pitch, ok = estimate_head_pose(landmarks, w, h)
                    if ok:
                        pose_calibrator.update(yaw, pitch)
                        if pose_calibrator.ready:
                            with state_lock:
                                system_state['calibrated'] = True
                            yn, pn = pose_calibrator.normalize(yaw, pitch)
                            pose_state = head_pose_state(yn, pn)
                            head_pose_ok = (pose_state == "NHIN DUONG")
                    
                    if args.show:
                        for p in pts:
                            cv2.circle(frame, (int(p[0]*w), int(p[1]*h)), 1, (0, 255, 255), -1)

            # --- ALWAYS UPDATE ALERT LOGIC (OUTSIDE FACE_DETECTION) ---
            # If no face, is_distracted=False and head_pose_ok=True -> Score will DECAY
            _, level = alert_manager.update(is_distracted, head_pose_ok)

            # Alert counters
            if level > last_level:
                with state_lock:
                    if level == 1: system_state['alerts_count']['warning'] += 1
                    elif level == 2: system_state['alerts_count']['critical'] += 1
            last_level = level

            # Feedback trigger
            with state_lock:
                muted = system_state['muted']
            
            if muted and level == 2:
                feedback.trigger(1)
            else:
                feedback.trigger(level)

            # --- UI DRAWING (ALWAYS) ---
            ui_color = (0, 255, 0) if level == 0 else ((0, 255, 255) if level == 1 else (0, 0, 255))
            status_text = "SAFE" if level == 0 else ("WARNING" if level == 1 else "CRITICAL!!")
            
            if not face_detected:
                cv2.putText(frame, "NO FACE DETECTED", (20, 50), 1, 1.5, (128, 128, 128), 2)
            else:
                cv2.putText(frame, f"STATUS: {status_text}", (20, 50), 1, 1.5, ui_color, 3)

            bar_w = int((alert_manager.score / MAX_SCORE) * 200)
            cv2.rectangle(frame, (20, 80), (220, 100), (200, 200, 200), 1)
            cv2.rectangle(frame, (20, 80), (20 + bar_w, 100), ui_color, -1)

            if face_detected and not pose_calibrator.ready:
                cv2.putText(frame, "CALIBRATING...", (20, h - 20), 1, 1.2, (255, 255, 0), 2)

            with state_lock:
                system_state.update({
                    'score': alert_manager.score,
                    'level': level,
                    'status': status_text,
                    'head_pose': {'yaw': float(yaw), 'pitch': float(pitch), 'state': pose_state},
                    'is_distracted': is_distracted,
                    'face_detected': face_detected,
                    'uptime': int(time.time() - start_time)
                })

            with lock:
                output_frame = frame.copy()
    finally:
        feedback.cleanup()
        cap.release()
        detector.close()

def gen_frames():
    global output_frame
    while True:
        with lock:
            if output_frame is None: continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed(): return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status')
def api_status():
    with state_lock: return jsonify(system_state)

@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    with state_lock: system_state['calibration_requested'] = True
    return jsonify({'success': True})

@app.route('/api/mute', methods=['POST'])
def api_mute():
    with state_lock:
        system_state['muted'] = not system_state['muted']
        return jsonify({'success': True, 'muted': system_state['muted']})

@app.route('/api/reset-stats', methods=['POST'])
def api_reset_stats():
    with state_lock:
        system_state['alerts_count'] = {'warning': 0, 'critical': 0}
        return jsonify({'success': True})

@app.route('/')
def index(): return render_template('index.html')

def main():
    parser = argparse.ArgumentParser(description="WakeGuard Integrated System")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--cam", type=int, default=CAMERA_INDEX)
    parser.add_argument("--port", type=int, default=5051)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    threading.Thread(target=vision_thread_func, args=(args,), daemon=True).start()
    print(f"Web server running and using camera {args.cam}")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True, use_reloader=False)

if __name__ == "__main__":
    main()
