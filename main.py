import os
import argparse
import cv2
import threading
import time
import numpy as np
from flask import Flask, Response
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

# Fix Protobuf issue on some environments
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

app = Flask(__name__)

# Global variables for thread communication
output_frame = None
lock = threading.Lock()

def resolve_model_path(model_arg: Path | None) -> Path:
    if model_arg and model_arg.exists():
        return model_arg
    if LATEST_MODEL_INFO.exists():
        p = Path(LATEST_MODEL_INFO.read_text().strip())
        if p.exists(): return p
    return DEFAULT_MODEL_PATH

def vision_thread_func(args):
    """
    Main Vision Thread: Constant AI processing and hardware control.
    """
    global output_frame
    
    # Init Engines
    model_path = resolve_model_path(args.model)
    classifier = load_onnx_model(model_path)
    detector = build_face_detector()
    alert_manager = AlertManager()
    feedback = FeedbackHAL()
    pose_calibrator = HeadPoseCalibrator(calibrate_seconds=CALIBRATION_SECONDS)
    
    # Camera Init
    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    print(f"🚀 [Vision] Monitoring started. Using camera: {args.cam}")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            import mediapipe as mp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            level = 0
            ui_color = (0, 255, 0) # Safe Green

            if result.face_landmarks:
                pts = extract_landmarks(result)
                if pts:
                    # 1. AI Distraction Inference
                    pred, _ = classifier.predict(normalize_landmarks(pts))
                    is_distracted = (pred == 1)

                    # 2. Head Pose Logic
                    landmarks = result.face_landmarks[0]
                    yaw, pitch, ok = estimate_head_pose(landmarks, w, h)
                    head_pose_ok = True
                    if ok:
                        pose_calibrator.update(yaw, pitch)
                        if pose_calibrator.ready:
                            yn, pn = pose_calibrator.normalize(yaw, pitch)
                            head_pose_ok = (head_pose_state(yn, pn) == "NHIN DUONG")

                    # 3. Decision & Alert (GPIO/Feedback)
                    _, level = alert_manager.update(is_distracted, head_pose_ok)
                    feedback.trigger(level)

                    # 4. Draw UI for Web Feed
                    ui_color = (0, 255, 0) if level == 0 else ((0, 165, 255) if level == 1 else (0, 0, 255))
                    status_text = "SAFE" if level == 0 else ("WARNING" if level == 1 else "CRITICAL!!")
                    cv2.putText(frame, f"STATUS: {status_text}", (20, 50), 1, 1.5, ui_color, 3)
                    
                    # Risk Progress Bar
                    bar_w = int((alert_manager.score / MAX_SCORE) * 200)
                    cv2.rectangle(frame, (20, 80), (220, 100), (200, 200, 200), 1)
                    cv2.rectangle(frame, (20, 80), (20 + bar_w, 100), ui_color, -1)
                    
                    if args.show:
                        for p in pts: cv2.circle(frame, (int(p[0]*w), int(p[1]*h)), 1, (0, 255, 255), -1)

            # Update shared frame safely
            with lock:
                output_frame = frame.copy()
    finally:
        feedback.cleanup()
        cap.release()
        detector.close()

def gen_frames():
    """
    Web Stream Generator: Pulls latest frame from Vision thread.
    """
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', output_frame)
            frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04) # Limit to ~25 FPS to save CPU

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
      <head>
        <title>WakeGuard Real-time Monitoring</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
      </head>
      <body style="background: #111; color: white; text-align: center; font-family: sans-serif; margin: 0; padding: 20px;">
        <h1 style="color: #ff4444;">🚨 WakeGuard Monitoring 🚨</h1>
        <div style="background: #222; padding: 10px; border-radius: 15px; display: inline-block;">
            <img src="/video_feed" style="width: 100%; max-width: 800px; border-radius: 10px;">
        </div>
        <p style="margin-top: 20px; font-size: 1.1em; color: #aaa;">Status: <span style="color: #00ff00;">ONLINE</span> | Protocol: MJPEG Stream</p>
      </body>
    </html>
    """

def main():
    parser = argparse.ArgumentParser(description="WakeGuard Integrated System")
    parser.add_argument("--model", type=Path, default=None, help="Path to ONNX model")
    parser.add_argument("--cam", type=int, default=CAMERA_INDEX, help="Camera index")
    parser.add_argument("--port", type=int, default=5050, help="Web port")
    parser.add_argument("--show", action="store_true", help="Show landmarks in stream")
    args = parser.parse_args()

    # 1. Start Vision Thread (Background)
    v_thread = threading.Thread(target=vision_thread_func, args=(args,), daemon=True)
    v_thread.start()
    
    # 2. Start Flask Server (Foreground)
    print(f"🌍 Web server running on http://0.0.0.0:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False, threaded=True)

if __name__ == "__main__":
    main()
