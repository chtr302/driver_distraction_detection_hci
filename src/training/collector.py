# --- FIX PROTOBUF ERROR ON PYTHON 3.11+ ---
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# ------------------------------------------

import time, cv2, platform
import mediapipe as mp
import numpy as np
import pandas as pd
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
CURRENT_DIR = Path(__file__).parent.resolve()
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
MODEL_PATH = SRC_DIR / "models" / "face_landmarker.task"
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_FILE = DATA_DIR / "3d_landmarks_full.csv"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_INNER = [78, 191, 80, 81, 82, 13, 312, 311]
ALL_POINTS = LEFT_EYE + RIGHT_EYE + MOUTH_INNER

STAGES = [
    {"label": 0, "desc": "BINH THUONG - Nhin Thang", "target": 800},
    {"label": 0, "desc": "BINH THUONG - Quay Trai/Phai - Cui/Ngua", "target": 600},
    {"label": 0, "desc": "BINH THUONG - Noi Chuyen / Cuoi", "target": 400},
    {"label": 0, "desc": "BINH THUONG - Deo Kinh / Thieu Sang", "target": 400},
    
    {"label": 1, "desc": "LIM DIM - Mat lo do (Mo 50%)", "target": 600},
    {"label": 1, "desc": "BUON NGU - Nham mat han (Thang)", "target": 600},
    {"label": 1, "desc": "BUON NGU - Nham mat han (Nghieng)", "target": 400},
    {"label": 1, "desc": "NGAP - Ha mieng to", "target": 600},
]

def play_feedback(type : str):
    os_name = platform.system()
    if os_name == "Darwin": # macOS
        sounds = {"start": "Tink", "complete": "Glass", "finish": "Hero"}
        sound_name = sounds.get(type, "Tink")
        os.system(f"afplay /System/Library/Sounds/{sound_name}.aiff &")
    elif os_name == "Windows":
        try:
            import winsound
            if type == "start": freq, dur = 1000, 100
            elif type == "complete": freq, dur = 1500, 300
            elif type == "finish": freq, dur = 2000, 600
            else: freq, dur = 1000, 100
            winsound.Beep(freq, dur)
        except ImportError:
            pass
    else:
        print(f"🔔 {type.upper()}")

def normalize_landmarks(landmarks):
    points = np.array(landmarks)
    center = np.mean(points, axis=0)
    dist = np.linalg.norm(points - center, axis=1)

    scale = np.max(dist)
    if scale < 1e-6: scale=1
    normalized = (points - center) / scale
    return normalized.flatten()

def main():
    if not MODEL_PATH.exists():
        print(f"LOI: Khong tim thay model tai: {MODEL_PATH}")
        print(f"Lenh tai: curl -L -o src/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options = base_options,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"khong dung duoc so 0")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Khong mo duoc cam, da thu index [0,1]")
            return
    
    data_list = []
    current_stage = 0
    collected_count = 0
    recording = False

    print("--- WAKEGUARD COLLECTOR ---")
    print("Space: Start/Pause | Q: Quit")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        h, w, _ = frame.shape

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            selected_pts = [[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in ALL_POINTS]

            # Visual feedback
            for p in selected_pts:
                cv2.circle(frame, (int(p[0]*w), int(p[1]*h)), 2, (0, 255, 255), -1)

            if recording and current_stage < len(STAGES):
                stage_info = STAGES[current_stage]
                if collected_count < stage_info['target']:
                    norm_data = normalize_landmarks(selected_pts)
                    entry = np.append(norm_data, stage_info['label'])
                    data_list.append(entry)
                    collected_count += 1
                else:
                    play_feedback("complete")
                    recording = False
                    collected_count = 0
                    current_stage += 1
                    if current_stage >= len(STAGES):
                        play_feedback("finish")
        
        # UI Rendering
        if current_stage < len(STAGES):
            info = STAGES[current_stage]
            color = (0, 255, 0) if recording else (0, 0, 255)
            cv2.putText(frame, f"Stage {current_stage+1}: {info['desc']}", (20, 40), 1, 1.2, (255,255,255), 2)
            cv2.putText(frame, f"Count: {collected_count}/{info['target']}", (20, 80), 1, 1.2, (255,255,0), 2)
            cv2.putText(frame, "REC" if recording else "PAUSED", (20, 120), 1, 1.2, color, 2)
        else:
            cv2.putText(frame, "DONE! Press Q to Save", (50, h//2), 1, 2.0, (0, 255, 0), 3)

        cv2.imshow("WakeGuard Collector", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == 32: # Space
            if current_stage < len(STAGES):
                recording = not recording
                if recording: play_feedback("start")

    if data_list:
        df = pd.DataFrame(data_list)
        cols = [f"p{i}_{axis}" for i in range(len(ALL_POINTS)) for axis in ['x', 'y', 'z']] + ['label']
        df.columns = cols
        df.to_csv(OUT_FILE, index=False)
        print(f"Saved {len(df)} rows to: {OUT_FILE}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
