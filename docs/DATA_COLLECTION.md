# Data Collection Guide

This document outlines the standard procedure for collecting high-quality training data for the WakeGuard model.

## Objective
To create a robust dataset that covers various lighting conditions, head angles, and fatigue states. The model is only as good as the data provided.

## Setup
1.  Ensure you have installed the project dependencies using `uv sync`.
2.  Ensure the `src/models/face_landmarker.task` file exists.
3.  Find a comfortable seating position.

## Running the Collector
Execute the script from the project root:

```bash
python src/training/collector.py
```

## Collection Stages
The script guides you through 8 distinct stages. Follow the on-screen instructions and audio cues.

### Labels Definition
- **Label 0 (Normal)**: Alert, attentive, eyes open.
- **Label 1 (Drowsy)**: Fatigue signs, closed eyes, yawning.

### Detailed Instructions per Stage

1.  **NORMAL - Looking Straight**
    - Action: Keep head straight, eyes open naturally. Blink normally.
    - Duration: ~800 frames.

2.  **NORMAL - Head Movements**
    - Action: Slowly turn head Left/Right and Up/Down while keeping eyes OPEN.
    - Purpose: Teaches the model that turning the head is not sleeping.
    - Duration: ~600 frames.

3.  **NORMAL - Talking/Smiling**
    - Action: Talk, smile, move mouth.
    - Purpose: Prevents the model from confusing talking with yawning.
    - Duration: ~400 frames.

4.  **NORMAL - Difficult Conditions**
    - Action: Wear glasses (if available), or turn off main lights.
    - Duration: ~400 frames.

5.  **DROWSY - Drowsy Eyes (Lim Dim)**
    - Action: Half-close your eyes, look tired, stare blankly.
    - Importance: Critical for early detection.
    - Duration: ~600 frames.

6.  **DROWSY - Closed Eyes (Straight)**
    - Action: Close eyes completely while facing forward.
    - Duration: ~600 frames.

7.  **DROWSY - Closed Eyes (Tilted)**
    - Action: Close eyes and let head drop slightly (nodding off) or turn to side.
    - Duration: ~400 frames.

8.  **DROWSY - Yawning**
    - Action: Open mouth wide, simulate a deep yawn. Cover mouth with hand occasionally to test occlusion.
    - Duration: ~600 frames.

## Tips for Best Results
- **Camera Position**: Do not fix the camera perfectly. Simulate a dashboard view (low looking up) or a visor view (high looking down).
- **Variety**: Change your distance from the camera slightly between stages.
- **Lighting**: Avoid strong backlighting (window behind you). Front lighting is best, but low light is also valuable data.
