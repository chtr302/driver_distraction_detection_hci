# System Architecture

## Design Philosophy
WakeGuard follows a modular architecture separating "Brain" (Logic) from "Muscle" (Hardware). This ensures cross-platform compatibility, allowing development on macOS/Windows and deployment on Raspberry Pi without code changes.

## Data Flow Pipeline

1.  **Input (Vision Sensor)**
    - Source: USB Camera.
    - Processing: MediaPipe Face Landmarker (Tasks API).
    - Output: 478 3D Face Landmarks.

2.  **Preprocessing (Feature Engineering)**
    - Extraction: Selects 20 critical landmarks (Eyes, Mouth).
    - Normalization: Converts absolute pixel coordinates (x, y, z) to relative coordinates normalized by face size. This removes dependencies on camera distance and face angle.
    - Flattening: Converts 3D points into a 1D vector (Feature Vector).

3.  **Inference (The Brain)**
    - Model: Multi-Layer Perceptron (MLP).
    - Format: ONNX (Open Neural Network Exchange).
    - Input: Normalized Feature Vector.
    - Output: Probability Score (0.0 - 1.0).

4.  **Decision Logic (The Gatekeeper)**
    - **Score Accumulator**: Uses a weighted counter mechanism (Water Tank). High scores fill the tank; low scores drain it.
    - **Thresholding**:
        - Yawn/Closed Eyes: Increases fatigue counter.
        - Distraction (Head Pose): Pauses eye analysis, triggers separate distraction timer.

5.  **Output (The Muscle)**
    - Interface: Hardware Abstraction Layer (HAL).
    - **Dev Mode (Mac/Win)**: Terminal logs, System sounds.
    - **Prod Mode (Pi)**: GPIO signals to MOSFET (Vibration) and Buzzer.

## Directory Organization

### src/core
Contains the business logic.
- `config.py`: Central configuration for thresholds and file paths.
- `vision.py`: Wrapper around MediaPipe.
- `inference.py`: ONNX Runtime engine.
- `engine.py`: State machine for drowsiness detection.

### src/muscle
Handles hardware interaction.
- `driver.py`: Abstract Base Class defining methods like `alert()`, `vibrate()`.
- `mock.py`: Implementation for development (safe).
- `peripheral.py`: Implementation for Raspberry Pi (RPi.GPIO).

### src/training
Development tools.
- `collector.py`: Data gathering wizard.
- `trainer.py`: PyTorch training script.
