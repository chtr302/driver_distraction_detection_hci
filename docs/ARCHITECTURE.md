# System Architecture

## I. Design Philosophy

WakeGuard is built on a modular, cross-platform architecture that strictly separates three core concerns:
1.  **AI Logic (`core`)**: The "brain" that processes vision data and makes decisions. It is platform-agnostic.
2.  **Hardware Interaction (`muscle`)**: The "muscle" that controls physical components (motors, buzzers). This layer is platform-specific.
3.  **Development/Training (`training`)**: The "lab" where data is collected and models are trained. This is not deployed to the final device.

This separation allows the team to develop and test the core logic on powerful macOS/Windows machines while the embedded-focused members can work on the Raspberry Pi hardware drivers in parallel.

## II. Data & Logic Pipeline (Input -> Process -> Output)

```
[Camera Frame] -> [1. Vision Engine] -> [2. Feature Vector] -> [3. Inference] -> [4. Decision Engine] -> [5. HAL] -> [Haptic/Audio Feedback]
```

### 1. Vision Engine (`core/vision.py`)
- **Source**: Raw image frame from a USB camera.
- **Processor**: `mediapipe.tasks.vision.FaceLandmarker`.
- **Output**: A list of 478 3D face landmarks (`NormalizedLandmark`).

### 2. Feature Vector Generation
- **Input**: All 478 landmarks.
- **Action**:
    1.  **Select**: Extracts 20 critical landmarks for eyes and mouth.
    2.  **Normalize**: Re-calculates landmark coordinates relative to the face's center point and scale. This crucial step makes the model robust against changes in head position, distance from the camera, and face size.
- **Output**: A flattened 1D NumPy array of 60 values (20 points * 3 axes), ready for the AI model.

### 3. Inference Engine (`core/inference.py`)
- **Framework**: `onnxruntime`. This engine is highly optimized for fast inference on ARM CPUs like the Raspberry Pi's.
- **Input**: The 60-value feature vector.
- **Output**: A single probability score (e.g., `0.92`), representing the model's confidence that the driver is drowsy.

### 4. Decision Engine (`core/engine.py`)
This is the state machine that prevents false alarms.
- **Input**: The drowsiness probability score and raw head pose data.
- **Algorithms**:
    - **Gatekeeper**: If the head's yaw/pitch angle exceeds a predefined threshold (e.g., `20` degrees), it assumes the driver is intentionally looking away. During this time, the drowsiness score is ignored to prevent misinterpreting a head turn as closed eyes.
    - **Water Tank**: A score accumulator. Consistently high drowsiness scores "fill the tank". If it overflows the threshold, an alert is triggered. Brief moments of normal behavior "drain the tank", preventing alerts from single, isolated events.

### 5. Hardware Abstraction Layer (HAL) (`muscle/`)
- **Purpose**: Provides a consistent interface for the Decision Engine to call, regardless of the underlying OS.
- **Interface (`driver.py`):** Defines abstract methods like `trigger_alert(level)`, `standby()`, `shutdown()`.
- **Implementations**:
    - `mock.py`: For macOS/Windows. `trigger_alert` prints to the console or plays a system sound.
    - `peripheral.py`: For Raspberry Pi. `trigger_alert` sends PWM signals to GPIO pins to control motors and buzzers.