# Driver Distraction Detection
Intelligent Driver Drowsiness & Distraction Detection System.

## Project Overview
WakeGuard is an embedded system designed to detect driver drowsiness and distraction in real-time. It leverages Computer Vision (MediaPipe Face Mesh) and Deep Learning (MLP) running on a Raspberry Pi 4 to analyze facial landmarks and provide multimodal feedback (Haptic, Audio, Visual).

## Hardware Specifications
- **Processor**: Raspberry Pi 4 Model B (4GB/8GB RAM recommended).
- **Camera**: Logitech BRIO 105 (or equivalent USB Webcam).
- **Feedback**:
  - Haptic: PWM Vibration Motors (GPIO Control).
  - Audio: Active Buzzer.
  - Visual: RGB LED.

## Software Stack
- **Language**: Python 3.10+
- **Package Manager**: uv
- **Core Libraries**: MediaPipe (Tasks API), OpenCV, NumPy, ONNX Runtime.
- **Training Framework**: PyTorch (MPS/CPU).

## Repository Structure
- `src/core`: Main logic engine (Vision pipeline, Decision making).
- `src/muscle`: Hardware Abstraction Layer (HAL) for Pi and Mac/Windows.
- `src/training`: Tools for data collection and model training.
- `src/models`: Binary model files (.task, .onnx).
- `data`: Raw and processed datasets.
- `docs`: Detailed documentation.

## Quick Start

### 1. Prerequisites
- Clone this repository.

### 2. Installation
Initialize the environment and install dependencies:

```bash
pip install requirements.txt
```

### 3. Setup Models
Download the required MediaPipe Face Landmarker model:

```bash
# MacOS/Linux
mkdir -p src/models
curl -L -o src/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### 4. Data Collection
To contribute to the dataset, run the collector tool:

```bash
python src/training/collector.py
```

See `docs/DATA_COLLECTION.md` for detailed instructions.
