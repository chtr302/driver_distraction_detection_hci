# WakeGuard
An embedded system for real-time, intelligent driver drowsiness and distraction detection.

## Project Philosophy
- **Edge First**: All processing is done locally on the Raspberry Pi 4 for maximum privacy and low latency. No internet connection is required for core functionality.
- **Developer Experience**: The project is architected for a seamless cross-platform development workflow. Code and train on a powerful machine (macOS/Windows), then deploy the finalized runtime to the Raspberry Pi.

## Core Technology
- **Hardware**: Raspberry Pi 4 Model B (4GB+), USB Webcam, Vibration Motors, Buzzer.
- **Language**: Python 3.11+
- **Package Manager**: `uv`
- **AI Stack**:
  - **Vision**: MediaPipe Face Landmarker (Tasks API).
  - **Inference**: ONNX Runtime (for lightweight model execution).
  - **Training**: PyTorch.

## Installation & Setup

This project uses `uv` for fast and reliable dependency management.

### 1. Prerequisite: Install `uv`
```bash
# On macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create Virtual Environment & Sync
After cloning the repository, run this command from the project root. It will create a `.venv` and install all necessary packages from `pyproject.toml` and `uv.lock`.

```bash
uv sync --dev
```
*The `--dev` flag installs development tools like `torch` and `pandas`.*

### 3. Download AI Model
The data collector requires the MediaPipe Face Landmarker model.

```bash
# Create the directory if it doesn't exist
mkdir -p src/models

# Download the model
curl -L -o src/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Usage

### Data Collection
To contribute to the project's dataset, run the data collector wizard:
```bash
python src/training/collector.py
```
Follow the on-screen instructions. Refer to `docs/DATA_COLLECTION.md` for a detailed guide on each stage.

### Model Training (Coming Soon)
```bash
python src/training/trainer.py
```

### Running the Main Application (Coming Soon)
```bash
python src/main.py
```

## Deployment (Docker -> Raspberry Pi)
The final application will be packaged into a Docker container and deployed to the Raspberry Pi to run as a headless `systemd` service. The Docker setup is pending completion of the core logic.