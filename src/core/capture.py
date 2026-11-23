"""Camera sources shared by the data collector, the bench tool and (later) main.py.

Every consumer must get frames through this module so training data and runtime
see identical pixels (spec: docs/design/data-collection.md, section 3).
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from src.core.config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH

COLOR_MODES = ("rgb", "gray3", "gray3_clahe")

_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


@dataclass
class Frame:
    rgb: np.ndarray      # HxWx3 uint8, RGB order, before color_mode is applied
    t_grab_ns: int       # time.monotonic_ns() right after the frame was returned


def apply_color_mode(rgb: np.ndarray, color_mode: str) -> np.ndarray:
    """Return the 3-channel RGB image fed to MediaPipe (it rejects 1-channel input)."""
    if color_mode == "rgb":
        return rgb
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if color_mode == "gray3_clahe":
        gray = _clahe.apply(gray)
    elif color_mode != "gray3":
        raise ValueError(f"unknown color_mode: {color_mode}")
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def brightness(rgb: np.ndarray) -> float:
    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())


class Picamera2Source:
    """CSI camera on the Pi. No horizontal flip: mirroring is for preview only."""

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT,
                 sensor_size: tuple[int, int] | None = None, tuning: str | None = None,
                 noise_reduction: str | None = None, fps: float | None = None):
        from picamera2 import Picamera2  # only importable on the Pi

        tuning_data = Picamera2.load_tuning_file(tuning) if tuning else None
        self._cam = Picamera2(tuning=tuning_data)
        controls = {}
        if fps:
            controls["FrameRate"] = fps
        if noise_reduction:
            from libcamera import controls as lc
            controls["NoiseReductionMode"] = getattr(lc.draft.NoiseReductionModeEnum, noise_reduction)
        extra = {"raw": {"size": sensor_size}} if sensor_size else {}  # raw=None would drop the raw stream
        config = self._cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}, controls=controls, **extra)
        self._cam.configure(config)
        applied = self._cam.camera_configuration()
        mode = (applied.get("raw") or {}).get("size") or (applied.get("sensor") or {}).get("output_size")
        self._info = {
            "backend": "picamera2", "width": width, "height": height, "flip": False,
            "sensor_mode": list(mode) if mode else None,
            "tuning": tuning or "default", "noise_reduction": noise_reduction or "default",
        }

    def start(self) -> None:
        self._cam.start()

    def read(self) -> Frame | None:
        bgr = self._cam.capture_array("main")  # picamera2 "RGB888" is B,G,R in memory
        t = time.monotonic_ns()
        return Frame(rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_grab_ns=t)

    def metadata(self) -> dict:
        return self._cam.capture_metadata()

    def stop(self) -> None:
        self._cam.stop()
        self._cam.close()

    def info(self) -> dict:
        return dict(self._info)


TUNING_DIR = "/usr/share/libcamera/ipa/rpi/vc4"
RESTART_BACKOFF_S = 5.0


class RpicamSource:
    """CSI camera via an `rpicam-vid` subprocess streaming raw I420 frames on stdout.

    Works with any Python version: MediaPipe 0.10.x needs Python <= 3.12, while the apt
    picamera2 bindings are built for the system Python 3.13. A reader thread keeps only the
    newest frame so a slow consumer never processes stale, queued frames.
    """

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT,
                 sensor_size: tuple[int, int] | None = None, tuning: str | None = None,
                 noise_reduction: str | None = None, fps: float | None = 30):
        self._w, self._h = width, height
        self._frame_bytes = width * height * 3 // 2
        self._cmd = ["rpicam-vid", "-t", "0", "-n", "--codec", "yuv420", "--flush",
                     "--width", str(width), "--height", str(height), "-o", "-"]
        if fps:
            self._cmd += ["--framerate", str(fps)]
        if sensor_size:
            self._cmd += ["--mode", f"{sensor_size[0]}:{sensor_size[1]}:10:P"]
        if tuning:
            self._cmd += ["--tuning-file", tuning if "/" in tuning else f"{TUNING_DIR}/{tuning}"]
        if noise_reduction:
            self._cmd += ["--denoise", noise_reduction]  # auto | off | cdn_off | cdn_fast | cdn_hq
        self._info = {
            "backend": "rpicam-vid", "width": width, "height": height, "flip": False,
            "sensor_mode": list(sensor_size) if sensor_size else "auto",
            "tuning": tuning or "default", "noise_reduction": noise_reduction or "default",
        }
        self._proc = None
        self._latest: Frame | None = None
        self._seq = 0
        self._read_seq = 0
        self._cond = threading.Condition()
        self.restarts = 0

    def start(self) -> None:
        self._proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                      bufsize=0)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        buf = bytearray(self._frame_bytes)
        view = memoryview(buf)
        while True:
            got = 0
            while got < self._frame_bytes:
                n = self._proc.stdout.readinto(view[got:])
                if not n:
                    with self._cond:
                        self._latest = None
                        self._seq = -1  # stream ended
                        self._cond.notify_all()
                    return
                got += n
            t = time.monotonic_ns()
            yuv = np.frombuffer(buf, dtype=np.uint8).reshape(self._h * 3 // 2, self._w)
            rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
            with self._cond:
                self._latest = Frame(rgb=rgb, t_grab_ns=t)
                self._seq += 1
                self._cond.notify_all()

    def read(self, timeout: float = 2.0) -> Frame | None:
        """Block until a frame newer than the last one returned arrives; None on timeout (never a repeat)."""
        with self._cond:
            self._cond.wait_for(lambda: self._seq != self._read_seq, timeout=timeout)
            if self._seq > 0 and self._seq != self._read_seq:
                self._read_seq = self._seq
                return self._latest
            ended = self._seq == -1
        if ended:  # rpicam-vid exited (camera unplugged / probe failed): back off instead of spinning
            time.sleep(RESTART_BACKOFF_S)
            self.stop()
            with self._cond:
                self._seq = self._read_seq = 0
                self._latest = None
            self.restarts += 1
            self.start()
        return None

    def metadata(self) -> dict:
        return {}

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def info(self) -> dict:
        return dict(self._info)


class Cv2Source:
    """USB/laptop webcam, for development off the Pi only."""

    def __init__(self, index: int = CAMERA_INDEX, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self._index, self._width, self._height = index, width, height
        self._cap = None

    def start(self) -> None:
        self._cap = cv2.VideoCapture(self._index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open camera index {self._index}")

    def read(self) -> Frame | None:
        ok, bgr = self._cap.read()
        t = time.monotonic_ns()
        if not ok:
            return None
        return Frame(rgb=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), t_grab_ns=t)

    def metadata(self) -> dict:
        return {}

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()

    def info(self) -> dict:
        return {"backend": "cv2", "width": self._width, "height": self._height, "flip": False,
                "sensor_mode": None, "tuning": None, "noise_reduction": None}


def open_source(backend: str, **kwargs):
    if backend == "rpicam":
        return RpicamSource(**kwargs)
    if backend == "picamera2":
        return Picamera2Source(**kwargs)
    if backend == "cv2":
        return Cv2Source(**{k: v for k, v in kwargs.items() if k in ("index", "width", "height")})
    raise ValueError(f"unknown backend: {backend}")
