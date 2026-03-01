import math
import time
import numpy as np

EPS = 1e-6

class HeadPoseCalibrator:
    def __init__(self, calibrate_seconds=3.0):
        self.calibrate_seconds = calibrate_seconds
        self.start_time = None
        self.samples = []
        self.base_yaw = 0.0
        self.base_pitch = 0.0
        self.ready = False

    def update(self, yaw, pitch):
        if self.ready:
            return

        if self.start_time is None:
            self.start_time = time.time()

        self.samples.append((yaw, pitch))

        if time.time() - self.start_time >= self.calibrate_seconds:
            ys = [s[0] for s in self.samples]
            ps = [s[1] for s in self.samples]
            self.base_yaw = float(np.mean(ys))
            self.base_pitch = float(np.mean(ps))
            self.ready = True

    def normalize(self, yaw, pitch):
        return yaw - self.base_yaw, pitch - self.base_pitch


def _np(p):
    return np.array([p.x, p.y, p.z], dtype=np.float32)


def _normalize(v):
    n = np.linalg.norm(v)
    if n < EPS:
        return v * 0.0
    return v / n


def estimate_head_pose(landmarks, img_w, img_h):
    """
    Ước lượng head pose 3D dựa trên trục mặt
    Hoạt động ổn định với mọi góc camera
    """

    try:
        left_eye = _np(landmarks[33])
        right_eye = _np(landmarks[263])
        nose = _np(landmarks[1])
        chin = _np(landmarks[152])
        mouth_l = _np(landmarks[61])
        mouth_r = _np(landmarks[291])
    except Exception:
        return 0.0, 0.0, False

    eye_center = (left_eye + right_eye) * 0.5
    mouth_center = (mouth_l + mouth_r) * 0.5

    # Trục X của khuôn mặt (trái -> phải)
    x_axis = _normalize(right_eye - left_eye)

    # Trục Y của khuôn mặt (trên -> dưới)
    y_axis = _normalize(mouth_center - eye_center)

    if np.linalg.norm(x_axis) < EPS or np.linalg.norm(y_axis) < EPS:
        return 0.0, 0.0, False

    # Trục Z hướng ra phía trước mặt
    z_axis = _normalize(np.cross(x_axis, y_axis))

    if z_axis[2] > 0:
        z_axis = -z_axis

    y_axis = _normalize(np.cross(z_axis, x_axis))

    # Vector nhìn thẳng
    forward = z_axis

    yaw = math.degrees(math.atan2(forward[0], -forward[2]))
    pitch = math.degrees(math.atan2(-forward[1], math.sqrt(forward[0]**2 + forward[2]**2)))

    return yaw, pitch, True


def head_pose_state(yaw, pitch, yaw_th=15, pitch_th=15):
    """
    Phân loại hướng nhìn
    """
    if abs(yaw) <= yaw_th and abs(pitch) <= pitch_th:
        return "NHIN DUONG"

    if pitch > pitch_th:
        return "NGUA LEN"
    if pitch < -pitch_th:
        return "CUI XUONG"
    if yaw > yaw_th:
        return "QUAY PHAI"
    if yaw < -yaw_th:
        return "QUAY TRAI"

    return "KHAC"
