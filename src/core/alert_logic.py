from src.core.config import (
    MAX_SCORE, THRESHOLD_LV1, THRESHOLD_LV2,
    SCORE_INCREMENT_MODEL, SCORE_INCREMENT_POSE, SCORE_DECAY
)

class AlertManager:
    def __init__(self):
        self.score = 0.0
        self.level = 0  # 0: Safe, 1: Warning, 2: Critical

    def update(self, is_distracted: bool, head_pose_ok: bool):
        """
        Cập nhật điểm số dựa trên kết quả AI và Head Pose.
        """
        penalty = 0
        if is_distracted:
            penalty += SCORE_INCREMENT_MODEL
        if not head_pose_ok:
            penalty += SCORE_INCREMENT_POSE

        if penalty > 0:
            self.score = min(MAX_SCORE, self.score + penalty)
        else:
            self.score = max(0.0, self.score - SCORE_DECAY)

        # Xác định cấp độ cảnh báo
        if self.score >= THRESHOLD_LV2:
            self.level = 2
        elif self.score >= THRESHOLD_LV1:
            self.level = 1
        else:
            self.level = 0
            
        return self.score, self.level
