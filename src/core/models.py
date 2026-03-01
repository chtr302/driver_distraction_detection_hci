import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple

class ONNXClassifier:
    """
    Hệ thống suy luận sử dụng ONNX Runtime (nhẹ hơn, phù hợp cho Raspberry Pi).
    """
    def __init__(self, model_path: Path):
        self.session = ort.InferenceSession(str(model_path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Lấy thông tin Input/Output
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        # Giả định out_dim từ shape (ví dụ: [1, 2] -> 2 classes)
        self.out_dim = self.output_shape[1]
        self.in_dim = self.input_shape[1]

    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Dự đoán trạng thái từ mảng features đã chuẩn hóa.
        Return: (class_id, probability)
        """
        # Đảm bảo đầu vào là float32 và có batch dimension [1, in_dim]
        if features.ndim == 1:
            features = features.reshape(1, -1).astype(np.float32)
        
        # Chạy Inference
        outputs = self.session.run([self.output_name], {self.input_name: features})
        logits = outputs[0]

        if self.out_dim == 1:
            # Binary classification (sigmoid giả lập)
            prob = 1 / (1 + np.exp(-logits[0][0]))
            pred = 1 if prob >= 0.5 else 0
        else:
            # Multi-class classification (softmax giả lập)
            exp_logits = np.exp(logits[0] - np.max(logits[0])) # stability
            probs = exp_logits / np.sum(exp_logits)
            pred = np.argmax(probs)
            prob = probs[pred]
            
        return int(pred), float(prob)

def load_onnx_model(model_path: Path) -> ONNXClassifier:
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    return ONNXClassifier(model_path)
