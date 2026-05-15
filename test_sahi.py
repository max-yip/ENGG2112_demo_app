from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2

model_path = "models/yolo26n.pt"
try:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.25,
        device="cpu"
    )
    img = cv2.imread("experiments.json") # invalid image just to see if model loads
    print("Model loaded successfully")
except Exception as e:
    print(f"Error: {e}")
