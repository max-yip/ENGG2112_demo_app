import cv2
import os
import time
from ultralytics import YOLO

def load_model(model_name, models_dir="models"):
    """Loads a model. Attempts YOLO first."""
    model_path = os.path.join(models_dir, model_name)
    try:
        # Try loading as a YOLO model using ultralytics
        model = YOLO(model_path)
        return model, "YOLO"
    except Exception as e:
        # If it fails, it might be a retinanet or faster rcnn model
        # For this prototype, we'll return None and log the error
        print(f"Failed to load {model_name} as YOLO: {e}")
        return None, f"Error: {str(e)}"

def process_video(video_path, model1, model2, conf_threshold=0.25):
    """
    Generator that processes video frame-by-frame using two models.
    Yields (frame_rgb1, frame_rgb2, metrics1, metrics2, is_finished).
    """
    if not os.path.exists(video_path):
        yield None, None, {"error": "Video not found"}, {"error": "Video not found"}, True
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, None, {"error": "Could not open video"}, {"error": "Could not open video"}, True
        return
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB for Streamlit displaying
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process Model 1
        res1_img = frame_rgb.copy()
        metrics1 = {"Speed": "N/A"}
        if model1 is not None:
            # inference
            results1 = model1(frame_rgb, verbose=False, conf=conf_threshold)
            res1_img = results1[0].plot()
            speed = results1[0].speed
            metrics1 = {"Speed": f"{speed['preprocess']:.1f}ms pre, {speed['inference']:.1f}ms inf, {speed['postprocess']:.1f}ms post"}
        else:
            metrics1 = {"error": "Model 1 unsupported"}
            
        # Process Model 2
        res2_img = frame_rgb.copy()
        metrics2 = {"Speed": "N/A"}
        if model2 is not None:
            # inference
            results2 = model2(frame_rgb, verbose=False, conf=conf_threshold)
            res2_img = results2[0].plot()
            speed = results2[0].speed
            metrics2 = {"Speed": f"{speed['preprocess']:.1f}ms pre, {speed['inference']:.1f}ms inf, {speed['postprocess']:.1f}ms post"}
        else:
            metrics2 = {"error": "Model 2 unsupported"}

        yield res1_img, res2_img, metrics1, metrics2, False

    cap.release()
    yield None, None, {}, {}, True # Finished
