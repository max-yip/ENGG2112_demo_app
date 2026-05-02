import cv2
import os
import time
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import retinanet_resnet50_fpn, fasterrcnn_resnet50_fpn
from ultralytics import YOLO

def load_model(model_name, models_dir="models"):
    """Loads a model. Supports YOLO and Torchvision models."""
    model_path = os.path.join(models_dir, model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if "retinanet" in model_name.lower():
        try:
            model = retinanet_resnet50_fpn(weights=None, num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, "Torchvision"
        except Exception as e:
            print(f"Failed to load {model_name} as RetinaNet: {e}")
            return None, f"Error: {str(e)}"
    elif "faster_rcnn" in model_name.lower() or "fasterrcnn" in model_name.lower():
        try:
            model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, "Torchvision"
        except Exception as e:
            print(f"Failed to load {model_name} as Faster R-CNN: {e}")
            return None, f"Error: {str(e)}"
    else:
        try:
            # Try loading as a YOLO model using ultralytics
            model = YOLO(model_path)
            return model, "YOLO"
        except Exception as e:
            print(f"Failed to load {model_name} as YOLO: {e}")
            return None, f"Error: {str(e)}"

def run_torchvision_inference(model, frame_rgb, conf_threshold):
    device = next(model.parameters()).device
    img_tensor = F.to_tensor(frame_rgb).to(device)
    
    start_time = time.time()
    with torch.no_grad():
        output = model([img_tensor])[0]
    inf_time = (time.time() - start_time) * 1000
    
    res_img = frame_rgb.copy()
    boxes = output['boxes'].cpu().numpy()
    scores = output['scores'].cpu().numpy()
    labels = output['labels'].cpu().numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        if score >= conf_threshold:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(res_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Assuming class 1 is Human
            label_text = f"Human: {score:.2f}" if label == 1 else f"Class {label}: {score:.2f}"
            cv2.putText(res_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    metrics = {"Speed": f"0.0ms pre, {inf_time:.1f}ms inf, 0.0ms post"}
    return res_img, metrics

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
            if isinstance(model1, YOLO):
                results1 = model1(frame_rgb, verbose=False, conf=conf_threshold)
                res1_img = results1[0].plot()
                speed = results1[0].speed
                metrics1 = {"Speed": f"{speed['preprocess']:.1f}ms pre, {speed['inference']:.1f}ms inf, {speed['postprocess']:.1f}ms post"}
            elif isinstance(model1, torch.nn.Module):
                res1_img, metrics1 = run_torchvision_inference(model1, frame_rgb, conf_threshold)
            else:
                metrics1 = {"error": "Unsupported model type"}
        else:
            metrics1 = {"error": "Model 1 unsupported"}
            
        # Process Model 2
        res2_img = frame_rgb.copy()
        metrics2 = {"Speed": "N/A"}
        if model2 is not None:
            if isinstance(model2, YOLO):
                results2 = model2(frame_rgb, verbose=False, conf=conf_threshold)
                res2_img = results2[0].plot()
                speed = results2[0].speed
                metrics2 = {"Speed": f"{speed['preprocess']:.1f}ms pre, {speed['inference']:.1f}ms inf, {speed['postprocess']:.1f}ms post"}
            elif isinstance(model2, torch.nn.Module):
                res2_img, metrics2 = run_torchvision_inference(model2, frame_rgb, conf_threshold)
            else:
                metrics2 = {"error": "Unsupported model type"}
        else:
            metrics2 = {"error": "Model 2 unsupported"}

        yield res1_img, res2_img, metrics1, metrics2, False

    cap.release()
    yield None, None, {}, {}, True # Finished
