import cv2
import os
import time
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import retinanet_resnet50_fpn, fasterrcnn_resnet50_fpn
from ultralytics import YOLO

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

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
            return model, "Torchvision", model_path
        except Exception as e:
            print(f"Failed to load {model_name} as RetinaNet: {e}")
            return None, f"Error: {str(e)}", model_path
    elif "faster_rcnn" in model_name.lower() or "fasterrcnn" in model_name.lower():
        try:
            model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, "Torchvision", model_path
        except Exception as e:
            print(f"Failed to load {model_name} as Faster R-CNN: {e}")
            return None, f"Error: {str(e)}", model_path
    else:
        try:
            # Try loading as a YOLO model using ultralytics
            model = YOLO(model_path)
            return model, "YOLO", model_path
        except Exception as e:
            print(f"Failed to load {model_name} as YOLO: {e}")
            return None, f"Error: {str(e)}", model_path

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

def process_video(video_path, model1, model2, conf_threshold=0.25, use_sahi1=False, use_sahi2=False, path1=None, path2=None):
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
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    sahi_model1 = None
    if use_sahi1 and SAHI_AVAILABLE and path1:
        sahi_model1 = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=path1,
            confidence_threshold=conf_threshold,
            device=device
        )
        
    sahi_model2 = None
    if use_sahi2 and SAHI_AVAILABLE and path2:
        sahi_model2 = AutoDetectionModel.from_pretrained(
            model_type='ultralytics',
            model_path=path2,
            confidence_threshold=conf_threshold,
            device=device
        )
        
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
                if use_sahi1 and sahi_model1 is not None:
                    start_time = time.time()
                    result = get_sliced_prediction(
                        frame_rgb,
                        sahi_model1,
                        slice_height=512,
                        slice_width=512,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                        verbose=False
                    )
                    inf_time = (time.time() - start_time) * 1000
                    
                    # Draw boxes
                    for object_prediction in result.object_prediction_list:
                        bbox = object_prediction.bbox.to_xyxy()
                        score = object_prediction.score.value
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(res1_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(res1_img, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    metrics1 = {"Speed": f"SAHI {inf_time:.1f}ms"}
                else:
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
                if use_sahi2 and sahi_model2 is not None:
                    start_time = time.time()
                    result = get_sliced_prediction(
                        frame_rgb,
                        sahi_model2,
                        slice_height=512,
                        slice_width=512,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                        verbose=False
                    )
                    inf_time = (time.time() - start_time) * 1000
                    
                    # Draw boxes
                    for object_prediction in result.object_prediction_list:
                        bbox = object_prediction.bbox.to_xyxy()
                        score = object_prediction.score.value
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(res2_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(res2_img, f"{score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    metrics2 = {"Speed": f"SAHI {inf_time:.1f}ms"}
                else:
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
